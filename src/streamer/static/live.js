/* Live scores, read in the browser from ESPN's public NFL scoreboard and box
   scores (they send Access-Control-Allow-Origin: *). Each league panel's live
   card carries both sides' starters, the league's scoring and a slice of the
   draws that priced the matchup; this turns each starter's stat line into
   fantasy points and rescores the draws: a finished game counts what he
   scored, one not started counts the draw, one in progress what he has plus
   the draw's share of the time left. P(win) is the share of draws you finish
   ahead. Polls once a minute while the page is open and a game is on.
   streamer/roster/live.py builds the card; DECISIONS.md, "Live scores in the
   browser". Also loadable in Node for the tests (module.exports). */
(function (root) {
  "use strict";
  var BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/";
  var POLL_MS = 60000;

  function num(x) { var v = parseFloat(x); return isFinite(v) ? v : 0; }
  function pair(x) { var m = String(x == null ? "" : x).split(/[\/-]/); return [num(m[0]), num(m[1])]; }
  function norm(name) {
    return String(name || "").toLowerCase().replace(/\b(jr|sr|ii|iii|iv|v)\b\.?/g, "").replace(/[^a-z]/g, "");
  }
  function teamOf(abbr, alias) { abbr = String(abbr || "").toUpperCase(); return (alias && alias[abbr]) || abbr; }

  // -- the box score -----------------------------------------------------
  /* Team -> {ath: id -> {"group.key": value}, names: normalised name -> id,
     def: {sacks, ints}, stats: team stat name -> displayValue}. */
  function boxStats(summary, alias) {
    var out = {}, box = (summary && summary.boxscore) || {};
    (box.players || []).forEach(function (t) {
      var tm = teamOf((t.team || {}).abbreviation, alias);
      var T = out[tm] = out[tm] || { ath: {}, names: {}, def: { sacks: 0, ints: 0 }, stats: {} };
      (t.statistics || []).forEach(function (g) {
        var keys = g.keys || [];
        (g.athletes || []).forEach(function (a) {
          var who = a.athlete || {}, id = String(who.id || ""), s = T.ath[id] = T.ath[id] || {};
          if (who.displayName) T.names[norm(who.displayName)] = id;
          keys.forEach(function (k, i) { s[g.name + "." + k] = (a.stats || [])[i]; });
          if (g.name === "defensive") T.def.sacks += num(s["defensive.sacks"]);
          if (g.name === "interceptions") T.def.ints += num(s["interceptions.interceptions"]);
        });
      });
    });
    (box.teams || []).forEach(function (t) {
      var tm = teamOf((t.team || {}).abbreviation, alias);
      var T = out[tm] = out[tm] || { ath: {}, names: {}, def: { sacks: 0, ints: 0 }, stats: {} };
      (t.statistics || []).forEach(function (st) { T.stats[st.name] = st.displayValue; });
    });
    return out;
  }

  function scoringPlays(summary, alias) {
    return ((summary && summary.scoringPlays) || []).map(function (p) {
      return { team: teamOf(((p.team || {}).abbreviation), alias), type: String(((p.type || {}).text) || ""),
               text: String(p.text || "") };
    });
  }

  /* Two-point conversions by normalised name: "(A Pass to B for Two-Point
     Conversion)" scores both, "(A Run for Two-Point Conversion)" the runner. */
  function twoPoints(plays) {
    var out = {};
    plays.forEach(function (p) {
      var m = /\(([^()]*?) (?:pass|Pass) to ([^()]*?) for Two-Point Conversion\)/.exec(p.text);
      var r = /\(([^()]*?) (?:run|Run) for Two-Point Conversion\)/.exec(p.text);
      if (/Failed/i.test(p.text)) return;
      if (m) { [m[1], m[2]].forEach(function (n) { out[norm(n)] = (out[norm(n)] || 0) + 1; }); }
      else if (r) { out[norm(r[1])] = (out[norm(r[1])] || 0) + 1; }
    });
    return out;
  }

  function find(T, player) {
    if (!T) return null;
    if (player.eid && T.ath[player.eid]) return T.ath[player.eid];
    var id = T.names[norm(player.n)];
    return id ? T.ath[id] : null;
  }

  // -- fantasy points ----------------------------------------------------
  function skillPoints(s, off, twoPt) {
    if (!s) return 0;
    var g = function (k) { return num(s[k]); };
    return g("passing.passingYards") * (off.pass_yards || 0) + g("passing.passingTouchdowns") * (off.pass_td || 0)
      + g("passing.interceptions") * (off.interception || 0)
      + g("rushing.rushingYards") * (off.rush_yards || 0) + g("rushing.rushingTouchdowns") * (off.rush_td || 0)
      + g("receiving.receptions") * (off.reception || 0) + g("receiving.receivingYards") * (off.reception_yards || 0)
      + g("receiving.receivingTouchdowns") * (off.reception_td || 0)
      + g("fumbles.fumblesLost") * (off.fumble_lost || 0) + (twoPt || 0) * (off.two_point || 0);
  }

  function fgValue(d, k) {
    return d >= 60 ? k.fg_60_plus : d >= 50 ? k.fg_50_59 : d >= 40 ? k.fg_40_49 : k.fg_0_39;
  }

  /* A kicker: field goals by distance (from the scoring plays), misses, and
     extra points. */
  function kickerPoints(s, plays, team, k) {
    if (!s) return 0;
    var fg = pair(s["kicking.fieldGoalsMade/fieldGoalAttempts"]), xp = pair(s["kicking.extraPointsMade/extraPointAttempts"]);
    var dists = [];
    plays.forEach(function (p) {
      var m = /(\d+)\s*Yd\s+Field Goal/i.exec(p.text);
      if (p.team === team && m && /Field Goal/i.test(p.type + " " + p.text)) dists.push(num(m[1]));
    });
    var pts = 0;
    for (var i = 0; i < fg[0]; i++) pts += (fgValue(i < dists.length ? dists[i] : 45, k) || 0);
    var miss = (k.fg_missed_0_39 + k.fg_missed_40_49 + k.fg_missed_50_59 + k.fg_missed_60_plus) / 4 || 0;
    pts += Math.max(fg[1] - fg[0], 0) * miss;
    pts += xp[0] * (k.pat_made || 0) + Math.max(xp[1] - xp[0], 0) * (k.pat_missed || 0);
    return pts;
  }

  function tier(value, tiers) {
    if (!tiers) return 0;
    for (var i = 0; i < tiers.length; i++) {
      var lo = tiers[i][0], hi = tiers[i][1];
      if (value >= lo && (hi === null || hi === undefined || value <= hi)) return tiers[i][2];
    }
    return 0;
  }

  function teamStat(T, names) {
    if (!T) return null;
    for (var i = 0; i < names.length; i++) if (T.stats[names[i]] != null) return T.stats[names[i]];
    return null;
  }

  /* A defence: sacks and interceptions from its defenders, fumbles from what
     the other side lost, touchdowns and safeties from the scoring plays,
     points and yards allowed from the score and the other side's yards. */
  function dstPoints(box, plays, team, opp, oppScore, dst) {
    var T = box[team], O = box[opp];
    if (!T) return 0;
    var pts = T.def.sacks * (dst.sack || 0) + T.def.ints * (dst.interception || 0);
    pts += num(teamStat(O, ["fumblesLost"])) * (dst.fumble_recovery || 0);
    plays.forEach(function (p) {
      if (p.team !== team) return;
      var t = p.type + " " + p.text;
      if (/Safety/i.test(p.type)) pts += dst.safety || 0;
      else if (/Blocked/i.test(t) && /Touchdown/i.test(t)) pts += dst.blocked_kick_td || 0;
      else if (/(Kickoff|Punt) Return/i.test(t) && /Touchdown/i.test(t)) pts += dst.return_td || 0;
      else if (/(Interception|Fumble) Return/i.test(t) && /Touchdown/i.test(t)) pts += dst.defensive_td || 0;
    });
    var fourth = pair(teamStat(O, ["fourthDownEff"]));
    pts += Math.max(fourth[1] - fourth[0], 0) * (dst.fourth_down_stop || 0);
    pts += tier(oppScore, dst.points_allowed_tiers);
    if (dst.yards_allowed_tiers) pts += tier(num(teamStat(O, ["totalYards"])), dst.yards_allowed_tiers);
    return pts;
  }

  /* One starter's points so far from his game's summary. */
  function pointsFor(player, game, summary, sc, alias) {
    var box = boxStats(summary, alias), plays = scoringPlays(summary, alias), tm = player.tm;
    if (player.pos === "DST") {
      var opp = game.home === tm ? game.away : game.home;
      return dstPoints(box, plays, tm, opp, game.score[opp] || 0, sc.dst);
    }
    var s = find(box[tm], player);
    if (player.pos === "K") return kickerPoints(s, plays, tm, sc.k);
    return skillPoints(s, sc.off, twoPoints(plays)[norm(player.n)]);
  }

  // -- the games ---------------------------------------------------------
  /* Share of the game still to play: 1 before kickoff, 0 when final. */
  function remaining(status) {
    var type = (status && status.type) || {}, state = type.state;
    if (state === "pre") return 1;
    if (state === "post" || type.completed) return 0;
    if (/HALFTIME/.test(type.name || "")) return 0.5;
    var period = num(status.period), clock = num(status.clock);
    if (period > 4) return 0.02;                     // overtime: next to nothing left
    var played = (period - 1) * 900 + (900 - clock);
    return Math.min(Math.max(1 - played / 3600, 0), 1);
  }

  function games(scoreboard, alias) {
    var out = {};
    ((scoreboard && scoreboard.events) || []).forEach(function (e) {
      var comp = (e.competitions || [])[0] || {}, g = { id: e.id, status: e.status || comp.status || {}, score: {} };
      (comp.competitors || []).forEach(function (c) {
        var tm = teamOf((c.team || {}).abbreviation, alias);
        g.score[tm] = num(c.score);
        if (c.homeAway === "home") g.home = tm; else g.away = tm;
      });
      g.left = remaining(g.status);
      g.state = (g.status.type || {}).state;
      g.label = (g.status.type || {}).shortDetail || "";
      if (g.home) out[g.home] = g;
      if (g.away) out[g.away] = g;
    });
    return out;
  }

  // -- rescoring the draws -----------------------------------------------
  function decode(b64) {
    var s = (typeof atob === "function" ? atob(b64) : Buffer.from(b64, "base64").toString("binary"));
    var u = new Uint8Array(s.length);
    for (var i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
    return new Int16Array(u.buffer);
  }

  /* Totals per draw for one side: rows[j] = {now, left, dst}, draws as
     players x n in tenths. */
  function totals(rows, draws, n, means) {
    var out = new Float64Array(n);
    rows.forEach(function (r, j) {
      var m = means[j], f = r.left, root = Math.sqrt(f);
      for (var i = 0; i < n; i++) {
        var d = draws[j * n + i] / 10, v;
        if (f >= 1) v = d;
        else if (f <= 0) v = r.now;
        else if (r.dst) v = (1 - f) * r.now + f * d;
        else v = Math.max(r.now + f * m + root * (d - m), r.now - 2);
        out[i] += v;
      }
    });
    return out;
  }

  function winShare(a, b) {
    var w = 0;
    for (var i = 0; i < a.length; i++) w += a[i] > b[i] ? 1 : a[i] === b[i] ? 0.5 : 0;
    return w / a.length;
  }

  function mean(a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return a.length ? s / a.length : 0; }

  /* The live picture for one card's data, given the scoreboard's games and
     the summaries already read (game id -> summary). */
  function evaluate(D, gamesByTeam, summaries) {
    function side(list) {
      return list.map(function (p) {
        var g = gamesByTeam[p.tm];
        if (!g) return { p: p, now: 0, left: p.m > 0 ? 1 : 0, dst: p.pos === "DST", label: "no game" };
        var sum = summaries[g.id], now = 0;
        if (g.state !== "pre" && sum) now = pointsFor(p, g, sum, D.sc, D.teams);
        var left = g.state !== "pre" && !sum ? 1 : g.left;    // no box score yet: still the draw
        return { p: p, now: now, left: left, dst: p.pos === "DST", label: g.label, state: g.state };
      });
    }
    var me = side(D.me), opp = side(D.opp);
    var S = decode(D.S), O = decode(D.O), n = D.n;
    var mineT = totals(me, S, n, D.me.map(function (p) { return p.m; }));
    var oppT = totals(opp, O, n, D.opp.map(function (p) { return p.m; }));
    var pre = function (rows) { return rows.map(function (r) { return { now: 0, left: 1, dst: r.dst }; }); };
    var mine0 = totals(pre(me), S, n, D.me.map(function (p) { return p.m; }));
    var opp0 = totals(pre(opp), O, n, D.opp.map(function (p) { return p.m; }));
    var sum = function (rows) { return rows.reduce(function (s, r) { return s + (r.state && r.state !== "pre" ? r.now : 0); }, 0); };
    return { me: me, opp: opp, p: winShare(mineT, oppT), p0: winShare(mine0, opp0),
             nowMe: sum(me), nowOpp: sum(opp), projMe: mean(mineT), projOpp: mean(oppT),
             started: me.concat(opp).some(function (r) { return r.state && r.state !== "pre"; }),
             done: me.concat(opp).every(function (r) { return !r.state || r.state === "post"; }) };
  }

  // -- the page ----------------------------------------------------------
  function getJSON(url) {
    return fetch(url, { cache: "no-store" }).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    });
  }

  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; });
  }

  function render(el, D, R) {
    var pct = function (x) { return Math.round(x * 100) + "%"; };
    var row = function (r) {
      var st = r.state === "post" ? "Final" : r.state === "in" ? r.label : r.label || "";
      var pts = r.state && r.state !== "pre" ? r.now.toFixed(1) : "–";
      return "<tr><td>" + esc(r.p.slot) + "</td><td>" + esc(r.p.n) + "</td><td class='sub'>" + esc(st) +
        "</td><td><b>" + pts + "</b></td></tr>";
    };
    var t = new Date();
    el.querySelector(".live-out").innerHTML =
      "<h3>Live</h3><p class='sub'>You <b>" + R.nowMe.toFixed(1) + "</b> – <b>" + R.nowOpp.toFixed(1) + "</b> " +
      esc(D.opp_name || "them") + " so far · projected " + R.projMe.toFixed(1) + " – " + R.projOpp.toFixed(1) +
      " · P(win) <b>" + pct(R.p) + "</b> (" + pct(R.p0) + " before kickoff)</p>" +
      "<div class='scroll'><table><thead><tr><th>Slot</th><th>You</th><th></th><th>Pts</th></tr></thead><tbody>" +
      R.me.map(row).join("") + "</tbody></table></div>" +
      "<div class='scroll'><table><thead><tr><th>Slot</th><th>Them</th><th></th><th>Pts</th></tr></thead><tbody>" +
      R.opp.map(row).join("") + "</tbody></table></div>" +
      "<p class='sub'>ESPN live box scores, scored with this league's rules; updated " +
      t.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) +
      (R.done ? " (all games final)." : ", every minute while this page is open.") +
      " Two-point conversions and blocked kicks are read from the play text and can lag; the platform's own score is final.</p>";
    el.hidden = false;
  }

  var cache = {};       // game id -> summary, for finished games

  function mount(uid) {
    var el = document.getElementById(uid);
    if (!el || el.dataset.liveMounted) return;
    el.dataset.liveMounted = "1";
    var D;
    try { D = JSON.parse(el.querySelector(".live-data").textContent); } catch (e) { return; }
    var teams = {};
    D.me.concat(D.opp).forEach(function (p) { teams[p.tm] = 1; });
    var url = BASE + "scoreboard?seasontype=2&week=" + D.week + "&dates=" + D.season;

    function tick() {
      if (!document.body.contains(el)) return;                       // the view was swapped out
      if (document.visibilityState === "hidden") { setTimeout(tick, POLL_MS); return; }
      getJSON(url).then(function (sb) {
        var g = games(sb, D.teams), ids = {};
        Object.keys(teams).forEach(function (tm) { var x = g[tm]; if (x && x.state !== "pre") ids[x.id] = x; });
        return Promise.all(Object.keys(ids).map(function (id) {
          if (cache[id]) return Promise.resolve();
          return getJSON(BASE + "summary?event=" + id).then(function (s) {
            if (ids[id].state === "post") cache[id] = s; else cache["live:" + id] = s;
          }).catch(function () {});
        })).then(function () {
          var sums = {};
          Object.keys(ids).forEach(function (id) { sums[id] = cache[id] || cache["live:" + id]; });
          var R = evaluate(D, g, sums);
          if (R.started) render(el, D, R);
          if (!R.done) setTimeout(tick, POLL_MS);
        });
      }).catch(function () { setTimeout(tick, POLL_MS * 2); });
    }
    tick();
  }

  var api = { mount: mount, boxStats: boxStats, pointsFor: pointsFor, skillPoints: skillPoints,
              kickerPoints: kickerPoints, dstPoints: dstPoints, remaining: remaining, games: games,
              evaluate: evaluate, twoPoints: twoPoints, scoringPlays: scoringPlays };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.StreamerLive = api;
  if (typeof document !== "undefined") {
    Array.prototype.forEach.call(document.querySelectorAll(".live-card"), function (el) { mount(el.id); });
  }
})(typeof window !== "undefined" ? window : this);
