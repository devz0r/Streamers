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
      var m = /\(([^()]*?) pass to ([^()]*?) for (?:a )?two[- ]point conversion\)/i.exec(p.text);
      var r = /\(([^()]*?) (?:run|rush) for (?:a )?two[- ]point conversion\)/i.exec(p.text);
      if (/two[- ]point[^()]*failed/i.test(p.text)) return;
      if (m) { [m[1], m[2]].forEach(function (n) { out[norm(n)] = (out[norm(n)] || 0) + 1; }); }
      else if (r) { out[norm(r[1])] = (out[norm(r[1])] || 0) + 1; }
    });
    return out;
  }

  /* What the box score leaves out, read from the play-by-play: kicks a team
     blocked, and the returning unit's own muffed punts recovered by the
     kicking team (a D/ST fumble lost). Team -> {blocked, lost}. */
  function unitPlays(summary, alias) {
    var out = {}, drives = (summary && summary.drives) || {};
    var all = (drives.previous || []).concat(drives.current ? [drives.current] : []);
    var add = function (tm, k) { out[tm] = out[tm] || { blocked: 0, lost: 0 }; out[tm][k] += 1; };
    var teams = {};
    all.forEach(function (d) { var tm = teamOf((d.team || {}).abbreviation, alias); if (tm) teams[tm] = 1; });
    all.forEach(function (d) {
      var offense = teamOf((d.team || {}).abbreviation, alias);
      var other = Object.keys(teams).filter(function (t) { return t !== offense; })[0];
      (d.plays || []).forEach(function (pl) {
        var type = String(((pl.type || {}).text) || ""), text = String(pl.text || "");
        if (!other || /no play/i.test(text)) return;
        if (/(punt|field goal|extra point|PAT)/i.test(type + " " + text) && /\bblocked\b/i.test(text)) add(other, "blocked");
        var rec = /RECOVERED by ([A-Z]{2,3})-/.exec(text);
        if (/punt/i.test(type) && /muff/i.test(text) && rec && teamOf(rec[1], alias) === offense) add(other, "lost");
      });
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
  function dstPoints(box, plays, team, opp, oppScore, dst, unit) {
    var T = box[team], O = box[opp];
    if (!T) return 0;
    var pts = T.def.sacks * (dst.sack || 0) + T.def.ints * (dst.interception || 0);
    var u = (unit && unit[team]) || { blocked: 0, lost: 0 };
    pts += u.blocked * (dst.blocked_kick || 0) + u.lost * (dst.fumble_lost || 0);
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
      return dstPoints(box, plays, tm, opp, game.score[opp] || 0, sc.dst, unitPlays(summary, alias));
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

  /* "Final", "Q3 7:12", "Half", or the kickoff as "Mon 8:15p": short enough
     for a phone. */
  function shortLabel(g, date) {
    var st = g.status || {}, type = st.type || {};
    if (type.state === "post" || type.completed) return "Final";
    if (type.state === "in") {
      if (/HALFTIME/.test(type.name || "")) return "Half";
      var q = num(st.period);
      return (q > 4 ? "OT" : "Q" + q) + " " + (st.displayClock || "");
    }
    var d = date ? new Date(date) : null;
    if (!d || isNaN(d)) return type.shortDetail || "";
    var h = d.getHours(), m = d.getMinutes();
    return ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][d.getDay()] + " " + ((h % 12) || 12) +
      (m ? ":" + (m < 10 ? "0" : "") + m : "") + (h < 12 ? "a" : "p");
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
      g.label = shortLabel(g, e.date);
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

  /* What a defence holds at kickoff, before anything has happened: the
     sheet's tiers for no points (and, where scored, no yards) allowed. */
  function dstBase(dst) {
    dst = dst || {};
    return tier(0, dst.points_allowed_tiers) + (dst.yards_allowed_tiers ? tier(0, dst.yards_allowed_tiers) : 0);
  }

  /* Totals per draw for one side: rows[j] = {now, left, dst}, draws as
     players x n in tenths. */
  function totals(rows, draws, n, means, base) {
    var out = new Float64Array(n);
    rows.forEach(function (r, j) {
      var m = means[j], f = r.left, root = Math.sqrt(f);
      for (var i = 0; i < n; i++) {
        var d = draws[j * n + i] / 10, v;
        if (f >= 1) v = d;
        else if (f <= 0) v = r.now;
        // A defence starts with the sheet's points for nothing allowed (`base`) and gives them back as
        // the game goes: it keeps what it has, less the share of the base still at risk. (Blending its
        // score toward the draw, as first shipped, threw away half of a big first half: graded worse
        // in all of 2022-2025, scripts/grade_live_odds.py.)
        else if (r.dst) v = r.now + f * (m - (base || 0)) + root * (d - m);
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

  function marginVar(a, b) {
    var n = a.length, m = 0, v = 0, i;
    for (i = 0; i < n; i++) m += a[i] - b[i];
    m /= n || 1;
    for (i = 0; i < n; i++) v += (a[i] - b[i] - m) * (a[i] - b[i] - m);
    return n ? v / n : 0;
  }
  function ncdf(x) {
    var t = 1 / (1 + 0.2316419 * Math.abs(x)), d = 0.3989423 * Math.exp(-x * x / 2);
    var p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    return x > 0 ? 1 - p : p;
  }
  function nppf(p) {
    var a = [-39.69683, 220.9461, -275.9285, 138.3578, -30.66480, 2.506628], b = [-54.47610, 161.5858, -155.6990, 66.80131, -13.28068];
    var c = [-0.007784894, -0.3223964, -2.400758, -2.549733, 4.374664, 2.938164], e = [0.007784696, 0.3224671, 2.445134, 3.754409], q, r;
    if (p < 0.02425) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((e[0] * q + e[1]) * q + e[2]) * q + e[3]) * q + 1); }
    if (p > 1 - 0.02425) { q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((e[0] * q + e[1]) * q + e[2]) * q + e[3]) * q + 1); }
    q = p - 0.5; r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  /* P(win) as it happens from P(win) as drawn: Phi(k Phi^-1(p)), with the
     exponent a function of `left`, the share of the pre-game uncertainty
     still to play (the margin's variance now over its variance before
     kickoff). With everything to play it is the lineup model's own k
     (lineup.calibrated: favourites on paper win a little less often than
     drawn), so the live number starts where the page's pre-game number
     stood. In-game the draws turn out slightly too cautious instead (a side
     drawn at 90-97% won 94.9%, 2022-2025), so the exponent rises past 1:
     k = K_END - (K_END - k0) left^K_POWER. Graded on 35,532 in-game states
     of 1,680 replayed games, the two numbers fitted on three seasons and
     scored on the fourth: better than the draws as they are and than a
     straight-line easing in all four (scripts/grade_live_odds.py;
     DECISIONS.md, "Grading the live win chance"). */
  var K_END = 1.12, K_POWER = 2;
  function calibrated(p, k, left) {
    if (!k || p <= 0 || p >= 1) return p;            // no mapping given; a decided game is decided
    var l = Math.min(Math.max(left, 0), 1), ke = K_END - (K_END - k) * Math.pow(l, K_POWER);
    return ncdf(ke * nppf(Math.min(Math.max(p, 1e-6), 1 - 1e-6)));
  }

  /* The live picture for one card's data, given the scoreboard's games and
     the summaries already read (game id -> summary). */
  function evaluate(D, gamesByTeam, summaries) {
    function side(list) {
      return list.map(function (p) {
        var g = gamesByTeam[p.tm];
        if (!g) return { p: p, now: 0, left: p.m > 0 ? 1 : 0, dst: p.pos === "DST", label: "no game" };
        var sum = summaries[g.id], now = 0;
        if (g.state !== "pre" && sum) now = Math.round(pointsFor(p, g, sum, D.sc, D.teams) * 100) / 100;   // leagues score to the hundredth
        var left = g.state !== "pre" && !sum ? 1 : g.left;    // no box score yet: still the draw
        return { p: p, now: now, left: left, dst: p.pos === "DST", label: g.label, state: g.state };
      });
    }
    var me = side(D.me), opp = side(D.opp);
    var S = decode(D.S), O = decode(D.O), n = D.n;
    var base = dstBase((D.sc || {}).dst);
    var mineT = totals(me, S, n, D.me.map(function (p) { return p.m; }), base);
    var oppT = totals(opp, O, n, D.opp.map(function (p) { return p.m; }), base);
    var pre = function (rows) { return rows.map(function (r) { return { now: 0, left: 1, dst: r.dst }; }); };
    var mine0 = totals(pre(me), S, n, D.me.map(function (p) { return p.m; }), base);
    var opp0 = totals(pre(opp), O, n, D.opp.map(function (p) { return p.m; }), base);
    var sum = function (rows) { return rows.reduce(function (s, r) { return s + (r.state && r.state !== "pre" ? r.now : 0); }, 0); };
    var raw = winShare(mineT, oppT), raw0 = winShare(mine0, opp0), v0 = marginVar(mine0, opp0);
    var left = v0 > 0 ? Math.min(marginVar(mineT, oppT) / v0, 1) : 0;
    return { me: me, opp: opp, mineT: mineT, oppT: oppT, raw: raw, raw0: raw0, left: left,
             p: calibrated(raw, D.k, left), p0: calibrated(raw0, D.k, 1),
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

  /* Where a player is headed: his points so far plus the projection's share
     of what is left (all of it before kickoff, none once final). */
  function headed(r) {
    if (!r) return -1;
    var started = r.state && r.state !== "pre";
    return (started ? r.now : 0) + (r.left || 0) * (r.p.m || 0);
  }

  /* Your starters against theirs, slot by slot, best against best; a check
     by whoever is headed higher. */
  function headToHead(R, D) {
    var bySlot = function (rows) {
      var g = {};
      rows.forEach(function (r) { (g[r.p.slot] = g[r.p.slot] || []).push(r); });
      Object.keys(g).forEach(function (k) { g[k].sort(function (a, b) { return headed(b) - headed(a); }); });
      return g;
    };
    var mine = bySlot(R.me), theirs = bySlot(R.opp), order = [];
    R.me.concat(R.opp).forEach(function (r) { if (order.indexOf(r.p.slot) < 0) order.push(r.p.slot); });
    var tick = " <span class='h2h-win'>&#10003;</span>";
    var cell = function (r, win) {
      if (!r) return "<td class='live-name'>–</td><td class='live-pts'></td>";
      var started = r.state && r.state !== "pre";
      var pts = started ? r.now.toFixed(2) : "<span class='opp'>" + (r.p.m || 0).toFixed(1) + "</span>";
      return "<td class='live-name'>" + esc(r.p.n) + " <span class='opp'>" + esc(r.label || "") +
        "</span></td><td class='live-pts'><b>" + pts + "</b>" + (win ? tick : "") + "</td>";
    };
    var rows = "";
    order.forEach(function (slot) {
      var a = mine[slot] || [], b = theirs[slot] || [];
      for (var i = 0; i < Math.max(a.length, b.length); i++) {
        var ha = headed(a[i]), hb = headed(b[i]);
        rows += "<tr><td class='live-slot'>" + esc(slot) + "</td>" + cell(a[i], ha > hb) + cell(b[i], hb > ha) + "</tr>";
      }
    });
    return "<table class='live-table live-h2h'><thead><tr><th></th><th>You</th><th>Pts</th><th>" +
      esc(D.opp_name || "Them") + "</th><th>Pts</th></tr></thead><tbody>" + rows + "</tbody></table>" +
      "<p class='sub'>Points so far (grey: projection, not started); the check is whoever is on track to " +
      "finish higher, his points plus the projection's share of the time left.</p>";
  }

  /* Your playoff and title odds with this game going the way the live P(win)
     says: the season model's odds, moved along the line between "if you win"
     and "if you lose" by how far P(win) has come from kickoff. Only this
     game: other matchups, and the games still to play, are held as modelled.
     Nothing moves before kickoff. */
  function seasonOdds(S, dp) {
    if (!S) return null;
    var clip = function (x) { return Math.min(Math.max(x, 0), 1); };
    return { po: clip(S.po + dp * (S.po_w - S.po_l)), ti: clip(S.ti + dp * (S.ti_w - S.ti_l)),
             po0: S.po, ti0: S.ti };
  }

  function seasonLine(D, R) {
    var o = seasonOdds(D.season, R.p - R.p0);
    if (!o) return "";
    var pct1 = function (x) { return (x * 100).toFixed(x < 0.1 ? 1 : 0) + "%"; };
    return "<p class='sub'>Season: playoffs <b>" + pct1(o.po) + "</b> (" + pct1(o.po0) + " before kickoff) · title <b>" +
      pct1(o.ti) + "</b> (" + pct1(o.ti0) + "). This game only, moved by the live P(win); other games count when they end.</p>";
  }

  function render(el, D, R) {
    var pct = function (x) { return Math.round(x * 100) + "%"; };
    var t = new Date();
    el.querySelector(".live-out").innerHTML =
      "<h3>Live</h3><p class='sub'>You <b>" + R.nowMe.toFixed(2) + "</b> – <b>" + R.nowOpp.toFixed(2) + "</b> " +
      esc(D.opp_name || "them") + " so far · projected " + R.projMe.toFixed(1) + " – " + R.projOpp.toFixed(1) +
      " · P(win) <b>" + pct(R.p) + "</b> (" + pct(R.p0) + " before kickoff)</p>" +
      seasonLine(D, R) +
      headToHead(R, D) +
      "<p class='sub'>ESPN live box scores, scored with this league's rules; updated " +
      t.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) +
      (R.done ? " (all games final)." : ", every minute while this page is open.") +
      " Two-point conversions and blocked kicks are read from the play text and can lag; the platform's own score is final.</p>";
    el.hidden = false;
  }

  var cache = {};       // game id -> summary, for finished games

  /* The live picture for one card's data, read now: the week's scoreboard,
     then the box score of each game under way or over (finished ones once). */
  function fetchNow(D) {
    var teams = {};
    D.me.concat(D.opp).forEach(function (p) { teams[p.tm] = 1; });
    var url = BASE + "scoreboard?seasontype=2&week=" + D.week + "&dates=" + D.season;
    return getJSON(url).then(function (sb) {
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
        R.at = new Date();
        return R;
      });
    });
  }

  function cardData(el) {
    try { return JSON.parse(el.querySelector(".live-data").textContent); } catch (e) { return null; }
  }

  /* Once games are on: the score charts read the same live draws
     (static/gamelog.js), and the "Win this week" tile shows the live P(win)
     (only your own: a tile facing another opponent is another team's view). */
  function remember(el, R, D) {
    var host = el.closest && el.closest("[data-league]");
    if (!host || typeof window === "undefined" || !R.started) return;
    (window.StreamerLiveLast = window.StreamerLiveLast || {})[host.getAttribute("data-league")] = R;
    Array.prototype.forEach.call(host.querySelectorAll(".kpi[data-win]"), function (k) {
      var small = k.querySelector("small"), b = k.querySelector("b"), bar = k.querySelector(".kbar i");
      if (!b || !small || small.textContent.replace(/ \u00b7 live$/, "") !== "vs " + (D.opp_name || "")) return;
      b.textContent = Math.round(R.p * 100) + "%";
      small.textContent = "vs " + (D.opp_name || "") + " \u00b7 live";
      if (bar) bar.style.width = Math.round(R.p * 100) + "%";
    });
  }

  function mount(uid) {
    var el = document.getElementById(uid);
    if (!el || el.dataset.liveMounted) return;
    el.dataset.liveMounted = "1";
    var D = cardData(el);
    if (!D) return;

    function tick() {
      if (!document.body.contains(el)) return;                       // the view was swapped out
      if (document.visibilityState === "hidden") { setTimeout(tick, POLL_MS); return; }
      fetchNow(D).then(function (R) {
        if (R.started) { render(el, D, R); remember(el, R, D); }
        if (!R.done) setTimeout(tick, POLL_MS);
      }).catch(function () { setTimeout(tick, POLL_MS * 2); });
    }
    tick();
  }

  /* The live picture for a league's card, read now (for the score charts):
     resolves to null when the league has no live card. */
  function now(league) {
    var el = document.querySelector('[data-league="' + league + '"] .live-card');
    var D = el && cardData(el);
    if (!D) return Promise.resolve(null);
    return fetchNow(D).then(function (R) { remember(el, R, D); R.D = D; return R; });
  }

  var api = { mount: mount, now: now, fetchNow: fetchNow, boxStats: boxStats, pointsFor: pointsFor, skillPoints: skillPoints,
              kickerPoints: kickerPoints, dstPoints: dstPoints, remaining: remaining, games: games, shortLabel: shortLabel,
              evaluate: evaluate, calibrated: calibrated, seasonOdds: seasonOdds, twoPoints: twoPoints, scoringPlays: scoringPlays, unitPlays: unitPlays };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.StreamerLive = api;
  if (typeof document !== "undefined") {
    Array.prototype.forEach.call(document.querySelectorAll(".live-card"), function (el) { mount(el.id); });
  }
})(typeof window !== "undefined" ? window : this);
