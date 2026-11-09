/* Tap a player's name anywhere on the page for his season so far, game by
   game. The data (static/game_log.py) is embedded once as #gamelog-data;
   names are found in the text under the tap, so tables drawn in the browser
   and the other teams' views work without marking each name up. */
(function () {
  "use strict";
  var node = document.getElementById("gamelog-data");
  if (!node) return;
  var D;
  try { D = JSON.parse(node.textContent); } catch (e) { return; }
  var SUFFIX = { jr: 1, sr: 1, ii: 1, iii: 1, iv: 1, v: 1 };

  function norm(t) {
    var s = String(t).normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase()
      .replace(/['’.]/g, "").replace(/[^a-z0-9 ]+/g, " ").trim().split(/ +/).filter(Boolean);
    while (s.length && SUFFIX[s[s.length - 1]]) s.pop();
    return s.join(" ");
  }
  function esc(t) { return String(t).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }

  // One log per league (each on its own scoring sheet), picked by the
  // league panel the tap is in.
  var byLeague = {}, leagues = Object.keys(D);
  leagues.forEach(function (lg) {
    var idx = byLeague[lg] = {};
    Object.keys(D[lg]).forEach(function (k) { var p = D[lg][k]; (idx[norm(p.n)] = idx[norm(p.n)] || []).push(p); });
  });
  function leagueOf(el) {
    var host = el && el.closest && el.closest("[data-league]");
    var lg = host && host.getAttribute("data-league");
    return byLeague[lg] ? lg : leagues[0];
  }

  // The player whose name is under the tap, from the text node at that point.
  function at(ev) {
    var tx, off, n;
    if (document.caretPositionFromPoint) {
      var cp = document.caretPositionFromPoint(ev.clientX, ev.clientY);
      if (cp) { n = cp.offsetNode; off = cp.offset; }
    } else if (document.caretRangeFromPoint) {
      var r = document.caretRangeFromPoint(ev.clientX, ev.clientY);
      if (r) { n = r.startContainer; off = r.startOffset; }
    }
    if (!n || n.nodeType !== 3) return null;
    var byName = byLeague[leagueOf(n.parentElement)];
    tx = n.nodeValue;
    var words = [], re = /\S+/g, m;
    while ((m = re.exec(tx))) words.push([m[0], m.index]);
    var best = null;
    for (var i = 0; i < words.length; i++) {
      for (var len = 1; len <= 4 && i + len <= words.length; len++) {
        var a = words[i][1], b = words[i + len - 1][1] + words[i + len - 1][0].length;
        if (off < a - 1 || off > b + 1) continue;
        var hit = byName[norm(tx.slice(a, b))];
        if (hit && len === 1 && !hit.every(function (x) { return x.p === "DST"; })) hit = null;   // a lone word only names a defence ("Bengals")
        if (hit && (!best || len > best.len)) best = { hit: hit, len: len };
      }
    }
    if (!best) return null;
    if (best.hit.length === 1) return best.hit[0];
    var row = ev.target.closest && ev.target.closest("tr, .card, li, p, div");
    var text = row ? row.textContent : "";
    return best.hit.filter(function (p) { return text.indexOf(" " + p.t) >= 0; })[0] || best.hit[0];
  }

  // ---- Charts: simulated score distributions (the same draws that give P(win)) ----
  // Plain histograms: each bar is the share of draws that landed in it, nothing
  // smoothed. A side with nothing left to play has one score: a solid line.
  function bars(d, x, y, h, from) {
    // d: {x0 (left edge), dx, y[]} -> a step outline closed on the baseline; `from` keeps bins at or right of it.
    var p = "", first = null, last = null;
    d.y.forEach(function (f, i) {
      var l = d.x0 + i * d.dx, r = l + d.dx;
      if (from !== undefined && l < from - 1e-9) return;
      if (first === null) { first = l; p += "M" + x(l).toFixed(1) + " " + h; }
      p += " L" + x(l).toFixed(1) + " " + y(f).toFixed(1) + " L" + x(r).toFixed(1) + " " + y(f).toFixed(1);
      last = r;
    });
    return first === null ? "" : p + " L" + x(last).toFixed(1) + " " + h + " Z";
  }
  function axis(lo, hi, x, h, step) {
    var s = "";
    for (var v = Math.ceil(lo / step) * step; v <= hi; v += step)
      s += '<line x1="' + x(v).toFixed(1) + '" x2="' + x(v).toFixed(1) + '" y1="' + h + '" y2="' + (h + 4) + '" stroke="currentColor" opacity=".4"/>' +
        '<text x="' + x(v).toFixed(1) + '" y="' + (h + 16) + '" text-anchor="middle" font-size="10" fill="currentColor" opacity=".7">' + v + '</text>';
    return s;
  }
  function marker(v, x, h, label, color, solid, row) {
    var ty = 12 + (row || 0) * 12;
    return '<line x1="' + x(v).toFixed(1) + '" x2="' + x(v).toFixed(1) + '" y1="' + (ty - 4) + '" y2="' + h + '" stroke="' + color + '" stroke-width="' + (solid ? 3 : 1.5) + '"' +
      (solid ? "" : ' stroke-dasharray="3 3"') + '/><text x="' + x(v).toFixed(1) + '" y="' + (ty - 6) + '" text-anchor="middle" font-size="10" font-weight="' +
      (solid ? 700 : 400) + '" fill="' + color + '">' + label + '</text>';
  }
  function span(d) { return d.fixed !== undefined ? [d.fixed, d.fixed] : [d.x0, d.x0 + d.y.length * d.dx]; }
  function chart(series, markers, shade, opts) {
    // series: [{d, color, label}] (d a histogram or {fixed}); markers: [[value, label, color]]; shade: {d, from, color}.
    var W = 560, H = 190, L = 10, R = 10;
    var ends = series.map(function (s) { return span(s.d); });
    var lo = Math.min.apply(null, ends.map(function (e) { return e[0]; }));
    var hi = Math.max.apply(null, ends.map(function (e) { return e[1]; }));
    var pad = Math.max(4, (hi - lo) * 0.04); lo -= pad; hi += pad;
    var dens = series.filter(function (s) { return s.d.fixed === undefined; });
    var top = Math.max.apply(null, [1e-9].concat(dens.map(function (s) { return Math.max.apply(null, s.d.y); }))) * 1.05;
    var x = function (v) { return L + (v - lo) / (hi - lo || 1) * (W - L - R); };
    var y = function (f) { return H - f / top * (H - 34); };
    var out = '<svg viewBox="0 0 ' + W + ' ' + (H + 22) + '" width="100%" role="img" aria-label="' + esc(opts.label) + '" style="display:block">';
    if (shade && shade.d.fixed === undefined)
      out += '<path d="' + bars(shade.d, x, y, H, shade.from) + '" fill="' + shade.color + '" opacity=".4"/>';
    dens.forEach(function (s) {
      out += '<path d="' + bars(s.d, x, y, H) + '" fill="' + s.color + '" fill-opacity="' + (shade ? 0 : .2) + '" stroke="' + s.color + '" stroke-width="1.5" stroke-linejoin="round"/>';
    });
    out += '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + H + '" y2="' + H + '" stroke="currentColor" opacity=".35"/>';
    out += axis(lo, hi, x, H, opts.step);
    var row = 0;
    markers.forEach(function (m) { out += marker(m[0], x, H, m[1], m[2], m[3], row++); });
    return out + "</svg>";
  }
  function playerChart(d) {
    if (d.fixed !== undefined)
      return '<p class="sub"><b>This week:</b> ' + d.fixed.toFixed(2) + " points, final (his game is over: one score, no spread left).</p>";
    var svg = chart([{ d: d, color: "#3b82f6" }], [[d.p50, "median " + d.p50.toFixed(1), "#3b82f6"]], null,
      { step: d.dx >= 1 ? 5 : 2, label: "simulated points this week" });
    return '<div class="gl-dist"><p class="sub"><b>Simulated points this week</b> (the draws behind your P(win), as of the last update; ' +
      "each bar the share of draws that landed there): four games in five he scores " + d.p10.toFixed(1) + "&ndash;" + d.p90.toFixed(1) +
      " (10th&ndash;90th percentile); median " + d.p50.toFixed(1) + ", mean " + d.mean.toFixed(1) + ".</p>" + svg + "</div>";
  }
  var M;
  try { M = JSON.parse((document.getElementById("matchup-dist") || {}).textContent || "null"); } catch (e) { M = null; }
  var matchSheet, matchLg = null, matchView = "scores";
  function closeMatch() { if (matchSheet) { matchSheet.remove(); matchSheet = null; matchLg = null; document.documentElement.classList.remove("gl-open"); } }

  // The live draws (static/live.js): points already scored counted, the rest
  // of each player's draw cut to the time he has left.
  function pctile(sorted, q) { return sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(q * sorted.length)))]; }
  function hist(arr, step, lo, hi) {
    var s = Array.prototype.slice.call(arr).sort(function (a, b) { return a - b; });
    if (s[s.length - 1] - s[0] < 0.05) { var v = s[0]; return { fixed: v, p10: v, p50: v, p90: v, mean: v }; }
    var mean = s.reduce(function (a, v) { return a + v; }, 0) / s.length;
    lo = lo === null ? Math.floor(pctile(s, 0.002) / step) * step : lo;
    hi = hi === null ? Math.ceil(pctile(s, 0.998) / step) * step : hi;
    if (hi - lo < step * 3) hi = lo + step * 3;
    var nb = Math.round((hi - lo) / step), h = new Array(nb).fill(0);
    s.forEach(function (v) { h[Math.min(nb - 1, Math.max(0, Math.floor((v - lo) / step)))]++; });
    return { x0: lo, dx: step, y: h.map(function (c) { return c / (s.length * step); }),
             p10: pctile(s, 0.1), p50: pctile(s, 0.5), p90: pctile(s, 0.9), mean: mean };
  }
  function liveMatch(L, base) {
    if (!L || !L.started || !L.mineT || !base) return null;
    var me = L.mineT, opp = L.oppT, n = me.length, mg = new Array(n), i;
    var both = Array.prototype.slice.call(me).concat(Array.prototype.slice.call(opp)).sort(function (a, b) { return a - b; });
    var meH = hist(me, 2, null, null), oppH = hist(opp, 2, null, null);
    var names = [base.names[0], (L.D && L.D.opp_name) || base.names[1]];
    var left = function (rows) { return rows.filter(function (r) { return r.state !== "post" && (r.p.m || 0) > 0 && r.left > 0; }).length; };
    if (meH.fixed !== undefined && oppH.fixed !== undefined)
      return { final: [meH.fixed, oppH.fixed], p: L.p, raw: L.p, names: names, live: true, at: L.at };
    var lo = Math.floor(pctile(both, 0.002) / 2) * 2, hi = Math.ceil(pctile(both, 0.998) / 2) * 2;
    if (meH.fixed === undefined) meH = hist(me, 2, lo, hi);
    if (oppH.fixed === undefined) oppH = hist(opp, 2, lo, hi);
    for (i = 0; i < n; i++) mg[i] = me[i] - opp[i];
    var pos = mg.filter(function (v) { return v > 0; }).sort(function (a, b) { return a - b; });
    var neg = mg.filter(function (v) { return v < 0; }).map(function (v) { return -v; }).sort(function (a, b) { return a - b; });
    return { me: meH, opp: oppH, margin: hist(mg, 2, null, null), raw: L.raw === undefined ? L.p : L.raw, p: L.p, p0: L.p0,
             names: names, live: true, n: n,
             at: L.at, now: [L.nowMe, L.nowOpp], left: [left(L.me), left(L.opp)],
             win_by: pos.length ? pctile(pos, 0.5) : 0, lose_by: neg.length ? pctile(neg, 0.5) : 0 };
  }
  function clock(t) { return t ? new Date(t).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) : ""; }
  function matchBody(m, status) {
    var a = m.names[0], b = m.names[1], pct = function (v) { return (v * 100).toFixed(1) + "%"; };
    var close = '<button type="button" class="gl-x" aria-label="Close">&times;</button>';
    var foot = '<p class="sub gl-status">' + status + "</p>";
    if (m.final)
      return '<div class="gl-head"><div><strong>' + (m.final[0] > m.final[1] ? "You win" : m.final[0] < m.final[1] ? "You lose" : "A tie") +
        '</strong> <span class="opp">' + esc(a) + " vs " + esc(b) + "</span></div>" + close + "</div>" +
        '<p class="sub"><b>' + m.final[0].toFixed(2) + " &ndash; " + m.final[1].toFixed(2) + "</b>: nobody left to play on either side, " +
        "so there is no distribution, just the result. The platform's own score is the one that counts.</p>" + foot;
    var cA = "#3b82f6", cB = "#f97316";
    var lab = function (name, d, c) {
      var nm = name.length <= 14 ? name : name.split(" ")[0];
      return d.fixed !== undefined ? [d.fixed, nm + " " + d.fixed.toFixed(2) + " final", c, true]
                                   : [d.p50, nm + " median " + d.p50.toFixed(0), c, false];
    };
    var scores = chart([{ d: m.me, color: cA }, { d: m.opp, color: cB }], [lab(a, m.me, cA), lab(b, m.opp, cB)], null,
      { step: 20, label: "each team's possible final score" });
    var margin = chart([{ d: m.margin, color: "#64748b" }], [[0, "tie", "#64748b", false]],
      { d: m.margin, from: 0, color: "#22c55e" }, { step: 10, label: "your score minus theirs" });
    var side = function (name, d, c, i) {
      var who = '<span style="color:' + c + '"><b>' + esc(name) + "</b></span>";
      if (d.fixed !== undefined) return who + ": <b>" + d.fixed.toFixed(2) + "</b>, final (nobody left to play, so one score)";
      var so = m.now ? " has " + m.now[i].toFixed(2) + " with " + m.left[i] + " to play;" : "";
      return who + so + " finishes " + d.p10.toFixed(0) + "&ndash;" + d.p90.toFixed(0) + " in four games of five (median " + d.p50.toFixed(0) + ")";
    };
    var n = (m.n || 5000).toLocaleString();
    var what = m.live ? "From " + n + " simulated finishes as of " + clock(m.at) + ": points already scored are counted, and each player still to play " +
        "keeps his draw cut to the time he has left. Each bar is the share of draws that landed there."
      : "From the " + n + " simulated games behind the percentage, as of the last update. Each bar is the share of draws that landed there.";
    var pnote = m.live ? pct(m.raw) + " of the simulated finishes go your way; the figure above is " + pct(m.p) +
        (m.p0 !== undefined ? " (" + pct(m.p0) + " before kickoff)" : "") + ", the same as the live card: the pull toward 50% " +
        "the model applies before kickoff, easing off as points are banked. Your lineup as set."
      : Math.abs(m.raw - m.p) < 0.0005 ? pct(m.raw) + " of the simulated games go your way."
      : "The simulation draws " + pct(m.raw) + " of games for you; the figure above is " + pct(m.p) +
        ": favourites on paper have won a little less often than drawn (2022&ndash;2025), so the model pulls the draw toward 50%.";
    var on = function (v) { return matchView === v; };
    return '<div class="gl-head"><div><strong>Win this week' + (m.live ? " (live)" : "") + ": " + pct(m.p) + '</strong> <span class="opp">' +
      esc(a) + " vs " + esc(b) + "</span></div>" + close + "</div>" +
      '<div class="chips"><button type="button" data-v="scores" aria-pressed="' + on("scores") + '">Scores</button>' +
      '<button type="button" data-v="margin" aria-pressed="' + on("margin") + '">Margin</button></div>' +
      '<div class="gl-v" data-v="scores"' + (on("scores") ? "" : " hidden") + '><p class="sub">' + what + "</p><p class=\"sub\">" +
      side(a, m.me, cA, 0) + "; " + side(b, m.opp, cB, 1) + ".</p>" + scores + "</div>" +
      '<div class="gl-v" data-v="margin"' + (on("margin") ? "" : " hidden") + '><p class="sub">Your score minus theirs. The green bars, right of zero, are your win chance. ' +
      "When you win it is by about " + m.win_by.toFixed(0) + " points, when you lose by about " + m.lose_by.toFixed(0) + ".</p>" + margin + "</div>" +
      '<p class="sub">' + pnote + "</p>" + foot;
  }
  function fill(m, status) {
    if (!matchSheet) return;
    matchSheet.querySelector(".gl-sheet").innerHTML = matchBody(m, status);
  }
  function openMatch(lg) {
    var base = M && M[lg]; if (!base) return false;
    closeMatch(); close();
    var cached = liveMatch(window.StreamerLiveLast && window.StreamerLiveLast[lg], base);
    matchSheet = document.createElement("div");
    matchSheet.className = "gl-back";
    matchSheet.innerHTML = '<div class="gl-sheet" role="dialog" aria-modal="true" aria-label="Win probability"></div>';
    document.body.appendChild(matchSheet);
    document.documentElement.classList.add("gl-open");
    matchLg = lg;
    var live = window.StreamerLive && window.StreamerLive.now;
    fill(cached || base, live ? "Checking the live scores&hellip;" : "Before kickoff, as of the last update.");
    matchSheet.addEventListener("click", function (e) {
      if (e.target === matchSheet || e.target.closest(".gl-x")) return closeMatch();
      var bt = e.target.closest("button[data-v]"); if (!bt) return;
      matchView = bt.getAttribute("data-v");
      matchSheet.querySelectorAll("button[data-v]").forEach(function (x) { x.setAttribute("aria-pressed", x === bt ? "true" : "false"); });
      matchSheet.querySelectorAll(".gl-v").forEach(function (x) { x.hidden = x.getAttribute("data-v") !== matchView; });
    });
    if (live) {
      var mine = matchSheet;
      window.StreamerLive.now(lg).then(function (R) {
        if (matchSheet !== mine) return;
        var m = liveMatch(R, base);
        if (m) fill(m, "Live: ESPN box scores read at " + clock(m.at) + ", scored with this league's rules. Tap again to refresh.");
        else fill(base, R ? "No game has kicked off yet: this is the pre-game picture, as of the last update."
                          : "No live card for this league: this is as of the last update.");
      }).catch(function () {
        if (matchSheet !== mine) return;
        fill(cached || base, "Couldn't reach the live scores just now" + (cached ? "; showing the last live read (" + clock(cached.at) + ")." : "; this is as of the last update."));
      });
    }
    return true;
  }

  var sheet;
  function close() { if (sheet) { sheet.remove(); sheet = null; document.documentElement.classList.remove("gl-open"); } }

  function open(p) {
    close();
    var played = p.g.filter(function (g) { return g.length > 2 && typeof g[2] === "number"; });
    var pts = played.map(function (g) { return g[2]; });
    var avg = pts.length ? pts.reduce(function (a, b) { return a + b; }, 0) / pts.length : 0;
    var top = Math.max.apply(null, pts.concat([1]));
    var bars = p.g.map(function (g) {
      if (g.length <= 2) return '<div class="gl-bar"><div class="gl-v"></div><div class="gl-col gl-na"></div><div class="gl-w">' + g[0] +
        '</div><div class="gl-t">' + (g[1] === "bye" ? "bye" : "&ndash;") + "</div></div>";
      var h = Math.max(3, Math.round(g[2] / top * 56));
      return '<div class="gl-bar"><div class="gl-v">' + g[2].toFixed(1) + '</div><div class="gl-col' + (g[2] >= avg ? " gl-hi" : "") +
        '" style="height:' + h + 'px"></div><div class="gl-w">' + g[0] + '</div><div class="gl-t">' + esc(g[1]) + "</div></div>";
    }).join("");
    var rows = p.g.slice().reverse().map(function (g) {
      if (g[1] === "bye") return '<tr class="gl-off"><td>' + g[0] + '</td><td colspan="4">Bye week</td></tr>';
      if (g.length <= 2) return '<tr class="gl-off"><td>' + g[0] + '</td><td colspan="4">Did not play (inactive, or no stat line)</td></tr>';
      return "<tr><td>" + g[0] + "</td><td>" + esc(g[1]) + "</td><td><b>" + g[2].toFixed(1) + "</b></td><td>" +
        (g[3] === null ? "&ndash;" : g[3] + "%") + '</td><td class="gl-line">' + (esc(g[4]) || '<span class="opp">on the field, no stats</span>') + "</td></tr>";
    }).join("");
    var v = "";
    if (p.v) {
      v = '<p class="sub">Season value ' + p.v[0].toFixed(1) + " a game";
      if (p.v[1] !== null) {
        var d = p.v[0] - p.v[1];
        v += ", " + (d >= 0 ? "up " : "down ") + Math.abs(d).toFixed(1) + " from " + p.v[1].toFixed(1) + " last week";
      }
      v += ".</p>";
    }
    var sum = played.length ? '<p class="sub"><b>' + played.length + " games</b>: " + avg.toFixed(1) + " a game, best " +
      Math.max.apply(null, pts).toFixed(1) + ", worst " + Math.min.apply(null, pts).toFixed(1) + ".</p>" : '<p class="sub">No games played yet.</p>';
    sheet = document.createElement("div");
    sheet.className = "gl-back";
    sheet.innerHTML = '<div class="gl-sheet" role="dialog" aria-modal="true" aria-label="' + esc(p.n) + ' game log">' +
      '<div class="gl-head"><div><strong>' + esc(p.n) + '</strong> <span class="opp">' + esc(p.p) + " " + esc(p.t) +
      '</span></div><button type="button" class="gl-x" aria-label="Close">&times;</button></div>' + v + sum +
      (p.d ? playerChart(p.d) : "") + '<div class="gl-bars">' + bars + '</div><div class="scroll"><table class="gl-table"><thead><tr><th>Wk</th><th>Opp</th><th>Pts</th><th>Snaps</th>' +
      '<th class="gl-line">Line</th></tr></thead><tbody>' + rows + "</tbody></table></div></div>";
    document.body.appendChild(sheet);
    document.documentElement.classList.add("gl-open");
    sheet.addEventListener("click", function (e) { if (e.target === sheet || e.target.closest(".gl-x")) close(); });
    sheet.querySelector(".gl-x").focus();
  }

  // Capture phase: a tap on a name opens the log instead of what the row
  // itself would do (expanding a pickup, say); links, buttons and form
  // controls keep their own behaviour.
  document.addEventListener("click", function (ev) {
    var win = ev.target.closest && ev.target.closest(".kpi[data-win]");
    if (win && !(matchSheet && matchSheet.contains(ev.target))) {
      var host = win.closest("[data-league]"), lg = host && host.getAttribute("data-league");
      if (openMatch(lg)) { ev.preventDefault(); ev.stopPropagation(); return; }
    }
    if (matchSheet && matchSheet.contains(ev.target)) return;
    if (sheet && sheet.contains(ev.target)) return;
    var t = ev.target;
    if (!t.closest || t.closest("a, button, select, input, textarea, label, summary, th, .gl-back")) return;
    var p = at(ev);
    if (!p) return;
    ev.preventDefault();
    ev.stopPropagation();
    open(p);
  }, true);
  document.addEventListener("keydown", function (ev) { if (ev.key === "Escape") { close(); closeMatch(); } });
})();
