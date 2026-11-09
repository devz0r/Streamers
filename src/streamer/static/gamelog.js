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
  var SVG = "http://www.w3.org/2000/svg";
  function curve(d, x, y, h) {
    // d: {x0, dx, y[]} -> an SVG path (area) in chart coordinates; x(v) maps a value to px, y(f) a density to px.
    var pts = d.y.map(function (f, i) { return [x(d.x0 + i * d.dx), y(f)]; });
    var top = pts.map(function (p, i) { return (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1); }).join(" ");
    return top + " L" + pts[pts.length - 1][0].toFixed(1) + " " + h + " L" + pts[0][0].toFixed(1) + " " + h + " Z";
  }
  function axis(lo, hi, x, h, step) {
    var s = "";
    for (var v = Math.ceil(lo / step) * step; v <= hi; v += step)
      s += '<line x1="' + x(v).toFixed(1) + '" x2="' + x(v).toFixed(1) + '" y1="' + h + '" y2="' + (h + 4) + '" stroke="currentColor" opacity=".4"/>' +
        '<text x="' + x(v).toFixed(1) + '" y="' + (h + 16) + '" text-anchor="middle" font-size="10" fill="currentColor" opacity=".7">' + v + '</text>';
    return s;
  }
  function marker(v, x, h, label, color) {
    return '<line x1="' + x(v).toFixed(1) + '" x2="' + x(v).toFixed(1) + '" y1="6" y2="' + h + '" stroke="' + color + '" stroke-width="1.5" stroke-dasharray="3 3"/>' +
      '<text x="' + x(v).toFixed(1) + '" y="12" text-anchor="middle" font-size="10" fill="' + color + '">' + label + '</text>';
  }
  function chart(series, markers, shade, opts) {
    // series: [{d, color}], markers: [[value, label, color]], shade: {d, from, color} fills a density right of `from`.
    var W = 560, H = 170, L = 8, R = 8, all = series.map(function (s) { return s.d; });
    var lo = Math.min.apply(null, all.map(function (d) { return d.x0; }));
    var hi = Math.max.apply(null, all.map(function (d) { return d.x0 + (d.y.length - 1) * d.dx; }));
    var top = Math.max.apply(null, all.map(function (d) { return Math.max.apply(null, d.y); })) * 1.08;
    var x = function (v) { return L + (v - lo) / (hi - lo || 1) * (W - L - R); };
    var y = function (f) { return H - f / top * (H - 22); };
    var out = '<svg viewBox="0 0 ' + W + ' ' + (H + 22) + '" width="100%" role="img" aria-label="' + esc(opts.label) + '" style="display:block">';
    if (shade) {
      var d = shade.d, idx = [];
      d.y.forEach(function (f, i) { if (d.x0 + i * d.dx >= shade.from) idx.push(i); });
      if (idx.length) {
        var sub = { x0: d.x0 + idx[0] * d.dx, dx: d.dx, y: idx.map(function (i) { return d.y[i]; }) };
        out += '<path d="' + curve(sub, x, y, H) + '" fill="' + shade.color + '" opacity=".35"/>';
      }
    }
    series.forEach(function (s) {
      out += '<path d="' + curve(s.d, x, y, H) + '" fill="' + (shade ? "none" : s.color) + '" fill-opacity="' + (shade ? 0 : .22) + '" stroke="' + s.color + '" stroke-width="2"/>';
    });
    out += '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + H + '" y2="' + H + '" stroke="currentColor" opacity=".35"/>';
    out += axis(lo, hi, x, H, opts.step);
    markers.forEach(function (m) { out += marker(m[0], x, H, m[1], m[2]); });
    return out + "</svg>";
  }
  function playerChart(d, unit) {
    if (d.fixed !== undefined)
      return '<p class="sub"><b>This week:</b> locked at ' + d.fixed.toFixed(1) + " points.</p>";
    var svg = chart([{ d: d, color: "#3b82f6" }], [[d.p50, "median " + d.p50.toFixed(1), "#3b82f6"]], null,
      { step: d.dx >= 1 ? 5 : 2, label: "simulated points this week" });
    return '<div class="gl-dist"><p class="sub"><b>Simulated points this week</b> (the draws behind your P(win)): ' +
      "four games in five he scores " + d.p10.toFixed(1) + "&ndash;" + d.p90.toFixed(1) + " (his 10th&ndash;90th percentile); median " + d.p50.toFixed(1) +
      ", mean " + d.mean.toFixed(1) + ".</p>" + svg + "</div>";
  }
  var M;
  try { M = JSON.parse((document.getElementById("matchup-dist") || {}).textContent || "null"); } catch (e) { M = null; }
  var matchSheet;
  function closeMatch() { if (matchSheet) { matchSheet.remove(); matchSheet = null; document.documentElement.classList.remove("gl-open"); } }
  function openMatch(lg) {
    var m = M && M[lg]; if (!m) return false;
    closeMatch(); close();
    var a = m.names[0], b = m.names[1];
    var scores = chart([{ d: m.me, color: "#3b82f6" }, { d: m.opp, color: "#f97316" }],
      [[m.me.p50, a.split(" ")[0] + " " + m.me.p50.toFixed(0), "#3b82f6"], [m.opp.p50, b.split(" ")[0] + " " + m.opp.p50.toFixed(0), "#f97316"]], null,
      { step: 20, label: "each team's possible score" });
    var margin = chart([{ d: m.margin, color: "#64748b" }], [[0, "tie", "#64748b"]],
      { d: m.margin, from: 0, color: "#22c55e" }, { step: 10, label: "your score minus theirs" });
    var pct = function (v) { return (v * 100).toFixed(1) + "%"; };
    matchSheet = document.createElement("div");
    matchSheet.className = "gl-back";
    matchSheet.innerHTML = '<div class="gl-sheet" role="dialog" aria-modal="true" aria-label="Win probability">' +
      '<div class="gl-head"><div><strong>Win this week: ' + pct(m.p) + '</strong> <span class="opp">' + esc(a) + " vs " + esc(b) +
      '</span></div><button type="button" class="gl-x" aria-label="Close">&times;</button></div>' +
      '<div class="chips"><button type="button" data-v="scores" aria-pressed="true">Scores</button>' +
      '<button type="button" data-v="margin" aria-pressed="false">Margin</button></div>' +
      '<div class="gl-v" data-v="scores"><p class="sub">Each team\'s possible score, from 5,000 simulated games. ' +
      '<span style="color:#3b82f6"><b>' + esc(a) + '</b></span> 10th&ndash;90th: ' + m.me.p10.toFixed(0) + "&ndash;" + m.me.p90.toFixed(0) +
      '; <span style="color:#f97316"><b>' + esc(b) + "</b></span>: " + m.opp.p10.toFixed(0) + "&ndash;" + m.opp.p90.toFixed(0) + ".</p>" + scores + "</div>" +
      '<div class="gl-v" data-v="margin" hidden><p class="sub">Your score minus theirs. The green area, right of zero, is your win chance. ' +
      "When you win it is by about " + m.win_by.toFixed(0) + " points, when you lose by about " + m.lose_by.toFixed(0) + ".</p>" + margin + "</div>" +
      '<p class="sub">The simulation draws ' + pct(m.raw) + ' of games for you; the figure above is ' + pct(m.p) +
      ': favourites on paper have won a little less often than drawn (2022&ndash;2025), so the model pulls the draw toward 50%.</p></div>';
    document.body.appendChild(matchSheet);
    document.documentElement.classList.add("gl-open");
    matchSheet.addEventListener("click", function (e) {
      if (e.target === matchSheet || e.target.closest(".gl-x")) return closeMatch();
      var b = e.target.closest("button[data-v]"); if (!b) return;
      matchSheet.querySelectorAll("button[data-v]").forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
      matchSheet.querySelectorAll(".gl-v").forEach(function (x) { x.hidden = x.getAttribute("data-v") !== b.getAttribute("data-v"); });
    });
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
