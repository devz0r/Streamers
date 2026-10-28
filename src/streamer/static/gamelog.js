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

  var byName = {};
  Object.keys(D).forEach(function (k) { var p = D[k]; (byName[norm(p.n)] = byName[norm(p.n)] || []).push(p); });

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
    tx = n.nodeValue;
    var words = [], re = /\S+/g, m;
    while ((m = re.exec(tx))) words.push([m[0], m.index]);
    var best = null;
    for (var i = 0; i < words.length; i++) {
      for (var len = 2; len <= 4 && i + len <= words.length; len++) {
        var a = words[i][1], b = words[i + len - 1][1] + words[i + len - 1][0].length;
        if (off < a - 1 || off > b + 1) continue;
        var hit = byName[norm(tx.slice(a, b))];
        if (hit && (!best || len > best.len)) best = { hit: hit, len: len };
      }
    }
    if (!best) return null;
    if (best.hit.length === 1) return best.hit[0];
    var row = ev.target.closest && ev.target.closest("tr, .card, li, p, div");
    var text = row ? row.textContent : "";
    return best.hit.filter(function (p) { return text.indexOf(" " + p.t) >= 0; })[0] || best.hit[0];
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
      '<div class="gl-bars">' + bars + '</div><div class="scroll"><table class="gl-table"><thead><tr><th>Wk</th><th>Opp</th><th>Pts</th><th>Snaps</th>' +
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
    if (sheet && sheet.contains(ev.target)) return;
    var t = ev.target;
    if (!t.closest || t.closest("a, button, select, input, textarea, label, summary, th, .gl-back")) return;
    var p = at(ev);
    if (!p) return;
    ev.preventDefault();
    ev.stopPropagation();
    open(p);
  }, true);
  document.addEventListener("keydown", function (ev) { if (ev.key === "Escape") close(); });
})();
