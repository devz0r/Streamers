/* Enhancements only: the page works without this script (the league switch
   and every tab are CSS). Adds: the league and tab you were on, kept across
   reloads and linkable (#espn-trades); "updated 12 min ago"; long reasoning
   folded to a few lines with "Show more"; and a search box and position
   filters over the open tab. */
(function () {
  "use strict";
  var KEY = "streamer:view";
  var $ = function (sel, root) { return (root || document).querySelector(sel); };
  var $$ = function (sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); };

  function load() { try { return JSON.parse(localStorage.getItem(KEY) || "{}") || {}; } catch (e) { return {}; } }
  function save(v) { try { localStorage.setItem(KEY, JSON.stringify(v)); } catch (e) { /* private mode */ } }
  function check(id) { var el = document.getElementById(id); if (el) { el.checked = true; return true; } return false; }
  function league() { var r = $("input.profile-radio:checked"); return r ? r.id.replace("profile-", "") : (($(".profile-panel") || {}).dataset || {}).league; }
  function tabOf(lg) { var r = $('input.sec-radio[name="sec-' + lg + '"]:checked'); return r ? r.id.replace("sec-" + lg + "-", "") : ""; }
  function panel(lg) { return document.getElementById("panel-" + lg); }
  function pane(lg) { var t = tabOf(lg); return t ? document.getElementById(lg + "-" + t) : null; }

  // -- where you were ------------------------------------------------------
  function restore() {
    var h = decodeURIComponent((location.hash || "").slice(1)), v = load();
    if (h) {
      var parts = h.split("-"), lg = parts.shift(), tab = parts.join("-");
      if (check("profile-" + lg) || panel(lg)) { if (tab) check("sec-" + lg + "-" + tab); return; }
    }
    if (v.league) check("profile-" + v.league);
    Object.keys(v.tabs || {}).forEach(function (lg) { check("sec-" + lg + "-" + v.tabs[lg]); });
  }
  function remember() {
    var lg = league(), v = load();
    v.league = lg; v.tabs = v.tabs || {};
    $$(".profile-panel").forEach(function (p) { var l = p.dataset.league, t = tabOf(l); if (t) v.tabs[l] = t; });
    save(v);
    var tab = tabOf(lg);
    if (history.replaceState && lg) history.replaceState(null, "", "#" + lg + (tab ? "-" + tab : ""));
  }

  // -- updated N minutes ago -------------------------------------------------
  function ago() {
    $$("time.ago").forEach(function (t) {
      var when = new Date(t.getAttribute("datetime")), min = Math.round((Date.now() - when) / 60000);
      if (!isFinite(min) || min < 0) return;
      var text = min < 1 ? "just now" : min < 60 ? min + " min ago" : min < 1440 ? Math.round(min / 60) + " h ago"
        : Math.round(min / 1440) + " days ago";
      t.textContent = text;
      t.title = when.toLocaleString();
      // Refreshed every two hours through the day: six without one is stale.
      t.classList.toggle("stale", min > 360);
      if (min > 360) t.textContent = text + " · may be out of date";
    });
  }

  // -- long reasoning, folded --------------------------------------------------
  function fold() {
    $$("div.why").forEach(function (el) {
      if (el.textContent.length < 240 || el.dataset.folded) return;
      el.dataset.folded = "1";
      el.classList.add("clamp");
      var b = document.createElement("button");
      b.type = "button"; b.className = "why-toggle"; b.textContent = "Show more";
      b.setAttribute("aria-expanded", "false");
      b.addEventListener("click", function () {
        var open = el.classList.toggle("clamp") === false;
        b.textContent = open ? "Show less" : "Show more";
        b.setAttribute("aria-expanded", String(open));
      });
      el.insertAdjacentElement("afterend", b);
    });
  }

  // -- search and position filters --------------------------------------------
  var POS = ["QB", "RB", "WR", "TE", "K", "DST"];
  var FILTERABLE = { roster: 1, waivers: 1, lineup: 1, streams: 0 };
  function groups(root) {
    // A table row and the note row under it go together; so does a card.
    var out = [];
    $$("tbody tr", root).forEach(function (tr) {
      var note = tr.classList.contains("note") || (tr.firstElementChild && tr.firstElementChild.classList.contains("why"));
      if (note && out.length && out[out.length - 1][0].parentNode === tr.parentNode) out[out.length - 1].push(tr);
      else out.push([tr]);
    });
    $$(".card, .news li", root).forEach(function (c) { out.push([c]); });
    return out;
  }
  function apply(lg) {
    var p = panel(lg), pn = pane(lg), bar = p && $(".tools", p);
    if (!pn || !bar) return;
    var q = ($(".find", bar).value || "").trim().toLowerCase();
    var pos = bar.dataset.pos || "";
    var tab = tabOf(lg);
    $(".chips", bar).classList.toggle("hide", !FILTERABLE[tab]);
    if (!FILTERABLE[tab]) pos = "";
    var shown = 0, total = 0;
    groups(pn).forEach(function (g) {
      var text = g.map(function (e) { return e.textContent; }).join(" ").toLowerCase();
      var chip = $(".pc", g[0]);
      var ok = (!q || text.indexOf(q) >= 0) && (!pos || (chip && chip.textContent.replace("/", "").toUpperCase() === pos));
      g.forEach(function (e) { e.classList.toggle("hide", !ok); });
      total += 1; shown += ok ? 1 : 0;
    });
    var msg = $(".nomatch", pn);
    if ((q || pos) && total && !shown) {
      if (!msg) { msg = document.createElement("p"); msg.className = "nomatch"; pn.insertBefore(msg, pn.firstChild); }
      msg.textContent = "Nothing on this tab matches" + (q ? " “" + q + "”" : "") + (pos ? " at " + pos : "") + ".";
    } else if (msg) msg.remove();
  }
  function tools() {
    $$(".profile-panel").forEach(function (p) {
      var lg = p.dataset.league, nav = $(".sec-labels", p);
      if (!nav || $(".tools", p)) return;
      var bar = document.createElement("div");
      bar.className = "tools";
      bar.innerHTML = '<input class="find" type="search" placeholder="Search players, teams, notes" aria-label="Search this tab" autocomplete="off">'
        + '<div class="chips" role="group" aria-label="Position">'
        + ["All"].concat(POS).map(function (x) {
          return '<button type="button" data-pos="' + (x === "All" ? "" : x) + '" aria-pressed="' + (x === "All") + '">' + (x === "DST" ? "D/ST" : x) + "</button>";
        }).join("") + "</div>";
      var after = $(".more-sheet", p) || nav;
      after.insertAdjacentElement("afterend", bar);
      $(".find", bar).addEventListener("input", function () { apply(lg); });
      $$(".chips button", bar).forEach(function (b) {
        b.addEventListener("click", function () {
          bar.dataset.pos = b.dataset.pos;
          $$(".chips button", bar).forEach(function (x) { x.setAttribute("aria-pressed", String(x === b)); });
          apply(lg);
        });
      });
      apply(lg);
    });
  }

  document.addEventListener("change", function (ev) {
    var t = ev.target;
    if (!t.classList) return;
    if (t.classList.contains("sec-radio") || t.classList.contains("profile-radio")) {
      $$(".more-toggle").forEach(function (m) { m.checked = false; });
      remember();
      var lg = league();
      if (lg) apply(lg);
      if (t.classList.contains("sec-radio")) {
        var nav = $(".sec-labels", panel(lg));
        var top = nav && getComputedStyle(nav).position === "sticky" ? nav.offsetTop : 0;
        if (window.scrollY > top) window.scrollTo(0, top);
      }
    }
  });
  document.addEventListener("keydown", function (ev) {
    if (ev.key === "/" && !/input|select|textarea/i.test((document.activeElement || {}).tagName || "")) {
      var f = $(".find", panel(league()));
      if (f) { ev.preventDefault(); f.focus(); }
    }
    if (ev.key === "Escape") $$(".more-toggle").forEach(function (m) { m.checked = false; });
  });

  window.addEventListener("hashchange", function () {
    restore();
    $$(".profile-panel").forEach(function (p) { apply(p.dataset.league); });
  });

  restore();
  tools();
  fold();
  ago();
  setInterval(ago, 60000);
  $$(".profile-panel").forEach(function (p) { apply(p.dataset.league); });
})();
