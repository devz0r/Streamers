// Trade evaluator: the page's side of streamer/roster/trade_eval.py.
// evaluate() mirrors the Python reference line for line, on the same data and
// the same random numbers; tests/test_trade_eval.py runs both and compares.
// Keep the two in step. No "<" followed by "/" anywhere: this file is inlined
// into a script element.
(function (g) {
  if (g.StreamerTrade) return;
  var SKILL = { QB: 1, RB: 1, WR: 1, TE: 1 };

  // -- random numbers the Python side reproduces exactly ------------------
  function fnv1a(text) {
    var bytes = new TextEncoder().encode(text), h = 0x811C9DC5;
    for (var i = 0; i < bytes.length; i++) h = Math.imul(h ^ bytes[i], 0x01000193) >>> 0;
    return h;
  }
  function fmix(x) {
    x ^= x >>> 16; x = Math.imul(x, 0x85EBCA6B); x ^= x >>> 13; x = Math.imul(x, 0xC2B2AE35); x ^= x >>> 16;
    return x >>> 0;
  }
  function u01(seed, s, k) {
    return fmix((seed ^ Math.imul(s + 1, 0x9E3779B1) ^ Math.imul(k + 1, 0x85EBCA77)) >>> 0) / 4294967296;
  }

  function Sim(data) {
    this.data = data; this.n = data.ns; this.k = data.weeks.length; this.cache = {};
  }
  Sim.prototype.levels = function (pid) {
    if (this.cache[pid]) return this.cache[pid];
    var p = this.data.players[pid], n = this.n, k = this.k, out = new Float64Array(n * k);
    if (p.a) {
      var seed = fnv1a(pid), alt = (seed ^ 0x5BD1E995) >>> 0;
      for (var s = 0; s < n; s++) {
        var u1 = u01(alt, s, 1000), u2 = u01(alt, s, 1001);
        var z = Math.sqrt(-2 * Math.log(Math.max(u1, 1e-12))) * Math.cos(2 * Math.PI * u2);
        for (var w = 0; w < k; w++) {
          var lv = Math.max(p.m[w] + p.s[w] * z, 0.01);
          out[s * k + w] = u01(seed, s, w) < p.a[w] ? lv : 0;
        }
      }
    }
    this.cache[pid] = out;
    return out;
  };
  Sim.prototype.mean = function (ids, windows) {
    var D = this.data, n = this.n, k = this.k;
    ids = ids.filter(function (pid) { return SKILL[D.players[pid].pos]; });
    if (!ids.length) return 0;
    var m = ids.length, lv = ids.map(this.levels, this), total = 0;
    var elig = D.sim_slots.map(function (sl) {
      return ids.map(function (pid) { return D.players[pid].el.indexOf(sl[0]) >= 0; });
    });
    var used = new Uint8Array(m), cur = new Float64Array(m);
    for (var s = 0; s < n; s++) {
      for (var w = 0; w < k; w++) {
        for (var j = 0; j < m; j++) {
          var win = windows && windows[ids[j]];
          cur[j] = (win && (w < win[0] || w >= win[1])) ? -1 : lv[j][s * k + w];
          used[j] = 0;
        }
        for (var q = 0; q < D.sim_slots.length; q++) {
          var rep = D.sim_slots[q][2], count = D.sim_slots[q][1], el = elig[q];
          for (var c = 0; c < count; c++) {
            var best = -Infinity, pick = -1;
            for (j = 0; j < m; j++) {
              var v = (el[j] && !used[j]) ? cur[j] : -2;
              if (v > best) { best = v; pick = j; }
            }
            if (best >= Math.max(rep, 0) + 1e-6 && best > -0.5) { total += best; used[pick] = 1; }
            else total += rep;
          }
        }
      }
    }
    return total / (n * k);
  };

  // -- lineups and rosters --------------------------------------------------
  function lineup(D, ids, key) {
    var ranked = ids.slice().sort(function (a, b) { return key(b) - key(a); }), assigned = {};
    D.slots.forEach(function (sl) {
      var taken = 0;
      for (var i = 0; i < ranked.length && taken < sl[1]; i++) {
        var pid = ranked[i];
        if (assigned[pid] || D.players[pid].el.indexOf(sl[0]) < 0) continue;
        assigned[pid] = sl[0]; taken++;
      }
    });
    return assigned;
  }
  function rosterValue(D, ids, key) {
    var a = lineup(D, ids, key), total = 0, bench = {};
    ids.forEach(function (pid) { if (a[pid]) total += key(pid); });
    ids.forEach(function (pid) {
      if (!a[pid]) { var pos = D.players[pid].pos; (bench[pos] = bench[pos] || []).push(key(pid)); }
    });
    Object.keys(bench).forEach(function (pos) {
      var vals = bench[pos].sort(function (x, y) { return y - x; }), level = D.seen_repl[pos] || 0, w = D.bench_w[pos] || [];
      for (var i = 0; i < Math.min(w.length, vals.length); i++) total += w[i] * Math.max(vals[i] - level, 0);
    });
    return total;
  }
  function counts(D, ids) {
    var c = {};
    ids.forEach(function (pid) { var p = D.players[pid]; if (!p.ir) c[p.pos] = (c[p.pos] || 0) + 1; });
    return c;
  }
  function keepsMinimums(D, ids, before) {
    var now = counts(D, ids), was = counts(D, before);
    return Object.keys(D.min_keep).every(function (pos) {
      return (now[pos] || 0) >= Math.min(D.min_keep[pos], was[pos] || 0);
    });
  }
  function makeRoom(D, ids, extra, keep, key) {
    var dropped = [], before = ids;
    for (var r = 0; r < extra; r++) {
      var worst = null, wv = Infinity;
      for (var i = 0; i < ids.length; i++) {
        var pid = ids[i];
        if (keep[pid] || D.players[pid].ir) continue;
        if (!keepsMinimums(D, ids.filter(function (q) { return q !== pid; }), before)) continue;
        if (key(pid) < wv) { wv = key(pid); worst = pid; }
      }
      if (worst === null) break;
      ids = ids.filter(function (q) { return q !== worst; });
      dropped.push(worst);
    }
    return [ids, dropped];
  }
  function consolidation(D, give, get, his) {
    if (give.length === get.length || !give.length || !get.length) return 0;
    var fewerIsGive = give.length < get.length, fewer = fewerIsGive ? give : get, more = fewerIsGive ? get : give;
    var lead = Math.max.apply(null, fewer.map(his)) - Math.max.apply(null, more.map(his));
    if (lead <= 0) return 0;
    var clear = Math.min(lead / D.params.gap, 1);
    return fewerIsGive ? clear : -clear;
  }
  function pAccept(D, gain, cons, engaged) {
    var q = D.params, x = gain + cons * q.cons;
    return q.ceiling * engaged / (1 + Math.exp(-(x - q.center) / q.scale));
  }
  function windowsFor(D, original, now) {
    var lag = D.lag, k = D.weeks.length, ids = now.slice(), win = {};
    now.forEach(function (pid) {
      var join = lag + (D.players[pid].pl ? 1 : 0);
      if (original.indexOf(pid) < 0 && join > 0) win[pid] = [join, k];
    });
    original.forEach(function (pid) {
      var leave = lag + (D.players[pid].pl ? 1 : 0);
      if (now.indexOf(pid) < 0 && leave > 0) { ids.push(pid); win[pid] = [0, leave]; }
    });
    return [ids, win];
  }
  function interp(x, xs, ys) {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
    for (var i = 1; i < xs.length; i++) {
      if (x <= xs[i]) return ys[i - 1] + (ys[i] - ys[i - 1]) * (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
    }
    return ys[ys.length - 1];
  }
  function read(D, curve, team, shift, who) {
    return interp(shift, D.grid, D[curve][team].map(function (row) { return row[who]; }));
  }
  function sign1(x) { return (x >= 0 ? '+' : '') + x.toFixed(1); }
  function pct1(x) { return (x * 100).toFixed(1) + '%'; }
  function set(list) { var o = {}; list.forEach(function (x) { o[x] = 1; }); return o; }

  // -- the evaluation -------------------------------------------------------
  function evaluate(D, partner, give, get, myDrops, sim) {
    sim = sim || new Sim(D);
    var P = D.players, me_i = -1, th_i = -1;
    D.teams.forEach(function (t, i) { if (t.me) me_i = i; if (t.id === partner) th_i = i; });
    var me = D.teams[me_i], th = D.teams[th_i];
    var ours = function (pid) { return P[pid].o; };
    var his = function (pid) { return pid in th.own ? th.own[pid] : P[pid].v; };
    var giveS = set(give), getS = set(get);
    var myNew = me.r.filter(function (pid) { return !giveS[pid]; }).concat(get);
    var theirNew = th.r.filter(function (pid) { return !getS[pid]; }).concat(give);
    var myDropsOut = [], theirDrops = [], extra = get.length - give.length, r;
    if (extra > 0) {
      var chosen = (myDrops || []).filter(function (pid) { return myNew.indexOf(pid) >= 0 && !getS[pid]; }).slice(0, extra);
      myNew = myNew.filter(function (pid) { return chosen.indexOf(pid) < 0; });
      r = makeRoom(D, myNew, extra - chosen.length, getS, function (pid) { return P[pid].o + P[pid].sd; });
      myNew = r[0]; myDropsOut = chosen.concat(r[1]);
    } else if (extra < 0) {
      r = makeRoom(D, theirNew, -extra, giveS, his);
      theirNew = r[0]; theirDrops = r[1];
    }
    r = windowsFor(D, me.r, myNew);
    var dMe = sim.mean(r[0], r[1]) - sim.mean(me.r);
    r = windowsFor(D, th.r, theirNew);
    var dTh = sim.mean(r[0], r[1]) - sim.mean(th.r);
    function odds(curve, who) {
      var b = D.base[curve][who];
      var now = b + (read(D, curve, me_i, dMe, who) - b) + (read(D, curve, th_i, dTh, who) - b);
      return [b, Math.min(Math.max(now, 0), 1)];
    }
    function lineupTotal(ids, key) {
      var a = lineup(D, ids, key), t = 0;
      ids.forEach(function (pid) { if (a[pid]) t += key(pid); });
      return t;
    }
    var seen = rosterValue(D, theirNew, his) - rosterValue(D, th.r, his);
    var cons = consolidation(D, give, get, his);
    var out = {
      shift_me: dMe, shift_them: dTh, lineup_me: lineupTotal(myNew, ours) - lineupTotal(me.r, ours),
      seen: seen, consolidation: cons, p_accept: pAccept(D, seen, cons, th.eng),
      my_drops: myDropsOut, their_drops: theirDrops,
      title_me: odds('title', me_i), title_them: odds('title', th_i),
      playoffs_me: odds('playoffs', me_i), playoffs_them: odds('playoffs', th_i),
      legal: keepsMinimums(D, myNew, me.r) && keepsMinimums(D, theirNew, th.r)
    };
    var why = reasons(D, out, th, give, get, theirNew, his);
    out.why_you = why[0]; out.why_them = why[1];
    var g2 = give.slice().sort(), t2 = get.slice().sort();
    out.exact = null;
    if (!(myDrops && myDrops.length)) {
      D.exact.forEach(function (e) {
        if (e[0] === partner && e[1].join('|') === g2.join('|') && e[2].join('|') === t2.join('|')) out.exact = e.slice(3);
      });
    }
    return out;
  }

  function reasons(D, out, th, give, get, theirNew, his) {
    var P = D.players, gap = D.params.gap_note;
    var you = ['your lineup ' + sign1(out.lineup_me) + ' a game on our projections, ' + sign1(out.shift_me) +
               ' a week once byes and injuries play out'];
    give.forEach(function (pid) {
      var p = P[pid];
      if (p.v - p.o >= gap) you.push('sell high: the market sees ' + p.n + ' at ' + p.v.toFixed(1) + ' a game, we project ' + p.o.toFixed(1));
    });
    get.forEach(function (pid) {
      var p = P[pid];
      if (p.o - p.v >= gap) you.push('buy low: we project ' + p.n + ' at ' + p.o.toFixed(1) + ', the market sees ' + p.v.toFixed(1));
      if (p.sig) you.push(p.n + ': ' + p.sig);
    });
    if (out.my_drops.length) you.push('you would drop ' + out.my_drops.map(function (pid) { return P[pid].n; }).join(' and ') + ' to make room');
    else if (get.length < give.length) you.push('opens a roster spot for a pickup');

    var them = ['as he would see it, his roster ' + sign1(out.seen) + ' a game'];
    get.forEach(function (pid) {
      if (pid in th.over && his(pid) >= P[pid].v + 0.3) {
        them.push('he starts ' + P[pid].n + ' over ' + th.over[pid] + ', so he rates him above the market\'s ' + P[pid].v.toFixed(1) + ' a game');
      }
    });
    var startOld = lineup(D, th.r, his), startNew = lineup(D, theirNew, his);
    var benched = th.r.filter(function (pid) { return startOld[pid] && theirNew.indexOf(pid) >= 0 && !startNew[pid]; });
    for (var i = 0; i < give.length; i++) {
      if (startNew[give[i]] && benched.length) {
        var worst = benched[0];
        benched.forEach(function (pid) { if (his(pid) < his(worst)) worst = pid; });
        them.push(P[give[i]].n + ' would start for him over ' + P[worst].n);
        break;
      }
    }
    if (out.consolidation >= 0.5) them.push('he gets the best player in the deal');
    else if (out.consolidation <= -0.5) them.push('he gives up the best player in the deal, a harder sell');
    give.forEach(function (pid) {
      var p = P[pid], pf = ('pf' in p) ? p.pf : p.o;
      if (p.lp != null && (p.lg || 0) >= 6 && p.lp >= p.o + 2) them.push('name value: ' + p.n + ' scored ' + p.lp.toFixed(1) + ' a game last season');
      if (p.np != null && (p.ng || 0) >= 2 && p.np >= pf + 3) them.push(p.n + ' is averaging ' + p.np.toFixed(1) + ' this season');
    });
    get.forEach(function (pid) {
      var p = P[pid], pf = ('pf' in p) ? p.pf : p.o;
      if (p.out || p.st) them.push(p.n + ' is listed ' + (p.st || 'out'));
      else if (p.np != null && (p.ng || 0) >= 2 && p.np <= pf - 3) them.push(p.n + ' has averaged only ' + p.np.toFixed(1) + ' so far');
    });
    if (th.acq != null && th.eng < 0.75) them.push('he rarely makes moves (' + th.acq + ' pickups), so any offer is a long shot');
    else if (th.acq != null && th.eng >= 0.95) them.push('an active manager (' + th.acq + ' pickups)');
    if (out.title_them[1] >= out.title_them[0]) them.push('his title odds rise too (' + pct1(out.title_them[0]) + ' to ' + pct1(out.title_them[1]) + ')');
    if (out.their_drops.length) them.push('he would drop ' + out.their_drops.map(function (pid) { return P[pid].n; }).join(' and ') + ' to make room');
    return [you, them];
  }

  // -- the page ---------------------------------------------------------------
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function label(p) {
    return p.n + ' ' + p.pos + (p.tm ? ' ' + p.tm : '') + (p.st ? ' (' + p.st + ')' : '') +
      ' · ours ' + p.o.toFixed(1) + ', market ' + p.v.toFixed(1);
  }
  function mount(uid) {
    var root = document.getElementById(uid);
    if (!root) return;
    var D = JSON.parse(root.querySelector('.te-data').textContent), sim = new Sim(D), P = D.players;
    var me = D.teams.filter(function (t) { return t.me; })[0];
    var partnerSel = root.querySelector('.te-partner'), giveBox = root.querySelector('.te-give'),
        getBox = root.querySelector('.te-get'), dropLine = root.querySelector('.te-dropline'),
        out = root.querySelector('.te-out'), sum = root.querySelector('.te-sum');
    var tradeable = function (pid) { return SKILL[P[pid].pos] && !P[pid].ir; };
    D.teams.forEach(function (t) {
      if (t.me) return;
      var o = el('option', null, t.n); o.value = t.id; partnerSel.appendChild(o);
    });
    var dropSel = null;
    function boxes(host, ids) {
      host.textContent = '';
      ids.filter(tradeable).forEach(function (pid) {
        var row = el('label', 'te-pick'), cb = el('input');
        cb.type = 'checkbox'; cb.value = pid; cb.addEventListener('change', function () { dropSel = null; calc(); });
        row.appendChild(cb); row.appendChild(document.createTextNode(' ' + label(P[pid])));
        host.appendChild(row);
      });
    }
    function picked(host) {
      return Array.prototype.filter.call(host.querySelectorAll('input'), function (c) { return c.checked; })
        .map(function (c) { return c.value; });
    }
    function line(cls, text) { var p = el('p', cls); p.textContent = text; return p; }
    function change(pair) { return pct1(pair[0]) + ' \u2192 ' + pct1(pair[1]); }
    function names(ids) {
      return ids.map(function (pid) { return P[pid].n + ' (' + P[pid].pos + ')'; }).join(' + ') || 'nothing';
    }
    function calc() {
      var give = picked(giveBox), get = picked(getBox);
      out.textContent = '';
      if (!give.length && !get.length) {
        dropLine.textContent = ''; dropSel = null;
        sum.textContent = 'Tick players on both sides to see the trade for both teams.';
        return;
      }
      var extra = get.length - give.length, myDrop = dropSel && dropSel.value ? [dropSel.value] : null;
      if (extra <= 0) { dropLine.textContent = ''; dropSel = null; myDrop = null; }
      var t0 = Date.now(), r = evaluate(D, partnerSel.value, give, get, myDrop, sim);
      if (extra > 0 && !dropSel) {
        dropLine.textContent = 'You would have to drop: ';
        dropSel = el('select');
        var auto = el('option', null, 'the lowest-value player (' + r.my_drops.map(function (pid) { return P[pid].n; }).join(', ') + ')');
        auto.value = ''; dropSel.appendChild(auto);
        me.r.filter(function (pid) { return give.indexOf(pid) < 0 && !P[pid].ir; }).forEach(function (pid) {
          var o = el('option', null, P[pid].n + ' ' + P[pid].pos); o.value = pid; dropSel.appendChild(o);
        });
        dropSel.addEventListener('change', calc);
        dropLine.appendChild(dropSel);
      }
      var yes = r.p_accept, word = yes >= 0.5 ? 'likely' : yes >= 0.25 ? 'possible' : 'long shot';
      var mine = (r.title_me[1] - r.title_me[0]) * 100, his = (r.title_them[1] - r.title_them[0]) * 100;
      var partnerName = partnerSel.options[partnerSel.selectedIndex].textContent;
      sum.textContent = 'Your title odds ' + sign1(mine) + ' \u00b7 his ' + sign1(his) +
        ' \u00b7 chance of a yes ~' + Math.round(yes * 100) + '%';
      var card = el('div', 'card');
      var head = el('div', 'row');
      head.appendChild(el('div', 'rank', '\u21c4'));
      var nm = el('div');
      nm.appendChild(el('span', 'name', 'Get ' + names(get)));
      nm.appendChild(document.createTextNode(' '));
      nm.appendChild(el('span', 'opp', 'from ' + partnerName + ' \u00b7 give ' + names(give)));
      head.appendChild(nm);
      head.appendChild(el('div', 'pts', sign1(mine)));
      card.appendChild(head);
      var meta = el('div', 'meta');
      ['title ' + change(r.title_me), 'playoffs ' + change(r.playoffs_me),
       'yes: ' + word + ' (~' + Math.round(yes * 100) + '%)',
       'his title ' + change(r.title_them) + ' (' + sign1(his) + ')', 'his playoffs ' + change(r.playoffs_them)]
        .forEach(function (x) { meta.appendChild(el('span', null, x)); });
      card.appendChild(meta);
      var why = el('div', 'why');
      why.appendChild(el('b', null, 'For you: ')); why.appendChild(document.createTextNode(r.why_you.join('; ') + '.'));
      why.appendChild(el('br'));
      why.appendChild(el('b', null, 'For him: ')); why.appendChild(document.createTextNode(r.why_them.join('; ') + '.'));
      if (r.exact) {
        why.appendChild(el('br'));
        why.appendChild(document.createTextNode('The full simulation priced this exact trade this run: ' +
          sign1(r.exact[0] * 100) + ' for you, ' + sign1(r.exact[1] * 100) + ' for him.'));
      }
      if (!r.legal) {
        why.appendChild(el('br'));
        why.appendChild(document.createTextNode('This leaves a team short at a position it must fill; the platform may refuse it.'));
      }
      card.appendChild(why);
      out.appendChild(card);
      var c = D.check || {};
      var acc = c.me_rmse != null ? ' Checked against the full simulation on the ' + c.n + ' trades it priced this run: ' +
        'typically within \u00b1' + c.me_rmse.toFixed(1) + ' points of your title odds and \u00b1' + c.them_rmse.toFixed(1) +
        ' of his.' : '';
      out.appendChild(line('sub', 'Estimated in ' + (Date.now() - t0) + ' ms from ' + D.ns.toLocaleString() +
        ' simulated seasons of weekly points, read against each team\u2019s title odds on the full simulation.' + acc));
    }
    partnerSel.addEventListener('change', function () { boxes(getBox, D.teams.filter(function (t) { return t.id === partnerSel.value; })[0].r); dropSel = null; calc(); });
    boxes(giveBox, me.r);
    boxes(getBox, D.teams.filter(function (t) { return t.id === partnerSel.value; })[0].r);
    calc();
  }

  g.StreamerTrade = { evaluate: evaluate, Sim: Sim, mount: mount, fnv1a: fnv1a, u01: u01 };
})(typeof window !== 'undefined' ? window : globalThis);
if (typeof module !== 'undefined') module.exports = globalThis.StreamerTrade;
