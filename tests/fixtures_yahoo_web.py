"""Yahoo fantasy pages, rebuilt from the structure `yahoo-probe --detail`
reported for real pages. Players and teams are invented."""

from __future__ import annotations

LEAGUE = "123456"


def _player_cell(pid: str, name: str, code: str, pos: str, injury: str = "", defense: bool = False) -> str:
    href = "/nfl/teams/tampa-bay/" if defense else f"/nfl/players/{pid}"
    tag = f'<span class="F-injury Fz-xxs Mstart-xs">{injury}</span>' if injury else ""
    return f"""
    <td class="Alt Ta-start player Bdrstart"><div class="Ov-h"><div class="D-f Jc-sb Ai-c">
      <div class="Ta-start Truncate">
        <div class="ysf-player-name Nowrap Relative Lh-xs">
          <a class="Nowrap name F-link playernote" href="{href}" data-ys-playerid="{pid}"
             data-ys-playernote-view="notes" id="playernote-{pid}" title="{name}">{name}</a>
          {tag}
          <span class="player-status D-ib Pstart-sm Nowrap">
            <a href="/nfl/players/{pid}/news" class="playernote" data-ys-playerid="{pid}">
              <span class="ysf-player-icon" title="Player Note">Player Note</span></a>
          </span>
          <span class="D-b"><span class="Fz-xxs">{code} - {pos}</span></span>
        </div>
        <div class="ysf-player-detail Nowrap Fz-xxs Lh-xs">
          <span class="ysf-game-status"><a class="F-reset D-if Ai-c" href="/nfl/x-y-1/">Sun 1:00 pm vs Den</a></span>
        </div>
      </div></div></div></td>"""


def _roster_row(slot: str, pid: str | None, name: str = "", code: str = "", pos: str = "",
                proj: str = "–", injury: str = "", defense: bool = False) -> str:
    player = (_player_cell(pid, name, code, pos, injury, defense) if pid
              else '<td class="Alt Ta-start player Bdrstart"><div>(Empty)</div></td>')
    return f"""
    <tr>
      <td class="Alt Ta-c pos headcol"><div>
        <span class="pos-label Btn-primary Btn-short Miwpx-40 Block Nowrap" data-pos="{slot}">{slot}</span></div></td>
      <td class="Ta-start edit Js-hidden Bdrstart"><div><select><option>{slot}</option><option>BN</option></select></div></td>
      {player}
      <td class="Ta-end Bdrstart"><div>9</div></td>
      <td class="Alt Ta-end Nowrap pts Bdrstart"><div>–</div></td>
      <td class="Ta-end Nowrap"><div>{proj}</div></td>
      <td class="Alt Ta-end Nowrap"><div>88%</div></td>
      <td class="Ta-end Nowrap"><div>97%</div></td>
      <td class="Alt Ta-end"><div><span class="F-faded">–</span></div></td>
      <td class="No-p Spacer"><div></div></td>
    </tr>"""


def _roster_table(table_id: str, group: str, rows: str) -> str:
    return f"""
    <table id="{table_id}" class="Table-plain Table">
      <thead>
        <tr><th class="Alt pos headcol"></th><th class="edit Js-hidden"></th><th class="Alt player"></th>
            <th></th><th colspan="4" class="Alt">Fantasy</th><th class="Alt">Passing</th><th class="No-p Spacer"></th></tr>
        <tr><th class="Alt pos headcol">Pos</th><th class="edit Js-hidden">Edit</th><th class="Alt player">{group}</th>
            <th>Bye</th><th class="Alt Nowrap pts">Fan Pts</th><th class="Nowrap">Proj Pts</th>
            <th class="Alt Nowrap">% Start</th><th class="Nowrap">% Ros</th><th class="Alt">Yds</th>
            <th class="No-p Spacer"></th></tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>"""


def team_page(prefix: str = "1") -> str:
    offense = "".join([
        _roster_row("QB", f"{prefix}01", "Quinn Arrow", "LAR", "QB", "17.84"),
        _roster_row("WR", f"{prefix}02", "Wade Runner", "Buf", "WR", "14.10", injury="Q"),
        _roster_row("WR", f"{prefix}03", "Wes Catch", "Min", "WR", "11.50"),
        _roster_row("RB", f"{prefix}04", "Rob Back", "Det", "RB", "13.20"),
        _roster_row("RB", None),                                          # empty slot
        _roster_row("TE", f"{prefix}05", "Ty End", "KC", "TE", "9.75"),
        _roster_row("W/R/T", f"{prefix}06", "Flex Man", "Was", "RB,WR", "10.05"),
        _roster_row("BN", f"{prefix}07", "Ben Chwarmer", "Jax", "WR", "6.40"),
        _roster_row("BN", f"{prefix}08", "Bo Nch", "NYJ", "RB", "5.00"),
        _roster_row("IR", f"{prefix}09", "Ira Hurt", "SF", "RB", "–", injury="IR"),
    ])
    kicker = _roster_row("K", f"{prefix}10", "Kip Kick", "Phi", "K", "8.40")
    defense = _roster_row("DEF", f"{prefix}11", "Tampa Bay", "TB", "DEF", "7.60", defense=True)
    return f"""<html><head><title>Test League - My Team | Fantasy Football | Yahoo! Sports</title></head><body>
      <a href="/f1/{LEAGUE}/5">My Team</a>
      {_roster_table("statTable0", "Offense", offense)}
      {_roster_table("statTable1", "Kickers", kicker)}
      {_roster_table("statTable2", "Defense/Special Teams", defense)}
    </body></html>"""


def league_page() -> str:
    rows = ""
    for tid, name, wlt, pf in ((2, "Alpha Squad", "2-0-0", "251.4"), (5, "My Team Name", "1-1-0", "230.1"),
                               (6, "The Rival", "1-1-0", "219.9")):
        rows += f"""
        <tr class="Linkable" data-linkable="true" data-target="/f1/{LEAGUE}/{tid}">
          <td class="First Ta-c Px-sm Tst-rank Relative"><span>{tid}</span></td>
          <td class="yfa-td-flex Grid-h-mid Px-sm Tst-manager Nowrap">
            <a class="Grid-u Pend-med Lh-1" href="/f1/{LEAGUE}/{tid}"><img class="Avatar-sm"></a>
            <a class="Grid-u F-reset Ell Mawpx-250" href="/f1/{LEAGUE}/{tid}">{name}</a></td>
          <td class="Nowrap Ta-c Px-sm Tst-wlt">{wlt}</td>
          <td class="Ta-c Px-sm">{pf}</td><td class="Ta-c Px-sm">200.0</td><td class="Ta-c Px-sm">W-1</td>
          <td class="Ta-c Px-sm"><span>3</span></td><td class="Ta-c Px-sm last"><span>4</span></td>
        </tr>"""
    return f"""<html><head><title>Test League | Fantasy Football | Yahoo! Sports</title></head><body>
      <nav><a href="/f1/{LEAGUE}/5">My Team</a></nav>
      <table id="standingstable" class="Table"><thead><tr>
        <th class="Tst-rank-th">Rank</th><th class="Tst-manager-th">Team</th><th class="Tst-wlt-th">W-L-T</th>
        <th>PF</th><th>PA</th><th>Streak</th><th>Waiver</th><th class="last">Moves</th></tr></thead>
      <tbody>{rows}</tbody></table></body></html>"""


def matchup_page(mine: str = "5", theirs: str = "6") -> str:
    return f"""<html><body>
      <a href="/f1/{LEAGUE}/{mine}">Mine</a><a href="/f1/{LEAGUE}/{theirs}">Theirs</a>
      <a href="/f1/{LEAGUE}/{mine}?week=3">Mine again</a></body></html>"""


def free_agent_page(projected: bool = True, week: int = 3) -> str:
    label = f"Projected Stats (Week {week})" if projected else "Season (2026)"
    rows = ""
    for pid, name, code, pos, pts, ros in (("901", "Fay Agent", "Min", "QB", "16.4", "41%"),
                                           ("902", "Walt Wire", "Hou", "WR", "9.8", "12%")):
        rows += f"""
        <tr>
          <td class="Alt"><div><a title="Add Player" href="/f1/{LEAGUE}/addplayer">+</a></div></td>
          <td class="Ta-c Bdrend"><div><a title="Add to Watch List" href="/f1/{LEAGUE}/addplayerwatch">*</a></div></td>
          {_player_cell(pid, name, code, pos)}
          <td class="Ta-start Nowrap Bdrend"><div>FA</div></td>
          <td class="Alt Ta-end F-faded Bdrend"><div>2</div></td>
          <td class="Ta-end Bdrend"><div>7</div></td>
          <td class="Alt Ta-end Nowrap pts"><div><span class="Fw-b">{pts}</span></div></td>
          <td class="Ta-end Nowrap"><div>120</div></td>
          <td class="Alt Ta-end"><div>88</div></td>
          <td class="Ta-end Nowrap"><div>{ros}</div></td>
        </tr>"""
    return f"""<html><body>
      <select name="stat1"><option>Season (2026)</option><option selected>{label}</option></select>
      <table class="Table Ta-start Fz-xs"><thead>
        <tr><th colspan="3" class="Alt"></th><th></th><th></th><th></th><th class="Alt">Fantasy</th>
            <th colspan="2">Rankings</th><th></th></tr>
        <tr><th class="Alt"></th><th></th><th class="Alt player">Offense</th><th>Owner</th><th>GP*</th>
            <th>Bye</th><th class="Alt Nowrap pts">Fan Pts</th><th>Pre-Season</th><th>Current</th><th>% Ros</th></tr>
      </thead><tbody>{rows}</tbody></table></body></html>"""
