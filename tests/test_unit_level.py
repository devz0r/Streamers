"""A D/ST's and kicker's season level: this week's line blended with points so far."""

from streamer.roster import unit_level


def test_a_defences_level_is_mostly_its_season_pulled_to_the_mean():
    c = {"DST": {"line": 0.2, "shrink": 48.0}, "K": {"line": 0.7, "shrink": 8.0}}
    pts = {"DST": ({"JAX": (77.5, 7)}, 8.3), "K": ({"DET": (72.0, 8)}, 8.2)}
    # Jacksonville, 13.3 this week against Baltimore, 11.1 a game over 7 games:
    # 0.2 * 13.3 + 0.8 * (77.5 + 48 * 8.3) / 55 = 9.6, not 13.3 all season.
    assert round(unit_level.level("DST", "JAX", 13.3, pts, c), 1) == 9.6
    # A unit with no games yet sits at the mean, its line aside.
    assert round(unit_level.level("DST", "NYJ", 10.0, pts, c), 2) == round(0.2 * 10 + 0.8 * 8.3, 2)
    # A kicker keeps most of his team's line: 0.7 * 8.8 + 0.3 * (72 + 8 * 8.2) / 16.
    assert round(unit_level.level("K", "DET", 8.8, pts, c), 2) == round(0.7 * 8.8 + 0.3 * (72 + 65.6) / 16, 2)
    # Nothing scored this season: the line, as before.
    assert unit_level.level("DST", "JAX", 13.3, {}, c) == 13.3
