import pandas as pd

from streamer.features.team_week import drive_table


def test_a_drive_counts_its_scrimmage_plays_only():
    def play(drive, kind, yard, result):
        return {"season": 2026, "week": 1, "game_id": "g", "posteam": "KC", "fixed_drive": drive,
                "fixed_drive_result": result, "yardline_100": yard, "play_type": kind}

    pbp = pd.DataFrame([play(1, "kickoff", 75, "Touchdown"), play(1, "pass", 60, "Touchdown"),
                        play(1, "run", 15, "Touchdown"), play(1, "extra_point", 15, "Touchdown"),
                        play(2, "punt", 70, "Punt"), play(3, "qb_kneel", 40, "End of half")])
    out = drive_table(pbp).set_index("fixed_drive")
    assert out["plays"].tolist() == [2, 0, 1] and str(out["plays"].dtype) == "int64"
    assert out["reached_rz"].tolist() == [1, 0, 0] and out["is_td"].tolist() == [1, 0, 0]
