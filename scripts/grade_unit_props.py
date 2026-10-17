"""Do the kicker kicking-points line and the opposing QB's interception line
add to our kicker and D/ST projections? Grades ``unit_log.parquet`` against
``history.parquet`` (what the units scored), leave one week out.

    python scripts/grade_unit_props.py

Prints only aggregates. ``roster.unit_props`` and DECISIONS.md, "Kicker props
and the opposing QB's interception line".
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    from streamer.config import get_config
    from streamer.roster import unit_props

    base = get_config()
    for profile in ("espn", "yahoo"):
        cfg = base.for_profile(profile)
        print(f"{profile}:")
        for pos, res in unit_props.grade(cfg).items():
            print(f"  {pos}: {res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
