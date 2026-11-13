"""Golden-output check: does a change leave the published page byte-identical?

    python scripts/golden_page.py baseline        # build the page from the current checkout, keep its digests
    ... change code (a refactor, a speed-up) ...
    python scripts/golden_page.py check           # build again, compare, show what moved

The page is built offline (``streamer --offline publish``) in a copy of the
working tree under the system temp directory, so nothing in the real
``docs/``, ``results/`` or ``data/`` is touched. Both builds read the same
inputs: the baseline records a fingerprint of everything the build reads
(``data/``, ``results/``, ``config.yaml``, ``odds_baseline.json``) and ``check``
refuses to compare if it has moved (a ``git pull`` of a refresh, say) -- run
``baseline`` again then. A change meant to alter output (a new feature, a
recalibration) shows exactly which files moved and by how much.

Times and other values that differ on every build are masked before
digesting (:data:`MASKS`). ``--twice`` builds the baseline twice and reports
any difference between the two builds, which is how a new source of
non-determinism is found; the baseline is only kept if they agree.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = Path(tempfile.gettempdir()) / "streamer_golden"
SKIP = {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules"}
#: What a build reads: its fingerprint decides whether two builds are comparable.
INPUTS = ("data", "results", "config.yaml", "odds_baseline.json", "reports")
#: Values that differ on every build, masked before digesting.
MASKS = [
    (re.compile(r'"generated_at":\s*"[^"]*"'), '"generated_at": "<time>"'),
    (re.compile(r"\b20\d\d-\d\d-\d\d[T ]\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:?\d\d)?"), "<time>"),
    (re.compile(r"\b\d{1,2}:\d\d(?::\d\d)? ?(?:AM|PM|am|pm)? ?(?:ET|UTC|EST|EDT)?\b(?= updated| ago|\))"), "<clock>"),
    (re.compile(r"in \d+(?:\.\d+)?s\b"), "in <n>s"),
]


def fingerprint() -> str:
    h = hashlib.sha256()
    for name in INPUTS:
        base = ROOT / name
        files = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file())
        for p in files:
            if "__pycache__" in p.parts:
                continue
            h.update(str(p.relative_to(ROOT)).encode())
            st = p.stat()
            h.update(str(st.st_size).encode())
            h.update(hashlib.sha256(p.read_bytes()).digest() if st.st_size < 4_000_000 else str(st.st_mtime_ns).encode())
    return h.hexdigest()[:16]


def build(label: str) -> tuple[Path, float]:
    """A copy of the working tree, the page built in it offline."""
    work = Path(tempfile.mkdtemp(prefix=f"golden_{label}_"))
    t = time.time()
    for item in ROOT.iterdir():
        if item.name in SKIP:
            continue
        dest = work / item.name
        shutil.copytree(item, dest, ignore=shutil.ignore_patterns(*SKIP)) if item.is_dir() else shutil.copy2(item, dest)
    env = {**os.environ, "PYTHONPATH": str(work / "src"), "PYTHONHASHSEED": "0"}
    week = [a for a in sys.argv if a.startswith("--week=")]
    cmd = [sys.executable, "-m", "streamer.cli", "--offline", "publish"] + (["--week", week[0].split("=")[1]] if week else [])
    if "--week" not in cmd:
        cmd += ["--week", str(newest_week())]
    r = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:])
        raise SystemExit(f"the {label} build failed (exit {r.returncode})")
    return work, time.time() - t


def newest_week() -> int:
    weeks = [int(re.search(r"week_(\d+)", p.name).group(1)) for p in (ROOT / "data" / "leagues").glob("*/week_*.json")]
    return max(weeks)


def masked(text: str) -> str:
    for pat, sub in MASKS:
        text = pat.sub(sub, text)
    return text


def digests(work: Path) -> dict[str, str]:
    out = {}
    docs = work / "docs"
    for p in sorted(docs.rglob("*")):
        if p.is_file() and p.suffix in (".html", ".json", ".js", ".css", ".txt"):
            out[str(p.relative_to(docs))] = hashlib.sha256(masked(p.read_text(encoding="utf-8", errors="replace")).encode()).hexdigest()[:16]
    # What the build wrote besides the page: its logs and results.
    for sub in ("results", "reports"):
        for p in sorted((work / sub).rglob("*")) if (work / sub).exists() else []:
            if p.is_file() and p.suffix in (".json", ".csv", ".md", ".txt"):
                out[f"{sub}/{p.relative_to(work / sub)}"] = hashlib.sha256(masked(p.read_text(encoding="utf-8", errors="replace")).encode()).hexdigest()[:16]
    return out


def show_diff(a_dir: Path, b_dir: Path, name: str, limit: int = 14) -> None:
    pa, pb = a_dir / ("docs" if not name.startswith(("results/", "reports/")) else "") / name, \
        b_dir / ("docs" if not name.startswith(("results/", "reports/")) else "") / name
    if not (pa.exists() and pb.exists()):
        return
    la, lb = masked(pa.read_text(encoding="utf-8", errors="replace")).splitlines(), \
        masked(pb.read_text(encoding="utf-8", errors="replace")).splitlines()
    shown = 0
    for line in difflib.unified_diff(la, lb, lineterm="", n=0):
        if line.startswith(("+++", "---", "@@")):
            continue
        print("      " + (line[:220] + (" ..." if len(line) > 220 else "")))
        shown += 1
        if shown >= limit:
            print("      ...")
            break


def compare(a: dict[str, str], b: dict[str, str], a_dir: Path | None = None, b_dir: Path | None = None) -> int:
    moved = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    for k in moved:
        what = "added" if k not in a else "removed" if k not in b else "changed"
        print(f"  {what:8s} {k}")
        if what == "changed" and a_dir and b_dir:
            show_diff(a_dir, b_dir, k)
    return len(moved)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["baseline", "check"])
    ap.add_argument("--twice", action="store_true", help="baseline: build twice and require the two to agree")
    ap.add_argument("--week", type=int, default=None)
    args, _ = ap.parse_known_args()
    STATE.mkdir(exist_ok=True)
    state_file = STATE / "baseline.json"
    fp = fingerprint()
    if args.mode == "baseline":
        work, secs = build("baseline")
        d = digests(work)
        print(f"baseline built in {secs:.0f}s: {len(d)} files, inputs {fp}")
        if args.twice:
            work2, secs2 = build("again")
            d2 = digests(work2)
            n = compare(d, d2, work, work2)
            print(f"second build in {secs2:.0f}s: {n} files differ from the first")
            shutil.rmtree(work2, ignore_errors=True)
            if n:
                print("the build is not deterministic; mask or fix what moved above, then run again")
                shutil.rmtree(work, ignore_errors=True)
                return 1
        old = json.loads(state_file.read_text()) if state_file.exists() else None
        if old and old.get("dir") and Path(old["dir"]).exists():
            shutil.rmtree(old["dir"], ignore_errors=True)
        state_file.write_text(json.dumps({"inputs": fp, "digests": d, "dir": str(work)}))
        print(f"kept {state_file}")
        return 0
    if not state_file.exists():
        raise SystemExit("no baseline: run `golden_page.py baseline` before the change")
    base = json.loads(state_file.read_text())
    if base["inputs"] != fp:
        raise SystemExit(f"the inputs moved since the baseline ({base['inputs']} -> {fp}): data, results or config changed "
                         "(a pulled refresh?). Run `baseline` again on the unchanged code.")
    work, secs = build("check")
    d = digests(work)
    print(f"check built in {secs:.0f}s: {len(d)} files")
    n = compare(base["digests"], d, Path(base["dir"]) if base.get("dir") and Path(base["dir"]).exists() else None, work)
    print("IDENTICAL: nothing in the page, its data or the results moved" if n == 0 else f"{n} files moved")
    shutil.rmtree(work, ignore_errors=True)
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main())
