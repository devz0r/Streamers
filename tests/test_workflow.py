"""The weekly workflow: shape checks that keep the publishing paths intact."""

from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "weekly.yml"


def _load() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_the_page_is_deployed_only_when_served_from_actions():
    wf = _load()
    steps = {s.get("name"): s for s in wf["jobs"]["weekly"]["steps"]}
    package = steps["Package the page"]
    assert "build_type == 'workflow'" in package["if"] and "docs/index.html" in package["if"]
    assert wf["jobs"]["deploy"]["needs"] == "weekly"
    assert wf["jobs"]["deploy"]["environment"]["name"] == "github-pages"
    assert {"pages", "id-token"} <= set(wf["permissions"])


def test_the_commit_step_still_commits_the_page_when_served_from_a_branch():
    run = next(s for s in _load()["jobs"]["weekly"]["steps"] if s.get("name") == "Commit the refreshed page")["run"]
    assert run.count("git add -A docs reports results data/leagues") == 2     # branch mode, and publish in Actions mode
    assert "git rm -r -q --cached" in run


def test_the_volatile_files_are_ignored():
    ignored = (WORKFLOW.parents[2] / ".gitignore").read_text()
    for path in ("docs/index.html", "docs/latest.json", "docs/views/"):
        assert path in ignored
