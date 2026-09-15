"""A result JSON must record the code that produced it.

Job 708549 measured an uncommitted `temporal_features.py` fix instead of the checkpoint --
23.3% of `total_rtt` on one cell, enough to flip the gate verdict -- and neither result JSON
could reveal it, because `run_provenance` recorded env vars and contracts but never the code
version. These tests drive `describe_code_provenance` against real throwaway repos, so they
fail if the stamp stops distinguishing the three cases a triage needs to tell apart:
different commit, same commit / different working tree, identical code.
"""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.placement.env_fingerprint import (  # noqa: E402
    CODE_PATHS,
    describe_code_provenance,
    format_code_banner,
)


def _git(repo, *args):
    subprocess.run(("git", "-C", str(repo)) + args, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    """A minimal git repo with one tracked file under a code path and one outside it."""
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "sim.py").write_text("RATE = 1\n")
    (tmp_path / "simulation_data").mkdir()
    (tmp_path / "simulation_data" / "REGISTRY.json").write_text("{}\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "init")
    return tmp_path


def test_clean_tree_records_commit_and_is_not_dirty(repo):
    p = describe_code_provenance(repo)
    assert p["git_available"] is True
    assert len(p["commit"]) == 40
    assert p["dirty"] is False
    assert p["changed_files"] == []
    assert p["code_paths"] == list(CODE_PATHS)


def test_uncommitted_code_edit_is_flagged_and_hashed(repo):
    clean = describe_code_provenance(repo)
    (repo / "src" / "sim.py").write_text("RATE = 2\n")
    dirty = describe_code_provenance(repo)

    # The 708549 case: same commit, different code, different number.
    assert dirty["commit"] == clean["commit"]
    assert dirty["dirty"] is True
    assert dirty["changed_files"] == ["src/sim.py"]
    assert dirty["diff_sha256"] != clean["diff_sha256"]


def test_untracked_code_file_counts_as_dirty(repo):
    (repo / "src" / "new_feature.py").write_text("X = 1\n")
    p = describe_code_provenance(repo)
    # An untracked module is importable and can change a result, so it must not read clean
    # -- even though `git diff` cannot see it and the diff hash is unchanged.
    assert p["dirty"] is True
    assert p["changed_files"] == ["src/new_feature.py"]


def test_data_refresh_outside_code_paths_does_not_mark_the_run_dirty(repo):
    (repo / "simulation_data" / "REGISTRY.json").write_text('{"refreshed": true}\n')
    p = describe_code_provenance(repo)
    # A dirty flag that fires on every registry refresh gets ignored, and an ignored check
    # protects nothing.
    assert p["dirty"] is False


def test_two_commits_are_distinguishable(repo):
    first = describe_code_provenance(repo)
    (repo / "src" / "sim.py").write_text("RATE = 3\n")
    _git(repo, "commit", "-aqm", "change")
    second = describe_code_provenance(repo)
    assert first["commit"] != second["commit"]
    assert second["dirty"] is False


def test_non_git_tree_is_reported_not_crashed(tmp_path):
    p = describe_code_provenance(tmp_path)
    assert p["git_available"] is False
    assert "git=unavailable" in format_code_banner(p)


def test_banner_is_single_line(repo):
    assert "\n" not in format_code_banner(describe_code_provenance(repo))
