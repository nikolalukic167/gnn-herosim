import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
import frozen_import as F  # noqa: E402


def commit(repo, body, msg):
    (repo / "m.py").write_text(body)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", msg], check=True)
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD:m.py"], check=True, capture_output=True, text=True).stdout.strip()


def test_loads_the_frozen_blob_not_the_working_file(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    old = commit(tmp_path, "X = 1\n", "frozen")
    new = commit(tmp_path, "X = 1\nY = 2\n", "additions")
    m_old = F.load_frozen(str(tmp_path), "m.py", old)
    assert m_old.X == 1 and not hasattr(m_old, "Y") and m_old.__frozen_blob__ == old and m_old.__name__ == f"m@{old[:8]}"
    m_new = F.load_frozen(str(tmp_path), "m.py", new)
    assert m_new.Y == 2 and m_new is not m_old and "m" not in sys.modules


def test_unknown_blob_fails_loud(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    commit(tmp_path, "X = 1\n", "only")
    with pytest.raises(SystemExit, match="no such blob"):
        F.load_frozen(str(tmp_path), "m.py", "0" * 40)


def test_the_three_study_readers_load_the_p9_blob():
    here = os.path.join(os.path.dirname(__file__), "..", "scripts_cosim")
    g = F.load_frozen(here, "uncut_arm_gate_read.py", "04e368af8fbd9eb462ae68945676cc850ea808a9")
    assert not hasattr(g, "read_own_state") and hasattr(g, "read_mp_twin") and hasattr(g, "contrast")
