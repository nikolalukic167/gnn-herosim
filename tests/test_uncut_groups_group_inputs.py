"""uncut_groups_s0_v1: the whole-group inputs differ from the minted ones only in each cfg's scheduler.batch_size."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
import uncut_groups_group_inputs as G  # noqa: E402


def _tree(root, bs=10):
    for tag in ("moderate", "heavy"):
        os.makedirs(root / f"wf1_{tag}" / "cfg"); os.makedirs(root / f"wf1_{tag}" / "wl")
        (root / f"wf1_{tag}" / "cfg" / "cc40s1.json").write_text(json.dumps({"scheduler": {"batch_size": bs, "batch_timeout": 1.0}, "nodes": {"x": 1}}))
        (root / f"wf1_{tag}" / "wl" / "grounded_g0_n50000.json").write_text('{"events": []}')
    (root / "manifest_heavy.json").write_text("{}")


def test_only_batch_size_changes_and_both_trees_get_a_manifest(tmp_path):
    src, dst = tmp_path / "inputs", tmp_path / "inputs_group"
    _tree(src)
    assert G.main(["--src", str(src), "--dst", str(dst)]) == 0
    cfg = json.load(open(dst / "wf1_heavy" / "cfg" / "cc40s1.json"))
    assert cfg == {"scheduler": {"batch_size": "group", "batch_timeout": 1.0}, "nodes": {"x": 1}}
    assert (dst / "wf1_heavy" / "wl" / "grounded_g0_n50000.json").read_bytes() == (src / "wf1_heavy" / "wl" / "grounded_g0_n50000.json").read_bytes()
    s, d = (src / "MD5SUMS").read_text().splitlines(), (dst / "MD5SUMS").read_text().splitlines()
    diff = [b.split("  ", 1)[1] for a, b in zip(s, d) if a != b]
    assert len(s) == len(d) == 5 and sorted(diff) == ["wf1_heavy/cfg/cc40s1.json", "wf1_moderate/cfg/cc40s1.json"]
    with pytest.raises(SystemExit, match="exists"):
        G.main(["--src", str(src), "--dst", str(dst)])


def test_refuses_a_cfg_that_is_not_the_minted_10(tmp_path):
    src = tmp_path / "inputs"
    _tree(src, bs=16)
    with pytest.raises(SystemExit, match="FAIL LOUD"):
        G.main(["--src", str(src), "--dst", str(tmp_path / "out")])
