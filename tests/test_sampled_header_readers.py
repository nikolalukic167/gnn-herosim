import json

import pytest

from src.notebooks.non_unique_lib.training_contract import load_sweep_minimum
from src.placement.sweep_status import sweep_status

HEADER = {"header": True, "exhaustive": False, "plan_space_size": 30}
ROW = {"placement_plan": {"0": [1, 2], "1": [1, 3]}, "rtt": 4.5}


def _ds(tmp_path, lines, n):
    (tmp_path / "placements").mkdir()
    (tmp_path / "placements" / "placements.jsonl").write_text("\n".join(json.dumps(x, separators=(",", ":")) for x in lines) + "\n")
    (tmp_path / "placement_metadata.json").write_text(json.dumps({"sweep_complete": True, "num_placements": n}))
    (tmp_path / "best.json").write_text("{}x")
    return tmp_path


def test_header_line_is_not_a_plan(tmp_path):
    d = _ds(tmp_path, [HEADER, ROW], 1)
    assert sweep_status(d) == (True, "complete")
    plan, rtt, _ = load_sweep_minimum(d / "placements" / "placements.jsonl")
    assert rtt == 4.5


def test_row_count_still_checked_with_header(tmp_path):
    assert sweep_status(_ds(tmp_path, [HEADER, ROW], 2))[0] is False


def test_headerless_unchanged(tmp_path):
    assert sweep_status(_ds(tmp_path, [ROW], 1))[0] is True
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"rtt": 1.0}) + "\n")
    with pytest.raises(RuntimeError):
        load_sweep_minimum(bad)
