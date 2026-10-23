"""llf_label_merge: duplicate rows must be equal, an unfinished state carries its first-pass label with a flag, trainer keys only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import llf_label_merge as M  # noqa: E402
import uncut_label_screen as L  # noqa: E402


def _row(gid, **kw):
    base = {"gid": gid, "plans": [[[1, 2]]], "cd_rtt": 3.0, "split": "train", "bucket": "le10", "n_tasks": 1, "method": "exhaustive", "current_label_source": "hand", "differs": False, "gain": 0.0,
            "rtt_current": 3.0, "rtt_sim": 3.0, "n_plans_replayed": 2, "current_plan": [[1, 2]], "best_tag": "label", "stored_label_rtt": 2.5, "stored_flows": True, "wall_s": 1.0,
            "winner_pool": "first_pass", "sfs_wall_s": 0.0, "hand_replan_differs": False}
    base.update(kw)
    return base


@pytest.fixture
def world(tmp_path, monkeypatch):
    states = [{"gid": "p/ds_1", "split": "train", "bucket": "le10", "small": True}, {"gid": "p/ds_2", "split": "train", "bucket": "le10", "small": True}]
    monkeypatch.setattr(L, "all_states", lambda split, corpus: states)
    monkeypatch.setattr(L, "_dataset_dir", lambda corpus, gid: f"{tmp_path}/{gid}")
    monkeypatch.setattr(L, "_stored_rows", lambda ds: ({"n_tasks": 1, "label_source": "hand", "state_id": "s", "trace_task_ids": [7]}, [{"rtt": 2.5, "placement_plan": {"0": [4, 5]}}]))
    states_json = tmp_path / "states.json"
    states_json.write_text(json.dumps({"states": [{"key": "s", "snapfile": str(tmp_path / "snap"), "trigger": 1}]}))
    (tmp_path / "snap.chosen.jsonl").write_text(json.dumps({"trigger_task_id": 1, "chosen": [[7, 4, 5]]}) + "\n")
    return tmp_path, states_json


def _run(tmp_path, states_json, files, extra=()):
    argv = ["--split", "x", "--corpus", "c", "--states", str(states_json), "--in", *map(str, files), "--out", str(tmp_path / "o.json"), "--stats", str(tmp_path / "st.json"), *extra]
    M.main(argv)
    return json.load(open(tmp_path / "o.json")), json.load(open(tmp_path / "st.json"))


def test_duplicates_must_be_equal_and_trainer_keys_only(world):
    tmp, sj = world
    a, b = tmp / "a.jsonl", tmp / "b.jsonl"
    a.write_text("\n".join(json.dumps(_row(g)) for g in ("p/ds_1", "p/ds_2")) + "\n")
    b.write_text(json.dumps(_row("p/ds_2", wall_s=99.0)) + "\n")            # same row, other wall time: accepted
    out, _ = _run(tmp, sj, [a, b])
    assert out["_meta"]["n"] == 2 and all(k not in out["p/ds_1"] for k in ("stored_label_rtt", "stored_flows", "wall_s", "winner_pool", "hand_replan_differs"))
    b.write_text(json.dumps(_row("p/ds_2", cd_rtt=9.0)) + "\n")             # a different row for the same state: refused
    with pytest.raises(SystemExit, match="twice with different rows"):
        _run(tmp, sj, [a, b])


def test_an_unfinished_state_carries_its_first_pass_label_with_a_flag(world):
    tmp, sj = world
    a = tmp / "a.jsonl"
    a.write_text(json.dumps(_row("p/ds_1")) + "\n")
    out, st = _run(tmp, sj, [a], ["--carry-first-pass", "p/ds_2"])
    e = out["p/ds_2"]
    assert e["relabel_unfinished"] is True and e["plans"] == [[[4, 5]]] and e["cd_rtt"] == 2.5 and e["method"] == "first_pass_carried" and "relabel_unfinished" not in out["p/ds_1"]
    assert out["_meta"]["relabel_unfinished"] == ["p/ds_2"] and st["n"] == 2
    with pytest.raises(SystemExit, match="listed in --carry-first-pass"):
        _run(tmp, sj, [a], ["--carry-first-pass", "p/ds_1"])
