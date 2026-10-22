import json

import pytest

from scripts_cosim import dagger_corpus as D
from scripts_cosim import uncut_arm1 as A


def _snaps(path, caps):
    with open(path, "w") as fh:
        for trig, t, n in caps:
            fh.write(json.dumps({"trigger_task_id": trig, "time": t, "fidelity": {"batch": [{"gid": i} for i in range(n)]}}) + "\n")


def _cells(tmp_path, caps, rungs=("moderate", "heavy")):
    cells, inputs = tmp_path / "cells", tmp_path / "in"
    cells.mkdir()
    for r in rungs:
        (inputs / f"wf1_{r}" / "cfg").mkdir(parents=True)
        (inputs / f"wf1_{r}" / "wl").mkdir(parents=True)
        (inputs / f"wf1_{r}" / "cfg" / "cc40s16201.json").write_text("{}")
        (inputs / f"wf1_{r}" / "wl" / "grounded_g0_n50000.json").write_text("{}")
        stem = cells / f"cc40s16201__g0{r}__ra_twin_eng_combfl_nosplit_k100_s1"
        _snaps(str(stem) + ".snapshots.jsonl", caps)
        (cells / (stem.name + ".summary.json")).write_text("{}")
    return cells, inputs


def test_states_take_first_capture_buckets_and_skip_over_256(tmp_path):
    cells, inputs = _cells(tmp_path, [(0, 5.0, 4), (3, 400.0, 30), (3, 400.01, 12), (9, 800.0, 300)])
    plan = {"16201__g0": [0, 3, 9]}
    r = D.make_states("twin", cells, "ra_twin_eng_combfl_nosplit_k100", 1, plan, inputs, [16201])
    by = {s["key"]: s for s in r["states"]}
    assert len(r["states"]) == 4 and r["groups_over_256"] == 2 and not r["missing"]       # 2 rungs x (trigger 0, trigger 3); the 300-task group is counted, not labelled
    s3 = by["dagger_twin:16201:g0:heavy:3"]
    assert s3["n_batch"] == 30 and s3["bucket"] == "11_50" and not s3["early"] and s3["check_arm"] is False
    assert by["dagger_twin:16201:g0:moderate:0"]["early"] and by["dagger_twin:16201:g0:moderate:0"]["bucket"] == "le10"
    assert s3["cell_summary"].endswith("_s1.summary.json") and s3["cell_config"].endswith("cc40s16201.json") and s3["trace"].endswith("grounded_g0_n50000.json")


def test_job_of_uses_the_states_own_cell_files_and_check_arm_flag(tmp_path):
    cells, inputs = _cells(tmp_path, [(0, 5.0, 4)])
    st = D.make_states("twin", cells, "ra_twin_eng_combfl_nosplit_k100", 1, {"16201__g0": [0]}, inputs, [16201])["states"][0]
    j = A.job_of(st, "/nowhere")
    assert j["cell_summary"] == st["cell_summary"] and j["cell_config"] == st["cell_config"] and j["check_arm"] is False and j["group_size"] == 4
    plain = A.job_of({k: v for k, v in st.items() if k not in ("cell_summary", "cell_config", "check_arm")}, "/root")
    assert plain["cell_summary"] == "/root/cap/A/cc40s16201__g0moderate__cd_ib0_s0.summary.json" and "check_arm" not in plain      # Arm 1 jobs are unchanged


def test_a_planned_trigger_that_was_never_captured_fails_loud_unless_allowed(tmp_path):
    cells, inputs = _cells(tmp_path, [(0, 5.0, 4)])
    with pytest.raises(SystemExit, match="never captured"):
        D.make_states("twin", cells, "ra_twin_eng_combfl_nosplit_k100", 1, {"16201__g0": [0, 6]}, inputs, [16201])
    r = D.make_states("twin", cells, "ra_twin_eng_combfl_nosplit_k100", 1, {"16201__g0": [0, 6]}, inputs, [16201], allow_missing=True)
    assert len(r["missing"]) == 2 and len(r["states"]) == 2


def test_a_topology_outside_the_training_set_fails_loud(tmp_path):
    cells, inputs = _cells(tmp_path, [(0, 5.0, 4)])
    with pytest.raises(SystemExit, match="not one of the allowed"):
        D.make_states("twin", cells, "t", 1, {"30149__g0": [0]}, inputs, [16201])


def _corpus(tmp_path, arm, n):
    for i in range(n):
        d = tmp_path / "new" / f"dagger_{arm}" / f"ds_{i:05d}" / "placements"
        d.mkdir(parents=True)
        (d / "placements.jsonl").write_text("{}\n")
    (tmp_path / "new" / f"dagger_{arm}" / "ds_99999").mkdir()           # a dataset without its placements file is not a dataset


def test_split_adds_new_datasets_to_train_only_and_new_only_frame(tmp_path):
    _corpus(tmp_path, "twin", 3)
    old = {"schema": "split_artifact_v1", "n_parents": 6, "train": ["gnn_datasets_wf1_train/ds_00000", "gnn_datasets_wf1_train/ds_00001"], "val": ["gnn_datasets_wf1_heldout/ds_00000", "gnn_datasets_wf1_heldout/ds_00001"],
           "test": ["gnn_datasets_wf1_train/ds_00007", "gnn_datasets_wf1_train/ds_00008"], "cache_dir": "x"}
    parents = D.new_parents(tmp_path / "new", "twin")
    assert parents == [f"gnn_datasets_wf1_dagger_twin/ds_{i:05d}" for i in range(3)]
    c = D.combined_split(old, parents, False)
    assert c["n_parents"] == 9 and len(c["train"]) == 5 and c["val"] == old["val"] and c["test"] == old["test"] and c["dagger"]["new_train"] == 3
    n = D.combined_split(old, parents, True)
    assert n["train"] == parents and n["n_parents"] == 3 and n["val"] == [] and n["test"] == []
    with pytest.raises(SystemExit, match="share an id"):
        D.combined_split({**old, "train": old["train"] + parents[:1]}, parents, False)
    out = tmp_path / "s.json"
    sha = D.write_split(c, out)
    assert len(sha) == 64 and json.loads(out.read_text())["n_parents"] == 9


def test_view_links_old_base_dirs_and_the_new_one(tmp_path):
    _corpus(tmp_path, "twin", 2)
    old = tmp_path / "oldview"
    (old / "gnn_datasets_wf1_train" / "ds_00000").mkdir(parents=True)
    (old / "gnn_datasets_wf1_heldout").mkdir()
    names = D.make_view(old, tmp_path / "new", "twin", tmp_path / "comb")
    assert names == ["gnn_datasets_wf1_heldout", "gnn_datasets_wf1_train", "gnn_datasets_wf1_dagger_twin"]
    assert (tmp_path / "comb" / "gnn_datasets_wf1_dagger_twin" / "ds_00000" / "placements" / "placements.jsonl").exists()
    with pytest.raises(SystemExit, match="exists"):
        D.make_view(old, tmp_path / "new", "twin", tmp_path / "comb")
