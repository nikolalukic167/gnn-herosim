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
    plan = {"16201__g0moderate": [0, 3, 9], "16201__g0heavy": [0, 3, 9]}
    r = D.make_states("twin", cells, "ra_twin_eng_combfl_nosplit_k100", 1, plan, inputs, [16201])
    by = {s["key"]: s for s in r["states"]}
    assert len(r["states"]) == 4 and r["groups_over_256"] == 2 and not r["missing"]       # 2 rungs x (trigger 0, trigger 3); the 300-task group is counted, not labelled
    s3 = by["dagger_twin:16201:g0:heavy:3"]
    assert s3["n_batch"] == 30 and s3["bucket"] == "11_50" and not s3["early"] and s3["check_arm"] is False
    assert by["dagger_twin:16201:g0:moderate:0"]["early"] and by["dagger_twin:16201:g0:moderate:0"]["bucket"] == "le10"
    assert s3["cell_summary"].endswith("_s1.summary.json") and s3["cell_config"].endswith("cc40s16201.json") and s3["trace"].endswith("grounded_g0_n50000.json")


def test_job_of_uses_the_states_own_cell_files_and_check_arm_flag(tmp_path):
    cells, inputs = _cells(tmp_path, [(0, 5.0, 4)])
    st = D.make_states("twin", cells, "ra_twin_eng_combfl_nosplit_k100", 1, {"16201__g0moderate": [0]}, inputs, [16201])["states"][0]
    j = A.job_of(st, "/nowhere")
    assert j["cell_summary"] == st["cell_summary"] and j["cell_config"] == st["cell_config"] and j["check_arm"] is False and j["group_size"] == 4
    plain = A.job_of({k: v for k, v in st.items() if k not in ("cell_summary", "cell_config", "check_arm")}, "/root")
    assert plain["cell_summary"] == "/root/cap/A/cc40s16201__g0moderate__cd_ib0_s0.summary.json" and "check_arm" not in plain      # Arm 1 jobs are unchanged


def test_uncaptured_triggers_are_counted_and_below_80_percent_stops(tmp_path):
    cells, inputs = _cells(tmp_path, [(i, 5.0 + i, 4) for i in range(10)])
    plan = {"16201__g0moderate": list(range(9)) + [99]}                                          # 9 of 10 captured: reported, not fatal, no top-up
    r = D.make_states("twin", cells, "ra_twin_eng_combfl_nosplit_k100", 1, plan, inputs, [16201])
    assert len(r["missing"]) == 1 and len(r["states"]) == 9 and r["captured_share"] == 0.9 and r["drawn"] == 10
    with pytest.raises(SystemExit, match="stop labelling and report to SCIENTIST"):
        D.make_states("twin", cells, "ra_twin_eng_combfl_nosplit_k100", 1, {"16201__g0moderate": list(range(7)) + [90, 91, 92]}, inputs, [16201])      # 7 of 10


def test_a_plan_key_without_a_rung_fails_loud(tmp_path):
    cells, inputs = _cells(tmp_path, [(0, 5.0, 4)])
    with pytest.raises(SystemExit, match="no rung suffix"):
        D.make_states("twin", cells, "t", 1, {"16201__g0": [0]}, inputs, [16201])


def test_a_topology_outside_the_training_set_fails_loud(tmp_path):
    cells, inputs = _cells(tmp_path, [(0, 5.0, 4)])
    with pytest.raises(SystemExit, match="not one of the allowed"):
        D.make_states("twin", cells, "t", 1, {"30149__g0moderate": [0]}, inputs, [16201])


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


def _wl(sizes, spacing=10.0):
    ev, pairs, t = [], [], 0.0
    for g, n in enumerate(sizes):
        for _ in range(n):
            ev.append({"peer_group": g, "timestamp": t})
        t += spacing
    return {"events": ev, "peer_exchange": pairs}


def _wls(topos=(16201, 16202)):
    sizes = ([3] * 30 + [20] * 30 + [70] * 10 + [150] * 4 + [300] * 2) * 8            # 76 groups per 10 s: ~30 % under 360 s per bucket; the 300-task groups are over 256
    return {(t, w, r): _wl(sizes) for t in topos for w in D.WINDOWS for r in D.RUNGS}


def test_draw_plan_total_strata_and_same_ids_for_every_call():
    wls = _wls()
    r = D.draw_plan(wls, 400, seed=7)
    assert r["total_drawn"] + r["shortfall"] == 400 and set(r["plan"]) == {f"{t}__{w}{x}" for t in (16201, 16202) for w in D.WINDOWS for x in D.RUNGS}
    assert D.draw_plan(wls, 400, seed=7)["plan"] == r["plan"] and D.draw_plan(wls, 400, seed=8)["plan"] != r["plan"]
    from scripts_cosim.uncut_arm1 import groups_of

    gs = {t: (size, at) for t, size, at in groups_of(wls[(16201, "g0", "moderate")])}
    for key, ids in r["plan"].items():
        assert ids == sorted(set(ids)) and all(gs[i][0] <= 256 for i in ids)                  # never a group over 256 tasks
    bucket_counts = {}
    for c in r["census"]:
        bucket_counts[c["bucket"]] = bucket_counts.get(c["bucket"], 0) + c["drawn"]
    assert sum(bucket_counts.values()) == r["total_drawn"] and bucket_counts["le10"] > bucket_counts["11_50"] > bucket_counts["51_100"] > bucket_counts["101_256"]
    early = sum(c["drawn"] for c in r["census"] if c["early"])
    assert 0.25 < early / r["total_drawn"] < 0.35


def test_draw_plan_reports_a_shortfall_and_does_not_top_up():
    wls = {(16201, w, r): _wl([3] * 5 + [20] * 5) for w in D.WINDOWS for r in D.RUNGS}        # 10 groups per cell, none above 20 tasks
    r = D.draw_plan(wls, 2000, seed=1)
    assert r["shortfall"] > 1000 and r["total_drawn"] <= 10 * 8 and all(len(v) <= 10 for v in r["plan"].values())


def test_largest_remainder_sums_to_total_and_breaks_ties_by_key():
    q = D._largest_remainder(10, {"a": 1, "b": 1, "c": 1})
    assert sum(q.values()) == 10 and q == {"a": 4, "b": 3, "c": 3}


def test_the_draw_is_identical_across_arms_and_deterministic_at_the_registered_seed():
    wls = _wls((16201, 16202, 16203, 16204))
    r1, r2 = D.draw_plan(wls), D.draw_plan(wls)                  # the draw has no arm argument: both arms read the same plan file
    assert D.SEED == 20261022 and D.TOTAL == 1000 and r1["plan"] == r2["plan"] and r1["seed"] == 20261022
    assert D.draw_plan(wls, seed=D.SEED + 1)["plan"] != r1["plan"]
    assert abs(r1["per_rung"]["moderate"] - r1["per_rung"]["heavy"]) <= 4 and r1["total_drawn"] <= 1000          # half per load, to within the quota rounding


def test_the_draw_meets_quotas_has_nothing_above_256_and_the_early_floor():
    wls = _wls((16201, 16202, 16203, 16204))
    r = D.draw_plan(wls)
    assert r["quotas"] == {"le10": 400, "11_50": 383, "51_100": 167, "101_256": 50} or sum(r["quotas"].values()) == 1000
    assert r["quotas"]["le10"] > r["quotas"]["11_50"] > r["quotas"]["51_100"] > r["quotas"]["101_256"] and abs(r["quotas"]["le10"] - 400) <= 1 and abs(r["quotas"]["101_256"] - 50) <= 1
    from scripts_cosim.uncut_arm1 import groups_of

    size = {t: n for t, n, _ in groups_of(_wl(([3] * 30 + [20] * 30 + [70] * 10 + [150] * 4 + [300] * 2) * 8))}
    assert all(size[i] <= 256 for ids in r["plan"].values() for i in ids)
    by_bucket = {}
    for c in r["census"]:
        by_bucket[c["bucket"]] = by_bucket.get(c["bucket"], 0) + c["drawn"]
    assert all(by_bucket[b] <= r["quotas"][b] for b in by_bucket) and r["early_floor_ok"] and r["early_share"] >= D.EARLY_FLOOR


def test_k1_commit_is_read_from_the_arm1_label_script_and_asserted(tmp_path):
    f = tmp_path / "labels.sbatch"
    f.write_text("python3 uncut_arm1.py label ids.json --shard 0/24 --k1-read-commit ca47cb62 2>&1\n")
    assert D.k1_commit_of_arm1(str(f), "ca47cb62") == "ca47cb62"
    with pytest.raises(SystemExit, match="expected exactly"):
        D.k1_commit_of_arm1(str(f), "1ccb4487")
    f.write_text("--k1-read-commit ca47cb62 ... --k1-read-commit 1ccb4487\n")
    with pytest.raises(SystemExit, match="expected exactly"):
        D.k1_commit_of_arm1(str(f), "ca47cb62")


def test_label_report_counts_mismatches_and_flags_every_drop():
    rows = [{"state_id": "a", "source": "exact"}, {"state_id": "b", "source": "hand", "chosen_match": False}, {"state_id": "c", "source": "ils", "chosen_match": True},
            {"state_id": "d", "source": "dropped", "reason": "OTHER", "error": "FAIL LOUD x"}]
    r = D.label_report(rows)
    assert r["by_source"] == {"exact": 1, "hand": 1, "ils": 1, "dropped": 1} and r["served_vs_redecision"] == {"checked": 2, "mismatched": 1}
    assert r["dropped"][0]["reason"] == "OTHER" and r["ok"] is False
    assert D.label_report(rows[:3])["ok"] is True


def test_multi_arm_view_and_split_and_merged_labels(tmp_path):
    _corpus(tmp_path, "twin_s1", 2)
    _corpus(tmp_path, "twin_s2", 3)
    parents = D.new_parents(tmp_path / "new", ["twin_s1", "twin_s2"])
    assert len(parents) == 5 and parents[0].startswith("gnn_datasets_wf1_dagger_twin_s1/") and parents[-1].startswith("gnn_datasets_wf1_dagger_twin_s2/")
    old = tmp_path / "oldview"
    (old / "gnn_datasets_wf1_train").mkdir(parents=True)
    assert D.make_view(old, tmp_path / "new", ["twin_s1", "twin_s2"], tmp_path / "comb")[-2:] == ["gnn_datasets_wf1_dagger_twin_s1", "gnn_datasets_wf1_dagger_twin_s2"]
    o, n = tmp_path / "o.json", tmp_path / "n.json"
    o.write_text(json.dumps({"_meta": {"v": 1}, "gnn_datasets_wf1_train/ds_00000": {"plans": [[[1, 2]]], "split": "train"}}))
    n.write_text(json.dumps({"_meta": {"v": 2}, parents[0]: {"plans": [[[3, 4]]], "split": "train"}}))
    r = D.merge_labels(o, n, tmp_path / "m.json")
    m = json.loads((tmp_path / "m.json").read_text())
    assert r["n_old"] == 1 and r["n_new"] == 1 and set(m) == {"_meta", "gnn_datasets_wf1_train/ds_00000", parents[0]}
    n.write_text(json.dumps({"gnn_datasets_wf1_train/ds_00000": {}}))
    with pytest.raises(SystemExit, match="in both label jsons"):
        D.merge_labels(o, n, tmp_path / "m2.json")


def test_check_cache_passes_an_unchanged_old_graph_set_and_fails_a_changed_one(tmp_path):
    import pickle

    import torch
    from torch_geometric.data import Data

    def g(v):
        return Data(x=torch.tensor([[v, 2.0]]), edge_index=torch.zeros((2, 0), dtype=torch.long))

    for name, graphs, ids in (("old", [g(1.0), g(3.0)], ["a", "b"]), ("same", [g(3.0), g(1.0), g(5.0)], ["b", "a", "c"]), ("changed", [g(1.0), g(3.5), g(5.0)], ["a", "b", "c"])):
        d = tmp_path / name
        d.mkdir()
        pickle.dump(graphs, open(d / "graphs.pkl", "wb"))
        pickle.dump(ids, open(d / "dataset_ids.pkl", "wb"))
    r = D.check_cache(tmp_path / "old", tmp_path / "same")
    assert r == {"old_graphs": 2, "new_graphs": 3, "added": 1, "old_graphs_identical": True}
    with pytest.raises(SystemExit, match="old graphs differ"):
        D.check_cache(tmp_path / "old", tmp_path / "changed")
