"""extend_corpus_slate: the plan bookkeeping, the reconstruction checks and the untouched-dataset guarantee run anywhere; the replay checks
(a-c) need a built accel corpus and the physics env, so they run only where EXTEND_TEST_CORPUS names one (datalab)."""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts_cosim import extend_corpus_slate as X  # noqa: E402
from scripts_cosim.make_warm_corpus import flag_candidates_per_task  # noqa: E402


def _cand(node, plat, drain=0.0):
    return {"node_id": node, "node_name": f"node{node}", "platform_id": plat, "queue_key": f"node{node}:{plat}", "queue_drain_seconds": drain,
            "platform_type": "rpiCpu"}


def _raw(sizes):
    tasks = [{"task_id": 10 + i, "task_type": "dnn1", "candidates": [_cand(n, 100 + n) for n in range(s)], "candidate_count": s}
             for i, s in enumerate(sizes)]
    specs = [{"node_name": f"node{n}", "platform_id": 100 + n, "queue_key": f"node{n}:{100 + n}", "candidate": True} for n in range(max(sizes))]
    return {"snapshot_id": 1, "tasks": tasks, "replicas_by_type": {"dnn1": specs}, "fidelity": {"batch": []}}


def _dataset(root: Path, name: str, sizes, top=2):
    raw = _raw(sizes)
    kept = {10 + i: {f"node{n}:{100 + n}" for n in range(min(s, top))} for i, s in enumerate(sizes)}
    flagged = flag_candidates_per_task(raw, kept)
    ids = [10 + i for i in range(len(sizes))]
    tc = [[[int(c["node_id"]), int(c["platform_id"])] for c in t["candidates"]] for t in flagged["tasks"]]
    d = root / name
    (d / "placements").mkdir(parents=True)
    spec = {"snapshot": raw, "task_candidates": tc, "sim_input": "/x", "candidate_slate": "declared_pruning_v1"}
    (d / "infrastructure.json").write_text(json.dumps({"live_snapshot_seed": {"fidelity_replay": spec}, "metadata": {}}))
    old_plans = 1
    for k in kept.values():
        old_plans *= len(k)
    draw = {"rule": "declared_pruning_v1", "plans": old_plans, "task_ids": ids, "per_task_candidates": [len(kept[i]) for i in ids]}
    (d / "warm_snapshot.json").write_text(json.dumps({"snapshot": flagged, "provenance": {"candidate_draw": draw, "cell_seed": 1, "source_tag": "t"}}))
    (d / "workload.json").write_text(json.dumps({"trace_task_ids": ids, "events": [{}] * len(ids)}))
    (d / "best.json").write_text(json.dumps({"file": "simulation_1_optimal.json", "rtt": 1.0}))
    (d / "placements" / "placements.jsonl").write_text("")
    return d


def test_plan_key_ignores_key_type_and_order():
    a = {"0": [1, 2], "1": [3, 4]}
    b = {1: (3, 4), 0: (1, 2)}
    assert X.plan_key(a) == X.plan_key(b)
    assert X.plan_combo(a) == ((1, 2), (3, 4))


def test_best_of_breaks_ties_by_plan():
    rows = [("l1", {"placement_plan": {"0": [2, 2]}, "rtt": 1.0}), ("l2", {"placement_plan": {"0": [1, 9]}, "rtt": 1.0}), ("l3", {"placement_plan": {"0": [0, 0]}, "rtt": 2.0})]
    assert X.best_of(rows) == (1.0, ((1, 9),))


def test_rebuild_checks_accept_a_consistent_dataset_and_refuse_a_foreign_slate(tmp_path):
    d = _dataset(tmp_path, "ds_00000", [4, 3, 2])
    v = X.slate_view(d)
    assert v["full_plans"] == 24 and v["old_plans"] == 2 * 2 * 2
    X.rebuild_checks(v)
    v["ws"]["snapshot"]["tasks"][0]["candidates"] = [_cand(9, 109)]   # not a subset of the raw candidates
    with pytest.raises(RuntimeError, match="not a subset"):
        X.rebuild_checks(v)


def test_provenance_records_the_old_slate_and_the_counts(tmp_path):
    d = _dataset(tmp_path, "ds_00000", [4, 3, 2])
    v = X.slate_view(d)
    prov = X.new_provenance(v, 8, 24)
    cd = prov["candidate_draw"]
    assert cd["rule"] == "declared_pruning_v1_uncapped" and cd["plans"] == 24 and cd["plans_before"] == 8 and cd["plans_after"] == 24
    assert cd["old_slate"]["rule"] == "declared_pruning_v1" and cd["old_slate"]["per_task_candidates"] == [2, 2, 2]
    assert cd["per_task_candidates"] == [4, 3, 2] and cd["full_plans"] == 24 and cd["pruned"] is False


def _tree_hash(root: Path):
    h = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return h


def test_a_dataset_above_the_cap_is_untouched_byte_for_byte_and_listed(tmp_path, monkeypatch):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    big = _dataset(src, "ds_00000", [4, 3, 2])         # 24 full plans
    _dataset(src, "ds_00001", [2, 2, 2])        # never pruned: already full
    before = _tree_hash(src)
    monkeypatch.setattr(sys, "argv", ["x", "--src", str(src), "--out", str(out), "--max-plans", "23", "--link-untouched"])
    assert X.main() == 0
    assert _tree_hash(src) == before
    rows = [json.loads(l) for l in (out / "extend_manifest.jsonl").read_text().splitlines()]
    assert {r["dataset"]: r["status"] for r in rows} == {"ds_00000": "untouched_over_cap", "ds_00001": "untouched_already_full"}
    assert (out / "ds_00000").is_symlink() and _tree_hash(out / "ds_00000") == _tree_hash(big)


def test_out_inside_src_is_refused(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    monkeypatch.setattr(sys, "argv", ["x", "--src", str(src), "--out", str(src / "new")])
    with pytest.raises(SystemExit, match="new directory"):
        X.main()


def test_census_counts_what_would_be_extended_and_writes_nothing_into_src(tmp_path, monkeypatch):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    _dataset(src, "ds_00000", [4, 3, 2])
    before = _tree_hash(src)
    monkeypatch.setattr(sys, "argv", ["x", "--src", str(src), "--out", str(out), "--census"])
    assert X.main() == 0
    rec = json.loads((out / "extend_census.jsonl").read_text())
    assert rec["status"] == "would_extend" and rec["new_plans"] == 24 - 8
    assert _tree_hash(src) == before


# ---- replay checks (a)-(c): need a built corpus and the physics env -------------------------------------------------------------
CORPUS = os.environ.get("EXTEND_TEST_CORPUS")
needs_corpus = pytest.mark.skipif(not CORPUS, reason="EXTEND_TEST_CORPUS (a built accel corpus + physics env) not set")


@needs_corpus
def test_replay_checks_on_a_scratch_copy():
    """EXTEND_TEST_CORPUS = the extended scratch --out directory the smoke wrote (from EXTEND_TEST_SRC, the source corpus)."""
    out = Path(CORPUS)
    src = Path(os.environ["EXTEND_TEST_SRC"])
    manifest = [json.loads(l) for l in (out / "extend_manifest.jsonl").read_text().splitlines()]
    done = [r for r in manifest if r["status"] == "extended"]
    assert done, "the smoke extended nothing"
    for r in done:
        new = out / r["dataset"]
        old_rows = X.read_rows(src / r["dataset"] / "placements" / "placements.jsonl")
        new_rows = X.read_rows(new / "placements" / "placements.jsonl")
        # (a) the stored plans re-scored on the extended infrastructure reproduce their rtt (the run's own check, repeated from the manifest)
        assert r["verified"]["n"] >= min(20, len(old_rows)) and r["verified"]["max_abs_diff"] <= 1e-9
        # (b) the old slate is a subset of the new placements, row for row
        assert {l for l, _ in old_rows} <= {l for l, _ in new_rows} and len(new_rows) == r["plans_after"] > len(old_rows)
        # (c) the argmin over the new set is at most the old best
        assert X.best_of(new_rows)[0] <= X.best_of(old_rows)[0] + 1e-12
        assert abs(json.loads((new / "best.json").read_text())["rtt"] - X.best_of(new_rows)[0]) < 1e-12
        ws = json.loads((new / "warm_snapshot.json").read_text())
        assert ws["provenance"]["candidate_draw"]["rule"] == "declared_pruning_v1_uncapped"
    # (d) the source corpus is unchanged: nothing under src was modified after the smoke started is checked by the caller's hash file
