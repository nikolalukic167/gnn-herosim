"""r1a_select_checkpoints: the configuration per arm is the lowest mean validation score over 3 seeds, a test-scored run is refused, and the
staged names are the ones the gate reads."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts_cosim"))
import joint_burst_v2_sidecheck as sc  # noqa: E402
import r1a_select_checkpoints as sel  # noqa: E402


def _stage(tmp_path, arms, vals, test_evaluated=False):
    split = tmp_path / "split.json"
    split.write_text("{}")
    sha = hashlib.sha256(split.read_bytes()).hexdigest()
    models = tmp_path / "models"
    models.mkdir()
    for arm in arms:
        kind, mp_off, raw, conv, phys = sc.RA_ARMS[f"ra_{arm}"]
        graph_free = kind != "gnn"
        side = {"arm_kind": kind, "disable_message_passing": mp_off, "partial_state_contract": "partial_state_v5", "load_seconds": True,
                "exchange_seconds": True, "full_context_ce_weight": 1.0, "dag_alpha_key": "inf", "mp_residual": False, "plan_raw": raw,
                "plan_raw_sum": False, "plan_raw_local": False, "mp_bipartite_edge_conv": conv and not graph_free,
                "mp_bipartite_edge_attr_zero": conv and not phys and not graph_free, "mp_peer_edges": not graph_free,
                "partial_state_feature_dim": 2 if raw else 27, "candidate_slate": "declared_pruning_v1", "physics_env": {"x": "y"},
                "set_heads": 4 if kind == "set_transformer" else None, "split_artifact": {"sha256": sha}}
        for k in range(6):
            for s in (1, 2, 3):
                stem = models / f"{sel.run_stem(arm, k)}-seed{s}"
                stem.with_suffix(".pt").write_bytes(f"{arm}{k}{s}".encode())
                stem.with_suffix(".contract.json").write_text(json.dumps(side))
                stem.with_suffix(".val.json").write_text(json.dumps({"best_val": vals(arm, k, s), "checkpoint_metric": "regret_masked_topo",
                                                                      "test_evaluated": test_evaluated}))
    return split, models


def _run(tmp_path, split, models, arms):
    out = tmp_path / "sel.json"
    argv = ["x", "--models-dir", str(models), "--inputs-dir", str(tmp_path / "inputs"), "--split", str(split), "--out", str(out), "--arms", *arms]
    old = sys.argv
    sys.argv = argv
    try:
        return sel.main(), json.loads(out.read_text())
    finally:
        sys.argv = old


def test_lowest_mean_config_wins_and_checkpoints_are_staged(tmp_path):
    arms = ["gnn_eng", "mlp_same"]
    # gnn_eng: config 3 has the lowest MEAN although config 1 has the single best seed; mlp_same: config 5
    def vals(arm, k, s):
        if arm == "gnn_eng":
            return {3: 1.0}.get(k, 5.0) + (0.1 * s) - (4.9 if (k, s) == (1, 1) else 0.0)
        return 2.0 if k == 5 else 3.0 + k
    split, models = _stage(tmp_path, arms, vals)
    rc, rep = _run(tmp_path, split, models, arms)
    assert rc == 0 and rep["arms"]["gnn_eng"]["config"] == 3 and rep["arms"]["mlp_same"]["config"] == 5
    assert rep["best_learned_arm_by_validation"] == "gnn_eng"
    for s in (1, 2, 3):
        assert (tmp_path / "inputs/models" / f"r1-attribution-v1-gnn-eng-seed{s}.pt").read_bytes() == f"gnn_eng3{s}".encode()
        assert (tmp_path / "inputs/models" / f"r1-attribution-v1-mlp-same-seed{s}.contract.json").is_file()


def test_a_test_scored_run_is_refused(tmp_path):
    split, models = _stage(tmp_path, ["gnn_eng"], lambda a, k, s: 1.0, test_evaluated=True)
    with pytest.raises(SystemExit, match="held-out topologies"):
        _run(tmp_path, split, models, ["gnn_eng"])


def test_a_missing_seed_is_refused(tmp_path):
    split, models = _stage(tmp_path, ["gnn_eng"], lambda a, k, s: 1.0)
    (models / f"{sel.run_stem('gnn_eng', 2)}-seed3.val.json").unlink()
    with pytest.raises(SystemExit, match="no validation record"):
        _run(tmp_path, split, models, ["gnn_eng"])


def test_a_checkpoint_trained_on_another_split_is_refused(tmp_path):
    split, models = _stage(tmp_path, ["gnn_eng"], lambda a, k, s: 1.0)
    split.write_text('{"other": 1}')
    with pytest.raises(SystemExit, match="different split"):
        _run(tmp_path, split, models, ["gnn_eng"])


def test_per_arm_selection_picks_what_the_all_arms_run_picks(tmp_path):
    arms = ["gnn_eng", "twin_eng"]
    vals = lambda arm, k, s: (k - 2) ** 2 + (0.3 if arm == "twin_eng" else 0.0) + 0.01 * s
    split, models = _stage(tmp_path, arms, vals)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    picks = {}
    for arm in arms:
        argv = ["x", "--models-dir", str(models), "--inputs-dir", str(tmp_path / "inputs_arm"), "--split", str(split),
                "--out", str(out_dir / f"selection_{arm}.json"), "--arm", arm]
        old, sys.argv = sys.argv, argv
        try:
            assert sel.main() == 0
        finally:
            sys.argv = old
        picks[arm] = json.loads((out_dir / f"selection_{arm}.json").read_text())["arms"][arm]["config"]
    argv = ["x", "--models-dir", str(models), "--inputs-dir", str(tmp_path / "inputs_all"), "--split", str(split), "--out", str(out_dir / "selection.json"), "--arms", *arms]
    old, sys.argv = sys.argv, argv
    try:
        assert sel.main() == 0
    finally:
        sys.argv = old
    allrun = json.loads((out_dir / "selection.json").read_text())
    assert picks == {a: allrun["arms"][a]["config"] for a in arms} == {"gnn_eng": 2, "twin_eng": 2}
    for a in arms:
        for s in (1, 2, 3):
            n = f"r1-attribution-v1-{a.replace('_', '-')}-seed{s}.pt"
            assert (tmp_path / "inputs_arm/models" / n).read_bytes() == (tmp_path / "inputs_all/models" / n).read_bytes()


def test_if_complete_exits_quietly_until_all_18_runs_are_scored(tmp_path, capsys):
    split, models = _stage(tmp_path, ["mlp_same"], lambda a, k, s: 1.0)
    (models / f"{sel.run_stem('mlp_same', 5)}-seed2.val.json").unlink()
    argv = ["x", "--models-dir", str(models), "--inputs-dir", str(tmp_path / "inputs"), "--split", str(split), "--out", str(tmp_path / "s.json"),
            "--arm", "mlp_same", "--if-complete"]
    old, sys.argv = sys.argv, argv
    try:
        assert sel.main() == 3
    finally:
        sys.argv = old
    assert not (tmp_path / "inputs").exists()
