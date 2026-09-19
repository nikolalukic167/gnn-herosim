"""route_b stage 2, W4 — scripts_cosim/eval_route_b_stage2_arm.py.

Three things pinned here:

1. The self-check positive control reaches 1e-9 acceptance (the same target as
   verify_route_b_scorer_agreement.py --check-decoder) by construction, since it
   dispatches straight into that function rather than a re-typed copy.
2. Sidecar-less checkpoints are refused for BOTH arm families (MLP: no
   inference_feature_layout; GNN: no .contract.json) — a checkpoint without a
   contract is not evidence (stage-1 rule).
3. --aggregate's pooled-sigma and per-draw-regret arithmetic, on small
   hand-built per-draw reports where the answer is known by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts_cosim")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eval_route_b_stage2_arm as ev  # noqa: E402

CACHE_DIR = REPO_ROOT / "simulation_data" / "graphs_cache_route_b_pilot_s_dag"
CORPUS = REPO_ROOT / "simulation_data" / "gnn_datasets_dag4_route_b_pilot_v1_arm_s"
FROZEN_REPORT = REPO_ROOT / "simulation_data" / "route_b_pilot_v1_arm_s_rtt.json"

needs_cache = pytest.mark.skipif(
    not CACHE_DIR.is_dir(), reason=f"cache not present at {CACHE_DIR}"
)
needs_corpus = pytest.mark.skipif(
    not (CORPUS.is_dir() and FROZEN_REPORT.is_file()),
    reason="stage-1 corpus / frozen report not present",
)


@needs_corpus
def test_self_check_reaches_1e9_acceptance(capsys, monkeypatch):
    # check_decoder looks up the report by the corpus key AS RECORDED in the report
    # JSON — which is the relative path the scorer was originally invoked with, not
    # whatever path this test passes. cwd=REPO_ROOT reproduces the documented CLI
    # invocation (scripts_cosim/eval_route_b_stage2_arm.py --self-check --corpus
    # simulation_data/...) exactly.
    monkeypatch.chdir(REPO_ROOT)

    class Args:
        corpus = "simulation_data/gnn_datasets_dag4_route_b_pilot_v1_arm_s"
        report = "simulation_data/route_b_pilot_v1_arm_s_rtt.json"
        task_types = "data/nofs-ids/task-types.json"
        alpha = None

    rc = ev.run_self_check(Args())
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK: masked_topo decoder reproduces" in out


# --- sidecar-less refusal ----------------------------------------------------------


def test_mlp_checkpoint_without_layout_is_refused(tmp_path):
    ckpt_path = tmp_path / "no_layout.pt"
    torch.save(
        {
            "model_state_dict": {},
            "input_dim": 25,
            "hidden_dim": 64,
            # inference_feature_layout deliberately omitted
        },
        ckpt_path,
    )
    with pytest.raises(ev.EvalError, match="declares no inference_feature_layout"):
        ev.load_arm(ckpt_path)


def test_mlp_checkpoint_with_bad_layout_is_refused(tmp_path):
    ckpt_path = tmp_path / "bad_layout.pt"
    torch.save(
        {
            "model_state_dict": {},
            "input_dim": 22,
            "hidden_dim": 64,
            "inference_feature_layout": "dim22",
        },
        ckpt_path,
    )
    with pytest.raises(ev.EvalError, match="unsupported MLP layout"):
        ev.load_arm(ckpt_path)


def test_gnn_checkpoint_without_sidecar_is_refused(tmp_path):
    # A bare state_dict (no "model_state_dict" key) is what save_checkpoint writes
    # for a GNN — this is what distinguishes it from an MLP checkpoint dict.
    ckpt_path = tmp_path / "no_sidecar.pt"
    torch.save({"task_encoder.net.0.weight": torch.zeros(4, 4)}, ckpt_path)
    assert not ckpt_path.with_suffix(".contract.json").is_file()
    with pytest.raises(ev.EvalError, match="no .contract.json sidecar"):
        ev.load_arm(ckpt_path)


def test_gnn_checkpoint_sidecar_without_partial_state_is_refused(tmp_path):
    ckpt_path = tmp_path / "stage1.pt"
    torch.save({"task_encoder.net.0.weight": torch.zeros(4, 4)}, ckpt_path)
    ckpt_path.with_suffix(".contract.json").write_text(
        json.dumps({"partial_state_edge_features": False})
    )
    with pytest.raises(ev.EvalError, match="not a stage-2 T2"):
        ev.load_arm(ckpt_path)


# --- --aggregate arithmetic ---------------------------------------------------------


def _fake_report(arm_type: str, regrets: dict) -> dict:
    return {
        "arm_type": arm_type,
        "per_dataset": [
            {
                "dataset_id": ds_id,
                "infeasible": False,
                "split": "train",
                "decode_regret_pct": {"registered": value},
            }
            for ds_id, value in regrets.items()
        ],
    }


def test_aggregate_mean_median_and_sigma(tmp_path):
    r1 = tmp_path / "r1.json"
    r2 = tmp_path / "r2.json"
    r3 = tmp_path / "r3.json"
    r1.write_text(json.dumps(_fake_report("mlp_dim63crk", {"a": 10.0, "b": 20.0})))
    r2.write_text(json.dumps(_fake_report("mlp_dim63crk", {"a": 10.0, "b": 30.0})))
    r3.write_text(json.dumps(_fake_report("mlp_dim63crk", {"a": 10.0, "b": 40.0})))

    out = tmp_path / "agg.json"
    result = ev.run_aggregate([r1, r2, r3], out)

    assert result["num_draws"] == 3
    # per-draw mean over the 2 datasets: (10+20)/2=15, (10+30)/2=20, (10+40)/2=25
    assert result["per_draw_train_regret_pct"] == pytest.approx([15.0, 20.0, 25.0])
    assert result["mean_train_regret_pct"] == pytest.approx(60.0 / 3)
    assert result["median_train_regret_pct"] == pytest.approx(20.0)
    # dataset "a" is constant (sigma 0) across draws; "b" is [20,30,40] (sigma=10)
    assert result["num_datasets_common_to_all_draws"] == 2
    assert result["pooled_per_dataset_paired_diff_sigma"] == pytest.approx(5.0)
    assert out.is_file()


def test_aggregate_rejects_mixed_arm_types(tmp_path):
    r1 = tmp_path / "r1.json"
    r2 = tmp_path / "r2.json"
    r1.write_text(json.dumps(_fake_report("mlp_dim63crk", {"a": 1.0})))
    r2.write_text(json.dumps(_fake_report("mlp_dim25cr", {"a": 1.0})))
    with pytest.raises(ev.EvalError, match="mix arm types"):
        ev.run_aggregate([r1, r2], tmp_path / "agg.json")


def test_aggregate_needs_at_least_one_report(tmp_path):
    with pytest.raises(ev.EvalError, match="at least one"):
        ev.run_aggregate([], tmp_path / "agg.json")


# --- end-to-end (needs a real cache + a trained checkpoint; run manually / in CI with
# the cache present, not gated on a checkpoint existing in this repo state) --------


@needs_cache
def test_gnn_arm_construction_infers_widths_from_state_dict(tmp_path):
    """The constructor-default hidden_dim=128 must not silently win over a checkpoint
    trained at a different width — this is the bug the A1 smoke caught: build_model
    must read hidden_dim/embedding_dim/num_layers off the state_dict's tensor shapes,
    not the TaskPlacementGNN defaults."""
    from src.policy.gnn.gnn_model import TaskPlacementGNN

    model = TaskPlacementGNN(
        task_feature_dim=7,
        platform_feature_dim=14,
        embedding_dim=32,
        hidden_dim=48,
        num_layers=2,
        mp_dag_edges=True,
        task_type_onehot_dim=4,
        partial_state_edge_dim=38,
    )
    ckpt_path = tmp_path / "fake_gnn.pt"
    torch.save(model.state_dict(), ckpt_path)
    ckpt_path.with_suffix(".contract.json").write_text(
        json.dumps(
            {
                "mp_dag_edges": True,
                "task_type_onehot_dim": 4,
                "dag_task_type_vocab": ["cnn", "dnn1", "dnn2", "rf"],
                "partial_state_edge_features": True,
                "partial_state_contract": "partial_state_v1",
                "partial_state_feature_dim": 38,
            }
        )
    )
    arm = ev.load_arm(ckpt_path)
    assert arm["arm_type"] == "gnn"
