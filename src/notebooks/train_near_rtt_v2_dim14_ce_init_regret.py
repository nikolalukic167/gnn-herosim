#!/usr/bin/env python3
from __future__ import annotations

"""Near-RTT v2 dim14 Phase B: ranking fine-tune from CE-only anchor.

Initializes from models/near-rtt-v2-dim14-ce-only.pt (Phase A) on the dim-14
1060 cache. CE stays at 1.0; regret is a light constant nudge (default 0.02).
Override regret via NEAR_RTT_REGRET_WEIGHT (e.g. 0.01, 0.02 for ablation).
"""

import os
import sys
import runpy
from pathlib import Path


def _regret_tag(weight: float) -> str:
    return f"r{int(round(weight * 100)):03d}"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    ce_only_ckpt = repo_root / "models" / "near-rtt-v2-dim14-ce-only.pt"
    if not ce_only_ckpt.exists():
        raise FileNotFoundError(
            f"Phase A checkpoint missing: {ce_only_ckpt}. "
            "Run train_near_rtt_v2_dim14_ce_only.py first."
        )

    regret_weight = float(os.environ.get("NEAR_RTT_REGRET_WEIGHT", "0.02"))
    regret_tag = _regret_tag(regret_weight)
    wandb_name = os.environ.get(
        "WANDB_RUN_NAME", f"near-rtt-v2-dim14-ce-init-{regret_tag}"
    )

    os.environ.setdefault("NEAR_RTT_LOSS_VARIANT", "near-rtt-v2-trash-exp")
    os.environ.setdefault("NEAR_RTT_SIDECAR_NAME", "valid_combos_near_rtt_v2_capped.pkl")
    os.environ.setdefault("NEAR_RTT_MARGIN_MODE", "exp")
    os.environ.setdefault("NEAR_RTT_MARGIN_CAP", "4.0")
    os.environ.setdefault("NEAR_RTT_MARGIN_EXP_SCALE", "0.75")
    os.environ.setdefault("NEAR_RTT_MARGIN_EXP_CLIP", "4.0")
    os.environ.setdefault("NEAR_RTT_TRASH_DELTA", "5.0")
    os.environ.setdefault("NEAR_RTT_TRASH_WEIGHT", "1.0")
    os.environ.setdefault("NEAR_RTT_FAR_WEIGHT", "0.75")
    os.environ.setdefault("NEAR_RTT_UNMAPPED_PENALTY", "8.0")
    os.environ.setdefault("TRAIN_INIT_CHECKPOINT", str(ce_only_ckpt))
    os.environ.setdefault("NEAR_RTT_CE_BASELINE_ACC", "0.244")
    os.environ.setdefault("NEAR_RTT_GREEDY_COLLAPSE_REL", "0.10")
    os.environ.setdefault("NEAR_RTT_PHASE_B_CHECKPOINT_METRIC", "seq_reforward_regret")
    os.environ.setdefault("NEAR_RTT_REGRET_RAMP", "0")
    os.environ.setdefault("WANDB_RUN_NAME", wandb_name)
    os.environ.setdefault(
        "WANDB_TAGS",
        f"near-rtt,loss-v2,trash-sidecar,dim14,ce-init,regret-{regret_tag},1060",
    )

    cache_dir = (
        repo_root
        / "simulation_data"
        / "artifacts"
        / "run_queue_big"
        / "graphs_cache_gnn_datasets_4tasks_1060"
    )

    trainer_path = Path(__file__).with_name("train_near_rtt.py")
    sys.argv = [
        str(trainer_path),
        "--cache-dir",
        str(cache_dir),
        "--regret-loss-weight",
        str(regret_weight),
        "--ce-loss-weight",
        "1",
        "--learning-rate",
        os.environ.get("NEAR_RTT_LEARNING_RATE", "2e-4"),
        "--epochs",
        os.environ.get("NEAR_RTT_EPOCHS", "40"),
        "--wandb-project",
        "gnn-near-rtt-jun2026",
    ]
    runpy.run_path(str(trainer_path), run_name="__main__")


if __name__ == "__main__":
    main()
