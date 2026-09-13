#!/usr/bin/env python3
from __future__ import annotations

"""Launch near-RTT v2 retrain on the dim-14 1060 cache.

Uses graphs_cache_gnn_datasets_4tasks_1060 rebuilt with:
  - initialized_snapshot backfill (1242 ds / 1230 valid)
  - is_warm fix (previous_task_type_name predicate)
  - platform_feature_dim=14 (shared_fate_signal dim-8 populated)

Same v2 trash-exp loss/sidecar params as near-rtt-v2-clean-1230.
"""

import os
import sys
import runpy
from pathlib import Path


def main() -> None:
    os.environ.setdefault("NEAR_RTT_LOSS_VARIANT", "near-rtt-v2-trash-exp")
    os.environ.setdefault("NEAR_RTT_SIDECAR_NAME", "valid_combos_near_rtt_v2_capped.pkl")
    os.environ.setdefault("NEAR_RTT_MARGIN_MODE", "exp")
    os.environ.setdefault("NEAR_RTT_MARGIN_CAP", "8.0")
    os.environ.setdefault("NEAR_RTT_MARGIN_EXP_SCALE", "0.75")
    os.environ.setdefault("NEAR_RTT_MARGIN_EXP_CLIP", "4.0")
    os.environ.setdefault("NEAR_RTT_TRASH_DELTA", "5.0")
    os.environ.setdefault("NEAR_RTT_TRASH_WEIGHT", "1.0")
    os.environ.setdefault("NEAR_RTT_FAR_WEIGHT", "0.75")
    os.environ.setdefault("NEAR_RTT_UNMAPPED_PENALTY", "8.0")
    os.environ.setdefault("WANDB_RUN_NAME", "near-rtt-v2-dim14-1060")
    os.environ.setdefault(
        "WANDB_TAGS",
        "near-rtt,loss-v2,trash-sidecar,dim14,initialized-snapshot,is-warm-fix,1060",
    )

    _REPO_ROOT = Path(__file__).resolve().parents[2]
    _CACHE_DIR = (
        _REPO_ROOT
        / "simulation_data"
        / "artifacts"
        / "run_queue_big"
        / "graphs_cache_gnn_datasets_4tasks_1060"
    )

    # Inject --cache-dir into argv so train_near_rtt.py argparse picks it up.
    if "--cache-dir" not in sys.argv:
        sys.argv.extend(["--cache-dir", str(_CACHE_DIR)])

    trainer_path = Path(__file__).with_name("train_near_rtt.py")
    runpy.run_path(str(trainer_path), run_name="__main__")


if __name__ == "__main__":
    main()
