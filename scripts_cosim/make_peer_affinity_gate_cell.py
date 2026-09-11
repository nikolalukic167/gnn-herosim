#!/usr/bin/env python3
"""peer_affinity_v1 stage 3: mint a live-gate topology cell in the corpus's own shape.

The prefix-conditioned checkpoints were fitted on 20-client / 6-server sparse topologies
at connection probability 0.6 (their sidecar's `corpus` block), and the partial-state
krank block is padded to KRANK_WIDTH = 6 candidate nodes, so a live gate must run on a
cell of that shape. This takes a corpus dataset's space config verbatim, changes ONLY the
topology seed (to one outside every seed the corpus used: 7001-7034 and 7100-7699), sets
the scheduler window the gate serves with, generates the co-sim infrastructure for the
cell and proves live-regeneration parity with `verify_live_infra_parity.verify_dataset`
before writing anything a run could pick up — the `make_full_corpus_siv1_gate_cells.py`
protocol.

Usage:
  PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH=. pipenv run python3 \
    scripts_cosim/make_peer_affinity_gate_cell.py --seed 7901 \
      --sweep-dir simulation_data/peer_affinity_live_gate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.verify_live_infra_parity import DEFAULT_SIM_INPUT, verify_dataset  # noqa: E402
from src.generate_infrastructure import generate_deterministic_infrastructure  # noqa: E402

BASE_CONFIG = (
    REPO_ROOT / "simulation_data" / "gnn_datasets_peer_affinity_v1_c3_x200_train" / "ds_00000"
    / "space_with_network.json"
)
CORPUS_SEEDS = set(range(7001, 7035)) | set(range(7100, 7700))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, required=True, help="topology seed; must be outside the corpus's")
    ap.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    ap.add_argument("--sweep-dir", type=Path, default=REPO_ROOT / "simulation_data" / "peer_affinity_live_gate")
    ap.add_argument("--batch-size", type=int, default=10, help="scheduler.batch_size the cell declares (= peer group)")
    ap.add_argument("--batch-timeout", type=float, default=0.02,
                    help="scheduler.batch_timeout: how long the group collector waits for late members")
    args = ap.parse_args()
    if args.seed in CORPUS_SEEDS:
        raise SystemExit(f"seed {args.seed} is a corpus seed; a gate cell must be unseen")

    base = json.loads(args.base_config.read_text())
    cell = f"cell_s{args.seed}"
    cfg_dir = args.sweep_dir / "configs"
    infra_dir = args.sweep_dir / "cell_infrastructure" / cell
    cfg_path = cfg_dir / f"{cell}.json"
    if cfg_path.exists():
        raise SystemExit(f"{cfg_path} exists — cells are minted once")

    space = json.loads(json.dumps(base))
    space["network"]["topology"]["seed"] = args.seed
    space["scheduler"] = {"batch_size": args.batch_size, "batch_timeout": args.batch_timeout}
    # Not read by the live path (no replica_plan / preinit there) but kept verbatim so the
    # cell's provenance says which corpus shape it was cut from.
    space["gate_cell"] = {
        "lineage": "peer_affinity_v1", "stage": 3, "base_config": str(args.base_config),
        "corpus_shape": {
            "client_node_count": space["nodes"]["client_nodes"]["count"],
            "server_node_count": space["nodes"]["server_nodes"]["count"],
            "topology_type": space["network"]["topology"]["type"],
            "connection_probability": space["network"]["topology"]["connection_probability"],
        },
    }
    infra_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    staged = infra_dir / "space_with_network.json"
    staged.write_text(json.dumps(space, indent=2))
    generate_deterministic_infrastructure(
        str(staged), DEFAULT_SIM_INPUT, str(infra_dir / "infrastructure.json"), args.seed
    )
    result = verify_dataset(infra_dir, DEFAULT_SIM_INPUT)
    if not result.ok:
        for finding in result.findings:
            print(f"  FAIL {finding}")
        raise SystemExit(f"{cell}: live regeneration does not match the co-sim infrastructure")
    cfg_path.write_text(json.dumps(space, indent=2))
    print(f"[gate-cell] {cell}: parity PASS; config {cfg_path}; infra {infra_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
