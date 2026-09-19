#!/usr/bin/env python3
"""Mint sealed live-gate cells for the `siv1_full_corpus` checkpoint, and prove their parity.

Why fresh cells rather than held-out corpus datasets: every one of the 52 datasets on disk
but absent from `graphs_cache_full_corpus_siv1_dim14` shares its exact topology cell
(clients, servers, type, connection_probability, seed) with a dataset the model *did*
train on — checked, 0/52 unseen. They were excluded for data quality (the
`exclude_bad31` oversample manifest), not held out as a topology holdout. So there is no
unseen cell to select, and one has to be minted.

Each minted cell keeps every axis the checkpoint's sidecar declares (20 clients / 20
servers, `sparse`, a connection probability drawn from the corpus's own six values) and
changes only the topology seed, to a value outside the 142 the corpus used. That makes the
cell in-distribution but not memorized — the honest first live gate.

For each cell this script:
  1. writes the space config,
  2. generates the co-sim `infrastructure.json` for it (cheap: topology + replica
     placement only, no brute-force sweep), so there is a corpus-side artifact to compare
     against, and
  3. runs `verify_live_infra_parity.py` over the pair, refusing to emit a cell whose live
     regeneration would not match.

Usage:
    PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH=$(pwd) \
      pipenv run python3 scripts_cosim/important/make_full_corpus_siv1_gate_cells.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.verify_live_infra_parity import DEFAULT_SIM_INPUT, verify_dataset  # noqa: E402
from src.generate_infrastructure import generate_deterministic_infrastructure  # noqa: E402

BASE_CONFIG = (
    REPO_ROOT
    / "simulation_data"
    / "gnn_datasets_4tasks_contention_v2"
    / "ds_00000"
    / "space_with_network.json"
)

# (cell name, connection_probability, topology seed). Every probability is one the corpus
# spans; every seed is outside the 142 it used (max was 609).
CELLS = [
    ("cell01_p25_s9001", 0.25, 9001),
    ("cell02_p35_s9002", 0.35, 9002),
    ("cell03_p15_s9003", 0.15, 9003),
    ("cell04_p50_s9004", 0.50, 9004),
    ("cell05_p20_s9005", 0.20, 9005),
]

DEFAULT_SWEEP_DIR = (
    REPO_ROOT / "simulation_data" / "normal_sim_sweeps" / "full_corpus_siv1_live_gate_20260820"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--sim-input", type=Path, default=DEFAULT_SIM_INPUT)
    parser.add_argument(
        "--backbone-bandwidth-mbps",
        type=float,
        default=None,
        help=(
            "Mint the cells with a `network.backbone` block at this per-link capacity "
            "(link_contention_v1 physics). Omitted, no backbone block is written and the "
            "cells are bit-identical to the stock ones. NOTE: the backbone must be present "
            "when the cell's corpus-side infrastructure.json is generated, not bolted on "
            "afterwards -- verify_live_infra_parity compares link_topology *presence*, so "
            "a live-only backbone fails parity by construction."
        ),
    )
    parser.add_argument("--backbone-n-core", type=int, default=4)
    parser.add_argument("--backbone-attach-degree", type=int, default=1)
    parser.add_argument("--backbone-chord-count", type=int, default=0)
    args = parser.parse_args()

    config_dir = args.sweep_dir / "configs"
    infra_dir = args.sweep_dir / "cell_infrastructure"
    config_dir.mkdir(parents=True, exist_ok=True)
    infra_dir.mkdir(parents=True, exist_ok=True)

    with open(BASE_CONFIG, "r") as handle:
        base = json.load(handle)

    failures = []
    for name, conn_prob, seed in CELLS:
        space_config = json.loads(json.dumps(base))  # deep copy
        space_config["network"]["topology"]["connection_probability"] = conn_prob
        space_config["network"]["topology"]["seed"] = seed

        if args.backbone_bandwidth_mbps is not None:
            # n_core=4 is the measured interior peak of the hub<->mesh sweep -- the only
            # configuration whose max additive-argmin regret cleared the 5% gate
            # (LINEAGES.md, link_contention_v1 closing sweep). attach_degree=1 + no chords
            # because chords and a second attachment both collapse route overlap (P0
            # pre-check).
            space_config.setdefault("network", {})["backbone"] = {
                "n_core": args.backbone_n_core,
                "attach_degree": args.backbone_attach_degree,
                "chord_count": args.backbone_chord_count,
                "bandwidth_mbps": args.backbone_bandwidth_mbps,
            }

        config_path = config_dir / f"{name}.json"
        with open(config_path, "w") as handle:
            json.dump(space_config, handle, indent=2)

        # The parity tool expects a dataset-shaped directory: space config + infrastructure.
        cell_dir = infra_dir / name
        cell_dir.mkdir(parents=True, exist_ok=True)
        with open(cell_dir / "space_with_network.json", "w") as handle:
            json.dump(space_config, handle, indent=2)

        generate_deterministic_infrastructure(
            str(config_path),
            args.sim_input,
            str(cell_dir / "infrastructure.json"),
            seed,
        )

        result = verify_dataset(cell_dir, args.sim_input)
        status = "PASS" if result.ok else "FAIL"
        print(f"\n[{status}] {name}  p={conn_prob} seed={seed}")
        for note in result.notes:
            print(f"    note: {note}")
        for finding in result.findings:
            print(f"    !! {finding}")
        if not result.ok:
            failures.append(name)

    print(f"\n{len(CELLS) - len(failures)}/{len(CELLS)} cells verified")
    print(f"configs:   {config_dir}")
    print(f"cell infra: {infra_dir}")
    if failures:
        print(f"FAILED cells: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
