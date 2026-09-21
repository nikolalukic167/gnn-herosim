#!/usr/bin/env python3
"""rollout_imitation_v1: mint additional training-topology cells for a fatter rollout corpus.

The joint_burst training cells (cc40s9201..9224) differ from one another in exactly one field --
network.topology.seed -- everything else (40 clients / 6 servers, replicas, prewarm, scheduler) is
shared, and the network is generated deterministically from the seed at run time. So a new training
topology is a clone of the base cell with a new seed. Exploitable (multi-candidate) decisions are
sparse per topology, so more topologies is the lever for a trainable corpus.

Usage (datalab): python3 scripts_cosim/rollout_imitation_v1_mint_cells.py --lo 9225 --hi 9320 \
    --configs-dir simulation_data/peer_affinity_live_gate/configs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE_SEED = 9201  # cc40s9201.json is the template


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs-dir", type=Path, required=True)
    ap.add_argument("--lo", type=int, required=True, help="first new seed (inclusive)")
    ap.add_argument("--hi", type=int, required=True, help="last new seed (inclusive)")
    ap.add_argument("--base-seed", type=int, default=BASE_SEED)
    a = ap.parse_args()

    base = json.load(open(a.configs_dir / f"cc40s{a.base_seed}.json"))
    made, skipped = 0, 0
    for seed in range(a.lo, a.hi + 1):
        out = a.configs_dir / f"cc40s{seed}.json"
        if out.exists():
            skipped += 1
            continue
        cfg = json.loads(json.dumps(base))
        cfg["network"]["topology"]["seed"] = seed
        json.dump(cfg, open(out, "w"))
        made += 1
    print(f"minted {made} cells (seeds {a.lo}-{a.hi}), skipped {skipped} existing; base cc40s{a.base_seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
