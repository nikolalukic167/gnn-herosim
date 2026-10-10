#!/usr/bin/env python3
"""Write the r1_attribution_v1 arm configs under experiments/: 7 learned arms x a 6-configuration grid, plus a smoke variant each.

Fair-comparison budget (docs/lineages/r1_attribution_v1.md, after Errica et al.): every arm gets the same grid, the same epoch
cap, the same early stopping and three training seeds (run_experiment.py --seed 1 2 3). The grid is learning rate {5e-4, 1e-3, 2e-3}
x width {64, 128} (embedding = hidden); depth is each arm's default (3 message-passing layers / 3 set blocks / 2 MLP hidden layers).
Everything else is the so1load recipe on partial_state_v5.

    python3 scripts_cosim/make_r1a_arm_configs.py            # (re)write experiments/r1_attribution_v1_*.yaml
    python3 scripts_cosim/make_r1a_arm_configs.py --check    # exit 1 when a committed file differs from what this would write

The production cache and split do not exist yet (they come from the corpus build); the configs name the paths the build is to produce.
The smoke variants point at the cache scripts_cosim/datalab/r1a_arms_smoke.sbatch builds from S6's smoke corpus.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "experiments"

CACHE = "simulation_data/graphs_cache_r1_attribution_v1_psv5_inf"
SPLIT = "experiments/r1_attribution_v1_split.json"
SMOKE_CACHE = "simulation_data/graphs_cache_r1_attribution_v1_smoke_psv5"
SMOKE_SPLIT = "experiments/r1_attribution_v1_smoke_split.json"

GRID = [(lr, width) for width in (64, 128) for lr in (5e-4, 1e-3, 2e-3)]

# so1load, with the v5 contract. NEAR_RTT_TRAIN_OBJECTIVE ce_only + full-context CE are the recipe's loss.
BASE_ENV = {
    "NEAR_RTT_SAVE_FINAL": "1",
    "NEAR_RTT_TASK_TYPE_ONEHOT": "1",
    "NEAR_RTT_PARTIAL_STATE_EDGES": "1",
    "NEAR_RTT_DAG_ALPHA_KEY": "inf",
    "NEAR_RTT_TIED_MAX_PLANS": "0",
    "NEAR_RTT_TRAIN_OBJECTIVE": "ce_only",
    "NEAR_RTT_VAL_EXACT_REGRET": "1",
    "NEAR_RTT_DECODE_REPLICA_REUSE": "1",
    "NEAR_RTT_DECODE_RELAX": "1",
    "PARTIAL_STATE_CONTRACT": "partial_state_v5",
    "PARTIAL_STATE_LOAD_SECONDS": "1",
    "PARTIAL_STATE_PEER_MASS": "1",
    "NEAR_RTT_FULL_CONTEXT_CE_WEIGHT": "1.0",
    "PARTIAL_STATE_EXCHANGE_SECONDS": "1",
    # arm and configuration are chosen on validation topologies only: the held-out topologies are not scored by training
    "NEAR_RTT_SKIP_FINAL_TEST": "1",
}
ENG_MP = {"NEAR_RTT_MP_PEER_EDGES": "1", "NEAR_RTT_MP_BIPARTITE_EDGE_CONV": "1", "NEAR_RTT_MP_BIPARTITE_EDGE_ATTR_ZERO": "1"}
TWIN_MP = {"NEAR_RTT_MP_PEER_EDGES": "1", "GNN_DISABLE_MESSAGE_PASSING": "1"}

# arm -> (env additions, one-line description)
ARMS = {
    "gnn_eng": (ENG_MP, "GNN-eng: engineered relational features (partial_state_v5) + message passing; the retrained so1load recipe."),
    "twin_eng": (TWIN_MP, "Twin-eng: GNN-eng with every message-passing layer skipped (the MP-OFF twin, trained separately)."),
    "mlp_same": ({"NEAR_RTT_ARM": "mlp_same"}, "MLP-same: GNN-eng's per-edge inputs scored by one MLP, no node encoders, no graph."),
    "gnn_raw": ({**ENG_MP, "NEAR_RTT_PLAN_RAW": "1"}, "GNN-raw: the raw committed plan instead of hand-built relational columns, message passing on."),
    "twin_raw": ({**TWIN_MP, "NEAR_RTT_PLAN_RAW": "1"}, "Twin-raw: GNN-raw with message passing off."),
    "gnn_eng_physmp": ({**ENG_MP, "NEAR_RTT_MP_BIPARTITE_EDGE_ATTR_ZERO": "0"},
                       "GNN-eng-physMP: GNN-eng with the physics edge columns live inside message passing (edge_attr not zeroed)."),
    "set_transformer": ({"NEAR_RTT_ARM": "set_transformer"},
                        "Set-transformer: GNN-eng's inputs and scorer, all-pairs set attention (SAB over tasks, ISAB over candidates) in place of the graph."),
}


def build(arm: str, k: int | None, lineage: str = "r1_attribution_v1", project: str = "gnn-r1-attribution-v1") -> tuple[str, str]:
    extra, what = ARMS[arm]
    smoke = k is None
    if smoke:
        lr, width = 2e-3, 64
        epochs, patience, min_epochs = 2, 2, 1
        cache, split = SMOKE_CACHE, SMOKE_SPLIT
        name = f"r1_attribution_v1_{arm}_smoke"
    else:
        lr, width = GRID[k]
        epochs, patience, min_epochs = 100, 60, 100
        cache, split = CACHE, SPLIT
        name = f"r1_attribution_v1_{arm}_g{k}"
        if lineage != "r1_attribution_v1":
            cache, split, name = f"simulation_data/graphs_cache_{lineage}_psv5_inf", f"experiments/{lineage}_split.json", f"{lineage}_{arm}_g{k}"
    cfg = {
        "trainer": "gnn",
        "lineage": lineage,
        "cache_dir": cache,
        "unset_env": ["TRAIN_INIT_CHECKPOINT", "NEAR_RTT_LABEL_OVERRIDE_JSON"],
        "env": {**BASE_ENV, "NEAR_RTT_SPLIT_ARTIFACT": split, **extra},
        "args": {
            "regret-loss-weight": 0, "ce-loss-weight": 1, "epochs": epochs, "patience": patience,
            "min-epochs": min_epochs, "learning-rate": lr, "embedding-dim": width, "hidden-dim": width,
        },
        "wandb": {
            "project": project,
            "run_name": name.replace("_", "-"),
            "tags": [lineage.replace("_", "-"), "partial-state-v5", arm.replace("_", "-")] + (["smoke"] if smoke else [f"grid{k}"]),
        },
    }
    head = (f"# {lineage} -- {what}\n"
            + ("# SMOKE (2 epochs, S6's smoke corpus): pipeline and determinism check; its checkpoints are never served as evidence.\n" if smoke else
               f"# Grid point {k} of 6: lr {lr:g}, embedding = hidden = {width}. 100-epoch cap, patience 60; seeds via run_experiment.py --seed 1|2|3.\n")
            + "# Generated by scripts_cosim/make_r1a_arm_configs.py -- edit the generator, not this file.\n")
    return name, head + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)


def all_configs() -> dict[str, str]:
    out = {}
    for arm in ARMS:
        for k in list(range(len(GRID))) + [None]:
            name, text = build(arm, k)
            out[name + ".yaml"] = text
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--lineage", help="write this lineage's configs instead (experiments/<lineage>_<arm>_g<k>.yaml, no smoke variants), e.g. scale_160_v1")
    ap.add_argument("--arms", help="comma-separated arms of --lineage (default: all)")
    ap.add_argument("--project", help="W&B project of --lineage (default: gnn-<lineage with dashes>)")
    a = ap.parse_args()
    bad = []
    configs = all_configs()
    if a.lineage:
        arms = a.arms.split(",") if a.arms else list(ARMS)
        unknown = sorted(set(arms) - set(ARMS))
        if unknown:
            raise SystemExit(f"FAIL LOUD: unknown arms {unknown}")
        project = a.project or "gnn-" + a.lineage.replace("_", "-")
        configs = {}
        for arm in arms:
            for k in range(len(GRID)):
                name, text = build(arm, k, a.lineage, project)
                configs[name + ".yaml"] = text
    for fname, text in configs.items():
        path = OUT / fname
        if a.check:
            if not path.is_file() or path.read_text() != text:
                bad.append(fname)
        else:
            path.write_text(text)
    if a.check and bad:
        print(f"stale or missing: {bad}", file=sys.stderr)
        return 1
    if not a.check:
        print(f"wrote {len(configs)} configs to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
