#!/usr/bin/env python3
"""r1_attribution_v1 diagnostic (read-only): permutation importance per feature group on the VALIDATION split only.

For one checkpoint, decode every validation batch with the served masked_topo decode (load_prefix_conditioned_gnn + decode_prefix_conditioned, the
path the gate serves) and score the decoded plan against the dataset's own sweep: regret = rtt(plan) - optimal rtt, in seconds, exactly the
trainer's val/regret_masked_topo. Then repeat with one feature group shuffled and report delta regret. Held-out (test) topologies are never read:
the split artifact's "val" parents are the only datasets touched, and the script refuses any other.

Shuffles (seeded, K repeats):
  task_features / task_type_onehot : permute the rows across the batch's tasks            (same permutation for the group's columns)
  platform groups                  : permute the rows across the graph's platforms        (queue = col 7, temporal = 9-11, usage = 13, flags = last 2, rest)
  edge_attr                        : permute the 5 columns' rows across the graph's task->platform edges
  partial-state groups             : permute the rows across the SCORED task's candidates, at every decode step, after the block is built
                                     (occupancy 0-3, capacity 4-6, exchange 7-9, rank 10-17, linkrank 18-21, load 22-24, push 25-26)
  ALL                              : every group shuffled at once -- the chance floor of this procedure
The unit of "column" is the group above (a column-by-column table would be 60+ rows); a within-group joint permutation keeps the group's internal
structure and removes only its link to the candidate.

  r1a_permutation_importance.py --ckpt models/<run>.pt --cache-dir <v5 cache> --split <split.json> --out report.json [--repeats 3] [--limit N]
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src" / "notebooks"))

PS_GROUPS = {"ps_occupancy": [0, 1, 2, 3], "ps_capacity": [4, 5, 6], "ps_exchange_pull": [7, 8, 9], "ps_rank": list(range(10, 18)),
             "ps_linkrank": [18, 19, 20, 21], "ps_load": [22, 23, 24], "ps_exchange_push": [25, 26]}
PLATFORM_GROUPS = {"platform_queue": [7], "platform_temporal": [9, 10, 11], "platform_usage": [13], "platform_flags": [14, 15]}


def platform_groups(width: int) -> Dict[str, List[int]]:
    named = {k: [c for c in v if c < width] for k, v in PLATFORM_GROUPS.items()}
    used = {c for v in named.values() for c in v}
    named["platform_other"] = [c for c in range(width) if c not in used]
    return named


def rtt_table(ds_dir: Path) -> Dict[tuple, float]:
    out = {}
    with open(ds_dir / "placements" / "placements.jsonl") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                plan = r["placement_plan"]
                out[tuple(tuple(plan[str(i)]) for i in range(len(plan)))] = float(r["rtt"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="first N validation graphs only (smoke)")
    a = ap.parse_args()

    from src.policy.gnn import partial_state_edges as pse
    from src.policy.gnn.prefix_serving import decode_prefix_conditioned, load_prefix_conditioned_gnn

    split = json.loads(a.split.read_text())
    val_parents = set(split["val"])
    if val_parents & set(split["test"]) or val_parents & set(split["train"]):
        raise SystemExit("FAIL LOUD: split artifact has overlapping val/test/train")
    meta = json.loads((a.cache_dir / "metadata.json").read_text())
    ids = pickle.load(open(a.cache_dir / "dataset_ids.pkl", "rb"))
    graphs = pickle.load(open(a.cache_dir / "graphs.pkl", "rb"))
    base = {Path(d).name: Path(d) for d in meta["base_dirs"]}
    items = [(i, g) for i, (i, g) in enumerate(zip(ids, graphs)) if str(i).split("@")[0] in val_parents]
    if a.limit:
        items = items[:a.limit]
    assert items and all("heldout" not in str(i) for i, _ in items), "FAIL LOUD: a held-out dataset is in the validation selection"
    model, options, side = load_prefix_conditioned_gnn(a.ckpt, device=torch.device("cpu"))
    model.eval()
    print(f"{a.ckpt.name}: arm {side.get('arm_kind', 'gnn')}, {len(items)} validation graphs", flush=True)

    tables = {}
    for did, g in items:
        corpus, ds = str(did).split("@")[0].split("/", 1)
        tables[did] = rtt_table(base[corpus] / ds)

    pw = platform_groups(int(graphs[0].platform_features.shape[1]))
    groups: Dict[str, Optional[Dict]] = {
        "task_features": {"attr": "task_features"}, "task_type_onehot": {"attr": "task_type_onehot4"},
        **{k: {"attr": "platform_features", "cols": v} for k, v in pw.items() if v},
        "edge_attr": {"attr": "edge_attr"}, **{k: {"ps": v} for k, v in PS_GROUPS.items()},
    }

    state = {"rng": None, "ps_cols": None}
    orig_refresh = pse.refresh_partial_state_edge_attr

    def shuffled_refresh(data, ctx, task_idx, committed):
        attr = orig_refresh(data, ctx, task_idx, committed)
        cols = state["ps_cols"]
        if cols:
            rows = pse.candidate_edge_rows(data)[int(task_idx)]
            if len(rows) > 1:
                perm = state["rng"].permutation(len(rows))
                r = torch.as_tensor(rows, dtype=torch.long)
                attr[r[:, None], torch.as_tensor(cols)[None, :]] = attr[r[perm][:, None], torch.as_tensor(cols)[None, :]].clone()
        return attr

    pse.refresh_partial_state_edge_attr = shuffled_refresh

    def perturbed(g, names: Sequence[str], rng) -> object:
        h = copy.copy(g)
        state["ps_cols"] = None
        ps_cols: List[int] = []
        for n in names:
            spec = groups[n]
            if "ps" in spec:
                ps_cols += spec["ps"]
                continue
            t = getattr(h, spec["attr"]).clone()
            cols = spec.get("cols")
            n_rows = t.shape[0]
            perm = torch.as_tensor(rng.permutation(n_rows))
            if cols is None:
                t = t[perm]
            else:
                t[:, cols] = t[perm][:, cols]
            setattr(h, spec["attr"], t)
        state["ps_cols"] = sorted(set(ps_cols)) or None
        h.partial_state_edge_attr = None if not hasattr(g, "partial_state_edge_attr") else g.partial_state_edge_attr.clone()
        return h

    def mean_regret(names: Sequence[str], seed: int):
        rng = np.random.default_rng(seed)
        state["rng"] = rng
        reg, unmapped = [], 0
        for did, g in items:
            h = perturbed(g, names, rng)
            plan = decode_prefix_conditioned(model, h, options)
            tab = tables[did]
            key = tuple(tuple(int(x) for x in p) for p in plan)
            if key in tab:
                reg.append(tab[key] - float(g.opt_rtt))
            else:
                unmapped += 1
        return float(np.mean(reg)), unmapped, len(reg)

    t0 = time.time()
    base_reg, base_unm, n = mean_regret([], 0)
    print(f"baseline regret {base_reg:.4f}s over {n} graphs ({base_unm} unmapped) [{time.time() - t0:.0f}s]", flush=True)
    rows: Dict[str, Dict] = {}
    for name in list(groups) + ["ALL"]:
        names = list(groups) if name == "ALL" else [name]
        vals = [mean_regret(names, 1000 + r) for r in range(a.repeats)]
        d = [v[0] - base_reg for v in vals]
        rows[name] = {"regret": [v[0] for v in vals], "delta": float(np.mean(d)), "delta_sd": float(np.std(d)), "unmapped": [v[1] for v in vals]}
        print(f"  {name:20s} delta {np.mean(d):+8.4f}s (sd {np.std(d):.4f}) regret {np.mean([v[0] for v in vals]):.4f}s [{time.time() - t0:.0f}s]", flush=True)
    pse.refresh_partial_state_edge_attr = orig_refresh
    a.out.write_text(json.dumps({"ckpt": a.ckpt.name, "arm": side.get("arm_kind", "gnn"), "n_val_graphs": n, "baseline_regret": base_reg,
                                 "baseline_unmapped": base_unm, "repeats": a.repeats, "groups": rows, "ps_groups": PS_GROUPS,
                                 "platform_groups": pw}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
