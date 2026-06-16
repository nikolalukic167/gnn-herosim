#!/usr/bin/env python3
"""GNN-necessity ablation: does message passing (and same-node aggregation) beat a
pointwise edge scorer on co-sim placement labels?

Three models share an IDENTICAL encoder + edge scorer; the ONLY difference is the
relational structure used between encoding and scoring:

  pointwise   : no message passing            (feature-parity MLP baseline)
  gnn_base    : GIN over task<->platform edges (bipartite message passing)
  gnn_node    : GIN over task<->platform + same-node platform<->platform edges
                (the node-aggregation prototype: signal a pointwise model cannot see)

Trained with identical CE loss / optimizer / splits / seed. We report, on a held-out
test split:
  - top-1 per-task placement accuracy (argmax logit == oracle-optimal logit index)
  - JOINT-PLAN RTT REGRET vs the brute-force optimum (the money metric): build the plan
    from per-task argmax, look up its RTT in that dataset's placements.jsonl, and compute
    (plan_rtt - opt_rtt)/opt_rtt.
  - greedy/pointwise-oracle baseline regret for reference.

node_edge_index is synthesized on the fly from queue_key_to_platform_meta so this runs on
any existing cache (old caches lack the attribute).

Run:
  pipenv run python3 scripts_cosim/gnn_necessity_ablation.py \
      --cache simulation_data/graphs_cache_XXX --corpus-root simulation_data --epochs 120
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.models import GIN


# --------------------------------------------------------------------------------------
# Model: shared encoder + scorer; toggle message passing.
# --------------------------------------------------------------------------------------
class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class EdgeScorer(nn.Module):
    def __init__(self, emb: int, hidden: int, edge_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(2 * emb + edge_dim, hidden)
        self.drop = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, et: Tensor, ep: Tensor, ea: Optional[Tensor]) -> Tensor:
        x = torch.cat([et, ep] + ([ea] if ea is not None else []), dim=-1)
        return self.fc2(self.drop(F.relu(self.fc1(x)))).squeeze(-1)


class AblationModel(nn.Module):
    def __init__(self, task_dim: int, plat_dim: int, edge_dim: int,
                 use_gin: bool, use_node_edges: bool,
                 emb: int = 64, hidden: int = 128, layers: int = 3):
        super().__init__()
        self.use_gin = use_gin
        self.use_node_edges = use_node_edges
        self.task_enc = MLPEncoder(task_dim, hidden, emb)
        self.plat_enc = MLPEncoder(plat_dim, hidden, emb)
        self.gin = GIN(in_channels=emb, hidden_channels=hidden, num_layers=layers, out_channels=emb) if use_gin else None
        self.post_drop = nn.Dropout(0.2)
        self.scorer = EdgeScorer(emb, hidden, edge_dim)

    def forward(self, g) -> List[Tensor]:
        nt, npl = int(g.n_tasks), int(g.n_platforms)
        te = self.task_enc(g.task_features)
        pe = self.plat_enc(g.platform_features)
        if self.use_gin:
            x0 = torch.cat([te, pe], dim=0)
            ei = g.edge_index
            if self.use_node_edges and getattr(g, "node_edge_index", None) is not None and g.node_edge_index.numel() > 0:
                ei = torch.cat([ei, g.node_edge_index.to(ei.device)], dim=1)
            # Residual around GIN: message passing AUGMENTS the per-node encoding instead of
            # replacing it. Prevents oversmoothing/training-collapse when many (e.g. same-node)
            # edges flood aggregation, and guarantees GNN capacity >= pointwise.
            x = x0 + self.post_drop(self.gin(x0, ei))
            te, pe = x[:nt], x[nt:]
        ei = g.edge_index
        ti, pj = ei[0], ei[1] - nt
        valid = (pj >= 0) & (pj < npl) & (ti < nt)
        ti, pj = ti[valid], pj[valid]
        ea = None
        if getattr(g, "edge_attr", None) is not None and g.edge_attr.numel() > 0:
            ea = g.edge_attr[valid]
        scores = self.scorer(te[ti], pe[pj], ea)
        return [scores[ti == t] for t in range(nt)]


# --------------------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------------------
def synth_node_edges(g, candidates_only: bool = True) -> Tensor:
    """Build same-node platform<->platform undirected edges from platform metadata.

    candidates_only: restrict to platforms reachable by >=1 task in THIS batch (they
    appear as a destination in edge_index). This is the principled scope (contention only
    matters among platforms tasks can actually use) AND avoids oversmoothing: connecting
    all 208 topology platforms floods the GIN (1428 edges vs 36 bipartite); restricting to
    candidates yields a handful of edges focused on real co-location competition.
    """
    meta = getattr(g, "queue_key_to_platform_meta", None) or {}
    nt = int(g.n_tasks)
    npl = int(g.n_platforms)
    cand: Optional[set] = None
    if candidates_only:
        ei = g.edge_index
        pj = (ei[1] - nt)
        valid = (pj >= 0) & (pj < npl) & (ei[0] < nt)
        cand = set(int(x) for x in pj[valid].tolist())
    node2pos: Dict[object, List[int]] = defaultdict(list)
    for v in meta.values():
        if "node_id" in v and "platform_pos" in v:
            pos = int(v["platform_pos"])
            if cand is None or pos in cand:
                node2pos[v["node_id"]].append(pos)
    s, d = [], []
    for pos in node2pos.values():
        pos = sorted(set(pos))
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                a, b = nt + pos[i], nt + pos[j]
                s += [a, b]; d += [b, a]
    if not s:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([s, d], dtype=torch.long)


def load_combos(ds_dir: Path) -> Optional[Dict[Tuple, float]]:
    """placements.jsonl -> {tuple(sorted task->(node,plat)) : min rtt}."""
    jp = ds_dir / "placements" / "placements.jsonl"
    if not jp.exists():
        return None
    lut: Dict[Tuple, float] = {}
    with jp.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            plan, rtt = rec.get("placement_plan"), rec.get("rtt")
            if plan is None or rtt is None:
                continue
            key = tuple(sorted((int(k), (int(v[0]), int(v[1]))) for k, v in plan.items()))
            r = float(rtt)
            if key not in lut or r < lut[key]:
                lut[key] = r
    return lut or None


def plan_key(plan: Dict[int, Tuple[int, int]]) -> Tuple:
    return tuple(sorted((t, (n, p)) for t, (n, p) in plan.items()))


# --------------------------------------------------------------------------------------
# Eval: RTT regret of a model's argmax joint plan vs the oracle optimum.
# --------------------------------------------------------------------------------------
def eval_regret(model, graphs, corpus_root: Path, device, lut_cache: dict) -> dict:
    model.eval()
    regrets, top1_hits, top1_n = [], 0, 0
    found, collided, missing = 0, 0, 0
    opt_recovered = 0
    per_ds: Dict[str, float] = {}
    with torch.no_grad():
        for g in graphs:
            dsid = str(g.dataset_id)
            lut = lut_cache.get(dsid)
            if not lut:
                continue
            opt_rtt = min(lut.values())
            logits = model(g.to(device))
            nt = int(g.n_tasks)
            l2p = g.task_logit_to_placement
            plan = {}
            ok = True
            for t in range(nt):
                if logits[t].numel() == 0 or t >= len(l2p) or len(l2p[t]) == 0:
                    ok = False
                    break
                idx = int(torch.argmax(logits[t]).item())
                if idx >= len(l2p[t]):
                    ok = False
                    break
                plan[t] = tuple(l2p[t][idx])
                # top-1 accuracy vs oracle label y
                if hasattr(g, "y") and t < g.y.numel() and int(g.y[t].item()) >= 0:
                    top1_n += 1
                    if idx == int(g.y[t].item()):
                        top1_hits += 1
            if not ok:
                continue
            if len(set(plan.values())) < len(plan):
                collided += 1
            key = plan_key(plan)
            if key in lut:
                found += 1
                rtt = lut[key]
                reg = (rtt - opt_rtt) / opt_rtt if opt_rtt > 0 else 0.0
                regrets.append(reg)
                per_ds[dsid] = reg
                if reg <= 1e-9:
                    opt_recovered += 1
            else:
                missing += 1
    regrets_arr = np.array(regrets) if regrets else np.array([0.0])
    return {
        "n_eval": len(graphs),
        "n_found": found,
        "n_missing_plan": missing,
        "n_collided_plan": collided,
        "top1_acc": (top1_hits / top1_n) if top1_n else float("nan"),
        "regret_mean": float(regrets_arr.mean()),
        "regret_median": float(np.median(regrets_arr)),
        "regret_p90": float(np.percentile(regrets_arr, 90)),
        "regret_max": float(regrets_arr.max()),
        "opt_recovered_frac": (opt_recovered / found) if found else float("nan"),
        "per_ds": per_ds,
    }


def greedy_baseline_regret(graphs, lut_cache: dict) -> dict:
    """Per-task marginal-argmin (pointwise oracle / Knative-like lower bound) regret."""
    regrets = []
    per_ds: Dict[str, float] = {}
    for g in graphs:
        dsid = str(g.dataset_id)
        lut = lut_cache.get(dsid)
        if not lut:
            continue
        opt_rtt = min(lut.values())
        # marginal_min[t][placement] = min rtt over combos with task t -> placement
        marg: Dict[int, Dict[Tuple[int, int], float]] = defaultdict(lambda: defaultdict(lambda: float("inf")))
        for key, rtt in lut.items():
            for t, pl in key:
                if rtt < marg[t][pl]:
                    marg[t][pl] = rtt
        plan = {t: min(d.items(), key=lambda kv: kv[1])[0] for t, d in marg.items()}
        k = plan_key(plan)
        if k in lut and opt_rtt > 0:
            reg = (lut[k] - opt_rtt) / opt_rtt
            regrets.append(reg)
            per_ds[dsid] = reg
    arr = np.array(regrets) if regrets else np.array([0.0])
    return {"regret_mean": float(arr.mean()), "regret_p90": float(np.percentile(arr, 90)),
            "regret_max": float(arr.max()), "n": len(regrets), "per_ds": per_ds}


def train_model(model, train_graphs, device, epochs: int, lr: float = 1e-3) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for ep in range(epochs):
        model.train()
        random.shuffle(train_graphs)
        tot = 0.0
        for g in train_graphs:
            gg = g.to(device)
            logits = model(gg)
            loss = 0.0
            n = 0
            for t in range(int(gg.n_tasks)):
                if t >= gg.y.numel():
                    continue
                yt = int(gg.y[t].item())
                if yt < 0 or logits[t].numel() == 0 or yt >= logits[t].numel():
                    continue
                loss = loss + F.cross_entropy(logits[t].unsqueeze(0), torch.tensor([yt], device=device))
                n += 1
            if n == 0:
                continue
            loss = loss / n
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"    epoch {ep:3d}  mean_ce={tot/max(len(train_graphs),1):.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--corpus-root", default="simulation_data")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  cache={args.cache}")

    graphs = pickle.load(open(os.path.join(args.cache, "graphs.pkl"), "rb"))
    if args.limit:
        graphs = graphs[: args.limit]
    # Always (re)build candidate-restricted same-node edges (override any cached full
    # version, which floods the GIN and oversmooths).
    for g in graphs:
        g.node_edge_index = synth_node_edges(g, candidates_only=True)
    task_dim = graphs[0].task_features.shape[1]
    plat_dim = graphs[0].platform_features.shape[1]
    edge_dim = graphs[0].edge_attr.shape[1] if graphs[0].edge_attr.numel() > 0 else 0
    print(f"n_graphs={len(graphs)}  task_dim={task_dim} plat_dim={plat_dim} edge_dim={edge_dim}")
    avg_node_edges = np.mean([g.node_edge_index.shape[1] for g in graphs])
    print(f"avg same-node edges/graph={avg_node_edges:.1f}")

    idx = list(range(len(graphs)))
    random.shuffle(idx)
    n_test = max(1, int(len(graphs) * args.test_frac))
    test_idx, train_idx = set(idx[:n_test]), idx[n_test:]
    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in sorted(test_idx)]
    print(f"train={len(train_graphs)} test={len(test_graphs)}")

    corpus_root = Path(args.corpus_root)
    # Pre-load + cache the RTT lookup per test dataset (shared by greedy + all models).
    lut_cache: Dict[str, dict] = {}
    for g in test_graphs:
        dsid = str(g.dataset_id)
        if dsid not in lut_cache:
            lut_cache[dsid] = load_combos(corpus_root / dsid)
    base = greedy_baseline_regret(test_graphs, lut_cache)
    print(f"\n[greedy/pointwise-oracle baseline on test] regret mean={base['regret_mean']*100:.2f}% "
          f"p90={base['regret_p90']*100:.2f}% max={base['regret_max']*100:.2f}% (n={base['n']})")
    # Stratify test datasets by COUPLING magnitude (greedy regret): the GNN can only help
    # where independent per-task choice is suboptimal.
    coupling_thresh = 0.01
    coupled_ids = {d for d, r in base["per_ds"].items() if r > coupling_thresh}
    print(f"[coupling] test datasets with greedy regret > {coupling_thresh*100:.0f}%: "
          f"{len(coupled_ids)}/{len(base['per_ds'])}")

    configs = [
        ("pointwise", dict(use_gin=False, use_node_edges=False)),
        ("gnn_base",  dict(use_gin=True,  use_node_edges=False)),
        ("gnn_node",  dict(use_gin=True,  use_node_edges=True)),
    ]
    results = {}
    for name, cfg in configs:
        print(f"\n=== training {name} ({cfg}) ===")
        torch.manual_seed(args.seed); random.seed(args.seed)
        model = AblationModel(task_dim, plat_dim, edge_dim, **cfg).to(device)
        nparam = sum(p.numel() for p in model.parameters())
        print(f"  params={nparam}")
        train_model(model, list(train_graphs), device, args.epochs)
        results[name] = eval_regret(model, test_graphs, corpus_root, device, lut_cache)

    print("\n" + "=" * 92)
    print(f"RESULTS  (corpus={Path(args.cache).name}, test n={len(test_graphs)})")
    print("=" * 92)
    hdr = f"{'model':<12}{'top1_acc':>10}{'regret_mean':>13}{'regret_p90':>12}{'regret_max':>12}{'opt_recov':>11}{'collide':>9}{'missing':>9}"
    print(hdr)
    for name, _ in configs:
        r = results[name]
        print(f"{name:<12}{r['top1_acc']*100:>9.1f}%{r['regret_mean']*100:>12.2f}%{r['regret_p90']*100:>11.2f}%"
              f"{r['regret_max']*100:>11.2f}%{r['opt_recovered_frac']*100:>10.1f}%{r['n_collided_plan']:>9}{r['n_missing_plan']:>9}")
    print(f"\ngreedy baseline regret: mean={base['regret_mean']*100:.2f}% p90={base['regret_p90']*100:.2f}%")

    # Stratified: regret on the COUPLED subset (where joint reasoning matters)
    if coupled_ids:
        print(f"\n--- COUPLED subset only (greedy regret > {coupling_thresh*100:.0f}%, n={len(coupled_ids)}) ---")
        g_c = np.array([base["per_ds"][d] for d in coupled_ids])
        print(f"{'model':<12}{'regret_mean':>13}{'regret_p90':>12}{'regret_max':>12}")
        for name, _ in configs:
            pd = results[name]["per_ds"]
            vals = np.array([pd[d] for d in coupled_ids if d in pd])
            if vals.size == 0:
                continue
            print(f"{name:<12}{vals.mean()*100:>12.2f}%{np.percentile(vals,90)*100:>11.2f}%{vals.max()*100:>11.2f}%")
        print(f"{'greedy':<12}{g_c.mean()*100:>12.2f}%{np.percentile(g_c,90)*100:>11.2f}%{g_c.max()*100:>11.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
