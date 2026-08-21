#!/usr/bin/env python3
"""GNN-necessity ablation: does message passing (and same-node aggregation) beat a
pointwise edge scorer on co-sim placement labels?

Four models share an IDENTICAL encoder + edge scorer; the ONLY difference is the
relational structure used between encoding and scoring:

  pointwise   : no message passing            (feature-parity MLP baseline)
  gnn_base    : GIN over task<->platform edges (bipartite message passing)
  gnn_node    : GIN over task<->platform + same-node platform<->platform edges
                (the node-aggregation prototype: signal a pointwise model cannot see)
  gnn_topo    : GIN over task<->platform edges + backbone/link network entities
                (use_network_entities=True; the only arm with topology awareness at all)

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
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.models import GIN

_NOTEBOOKS = Path(__file__).resolve().parents[1] / "src" / "notebooks"
if str(_NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS))
from src.placement.network_graph import (  # noqa: E402
    NET_LINK_FEATURE_DIM,
    NET_NODE_FEATURE_DIM,
)
from src.policy.gnn.gnn_model import split_task_platform_embeddings  # noqa: E402
from non_unique_lib.training_contract import (  # noqa: E402
    assert_zero_parent_overlap,
    split_ids_by_canonical_parent,
    split_ids_by_topology_size,
    topology_sizes_by_parent,
)

_SCRIPTS_COSIM = Path(__file__).resolve().parent
if str(_SCRIPTS_COSIM) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_COSIM))
from gate_statistics import (  # noqa: E402
    PHASE4_TIERS,
    escalation_note,
    format_comparison_table,
    paired_regret_comparison,
    phase4_verdict,
    power_note,
)


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
                 use_network_entities: bool = False,
                 net_node_dim: int = NET_NODE_FEATURE_DIM,
                 net_link_dim: int = NET_LINK_FEATURE_DIM,
                 emb: int = 64, hidden: int = 128, layers: int = 3):
        super().__init__()
        self.use_gin = use_gin
        self.use_node_edges = use_node_edges
        # Network entities come from the cache, which builds them through the SAME
        # src/placement/network_graph.py path live inference uses. This harness deliberately
        # does not construct any graph of its own — a third construction is how the formulas
        # this repo has already had to de-duplicate (queue, topology, temporal) got their
        # divergences in the first place.
        self.use_network_entities = use_network_entities
        self.task_enc = MLPEncoder(task_dim, hidden, emb)
        self.plat_enc = MLPEncoder(plat_dim, hidden, emb)
        if use_network_entities:
            self.net_node_enc = MLPEncoder(net_node_dim, hidden, emb)
            self.net_link_enc = MLPEncoder(net_link_dim, hidden, emb)
        self.gin = GIN(in_channels=emb, hidden_channels=hidden, num_layers=layers, out_channels=emb) if use_gin else None
        self.post_drop = nn.Dropout(0.2)
        self.scorer = EdgeScorer(emb, hidden, edge_dim)

    def forward(self, g) -> List[Tensor]:
        nt, npl = int(g.n_tasks), int(g.n_platforms)
        te = self.task_enc(g.task_features)
        pe = self.plat_enc(g.platform_features)
        if self.use_gin:
            blocks = [te, pe]
            extra = []
            if self.use_node_edges and getattr(g, "node_edge_index", None) is not None and g.node_edge_index.numel() > 0:
                extra.append(g.node_edge_index.to(g.edge_index.device))
            if self.use_network_entities:
                missing = [
                    n for n in ("net_node_features", "net_link_features", "net_edge_index")
                    if getattr(g, n, None) is None
                ]
                if missing:
                    raise ValueError(
                        f"FAIL LOUD: use_network_entities is on but the cached graph is "
                        f"missing {missing}. Rebuild the cache with "
                        f"NETWORK_GRAPH_CONTRACT=core_v1 over a corpus that has a "
                        f"link_topology."
                    )
                blocks.extend([
                    self.net_node_enc(g.net_node_features),
                    self.net_link_enc(g.net_link_features),
                ])
                if g.net_edge_index.numel() > 0:
                    extra.append(g.net_edge_index.to(g.edge_index.device))
            x0 = torch.cat(blocks, dim=0)
            ei = torch.cat([g.edge_index] + extra, dim=1) if extra else g.edge_index
            # Residual around GIN: message passing AUGMENTS the per-node encoding instead of
            # replacing it. Prevents oversmoothing/training-collapse when many (e.g. same-node)
            # edges flood aggregation, and guarantees GNN capacity >= pointwise.
            x = x0 + self.post_drop(self.gin(x0, ei))
            # Shared bounded split — see split_task_platform_embeddings for why this is not
            # written inline as `x[nt:]`.
            te, pe = split_task_platform_embeddings(x, nt, npl)
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
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"{jp}:{line_number}: invalid JSON") from exc
            plan, rtt = rec.get("placement_plan"), rec.get("rtt")
            if plan is None or rtt is None:
                raise RuntimeError(
                    f"{jp}:{line_number}: missing placement_plan or rtt"
                )
            key = tuple(sorted((int(k), (int(v[0]), int(v[1]))) for k, v in plan.items()))
            r = float(rtt)
            if key not in lut or r < lut[key]:
                lut[key] = r
    return lut or None


def plan_key(plan: Dict[int, Tuple[int, int]]) -> Tuple:
    return tuple(sorted((t, (n, p)) for t, (n, p) in plan.items()))


# --------------------------------------------------------------------------------------
# Preflight: the labels the models are trained and scored against must BE the sweep optima.
#
# The separability gate (separability_diagnostic.py) measures coupling in placements.jsonl.
# The ablation measures models against cache labels. Those are two different objects, and
# nothing used to check they agree -- so a cache whose labels came from
# optimal_result.json's `sample.placement_plan` (which is not guaranteed to be the sweep
# minimum) would silently invalidate a run while every other check passed. Audit here, on
# every graph, before a single epoch is spent.
# --------------------------------------------------------------------------------------
def label_plan(g) -> Optional[Dict[int, Tuple[int, int]]]:
    """Decode a graph's oracle label into a joint (node, platform) plan."""
    if not hasattr(g, "y"):
        return None
    l2p = g.task_logit_to_placement
    plan: Dict[int, Tuple[int, int]] = {}
    for t in range(int(g.n_tasks)):
        if t >= g.y.numel() or t >= len(l2p):
            return None
        idx = int(g.y[t].item())
        if idx < 0 or idx >= len(l2p[t]):
            return None
        plan[t] = tuple(l2p[t][idx])
    return plan or None


def audit_label_provenance(graphs, corpus_root: Path) -> dict:
    """Fail loud unless every cache label is its dataset's placements.jsonl minimum.

    Streams each sweep once, keeping only the running minimum and the label combo's RTT,
    so this stays cheap on corpora far larger than the graph cache itself.
    """
    n_checked = 0
    undecodable: List[str] = []
    absent: List[str] = []
    drifted: List[Tuple[str, float]] = []

    for g in graphs:
        dsid = str(g.dataset_id)
        plan = label_plan(g)
        if plan is None:
            undecodable.append(dsid)
            continue
        key = plan_key(plan)
        jp = corpus_root / dsid / "placements" / "placements.jsonl"
        if not jp.is_file():
            raise RuntimeError(f"Label audit: missing placements.jsonl for {dsid}: {jp}")

        min_rtt: Optional[float] = None
        label_rtt: Optional[float] = None
        with jp.open() as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception as exc:
                    raise RuntimeError(f"{jp}:{line_number}: invalid JSON") from exc
                rplan, rtt = rec.get("placement_plan"), rec.get("rtt")
                if rplan is None or rtt is None:
                    raise RuntimeError(
                        f"{jp}:{line_number}: missing placement_plan or rtt"
                    )
                r = float(rtt)
                rkey = tuple(
                    sorted((int(k), (int(v[0]), int(v[1]))) for k, v in rplan.items())
                )
                if min_rtt is None or r < min_rtt:
                    min_rtt = r
                if rkey == key and (label_rtt is None or r < label_rtt):
                    label_rtt = r
        if min_rtt is None:
            raise RuntimeError(f"Label audit: empty sweep for {dsid}: {jp}")

        n_checked += 1
        if label_rtt is None:
            absent.append(dsid)
        elif label_rtt > min_rtt + 1e-9:
            drifted.append((dsid, (label_rtt - min_rtt) / min_rtt))

    regrets = [r for _, r in drifted]
    report = {
        "n_checked": n_checked,
        "n_label_is_sweep_min": n_checked - len(drifted) - len(absent),
        "n_label_suboptimal": len(drifted),
        "n_label_absent_from_sweep": len(absent),
        "n_label_undecodable": len(undecodable),
        "label_regret_mean": float(np.mean(regrets)) if regrets else 0.0,
        "label_regret_max": float(np.max(regrets)) if regrets else 0.0,
        "worst_datasets": [
            {"dataset_id": d, "regret": r}
            for d, r in sorted(drifted, key=lambda kv: -kv[1])[:10]
        ],
    }

    failures = []
    if undecodable:
        failures.append(f"{len(undecodable)} graphs have undecodable labels: {undecodable[:5]}")
    if absent:
        failures.append(
            f"{len(absent)} labels are absent from their sweep: {absent[:5]}"
        )
    if drifted:
        worst = sorted(drifted, key=lambda kv: -kv[1])[:5]
        failures.append(
            f"{len(drifted)}/{n_checked} labels are not the sweep minimum "
            f"(mean regret {100 * report['label_regret_mean']:.2f}%, "
            f"max {100 * report['label_regret_max']:.2f}%); worst: "
            + ", ".join(f"{d} +{100 * r:.1f}%" for d, r in worst)
        )
    if failures:
        raise RuntimeError(
            "LABEL PROVENANCE AUDIT FAILED -- the cache labels are not the brute-force "
            "optima, so every number this ablation would print is measured against the "
            "wrong target. Rebuild the cache with prepare_graphs_cache.py (label_source="
            "placements.jsonl_sweep_minimum) before rerunning.\n  - "
            + "\n  - ".join(failures)
        )
    return report


# --------------------------------------------------------------------------------------
# Eval: RTT regret of a model's argmax joint plan vs the oracle optimum.
# --------------------------------------------------------------------------------------
def eval_regret(model, graphs, corpus_root: Path, device, lut_cache: dict) -> dict:
    model.eval()
    regrets, top1_hits, top1_n = [], 0, 0
    found, collided, missing_collided, missing_clean = 0, 0, 0, 0
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
            has_collision = len(set(plan.values())) < len(plan)
            if has_collision:
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
            elif has_collision:
                # Expected: a colliding plan (two tasks on the same platform) is not a
                # jointly-feasible combination, so a brute-force sweep correctly omits it.
                missing_collided += 1
            else:
                # No collision explains the absence -- the sweep should contain every
                # jointly-feasible combination, so this is a corpus/harness bug.
                missing_clean += 1
    regrets_arr = np.array(regrets) if regrets else np.array([0.0])
    return {
        "n_eval": len(graphs),
        "n_found": found,
        "n_missing_collided": missing_collided,
        "n_missing_clean": missing_clean,
        "n_missing_plan": missing_collided + missing_clean,
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
    ap.add_argument(
        "--test-frac",
        type=float,
        default=0.3,
        help="Holdout parent fraction before val/test split (default 0.3 → 70/15/15)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--models",
        nargs="+",
        choices=["pointwise", "gnn_base", "gnn_node", "gnn_topo"],
        default=["pointwise", "gnn_base", "gnn_node"],
    )
    ap.add_argument("--expected-graphs", type=int, default=0)
    ap.add_argument(
        "--skip-label-audit",
        action="store_true",
        help="Skip the sweep-minimum label provenance preflight (NOT for reported runs)",
    )
    ap.add_argument("--output", type=Path)
    ap.add_argument(
        "--power-tier",
        default="tier_0.02",
        choices=[t["name"] for t in PHASE4_TIERS],
        help=(
            "Which pre-registered power tier this run is being read at "
            "(gate_statistics.PHASE4_TIERS). Decides the escalation target when the "
            "win_rate CI straddles 0.5; does not change any computed statistic."
        ),
    )
    ap.add_argument(
        "--nondeterministic",
        action="store_true",
        help=(
            "Disable torch deterministic algorithms. Reproduces the pre-2026-08-19 "
            "behaviour where two identical commands give different GIN results. "
            "NOT for reported runs -- see the note in main()."
        ),
    )
    ap.add_argument(
        "--split-mode",
        choices=["canonical_parent", "copy_shuffle", "topology_size"],
        default="canonical_parent",
        help=(
            "canonical_parent = training-contract 70/15/15; copy_shuffle = legacy; "
            "topology_size = topology_transfer_v1 Phase 4 gate (train on "
            "--train-sizes, test on --held-out-sizes, val is a held-out slice of "
            "the train sizes only)"
        ),
    )
    ap.add_argument(
        "--train-sizes", type=int, nargs="+", default=[20, 28, 40],
        help="topology_size split only: server_node_count values used for train/val",
    )
    ap.add_argument(
        "--held-out-sizes", type=int, nargs="+", default=[60, 80],
        help="topology_size split only: server_node_count values held out as test",
    )
    args = ap.parse_args()

    # REPRODUCIBILITY. Measured 2026-08-19: at a FIXED seed on CPU, `pointwise` is
    # bit-identical run to run while `gnn_base`/`gnn_node` are not -- the training loss
    # itself diverges (mean_ce 0.9604 vs 0.9601 at epoch 5). So the non-reproducibility
    # recorded in this repo is NOT CUDA-specific, as the pre-registered gate's control 1
    # assumed; it is a non-deterministic op in the GIN autograd path and it fires on CPU
    # too. It is not intra-op threading (OMP_NUM_THREADS=1 still diverges) and not
    # PYTHONHASHSEED. This one line makes all three models bit-identical across repeated
    # identical commands, verified on every reported statistic.
    #
    # Why this matters for the gate: without it, ">=3 seeds" conflates seed effects with
    # run-to-run training noise, and the noise is the LARGER of the two -- win_rate moved
    # 0.517 -> 0.550 between two identical seed-44 runs, against a seed-to-seed spread of
    # only 0.517-0.533. Any seed spread measured without this is uninterpretable.
    if not args.nondeterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  cache={args.cache}  "
          f"deterministic={not args.nondeterministic}")

    graphs = pickle.load(open(os.path.join(args.cache, "graphs.pkl"), "rb"))
    dataset_ids_path = os.path.join(args.cache, "dataset_ids.pkl")
    if os.path.isfile(dataset_ids_path):
        dataset_ids = pickle.load(open(dataset_ids_path, "rb"))
    else:
        dataset_ids = [str(g.dataset_id) for g in graphs]
    if args.expected_graphs and len(graphs) != args.expected_graphs:
        raise RuntimeError(
            f"Cache graph count {len(graphs)} != expected {args.expected_graphs}"
        )
    if args.limit:
        graphs = graphs[: args.limit]
        dataset_ids = dataset_ids[: args.limit]
    if len(graphs) != len(dataset_ids):
        raise RuntimeError(
            f"graphs ({len(graphs)}) != dataset_ids ({len(dataset_ids)})"
        )
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

    corpus_root = Path(args.corpus_root)
    if args.skip_label_audit:
        label_audit = {"skipped": True}
        print("\n[label audit] SKIPPED -- results are not reportable")
    else:
        label_audit = audit_label_provenance(graphs, corpus_root)
        print(
            f"\n[label audit] {label_audit['n_label_is_sweep_min']}/"
            f"{label_audit['n_checked']} labels are the placements.jsonl minimum -> OK"
        )

    if args.split_mode == "canonical_parent":
        (
            train_graphs,
            train_ids,
            val_graphs,
            val_ids,
            test_graphs,
            test_ids,
        ) = split_ids_by_canonical_parent(
            graphs,
            dataset_ids,
            test_size=args.test_frac,
            val_fraction_of_holdout=0.5,
            random_state=args.seed,
        )
        assert_zero_parent_overlap(train_ids, val_ids, test_ids)
        print(
            f"split=canonical_parent train={len(train_graphs)} "
            f"val={len(val_graphs)} test={len(test_graphs)}"
        )
    elif args.split_mode == "topology_size":
        sizes_by_parent = topology_sizes_by_parent(dataset_ids, corpus_root)
        (
            train_graphs,
            train_ids,
            val_graphs,
            val_ids,
            test_graphs,
            test_ids,
        ) = split_ids_by_topology_size(
            graphs,
            dataset_ids,
            sizes_by_parent,
            train_sizes=args.train_sizes,
            held_out_sizes=args.held_out_sizes,
            val_fraction_of_train=0.15,
            random_state=args.seed,
        )
        assert_zero_parent_overlap(train_ids, val_ids, test_ids)
        print(
            f"split=topology_size train_sizes={args.train_sizes} "
            f"held_out_sizes={args.held_out_sizes} train={len(train_graphs)} "
            f"val={len(val_graphs)} test={len(test_graphs)}"
        )
    else:
        idx = list(range(len(graphs)))
        random.shuffle(idx)
        n_test = max(1, int(len(graphs) * args.test_frac))
        test_idx, train_idx = set(idx[:n_test]), idx[n_test:]
        train_graphs = [graphs[i] for i in train_idx]
        test_graphs = [graphs[i] for i in sorted(test_idx)]
        val_graphs, val_ids, train_ids, test_ids = [], [], [], []
        print(f"split=copy_shuffle train={len(train_graphs)} test={len(test_graphs)}")

    # Pre-load + cache the RTT lookup per test dataset (shared by greedy + all models).
    lut_cache: Dict[str, dict] = {}
    for g in test_graphs:
        dsid = str(g.dataset_id)
        if dsid not in lut_cache:
            lut_cache[dsid] = load_combos(corpus_root / dsid)
            if not lut_cache[dsid]:
                raise RuntimeError(f"Missing RTT sweep for retained graph: {dsid}")
    base = greedy_baseline_regret(test_graphs, lut_cache)
    print(f"\n[greedy/pointwise-oracle baseline on test] regret mean={base['regret_mean']*100:.2f}% "
          f"p90={base['regret_p90']*100:.2f}% max={base['regret_max']*100:.2f}% (n={base['n']})")
    # Stratify test datasets by COUPLING magnitude (greedy regret): the GNN can only help
    # where independent per-task choice is suboptimal.
    coupling_thresh = 0.01
    coupled_ids = {d for d, r in base["per_ds"].items() if r > coupling_thresh}
    print(f"[coupling] test datasets with greedy regret > {coupling_thresh*100:.0f}%: "
          f"{len(coupled_ids)}/{len(base['per_ds'])}")

    all_configs = [
        ("pointwise", dict(use_gin=False, use_node_edges=False)),
        ("gnn_base",  dict(use_gin=True,  use_node_edges=False)),
        ("gnn_node",  dict(use_gin=True,  use_node_edges=True)),
        # Only arm that gives the model access to backbone/link topology at all --
        # gnn_base/gnn_node never see network entities, so a loss for them says nothing
        # about whether topology-aware message passing helps. Requires a cache built with
        # NETWORK_GRAPH_CONTRACT=core_v1 (raises loudly otherwise, see AblationModel).
        ("gnn_topo",  dict(use_gin=True,  use_node_edges=False, use_network_entities=True)),
    ]
    configs = [(name, cfg) for name, cfg in all_configs if name in args.models]
    results = {}
    for name, cfg in configs:
        print(f"\n=== training {name} ({cfg}) ===")
        torch.manual_seed(args.seed); random.seed(args.seed)
        model = AblationModel(task_dim, plat_dim, edge_dim, **cfg).to(device)
        nparam = sum(p.numel() for p in model.parameters())
        print(f"  params={nparam}")
        train_model(model, list(train_graphs), device, args.epochs)
        results[name] = eval_regret(model, test_graphs, corpus_root, device, lut_cache)
        if results[name]["n_missing_clean"]:
            raise RuntimeError(
                f"{name}: {results[name]['n_missing_clean']} predicted plans absent from "
                "the retained full placement sweep with NO collision to explain it -- a "
                "brute-force sweep should contain every jointly-feasible combination, so "
                "this is a corpus/harness bug, not a model artifact. "
                f"(separately, {results[name]['n_missing_collided']} plans were missing "
                "BECAUSE they collided -- that's expected and reported, not raised on)"
            )

    print("\n" + "=" * 92)
    print(f"RESULTS  (corpus={Path(args.cache).name}, test n={len(test_graphs)})")
    print("=" * 92)
    hdr = (f"{'model':<12}{'top1_acc':>10}{'regret_mean':>13}{'regret_p90':>12}{'regret_max':>12}"
           f"{'opt_recov':>11}{'collide':>9}{'miss_coll':>11}{'miss_clean':>12}")
    print(hdr)
    for name, _ in configs:
        r = results[name]
        print(f"{name:<12}{r['top1_acc']*100:>9.1f}%{r['regret_mean']*100:>12.2f}%{r['regret_p90']*100:>11.2f}%"
              f"{r['regret_max']*100:>11.2f}%{r['opt_recovered_frac']*100:>10.1f}%{r['n_collided_plan']:>9}"
              f"{r['n_missing_collided']:>11}{r['n_missing_clean']:>12}")
    print(f"\ngreedy baseline regret: mean={base['regret_mean']*100:.2f}% p90={base['regret_p90']*100:.2f}%")

    # PRIMARY gate statistics. The regret table above is reported for continuity with
    # earlier LINEAGES rows but is NOT gated on: measured on shallow_v1, a decision rule
    # of constant expressive power still drifts 2.58x in regret_mean across sweep-size
    # bins, and trimming/log-scaling do not fix it (2.27 / 2.64). See gate_statistics.py.
    comparisons = {}
    verdicts = {}
    ref_name = "pointwise"
    if ref_name in results:
        for name, _ in configs:
            if name == ref_name:
                continue
            comparisons[name] = paired_regret_comparison(
                results[name]["per_ds"], results[ref_name]["per_ds"]
            )
        if comparisons:
            print()
            print(format_comparison_table(
                comparisons, ref_name, [n for n, _ in configs if n != ref_name]
            ))
            for name, cmp in comparisons.items():
                print(f"  {name}: {power_note(cmp)}")
            # Pre-registered escalation rule (gate_statistics.PHASE4_TIERS, fixed
            # 2026-08-19 before any topo_transfer_v1 corpus existed). A CI straddling
            # 0.5 is an under-powered result, not a null one, and the two must not be
            # reported the same way -- only a CI excluding 0.5 on the reference's side
            # licenses "does not transfer".
            print()
            for name, cmp in comparisons.items():
                verdicts[name] = phase4_verdict(cmp, tier_name=args.power_tier)
                print(f"  {name}: {escalation_note(verdicts[name])}")

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
    if args.output:
        coupled_results = {}
        for name, _ in configs:
            values = np.array(
                [
                    results[name]["per_ds"][dataset_id]
                    for dataset_id in coupled_ids
                    if dataset_id in results[name]["per_ds"]
                ]
            )
            coupled_results[name] = {
                "n": int(values.size),
                "regret_mean": float(values.mean()) if values.size else None,
                "regret_p90": (
                    float(np.percentile(values, 90)) if values.size else None
                ),
                "regret_max": float(values.max()) if values.size else None,
            }
        payload = {
            "schema_version": 6,
            "paired_comparisons": comparisons,
            "phase4_verdicts": verdicts,
            "power_tier": args.power_tier,
            "paired_reference": ref_name,
            "label_audit": label_audit,
            "cache": str(args.cache),
            "corpus_root": str(corpus_root),
            "seed": args.seed,
            "epochs": args.epochs,
            "test_fraction": args.test_frac,
            "split_mode": args.split_mode,
            "train_sizes": args.train_sizes if args.split_mode == "topology_size" else None,
            "held_out_sizes": args.held_out_sizes if args.split_mode == "topology_size" else None,
            "n_graphs": len(graphs),
            "n_train": len(train_graphs),
            "n_val": len(val_graphs),
            "n_test": len(test_graphs),
            "models": args.models,
            "greedy_baseline": base,
            "coupled_threshold": coupling_thresh,
            "coupled_dataset_ids": sorted(coupled_ids),
            "coupled_results": coupled_results,
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"\nFrozen ablation report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
