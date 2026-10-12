"""plan_rerank_v1 Phase 1: the learned plan selector over a batch's candidate pool.

The pool (S6, rp/served-decode-check 981f263a; scripts_cosim/plan_rerank_branch_pool.py) is the step-0 branch completions of the
served GNN's masked_topo decode (the first task in decode order fixed to each of its candidates, the rest greedy) together with the
argmin-S plan(s) over the declared slate. Every candidate carries S and its components, the base GNN's plan score and its summed step
logits. The selector picks one plan per batch.

One scorer, two heads (the variants Phase 1 registers):
  (a) "a"  cost = (S - S_min) / sigma + h(P): S plus a learned residual, in units of sigma (the residual is Delta = sigma * h seconds);
  (b) "b"  cost = h(P): a direct scorer.
h(P) = MLP([e(P), g(P)]). e(P) is a plan-level GNN readout: the graph with EVERY task committed at P in the raw-plan encoding
(src/policy/gnn/plan_raw.py; committed flag and committed count on every bipartite edge, typed same-node edges, committed-load
channel), one encode, the mean over tasks of the scorer's output at the task's committed candidate. g(P) are plan-level scalars
(PLAN_FEATURES), normalised within the pool. The MP-OFF twin is the same model with message passing skipped
(TaskPlacementGNN._disable_mp), trained separately with the same recipe, pool and features.

sigma = max(S_min, SIGMA_FLOOR), S_min the smallest S in the pool: a label-free scale known at serving. The selection is the
argmin of cost; a cost tie (within TIE_EPS) goes to an argmin-S candidate, then to the lowest plan in (node id, platform id) order.

Train and serve go through `pool_inputs` and `PlanSelector.costs` on a row of the pool schema (the training row from the pool
jsonl, the serving row built live by src/policy/plan_rerank/live.py), so the two paths cannot price a pool differently.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

KIND = "plan_selector_v1"
LINEAGE = "plan_rerank_v1"
VARIANTS = ("a", "b")
S_COMP_KEYS = ("drain_platform", "drain_batch", "cold", "cold_dup", "exec_lat", "exch")
PLAN_FEATURES = ("s", *S_COMP_KEYS, "plan_score", "step_logit_sum", "is_argmin_s", "is_branch", "log1p_sigma", "log1p_n_tasks")
SIGMA_FLOOR = 1e-3
TIE_EPS = 1e-9

Plan = Tuple[Tuple[int, int], ...]


def plan_key(plan: Sequence[Sequence[Any]]) -> Plan:
    return tuple((int(c[0]), int(c[1])) for c in plan)


def pool_inputs(row: Mapping[str, Any]) -> Tuple[List[Plan], Tensor, Tensor, List[Dict[str, Any]]]:
    """(plans, S [P], features [P, F], candidates) of one pool row, candidates in canonical (plan) order. The row schema is the pool
    jsonl's: n_tasks and candidates[{plan, sources, s, s_comps, plan_score, step_logit_sum}]; rtt is ignored here."""
    cands = sorted(row["candidates"], key=lambda c: plan_key(c["plan"]))
    if not cands:
        raise ValueError("FAIL LOUD: an empty candidate pool")
    plans = [plan_key(c["plan"]) for c in cands]
    if len(set(plans)) != len(plans):
        raise ValueError("FAIL LOUD: a pool lists the same plan twice")
    n = int(row["n_tasks"])
    if any(len(p) != n for p in plans):
        raise ValueError(f"FAIL LOUD: a plan's length differs from n_tasks={n}")
    s = [float(c["s"]) for c in cands]
    for c, sv in zip(cands, s):
        tot = sum(float(c["s_comps"][k]) for k in S_COMP_KEYS if k != "cold_dup")
        if abs(tot - sv) > 1e-9 * max(1.0, abs(sv)):
            raise ValueError(f"FAIL LOUD: S {sv} != the sum of its components {tot}")
    s_min = min(s)
    sigma = max(s_min, SIGMA_FLOOR)
    raw: Dict[str, List[float]] = {k: [] for k in PLAN_FEATURES}
    for c, sv in zip(cands, s):
        src = list(c["sources"])
        raw["s"].append(sv)
        for k in S_COMP_KEYS:
            raw[k].append(float(c["s_comps"][k]))
        raw["plan_score"].append(float(c["plan_score"]))
        raw["step_logit_sum"].append(float(c["step_logit_sum"]))
        raw["is_argmin_s"].append(1.0 if "argmin_s" in src else 0.0)
        raw["is_branch"].append(1.0 if any(str(x).startswith("branch") for x in src) else 0.0)
        raw["log1p_sigma"].append(math.log1p(sigma))
        raw["log1p_n_tasks"].append(math.log1p(n))
    cols = []
    for k in PLAN_FEATURES:
        v = raw[k]
        if k in ("s", *S_COMP_KEYS):
            m = math.fsum(v) / len(v)
            cols.append([(x - m) / sigma for x in v])
        elif k in ("plan_score", "step_logit_sum"):
            m = math.fsum(v) / len(v)
            cols.append([(x - m) / max(1, n) for x in v])
        else:
            cols.append(list(v))
    feats = torch.tensor(cols, dtype=torch.float32).t().contiguous()
    return plans, torch.tensor(s, dtype=torch.float64), feats, cands


def select(costs: Sequence[float], plans: Sequence[Plan], cands: Sequence[Mapping[str, Any]]) -> int:
    """Index of the selected plan: the lowest cost; a tie within TIE_EPS goes to an argmin-S candidate, then to the lowest plan."""
    best = min(float(c) for c in costs)
    tied = [i for i, c in enumerate(costs) if float(c) <= best + TIE_EPS]
    return min(tied, key=lambda i: ("argmin_s" not in cands[i]["sources"], plans[i]))


class PlanSelector(nn.Module):
    def __init__(self, gnn: nn.Module, n_features: int, hidden_dim: int, variant: str, dropout: float = 0.1) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"FAIL LOUD: variant must be one of {VARIANTS}, got {variant!r}")
        if not getattr(gnn, "plan_raw", False):
            raise ValueError("FAIL LOUD: the selector's GNN must be built with plan_raw (the committed-plan encoding)")
        self.gnn = gnn
        self.variant = variant
        self.head = nn.Sequential(nn.Linear(1 + n_features, hidden_dim), nn.ReLU(), nn.Dropout(p=dropout),
                                  nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def plan_readout(self, data: Any, plans: Sequence[Plan]) -> Tensor:
        """[P]: per plan, the mean over tasks of the GNN scorer's output at the task's committed candidate, with every task committed."""
        from src.policy.gnn.plan_raw import plan_raw_edge_attr

        g = copy.copy(data)  # the encode writes partial_state_edge_attr and memo attributes; never on the caller's graph
        n = int(g.n_tasks)
        tl = [[plan_key([c])[0] for c in g.task_logit_to_placement[t]] for t in range(n)]
        out = []
        for plan in plans:
            committed = {t: plan[t] for t in range(n)}
            g.partial_state_edge_attr = plan_raw_edge_attr(g, committed, bool(getattr(self.gnn, "plan_raw_local", False)))
            task_emb, platform_emb = self.gnn._encode(g)
            logits = self.gnn._score(task_emb, platform_emb, g)
            out.append(torch.stack([logits[t][tl[t].index(plan[t])] for t in range(n)]).mean())
        return torch.stack(out)

    def costs(self, data: Any, row: Mapping[str, Any]) -> Tuple[Tensor, List[Plan], List[Dict[str, Any]]]:
        """(cost [P], plans, candidates) of one pool row; lower is better."""
        plans, s, feats, cands = pool_inputs(row)
        e = self.plan_readout(data, plans)
        h = self.head(torch.cat([e.unsqueeze(1), feats.to(e.device)], dim=1)).squeeze(1)
        if self.variant == "a":
            sigma = max(float(s.min()), SIGMA_FLOOR)
            return ((s - s.min()) / sigma).to(h.dtype).to(h.device) + h, plans, cands
        return h, plans, cands


def listwise_loss(cost: Tensor, rtt: Sequence[float], tau_rel: float) -> Tensor:
    """Cross-entropy of the model's softmax(-cost / tau) against softmax(-(RTT - RTT_min) / (tau * sigma_rtt)), sigma_rtt = RTT_min.
    Under (a) the residual that reproduces the target exactly is h = (RTT - S) / sigma up to a constant, when sigma = RTT_min."""
    r = torch.tensor([float(x) for x in rtt], dtype=torch.float64)
    scale = max(float(r.min()), SIGMA_FLOOR)
    q = torch.softmax(-(r - r.min()) / (tau_rel * scale), dim=0).to(cost.dtype).to(cost.device)
    return -(q * torch.log_softmax(-cost / tau_rel, dim=0)).sum()


def pool_weight(rtt: Sequence[float], cap: float) -> float:
    """A pool's weight in the loss: its relative RTT spread (max - min) / min, capped; 0 when every candidate has the same RTT."""
    lo, hi = min(rtt), max(rtt)
    return min(cap, (hi - lo) / max(lo, SIGMA_FLOOR))


def gnn_kwargs_from_contract(c: Mapping[str, Any]) -> Dict[str, Any]:
    g = c["gnn"]
    return dict(task_feature_dim=int(g["task_feature_dim"]), platform_feature_dim=int(g["platform_feature_dim"]),
                embedding_dim=int(g["embedding_dim"]), hidden_dim=int(g["hidden_dim"]), num_layers=int(g["num_layers"]),
                edge_dim=int(g["edge_dim"]), dropout=float(g["dropout"]), post_gin_dropout=float(g["dropout"]),
                normalize_platform_inputs=bool(g["normalize_platform_inputs"]),
                mp_residual=False, mp_node_edges=False, mp_network_entities=False, mp_dag_edges=False,
                mp_peer_edges=bool(g["mp_peer_edges"]), mp_platform_edges=True,
                mp_bipartite_edge_conv=True, mp_bipartite_edge_attr_zero=bool(g["mp_bipartite_edge_attr_zero"]),
                mp_bipartite_aggr="mean", mp_bipartite_hetero=False,
                task_type_onehot_dim=int(g["task_type_onehot_dim"]), partial_state_edge_dim=int(g["partial_state_edge_dim"]),
                plan_raw=True, plan_raw_sum=bool(g["plan_raw_sum"]), plan_raw_local=False)


def build_selector(contract: Mapping[str, Any]) -> PlanSelector:
    """The selector of a contract, every architecture flag explicit (nothing read from the environment: the host process serves a
    base GNN whose own flags live there). MP off is set on the instance, as the trainer set it."""
    from src.policy.gnn.gnn_model import TaskPlacementGNN

    gnn = TaskPlacementGNN(**gnn_kwargs_from_contract(contract))
    gnn._disable_mp = bool(contract["disable_message_passing"])
    return PlanSelector(gnn, len(contract["feature_names"]), int(contract["gnn"]["hidden_dim"]), contract["variant"],
                        float(contract["gnn"]["dropout"]))


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_selector(path: Path, *, base_checkpoint: Optional[Path], allow_smoke: bool = False) -> Tuple[PlanSelector, Dict[str, Any]]:
    """A selector checkpoint and its mandatory sidecar. Fails loud on a foreign kind, a feature list that is not this code's, a base
    checkpoint whose bytes are not the ones the pool was built with, and (unless allowed) a smoke checkpoint."""
    path = Path(path)
    side = path.with_suffix(".contract.json")
    if not side.is_file():
        raise RuntimeError(f"FAIL LOUD: {path.name}: no .contract.json sidecar")
    c = json.loads(side.read_text())
    if c.get("kind") != KIND:
        raise RuntimeError(f"FAIL LOUD: {path.name}: sidecar kind {c.get('kind')!r} is not {KIND}")
    if tuple(c.get("feature_names") or ()) != PLAN_FEATURES:
        raise RuntimeError(f"FAIL LOUD: {path.name}: trained on features {c.get('feature_names')}, this code computes {PLAN_FEATURES}")
    if c.get("smoke") and not allow_smoke:
        raise RuntimeError(f"FAIL LOUD: {path.name} is a smoke checkpoint; it serves only with the smoke flag")
    if base_checkpoint is not None:
        got = md5_of(Path(base_checkpoint))
        if got != c.get("base_checkpoint_md5"):
            raise RuntimeError(f"FAIL LOUD: {path.name} was trained on the pool of base {c.get('base_checkpoint')} "
                               f"(md5 {c.get('base_checkpoint_md5')}); the served base {base_checkpoint} has md5 {got}")
    from src.policy.gnn.prefix_serving import _task_type_vocab

    if list(c.get("dag_task_type_vocab") or []) != list(_task_type_vocab()):
        raise RuntimeError(f"FAIL LOUD: {path.name}: task-type vocab {c.get('dag_task_type_vocab')} != live {list(_task_type_vocab())}")
    model = build_selector(c)
    model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    model.eval()
    return model, c
