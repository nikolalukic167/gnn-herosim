#!/usr/bin/env python3
"""raw_plan_v2 train/serve parity (docs/lineages/raw_plan_v2.md, Step 3).

  raw_plan_v2_parity.py cache  --ckpt <pt> --config <experiments yaml> --cache <graphs_cache dir> [--n-graphs 20]
  raw_plan_v2_parity.py replay --ckpt <pt> --config <experiments yaml> --dump <dir written under GNN_PLAN_RAW_DUMP>

Two models from one checkpoint:
- **serving**: `prefix_serving.load_prefix_conditioned_gnn`, i.e. what the live gate builds from the sidecar;
- **trainer**: built from the experiment YAML's env (the source of truth the trainer read), not from the sidecar.

`cache`: on strided cached graphs, a one-pass id-order greedy decode plus two self-refine passes. At every step,
both models score the same committed set through the shared closure. The raw-plan block and every task's logits
must agree bitwise.

`replay`: graphs and logits the live simulator dumped are re-scored by the trainer model and must match bitwise.
Fails loud (exit 1) on the first mismatch, or on an empty dump.
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
from typing import Any, Dict, List, Mapping, Tuple

import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.policy.gnn.gnn_model import TaskPlacementGNN  # noqa: E402
from src.policy.gnn.partial_state_edges import make_partial_state_score_fn  # noqa: E402
from src.policy.gnn.plan_raw import plan_raw_dim, plan_raw_edge_attr  # noqa: E402
from src.policy.gnn.prefix_serving import load_prefix_conditioned_gnn  # noqa: E402


def _flag(env: Mapping[str, str], name: str) -> bool:
    return str(env.get(name, "0")) == "1"


def trainer_model(ckpt: str, config: str) -> TaskPlacementGNN:
    env = (yaml.safe_load(open(config)) or {}).get("env") or {}
    sd = torch.load(ckpt, map_location="cpu")
    prev = os.environ.get("GNN_DISABLE_MESSAGE_PASSING")
    os.environ["GNN_DISABLE_MESSAGE_PASSING"] = "1" if _flag(env, "GNN_DISABLE_MESSAGE_PASSING") else "0"
    try:
        model = TaskPlacementGNN(
            task_feature_dim=int(sd["task_encoder.net.0.weight"].shape[1]) - (4 if _flag(env, "NEAR_RTT_TASK_TYPE_ONEHOT") else 0),
            platform_feature_dim=int(sd["platform_encoder.net.0.weight"].shape[1]),
            embedding_dim=int(sd["task_encoder.net.4.weight"].shape[0]),
            hidden_dim=int(sd["task_encoder.net.0.weight"].shape[0]),
            num_layers=sum(1 for k in sd if k.startswith("gin.convs.") and k.endswith(".nn.lins.0.weight")),
            mp_peer_edges=_flag(env, "NEAR_RTT_MP_PEER_EDGES"),
            mp_bipartite_edge_conv=_flag(env, "NEAR_RTT_MP_BIPARTITE_EDGE_CONV"),
            mp_bipartite_edge_attr_zero=_flag(env, "NEAR_RTT_MP_BIPARTITE_EDGE_ATTR_ZERO"),
            task_type_onehot_dim=4 if _flag(env, "NEAR_RTT_TASK_TYPE_ONEHOT") else 0,
            partial_state_edge_dim=plan_raw_dim(_flag(env, "NEAR_RTT_PLAN_RAW_LOCAL")),
            plan_raw=_flag(env, "NEAR_RTT_PLAN_RAW"),
            plan_raw_sum=_flag(env, "NEAR_RTT_PLAN_RAW_SUM"),
            plan_raw_local=_flag(env, "NEAR_RTT_PLAN_RAW_LOCAL"),
        )
    finally:
        if prev is None:
            os.environ.pop("GNN_DISABLE_MESSAGE_PASSING", None)
        else:
            os.environ["GNN_DISABLE_MESSAGE_PASSING"] = prev
    model.load_state_dict(sd)
    model.eval()
    return model


def _fail(msg: str) -> None:
    print(f"FAIL LOUD: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _same(a: List[torch.Tensor], b: List[torch.Tensor], where: str) -> None:
    if len(a) != len(b) or any(not torch.equal(x, y) for x, y in zip(a, b)):
        worst = max((float((x - y).abs().max()) for x, y in zip(a, b) if x.numel()), default=float("nan"))
        _fail(f"logits differ at {where} (max |d| {worst:.3e})")


def _scores(model: Any, graph: Any, committed: Mapping[int, Tuple[int, int]]) -> List[torch.Tensor]:
    with torch.no_grad():
        fn = make_partial_state_score_fn(model, graph, None)
        return [fn(t, committed).detach().clone() for t in range(int(graph.n_tasks))]


def _argmax(logits: torch.Tensor, cands) -> Tuple[int, int]:
    return min((-float(logits[i]), (int(c[0]), int(c[1]))) for i, c in enumerate(cands))[1]


def run_cache(serving: Any, trainer: Any, cache: str, n_graphs: int) -> int:
    graphs = pickle.load(open(os.path.join(cache, "graphs.pkl"), "rb"))
    stride = max(1, len(graphs) // n_graphs)
    checked = 0
    for gi in list(range(0, len(graphs), stride))[:n_graphs]:
        g = graphs[gi]
        tl = g.task_logit_to_placement
        n = int(g.n_tasks)
        plan: Dict[int, Tuple[int, int]] = {}
        steps = [("decode", t) for t in range(n)] + [("refine", t) for _ in range(2) for t in range(n)]
        for phase, t in steps:
            committed = dict(plan) if phase == "decode" else {j: p for j, p in plan.items() if j != t}
            attr_s = plan_raw_edge_attr(g, committed, bool(getattr(trainer, 'plan_raw_local', False)))
            attr_t = plan_raw_edge_attr(g, committed, bool(getattr(trainer, 'plan_raw_local', False)))
            if not torch.equal(attr_s, attr_t):
                _fail(f"graph {gi} {phase} t={t}: raw-plan block not deterministic")
            ls, lt = _scores(serving, g, committed), _scores(trainer, g, committed)
            _same(ls, lt, f"graph {gi} {phase} t={t} |committed|={len(committed)}")
            plan[t] = _argmax(ls[t], tl[t])
            checked += 1
    return checked


def run_replay(trainer: Any, dump: str) -> int:
    files = sorted(glob.glob(os.path.join(dump, "call_*.pt")))
    if not files:
        _fail(f"no dumped calls under {dump}: the instrument never wrote")
    for f in files:
        rec = torch.load(f, map_location="cpu", weights_only=False)
        _same(rec["logits"], _scores(trainer, rec["graph"], rec["committed"]), os.path.basename(f))
    return len(files)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("cache", "replay"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache")
    ap.add_argument("--dump")
    ap.add_argument("--n-graphs", type=int, default=20)
    a = ap.parse_args()
    torch.use_deterministic_algorithms(True)
    trainer = trainer_model(a.ckpt, a.config)
    if a.mode == "cache":
        serving, _, _ = load_prefix_conditioned_gnn(a.ckpt, adopt_env=True)
        if serving.state_dict().keys() != trainer.state_dict().keys():
            _fail("serving and trainer models have different parameter sets")
        for flag in ("plan_raw", "plan_raw_sum", "plan_raw_local", "mp_bipartite_edge_attr_zero", "_disable_mp", "mp_bipartite_aggr"):
            if getattr(serving, flag, None) != getattr(trainer, flag, None):
                _fail(f"{flag}: serving {getattr(serving, flag, None)!r} != trainer {getattr(trainer, flag, None)!r}")
        n = run_cache(serving, trainer, a.cache, a.n_graphs)
    else:
        n = run_replay(trainer, a.dump)
    print(f"[parity] {a.mode} ok: {os.path.basename(a.ckpt)} -- {n} scored steps, bitwise identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
