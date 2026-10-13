"""uncut_groups_s0_v1 Arm 2: one training step on a real large Arm 1 state, with the yaml's flags, for wall time and peak memory.

The loss is the trainer's own `loss_tied_teacher_forced_ce`, taken from the source of src/notebooks/train_near_rtt.py (which trains at
import and cannot be imported) and executed against the yaml's configuration, so what is timed is the code that will train.

    python3 scripts_cosim/uncut_arm2_step_smoke.py --cache <cache_dir> --yaml experiments/uncut_arm2_gnn_eng.yaml --out smoke.json
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import pickle
import resource
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--min-tasks", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-tasks", type=int, default=0, help="pick the graph whose n_tasks is closest to this instead of the largest")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.yaml))
    env = {str(k): str(v) for k, v in (cfg.get("env") or {}).items()}
    os.environ.update(env)
    for key in ("PARTIAL_STATE_CONTRACT", "PARTIAL_STATE_LOAD_SECONDS", "PARTIAL_STATE_PEER_MASS", "PARTIAL_STATE_EXCHANGE_SECONDS"):
        os.environ.setdefault(key, env.get(key, ""))

    from src.policy.gnn import arm2 as _arm2
    from src.policy.gnn.gnn_model import TaskPlacementGNN
    from src.policy.gnn.partial_state_edges import make_partial_state_score_fn
    from src.policy.gnn.seq_decode import topological_task_order
    from src.policy.tabular.reduced_features import build_partial_state_context_from_graph, partial_state_feature_dim

    tree = ast.parse((REPO / "src/notebooks/train_near_rtt.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "loss_tied_teacher_forced_ce")
    ns: Dict[str, Any] = dict(
        torch=torch, F=F, math=__import__("math"), _arm2=_arm2, Dict=Dict, List=List, Tuple=Tuple, Any=Any, Sequence=Sequence,
        Tensor=torch.Tensor, topological_task_order=topological_task_order,
        make_partial_state_score_fn=make_partial_state_score_fn,
        build_partial_state_context_from_graph=build_partial_state_context_from_graph, _TRAIN_SEED=1,
        _ANCHOR_STATS={s: {"ce": 0.0, "floor": 0.0, "top1": 0.0, "groups": 0.0} for s in ("train", "val")},
        NEAR_CFG=types.SimpleNamespace(
            dag_alpha_key=env.get("NEAR_RTT_DAG_ALPHA_KEY", "inf"), tied_max_plans=int(env.get("NEAR_RTT_TIED_MAX_PLANS", "0")),
            anchor_ce_weight=float(env.get("NEAR_RTT_ANCHOR_CE_WEIGHT", "0.5")),
            score_task_rows_only=env.get("NEAR_RTT_SCORE_TASK_ROWS_ONLY", "0") == "1",
        ),
    )
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "train_near_rtt.py:loss_tied_teacher_forced_ce", "exec"), ns)
    loss_fn = ns["loss_tied_teacher_forced_ce"]

    with open(Path(args.cache) / "graphs.pkl", "rb") as fh:
        graphs = pickle.load(fh)
    cands = [g for g in graphs if int(g.n_tasks) >= args.min_tasks and (getattr(g, "tied_optimal_logit_plans", {}) or {}).get("inf")]
    if not cands:
        raise SystemExit(f"FAIL LOUD: no graph with >= {args.min_tasks} tasks and an 'inf' label in {args.cache}")
    graph = (min(cands, key=lambda g: abs(int(g.n_tasks) - args.target_tasks)) if args.target_tasks else max(cands, key=lambda g: int(g.n_tasks)))
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)
    args_cfg = cfg.get("args") or {}
    emb, hid = int(args_cfg.get("embedding-dim", 128)), int(args_cfg.get("hidden-dim", 128))
    contract = env.get("PARTIAL_STATE_CONTRACT", "partial_state_v5")
    model = TaskPlacementGNN(
        task_feature_dim=int(graph.task_features.shape[1]), platform_feature_dim=int(graph.platform_features.shape[1]),
        embedding_dim=emb, hidden_dim=hid, num_layers=3, dropout=0.1, post_gin_dropout=0.1,
        normalize_platform_inputs=int(graph.platform_features.shape[1]) == 21,
        mp_peer_edges=env.get("NEAR_RTT_MP_PEER_EDGES") == "1",
        mp_bipartite_edge_conv=env.get("NEAR_RTT_MP_BIPARTITE_EDGE_CONV") == "1",
        mp_bipartite_edge_attr_zero=env.get("NEAR_RTT_MP_BIPARTITE_EDGE_ATTR_ZERO") == "1",
        task_type_onehot_dim=4, partial_state_edge_dim=partial_state_feature_dim(contract),
        reencode_every=env.get("GNN_REENCODE_EVERY", "inf"), group_anchor=env.get("GNN_GROUP_ANCHOR") == "1",
    ).to(device)
    model.train()
    ns["_TRAIN_SEED"] = 1
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = graph.to(device) if hasattr(graph, "to") else graph
    k = int(env.get("NEAR_RTT_RANDOM_PREFIX_K", "0"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    loss, _ = loss_fn(model, data, device, full_context_weight=float(env.get("NEAR_RTT_FULL_CONTEXT_CE_WEIGHT", "0")), random_prefix_k=k)
    t_fwd = time.perf_counter() - t0
    opt.zero_grad()
    loss.backward()
    opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    out = {"yaml": args.yaml, "dataset_id": str(getattr(graph, "dataset_id", "?")), "n_tasks": int(graph.n_tasks),
           "n_labels": len(graph.tied_optimal_logit_plans["inf"]), "loss": float(loss.detach()), "forward_s": t_fwd, "step_s": wall,
           "device": str(device), "threads": torch.get_num_threads(), "rows_only": env.get("NEAR_RTT_SCORE_TASK_ROWS_ONLY") == "1", "peak_cuda_mb": (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None,
           "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
           "anchor_stats": ns["_ANCHOR_STATS"]["train"]}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
