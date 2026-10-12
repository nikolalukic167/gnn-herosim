#!/usr/bin/env python3
"""plan_rerank_v1 Phase 1 trainer: the plan selector (src/policy/plan_rerank/selector.py) on the branch-pool tables.

Run through run_experiment.py (trainer: plan_selector), never directly in a gate. Trains on the TRAIN pool (training topologies),
selects the epoch on the VAL pool only (pooled regret of the selected plan against the sweep optimum, ties to the earlier epoch),
never touches a test id: every train pool table must be in the split's train list, every val table in its val list, none in test,
and the split's train and val topologies must be disjoint.

Loss: soft listwise cross-entropy (selector.listwise_loss, tau fixed by --tau-rel before training), pools weighted by their capped
relative RTT spread (selector.pool_weight); a pool whose candidates all share one RTT has weight 0 and is skipped.
Message passing off (the twin): GNN_DISABLE_MESSAGE_PASSING=1 in the config's env, recorded in the sidecar.

Writes <out-dir>/<run name>.pt (the selected epoch) and its .contract.json, and <run name>-final.pt with its own sidecar. Logs every
epoch to Weights & Biases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from src.policy.plan_rerank import selector as S  # noqa: E402

SEED = int(os.environ.get("PRSEL_TRAIN_SEED", "1"))


def _seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _flag(name: str) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw not in ("", "0", "1", "true", "false", "yes", "no"):
        raise ValueError(f"FAIL LOUD: {name}={raw!r}")
    return raw in ("1", "true", "yes")


def load_rows(path: Path, limit: int) -> List[Dict[str, Any]]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows[:limit] if limit > 0 else rows


def check_split(train_rows, val_rows, split: Dict[str, Any]) -> None:
    tr, va, te = set(split["train"]), set(split["val"]), set(split["test"])
    for name, rows, allowed in (("train", train_rows, tr), ("val", val_rows, va)):
        bad = [r["ds"] for r in rows if r["ds"] not in allowed]
        if bad:
            raise SystemExit(f"FAIL LOUD: {len(bad)} {name} pool tables are not in the split's {name} list, e.g. {bad[:3]}")
        leak = [r["ds"] for r in rows if r["ds"] in te]
        if leak:
            raise SystemExit(f"FAIL LOUD: {len(leak)} {name} pool tables are test ids, e.g. {leak[:3]}")
    topo = split.get("topologies") or {}
    if topo:
        both = set(map(str, topo.get("train") or [])) & set(map(str, topo.get("val") or []))
        if both:
            raise SystemExit(f"FAIL LOUD: the split's train and val topologies overlap: {sorted(both)[:5]}")


def attach_graphs(rows, graph_of: Dict[str, Any]) -> List[Any]:
    out = []
    for r in rows:
        g = graph_of.get(r["ds"])
        if g is None:
            raise SystemExit(f"FAIL LOUD: {r['ds']} has no graph in the cache")
        n = int(g.n_tasks)
        if n != int(r["n_tasks"]):
            raise SystemExit(f"FAIL LOUD: {r['ds']}: graph n_tasks {n} != pool n_tasks {r['n_tasks']}")
        tl = [[S.plan_key([c])[0] for c in g.task_logit_to_placement[t]] for t in range(n)]
        for c in r["candidates"]:
            p = S.plan_key(c["plan"])
            if any(p[t] not in tl[t] for t in range(n)):
                raise SystemExit(f"FAIL LOUD: {r['ds']}: candidate {p} is outside the graph's candidate slate")
        out.append(g)
    return out


def evaluate(model: S.PlanSelector, rows, graphs) -> Dict[str, float]:
    model.eval()
    res: Dict[str, Dict[str, list]] = {}
    with torch.no_grad():
        for r, g in zip(rows, graphs):
            cost, plans, cands = model.costs(g, r)
            k = S.select([float(c) for c in cost], plans, cands)
            rtt = float(cands[k]["rtt"])
            amin = min((c for c in cands if "argmin_s" in c["sources"]), key=lambda c: float(c["rtt"]))
            for rung in ["all"] + ([str(r["rung"])] if r.get("rung") else []):
                d = res.setdefault(rung, {"ex": [], "opt": [], "ex_s": []})
                d["ex"].append(rtt - float(r["opt"]))
                d["ex_s"].append(float(amin["rtt"]) - float(r["opt"]))
                d["opt"].append(float(r["opt"]))
    out: Dict[str, float] = {}
    for rung, d in res.items():
        so = math.fsum(d["opt"])
        ex = sorted(d["ex"], reverse=True)
        out[f"{rung}/pooled"] = math.fsum(ex) / so
        out[f"{rung}/trim3"] = math.fsum(ex[3:]) / so
        out[f"{rung}/hit_opt"] = sum(e <= 1e-9 for e in ex) / len(ex)
        out[f"{rung}/argmin_s_pooled"] = math.fsum(d["ex_s"]) / so
        out[f"{rung}/n"] = float(len(ex))
    return out


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--pool-train", required=True)
    ap.add_argument("--pool-val", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--base-checkpoint", required=True, help="the base GNN whose decode the pool's branches come from")
    ap.add_argument("--variant", required=True, choices=S.VARIANTS)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--embedding-dim", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--tau-rel", type=float, default=0.05)
    ap.add_argument("--weight-cap", type=float, default=1.0)
    ap.add_argument("--pools-per-step", type=int, default=16)
    ap.add_argument("--max-train-pools", type=int, default=0, help="smoke only: the first N train pools")
    ap.add_argument("--max-val-pools", type=int, default=0, help="smoke only: the first N val pools")
    ap.add_argument("--smoke", action="store_true", help="mark the checkpoint smoke: the live loader refuses it without the smoke flag")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--wandb-project", default="gnn-plan-rerank-v1")
    a = ap.parse_args()
    _seed_everything()
    if (a.max_train_pools or a.max_val_pools) and not a.smoke:
        raise SystemExit("FAIL LOUD: a pool cap is a smoke setting; pass --smoke")
    run_name = os.environ.get("WANDB_RUN_NAME", "").strip()
    if not run_name:
        raise SystemExit("FAIL LOUD: WANDB_RUN_NAME is the checkpoint name; run through run_experiment.py")
    mp_off = _flag("GNN_DISABLE_MESSAGE_PASSING")

    import wandb

    cache = Path(a.cache_dir)
    split = json.load(open(a.split))
    train_rows = load_rows(Path(a.pool_train), a.max_train_pools)
    val_rows = load_rows(Path(a.pool_val), a.max_val_pools)
    check_split(train_rows, val_rows, split)
    with open(cache / "graphs.pkl", "rb") as fh:
        graphs = pickle.load(fh)
    with open(cache / "dataset_ids.pkl", "rb") as fh:
        ids = pickle.load(fh)
    if len(graphs) != len(ids):
        raise SystemExit(f"FAIL LOUD: {len(graphs)} graphs, {len(ids)} dataset ids")
    want = {r["ds"] for r in train_rows} | {r["ds"] for r in val_rows}
    graph_of = {str(i): g for i, g in zip(ids, graphs) if str(i) in want}
    del graphs
    train_g = attach_graphs(train_rows, graph_of)
    val_g = attach_graphs(val_rows, graph_of)
    meta = json.loads((cache / "metadata.json").read_text())
    g0 = train_g[0]
    onehot = getattr(g0, "task_type_onehot4", None)
    from src.policy.gnn.plan_raw import PLAN_RAW_DIM

    contract: Dict[str, Any] = {
        "kind": S.KIND, "lineage": S.LINEAGE, "variant": a.variant, "disable_message_passing": mp_off,
        "feature_names": list(S.PLAN_FEATURES), "tau_rel": a.tau_rel, "weight_cap": a.weight_cap, "sigma_floor": S.SIGMA_FLOOR,
        "gnn": {"task_feature_dim": int(g0.task_features.shape[1]), "platform_feature_dim": int(g0.platform_features.shape[1]),
                "edge_dim": int(g0.edge_attr.shape[1]), "embedding_dim": a.embedding_dim, "hidden_dim": a.hidden_dim,
                "num_layers": a.num_layers, "dropout": a.dropout, "normalize_platform_inputs": meta.get("inference_feature_layout") == "atomic21",
                "task_type_onehot_dim": int(onehot.shape[1]) if onehot is not None else 0, "mp_peer_edges": True,
                "mp_bipartite_edge_attr_zero": True, "plan_raw_sum": True, "partial_state_edge_dim": PLAN_RAW_DIM},
        "dag_task_type_vocab": meta.get("dag_task_type_vocab"), "candidate_slate": meta.get("candidate_slate"),
        "partial_state_contract_of_cache": meta.get("partial_state_contract"), "physics_env": meta.get("physics_env"),
        "base_checkpoint": Path(a.base_checkpoint).name, "base_checkpoint_md5": S.md5_of(Path(a.base_checkpoint)),
        "cache_dir": str(cache), "cache_metadata_sha256": sha256_of(cache / "metadata.json"),
        "pool_train": {"path": str(a.pool_train), "sha256": sha256_of(Path(a.pool_train)), "n": len(train_rows)},
        "pool_val": {"path": str(a.pool_val), "sha256": sha256_of(Path(a.pool_val)), "n": len(val_rows)},
        "split": {"path": str(a.split), "sha256": sha256_of(Path(a.split))},
        "train_seed": SEED, "deterministic_algorithms": True, "smoke": bool(a.smoke),
        "learning_rate": a.learning_rate, "weight_decay": a.weight_decay, "pools_per_step": a.pools_per_step, "epochs": a.epochs,
        "experiment_config": os.environ.get("HEROSIM_EXPERIMENT_CONFIG"),
    }
    if contract["gnn"]["task_type_onehot_dim"] != 4 or not contract["dag_task_type_vocab"]:
        raise SystemExit("FAIL LOUD: the selector needs the 4-way task-type one-hot and the cache's vocab (mp_peer_edges)")
    model = S.build_selector(contract)
    torch.manual_seed(SEED)
    model.apply(init_weights)
    opt = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)
    weights = [S.pool_weight([float(c["rtt"]) for c in r["candidates"]], a.weight_cap) if len(r["candidates"]) > 1 else 0.0
               for r in train_rows]
    live = [i for i, w in enumerate(weights) if w > 0]
    if not live:
        raise SystemExit("FAIL LOUD: no train pool has an RTT spread")
    wandb.init(project=a.wandb_project, name=run_name, tags=[t for t in os.environ.get("WANDB_TAGS", "").split(",") if t],
               config={**{k: v for k, v in contract.items() if k != "gnn"}, **{f"gnn_{k}": v for k, v in contract["gnn"].items()},
                       "train_pools_weighted": len(live)})
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"{run_name}.pt"
    rng = random.Random(SEED)
    best, best_epoch = None, None
    t_start = time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        order = list(live)
        rng.shuffle(order)
        total, wsum = 0.0, 0.0
        for k in range(0, len(order), a.pools_per_step):
            chunk = order[k:k + a.pools_per_step]
            opt.zero_grad()
            loss = torch.zeros(())
            w_chunk = math.fsum(weights[i] for i in chunk)
            for i in chunk:
                cost, _plans, cands = model.costs(train_g[i], train_rows[i])
                loss = loss + weights[i] * S.listwise_loss(cost, [float(c["rtt"]) for c in cands], a.tau_rel)
            loss = loss / w_chunk
            loss.backward()
            opt.step()
            total += float(loss) * w_chunk
            wsum += w_chunk
        val = evaluate(model, val_rows, val_g)
        log = {"epoch": epoch, "train/loss": total / wsum, **{f"val/{k}": v for k, v in val.items()},
               "wall_s": time.time() - t_start}
        wandb.log(log, step=epoch)
        print(json.dumps(log), flush=True)
        if best is None or val["all/pooled"] < best - 1e-12:
            best, best_epoch = val["all/pooled"], epoch
            torch.save(model.state_dict(), best_path)
            (best_path.with_suffix(".contract.json")).write_text(json.dumps(
                {**contract, "selected_epoch": epoch, "selected_on": "val pooled regret", "val": val}, indent=1))
    final = out_dir / f"{run_name}-final.pt"
    torch.save(model.state_dict(), final)
    final.with_suffix(".contract.json").write_text(json.dumps({**contract, "selected_epoch": a.epochs, "selected_on": "final", "val": val},
                                                              indent=1))
    wandb.log({"selected_epoch": best_epoch, "selected_val_pooled": best})
    wandb.finish()
    print(f"selected epoch {best_epoch} (val pooled regret {best:.6f}) -> {best_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
