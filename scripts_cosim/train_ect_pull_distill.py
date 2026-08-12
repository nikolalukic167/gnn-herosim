#!/usr/bin/env python3
"""
Phase 3: fine-tune GNN student on ect_pull distill corpus (hard CE + soft KL).

L = (1-α) CE(y*, ŷ) + α τ² KL(q || softmax(z/τ))

Default: init from oracle_split dim16 CE checkpoint, train on harvested frames.

Usage:
    pipenv run python3 scripts_cosim/train_ect_pull_distill.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.policy.gnn.gnn_model import TaskPlacementGNN  # noqa: E402


DEFAULT_CACHE = (
    PROJECT_ROOT
    / "simulation_data/graphs_cache_regime_b_ect_pull_distill_oracle_split_v1"
)
DEFAULT_INIT = (
    PROJECT_ROOT
    / "models/near-rtt-v2-regime-b-oracle-split-cosim-dim16-ce-only.pt"
)
DEFAULT_OUT = (
    PROJECT_ROOT
    / "models/near-rtt-v2-regime-b-oracle-split-v1-ect-pull-distill.pt"
)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move(graph: Data, device: torch.device) -> Data:
    g = graph.clone()
    g.task_features = g.task_features.to(device)
    g.platform_features = g.platform_features.to(device)
    g.edge_index = g.edge_index.to(device)
    if getattr(g, "edge_attr", None) is not None and g.edge_attr.numel() > 0:
        g.edge_attr = g.edge_attr.to(device)
    if getattr(g, "node_edge_index", None) is not None and g.node_edge_index.numel() > 0:
        g.node_edge_index = g.node_edge_index.to(device)
    g.y = g.y.to(device)
    g.teacher_soft = g.teacher_soft.to(device)
    if getattr(g, "teacher_ect", None) is not None:
        g.teacher_ect = g.teacher_ect.to(device)
    return g


def distill_loss(
    logits_per_task: List[torch.Tensor],
    data: Data,
    *,
    alpha: float,
    tau: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if not logits_per_task or logits_per_task[0].numel() == 0:
        raise RuntimeError("FAIL LOUD: empty student logits")
    logits = logits_per_task[0]
    y = data.y.view(-1).long()
    if y.numel() != 1:
        raise RuntimeError(f"FAIL LOUD: expected single-task y, got shape {tuple(y.shape)}")
    target = int(y[0].item())
    if target < 0 or target >= logits.numel():
        raise RuntimeError(
            f"FAIL LOUD: y={target} out of range for n_logits={logits.numel()}"
        )

    soft = data.teacher_soft.float().view(-1)
    if soft.numel() != logits.numel():
        raise RuntimeError(
            f"FAIL LOUD: soft={soft.numel()} != logits={logits.numel()}"
        )

    ce = F.cross_entropy(logits.unsqueeze(0), y.view(1))
    tau_safe = max(float(tau), 1e-6)
    log_pred = F.log_softmax(logits / tau_safe, dim=0)
    # KL(q || p): sum q * (log q - log p)
    kl = F.kl_div(log_pred, soft, reduction="sum", log_target=False)
    loss = (1.0 - float(alpha)) * ce + float(alpha) * (tau_safe ** 2) * kl

    with torch.no_grad():
        pred = int(logits.argmax().item())
        stats = {
            "ce": float(ce.item()),
            "kl": float(kl.item()),
            "loss": float(loss.item()),
            "acc": float(pred == target),
            "n_logits": float(logits.numel()),
        }
    return loss, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.5, help="soft KL weight")
    parser.add_argument("--tau", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--from-scratch", action="store_true")
    args = parser.parse_args()

    _set_seed(args.seed)
    cache_dir = args.cache_dir.resolve()
    graphs_path = cache_dir / "graphs.pkl"
    if not graphs_path.is_file():
        raise FileNotFoundError(f"FAIL LOUD: missing distill cache {graphs_path}")

    with graphs_path.open("rb") as fh:
        graphs: List[Data] = pickle.load(fh)
    if not graphs:
        raise RuntimeError("FAIL LOUD: empty distill graphs.pkl")

    plat_dim = int(graphs[0].platform_features.size(-1))
    task_dim = int(graphs[0].task_features.size(-1))
    if plat_dim < 16 or task_dim != 3:
        raise RuntimeError(
            f"FAIL LOUD: expected dim24 (task=3,plat>=16); got task={task_dim} plat={plat_dim}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TaskPlacementGNN(
        task_feature_dim=task_dim,
        platform_feature_dim=plat_dim,
        embedding_dim=64,
        hidden_dim=64,
        num_layers=3,
        edge_dim=5,
    ).to(device)

    init_md5 = None
    if not args.from_scratch:
        init_path = args.init_checkpoint.resolve()
        if not init_path.is_file():
            raise FileNotFoundError(f"FAIL LOUD: init checkpoint missing: {init_path}")
        state = torch.load(init_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        init_md5 = _md5(init_path)
        print(f"Init from {init_path} md5={init_md5[:12]}…")
    else:
        print("Training from scratch (no CE init)")

    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    print(
        f"Distill train: n={len(graphs)} epochs={args.epochs} α={args.alpha} "
        f"τ={args.tau} device={device}"
    )

    history: List[Dict[str, float]] = []
    best_loss = float("inf")
    best_state = None
    for epoch in range(int(args.epochs)):
        model.train()
        order = list(range(len(graphs)))
        random.shuffle(order)
        running = {"ce": 0.0, "kl": 0.0, "loss": 0.0, "acc": 0.0}
        for idx in order:
            g = _move(graphs[idx], device)
            opt.zero_grad()
            logits = model(g)
            loss, stats = distill_loss(
                logits, g, alpha=float(args.alpha), tau=float(args.tau)
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"FAIL LOUD: non-finite distill loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k in running:
                running[k] += stats[k]
        n = float(len(graphs))
        row = {k: running[k] / n for k in running}
        row["epoch"] = float(epoch)
        history.append(row)
        if row["loss"] < best_loss:
            best_loss = row["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0 or epoch == int(args.epochs) - 1:
            print(
                f"Epoch {epoch:3d}/{args.epochs}  loss={row['loss']:.4f}  "
                f"ce={row['ce']:.4f}  kl={row['kl']:.4f}  acc={row['acc']*100:.1f}%"
            )

    if best_state is None:
        raise RuntimeError("FAIL LOUD: no best state saved")
    if history[-1]["acc"] < 0.99:
        # With 12 frames we expect near-perfect teacher imitation after fine-tune.
        print(
            f"WARN: final train acc={history[-1]['acc']*100:.1f}% < 99% — "
            "student may not have fit the teacher trajectory"
        )

    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out)
    meta = {
        "md5": _md5(out),
        "path": out.as_posix(),
        "platform_feature_dim": plat_dim,
        "task_feature_dim": task_dim,
        "train_objective": "ect_pull_distill",
        "alpha": float(args.alpha),
        "tau": float(args.tau),
        "epochs": int(args.epochs),
        "n_graphs": len(graphs),
        "cache_dir": str(cache_dir),
        "init_checkpoint": None if args.from_scratch else str(args.init_checkpoint.resolve()),
        "init_md5": init_md5,
        "best_loss": best_loss,
        "final_acc": history[-1]["acc"],
        "phase": "phase3",
    }
    out.with_suffix(out.suffix + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out.parent / (out.stem + ".train_history.json")).write_text(
        json.dumps(history, indent=2) + "\n"
    )
    print(f"Wrote {out} md5={meta['md5']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
