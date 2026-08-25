"""
Train PointwiseEdgeMLP with dim22 features (3-task + 14-plat + 5-edge = 22-d)
directly from a sequential graph cache.

Fix vs train_mlp_ce_reduced.py:
  - Keeps all 14 platform features → includes is_cold/shared_fate (dim 8),
    has_dnn1/dnn2 (dims 5-6), temporal dims (9-11), target_concurrency (12)
  - Keeps all 5 edge features    → includes is_warm (dim 2), critical for
    node_disk_v2 physics where warm = skip 31s pull
  - Sets inference_feature_layout='dim22' so the scheduler builds the matching
    22-d feature bundle at serve time (use_dim22=True path in feature_builder.py)

The residual train/serve mismatch at platform dim-7 (raw queue in training vs
p90-normalized at inference) is the same one present in the working batch_edge_mlp.pt
(dim22, v6.5 cache) and is tolerable because 21 other features provide signal.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_repo = _Path(__file__).resolve().parents[3]
if str(_repo) not in _sys.path:
    _sys.path.insert(0, str(_repo))

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from src.policy.tabular.mlp_model import PointwiseEdgeMLP
from src.policy.tabular.reduced_features import (
    DIM22_FEATURE_COLUMN_NAMES,
    DIM22_FEATURE_DIM,
    apply_reduced_features_to_graph,
    dim22_rows_to_dataframe,
    enrich_task_features_with_src_norm,
    extract_rows_dim22_from_graph,
    should_emit_graph,
    validate_dim22_frame,
)
from src.policy.tabular.train_ranker import split_by_parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train dim22 PointwiseEdgeMLP from a sequential graph cache."
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-graphs", type=int, default=64)
    parser.add_argument("--regime", choices=("batch", "single"), default="batch")
    return parser.parse_args()


def load_sequential_cache(cache_dir: Path):
    meta_path = cache_dir / "metadata.json"
    graphs_path = cache_dir / "graphs.pkl"
    ids_path = cache_dir / "dataset_ids.pkl"
    for path in (meta_path, graphs_path, ids_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    if not metadata.get("sequential_counterfactual") and not metadata.get("single_task"):
        raise ValueError(f"Cache at {cache_dir} is not a sequential tabular source")
    with open(graphs_path, "rb") as f:
        graphs = pickle.load(f)
    with open(ids_path, "rb") as f:
        dataset_ids = pickle.load(f)
    if len(graphs) != len(dataset_ids):
        raise ValueError(f"graphs ({len(graphs)}) != dataset_ids ({len(dataset_ids)})")
    return metadata, graphs, dataset_ids


def build_graph_dataset(df: pd.DataFrame) -> List[Tuple[np.ndarray, int]]:
    samples = []
    for _, grp in df.groupby("graph_id", sort=True):
        X = grp[DIM22_FEATURE_COLUMN_NAMES].values.astype(np.float32)
        y = int(grp["y_logit"].iloc[0])
        if y < 0 or y >= X.shape[0]:
            raise ValueError(f"y_logit={y} out of range for graph with {X.shape[0]} edges")
        samples.append((X, y))
    return samples


def edge_accuracy_from_dataset(model, dataset, device) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for X_np, y in dataset:
            x = torch.from_numpy(X_np).to(device)
            scores = model(x)
            if int(scores.argmax().item()) == y:
                correct += 1
    return correct / max(len(dataset), 1)


def grouped_ce_loss(model, batch, device) -> torch.Tensor:
    total_loss = torch.tensor(0.0, device=device)
    for X_np, y in batch:
        x = torch.from_numpy(X_np).to(device)
        logits = model(x)
        target = torch.tensor(y, dtype=torch.long, device=device)
        total_loss = total_loss + F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
    return total_loss / len(batch)


def extract_dim22_dataframe(args: argparse.Namespace, metadata, graphs, dataset_ids) -> pd.DataFrame:
    repo_root = (args.project_root or Path(__file__).resolve().parents[3]).resolve()
    all_rows = []
    emitted = 0
    for graph, graph_id in tqdm(
        zip(graphs, dataset_ids), total=len(graphs), desc="extract-dim22"
    ):
        if not should_emit_graph(graph, regime=args.regime, exclude_prefix_augment=True):
            continue
        # Only enrich src_norm — do NOT slice platform or edge features
        enrich_task_features_with_src_norm(graph, repo_root)
        rows, skip_reason = extract_rows_dim22_from_graph(graph, graph_id)
        if skip_reason or not rows:
            continue
        all_rows.extend(rows)
        emitted += 1

    if not all_rows:
        raise RuntimeError("No dim22 rows extracted from cache")

    df = dim22_rows_to_dataframe(all_rows)
    stats = validate_dim22_frame(df)
    print(
        f"[MLP dim22] extracted {stats['num_rows']:,} rows / {stats['num_graphs']:,} graphs "
        f"from {emitted:,} seq graphs (cache={metadata.get('version')})",
        flush=True,
    )
    return df


def main() -> None:
    args = parse_args()
    # --random-state seeds the parent split and the batch order; without these two lines
    # the model's weight INIT comes from torch's unseeded global RNG, so two identical
    # invocations produce different weights. That is what made every MLP checkpoint
    # before 2026-08-24 an unreproducible draw (see train_mlp_dim22_from_batch.py, which
    # was fixed first). Checkpoints record `torch_seeded` so a seeded one can be told
    # from a drawn one.
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = args.cache_dir.resolve()

    print(
        f"[MLP dim22] device={device} input_dim={DIM22_FEATURE_DIM} "
        f"(3-task + 14-plat + 5-edge, includes is_warm + is_cold)",
        flush=True,
    )

    metadata, graphs, dataset_ids = load_sequential_cache(cache_dir)
    df = extract_dim22_dataframe(args, metadata, graphs, dataset_ids)

    train_df, test_df = split_by_parent(df, args.test_size, args.random_state)
    print(
        f"[MLP dim22] train {len(train_df):,} rows / {train_df['graph_id'].nunique():,} graphs  "
        f"| test {len(test_df):,} rows / {test_df['graph_id'].nunique():,} graphs",
        flush=True,
    )

    train_set = build_graph_dataset(train_df)
    test_set = build_graph_dataset(test_df)
    rng = random.Random(args.random_state)

    model = PointwiseEdgeMLP(input_dim=DIM22_FEATURE_DIM, hidden_dim=args.hidden_dim).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[MLP dim22] params={param_count:,}", flush=True)

    optimizer = Adam(model.parameters(), lr=args.lr)
    best_val_acc = -1.0
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_set)
        epoch_loss = 0.0
        n_batches = 0
        i = 0
        while i < len(train_set):
            batch = train_set[i : i + args.batch_graphs]
            optimizer.zero_grad()
            loss = grouped_ce_loss(model, batch, device)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            i += args.batch_graphs

        train_acc = edge_accuracy_from_dataset(model, train_set, device)
        val_acc = edge_accuracy_from_dataset(model, test_set, device)
        avg_loss = epoch_loss / max(n_batches, 1)
        history.append({"epoch": epoch, "loss": avg_loss, "train_acc": train_acc, "val_acc": val_acc})
        print(
            f"  epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  "
            f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}",
            flush=True,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[MLP dim22] early stop at epoch {epoch} (patience={args.patience})", flush=True)
                break

    if best_state is None:
        raise RuntimeError("[MLP dim22] No best state captured — training produced no improvement")

    model.load_state_dict(best_state)
    train_acc_final = edge_accuracy_from_dataset(model, train_set, device)
    val_acc_final = edge_accuracy_from_dataset(model, test_set, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "input_dim": DIM22_FEATURE_DIM,
            "hidden_dim": args.hidden_dim,
            "inference_feature_layout": "dim22",
        },
        str(args.output),
    )

    meta = {
        "cache_dir": str(cache_dir),
        "cache_version": metadata.get("version"),
        "output": str(args.output),
        "train_edge_accuracy": float(train_acc_final),
        "val_edge_accuracy": float(val_acc_final),
        "best_val_edge_accuracy": float(best_val_acc),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_graphs": int(train_df["graph_id"].nunique()),
        "test_graphs": int(test_df["graph_id"].nunique()),
        "hidden_dim": args.hidden_dim,
        "input_dim": DIM22_FEATURE_DIM,
        "inference_feature_layout": "dim22",
        "epochs_run": len(history),
        "epochs_max": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "random_state": args.random_state,
        "torch_seeded": True,
        "test_size": args.test_size,
        "param_count": param_count,
        "fix_note": (
            "dim22 layout: keeps is_warm (edge dim 2), is_cold (plat dim 8), "
            "has_dnn1/2 (plat dims 5-6), temporal dims. "
            "Fixes ce_reduced which dropped all three, causing catastrophic hub failure."
        ),
    }
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[+] Saved model  -> {args.output}", flush=True)
    print(f"[+] Saved meta   -> {meta_path}", flush=True)
    print(
        f"    train_edge_acc={train_acc_final:.4f}  val_edge_acc={val_acc_final:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
