"""
Train PointwiseEdgeMLP with dim22 features from a batch PyG graph cache
(prepare_graphs_cache.py — same cache as GNN wssm).

Platform encoding matches dim22 inference: normalized queue, shared_fate, usage_ratio.
One training sample per task per batch graph (4 decisions × 940 graphs ≈ 3760 graphs).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_repo = _Path(__file__).resolve().parents[3]
if str(_repo) not in _sys.path:
    _sys.path.insert(0, str(_repo))

import argparse
import json
import os
import pickle
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from src.placement.queue_features import (
    DEFAULT_QUEUE_FEATURE_CONTRACT,
    validate_queue_feature_contract,
)
from src.policy.tabular.mlp_model import PointwiseEdgeMLP
from src.policy.tabular.reduced_features import (
    DIM22_FEATURE_DIM,
    DIM24_FEATURE_DIM,
    dim22_rows_to_dataframe,
    extract_rows_dim22_from_batch_graph,
    validate_dim22_frame,
)
from src.policy.tabular.train_ranker import split_by_parent_three_way


def _feature_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if str(c).startswith("x_")]
    if not cols:
        raise ValueError("No feature columns (x_*) in extracted dataframe")
    return sorted(cols, key=lambda c: int(str(c).split("_", 1)[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train dim22/dim24 PointwiseEdgeMLP from a batch graph cache."
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-size", type=float, default=0.15,
                        help="Canonical-parent validation fraction")
    parser.add_argument("--test-size", type=float, default=0.15,
                        help="Canonical-parent held-out test fraction")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-graphs", type=int, default=64)
    parser.add_argument("--min-batch-tasks", type=int, default=2,
                        help="Skip batch graphs with fewer tasks (GNN/MLP deploy range)")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    return parser.parse_args()


def load_batch_cache(cache_dir: Path):
    meta_path = cache_dir / "metadata.json"
    graphs_path = cache_dir / "graphs.pkl"
    ids_path = cache_dir / "dataset_ids.pkl"
    for path in (meta_path, graphs_path, ids_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    if metadata.get("sequential_counterfactual"):
        raise ValueError(
            f"Cache at {cache_dir} is sequential — use train_mlp_dim22_from_seq.py instead"
        )
    with open(graphs_path, "rb") as f:
        graphs = pickle.load(f)
    with open(ids_path, "rb") as f:
        dataset_ids = pickle.load(f)
    if len(graphs) != len(dataset_ids):
        raise ValueError(f"graphs ({len(graphs)}) != dataset_ids ({len(dataset_ids)})")
    return metadata, graphs, dataset_ids


def build_graph_dataset(df: pd.DataFrame) -> List[Tuple[np.ndarray, int]]:
    feature_cols = _feature_columns(df)
    samples = []
    for _, grp in df.groupby("graph_id", sort=True):
        X = grp[feature_cols].values.astype(np.float32)
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
    all_rows = []
    emitted = 0
    skipped_small = 0
    for graph, graph_id in tqdm(
        zip(graphs, dataset_ids), total=len(graphs), desc="extract-batch-edges"
    ):
        n_tasks = int(getattr(graph, "n_tasks", 0))
        if n_tasks < args.min_batch_tasks:
            skipped_small += 1
            continue
        rows, skip_reason = extract_rows_dim22_from_batch_graph(graph, str(graph_id))
        if skip_reason or not rows:
            raise RuntimeError(f"Failed to extract {graph_id}: {skip_reason}")
        all_rows.extend(rows)
        emitted += 1

    if not all_rows:
        raise RuntimeError("No dim22/dim24 rows extracted from batch cache")

    df = dim22_rows_to_dataframe(all_rows)
    stats = validate_dim22_frame(df)
    print(
        f"[MLP batch] extracted {stats['num_rows']:,} rows / {stats['num_graphs']:,} decision graphs "
        f"from {emitted:,} batch graphs (skipped {skipped_small} with n_tasks<{args.min_batch_tasks}; "
        f"feature_dim={stats['feature_dim']} layout={stats['inference_feature_layout']} "
        f"cache={metadata.get('version')})",
        flush=True,
    )
    return df


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = args.cache_dir.resolve()

    metadata, graphs, dataset_ids = load_batch_cache(cache_dir)
    df = extract_dim22_dataframe(args, metadata, graphs, dataset_ids)
    feature_cols = _feature_columns(df)
    input_dim = len(feature_cols)
    if input_dim not in (DIM22_FEATURE_DIM, DIM24_FEATURE_DIM):
        raise RuntimeError(
            f"[MLP batch] Unexpected input_dim={input_dim}; "
            f"expected {DIM22_FEATURE_DIM} or {DIM24_FEATURE_DIM}"
        )
    layout = "dim24" if input_dim == DIM24_FEATURE_DIM else "dim22"
    # Caches built before CACHE_VERSION 5.7 carry no contract field and are legacy_v0 by
    # construction; the checkpoint records it so inference cannot serve the wrong scaling.
    queue_feature_contract = validate_queue_feature_contract(
        metadata.get("queue_feature_contract") or DEFAULT_QUEUE_FEATURE_CONTRACT
    )
    print(
        f"[MLP batch] device={device} input_dim={input_dim} layout={layout} "
        f"queue_feature_contract={queue_feature_contract} "
        f"(batch cache — norm queue + shared_fate"
        f"{' + pull observables' if layout == 'dim24' else ''}, matches GNN + inference)",
        flush=True,
    )

    train_df, val_df, test_df = split_by_parent_three_way(
        df,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(
        f"[MLP batch] train {len(train_df):,} rows / {train_df['graph_id'].nunique():,} graphs / "
        f"{train_df['parent_dataset_id'].nunique():,} parents  | "
        f"val {len(val_df):,} rows / {val_df['graph_id'].nunique():,} graphs / "
        f"{val_df['parent_dataset_id'].nunique():,} parents  | "
        f"test {len(test_df):,} rows / {test_df['graph_id'].nunique():,} graphs / "
        f"{test_df['parent_dataset_id'].nunique():,} parents",
        flush=True,
    )

    train_set = build_graph_dataset(train_df)
    val_set = build_graph_dataset(val_df)
    test_set = build_graph_dataset(test_df)
    rng = random.Random(args.random_state)

    model = PointwiseEdgeMLP(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[MLP batch] params={param_count:,}", flush=True)

    optimizer = Adam(model.parameters(), lr=args.lr)
    best_val_acc = -1.0
    best_state = None
    no_improve = 0
    history = []

    wandb_run = None
    if args.wandb_project:
        import wandb

        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or "nikolalukic167-tu-wien",
            name=args.wandb_run_name,
            config={
                "model": "PointwiseEdgeMLP",
                "input_dim": input_dim,
                "inference_feature_layout": layout,
                "queue_feature_contract": queue_feature_contract,
                "hidden_dim": args.hidden_dim,
                "cache_dir": str(cache_dir),
                "cache_version": metadata.get("version"),
                "epochs": args.epochs,
                "patience": args.patience,
                "lr": args.lr,
                "val_size": args.val_size,
                "test_size": args.test_size,
                "train_graphs": int(train_df["graph_id"].nunique()),
                "val_graphs": int(val_df["graph_id"].nunique()),
                "test_graphs": int(test_df["graph_id"].nunique()),
                "train_parents": int(train_df["parent_dataset_id"].nunique()),
                "val_parents": int(val_df["parent_dataset_id"].nunique()),
                "test_parents": int(test_df["parent_dataset_id"].nunique()),
            },
            tags=[
                t
                for t in os.environ.get(
                    "WANDB_TAGS", f"mlp,{layout},batchcache"
                ).split(",")
                if t
            ],
        )

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
        val_acc = edge_accuracy_from_dataset(model, val_set, device)
        avg_loss = epoch_loss / max(n_batches, 1)
        history.append({"epoch": epoch, "loss": avg_loss, "train_acc": train_acc, "val_acc": val_acc})
        if wandb_run is not None:
            import wandb

            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": avg_loss,
                    "train/edge_acc": train_acc,
                    "val/edge_acc": val_acc,
                }
            )
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
                print(f"[MLP batch] early stop at epoch {epoch} (patience={args.patience})", flush=True)
                break

    if best_state is None:
        raise RuntimeError("[MLP batch] No best state captured")

    model.load_state_dict(best_state)
    train_acc_final = edge_accuracy_from_dataset(model, train_set, device)
    val_acc_final = edge_accuracy_from_dataset(model, val_set, device)
    test_acc_final = edge_accuracy_from_dataset(model, test_set, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "inference_feature_layout": layout,
            "queue_feature_contract": queue_feature_contract,
        },
        str(args.output),
    )

    meta = {
        "cache_dir": str(cache_dir),
        "cache_version": metadata.get("version"),
        "cache_type": "batch",
        "output": str(args.output),
        "train_edge_accuracy": float(train_acc_final),
        "val_edge_accuracy": float(val_acc_final),
        "test_edge_accuracy": float(test_acc_final),
        "best_val_edge_accuracy": float(best_val_acc),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_graphs": int(train_df["graph_id"].nunique()),
        "val_graphs": int(val_df["graph_id"].nunique()),
        "test_graphs": int(test_df["graph_id"].nunique()),
        "train_parents": int(train_df["parent_dataset_id"].nunique()),
        "val_parents": int(val_df["parent_dataset_id"].nunique()),
        "test_parents": int(test_df["parent_dataset_id"].nunique()),
        "hidden_dim": args.hidden_dim,
        "input_dim": input_dim,
        "inference_feature_layout": layout,
        "queue_feature_contract": queue_feature_contract,
        "epochs_run": len(history),
        "epochs_max": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "random_state": args.random_state,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "param_count": param_count,
        "split_note": "Canonical-parent 70/15/15; early-stop on val; test reported once at end.",
        "fix_note": (
            f"Trained from batch cache (same as GNN) — platform features match {layout} inference."
        ),
    }
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[+] Saved model  -> {args.output}", flush=True)
    print(f"[+] Saved meta   -> {meta_path}", flush=True)
    print(
        f"    train_edge_acc={train_acc_final:.4f}  val_edge_acc={val_acc_final:.4f}  "
        f"test_edge_acc={test_acc_final:.4f}",
        flush=True,
    )
    if wandb_run is not None:
        import wandb

        wandb.summary["best_val_edge_acc"] = float(best_val_acc)
        wandb.summary["final_train_edge_acc"] = float(train_acc_final)
        wandb.summary["final_val_edge_acc"] = float(val_acc_final)
        wandb.summary["final_test_edge_acc"] = float(test_acc_final)
        wandb.finish()


if __name__ == "__main__":
    main()
