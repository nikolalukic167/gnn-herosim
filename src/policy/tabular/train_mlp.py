"""
Train PointwiseEdgeMLP with grouped cross-entropy on Regime A batch_edges parquet.

Each graph_id group forms one softmax; y_logit is the oracle edge index within
the group.  Train/test split is by parent_dataset_id (same as train_ranker.py).

Usage:
  pipenv run python3 -m src.policy.tabular.train_mlp \\
    --input simulation_data/artifacts/tabular/batch_edges.parquet \\
    --output models/tabular/batch_edge_mlp.pt \\
    --test-size 0.2 --random-state 42 --epochs 100 --patience 10
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.policy.tabular.constants import FEATURE_COLUMN_NAMES, FEATURE_DIM
from src.policy.tabular.graph_extraction import validate_extracted_frame
from src.policy.tabular.mlp_model import PointwiseEdgeMLP
from src.policy.tabular.train_ranker import load_frame, split_by_parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PointwiseEdgeMLP with grouped CE.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-graphs", type=int, default=64,
                        help="Number of graphs per gradient step")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def build_graph_dataset(
    df: pd.DataFrame,
) -> List[Tuple[np.ndarray, int]]:
    """Return list of (X:[N,22], y:int) pairs, one per graph_id, sorted by graph_id."""
    feat_cols = FEATURE_COLUMN_NAMES
    samples = []
    for _, grp in df.groupby("graph_id", sort=True):
        X = grp[feat_cols].values.astype(np.float32)
        y = int(grp["y_logit"].iloc[0])
        n = X.shape[0]
        if y < 0 or y >= n:
            raise ValueError(
                f"y_logit={y} out of range for graph with {n} edges "
                f"(graph_id={grp['graph_id'].iloc[0]!r})"
            )
        samples.append((X, y))
    return samples


def edge_accuracy_from_dataset(
    model: PointwiseEdgeMLP,
    dataset: List[Tuple[np.ndarray, int]],
    device: torch.device,
) -> float:
    """Fraction of graphs where argmax score equals y_logit (oracle index)."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for X_np, y in dataset:
            x = torch.from_numpy(X_np).to(device)
            scores = model(x)
            if int(scores.argmax().item()) == y:
                correct += 1
    return correct / max(len(dataset), 1)


def grouped_ce_loss(
    model: PointwiseEdgeMLP,
    batch: List[Tuple[np.ndarray, int]],
    device: torch.device,
) -> torch.Tensor:
    """CE loss averaged over a mini-batch of graphs."""
    total_loss = torch.tensor(0.0, device=device)
    for X_np, y in batch:
        x = torch.from_numpy(X_np).to(device)
        logits = model(x)
        target = torch.tensor(y, dtype=torch.long, device=device)
        total_loss = total_loss + F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
    return total_loss / len(batch)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MLP train] device={device}", flush=True)

    df = load_frame(args.input)
    validate_extracted_frame(df)
    print(f"[MLP train] loaded {len(df):,} rows, {df['graph_id'].nunique():,} graphs", flush=True)

    train_df, test_df = split_by_parent(df, args.test_size, args.random_state)
    print(
        f"[MLP train] train {len(train_df):,} rows / {train_df['graph_id'].nunique():,} graphs  "
        f"| test {len(test_df):,} rows / {test_df['graph_id'].nunique():,} graphs",
        flush=True,
    )

    train_set = build_graph_dataset(train_df)
    test_set = build_graph_dataset(test_df)

    rng = random.Random(args.random_state)

    model = PointwiseEdgeMLP(hidden_dim=args.hidden_dim).to(device)
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
                print(f"[MLP train] early stop at epoch {epoch} (patience={args.patience})", flush=True)
                break

    if best_state is None:
        raise RuntimeError("[MLP train] No best state captured — training produced no improvement")

    model.load_state_dict(best_state)
    train_acc_final = edge_accuracy_from_dataset(model, train_set, device)
    val_acc_final = edge_accuracy_from_dataset(model, test_set, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "input_dim": FEATURE_DIM,
            "hidden_dim": args.hidden_dim,
        },
        str(args.output),
    )

    parquet_meta = {}
    meta_path_src = Path(str(args.input) + ".meta.json")
    if meta_path_src.exists():
        with open(meta_path_src) as f:
            parquet_meta = json.load(f)

    meta = {
        "input": str(args.input),
        "output": str(args.output),
        "train_edge_accuracy": float(train_acc_final),
        "val_edge_accuracy": float(val_acc_final),
        "best_val_edge_accuracy": float(best_val_acc),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_graphs": int(train_df["graph_id"].nunique()),
        "test_graphs": int(test_df["graph_id"].nunique()),
        "hidden_dim": args.hidden_dim,
        "input_dim": FEATURE_DIM,
        "epochs_run": len(history),
        "epochs_max": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "random_state": args.random_state,
        "test_size": args.test_size,
        "cache_provenance": parquet_meta,
    }
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[+] Saved model  -> {args.output}", flush=True)
    print(f"[+] Saved meta   -> {meta_path}", flush=True)
    print(
        f"    train_edge_acc={train_acc_final:.4f}  val_edge_acc={val_acc_final:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
