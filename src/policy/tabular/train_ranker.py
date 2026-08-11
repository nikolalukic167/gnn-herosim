#!/usr/bin/env python3
"""
Train a grouped XGBoost edge ranker for Regime A (batch) tabular placement.

Groups rows by graph_id so training aligns with per-decision argmax decode.

Usage:
  pipenv run python3 -m src.policy.tabular.train_ranker \\
    --input simulation_data/artifacts/tabular/batch_edges.parquet \\
    --output models/tabular/batch_edge_ranker.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from src.policy.tabular.constants import FEATURE_COLUMN_NAMES
from src.policy.tabular.graph_extraction import validate_extracted_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train grouped XGBoost edge ranker.")
    parser.add_argument("--input", type=Path, required=True, help="Parquet/CSV from prepare_tabular_dataset.py")
    parser.add_argument("--output", type=Path, required=True, help="Output model path (.json)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction (by parent_dataset_id)")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--num-boost-round", type=int, default=200)
    parser.add_argument("--early-stopping-rounds", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    return parser.parse_args()


def load_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {suffix}")


def split_by_parent(
    df: pd.DataFrame, test_size: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unique_parents = df["parent_dataset_id"].unique()
    if len(unique_parents) < 2:
        raise ValueError("Need at least two parent_dataset_id groups for train/test split")
    train_parents, test_parents = train_test_split(
        unique_parents,
        test_size=test_size,
        random_state=random_state,
    )
    train_parents = set(train_parents)
    test_parents = set(test_parents)
    train_df = df[df["parent_dataset_id"].isin(train_parents)].copy()
    test_df = df[df["parent_dataset_id"].isin(test_parents)].copy()
    return train_df, test_df


def split_by_parent_three_way(
    df: pd.DataFrame,
    *,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Canonical-parent 70/15/15 split. Val is for selection; test is held out."""
    unique_parents = df["parent_dataset_id"].unique()
    if len(unique_parents) < 3:
        raise ValueError(
            f"Need at least three parent_dataset_id groups for train/val/test; got {len(unique_parents)}"
        )
    holdout = val_size + test_size
    if not (0.0 < holdout < 1.0):
        raise ValueError(f"val_size+test_size must be in (0,1); got {holdout}")
    train_parents, temp_parents = train_test_split(
        unique_parents,
        test_size=holdout,
        random_state=random_state,
    )
    relative_test = test_size / holdout
    val_parents, test_parents = train_test_split(
        temp_parents,
        test_size=relative_test,
        random_state=random_state,
    )
    train_df = df[df["parent_dataset_id"].isin(set(train_parents))].copy()
    val_df = df[df["parent_dataset_id"].isin(set(val_parents))].copy()
    test_df = df[df["parent_dataset_id"].isin(set(test_parents))].copy()
    train_p = set(train_df["parent_dataset_id"].unique())
    val_p = set(val_df["parent_dataset_id"].unique())
    test_p = set(test_df["parent_dataset_id"].unique())
    if train_p & val_p or train_p & test_p or val_p & test_p:
        raise RuntimeError("parent overlap after three-way split")
    return train_df, val_df, test_df


def sort_for_ranking(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["graph_id", "logit_idx"], kind="mergesort").reset_index(drop=True)


def group_sizes(df: pd.DataFrame) -> List[int]:
    return df.groupby("graph_id", sort=False).size().tolist()


def build_dmatrix(df: pd.DataFrame) -> xgb.DMatrix:
    dmat = xgb.DMatrix(df[FEATURE_COLUMN_NAMES], label=df["y_class"])
    dmat.set_group(group_sizes(df))
    return dmat


def edge_accuracy(df: pd.DataFrame, scores: np.ndarray) -> float:
    """Fraction of graph groups where argmax score hits y_class==1."""
    df = df.copy()
    df["_score"] = scores
    correct = 0
    total = 0
    for _, group in df.groupby("graph_id", sort=False):
        total += 1
        best_idx = int(group["_score"].values.argmax())
        if int(group.iloc[best_idx]["y_class"]) == 1:
            correct += 1
    return correct / max(total, 1)


def main() -> None:
    args = parse_args()
    df = load_frame(args.input)
    validate_extracted_frame(df)

    train_df, test_df = split_by_parent(df, args.test_size, args.random_state)
    train_df = sort_for_ranking(train_df)
    test_df = sort_for_ranking(test_df)

    dtrain = build_dmatrix(train_df)
    dtest = build_dmatrix(test_df)

    params = {
        "objective": "rank:pairwise",
        "eval_metric": "ndcg",
        "tree_method": "hist",
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": args.random_state,
    }

    evals = [(dtrain, "train"), (dtest, "test")]
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=args.num_boost_round,
        evals=evals,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=25,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(args.output))

    train_scores = booster.predict(dtrain)
    test_scores = booster.predict(dtest)
    metrics = {
        "train_edge_accuracy": edge_accuracy(train_df, train_scores),
        "test_edge_accuracy": edge_accuracy(test_df, test_scores),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_graphs": int(train_df["graph_id"].nunique()),
        "test_graphs": int(test_df["graph_id"].nunique()),
        "params": params,
        "best_iteration": int(getattr(booster, "best_iteration", args.num_boost_round)),
    }
    meta_path = args.output.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[+] Saved model -> {args.output}")
    print(f"[+] Metrics -> {meta_path}")
    print(
        f"    train edge accuracy={metrics['train_edge_accuracy']:.4f} "
        f"test edge accuracy={metrics['test_edge_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
