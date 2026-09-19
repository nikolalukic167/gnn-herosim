#!/usr/bin/env python3
"""Unit tests for the live dim7/dim13 chosen-vs-shortest probe and no-MP flag.

Run: pipenv run python3 -m pytest scripts_cosim/test_gnn_queue_feature_probe.py -q
"""
from __future__ import annotations

import os

import pytest
import numpy as np
import torch

from src.policy.gnn.seq_decode import (
    GnnDecodeRunStats,
    record_queue_feature_discrimination,
)


def test_dim7_blind_when_relative_feature_ties_on_a_raw_pile() -> None:
    """Two machines at 30000 vs 32000 can share dim7≈1.0 — the probe must flag that."""
    stats = GnnDecodeRunStats()
    pf = np.zeros((2, 14), dtype=np.float32)
    pf[0, 7] = 1.00
    pf[1, 7] = 1.01
    pf[0, 13] = 2.8
    pf[1, 13] = 2.82
    logits = [torch.tensor([0.1, 4.0])]  # prefers the longer line, large margin
    record_queue_feature_discrimination(
        stats,
        combo=((0, 1),),
        logits_per_task=logits,
        task_logit_to_placement={0: [(0, 0), (0, 1)]},
        queue_snapshot={"n:0": 30000, "n:1": 32000},
        task_logit_to_queue_key={0: ["n:0", "n:1"]},
        platform_features=pf,
        queue_key_to_platform_meta={
            "n:0": {"platform_pos": 0},
            "n:1": {"platform_pos": 1},
        },
    )
    summary = stats.summary()["queue_feature_discrimination"]
    assert stats.feature_probe_tasks == 1
    assert summary["dim7_blind_rate"] == 1.0
    assert summary["frac_raw_gap_ge_10"] == 1.0
    assert summary["logit_tied_rate"] == 0.0
    assert summary["confident_worse_queue_rate"] == 1.0


def test_discriminating_dim7_is_not_blind() -> None:
    stats = GnnDecodeRunStats()
    pf = np.zeros((2, 14), dtype=np.float32)
    pf[0, 7] = 0.1
    pf[1, 7] = 1.5
    pf[0, 13] = 0.2
    pf[1, 13] = 2.5
    logits = [torch.tensor([4.0, 0.1])]  # correctly prefers the short line
    record_queue_feature_discrimination(
        stats,
        combo=((0, 0),),
        logits_per_task=logits,
        task_logit_to_placement={0: [(0, 0), (0, 1)]},
        queue_snapshot={"n:0": 5, "n:1": 400},
        task_logit_to_queue_key={0: ["n:0", "n:1"]},
        platform_features=pf,
        queue_key_to_platform_meta={
            "n:0": {"platform_pos": 0},
            "n:1": {"platform_pos": 1},
        },
    )
    summary = stats.summary()["queue_feature_discrimination"]
    assert summary["dim7_blind_rate"] == 0.0
    assert summary["confident_worse_queue_rate"] == 0.0
    assert stats.dim7_chosen[0] == pytest.approx(0.1, rel=0, abs=1e-6)
    assert stats.dim7_minq[0] == pytest.approx(0.1, rel=0, abs=1e-6)


def test_missing_meta_fails_loud() -> None:
    stats = GnnDecodeRunStats()
    pf = np.zeros((1, 14), dtype=np.float32)
    try:
        record_queue_feature_discrimination(
            stats,
            combo=((0, 0),),
            logits_per_task=[torch.tensor([1.0])],
            task_logit_to_placement={0: [(0, 0)]},
            queue_snapshot={"n:0": 3},
            task_logit_to_queue_key={0: ["n:0"]},
            platform_features=pf,
            queue_key_to_platform_meta={},
        )
    except RuntimeError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for missing meta")


def test_disable_message_passing_skips_gin(monkeypatch) -> None:
    from src.policy.gnn.gnn_model import TaskPlacementGNN
    from torch_geometric.data import Data

    monkeypatch.setenv("GNN_DISABLE_MESSAGE_PASSING", "1")
    model = TaskPlacementGNN(
        task_feature_dim=3,
        platform_feature_dim=14,
        embedding_dim=8,
        hidden_dim=16,
        num_layers=3,
        edge_dim=5,
    )
    called = {"gin": False}
    orig = model.gin.forward

    def boom(*args, **kwargs):
        called["gin"] = True
        return orig(*args, **kwargs)

    model.gin.forward = boom  # type: ignore[method-assign]
    n_tasks, n_plat = 2, 3
    data = Data(
        edge_index=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        n_tasks=n_tasks,
        n_platforms=n_plat,
        task_features=torch.randn(n_tasks, 3),
        platform_features=torch.randn(n_plat, 14),
        edge_attr=torch.randn(2, 5),
    )
    logits = model(data)
    assert called["gin"] is False
    assert len(logits) == n_tasks
