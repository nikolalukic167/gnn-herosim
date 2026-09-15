"""route_b stage 2 W2: tied-dim25cr label parity with dim63crk.

A3's plain dim25cr path labels from graph.y (the unconstrained sweep minimum), but §3
requires "same labels, same alpha" across arms — A1/A2 teacher-force along the alpha=2.0
tied-optimal plan set. extract_rows_dim25cr_tied_from_batch_graph shares the exact same
tied-plan loop as extract_rows_dim63crk_from_batch_graph (one function,
_extract_tied_plan_rows_from_batch_graph, with include_partial_state toggled) so the two
extractions cannot silently drift into different labels or different plan walks.

These tests pin: byte-identical targets/graph_ids on the same graph, identical static 25
columns (dim22 + the 3 candidate-relative columns), and determinism.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.policy.tabular.reduced_features import (  # noqa: E402
    DIM25CR_FEATURE_DIM,
    DIM63CRK_FEATURE_DIM,
    extract_rows_dim25cr_tied_from_batch_graph,
    extract_rows_dim63crk_from_batch_graph,
)

CACHE = REPO_ROOT / "simulation_data" / "graphs_cache_route_b_smoke_s_dag" / "graphs.pkl"
ALPHA = "2.0"


@pytest.fixture(scope="module")
def graphs():
    if not CACHE.exists():
        pytest.skip(f"no --dag-partial-state cache at {CACHE}")
    with open(CACHE, "rb") as fh:
        return pickle.load(fh)


def test_tied_dim25cr_width_is_dim25cr(graphs):
    graph = graphs[0]
    rows, skip = extract_rows_dim25cr_tied_from_batch_graph(
        graph, "tied25", alpha_key=ALPHA
    )
    assert skip is None
    assert rows
    for row in rows:
        assert row.features.shape[0] == DIM25CR_FEATURE_DIM


def test_tied_dim25cr_targets_and_graph_ids_match_dim63crk(graphs):
    """The tied-plan walk is shared code: targets and graph_ids must be byte-identical
    row-for-row, since both extractors visit the same order, same plans, same labels."""
    for graph in graphs:
        crk_rows, crk_skip = extract_rows_dim63crk_from_batch_graph(
            graph, "cmp", alpha_key=ALPHA
        )
        cr_rows, cr_skip = extract_rows_dim25cr_tied_from_batch_graph(
            graph, "cmp", alpha_key=ALPHA
        )
        assert crk_skip is None and cr_skip is None
        assert len(crk_rows) == len(cr_rows)
        for a, b in zip(crk_rows, cr_rows):
            assert a.graph_id == b.graph_id
            assert a.parent_dataset_id == b.parent_dataset_id
            assert a.task_idx == b.task_idx
            assert a.logit_idx == b.logit_idx
            assert a.node_id == b.node_id
            assert a.platform_id == b.platform_id
            assert a.y_logit == b.y_logit
            assert a.y_class == b.y_class


def test_tied_dim25cr_static_25_columns_identical_to_dim63crk_prefix(graphs):
    """The first 25 columns (dim22 + the 3 CR columns) must be identical between the
    two layouts — the partial-state block is the ONLY thing include_partial_state
    toggles, never the static columns computed ahead of it."""
    for graph in graphs:
        crk_rows, _ = extract_rows_dim63crk_from_batch_graph(graph, "cmp", alpha_key=ALPHA)
        cr_rows, _ = extract_rows_dim25cr_tied_from_batch_graph(graph, "cmp", alpha_key=ALPHA)
        for a, b in zip(crk_rows, cr_rows):
            assert a.features.shape[0] == DIM63CRK_FEATURE_DIM
            assert b.features.shape[0] == DIM25CR_FEATURE_DIM
            np.testing.assert_array_equal(a.features[:DIM25CR_FEATURE_DIM], b.features)


def test_tied_dim25cr_determinism(graphs):
    graph = graphs[0]
    rows1, _ = extract_rows_dim25cr_tied_from_batch_graph(graph, "det", alpha_key=ALPHA)
    rows2, _ = extract_rows_dim25cr_tied_from_batch_graph(graph, "det", alpha_key=ALPHA)
    assert len(rows1) == len(rows2)
    for a, b in zip(rows1, rows2):
        assert a.graph_id == b.graph_id
        assert a.y_logit == b.y_logit
        np.testing.assert_array_equal(a.features, b.features)


def test_tied_dim25cr_refuses_non_dag_graph():
    """A graph carrying no tied_optimal_logit_plans (not a DAG cache) must skip loudly
    via the returned skip_reason, same refusal message pattern as dim63crk."""

    class _FakeGraph:
        n_tasks = 4

    rows, skip = extract_rows_dim25cr_tied_from_batch_graph(_FakeGraph(), "fake")
    assert rows == []
    assert skip is not None
    assert "tied_optimal_logit_plans" in skip


# --- trainer --tied-labels flag validation (subprocess, fast-fail before training) ----

BATCH_CACHE_NON_DAG = (
    REPO_ROOT / "simulation_data" / "graphs_cache_regime_b_oracle_split_cosim"
)

needs_smoke_dag = pytest.mark.skipif(
    not CACHE.parent.is_dir(), reason=f"cache not present at {CACHE.parent}"
)
needs_non_dag = pytest.mark.skipif(
    not BATCH_CACHE_NON_DAG.is_dir(), reason=f"cache not present at {BATCH_CACHE_NON_DAG}"
)


def _run_trainer(args):
    env = dict(os.environ)
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "PIPENV_IGNORE_VIRTUALENVS": "1",
            "PYTHONPATH": str(REPO_ROOT),
            "WANDB_MODE": "disabled",
        }
    )
    return subprocess.run(
        [sys.executable, "src/policy/tabular/train_mlp_dim22_from_batch.py", *args],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


def test_tied_labels_requires_candidate_relative_queue(tmp_path):
    result = _run_trainer(
        [
            "--cache-dir", str(CACHE.parent),
            "--output", str(tmp_path / "mlp.pt"),
            "--epochs", "1",
            "--tied-labels",
        ]
    )
    assert result.returncode != 0
    assert "--candidate-relative-queue" in result.stderr


def test_tied_labels_rejects_alongside_partial_state(tmp_path):
    result = _run_trainer(
        [
            "--cache-dir", str(CACHE.parent),
            "--output", str(tmp_path / "mlp.pt"),
            "--epochs", "1",
            "--candidate-relative-queue",
            "--partial-state",
            "--tied-labels",
        ]
    )
    assert result.returncode != 0
    assert "redundant with --partial-state" in result.stderr


@needs_non_dag
def test_tied_labels_refuses_a_non_dag_cache(tmp_path):
    result = _run_trainer(
        [
            "--cache-dir", str(BATCH_CACHE_NON_DAG),
            "--output", str(tmp_path / "mlp.pt"),
            "--epochs", "1",
            "--candidate-relative-queue",
            "--tied-labels",
        ]
    )
    assert result.returncode != 0
    assert "dag_partial_state" in result.stderr


@needs_smoke_dag
def test_tied_labels_trains_and_stamps_label_mode(tmp_path):
    out = tmp_path / "mlp.pt"
    result = _run_trainer(
        [
            "--cache-dir", str(CACHE.parent),
            "--output", str(out),
            "--epochs", "1",
            "--random-state", "4242",
            "--candidate-relative-queue",
            "--tied-labels",
        ]
    )
    assert result.returncode == 0, (
        f"trainer exited {result.returncode}\n--- stdout ---\n{result.stdout[-3000:]}"
        f"\n--- stderr ---\n{result.stderr[-3000:]}"
    )
    meta = json.loads((tmp_path / "mlp.pt.meta.json").read_text())
    assert meta["inference_feature_layout"] == "dim25cr"
    assert meta["label_mode"] == "any_of_k_tied_optimal"
    assert meta["label_alpha_key"] == "2.0"
    assert meta["tied_labels"] is True

    import torch

    checkpoint = torch.load(str(out), map_location="cpu", weights_only=False)
    assert checkpoint["label_mode"] == "any_of_k_tied_optimal"
    assert checkpoint["label_alpha_key"] == "2.0"
    assert checkpoint["tied_labels"] is True
