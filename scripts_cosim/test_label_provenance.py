#!/usr/bin/env python3
"""The label-provenance preflight must fail loud when cache labels drift off the sweep.

Regression guard for the 2026-08-17 retraction: the separability gate measures coupling in
placements.jsonl while the ablation measures models against cache labels, and nothing
checked the two describe the same optimum.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gnn_necessity_ablation import audit_label_provenance  # noqa: E402


class FakeGraph:
    """Minimal stand-in for the cached PyG graph fields the audit reads."""

    def __init__(self, dataset_id, label_indices, logit_to_placement):
        self.dataset_id = dataset_id
        self.y = torch.tensor(label_indices, dtype=torch.long)
        self.n_tasks = len(label_indices)
        self.task_logit_to_placement = logit_to_placement


def write_sweep(corpus_root: Path, dataset_id: str, rows):
    ds_dir = corpus_root / dataset_id / "placements"
    ds_dir.mkdir(parents=True, exist_ok=True)
    with (ds_dir / "placements.jsonl").open("w") as f:
        for plan, rtt in rows:
            f.write(
                json.dumps(
                    {
                        "placement_plan": {str(t): list(p) for t, p in plan.items()},
                        "rtt": rtt,
                    }
                )
                + "\n"
            )


# Two tasks, two candidate placements each. Plan {(0,0),(0,0)} is the optimum at 1.0.
CANDIDATES = [[(0, 0), (1, 1)], [(0, 0), (1, 1)]]
SWEEP = [
    ({0: (0, 0), 1: (0, 0)}, 1.0),
    ({0: (0, 0), 1: (1, 1)}, 2.0),
    ({0: (1, 1), 1: (0, 0)}, 2.0),
    ({0: (1, 1), 1: (1, 1)}, 4.0),
]


@pytest.fixture()
def corpus(tmp_path):
    write_sweep(tmp_path, "ds_00000", SWEEP)
    return tmp_path


def test_sweep_minimum_label_passes(corpus):
    graphs = [FakeGraph("ds_00000", [0, 0], CANDIDATES)]
    report = audit_label_provenance(graphs, corpus)
    assert report["n_checked"] == 1
    assert report["n_label_is_sweep_min"] == 1
    assert report["label_regret_max"] == 0.0


def test_suboptimal_label_fails_loud(corpus):
    """Label points at the 4.0 plan while the sweep minimum is 1.0."""
    graphs = [FakeGraph("ds_00000", [1, 1], CANDIDATES)]
    with pytest.raises(RuntimeError, match="LABEL PROVENANCE AUDIT FAILED"):
        audit_label_provenance(graphs, corpus)


def test_suboptimal_label_reports_regret(corpus):
    graphs = [FakeGraph("ds_00000", [0, 1], CANDIDATES)]
    with pytest.raises(RuntimeError) as exc:
        audit_label_provenance(graphs, corpus)
    # (2.0 - 1.0) / 1.0 = 100%
    assert "max 100.00%" in str(exc.value)


def test_label_absent_from_sweep_fails(tmp_path):
    """A label whose plan was never enumerated is a harder failure than drift."""
    write_sweep(tmp_path, "ds_00000", SWEEP[:1])
    graphs = [FakeGraph("ds_00000", [1, 1], CANDIDATES)]
    with pytest.raises(RuntimeError, match="absent from their sweep"):
        audit_label_provenance(graphs, tmp_path)


def test_missing_placements_jsonl_fails(tmp_path):
    (tmp_path / "ds_00000").mkdir()
    graphs = [FakeGraph("ds_00000", [0, 0], CANDIDATES)]
    with pytest.raises(RuntimeError, match="missing placements.jsonl"):
        audit_label_provenance(graphs, tmp_path)
