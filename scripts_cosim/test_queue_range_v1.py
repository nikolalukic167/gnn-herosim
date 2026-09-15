#!/usr/bin/env python3
"""queue_range_v1 -- the serving overrides and the mechanism instrument.

Registered in docs/lineages/queue_range_v1.md. These tests exist because two of the three
ways this experiment could silently become a different experiment are code, not statistics:
a knob that never reaches the feature builder, and counters the orchestrator's scalar
whitelist drops on the way out.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.placement.queue_features import (  # noqa: E402
    CORPUS_DIM13_CANDIDATE_MAX,
    CORPUS_DIM7_CANDIDATE_MAX,
    QUEUE_RANGE_BLIND_DIM7,
    QUEUE_RANGE_BLIND_RAW,
    apply_serve_clamp,
    serve_dim7_clamp,
    serve_dim13_clamp,
    serve_queue_divisor_override,
)

ENVS = ("GNN_QUEUE_SERVE_DIVISOR", "GNN_QUEUE_SERVE_DIM7_CLAMP",
        "GNN_QUEUE_SERVE_DIM13_CLAMP")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ENVS:
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------- the overrides themselves
def test_all_overrides_default_off():
    assert serve_queue_divisor_override() is None
    assert serve_dim7_clamp() is None
    assert serve_dim13_clamp() is None


def test_blank_is_off_not_zero(monkeypatch):
    monkeypatch.setenv("GNN_QUEUE_SERVE_DIVISOR", "   ")
    assert serve_queue_divisor_override() is None


@pytest.mark.parametrize("bad", ["0", "-1", "nope", "nan", "inf"])
def test_non_positive_or_unparseable_raises(monkeypatch, bad):
    monkeypatch.setenv("GNN_QUEUE_SERVE_DIVISOR", bad)
    with pytest.raises(ValueError):
        serve_queue_divisor_override()


def test_clamp_is_a_no_op_below_the_clamp():
    assert apply_serve_clamp(3.0, 42.0) == 3.0
    assert apply_serve_clamp(42.0, 42.0) == 42.0
    assert apply_serve_clamp(4200.0, 42.0) == 42.0
    assert apply_serve_clamp(4200.0, None) == 4200.0


def test_corpus_constants_are_the_measured_ones():
    # graphs_cache_drainable_objective_v1_v1, 516 datasets, candidate platforms only:
    # dim7 p50 12 / p90 25 / p99 34 / max 42 ; dim13 max 0.44. Changing these changes what
    # "inside the trained range" means, so they are pinned.
    assert CORPUS_DIM7_CANDIDATE_MAX == 42.0
    assert CORPUS_DIM13_CANDIDATE_MAX == 0.44


# ------------------------------------------------------- the override reaching the builder
def _build(monkeypatch, all_depths, cand_depths, **env):
    """dim7 for the candidate rows, through the real builder's own divisor rule.

    `all_depths` is the whole snapshot -- the divisor is a p90 over every platform, and in
    the corpus >= 90 % of them are idle (dim7 p90 is 0.0 over all 69,144 platform rows and
    25.0 over the 2,544 candidate rows), which is exactly why the divisor is 1.0 there.
    """
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    from src.placement.queue_features import (
        queue_depth_norm, serve_queue_divisor_override as ovr,
        serve_dim7_clamp as c7,
    )
    divisor = queue_depth_norm(all_depths, "scheduler_adaptive", "legacy_v0")
    override = ovr()
    if override is not None:
        divisor = override
    return [apply_serve_clamp(d / divisor, c7()) for d in cand_depths]


IDLE = [0.0] * 40
SHALLOW = [0.0, 5.0, 10.0, 20.0]
DEEP = [1000.0, 1005.0, 1010.0, 1020.0]


def test_the_corpus_divisor_really_is_one(monkeypatch):
    from src.placement.queue_features import queue_depth_norm
    assert queue_depth_norm(IDLE + SHALLOW, "scheduler_adaptive", "legacy_v0") == 1.0


def test_mostly_idle_snapshot_keeps_the_divisor_at_one(monkeypatch):
    """Failure mode ONE: the column leaves its trained range.

    The divisor is a p90 over EVERY platform, so while >= 10 % of platforms are idle it
    pins at 1.0 and dim7 is the raw queue depth -- the training semantics, and unbounded.
    A candidate 1,000 tasks deep therefore reads as 1000.0 against a corpus maximum of 42.
    """
    got = _build(monkeypatch, IDLE + DEEP, DEEP)
    assert got == DEEP
    assert max(got) > CORPUS_DIM7_CANDIDATE_MAX * 20


def test_busy_snapshot_compresses_the_column(monkeypatch):
    """Failure mode TWO: the column stops resolving.

    Once most platforms are busy the divisor inflates (capped at 100 under legacy_v0), and
    the SAME 20-task gap between candidates reads as 0.2 of feature. Which of the two modes
    a live cell is actually in is Q0's job; both are out-of-contract, and the intervention
    is the same.
    """
    busy_shallow = [float(v) for v in range(1, 41)] + SHALLOW
    busy_deep = [1000.0 + v for v in range(1, 41)] + DEEP
    shallow = _build(monkeypatch, busy_shallow, SHALLOW)
    deep = _build(monkeypatch, busy_deep, DEEP)
    s_gap = max(shallow) - min(shallow)
    d_gap = max(deep) - min(deep)
    assert d_gap < s_gap
    from src.placement.queue_features import queue_depth_norm
    assert queue_depth_norm(busy_deep, "scheduler_adaptive", "legacy_v0") > 1.0


def test_pinned_divisor_restores_the_training_semantics(monkeypatch):
    got = _build(monkeypatch, IDLE + DEEP, DEEP, GNN_QUEUE_SERVE_DIVISOR=1.0)
    assert got == DEEP
    assert max(got) - min(got) == pytest.approx(20.0)


def test_clamp_keeps_the_column_inside_the_trained_range(monkeypatch):
    got = _build(monkeypatch, IDLE + DEEP, DEEP, GNN_QUEUE_SERVE_DIVISOR=1.0,
                 GNN_QUEUE_SERVE_DIM7_CLAMP=CORPUS_DIM7_CANDIDATE_MAX)
    assert max(got) <= CORPUS_DIM7_CANDIDATE_MAX
    # And it is honest about what it costs: above the clamp the ordering is gone.
    assert len(set(got)) == 1


def test_clamp_preserves_order_inside_the_range(monkeypatch):
    got = _build(monkeypatch, IDLE + SHALLOW, SHALLOW, GNN_QUEUE_SERVE_DIVISOR=1.0,
                 GNN_QUEUE_SERVE_DIM7_CLAMP=CORPUS_DIM7_CANDIDATE_MAX)
    assert got == SHALLOW


def test_feature_builder_reads_the_override(monkeypatch):
    """The knob must reach the module the graph is actually built in, not just its own."""
    import importlib
    fb = importlib.import_module("src.policy.tabular.feature_builder")
    assert fb.serve_queue_divisor_override is serve_queue_divisor_override
    assert fb.serve_dim7_clamp is serve_dim7_clamp
    assert fb.serve_dim13_clamp is serve_dim13_clamp
    src = Path(fb.__file__).read_text()
    # dim7 and dim13 must both go through the clamp; a half-applied intervention is a
    # different experiment than the one registered.
    assert "apply_serve_clamp(\n                float(queue_len_raw) / float(queue_norm)" in src
    assert "dim13_feat = apply_serve_clamp(" in src


# --------------------------------------------------------------------- the instrument
class _Pf:
    def __init__(self, rows):
        self._rows = rows

    def size(self, dim):
        return len(self._rows) if dim == 0 else len(self._rows[0])

    def __getitem__(self, idx):
        pos, col = idx
        return _Cell(self._rows[pos][col])


class _Cell:
    def __init__(self, v):
        self._v = v

    def item(self):
        return self._v


class _Graph:
    pass


def _stub_graph(depths, dim7s, divisor):
    g = _Graph()
    keys = [f"n{i}:0" for i in range(len(depths))]
    g.platform_features = _Pf([[0.0] * 7 + [d7] + [0.0] * 6 for d7 in dim7s])
    g.queue_key_to_platform_meta = {k: {"platform_pos": i} for i, k in enumerate(keys)}
    g.task_logit_to_queue_key = {0: list(keys)}
    g.queue_snapshot = {k: depths[i] for i, k in enumerate(keys)}
    g.queue_norm_divisor = divisor
    return g


class _Sched:
    """The recorder under test, lifted off GNNScheduler without its SimPy machinery."""

    def __init__(self, now):
        from src.policy.gnn.scheduler import GNNScheduler
        self._queue_range_records = []
        self.env = type("E", (), {"now": now})()
        self._record_queue_range = GNNScheduler._record_queue_range.__get__(self)
        for name in ("queue_range_records", "qr_batches", "qr_blind_batches",
                     "qr_divisor_above_one_batches", "qr_dim7_over_corpus_batches"):
            setattr(type(self), name, getattr(GNNScheduler, name))


def test_recorder_writes_one_row_per_batch():
    s = _Sched(now=12.5)
    s._record_queue_range(_stub_graph([0, 10, 20], [0.0, 10.0, 20.0], 1.0))
    assert s.qr_batches == 1
    row = s.queue_range_records[0]
    assert row[0] == 12.5 and row[1] == 1.0
    assert row[2] == 20.0 and row[3] == 20.0      # raw spread, dim7 spread
    assert row[4] == 0.0                           # not blind: the column sees the pile
    assert row[5] == 20.0


def test_recorder_flags_a_compressed_column_as_blind():
    """1000 tasks of difference that reach the model as 0.02 of feature."""
    s = _Sched(now=1.0)
    s._record_queue_range(_stub_graph([5000, 5500, 6000], [0.50, 0.51, 0.52], 100.0))
    assert s.qr_blind_batches == 1
    assert s.queue_range_records[0][2] >= QUEUE_RANGE_BLIND_RAW
    assert s.queue_range_records[0][3] < QUEUE_RANGE_BLIND_DIM7
    assert s.qr_divisor_above_one_batches == 1


def test_recorder_counts_out_of_range_batches():
    s = _Sched(now=1.0)
    s._record_queue_range(_stub_graph([0, 50], [0.0, 50.0], 1.0))
    s._record_queue_range(_stub_graph([0, 40], [0.0, 40.0], 1.0))
    assert s.qr_dim7_over_corpus_batches == 1      # 50 > 42, 40 is inside


def test_recorder_ignores_a_layout_without_the_column():
    s = _Sched(now=1.0)
    g = _stub_graph([0, 10], [0.0, 10.0], 1.0)
    g.platform_features = _Pf([[0.0] * 8, [0.0] * 8])   # ce_reduced-width rows
    s._record_queue_range(g)
    assert s.qr_batches == 0


def test_counters_survive_the_orchestrator_whitelist():
    """The trap that made checkpoint_mp_config's guard never fire."""
    src = (REPO_ROOT / "src/placement/orchestrator.py").read_text()
    for name in ("qr_batches", "qr_blind_batches", "qr_divisor_above_one_batches",
                 "qr_dim7_over_corpus_batches"):
        assert f'"{name}"' in src, f"{name} is not in _scheduler_counters' whitelist"
    # The trace is a list; the scalar filter would drop it without the explicit passthrough.
    assert '"queue_range_records"' in src
    assert "isinstance(value, list)" in src


def test_scheduler_exposes_them_as_properties():
    from src.policy.gnn.scheduler import GNNScheduler
    for name in ("qr_batches", "qr_blind_batches", "qr_divisor_above_one_batches",
                 "qr_dim7_over_corpus_batches", "queue_range_records"):
        assert isinstance(getattr(GNNScheduler, name), property), name
