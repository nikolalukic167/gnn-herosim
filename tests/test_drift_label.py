"""Tests for the drainable_objective_v1 shaped label (scripts_cosim/drift_label.py).

Committed before any datum the label produces exists. Run:

    PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH=. pipenv run python3 \
        -m pytest tests/test_drift_label.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts_cosim.drift_label import (
    DriftLabelError,
    StateContext,
    added_seconds,
    approx_comm,
    build_state_context,
    drain_table_key,
    externality_seconds,
    load_drain_table,
    parse_label_objective,
    shaped_rows,
    shaped_value,
)

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# A two-platform synthetic whose externality can be worked out by hand
# --------------------------------------------------------------------------


def _ctx(
    *,
    drain=(2.0, 2.0),
    backlog=(5, 0),
    lam=(1.0, 1.0),
    n_tasks=4,
) -> StateContext:
    """Platforms 10 (type "a") and 20 (type "b"), all tasks of type "t"."""
    return StateContext(
        platform_type_of={10: "a", 20: "b"},
        node_of={10: "n0", 20: "n1"},
        queue_key_of={10: "n0:10", 20: "n1:20"},
        backlog_counts={("t", 10): backlog[0], ("t", 20): backlog[1]},
        task_type_names=["t"] * n_tasks,
        drain={("t", "a"): drain[0], ("t", "b"): drain[1]},
        lam={10: lam[0], 20: lam[1]},
    )


def test_backlog_and_added_are_counts_times_drain():
    ctx = _ctx()
    assert ctx.backlog_seconds() == {10: 10.0}  # 5 queued x 2 s
    plan = {0: (0, 10), 1: (0, 10), 2: (1, 20), 3: (1, 20)}
    assert added_seconds(plan, ctx) == {10: 4.0, 20: 4.0}


def test_externality_matches_the_closed_form_computed_by_hand():
    ctx = _ctx()
    # Platform 10: B = 10, A = 4, lam = 1  -> 0.5 * ((14)^2 - (10)^2) = 48
    # Platform 20: B = 0,  A = 4, lam = 1  -> 0.5 * ((4)^2  - 0)      =  8
    plan = {0: (0, 10), 1: (0, 10), 2: (1, 20), 3: (1, 20)}
    assert externality_seconds(plan, ctx) == pytest.approx(56.0)


def test_stacking_on_the_deep_platform_costs_more_than_spreading():
    """The whole point of the label: concentrating on a loaded server is charged."""
    ctx = _ctx()
    spread = {0: (0, 10), 1: (0, 10), 2: (1, 20), 3: (1, 20)}
    stacked = {0: (0, 10), 1: (0, 10), 2: (0, 10), 3: (0, 10)}
    assert externality_seconds(stacked, ctx) > externality_seconds(spread, ctx)
    # B = 10, A = 8 -> 0.5 * (18^2 - 10^2) = 112 against 56.
    assert externality_seconds(stacked, ctx) == pytest.approx(112.0)


def test_the_term_is_superlinear_in_added_work():
    """Doubling A more than doubles the charge -- that is the quadratic doing its job."""
    ctx = _ctx(backlog=(0, 0))
    one = externality_seconds({0: (0, 10)}, ctx)
    two = externality_seconds({0: (0, 10), 1: (0, 10)}, ctx)
    assert two > 2 * one


def test_a_deeper_backlog_makes_the_same_placement_cost_more():
    shallow = _ctx(backlog=(1, 0))
    deep = _ctx(backlog=(20, 0))
    plan = {0: (0, 10), 1: (0, 10)}
    assert externality_seconds(plan, deep) > externality_seconds(plan, shallow)


def test_zero_arrival_rate_platforms_contribute_nothing():
    """No future arrivals means no externality, however deep the queue."""
    ctx = _ctx(lam=(0.0, 1.0), backlog=(50, 0))
    assert externality_seconds({0: (0, 10)}, ctx) == 0.0


def test_v_zero_returns_the_sweep_value_bit_for_bit():
    ctx = _ctx()
    rows = [({0: (0, 10)}, 123.456789), ({0: (1, 20)}, 0.0)]
    out = shaped_rows(rows, ctx, 0.0)
    assert [v for _p, v in out] == [123.456789, 0.0]
    assert shaped_value(1.5, {0: (0, 10)}, ctx, 0.0) == 1.5


def test_v_scales_the_shaped_term_linearly():
    ctx = _ctx()
    plan = {0: (0, 10), 1: (0, 10)}
    base = externality_seconds(plan, ctx)
    assert shaped_value(100.0, plan, ctx, 1.0) == pytest.approx(100.0 + base)
    assert shaped_value(100.0, plan, ctx, 2.0) == pytest.approx(100.0 + 2 * base)


def test_shaped_rows_preserves_order_and_length():
    ctx = _ctx()
    rows = [({0: (0, 10)}, 10.0), ({0: (1, 20)}, 20.0), ({0: (0, 10)}, 30.0)]
    out = shaped_rows(rows, ctx, 1.0)
    assert len(out) == 3
    assert [p for p, _v in out] == [p for p, _v in rows]


# --------------------------------------------------------------------------
# Fail loud
# --------------------------------------------------------------------------


def test_missing_drain_entry_is_fatal_not_defaulted():
    ctx = _ctx()
    bad = {0: (0, 999)}  # platform not in the state
    with pytest.raises(DriftLabelError, match="replica_placements"):
        added_seconds(bad, ctx)


def test_a_plan_task_beyond_the_declared_batch_is_fatal():
    ctx = _ctx(n_tasks=2)
    with pytest.raises(DriftLabelError, match="declares 2 tasks"):
        added_seconds({0: (0, 10), 5: (0, 10)}, ctx)


def test_drain_table_pointing_at_a_missing_file_is_fatal(monkeypatch, tmp_path):
    monkeypatch.setenv("HEROSIM_BACKLOG_DRAIN_TABLE", str(tmp_path / "nope.json"))
    with pytest.raises(DriftLabelError, match="does not exist"):
        load_drain_table()


def test_unset_drain_table_is_none_not_an_error(monkeypatch):
    monkeypatch.delenv("HEROSIM_BACKLOG_DRAIN_TABLE", raising=False)
    assert load_drain_table() is None


def test_a_non_positive_measured_drain_is_fatal(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"drain_seconds_per_item": {"t|a": 0.0}}))
    with pytest.raises(DriftLabelError, match="non-positive"):
        load_drain_table(p)


def test_parse_label_objective():
    assert parse_label_objective("rtt") == ("rtt", 0.0)
    assert parse_label_objective("rtt_drift:1.0") == ("rtt_drift", 1.0)
    assert parse_label_objective("rtt_drift:0.5") == ("rtt_drift", 0.5)
    with pytest.raises(DriftLabelError, match="explicit coefficient"):
        parse_label_objective("rtt_drift")
    with pytest.raises(DriftLabelError, match="unknown label objective"):
        parse_label_objective("makespan")


# --------------------------------------------------------------------------
# The default clock must be the simulator's own, not a copy that drifted
# --------------------------------------------------------------------------


def test_approx_comm_agrees_with_the_simulator_helper():
    from src.placement.live_snapshot_seed import _approx_comm

    task_type = {"stateSize": {"app": {"input": 153600, "output": 8000}}}
    assert approx_comm(task_type) == pytest.approx(_approx_comm(task_type))


def test_default_drain_is_execution_plus_comm_exactly():
    """The default clock must reproduce seed_virtual_warmup's per-item charge."""
    task_type = {
        "executionTime": {"rpiCpu": 0.0029},
        "stateSize": {"app": {"input": 153600, "output": 8000}},
    }
    expected = 0.0029 + approx_comm(task_type)
    ctx = StateContext(
        platform_type_of={1: "rpiCpu"},
        node_of={1: "n"},
        queue_key_of={1: "n:1"},
        backlog_counts={("dnn1", 1): 1},
        task_type_names=["dnn1"],
        drain={("dnn1", "rpiCpu"): expected},
        lam={1: 1.0},
    )
    assert ctx.backlog_seconds()[1] == pytest.approx(expected)


# --------------------------------------------------------------------------
# On a real dataset
# --------------------------------------------------------------------------

CORPUS = REPO / "simulation_data" / "gnn_datasets_peer_affinity_v1_c3_x200_train2"
TASK_TYPES = REPO / "data" / "nofs-ids" / "task-types.json"


@pytest.mark.skipif(
    not (CORPUS / "ds_00000" / "infrastructure.json").exists() or not TASK_TYPES.exists(),
    reason="peer_affinity x200 train2 corpus not on this machine",
)
def test_build_state_context_on_a_real_dataset():
    db = json.loads(TASK_TYPES.read_text())
    ctx = build_state_context(CORPUS / "ds_00000", db, arrival_rate=0.46)

    assert ctx.task_type_names and len(ctx.task_type_names) == 10
    assert ctx.platform_type_of, "no replicas parsed"
    # Every platform that serves a type gets a positive share of the arrival rate.
    assert all(v > 0 for v in ctx.lam.values())
    assert sum(ctx.lam.values()) == pytest.approx(0.46, rel=1e-9)
    # The backlog is on the corpus's own clock, which is what this lineage calls wrong:
    # a queued item costs well under a second.
    backlog = ctx.backlog_seconds()
    assert backlog, "this dataset has no seeded queue at all"
    per_item = [
        backlog[pid] / sum(c for (_t, p), c in ctx.backlog_counts.items() if p == pid)
        for pid in backlog
    ]
    assert max(per_item) < 2.0


@pytest.mark.skipif(
    not (CORPUS / "ds_00000" / "placements" / "placements.jsonl").exists()
    or not TASK_TYPES.exists(),
    reason="peer_affinity x200 train2 corpus not on this machine",
)
def test_relabelling_a_real_sweep_changes_the_argmin_or_says_it_does_not():
    """Not a bar -- a smoke that the relabel runs end to end on a real sweep and that
    V = 0 is the identity on it."""
    from scripts_cosim.score_route_b_contention import load_rows

    ds = CORPUS / "ds_00000"
    db = json.loads(TASK_TYPES.read_text())
    rows = load_rows(ds, "rtt")[:2000]
    ctx = build_state_context(ds, db, arrival_rate=0.46)

    assert shaped_rows(rows, ctx, 0.0) == [(p, float(v)) for p, v in rows]
    shaped = shaped_rows(rows, ctx, 1.0)
    assert all(s >= r - 1e-9 for (_p, s), (_q, r) in zip(shaped, rows)), (
        "the shaped term must never reduce a plan's cost"
    )


# --------------------------------------------------------------------------
# The simulator side of the clock: HEROSIM_BACKLOG_DRAIN_TABLE
#
# These run the real Platform.seed_virtual_warmup, not a reimplementation, because the
# whole point of the knob is that a corpus generated with it is on the clock the label
# assumes. A drift between the two would be invisible everywhere else.
# --------------------------------------------------------------------------


def _seed_one_platform(monkeypatch, table_path=None, count=5):
    """Import infrastructure fresh under the requested env and seed one backlog."""
    import importlib
    import sys

    if table_path is None:
        monkeypatch.delenv("HEROSIM_BACKLOG_DRAIN_TABLE", raising=False)
    else:
        monkeypatch.setenv("HEROSIM_BACKLOG_DRAIN_TABLE", str(table_path))
    sys.modules.pop("src.placement.infrastructure", None)
    infra = importlib.import_module("src.placement.infrastructure")

    task_type = {
        "executionTime": {"rpiCpu": 0.0029},
        "coldStartDuration": {"rpiCpu": 0.33},
        "stateSize": {"nofs-dnn1": {"input": 153600, "output": 8000}},
    }
    plat = object.__new__(infra.Platform)
    plat.type = {"shortName": "rpiCpu"}
    plat.virtual_warmup_count = 0
    plat.virtual_warmup_total_time = 0.0
    plat.virtual_warmup_task_type = None
    plat.node = type("N", (), {"storage": type("S", (), {"items": []})(), "network": {}})()
    plat.seed_virtual_warmup(task_type, "dnn1", count)
    return plat.virtual_warmup_total_time, infra


def test_unset_drain_table_reproduces_the_exec_plus_comm_backlog(monkeypatch):
    total, _infra = _seed_one_platform(monkeypatch, None, count=5)
    # cold 0.33 + 5 x (0.0029 + comm), comm computed with the no-storage fallback
    # (100 MB/s, 1 ms), which is what this synthetic node exposes.
    comm = (153600 / (100 * 1024 * 1024) + 0.001) + (8000 / (100 * 1024 * 1024) + 0.001)
    assert total == pytest.approx(0.33 + 5 * (0.0029 + comm))


def test_a_measured_table_reprices_the_backlog_and_only_the_backlog(monkeypatch, tmp_path):
    table = tmp_path / "drain.json"
    table.write_text(json.dumps({"drain_seconds_per_item": {drain_table_key("dnn1", "rpiCpu"): 4.0}}))
    total, _infra = _seed_one_platform(monkeypatch, table, count=5)
    # Cold start is a property of the sandbox, not of a queued item, so it is unchanged.
    assert total == pytest.approx(0.33 + 5 * 4.0)


def test_a_table_missing_the_corpus_type_is_fatal(monkeypatch, tmp_path):
    table = tmp_path / "drain.json"
    table.write_text(json.dumps({"drain_seconds_per_item": {"other|rpiCpu": 4.0}}))
    with pytest.raises(RuntimeError, match="no entry for"):
        _seed_one_platform(monkeypatch, table)


def test_a_missing_table_path_is_fatal_at_import(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="does not exist"):
        _seed_one_platform(monkeypatch, tmp_path / "absent.json")


@pytest.fixture(autouse=True)
def _restore_infrastructure_module():
    """These tests re-import the simulator under a different env; put it back."""
    yield
    import importlib
    import sys

    sys.modules.pop("src.placement.infrastructure", None)
    importlib.import_module("src.placement.infrastructure")


def test_the_clock_applies_on_the_path_the_label_is_built_from(monkeypatch, tmp_path):
    """Where the clock applies was MEASURED, not assumed (2026-09-14).

    The co-sim sweep rebuilds each placement's starting state from the captured snapshot
    through `seed_virtual_warmup` -- the hook. The initial warmup that produces that
    snapshot uses real `create_warmup_tasks` instances, and a surcharge added to their
    service path recorded ZERO applications across a whole dataset generation. This test
    pins the hook that fires and the counters that let a corpus prove it.
    """
    table = tmp_path / "drain.json"
    table.write_text(json.dumps({"drain_seconds_per_item": {"dnn1|rpiCpu": 7.0}}))
    total, infra = _seed_one_platform(monkeypatch, table, count=4)
    assert total == pytest.approx(0.33 + 4 * 7.0)
    # The counters say how much backlog was repriced and by how much.
    assert infra.BACKLOG_SURCHARGE_COUNTERS["applications"] == 4
    assert infra.BACKLOG_SURCHARGE_COUNTERS["seconds"] > 27.0


def test_counters_stay_zero_without_a_table(monkeypatch):
    total, infra = _seed_one_platform(monkeypatch, None, count=4)
    assert infra.BACKLOG_SURCHARGE_COUNTERS["applications"] == 0
