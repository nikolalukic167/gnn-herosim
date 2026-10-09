"""backlog_corpus_v1: synthetic backlog injection, the seeded label clock, and the in-flight
capture fix (docs/lineages/backlog_corpus_v1.md).

    PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH=. pipenv run python3 \
        -m pytest tests/test_backlog_corpus.py -q
"""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest

from scripts_cosim.drift_label import DriftLabelError, build_state_context
from src.placement import live_audit
from src.placement.live_snapshot_seed import _approx_comm, inject_synthetic_backlog, seeded_backlog_seconds

TASK_TYPE = {
    "executionTime": {"rpiCpu": 1.0, "xavierGpu": 0.5},
    "coldStartDuration": {"rpiCpu": 0.0, "xavierGpu": 0.0},
    "stateSize": {"app": {"input": 0, "output": 0}},
}
DB = {"dnn1": TASK_TYPE}
ITEM = 1.0 + _approx_comm(TASK_TYPE)  # one queued dnn1 on rpiCpu: execution + storage I/O


def _seed_block():
    platforms = [
        {"node_name": "n0", "platform_id": 1, "queue_length": 0, "current_task_remaining": 0.0,
         "comm_remaining": 0.0, "task_type_hint": "dnn1", "queue_drain_seconds": 0.0},
        {"node_name": "n1", "platform_id": 2, "queue_length": 2, "current_task_remaining": 0.0,
         "comm_remaining": 0.0, "task_type_hint": "dnn1", "queue_drain_seconds": 0.0},
        {"node_name": "n2", "platform_id": 3, "queue_length": 0, "current_task_remaining": 0.0,
         "comm_remaining": 0.0, "task_type_hint": "dnn1", "queue_drain_seconds": 0.0},
    ]
    return {
        "platforms": [dict(p) for p in platforms],
        "replicas_by_type": {"dnn1": [dict(p, candidate=True) for p in platforms]},
    }


# --- seeded_backlog_seconds -------------------------------------------------------------

def test_captured_backlog_is_unchanged_without_synthetic_fields():
    spec = {"queue_length": 2, "current_task_remaining": 0.5, "comm_remaining": 0.0}
    assert seeded_backlog_seconds(spec, TASK_TYPE, "rpiCpu") == pytest.approx(0.5 + 2 * ITEM)
    assert seeded_backlog_seconds({"queue_length": 0}, TASK_TYPE, "rpiCpu") is None


def test_synthetic_seconds_add_to_the_captured_clock():
    spec = {"queue_length": 2, "synthetic_backlog_seconds": 7.0, "synthetic_queue_length": 2}
    assert seeded_backlog_seconds(spec, TASK_TYPE, "rpiCpu") == pytest.approx(2 * ITEM + 7.0)
    idle = {"queue_length": 0, "synthetic_backlog_seconds": 3.0, "synthetic_queue_length": 1}
    assert seeded_backlog_seconds(idle, TASK_TYPE, "rpiCpu") == pytest.approx(3.0)
    assert seeded_backlog_seconds(idle, None, "rpiCpu") is None


# --- inject_synthetic_backlog -----------------------------------------------------------

def test_injection_is_deterministic_and_consistent_across_the_seed():
    a, b = _seed_block(), _seed_block()
    ra = inject_synthetic_backlog(a, random.Random(5), mean_seconds=9.0, busy_prob=1.0, task_seconds=4.5)
    rb = inject_synthetic_backlog(b, random.Random(5), mean_seconds=9.0, busy_prob=1.0, task_seconds=4.5)
    assert ra == rb and a == b
    assert len(ra["per_key"]) == 3
    by_key = {f"{p['node_name']}:{p['platform_id']}": p for p in a["platforms"]}
    for spec in a["replicas_by_type"]["dnn1"]:
        twin = by_key[f"{spec['node_name']}:{spec['platform_id']}"]
        assert spec["synthetic_backlog_seconds"] == twin["synthetic_backlog_seconds"] > 0
        assert spec["synthetic_queue_length"] == twin["synthetic_queue_length"] >= 1


def test_rung_zero_injects_nothing():
    block = _seed_block()
    before = json.dumps(block, sort_keys=True)
    record = inject_synthetic_backlog(block, random.Random(1), mean_seconds=0.0, busy_prob=1.0, task_seconds=4.5)
    assert record["per_key"] == {}
    assert json.dumps(block, sort_keys=True) == before


def test_mean_backlog_tracks_the_rung():
    rng = random.Random(0)
    totals = []
    for _ in range(400):
        block = _seed_block()
        rec = inject_synthetic_backlog(block, rng, mean_seconds=12.0, busy_prob=1.0, task_seconds=4.5)
        totals += [s for _k, s in rec["per_key"].values()]
    assert sum(totals) / len(totals) == pytest.approx(12.0, rel=0.1)


# --- the drift label's seeded clock -----------------------------------------------------

def _dataset(tmp_path, block):
    infra = {
        "replica_placements": {"dnn1": [
            {"node_name": p["node_name"], "platform_id": p["platform_id"], "platform_type": "rpiCpu"}
            for p in block["platforms"]
        ]},
        "queue_distributions": {"dnn1": {
            f"{p['node_name']}:{p['platform_id']}": p["queue_length"] + p.get("synthetic_queue_length", 0)
            for p in block["platforms"]
        }},
        "live_snapshot_seed": block,
    }
    (tmp_path / "infrastructure.json").write_text(json.dumps(infra))
    return tmp_path


def test_seeded_clock_prices_exactly_what_the_simulator_replays(tmp_path):
    block = _seed_block()
    inject_synthetic_backlog(block, random.Random(3), mean_seconds=10.0, busy_prob=1.0, task_seconds=4.5)
    ds = _dataset(tmp_path, block)
    ctx = build_state_context(ds, DB, arrival_rate=0.46, task_type_names=["dnn1"], backlog_clock="seeded")
    for p in block["platforms"]:
        assert ctx.backlog_seconds()[p["platform_id"]] == pytest.approx(
            seeded_backlog_seconds(p, TASK_TYPE, "rpiCpu")
        )


def test_a_synthetic_corpus_refuses_the_counts_clock(tmp_path):
    block = _seed_block()
    inject_synthetic_backlog(block, random.Random(3), mean_seconds=10.0, busy_prob=1.0, task_seconds=4.5)
    with pytest.raises(DriftLabelError, match="synthetic backlog"):
        build_state_context(_dataset(tmp_path, block), DB, arrival_rate=0.46, task_type_names=["dnn1"])


def test_counts_clock_is_unchanged_on_a_captured_corpus(tmp_path):
    ctx = build_state_context(_dataset(tmp_path, _seed_block()), DB, arrival_rate=0.46, task_type_names=["dnn1"])
    assert ctx.seeded_backlog is None
    assert ctx.backlog_seconds() == {2: pytest.approx(2 * ITEM)}


# --- in-flight capture ------------------------------------------------------------------

def _platform(now, end, busy=True):
    return SimpleNamespace(
        env=SimpleNamespace(now=now), current_task=object() if busy else None,
        inflight_service_end=end, id=1,
    )


def test_inflight_remaining_reads_the_recorded_service_end():
    assert live_audit.inflight_remaining_seconds(_platform(10.0, 14.5)) == pytest.approx(4.5)
    assert live_audit.inflight_remaining_seconds(_platform(20.0, 14.5)) == 0.0
    assert live_audit.inflight_remaining_seconds(_platform(10.0, None)) is None
    assert live_audit.inflight_remaining_seconds(_platform(10.0, 14.5, busy=False)) is None


def test_capture_mode_defaults_to_legacy_and_rejects_unknown(monkeypatch):
    monkeypatch.delenv(live_audit.INFLIGHT_CAPTURE_ENV, raising=False)
    assert live_audit.inflight_capture_mode() == "legacy"
    monkeypatch.setenv(live_audit.INFLIGHT_CAPTURE_ENV, "bogus")
    with pytest.raises(ValueError):
        live_audit.inflight_capture_mode()


def test_service_end_mode_overrides_the_legacy_capture(monkeypatch):
    legacy = {"current_task_remaining": 0.0, "comm_remaining": 0.001, "cold_start_remaining": 0.0}
    sched = SimpleNamespace(_capture_temporal_state_for_replicas=lambda reps: {"n0:1": legacy})
    node = SimpleNamespace(node_name="n0")
    plat = _platform(10.0, 14.0)
    monkeypatch.setenv(live_audit.INFLIGHT_CAPTURE_ENV, "legacy")
    assert live_audit.temporal_state_of(sched, node, plat) == legacy
    monkeypatch.setenv(live_audit.INFLIGHT_CAPTURE_ENV, "service_end_v1")
    got = live_audit.temporal_state_of(sched, node, plat)
    assert got["current_task_remaining"] == pytest.approx(4.0) and got["comm_remaining"] == 0.0


def test_synthetic_seed_passes_explicit_seconds_and_skips_the_drain_table():
    """A platform type the measured drain table never covered (xavierDla) must still seed:
    the synthetic backlog is priced in seconds, so no per-item clock is consulted."""
    from src.placement.live_snapshot_seed import _seed_platform_state

    calls = []
    plat = SimpleNamespace(
        initialized=SimpleNamespace(triggered=True), type={"shortName": "xavierDla"},
        seed_virtual_warmup=lambda *a, **k: calls.append((a, k)), virtual_warmup_total_time=0.0,
    )
    sim = SimpleNamespace(task_types={"dnn1": TASK_TYPE})
    spec = {"node_name": "n0", "platform_id": 1, "queue_length": 0, "task_type_hint": "dnn1",
            "synthetic_queue_length": 3, "synthetic_backlog_seconds": 12.5}
    _seed_platform_state({("n0", 1): (None, plat)}, sim, spec)
    assert calls == [(( TASK_TYPE, "dnn1", 3), {"total_seconds": 12.5})]
    assert plat.virtual_warmup_total_time == 12.5
    captured = {"node_name": "n0", "platform_id": 1, "queue_length": 2, "task_type_hint": "dnn1"}
    calls.clear()
    _seed_platform_state({("n0", 1): (None, plat)}, sim, captured)
    assert calls[0][1] == {}


def test_legacy_serving_backlog_equals_what_the_cache_reads_for_a_busy_platform(monkeypatch):
    """r1_attribution_v1 ds_02000: the fidelity replay shows backlog 0 where the cache holds 0.55 s on a platform running a task, because a replayed
    ghost is not `platform.current_task`. The cache reads the snapshot's candidate payload (live_audit.temporal_state_of -> the scheduler's
    _capture_temporal_state_for_replicas); serving reads candidate_backlog_seconds -> the same function. Pinned on a real busy platform."""
    from src.placement.live_snapshot_seed import seeded_backlog_seconds
    from src.policy.gnn.scheduler import GNNScheduler

    monkeypatch.delenv(live_audit.INFLIGHT_CAPTURE_ENV, raising=False)
    monkeypatch.setattr(live_audit, "platform_queue_drain_seconds", lambda *a, **k: 0.0)
    monkeypatch.setattr(live_audit, "orchestrator_of", lambda s: None)
    task_type = {"name": "cnn", "executionTime": {"xavierCpu": 0.7055338}, "stateSize": {"nofs-cnn": {"output": 11264}}}
    running = SimpleNamespace(cold_started=False, started_time=100.0, type=task_type, application=SimpleNamespace(type={"name": "nofs-cnn"}))
    plat = SimpleNamespace(id=220, env=SimpleNamespace(now=100.1567), current_task=running, type={"shortName": "xavierCpu"},
                           inflight_service_end=None)
    node = SimpleNamespace(node_name="node4")
    sched = SimpleNamespace(env=plat.env)
    sched._capture_temporal_state_for_replicas = lambda reps: GNNScheduler._capture_temporal_state_for_replicas(sched, reps)
    temporal = live_audit.temporal_state_of(sched, node, plat)
    assert temporal["current_task_remaining"] == pytest.approx(0.7055338 - 0.1567) and temporal["comm_remaining"] > 0.0
    served = live_audit.candidate_backlog_seconds(sched, node, plat)
    spec = {"queue_length": 0, "queue_drain_seconds": 0.0, **{k: temporal[k] for k in ("current_task_remaining", "comm_remaining")}}
    assert served == pytest.approx(seeded_backlog_seconds(spec, task_type, "xavierCpu"))
    assert served == pytest.approx(0.7055338 - 0.1567 + temporal["comm_remaining"])
