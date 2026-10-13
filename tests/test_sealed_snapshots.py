"""uncut_groups_s0_v1 K1 / identity (m): the snapshot hook captures exactly the sealed batches, named before the run."""
import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
import fresh_topo_burst_v1_gate as G  # noqa: E402
from src.placement import live_audit  # noqa: E402


class _Reached(Exception):
    pass


def _sched():
    def boom(*_a, **_k):
        raise _Reached()
    return SimpleNamespace(env=SimpleNamespace(now=1.0), _capture_full_queue_snapshot=boom, _get_valid_replicas=boom, nodes=SimpleNamespace(items=[]))


def test_only_sealed_triggers_are_captured(tmp_path, monkeypatch):
    trig = tmp_path / "t.json"; trig.write_text("[5, 9]")
    monkeypatch.setenv("LIVE_AUDIT_SNAPSHOT_PATH", str(tmp_path / "s.jsonl"))
    monkeypatch.setenv("LIVE_AUDIT_TRIGGER_IDS", str(trig))
    monkeypatch.setenv("LIVE_AUDIT_MIN_BATCH_SIZE", "100")  # the size filter does not apply to a sealed sample
    s = _sched()
    live_audit.maybe_capture_batch_live_audit_snapshot(s, None, [SimpleNamespace(id=6)], "p")  # not sealed: returns quietly
    with pytest.raises(_Reached):
        live_audit.maybe_capture_batch_live_audit_snapshot(s, None, [SimpleNamespace(id=9), SimpleNamespace(id=10)], "p")


def test_trigger_file_must_be_a_list_of_ids(tmp_path, monkeypatch):
    trig = tmp_path / "t.json"; trig.write_text('{"a": 1}')
    monkeypatch.setenv("LIVE_AUDIT_SNAPSHOT_PATH", str(tmp_path / "s.jsonl"))
    monkeypatch.setenv("LIVE_AUDIT_TRIGGER_IDS", str(trig))
    with pytest.raises(ValueError, match="FAIL LOUD"):
        live_audit.maybe_capture_batch_live_audit_snapshot(_sched(), None, [SimpleNamespace(id=1)], "p")


def test_gate_plan_lookup(tmp_path, monkeypatch):
    monkeypatch.delenv("R1A_SNAPSHOT_PLAN", raising=False)
    assert G.sealed_triggers(19461, "g0heavy") is None
    plan = tmp_path / "plan.json"; plan.write_text(json.dumps({"19461__g0heavy": [30, 7], "19462__g0heavy": "x"}))
    monkeypatch.setenv("R1A_SNAPSHOT_PLAN", str(plan))
    assert G.sealed_triggers(19461, "g0heavy") == [7, 30]
    assert G.sealed_triggers(19461, "g1heavy") is None
    with pytest.raises(SystemExit):
        G.sealed_triggers(19462, "g0heavy")


def test_snapshot_estimate_prices_peer_exchange_without_booking_it(monkeypatch):
    """identity (m) failure, 2026-10-13: the fidelity capture's input-stage estimate booked each popped task's peer
    exchange into the fabric's access-class tally (8 counters moved; the trajectory did not). record=False prices only."""
    from src.placement import infrastructure as inf
    monkeypatch.setenv("HEROSIM_PEER_EXCHANGE", "1")
    booked = []
    fabric = SimpleNamespace(access_classes=True, record_exchange=lambda a, b, t: booked.append((a, b, t)))
    peer = SimpleNamespace(platform=None, planned_node_name="n2")
    orch = SimpleNamespace(peer_exchange={1: {2: 1e6}}, task_by_id={2: peer})
    p = inf.Platform.__new__(inf.Platform)
    p.node = SimpleNamespace(node_name="n1", orchestrator_ref=orch, fabric=fabric, network_map={"n2": {"latency": 0.01}})
    monkeypatch.setattr(inf.Platform, "_payload_transfer_time", lambda self, n, b: 0.5, raising=False)
    monkeypatch.setattr(inf, "_AUDIT", None)
    priced = p._peer_exchange_time(SimpleNamespace(id=1), record=False)
    assert booked == [] and priced == pytest.approx(0.51)
    assert p._peer_exchange_time(SimpleNamespace(id=1)) == pytest.approx(0.51) and len(booked) == 1


def _chosen_env(tmp_path, monkeypatch):
    snap = tmp_path / "s.jsonl"
    monkeypatch.setenv("LIVE_AUDIT_SNAPSHOT_PATH", str(snap))
    monkeypatch.setenv("GNN_ARM_KIND", "cd_exactS_ib0")
    return str(snap)


def test_served_plan_round_trips_for_the_sealed_batch_only(tmp_path, monkeypatch):
    """K1(b): the served plan of a captured sealed batch is written after the decision and read back by trigger id."""
    snap = _chosen_env(tmp_path, monkeypatch)
    s = SimpleNamespace(_audit_pending_chosen=(9, 0), _live_audit_policy_name="p")
    t9, t10, t11 = (SimpleNamespace(id=i) for i in (9, 10, 11))
    live_audit.record_chosen(s, [SimpleNamespace(id=4)], [], {}, 5, None)  # another batch: nothing written, still pending
    assert s._audit_pending_chosen == (9, 0) and live_audit.load_chosen(snap) == {}
    live_audit.record_chosen(s, [t9, t10, t11], [t9, t11], {0: (3, 7), 1: (5, 2)}, 1234, True)
    assert s._audit_pending_chosen is None
    rec = live_audit.load_chosen(snap)[9]
    assert rec["chosen"] == [[9, 3, 7], [11, 5, 2]] and rec["deferred"] == [10]
    assert rec["arm"] == "cd_exactS_ib0" and rec["plan_evals"] == 1234 and rec["cap_hit"] is True and rec["snapshot_id"] == 0
    live_audit.record_chosen(s, [t9], [t9], {0: (1, 1)}, 1, False)  # not pending any more: no second record
    assert len(open(live_audit.chosen_path(snap)).read().splitlines()) == 1


def test_snapshots_without_a_sidecar_still_load(tmp_path):
    snap = tmp_path / "old.jsonl"
    snap.write_text(json.dumps({"snapshot_id": 0, "trigger_task_id": 5, "chosen": None, "tasks": []}) + "\n")
    assert live_audit.load_chosen(str(snap)) == {}
    assert json.loads(snap.read_text().splitlines()[0])["chosen"] is None


def test_the_capture_marks_a_sealed_batch_pending_and_unsealed_runs_never_do(tmp_path, monkeypatch):
    trig = tmp_path / "t.json"; trig.write_text("[9]")
    monkeypatch.setenv("LIVE_AUDIT_SNAPSHOT_PATH", str(tmp_path / "s.jsonl"))
    monkeypatch.setenv("LIVE_AUDIT_TRIGGER_IDS", str(trig))
    monkeypatch.setattr(live_audit, "_task_payload", lambda *a: {})
    monkeypatch.setattr(live_audit, "_replicas_by_type_payload", lambda *a: {})
    monkeypatch.setattr(live_audit.snapshot_fidelity, "enabled", lambda: False)
    s = SimpleNamespace(env=SimpleNamespace(now=1.0), _capture_full_queue_snapshot=lambda: {}, nodes=SimpleNamespace(items=[]))
    live_audit.maybe_capture_batch_live_audit_snapshot(s, None, [SimpleNamespace(id=9)], "p")
    assert s._audit_pending_chosen == (9, 0)
    monkeypatch.delenv("LIVE_AUDIT_TRIGGER_IDS")
    s2 = SimpleNamespace(env=SimpleNamespace(now=1.0), _capture_full_queue_snapshot=lambda: {}, nodes=SimpleNamespace(items=[]))
    monkeypatch.setattr(live_audit, "_batch_qualifies", lambda *a: True)
    live_audit.maybe_capture_batch_live_audit_snapshot(s2, None, [SimpleNamespace(id=9)], "p")
    assert getattr(s2, "_audit_pending_chosen", None) is None


def test_the_scheduler_records_the_served_plan_before_the_charge():
    import inspect
    from src.policy.gnn import scheduler
    src = inspect.getsource(scheduler.GNNScheduler._process_task_batch_prefix)
    assert src.index("record_chosen(self, batch_tasks, decodable, placements") < src.index("decision_cost.rate_ms(len(decodable))")


def test_a_retry_batch_led_by_a_captured_trigger_takes_no_slot(tmp_path, monkeypatch):
    """gate_k1 (2026-10-13): 3 cells wrote wanted == written with one sealed id missing; a postponed task's retry led a
    second batch with an already-captured trigger and used the last slot. Only the first batch per trigger is captured."""
    trig = tmp_path / "t.json"; trig.write_text("[9, 20]")
    snap = tmp_path / "s.jsonl"
    monkeypatch.setenv("LIVE_AUDIT_SNAPSHOT_PATH", str(snap))
    monkeypatch.setenv("LIVE_AUDIT_TRIGGER_IDS", str(trig))
    monkeypatch.setenv("LIVE_AUDIT_MAX_SNAPSHOTS", "2")
    monkeypatch.setattr(live_audit, "_task_payload", lambda *a: {})
    monkeypatch.setattr(live_audit, "_replicas_by_type_payload", lambda *a: {})
    monkeypatch.setattr(live_audit.snapshot_fidelity, "enabled", lambda: False)
    s = SimpleNamespace(env=SimpleNamespace(now=1.0), _capture_full_queue_snapshot=lambda: {}, nodes=SimpleNamespace(items=[]))
    for tid in (9, 9, 20):
        live_audit.maybe_capture_batch_live_audit_snapshot(s, None, [SimpleNamespace(id=tid)], "p")
    assert [json.loads(l)["trigger_task_id"] for l in snap.read_text().splitlines()] == [9, 20]
    assert s._audit_pending_chosen == (20, 1)
