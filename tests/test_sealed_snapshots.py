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
