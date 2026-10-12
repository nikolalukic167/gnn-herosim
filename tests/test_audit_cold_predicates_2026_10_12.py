"""Audit 2026-10-12 (declared-slate cold-cost defect, gate-tools section "2026-10-12"): pin, at their source on this
branch, the three cold-start predicates that the record says disagree, and the warmth-contract hole in the loader.

  * the simulation charges table cold unless the platform's PREVIOUS task has the same type (warmth.sandbox_is_warm,
    applied in infrastructure.Platform.run);
  * CD's `incoming_cold_start_time` follows that rule, except that it charges 0 whenever the platform has a
    `current_task` (gate-tools "second, smaller deviation"; under HEROSIM_REPLICA_RELEASE=1 `current_task` is set during
    compute at infrastructure.py:1857 and cleared at :1932);
  * the snapshot form `expected_completion_from_snapshot_candidate` charges 0 whenever the payload says
    `initialized` -- a platform-level event, not a previous-task predicate. It is the formula the declared slate
    (src/placement/declared_slate.py on rp/slate-fix, not on this branch) names as its source, so this is where the
    served-slate mispricing originates on this branch;
  * `check_checkpoint_corpus_compatibility` only guards warmth when the sidecar's `corpus` block carries
    `warmth_physics`. The accel g4 and slate-fix-v1 sidecars (datalab models/) carry no such key, so a run under the
    other warmth physics (~100x RTT) would load without error. Pinned here so a future contract key is a visible change.

These tests describe current behaviour; they are not a physics change.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.placement.scheduling_cost import (
    expected_completion_from_snapshot_candidate,
    incoming_cold_start_time,
)
from src.placement.warmth import sandbox_is_warm

COLD = 38.0


def _task(type_name: str):
    return SimpleNamespace(type={"name": type_name, "coldStartDuration": {"xavierCpu": COLD}})


def _platform(previous=None, current=None):
    return SimpleNamespace(type={"shortName": "xavierCpu"}, previous_task=previous, current_task=current)


def test_cd_cost_follows_the_simulation_when_the_platform_is_idle():
    incoming = _task("cnn")
    assert incoming_cold_start_time(incoming, _platform()) == COLD
    assert sandbox_is_warm(_platform(), incoming) is False
    same = _platform(previous=_task("cnn"))
    assert incoming_cold_start_time(incoming, same) == 0.0 and sandbox_is_warm(same, incoming) is True
    other = _platform(previous=_task("dnn1"))
    assert incoming_cold_start_time(incoming, other) == COLD and sandbox_is_warm(other, incoming) is False


def test_cd_cost_charges_no_cold_on_a_busy_platform_but_the_simulation_would():
    """gate-tools 2026-10-12, second deviation: `current_task` set, previous task of another type."""
    incoming = _task("cnn")
    busy = _platform(previous=_task("dnn1"), current=_task("dnn1"))
    assert incoming_cold_start_time(incoming, busy) == 0.0
    assert sandbox_is_warm(busy, incoming) is False  # the simulation charges COLD here


def test_snapshot_form_prices_cold_by_initialized_not_by_previous_task():
    """The payload says the platform is `initialized` but its previous task (if any) is of another type: the
    simulation charges COLD, the snapshot formula charges 0. Absent `initialized` raises KeyError since audit m3 (rp/slate-fix
    73d19ebf); it used to default to warm."""
    base = {"execution_time": 1.0, "cold_start_time": COLD, "network_latency": 0.0}
    assert expected_completion_from_snapshot_candidate({**base, "initialized": True}, 0, 0) == 1.0
    assert expected_completion_from_snapshot_candidate({**base, "initialized": False}, 0, 0) == 1.0 + COLD
    with pytest.raises(KeyError):  # audit m3: key absent is an error, no longer "warm"
        expected_completion_from_snapshot_candidate(dict(base), 0, 0)


def test_corpus_compat_does_not_guard_warmth_when_the_sidecar_omits_it(tmp_path: Path, monkeypatch):
    from src.executesimulation import check_checkpoint_corpus_compatibility

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"")
    sidecar_without_warmth = {"corpus": {"collections": ["x"], "n_datasets": 1}}
    ckpt.with_suffix(".contract.json").write_text(json.dumps(sidecar_without_warmth))
    space = {"nodes": {"client_nodes": {"count": 160}, "server_nodes": {"count": 24}}}
    monkeypatch.setenv("HEROSIM_WARMTH_PHYSICS", "platform_reuse_v1")
    check_checkpoint_corpus_compatibility(ckpt, "m.pt", space)  # pinned: no error, the hole is open

    ckpt.with_suffix(".contract.json").write_text(
        json.dumps({"corpus": {"collections": ["x"], "n_datasets": 1, "warmth_physics": "node_disk_v2"}}))
    with pytest.raises(ValueError, match="warmth physics"):
        check_checkpoint_corpus_compatibility(ckpt, "m.pt", space)
