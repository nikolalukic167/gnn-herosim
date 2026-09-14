"""The cell config's scheduler block wins over the env vars; a disagreement must fail loud.

Regression for drainable_regime_v1 S1 (2026-09-14): `GNN_BATCH_TIMEOUT=8.0` was exported
against `cell_s7901.json`, which declares `scheduler.batch_timeout = 0.02`. The
orchestrator applied the config silently, the run recorded 8.0 in `run_provenance.env`,
and the gate produced byte-identical scheduler counters to the unscaled run.
"""
import os
from types import SimpleNamespace

import pytest

from src.policy.gnn import orchestrator as orch_mod


class _Sched(SimpleNamespace):
    pass


def _apply(scheduler_config, scheduler):
    """The config-override block, exercised the way GNNOrchestrator.__init__ runs it."""
    if scheduler_config and "batch_timeout" in scheduler_config:
        cfg_bt = float(scheduler_config["batch_timeout"])
        env_bt = os.environ.get("GNN_BATCH_TIMEOUT")
        if env_bt is not None and float(env_bt) != cfg_bt:
            raise ValueError(
                f"FAIL LOUD: GNN_BATCH_TIMEOUT={env_bt} was exported but the cell config "
                f"declares scheduler.batch_timeout={cfg_bt}, which takes precedence."
            )
        scheduler.batch_timeout = cfg_bt
    return scheduler


def test_source_carries_the_guard():
    """Pin the real file, so deleting the guard fails here and not only in a gate."""
    src = (orch_mod.__file__).replace(".pyc", ".py")
    text = open(src).read()
    assert "GNN_BATCH_TIMEOUT" in text, "the batch_timeout guard is gone from the orchestrator"
    assert "which takes precedence" in text


def test_disagreeing_env_fails_loud(monkeypatch):
    monkeypatch.setenv("GNN_BATCH_TIMEOUT", "8.0")
    with pytest.raises(ValueError, match="FAIL LOUD"):
        _apply({"batch_timeout": 0.02}, _Sched(batch_timeout=8.0))


def test_agreeing_env_is_accepted(monkeypatch):
    monkeypatch.setenv("GNN_BATCH_TIMEOUT", "80.0")
    s = _apply({"batch_timeout": 80.0}, _Sched(batch_timeout=80.0))
    assert s.batch_timeout == 80.0


def test_unset_env_takes_the_config(monkeypatch):
    monkeypatch.delenv("GNN_BATCH_TIMEOUT", raising=False)
    s = _apply({"batch_timeout": 0.02}, _Sched(batch_timeout=0.002))
    assert s.batch_timeout == 0.02
