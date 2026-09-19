"""Target concurrency is a cluster parameter, and until 2026-09-14 no gate had ever varied it.

`HEROSIM_QUEUE_LENGTH` sets the Knative autoscaler's target concurrency per replica: scale-up
fires at `ceil(total_queued / Q) - replicas > 0` (`src/policy/knative_network/autoscaler.py:89`)
for EVERY policy in the family, learned arms included, because the value reaches them all
through the same `SimulationPolicy.queue_length`.

Why it matters enough to pin: at the default Q = 100 against platforms that serve one task at a
time, a cluster that is ~98.5 % idle still packs every arm onto one or two replicas per type,
and placement collapses to queue balancing. Every co-sim corpus state was also captured at 100
(`src/executecosimulation.py:1348,1601,2532,2667`), so a live run at another Q is a different
cluster from the one the corpus describes and the result JSON must say which it was.

These tests pin the resolution order and the fact that the resolved value is recorded.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.executesimulation import _resolve_queue_length  # noqa: E402
from src.placement.constants import QUEUE_LENGTH  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("HEROSIM_QUEUE_LENGTH", raising=False)


def test_default_is_the_constant_every_corpus_was_captured_under():
    assert _resolve_queue_length() == QUEUE_LENGTH == 100


def test_env_var_is_honoured(monkeypatch):
    monkeypatch.setenv("HEROSIM_QUEUE_LENGTH", "4")
    assert _resolve_queue_length() == 4


def test_explicit_argument_beats_the_env_var(monkeypatch):
    """`--queue-length` is the CLI path; it must win, or a sweep script and an exported
    variable could disagree silently."""
    monkeypatch.setenv("HEROSIM_QUEUE_LENGTH", "4")
    assert _resolve_queue_length(10) == 10


def test_an_empty_env_var_is_not_zero(monkeypatch):
    """`export HEROSIM_QUEUE_LENGTH=` in a shell sets it to the empty string. Zero would make
    the autoscaler divide by zero; falling back to the default is the only safe reading."""
    monkeypatch.setenv("HEROSIM_QUEUE_LENGTH", "")
    assert _resolve_queue_length() == QUEUE_LENGTH
    monkeypatch.setenv("HEROSIM_QUEUE_LENGTH", "   ")
    assert _resolve_queue_length() == QUEUE_LENGTH


def test_a_non_numeric_value_fails_loudly(monkeypatch):
    """Never silently serve the default when the operator asked for something specific."""
    monkeypatch.setenv("HEROSIM_QUEUE_LENGTH", "lots")
    with pytest.raises(ValueError):
        _resolve_queue_length()


def test_the_gate_sbatch_records_the_resolved_value():
    """The summary is all that survives a gate run (the raw JSON is deleted unless KEEP_RAW=1),
    so if the summary does not carry `queue_length`, a run at a non-default Q is unreadable
    afterwards and looks exactly like a run at 100."""
    sbatch = (
        Path(__file__).resolve().parents[1]
        / "scripts_cosim/datalab/peer_affinity_v1_stage3_live_gate.sbatch"
    ).read_text()
    assert 's["queue_length"] = d.get("queue_length")' in sbatch
    assert "HEROSIM_QUEUE_LENGTH" in sbatch, "the knob is not echoed into the job log"
