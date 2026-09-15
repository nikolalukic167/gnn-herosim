"""Tests for the drainable_debug_v1 D3 target-concurrency read, written before it was run."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts_cosim.drainable_debug_d3_read import (  # noqa: E402
    D3_ARTIFACT_MAX_FRACTION,
    D3_LOW_Q,
    D3_REFERENCE_Q,
    read_rung,
    verdict,
)


def _write(results: Path, rung: int, q: int, latency: float, recorded_q=None):
    d = results / f"dbg_D3_f{rung}_q{q}"
    d.mkdir(parents=True)
    (d / "knative_network.summary.json").write_text(json.dumps({
        "averageElapsedTime": latency, "averageQueueTime": latency - 8.0,
        "scaleEventCount": 1000, "averageOccupation": 0.02, "unusedPlatforms": 88.0,
        "endTime": 100000.0, "queue_length": q if recorded_q is None else recorded_q,
    }))


def test_a_run_whose_recorded_q_disagrees_with_its_directory_fails_loud(tmp_path):
    """The parent lineage lost three reads to env vars that did not reach the simulator. The
    summary records what the run resolved, so the two must be checked against each other."""
    _write(tmp_path, 4000, 4, 20.0, recorded_q=100)
    with pytest.raises(SystemExit) as exc:
        read_rung(tmp_path, 4000, [4])
    assert "did not run at the intended target concurrency" in str(exc.value)


def test_artifact_fires_when_a_low_q_more_than_halves_the_latency(tmp_path):
    for q, lat in ((1, 14.0), (2, 16.0), (4, 19.0), (10, 28.0), (100, 141.0)):
        _write(tmp_path, 2000, q, lat)
    v = verdict(read_rung(tmp_path, 2000, [1, 2, 4, 10, 100]))
    assert v["best_low_q"] == 1
    assert v["fraction_of_reference"] == pytest.approx(14.0 / 141.0)
    assert v["verdict"] == "TARGET-CONCURRENCY-ARTIFACT"


def test_the_regime_is_real_when_the_default_is_already_close(tmp_path):
    for q, lat in ((1, 18.4), (2, 19.8), (4, 21.5), (10, 25.5), (100, 25.95)):
        _write(tmp_path, 4000, q, lat)
    v = verdict(read_rung(tmp_path, 4000, [1, 2, 4, 10, 100]))
    assert v["fraction_of_reference"] == pytest.approx(18.4 / 25.95)
    assert v["verdict"] == "REGIME-IS-REAL"


def test_a_missing_reference_is_void(tmp_path):
    _write(tmp_path, 4000, 1, 18.0)
    assert verdict(read_rung(tmp_path, 4000, [1, 100]))["verdict"] == "VOID"


def test_the_bar_is_the_registered_one():
    assert D3_ARTIFACT_MAX_FRACTION == 0.50
    assert D3_LOW_Q == (1, 2, 4)
    assert D3_REFERENCE_Q == 100
