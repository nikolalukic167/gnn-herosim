#!/usr/bin/env python3
"""Unit tests for sweep metric extraction and warmth-physics provenance."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts_cosim.sweep_metrics import (  # noqa: E402
    MetricExtractionError,
    extract_number_array,
    load_metrics,
)
from src.placement.warmth import (  # noqa: E402
    NODE_DISK_V2,
    PLATFORM_REUSE_V1,
    ImplicitWarmthPhysicsError,
    describe_warmth_physics,
    require_explicit_warmth_physics,
)

SEALED = (
    ROOT
    / "simulation_data/normal_sim_sweeps"
    / "contention_v2_873_v5.5_sealed_holdout_20260806/results"
)

# Frozen from the sealed holdout (5 seeds, seed-averaged). MLP wins p99 on all
# four configs; the GNN's offline collision-robustness edge does not appear in
# the live tail either. Guards against silent metric-index drift.
SEALED_P99_MEANS = {
    "balanced_p50": {"knative": 58.3, "gnn": 48.7, "mlp_dim22": 25.0},
    "balanced_p60": {"knative": 36.1, "gnn": 25.3, "mlp_dim22": 24.4},
    "client_heavy_p50": {"knative": 90.6, "gnn": 57.2, "mlp_dim22": 31.6},
    "server_heavy_p50": {"knative": 33.1, "gnn": 25.5, "mlp_dim22": 24.4},
}


def _write_result(tmp_path: Path, payload: dict, name: str = "r.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2))
    return path


def test_quantile_indices_match_statistics_quantiles(tmp_path: Path):
    values = [float(i) for i in range(1, 1001)]
    dist = statistics.quantiles(values, n=100)
    path = _write_result(
        tmp_path,
        {"total_rtt": 12.5, "stats": {"taskResponseTimeDistribution": dist}},
    )
    metrics = load_metrics(path)
    assert metrics["p50"] == pytest.approx(dist[49])
    assert metrics["p90"] == pytest.approx(dist[89])
    assert metrics["p99"] == pytest.approx(dist[98])


def test_missing_total_rtt_fails_loud(tmp_path: Path):
    path = _write_result(tmp_path, {"stats": {"taskResponseTimeDistribution": [1.0]}})
    with pytest.raises(MetricExtractionError, match="missing total_rtt"):
        load_metrics(path)


def test_non_positive_total_rtt_fails_loud(tmp_path: Path):
    path = _write_result(tmp_path, {"total_rtt": 0.0, "stats": {}})
    with pytest.raises(MetricExtractionError, match="non-positive"):
        load_metrics(path)


def test_missing_tail_fails_loud_unless_waived(tmp_path: Path):
    path = _write_result(tmp_path, {"total_rtt": 5.0, "stats": {"averageQueueTime": 1.0}})
    with pytest.raises(MetricExtractionError, match="taskResponseTimeDistribution"):
        load_metrics(path)
    metrics = load_metrics(path, require_tail=False)
    assert metrics["p99"] is None
    assert metrics["total_rtt"] == pytest.approx(5.0)


def test_wrong_quantile_count_fails_loud(tmp_path: Path):
    path = _write_result(
        tmp_path,
        {"total_rtt": 1.0, "stats": {"taskResponseTimeDistribution": [1.0, 2.0, 3.0]}},
    )
    with pytest.raises(MetricExtractionError, match="99 quantile cut points"):
        load_metrics(path)


def test_array_extraction_survives_a_large_leading_blob(tmp_path: Path):
    dist = [float(i) for i in range(1, 100)]
    payload = {
        "total_rtt": 3.0,
        "stats": {
            "taskResults": [{"pad": "x" * 64} for _ in range(60000)],
            "taskResponseTimeDistribution": dist,
        },
    }
    path = _write_result(tmp_path, payload, name="big.json")
    assert path.stat().st_size > 4 << 20
    assert extract_number_array(path, "taskResponseTimeDistribution") == dist


def test_physics_provenance_reports_source():
    assert describe_warmth_physics("node_disk_v2") == {
        "warmth_physics": NODE_DISK_V2,
        "warmth_physics_source": "config",
    }
    assert describe_warmth_physics(None, "node_disk_v2") == {
        "warmth_physics": NODE_DISK_V2,
        "warmth_physics_source": "env",
    }
    implicit = describe_warmth_physics(None, "")
    assert implicit["warmth_physics"] == PLATFORM_REUSE_V1
    assert implicit["warmth_physics_source"] == "default"


def test_strict_mode_rejects_implicit_physics(monkeypatch):
    implicit = {"warmth_physics": PLATFORM_REUSE_V1, "warmth_physics_source": "default"}
    monkeypatch.delenv("HEROSIM_REQUIRE_EXPLICIT_PHYSICS", raising=False)
    require_explicit_warmth_physics(implicit)

    monkeypatch.setenv("HEROSIM_REQUIRE_EXPLICIT_PHYSICS", "1")
    with pytest.raises(ImplicitWarmthPhysicsError, match="was not declared"):
        require_explicit_warmth_physics(implicit)
    require_explicit_warmth_physics(
        {"warmth_physics": NODE_DISK_V2, "warmth_physics_source": "env"}
    )


@pytest.mark.skipif(not SEALED.is_dir(), reason="sealed holdout results not on this host")
@pytest.mark.parametrize("config", sorted(SEALED_P99_MEANS))
def test_sealed_holdout_p99_is_stable(config: str):
    for tag, expected in SEALED_P99_MEANS[config].items():
        paths = sorted(SEALED.glob(f"{config}_s*_{tag}.json"))
        paths = [p for p in paths if not p.name.endswith(".decode_stats.json")]
        assert len(paths) == 5, f"expected 5 seeds for {config}/{tag}, got {len(paths)}"
        p99s = [load_metrics(p)["p99"] for p in paths]
        assert statistics.mean(p99s) == pytest.approx(expected, abs=0.05)
