"""drainable_regime_v1 S0 read -- the registered bars, on synthetic screens.

Written before the screen's data exists (docs/lineages/drainable_regime_v1.md). Each test
builds summaries and audit snapshots on disk and runs the read end to end as a subprocess,
so the bars, the control and the refusals are exercised through the real entry point.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
READ = ROOT / "scripts_cosim/drainable_regime_s0_read.py"
FACTORS = [300.0, 1000.0, 2000.0, 4000.0]


def write_summary(results: Path, policy: str, factor: float, *, total_rtt: float,
                  peer: float, rendezvous: float, queue_share_pct: float) -> None:
    results.mkdir(parents=True, exist_ok=True)
    elapsed = 1000.0
    (results / f"{policy}_f{int(factor)}.summary.json").write_text(json.dumps({
        "arm": policy, "policy_name": policy, "num_tasks": 450729,
        "total_rtt": total_rtt, "totalPeerExchangeTime": peer,
        "totalPeerRendezvousWait": rendezvous,
        "averageElapsedTime": elapsed,
        "averageQueueTime": elapsed * queue_share_pct / 100.0,
        "averageCommunicationsTime": 3.4, "averageExecutionTime": 0.03,
        "averageOccupation": 0.06, "endTime": 1e5, "scaleEventCount": 100,
        "arrival_span_s": 169.5 * factor,
    }))


def write_snapshots(snaps: Path, factor: float, *, n_aligned: int, n_unaligned: int,
                    busy_queue: float) -> None:
    snaps.mkdir(parents=True, exist_ok=True)
    # "aligned" is the bridge's own test: exactly 10 tasks, consecutive ids, first id a
    # multiple of 10. Unaligned batches here start mid-group, which is what a low arrival
    # rate produces when a peer group no longer co-arrives inside the batch window.
    lines = []
    for i in range(n_aligned + n_unaligned):
        aligned = i < n_aligned
        first = i * 10 if aligned else i * 10 + 3
        tasks = [{
            "task_id": first + t, "task_type": "dnn1",
            "candidates": [
                {"queue_key": "node0:104", "queue_length": busy_queue},
                {"queue_key": "node1:108", "queue_length": 0},
            ],
        } for t in range(10 if aligned else 3)]
        lines.append(json.dumps({"snapshot_id": i, "time": float(i), "tasks": tasks}))
    (snaps / f"f{int(factor)}.jsonl").write_text("\n".join(lines) + "\n")


def build(tmp_path: Path, rungs: dict) -> tuple:
    """rungs: factor -> dict(peer_share_pct, rendezvous_share_pct, queue_share_pct, busy, aligned)."""
    results, snaps = tmp_path / "results", tmp_path / "snapshots"
    for factor, spec in rungs.items():
        total = 1.0e10
        write_summary(results, "knative_network_batch", factor, total_rtt=total,
                      peer=total * spec["peer_share_pct"] / 100.0,
                      rendezvous=total * spec.get("rendezvous_share_pct", 0.0) / 100.0,
                      queue_share_pct=spec["queue_share_pct"])
        write_summary(results, "knative_network", factor, total_rtt=total * 1.2,
                      peer=total * spec["peer_share_pct"] / 100.0,
                      rendezvous=0.0, queue_share_pct=spec["queue_share_pct"])
        write_snapshots(snaps, factor, n_aligned=spec.get("aligned", 20), n_unaligned=2,
                        busy_queue=spec.get("busy", 5.0))
    return results, snaps


def run_read(tmp_path: Path, results: Path, snaps: Path) -> dict:
    out = tmp_path / "read.json"
    proc = subprocess.run(
        [sys.executable, str(READ), "--results-dir", str(results), "--snapshots-dir", str(snaps),
         "--output", str(out)],
        capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(out.read_text())


OVERLOADED = {"peer_share_pct": 0.01, "queue_share_pct": 99.99, "busy": 590.0, "aligned": 20}


def test_go_when_a_drainable_rung_clears_every_bar(tmp_path: Path) -> None:
    r = run_read(tmp_path, *build(tmp_path, {
        300.0: dict(OVERLOADED),
        1000.0: {"peer_share_pct": 5.0, "queue_share_pct": 95.0, "busy": 100.0, "aligned": 20},
        2000.0: {"peer_share_pct": 35.0, "queue_share_pct": 60.0, "busy": 8.0, "aligned": 20},
        4000.0: {"peer_share_pct": 55.0, "queue_share_pct": 30.0, "busy": 2.0, "aligned": 20},
    }))
    assert r["verdict"] == "GO"
    assert r["passing_factors"] == [2000.0, 4000.0]
    assert r["S5_control_holds"] is True


def test_no_go_when_peer_share_never_reaches_the_bar(tmp_path: Path) -> None:
    r = run_read(tmp_path, *build(tmp_path, {
        300.0: dict(OVERLOADED),
        1000.0: {"peer_share_pct": 1.0, "queue_share_pct": 98.0, "busy": 20.0},
        2000.0: {"peer_share_pct": 9.9, "queue_share_pct": 80.0, "busy": 5.0},
        4000.0: {"peer_share_pct": 19.99, "queue_share_pct": 50.0, "busy": 1.0},
    }))
    assert r["verdict"] == "NO-GO"
    assert r["passing_factors"] == []


def test_rendezvous_wait_blocks_a_rung_that_clears_the_mechanism_bar(tmp_path: Path) -> None:
    """S3: peer cost that is waiting for a partner to arrive is not placement-sensitive."""
    r = run_read(tmp_path, *build(tmp_path, {
        300.0: dict(OVERLOADED),
        1000.0: {"peer_share_pct": 5.0, "queue_share_pct": 95.0, "busy": 50.0},
        2000.0: {"peer_share_pct": 60.0, "queue_share_pct": 20.0, "busy": 3.0,
                 "rendezvous_share_pct": 55.0},
        4000.0: {"peer_share_pct": 70.0, "queue_share_pct": 10.0, "busy": 1.0,
                 "rendezvous_share_pct": 80.0},
    }))
    assert r["verdict"] == "NO-GO"
    assert [x["S1_mechanism"] for x in r["rungs"]][2:] == [True, True]
    assert [x["S3_placement_sensitivity"] for x in r["rungs"]][2:] == [False, False]


def test_out_of_trained_range_queue_blocks_a_rung(tmp_path: Path) -> None:
    """S2: a rung whose queue column is above the cold-corpus maximum cannot be trained on."""
    r = run_read(tmp_path, *build(tmp_path, {
        300.0: dict(OVERLOADED),
        1000.0: {"peer_share_pct": 30.0, "queue_share_pct": 70.0, "busy": 43.0},
        2000.0: {"peer_share_pct": 30.0, "queue_share_pct": 70.0, "busy": 42.0},
        4000.0: {"peer_share_pct": 30.0, "queue_share_pct": 70.0, "busy": 100.0},
    }))
    assert r["passing_factors"] == [2000.0]
    assert r["verdict"] == "GO"


def test_too_few_aligned_snapshots_blocks_a_rung(tmp_path: Path) -> None:
    """S4: a rung whose peer groups never co-arrive cannot host the corpus bridge."""
    r = run_read(tmp_path, *build(tmp_path, {
        300.0: dict(OVERLOADED),
        1000.0: {"peer_share_pct": 30.0, "queue_share_pct": 70.0, "busy": 5.0, "aligned": 11},
        2000.0: {"peer_share_pct": 30.0, "queue_share_pct": 70.0, "busy": 5.0, "aligned": 12},
        4000.0: {"peer_share_pct": 30.0, "queue_share_pct": 70.0, "busy": 5.0, "aligned": 0},
    }))
    assert r["passing_factors"] == [2000.0]


def test_void_when_the_control_rung_does_not_reproduce_the_overloaded_regime(tmp_path: Path) -> None:
    """S5: if the rescale moved anything but the arrival rate, no rung may be read."""
    r = run_read(tmp_path, *build(tmp_path, {
        300.0: {"peer_share_pct": 40.0, "queue_share_pct": 50.0, "busy": 5.0},
        1000.0: {"peer_share_pct": 40.0, "queue_share_pct": 50.0, "busy": 5.0},
        2000.0: {"peer_share_pct": 40.0, "queue_share_pct": 50.0, "busy": 5.0},
        4000.0: {"peer_share_pct": 40.0, "queue_share_pct": 50.0, "busy": 5.0},
    }))
    assert r["verdict"] == "VOID"


def test_missing_summary_refuses_to_read(tmp_path: Path) -> None:
    results, snaps = build(tmp_path, {300.0: dict(OVERLOADED)})
    out = tmp_path / "read.json"
    proc = subprocess.run(
        [sys.executable, str(READ), "--results-dir", str(results), "--snapshots-dir", str(snaps),
         "--output", str(out)], capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode != 0
    assert "not complete, refusing to read" in proc.stderr


def test_percentile_matches_a_known_series() -> None:
    sys.path.insert(0, str(ROOT / "scripts_cosim"))
    from drainable_regime_s0_read import percentile
    series = list(range(1, 11))
    assert percentile(series, 50) == pytest.approx(5.5)
    assert percentile(series, 90) == pytest.approx(9.1)
    assert percentile([], 90) == 0.0
    assert percentile([7.0], 90) == 7.0
