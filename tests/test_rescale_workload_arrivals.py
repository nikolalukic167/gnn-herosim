"""The arrival rescaler: stretching and truncation must move the rate and nothing else.

Used by drainable_regime_v1's S0 load ladder (docs/lineages/drainable_regime_v1.md), whose
S5 control asserts the overloaded rung still reproduces the overloaded regime -- which is only
meaningful if the rescale provably touches timestamps and the event count, never the payloads.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "scripts_cosim/rescale_workload_arrivals.py"


def make_trace(path: Path, n: int = 40) -> dict:
    events = [{"timestamp": i * 0.5,
               "application": {"name": "nofs-dnn1", "dag": {"dnn1": []},
                               "demand_scale": {"dnn1": 1.0 + i / 100.0}},
               "qos": {"name": "medium"}, "node_name": f"client_node{i % 4}",
               "peer_group": i // 10} for i in range(n)]
    d = {"rps": 150, "duration": 20.0, "events": events,
         "peer_exchange": [[0, 5, 1e8], [3, 9, 2e8], [11, 19, 3e8], [12, 35, 4e8]],
         "peer_augmentation": {"seed": 7300, "group_size": 10}}
    path.write_text(json.dumps(d))
    return d


def run(args: list) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOL)] + [str(a) for a in args],
                          capture_output=True, text=True, cwd=ROOT)


def test_stretch_scales_timestamps_and_leaves_payloads_identical(tmp_path: Path) -> None:
    src, dst = tmp_path / "in.json", tmp_path / "out.json"
    before = make_trace(src)
    assert run(["--workload", src, "--factor", 1000, "--output", dst]).returncode == 0
    after = json.loads(dst.read_text())
    assert [e["timestamp"] for e in after["events"]] == [
        e["timestamp"] * 1000 for e in before["events"]]
    for a, b in zip(after["events"], before["events"]):
        assert a["application"] == b["application"]
        assert a["node_name"] == b["node_name"] and a["peer_group"] == b["peer_group"]
    assert after["peer_exchange"] == before["peer_exchange"]
    r = after["arrival_rescale"]
    assert r["rate_before_per_s"] == r["rate_after_per_s"] * 1000
    assert r["truncation"] is None


def test_truncation_drops_peer_pairs_that_reach_past_the_cut(tmp_path: Path) -> None:
    src, dst = tmp_path / "in.json", tmp_path / "out.json"
    make_trace(src)
    assert run(["--workload", src, "--factor", 2, "--output", dst,
                "--max-events", 20]).returncode == 0
    after = json.loads(dst.read_text())
    assert len(after["events"]) == 20
    # [12, 35] reaches past the cut and goes; [11, 19] is wholly inside and stays.
    assert after["peer_exchange"] == [[0, 5, 1e8], [3, 9, 2e8], [11, 19, 3e8]]
    assert after["arrival_rescale"]["truncation"]["peer_pairs_before"] == 4
    assert after["arrival_rescale"]["truncation"]["peer_pairs_after"] == 3


def test_truncation_must_not_cut_a_peer_group_in_half(tmp_path: Path) -> None:
    src, dst = tmp_path / "in.json", tmp_path / "out.json"
    make_trace(src)
    p = run(["--workload", src, "--factor", 2, "--output", dst, "--max-events", 25])
    assert p.returncode != 0
    assert "not a multiple of" in (p.stdout + p.stderr)
    assert not dst.exists()


def test_rejects_a_cut_longer_than_the_trace_and_a_nonpositive_factor(tmp_path: Path) -> None:
    src, dst = tmp_path / "in.json", tmp_path / "out.json"
    make_trace(src)
    assert run(["--workload", src, "--factor", 2, "--output", dst,
                "--max-events", 400]).returncode != 0
    assert run(["--workload", src, "--factor", 0, "--output", dst]).returncode != 0


def test_truncated_ids_stay_aligned_for_the_corpus_bridge(tmp_path: Path) -> None:
    """A truncated trace must still let batch ids 0..N-1 form whole peer groups."""
    src, dst = tmp_path / "in.json", tmp_path / "out.json"
    make_trace(src)
    assert run(["--workload", src, "--factor", 100, "--output", dst,
                "--max-events", 30]).returncode == 0
    after = json.loads(dst.read_text())
    assert len(after["events"]) % 10 == 0
    assert max(max(p[0], p[1]) for p in after["peer_exchange"]) < len(after["events"])
