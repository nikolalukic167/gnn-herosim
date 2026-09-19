"""`--skip-events`: disjoint arrival windows out of the production trace.

Every live gate in this programme has read the SAME first 50,000 events of a 450,729-event
trace. `unsaturated_scale_v1` measured the cost: between the two halves of that one window the
arms' median result against reactive swings 19 pp and flips sign, against an effect of 5-8 pp.
`unsaturated_scale_v2` therefore replicates across disjoint windows, and this file pins the
thing that would silently invalidate every trace already on disk:

  * `--skip-events 0` is BYTE-IDENTICAL to omitting the flag — no existing trace moves;
  * a window is rebased so it starts where window 0 starts, or every policy idles through the
    skipped span before its first arrival;
  * `peer_exchange` is re-indexed into the window, and a pair with one endpoint outside is
    DROPPED, never left pointing at whatever task now occupies that index.

Run: pipenv run python3 -m pytest tests/test_rescale_window_offset.py -q
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts_cosim" / "rescale_workload_arrivals.py"
GROUP = 10


def _trace(n_events: int = 120) -> dict:
    """A trace with the shape the real one has: one task per event, peer pairs inside groups."""
    events = [{"timestamp": 0.01 * i,
               "application": {"name": f"app{i % 3}", "dag": {f"t{i % 3}": []},
                               "demand_scale": {f"t{i % 3}": 1.0 + 0.001 * i}},
               "qos": {"name": "medium", "maxDurationDeviation": 15},
               "node_name": "client_node0", "peer_group": i // GROUP}
              for i in range(n_events)]
    pairs = []
    for g in range(n_events // GROUP):
        base = g * GROUP
        pairs.append([base, base + 1, 2.0e8])
        pairs.append([base + 2, base + 3, 1.0e8])
    return {"rps": 150, "duration": 1.0, "events": events, "peer_exchange": pairs}


def _run(tmp_path, src: Path, out_name: str, *extra) -> Path:
    out = tmp_path / out_name
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--workload", str(src), "--factor", "4000",
         "--output", str(out), *extra],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert r.returncode == 0, r.stderr
    return out


def _fail(tmp_path, src: Path, *extra) -> str:
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--workload", str(src), "--factor", "4000",
         "--output", str(tmp_path / "nope.json"), *extra],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert r.returncode != 0, "expected a refusal"
    return r.stdout + r.stderr


@pytest.fixture()
def src(tmp_path) -> Path:
    p = tmp_path / "src.json"
    p.write_text(json.dumps(_trace()))
    return p


def test_skip_zero_is_byte_identical_to_omitting_the_flag(src, tmp_path):
    """THE regression that matters: every trace on disk was built without this flag."""
    a = _run(tmp_path, src, "a.json", "--max-events", "50")
    b = _run(tmp_path, src, "b.json", "--max-events", "50", "--skip-events", "0")
    assert a.read_bytes() == b.read_bytes()


def test_a_window_is_rebased_so_it_starts_where_window_zero_starts(src, tmp_path):
    w0 = json.loads(_run(tmp_path, src, "w0.json", "--max-events", "50").read_text())
    w1 = json.loads(_run(tmp_path, src, "w1.json", "--max-events", "50",
                         "--skip-events", "50").read_text())
    assert w0["events"][0]["timestamp"] == pytest.approx(w1["events"][0]["timestamp"])
    # ...and the window is genuinely a different slice of the trace, not the same one again
    assert w0["events"][0]["application"]["demand_scale"] != \
        w1["events"][0]["application"]["demand_scale"]
    assert len(w1["events"]) == 50
    assert w1["arrival_rescale"]["truncation"]["window"] == [50, 100]
    assert w1["arrival_rescale"]["truncation"]["skip_events"] == 50


def test_the_window_spans_the_same_wall_time_as_window_zero(src, tmp_path):
    """Same event count at the same rescale factor, so the rungs stay comparable."""
    w0 = json.loads(_run(tmp_path, src, "w0.json", "--max-events", "50").read_text())
    w1 = json.loads(_run(tmp_path, src, "w1.json", "--max-events", "50",
                         "--skip-events", "50").read_text())
    assert w0["arrival_rescale"]["span_after_s"] == pytest.approx(
        w1["arrival_rescale"]["span_after_s"], rel=1e-9)


def test_peer_pairs_are_reindexed_into_the_window_and_stay_in_range(src, tmp_path):
    w1 = json.loads(_run(tmp_path, src, "w1.json", "--max-events", "50",
                         "--skip-events", "50").read_text())
    pairs = w1["peer_exchange"]
    assert pairs, "the window must keep its own pairs"
    for p in pairs:
        assert 0 <= p[0] < 50 and 0 <= p[1] < 50
    # the payload column survives re-indexing
    assert all(len(p) == 3 for p in pairs)
    # window 1 holds groups 5-9, each contributing 2 pairs
    assert len(pairs) == 10


def test_a_pair_reaching_outside_the_window_is_dropped_not_misaimed(tmp_path):
    t = _trace(60)
    t["peer_exchange"].append([25, 35, 5.0e7])       # straddles the window boundary at 30
    src = tmp_path / "straddle.json"
    src.write_text(json.dumps(t))
    w = json.loads(_run(tmp_path, src, "w.json", "--max-events", "30",
                        "--skip-events", "30").read_text())
    assert all(0 <= p[0] < 30 and 0 <= p[1] < 30 for p in w["peer_exchange"])
    assert [5, 35] not in w["peer_exchange"] and [-5, 5] not in w["peer_exchange"]


def test_the_refusals(src, tmp_path):
    assert "multiple of --group-size" in _fail(tmp_path, src, "--max-events", "50",
                                               "--skip-events", "7")
    assert "needs --max-events" in _fail(tmp_path, src, "--skip-events", "50")
    assert "exceeds the trace" in _fail(tmp_path, src, "--max-events", "100",
                                        "--skip-events", "100")
    assert "past the trace" in _fail(tmp_path, src, "--max-events", "10",
                                     "--skip-events", "200")
    assert "non-negative" in _fail(tmp_path, src, "--max-events", "10",
                                   "--skip-events", "-10")


def test_windows_are_disjoint_in_the_events_they_carry(src, tmp_path):
    seen = []
    for k in (0, 40, 80):
        w = json.loads(_run(tmp_path, src, f"w{k}.json", "--max-events", "40",
                            "--skip-events", str(k)).read_text())
        seen.append({e["application"]["demand_scale"][next(iter(e["application"]["dag"]))]
                     for e in w["events"]})
    assert not (seen[0] & seen[1]) and not (seen[1] & seen[2]) and not (seen[0] & seen[2])
