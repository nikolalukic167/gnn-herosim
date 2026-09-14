"""End-to-end tests for the registered drainable_regime_v1 S1 read.

The bars live in the read tool; these tests pin the decisions it makes, so a later edit
that moves a bar fails here rather than silently rewriting a verdict.
"""
import json
import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parent.parent / "scripts_cosim" / "drainable_regime_s1_read.py"
SPAN = 108602.985


def _arm(d: Path, name: str, rtt: float, peer_frac=0.25, end=108681.0):
    (d / f"{name}.summary.json").write_text(json.dumps({
        "total_rtt": rtt, "totalPeerExchangeTime": rtt * peer_frac, "endTime": end,
        "averageQueueTime": 1.0, "averageElapsedTime": 2.0, "arm": name}))


def _config(d: Path, gnn_rtts, mpoff_rtts, kn=1000.0, knb=1100.0, peer_frac=0.25, end=108681.0):
    d.mkdir(parents=True, exist_ok=True)
    _arm(d, "knative_network", kn, peer_frac, end)
    _arm(d, "knative_network_batch", knb, peer_frac, end)
    for i, v in enumerate(gnn_rtts, 1):
        _arm(d, f"gnn_s{i}", v, peer_frac, end)
    for i, v in enumerate(mpoff_rtts, 1):
        _arm(d, f"mpoff_s{i}", v, peer_frac, end)


def _run(cap: Path, unc: Path, out: Path, snapshots=None):
    cmd = [sys.executable, str(TOOL), "--capped-dir", str(cap), "--uncapped-dir", str(unc),
           "--arrival-span-s", str(SPAN), "--output", str(out)]
    if snapshots is not None:
        cmd += ["--snapshots", str(snapshots)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout, json.loads(out.read_text())


def test_gnn_needed_when_every_seed_wins(tmp_path):
    gnn = [100.0 + i for i in range(16)]
    mpoff = [120.0 + i for i in range(16)]
    _config(tmp_path / "cap", gnn, mpoff)
    _config(tmp_path / "unc", gnn, mpoff)
    stdout, res = _run(tmp_path / "cap", tmp_path / "unc", tmp_path / "o.json")
    assert res["capped"]["b1"]["verdict"] == "GNN-NEEDED"
    assert res["capped"]["b1"]["wins"] == 16
    assert res["capped"]["b1"]["pct"] > 0
    assert res["b3_cap_contingent"] is False


def test_pointwise_better_when_every_seed_loses(tmp_path):
    gnn = [120.0 + i for i in range(16)]
    mpoff = [100.0 + i for i in range(16)]
    _config(tmp_path / "cap", gnn, mpoff)
    _config(tmp_path / "unc", gnn, mpoff)
    _, res = _run(tmp_path / "cap", tmp_path / "unc", tmp_path / "o.json")
    assert res["capped"]["b1"]["verdict"] == "POINTWISE-BETTER"
    assert res["capped"]["b1"]["pct"] < 0


def test_tie_when_seeds_split(tmp_path):
    gnn = [100.0 + i for i in range(16)]
    mpoff = [100.0 + i + (5.0 if i % 2 == 0 else -5.0) for i in range(16)]
    _config(tmp_path / "cap", gnn, mpoff)
    _config(tmp_path / "unc", gnn, mpoff)
    _, res = _run(tmp_path / "cap", tmp_path / "unc", tmp_path / "o.json")
    assert res["capped"]["b1"]["verdict"] == "TIE"


def test_b4_failure_voids_the_read(tmp_path):
    gnn = [100.0 + i for i in range(16)]
    mpoff = [120.0 + i for i in range(16)]
    # peer share 1% -- below the 20% bar
    _config(tmp_path / "cap", gnn, mpoff, peer_frac=0.01)
    _config(tmp_path / "unc", gnn, mpoff, peer_frac=0.01)
    stdout, res = _run(tmp_path / "cap", tmp_path / "unc", tmp_path / "o.json")
    assert res["verdict"] == "VOID"
    assert "capped" not in res
    assert "B4 failed" in stdout


def test_b4_failure_on_cooldown_voids_the_read(tmp_path):
    gnn = [100.0 + i for i in range(16)]
    mpoff = [120.0 + i for i in range(16)]
    # endTime 10x the arrival span -> 90% cool-down, above the 5% bar
    _config(tmp_path / "cap", gnn, mpoff, end=SPAN * 10)
    _config(tmp_path / "unc", gnn, mpoff, end=SPAN * 10)
    _, res = _run(tmp_path / "cap", tmp_path / "unc", tmp_path / "o.json")
    assert res["verdict"] == "VOID"


def test_cap_contingent_when_the_two_configs_disagree(tmp_path):
    gnn = [100.0 + i for i in range(16)]
    mpoff = [120.0 + i for i in range(16)]
    _config(tmp_path / "cap", gnn, mpoff)
    _config(tmp_path / "unc", mpoff, gnn)          # verdict flips
    _, res = _run(tmp_path / "cap", tmp_path / "unc", tmp_path / "o.json")
    assert res["b3_cap_contingent"] is True
    assert "CAP-CONTINGENT" in res["verdict"]


def test_b5_marks_out_of_range_but_still_reads(tmp_path):
    gnn = [100.0 + i for i in range(16)]
    mpoff = [120.0 + i for i in range(16)]
    _config(tmp_path / "cap", gnn, mpoff)
    _config(tmp_path / "unc", gnn, mpoff)
    snap = tmp_path / "s.jsonl"
    snap.write_text("\n".join(
        json.dumps({"full_queue_snapshot": {"a:1": 500, "b:2": 0}})
        for _ in range(20)))
    _, res = _run(tmp_path / "cap", tmp_path / "unc", tmp_path / "o.json", snapshots=snap)
    assert res["b5"]["holds"] is False
    assert res["b5"]["dim7_p90"] == 500
    assert "OUT-OF-RANGE" in res["verdict"]


def test_b5_holds_inside_the_trained_range(tmp_path):
    gnn = [100.0 + i for i in range(16)]
    mpoff = [120.0 + i for i in range(16)]
    _config(tmp_path / "cap", gnn, mpoff)
    _config(tmp_path / "unc", gnn, mpoff)
    snap = tmp_path / "s.jsonl"
    snap.write_text("\n".join(
        json.dumps({"full_queue_snapshot": {"a:1": 6, "b:2": 0}})
        for _ in range(20)))
    _, res = _run(tmp_path / "cap", tmp_path / "unc", tmp_path / "o.json", snapshots=snap)
    assert res["b5"]["holds"] is True
    assert "OUT-OF-RANGE" not in res["verdict"]


def test_missing_arm_fails_loud(tmp_path):
    gnn = [100.0 + i for i in range(15)]           # one seed short
    mpoff = [120.0 + i for i in range(16)]
    _config(tmp_path / "cap", gnn, mpoff)
    _config(tmp_path / "unc", gnn, mpoff)
    r = subprocess.run(
        [sys.executable, str(TOOL), "--capped-dir", str(tmp_path / "cap"),
         "--uncapped-dir", str(tmp_path / "unc"), "--arrival-span-s", str(SPAN),
         "--output", str(tmp_path / "o.json")], capture_output=True, text=True)
    assert r.returncode != 0
    assert "FAIL LOUD" in r.stderr or "FAIL LOUD" in r.stdout


def test_b5_is_not_applicable_when_nothing_ever_queues(tmp_path):
    """An empty busy set must NOT pass the bar vacuously -- that defect shipped once."""
    gnn = [100.0 + i for i in range(16)]
    mpoff = [120.0 + i for i in range(16)]
    _config(tmp_path / "cap", gnn, mpoff)
    _config(tmp_path / "unc", gnn, mpoff)
    snap = tmp_path / "s.jsonl"
    snap.write_text("\n".join(
        json.dumps({"full_queue_snapshot": {"a:1": 0, "b:2": 0}}) for _ in range(20)))
    stdout, res = _run(tmp_path / "cap", tmp_path / "unc", tmp_path / "o.json", snapshots=snap)
    assert res["b5"]["holds"] is None
    assert res["b5"]["n_busy"] == 0
    assert "NOT-APPLICABLE" in stdout
