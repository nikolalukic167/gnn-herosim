"""End-to-end tests for the drainable_serving_config_v1 read.

The constants in the tool are the registered bars; these tests pin the decisions so a
later edit that moves one fails here rather than silently rewriting a verdict.
"""
import json
import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parent.parent / "scripts_cosim" / "drainable_serving_config_read.py"
SPAN = 108602.985
D_PAIRS = 83788.0


def _arm(d, name, rtt, batches=5948, decoded=50000, pairs=70000, peer_frac=0.25, end=108681.0):
    (d / f"{name}.summary.json").write_text(json.dumps({
        "total_rtt": rtt, "totalPeerExchangeTime": rtt * peer_frac, "endTime": end,
        "averageWaitTime": 1.0, "arm": name,
        "schedulerCounters": {"prefix_batches": batches, "prefix_tasks_decoded": decoded,
                              "prefix_pairs_in_batch": pairs,
                              "peer_group_incomplete_batches": 100}}))


def _cfg(d: Path, gnn, mpoff, kn=1000.0, knb=1100.0, **kw):
    d.mkdir(parents=True, exist_ok=True)
    _arm(d, "knative_network", kn, **kw)
    _arm(d, "knative_network_batch", knb, **kw)
    for i, v in enumerate(gnn, 1):
        _arm(d, f"gnn_s{i}", v, **kw)
    for i, v in enumerate(mpoff, 1):
        _arm(d, f"mpoff_s{i}", v, **kw)


def _run(specs, out, c1="A"):
    cmd = [sys.executable, str(TOOL), "--reference-pairs", str(D_PAIRS),
           "--arrival-span-s", str(SPAN), "--c1-config", c1, "--output", str(out)]
    for s in specs:
        cmd += ["--config", s]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout, json.loads(out.read_text())


def test_c1_does_not_explain_when_nobatch_is_still_far_behind(tmp_path):
    a = tmp_path / "A"
    _cfg(a, [3000.0 + i for i in range(16)], [2900.0 + i for i in range(16)],
         batches=50000, decoded=50000, pairs=0)
    stdout, res = _run([f"A:nobatch:{a}"], tmp_path / "o.json")
    assert res["c1"]["verdict"] == "BATCHING-DOES-NOT-EXPLAIN"
    assert res["configs"]["A"]["c3_gnn"]["verdict"] == "REACTIVE-WINS"


def test_c1_explains_when_nobatch_reaches_parity(tmp_path):
    a = tmp_path / "A"
    _cfg(a, [1050.0 + i for i in range(16)], [1060.0 + i for i in range(16)],
         batches=50000, decoded=50000, pairs=0)
    _, res = _run([f"A:nobatch:{a}"], tmp_path / "o.json")
    assert res["c1"]["verdict"] == "BATCHING-EXPLAINS"


def test_c1_partial_between_the_thresholds(tmp_path):
    a = tmp_path / "A"
    _cfg(a, [1600.0 + i for i in range(16)], [1590.0 + i for i in range(16)],
         batches=50000, decoded=50000, pairs=0)
    _, res = _run([f"A:nobatch:{a}"], tmp_path / "o.json")
    assert res["c1"]["verdict"] == "PARTIAL"


def test_peergroup_config_voids_on_pair_retention(tmp_path):
    """Config B's failure: mean batch fine, in-batch pairs only ~52% of the reference."""
    b = tmp_path / "B"
    _cfg(b, [900.0 + i for i in range(16)], [950.0 + i for i in range(16)],
         batches=10000, decoded=50000, pairs=43400)
    stdout, res = _run([f"B:peergroup:{b}"], tmp_path / "o.json", c1="A")
    assert res["configs"]["B"]["verdict"] == "VOID"
    assert "c2" not in res["configs"]["B"]
    assert res["c1"]["verdict"] == "UNREADABLE"


def test_peergroup_config_voids_on_mean_batch(tmp_path):
    b = tmp_path / "B"
    _cfg(b, [900.0 + i for i in range(16)], [950.0 + i for i in range(16)],
         batches=14000, decoded=50000, pairs=83000)
    _, res = _run([f"B:peergroup:{b}"], tmp_path / "o.json")
    assert res["configs"]["B"]["verdict"] == "VOID"


def test_nobatch_voids_if_it_actually_batched(tmp_path):
    a = tmp_path / "A"
    _cfg(a, [900.0 + i for i in range(16)], [950.0 + i for i in range(16)],
         batches=5000, decoded=50000, pairs=0)
    _, res = _run([f"A:nobatch:{a}"], tmp_path / "o.json")
    assert res["configs"]["A"]["verdict"] == "VOID"


def test_rescued_config_is_named(tmp_path):
    e = tmp_path / "E"
    _cfg(e, [800.0 + i for i in range(16)], [950.0 + i for i in range(16)],
         batches=8000, decoded=50000, pairs=70000)
    _, res = _run([f"E:peergroup:{e}"], tmp_path / "o.json", c1="A")
    assert res["configs"]["E"]["c2"]["verdict"] == "GNN-NEEDED"
    assert res["configs"]["E"]["c3_gnn"]["verdict"] == "LEARNED-WINS"
    assert res["rescued_configs"] == ["E"]


def test_c5_failure_voids_everything(tmp_path):
    a = tmp_path / "A"
    _cfg(a, [3000.0 + i for i in range(16)], [2900.0 + i for i in range(16)],
         batches=50000, decoded=50000, pairs=0, peer_frac=0.01)
    _, res = _run([f"A:nobatch:{a}"], tmp_path / "o.json")
    assert res["verdict"] == "VOID"
    assert "c1" not in res


def test_missing_arm_fails_loud(tmp_path):
    a = tmp_path / "A"
    _cfg(a, [3000.0 + i for i in range(15)], [2900.0 + i for i in range(16)],
         batches=50000, decoded=50000, pairs=0)
    r = subprocess.run(
        [sys.executable, str(TOOL), "--config", f"A:nobatch:{a}", "--reference-pairs",
         str(D_PAIRS), "--arrival-span-s", str(SPAN), "--c1-config", "A",
         "--output", str(tmp_path / "o.json")], capture_output=True, text=True)
    assert r.returncode != 0
    assert "FAIL LOUD" in (r.stdout + r.stderr)
