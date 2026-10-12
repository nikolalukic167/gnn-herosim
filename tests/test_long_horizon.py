"""long_horizon_s0_v1: the hourly-bin instrument and the L24 mint's concatenation."""
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.placement.hourly_bins import FIELDS, hourly_bins
from scripts_cosim import long_horizon_mint as lhm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))


def _task(disp, done, internal=False, **kw):
    t = SimpleNamespace(dispatched_time=disp, done_time=done, elapsed_time=None, is_internal=internal)
    for attr, _k in FIELDS:
        if attr != "elapsed_time":
            setattr(t, attr, kw.get(attr, 0.0))
    return t


def test_bins_weighted_means_equal_whole_run():
    rng = random.Random(3)
    tasks = []
    for _ in range(2000):
        d = rng.uniform(0, 10_000)
        tasks.append(_task(d, d + rng.expovariate(0.5), queue_time=rng.random(), cold_start_time=rng.random() * 0.1,
                           peer_exchange_time=rng.random() * 3, execution_time=0.06))
    tasks.append(_task(5.0, 6.0, internal=True, queue_time=99.0))
    tasks.append(_task(7.0, None))
    out = hourly_bins(tasks, 3600.0)
    rows = [t for t in tasks if not t.is_internal and t.done_time is not None]
    assert out["n_tasks"] == len(rows) == sum(b["n"] for b in out["bins"])
    assert len(out["bins"]) == 3
    for attr, key in FIELDS:
        whole = sum((t.done_time - t.dispatched_time) if attr == "elapsed_time" else getattr(t, attr) for t in rows) / len(rows)
        binned = sum(b[key] * b["n"] for b in out["bins"] if b["n"]) / out["n_tasks"]
        assert binned == pytest.approx(whole, rel=1e-12)


def test_bins_assignment_in_system_and_stability():
    tasks = [_task(10, 20), _task(3599, 3700), _task(3600, 3601), _task(7300, 7310)]
    out = hourly_bins(tasks, 3600.0)
    b = out["bins"]
    assert [x["n"] for x in b] == [2, 1, 1]
    assert b[0]["in_system_at_end"] == 1  # the 3599 task is still running at 3600
    assert b[0]["last_arrival_done_minus_end_s"] == pytest.approx(100.0)
    assert b[1]["in_system_at_end"] == 0
    assert b[0]["latency"] == pytest.approx((10 + 101) / 2)


def _window(n_groups, t0, seed):
    rng = random.Random(seed)
    events, pairs, t = [], [], t0
    for g in range(n_groups):
        size = rng.randint(1, 4)
        first = len(events)
        for _ in range(size):
            events.append({"timestamp": t + rng.random() * 0.01, "node_name": "client_node0", "peer_group": g,
                           "application": {"name": "nofs-cnn"}, "qos": {}})
        for i in range(first, len(events) - 1):
            pairs.append([i, i + 1, 1e6])
        t += rng.expovariate(1.0)
    order = sorted(range(len(events)), key=lambda i: (events[i]["timestamp"], i))
    new = {o: k for k, o in enumerate(order)}
    return {"events": [events[i] for i in order], "peer_exchange": sorted([min(new[a], new[b]), max(new[a], new[b]), x]
                                                                          for a, b, x in pairs)}


def test_concat_prefix_and_reindex():
    ws = [_window(30, 1.0, s) for s in range(5)]
    horizon = 80.0
    doc, cycles = lhm.concatenate(ws, horizon)
    n0 = len(ws[0]["events"])
    assert doc["events"][:n0] == ws[0]["events"]
    assert [p for p in doc["peer_exchange"] if p[1] < n0] == ws[0]["peer_exchange"]
    ts = [e["timestamp"] for e in doc["events"]]
    assert ts == sorted(ts)
    first = {}
    for e in doc["events"]:
        first.setdefault(e["peer_group"], e["timestamp"])
    assert max(first.values()) < horizon
    groups = [e["peer_group"] for e in doc["events"]]
    assert len(set(groups)) == len(first)
    for i, j, _b in doc["peer_exchange"]:
        assert i < j and doc["events"][i]["peer_group"] == doc["events"][j]["peer_group"]
    assert sum(c["n_tasks_kept"] for c in cycles) == len(doc["events"])


def test_remap_matches_script_rule(tmp_path):
    import json, subprocess, sys
    wl = {"events": [{"node_name": f"client_node{(g * 7 + k) % 20}", "peer_group": g, "timestamp": g + k / 10}
                     for g in range(40) for k in range(3)]}
    src, dst = tmp_path / "a.json", tmp_path / "b.json"
    src.write_text(json.dumps(wl))
    subprocess.run([sys.executable, "scripts_cosim/scale_probe_remap_clients.py", str(src), str(dst), "--clients", "160"],
                   check=True, capture_output=True)
    assert lhm.remap_clients(json.loads(src.read_text()), 160) == json.loads(dst.read_text())


def test_harness_l24_cells_and_grid(monkeypatch):
    import importlib
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "scripts_cosim"))
    import fresh_topo_burst_v1_gate as G
    for k, v in (("WF1_RUNGS", "moderate,heavy"), ("HEROSIM_TRANSFER_MODEL", "pipelined"), ("HEROSIM_REPLICA_RELEASE", "1"),
                 ("HEROSIM_SCALEOUT", "kpa"), ("GATE_FIXED_POLICY_TIME_SCALE", "1.0"), ("HEROSIM_SHARED_AUTOSCALER", "0"),
                 ("R1A_TOPOS", "16900"), ("R1A_WINDOWS", "L24"), ("R1A_ARMS", "cd,ra_gnn_eng_nosplit,ra_gnn_eng_cdxapply"),
                 ("R1A_SEEDS", "1")):
        monkeypatch.setenv(k, v)
    importlib.reload(G)
    cells = G.r1a_tasks(None)
    assert {(c["window"], c["kind"], c["seed"]) for c in cells} == {
        (w, k, s) for w in ("L24moderate", "L24heavy") for k, s in (("cd", 0), ("ra_gnn_eng_nosplit", 1), ("ra_gnn_eng_cdxapply", 1))}
    assert G.LONG_HORIZON["L24heavy"] == ("long_horizon_L24", "wf1_heavy")
    assert not any(w.startswith("L24") for w in G.WF1_LADDER)
    monkeypatch.delenv("R1A_WINDOWS")
    importlib.reload(G)
    assert not any(c["window"].startswith("L24") for c in G.r1a_tasks(None))


def _summaries(tmp_path, gnn_trend_pp, missing=None):
    import json
    import long_horizon_s0_v1_read as R
    for rung in ("moderate", "heavy"):
        for arm in set(R.GNN) | set(R.HAND[rung]):
            if (rung, arm) == missing:
                continue
            bins = []
            for b in range(24):
                lat = 2.0
                if arm in R.GNN:
                    lat = 2.0 * (1 + (-0.05 + gnn_trend_pp / 100.0 * b / 23.0))
                bins.append({"bin": b, "n": 100, "latency": lat})
            s = {"hourlyBins": {"bins": bins}, "num_tasks": 2400}
            (tmp_path / f"cc40s16900__L24{rung}__{arm}_s{R.SEED[arm]}.summary.json").write_text(json.dumps(s))


@pytest.mark.parametrize("trend,missing,want", [(0.5, None, "HORIZON-FLAT"), (-3.0, None, "HORIZON-GNN-GAINS"),
                                                (3.0, None, "HORIZON-HAND-GAINS"), (0.0, ("heavy", "cd"), "INCOMPLETE")])
def test_reader_bar(tmp_path, trend, missing, want, capsys):
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "scripts_cosim"))
    import long_horizon_s0_v1_read as R
    _summaries(tmp_path, trend, missing)
    _sys.argv = ["r", "--out", str(tmp_path), "--json", str(tmp_path / "r.json")]
    R.main()
    import json
    assert json.load(open(tmp_path / "r.json"))["verdict"] == want
