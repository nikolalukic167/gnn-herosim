"""Steady-state summary keys of fresh_topo_burst_v1_gate (expert's ruling 2026-10-12): mean latency after a warm-up cut, decile
means in dispatch order, and the compact per-task (dispatchedTime, doneTime) file."""
import importlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts_cosim"))


def _gate():
    return importlib.import_module("fresh_topo_burst_v1_gate")


def _rows(n=1000):
    # task i dispatched at 1.0 * i s with latency 10 s for the first 100 tasks (the cold opening), 1 s after
    rows = [{"taskId": i, "dispatchedTime": float(i), "doneTime": float(i) + (10.0 if i < 100 else 1.0)} for i in range(n)]
    rows.append({"taskId": -1, "dispatchedTime": 0.0, "doneTime": 1e6})  # a non-workload row must not count
    return rows[::-1]  # input order must not matter


def test_steady_state_cuts_and_deciles():
    s = _gate().steady_state_latency(_rows())
    assert s["after_360s"] == {"T_s": 360.0, "n": 640, "mean": 1.0}
    p5 = s["after_p5_dispatch"]
    assert p5["T_s"] == 49.0 and p5["n"] == 951 and abs(p5["mean"] - (51 * 10 + 900) / 951) < 1e-12
    assert s["decile_mean"] == [10.0] + [1.0] * 9
    assert sum(s["decile_mean"]) / 10 == (100 * 10 + 900) / 1000  # equal deciles: their mean is averageElapsedTime's quantity
    assert _gate().steady_state_latency([]) is None and _gate().steady_state_latency(None) is None


def test_task_times_file_round_trips(tmp_path):
    p = str(tmp_path / "cell.tasks.npy")
    assert _gate().write_task_times(_rows(), p) == 1000
    a = np.load(p)
    assert a.dtype == np.float32 and a.shape == (1000, 2)
    assert (np.diff(a[:, 0]) >= 0).all() and a[0].tolist() == [0.0, 10.0]
    assert not (tmp_path / "cell.tasks.npy.partial").exists()
    assert _gate().write_task_times([], str(tmp_path / "none.tasks.npy")) is None


def test_summary_carries_python_env_and_fingerprint_additively():
    """run_provenance.python_env / env_fingerprint are copied into the summary next to `code`; every earlier key keeps its source line."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "scripts_cosim" / "fresh_topo_burst_v1_gate.py").read_text()
    assert 'out["code"] = (doc.get("run_provenance") or {}).get("code")' in src
    assert 'out["python_env"] = (doc.get("run_provenance") or {}).get("python_env")' in src
    assert 'out["env_fingerprint"] = (doc.get("run_provenance") or {}).get("env_fingerprint")' in src
