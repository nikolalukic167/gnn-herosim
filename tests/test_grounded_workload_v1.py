import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
import grounded_workload_v1_mint as M  # noqa: E402
import grounded_workload_v1_read as R  # noqa: E402

TOPOS = list(range(9001, 9013))


def _write(d, topo, window, kind, seed, elapsed, queue=1.0):
    name = f"cc40s{topo}__{window}__{kind}_s{seed}"
    with open(os.path.join(d, name + ".summary.json"), "w") as fh:
        json.dump({"arm": name, "topology": topo, "window": window, "checkpoint_seed": seed,
                   "averageElapsedTime": elapsed, "averageQueueTime": queue}, fh)


def _gate(d, learned_factor, inadmissible=()):
    for t in TOPOS:
        for w in R.WINDOWS:
            q = 9.0 if (t, w) in inadmissible else 1.0
            _write(d, t, w, "reactive", 0, 10.0, q)
            _write(d, t, w, "cd", 0, 10.0)
            for s in R.SEEDS:
                _write(d, t, w, R.PRIMARY, s, 10.0 * learned_factor)


def test_confirmed_when_learned_is_faster_everywhere(tmp_path):
    _gate(str(tmp_path), 0.9)
    c = R.contrast(R.load(str(tmp_path)), R.admissible(R.load(str(tmp_path)), TOPOS), TOPOS, R.PRIMARY, "cd")
    assert c["label"] == "CONFIRMED" and c["median_pct"] == pytest.approx(-10.0)


def test_inadmissible_cells_are_excluded(tmp_path):
    bad = {(t, "g0") for t in TOPOS}
    _gate(str(tmp_path), 1.0, bad)
    for t in TOPOS:
        for s in R.SEEDS:
            _write(str(tmp_path), t, "g0", R.PRIMARY, s, 1.0)  # a huge "win" on an inadmissible cell
    s = R.load(str(tmp_path))
    c = R.contrast(s, R.admissible(s, TOPOS), TOPOS, R.PRIMARY, "cd")
    assert c["label"] == "NOT-SEPARATED" and c["median_pct"] == 0.0


def test_missing_reactive_is_not_admissible(tmp_path):
    _gate(str(tmp_path), 0.9)
    for t in TOPOS[:5]:
        for w in R.WINDOWS:
            os.remove(os.path.join(str(tmp_path), f"cc40s{t}__{w}__reactive_s0.summary.json"))
    s = R.load(str(tmp_path))
    c = R.contrast(s, R.admissible(s, TOPOS), TOPOS, R.PRIMARY, "cd")
    assert c["label"] == "DESIGN-SHORT" and c["n_topologies"] == 7


def test_mint_keeps_rate_order_and_group_peers():
    lib = {"groups": [{"t0_ms": 1000 * i, "offsets_ms": [0, 1, 5][: 1 + i % 3]} for i in range(60)]}
    base = {"rps": 1, "duration": 1, "events": [
        {"timestamp": 2.0 * i, "application": {"name": "nofs-dnn1", "dag": {"dnn1": []}}, "qos": {"name": "m"},
         "node_name": "c0"} for i in range(100)]}
    doc, meta = M.mint(lib, base, seed=1, n_tasks=100)
    ts = [e["timestamp"] for e in doc["events"]]
    assert len(ts) == 100 and ts == sorted(ts)
    assert meta["rate_per_s"] == pytest.approx(100 / (ts[-1] - ts[0]), rel=1e-9)
    for i, j, _b in doc["peer_exchange"]:
        assert doc["events"][i]["peer_group"] == doc["events"][j]["peer_group"]


def test_group_size_makes_exact_groups_at_the_same_task_rate():
    lib = {"groups": [{"t0_ms": 1000 * i, "offsets_ms": [0, 1, 5][: 1 + i % 3]} for i in range(200)]}
    base = {"rps": 1, "duration": 1, "events": [
        {"timestamp": 2.0 * i, "application": {"name": "nofs-dnn1", "dag": {"dnn1": []}}, "qos": {"name": "m"},
         "node_name": "c0"} for i in range(300)]}
    ref, ref_meta = M.mint(lib, base, seed=1, n_tasks=300)
    for g in (8, 12, 16):
        doc, meta = M.mint(lib, base, seed=1, n_tasks=300, group_size=g)
        sizes = {}
        for e in doc["events"]:
            sizes[e["peer_group"]] = sizes.get(e["peer_group"], 0) + 1
        assert sorted(sizes.values(), reverse=True)[0] == g and sum(sizes.values()) == 300
        assert sorted(sizes.values()).count(g) == 300 // g
        assert meta["rate_per_s"] == pytest.approx(ref_meta["rate_per_s"], rel=0.02)
        assert [e["application"]["name"] for e in doc["events"]] == [e["application"]["name"] for e in ref["events"]]
        for i, j, _b in doc["peer_exchange"]:
            assert doc["events"][i]["peer_group"] == doc["events"][j]["peer_group"]
    with pytest.raises(SystemExit):
        M.mint(lib, base, seed=1, n_tasks=300, merge_k=2, group_size=8)
