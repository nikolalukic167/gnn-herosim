"""peer_affinity_v1: corpus-level guards for HEROSIM_PEER_EXCHANGE, on a real dataset.

(a) Bit-identity: replaying a stored arm_b0 optimal plan reproduces its stored total_rtt
    to 1e-9 with the flag unset AND with the flag set (the trace has no peer table).
(b) The physics fires in a real episode: the same infrastructure with a batch of four
    single-task events and a peer table, forced co-located vs forced remote, differs by
    exactly the sum of hops x bytes / bottleneck + latency over the remote pairs.

Skips (never fails) when the dataset is not on this machine. ~10 s per replay.
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DS = REPO_ROOT / "simulation_data/gnn_datasets_dag4_route_b_pilot_v1_arm_b0/ds_00000"
pytestmark = pytest.mark.skipif(not (DS / "optimal_result.json").exists(), reason="arm_b0 ds_00000 not on this machine")


def _load():
    from src.executecosimulation import cosim_keep_alive, QUEUE_LENGTH  # noqa: F401
    o = json.load(open(DS / "optimal_result.json"))
    return o


def _run(config, sim_inputs):
    from src.executecosimulation import cosim_keep_alive, QUEUE_LENGTH, execute_simulation
    os.environ["SIM_FORCE_FULL_STATS"] = "1"
    return execute_simulation(config, sim_inputs, "determined_determined", cache_policy="fifo",
                              task_priority="fifo", keep_alive=cosim_keep_alive(), queue_length=QUEUE_LENGTH)


def _normalise(plan):
    return {int(k): (int(v[0]), int(v[1])) for k, v in plan.items()}


@pytest.mark.parametrize("flag", [None, "1"])
def test_stored_optimal_replays_bit_identically(monkeypatch, flag):
    monkeypatch.setenv("HEROSIM_COSIM_KEEP_ALIVE", "1000000")
    monkeypatch.delenv("HEROSIM_DATA_LOCALITY", raising=False)
    if flag is None:
        monkeypatch.delenv("HEROSIM_PEER_EXCHANGE", raising=False)
    else:
        monkeypatch.setenv("HEROSIM_PEER_EXCHANGE", flag)
    o = _load()
    config = copy.deepcopy(o["config"])
    config["infrastructure"]["forced_placements"] = _normalise(o["sample"]["placement_plan"])
    result = _run(config, o["sim_inputs"])
    old = o["stats"]["total_rtt"]; new = result["stats"]["total_rtt"]
    assert abs(new - old) <= 1e-9 * max(1.0, abs(old)), (old, new)
    assert result["stats"].get("totalPeerExchangeTime", 0.0) == 0.0


def _batch_workload(o, node_names):
    """Four single-task events (one per type) submitted together, plus a peer table."""
    base = o["config"]["workload"]["events"][0]
    events = []
    for t in ("dnn1", "dnn2", "rf", "cnn"):
        ev = copy.deepcopy(base)
        ev["application"] = {"name": f"nofs-{t}", "dag": {t: []}}
        ev["timestamp"] = 0.0
        events.append(ev)
    return {"rps": 1, "duration": 1, "events": events,
            "peer_exchange": [[0, 1, 5e6], [0, 2, 20e6], [2, 3, 1e6]]}


def test_peer_exchange_fires_in_a_real_episode(monkeypatch):
    monkeypatch.setenv("HEROSIM_COSIM_KEEP_ALIVE", "1000000")
    monkeypatch.delenv("HEROSIM_DATA_LOCALITY", raising=False)
    monkeypatch.setenv("HEROSIM_PEER_EXCHANGE", "1")
    o = _load()
    infra = json.load(open(DS / "infrastructure.json"))
    reps = infra["replica_placements"]
    # one replica per type on ONE node (co-located) vs. spread over distinct nodes where possible
    by_type = {t: {r["node_name"]: r for r in reps[t]} for t in ("dnn1", "dnn2", "rf", "cnn")}
    common = set.intersection(*(set(d) for d in by_type.values()))
    if not common:
        pytest.skip("no node hosts all four types on this dataset")
    home = sorted(common)[0]
    plan_col = {}
    stored = _normalise(o["sample"]["placement_plan"])
    node_index_of = {}
    with open(DS / "placements/placements.jsonl") as fh:  # platform_id -> global node index
        for line in fh:
            for _t, (ni, pid) in json.loads(line)["placement_plan"].items():
                node_index_of[int(pid)] = int(ni)
    for i, t in enumerate(("dnn1", "dnn2", "rf", "cnn")):
        r = by_type[t][home]; plan_col[i] = (node_index_of[r["platform_id"]], r["platform_id"])
    # remote plan: move task 0 (dnn1) to another node if one exists
    others = [n for n in by_type["dnn1"] if n != home]
    if not others:
        pytest.skip("dnn1 has a single hosting node on this dataset")
    away = sorted(others)[0]
    plan_rem = dict(plan_col)
    r0 = by_type["dnn1"][away]; plan_rem[0] = (node_index_of[r0["platform_id"]], r0["platform_id"])

    def run(plan):
        config = copy.deepcopy(o["config"])
        config["workload"] = _batch_workload(o, None)
        config["infrastructure"]["forced_placements"] = plan
        config["infrastructure"]["scheduler"] = {"batch_size": 4, "batch_timeout": 0.1}
        return _run(config, o["sim_inputs"])["stats"]

    s_col = run(plan_col); s_rem = run(plan_rem)
    assert s_col["totalPeerExchangeTime"] == 0.0
    # task 0 now exchanges 5e6 with task 1 and 20e6 with task 2 across (away, home); charged on BOTH ends
    from src.placement.network_fabric import route_links
    lt = infra["link_topology"]; hops = route_links(lt["routes"], away, home)
    bneck = min(float(lt["links"][h]["bandwidth_mbps"]) for h in hops)
    lat = float(infra["network_maps"][home][away])
    per_pair = lambda b: len(hops) * b / (bneck * 1024 * 1024) + lat  # noqa: E731
    expected = 2 * (per_pair(5e6) + per_pair(20e6))
    assert s_rem["totalPeerExchangeTime"] == pytest.approx(expected, rel=1e-9)
    assert s_rem["totalPeerExchangeTime"] > 0.0
