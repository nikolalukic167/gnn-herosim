"""peer_affinity_v1 stage 3: live serving of prefix-conditioned checkpoints.

What is pinned, in the order it can break:

1. The loader refuses a prefix-conditioned checkpoint under any decode but masked_topo,
   refuses a cap rung that disagrees with the sidecar, and otherwise builds the model
   with the sidecar's options attached.
2. The batch-size range follows the decode mode ([2,4] argmax family, [1,16] masked_topo)
   and peer-group batching needs masked_topo.
3. A partial_state_v2 context accepts an empty in-batch peer table with a positive norm
   (a lone live task) and still refuses a zero norm (a peer-less cache dataset).
4. The input-stage rendezvous waits ONLY for peers that are neither placed nor planned.
5. Real engine, reactive arm: a peer batch placed per arrival — where the physics used
   to fail loud on an unplanned peer — now completes and charges the exchange.
6. Real engine, the pin that makes stage 3 a result rather than a hope: the live path
   reproduces the offline read bit for bit on held-out co-sim datasets (graph, plan,
   RTT). Skips when the T1b artefacts are not on this machine.

Run: pipenv run python3 -m pytest tests/test_prefix_serving.py -q
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CKPT = REPO_ROOT / "models/peer-affinity-v1-t1b-gnn-lr2e3-seed1.pt"
REPORT = REPO_ROOT / "simulation_data/peer_affinity_t1b_reports/peer-affinity-v1-t1b-gnn-lr2e3-seed1.json"
CACHE = REPO_ROOT / "simulation_data/graphs_cache_peer_affinity_v1_t1b"
SPLIT = REPO_ROOT / "experiments/peer_affinity_v1_t1b_split.json"
ARM_B0 = REPO_ROOT / "simulation_data/gnn_datasets_dag4_route_b_pilot_v1_arm_b0/ds_00000"

needs_ckpt = pytest.mark.skipif(not CKPT.exists(), reason="T1b seed-1 checkpoint not on this machine")


@pytest.fixture(autouse=True)
def _restore_environment():
    """The live loader ADOPTS contracts into os.environ (partial-state contract, peer
    mass, MP flag, feature layout). monkeypatch only restores keys it touched itself,
    so without this the adopted values leak into every test collected after this file
    (measured: tests/test_tied_dim25cr_features.py and the A1 determinism tests fail
    on a partial_state_v2 contract they never set)."""
    snapshot = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(snapshot)


def _clean_env(monkeypatch):
    for name in ("GNN_DECODE_MODE", "GNN_BATCH_SIZE", "GNN_BATCH_BY_PEER_GROUP", "GNN_PREFIX_ALPHA_KEY",
                 "PARTIAL_STATE_CONTRACT", "PARTIAL_STATE_PEER_MASS", "GNN_DISABLE_MESSAGE_PASSING",
                 "INFERENCE_FEATURE_LAYOUT", "GNN_MP_NODE_EDGES", "GNN_MP_DAG_EDGES",
                 "QUEUE_FEATURE_CONTRACT", "TOPOLOGY_FEATURE_CONTRACT", "NETWORK_GRAPH_CONTRACT",
                 "HEROSIM_WARMTH_PHYSICS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HEROSIM_GNN_DEVICE", "cpu")


# 1. loader ---------------------------------------------------------------------------

@needs_ckpt
def test_loader_refuses_prefix_checkpoint_without_masked_topo(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("GNN_DECODE_MODE", "argmax")
    from src.executesimulation import load_gnn_model
    with pytest.raises(ValueError, match="masked_topo"):
        load_gnn_model(CKPT, space_config=None)


@needs_ckpt
def test_loader_builds_prefix_model_with_sidecar_options(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("GNN_DECODE_MODE", "masked_topo")
    from src.executesimulation import load_gnn_model
    model, _device = load_gnn_model(CKPT, space_config=None)
    opts = model.prefix_serving_options
    assert opts.alpha_key == "2.5" and opts.allow_replica_reuse and opts.relax_on_stuck
    assert opts.mp_peer_edges and opts.peer_mass and opts.partial_state_contract == "partial_state_v2"
    assert int(model.partial_state_edge_dim) == 38 and int(model.task_type_onehot_dim) == 4
    # contracts adopted into the environment, as the queue/topology contracts are
    assert os.environ["PARTIAL_STATE_CONTRACT"] == "partial_state_v2"
    assert os.environ["PARTIAL_STATE_PEER_MASS"] == "1"
    assert os.environ["GNN_DISABLE_MESSAGE_PASSING"] == "0"


@needs_ckpt
def test_loader_refuses_conflicting_alpha_rung(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("GNN_DECODE_MODE", "masked_topo")
    monkeypatch.setenv("GNN_PREFIX_ALPHA_KEY", "2.0")
    from src.policy.gnn.prefix_serving import PrefixServingError, load_prefix_conditioned_gnn
    with pytest.raises(PrefixServingError, match="dag_alpha_key"):
        load_prefix_conditioned_gnn(CKPT)


# 2. batch range + peer-group guard -------------------------------------------------------

def test_batch_range_follows_decode_mode(monkeypatch):
    _clean_env(monkeypatch)
    from src.policy.gnn import scheduler as S
    assert S._gnn_batch_range() == (2, 4)
    monkeypatch.setenv("GNN_BATCH_SIZE", "10")
    with pytest.raises(ValueError, match="ceiling"):
        S._read_gnn_batch_size()
    monkeypatch.setenv("GNN_DECODE_MODE", "masked_topo")
    assert S._gnn_batch_range() == (1, 16)
    assert S._read_gnn_batch_size() == 10


def test_peer_group_batching_requires_masked_topo(monkeypatch):
    _clean_env(monkeypatch)
    from src.policy.gnn import scheduler as S
    assert S._read_batch_by_peer_group() is False
    monkeypatch.setenv("GNN_BATCH_BY_PEER_GROUP", "1")
    with pytest.raises(ValueError, match="masked_topo"):
        S._read_batch_by_peer_group()
    monkeypatch.setenv("GNN_DECODE_MODE", "masked_topo")
    assert S._read_batch_by_peer_group() is True


# 3. v2 context on a lone live task -------------------------------------------------------

def _ctx(peer_pairs, peer_norm):
    from src.policy.tabular.reduced_features import PartialStateContext
    return PartialStateContext(
        node_caps={}, demand={}, node_of={}, task_type_index={}, parents={}, route_hops_bneck={},
        payload_bytes=0.0, transfer_norm=0.0, node_rank={}, ingress_links={}, core_links=frozenset(),
        peer_pairs=peer_pairs, node_exchange={}, peer_norm=peer_norm, cand_nodes={},
        contract="partial_state_v2",
    )


def test_v2_context_accepts_empty_pairs_with_positive_norm_and_refuses_zero_norm(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("PARTIAL_STATE_CONTRACT", "partial_state_v2")
    ctx = _ctx({}, 1.0)
    assert ctx.peer_pairs == {} and ctx.peer_norm == 1.0
    with pytest.raises(ValueError, match="peer_norm"):
        _ctx({}, 0.0)
    with pytest.raises(ValueError, match="peer_norm"):
        _ctx({(0, 1): 5.0, (1, 0): 5.0}, 0.0)


# 4. rendezvous events ---------------------------------------------------------------------

class _FakeOrch:
    def __init__(self, table, tasks):
        self.peer_exchange = table
        self.task_by_id = tasks
        self.minted = []

    def peer_ready_event(self, task_id):
        self.minted.append(task_id)
        return ("event", task_id)


class _FakeNode:
    def __init__(self, orch):
        self.node_name = "node0"
        self.network_map = {}
        self.network = {"bandwidth": 100.0}
        self.orchestrator_ref = orch


class _FakePlat:
    def __init__(self, node):
        self.node = node


class _FakeTask:
    def __init__(self, tid, placed=False, planned: Optional[str] = None):
        self.id = tid
        self.platform = _FakePlat(_FakeNode(None)) if placed else None
        self.planned_node_name = planned


def test_rendezvous_waits_only_for_unplanned_peers(monkeypatch):
    from src.placement.infrastructure import Platform
    table = {0: {1: 1e6, 2: 1e6, 3: 1e6, 9: 1e6}}
    tasks = {0: _FakeTask(0), 1: _FakeTask(1, placed=True), 2: _FakeTask(2, planned="node4"), 3: _FakeTask(3)}
    orch = _FakeOrch(table, tasks)
    platform = Platform.__new__(Platform)
    platform.node = _FakeNode(orch)
    monkeypatch.delenv("HEROSIM_PEER_EXCHANGE", raising=False)
    assert Platform._peer_rendezvous_events(platform, tasks[0]) == []
    monkeypatch.setenv("HEROSIM_PEER_EXCHANGE", "1")
    events = Platform._peer_rendezvous_events(platform, tasks[0])
    # 1 is placed, 2 is planned -> no wait; 3 is unplanned, 9 does not exist yet -> wait
    assert events == [("event", 3), ("event", 9)]
    assert orch.minted == [3, 9]


# 5. real engine: reactive arm on a peer batch --------------------------------------------

@pytest.mark.skipif(not (ARM_B0 / "optimal_result.json").exists(), reason="arm_b0 ds_00000 not on this machine")
def test_reactive_arm_completes_a_peer_batch_via_rendezvous(monkeypatch):
    monkeypatch.setenv("HEROSIM_COSIM_KEEP_ALIVE", "1000000")
    monkeypatch.setenv("HEROSIM_PEER_EXCHANGE", "1")
    monkeypatch.setenv("SIM_FORCE_FULL_STATS", "1")
    monkeypatch.delenv("HEROSIM_DATA_LOCALITY", raising=False)
    from src.executecosimulation import QUEUE_LENGTH, cosim_keep_alive, execute_simulation
    o = json.load(open(ARM_B0 / "optimal_result.json"))
    base = o["config"]["workload"]["events"][0]
    events = []
    for k, t in enumerate(("dnn1", "dnn2", "rf", "cnn")):
        ev = copy.deepcopy(base)
        ev["application"] = {"name": f"nofs-{t}", "dag": {t: []}}
        # staggered arrivals: task 0's input stage can start before task 3 exists
        ev["timestamp"] = 0.0 + 0.5 * k
        events.append(ev)
    cfg = copy.deepcopy(o["config"])
    cfg["infrastructure"].pop("forced_placements", None)
    cfg["workload"] = {"rps": 1, "duration": 2, "events": events,
                       "peer_exchange": [[0, 3, 50e6], [1, 2, 20e6], [0, 2, 5e6]]}
    stats = execute_simulation(cfg, o["sim_inputs"], "kn_network_kn_network", cache_policy="fifo",
                               task_priority="fifo", keep_alive=cosim_keep_alive(),
                               queue_length=QUEUE_LENGTH)["stats"]
    assert int(stats["num_tasks"]) == 4
    assert float(stats["totalPeerExchangeTime"]) >= 0.0
    assert float(stats["totalPeerRendezvousWait"]) >= 0.0
    waits = {tr["taskId"]: tr["peerRendezvousWait"] for tr in stats["taskResults"]}
    # task 0 lists task 3 (arrives 1.5 s later) as a peer: it must have waited for it
    assert waits[0] > 0.0


# 6. real engine: live path == offline read on held-out datasets ---------------------------

@pytest.mark.skipif(
    not (CKPT.exists() and REPORT.exists() and (CACHE / "graphs.pkl").exists() and SPLIT.exists()),
    reason="T1b checkpoint / report / cache not on this machine",
)
def test_live_serving_reproduces_the_offline_read(monkeypatch):
    import pickle
    _clean_env(monkeypatch)
    from scripts_cosim.peer_affinity_live_serve_check import _one_dataset
    ids = json.loads(SPLIT.read_text())["test"][:2]
    dataset_ids = [str(d) for d in pickle.load(open(CACHE / "dataset_ids.pkl", "rb"))]
    graphs = pickle.load(open(CACHE / "graphs.pkl", "rb"))
    report = {e["dataset_id"]: e["decoded_combo"] for e in json.loads(REPORT.read_text())["per_dataset"]}
    for ds in ids:
        if not (REPO_ROOT / "simulation_data" / ds / "optimal_result.json").exists():
            pytest.skip(f"{ds} not on this machine")
        result = _one_dataset({
            "dataset_id": ds, "checkpoint": str(CKPT), "cache_graph": graphs[dataset_ids.index(ds)],
            "report_combo": report[ds], "replay_rtt": None,
        })
        assert result["mismatches"] == [], result["mismatches"]
        assert result["n_batches"] == 1
