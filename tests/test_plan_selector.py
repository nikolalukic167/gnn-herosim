"""plan_rerank_v1 Phase 1: the plan selector scores one pool identically in training and in serving.

Training reads a pool row from the jsonl and its graph from the pickled cache, with the model the trainer holds in memory; serving
builds the row in memory (tuple plans, live floats, any candidate order) and scores it with the model the live loader rebuilds from
the .pt and its sidecar, whatever the serving process's environment says about message passing. Both must give the same costs to
the bit and the same selected plan. The trainer itself is run end to end on a synthetic cache, twice at one seed (bit-identical
weights), and its checkpoint is served through the loader.
"""
from __future__ import annotations

import copy
import itertools
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from src.policy.plan_rerank import selector as S
from src.policy.plan_rerank.pool import plan_score, step0_branches, teacher_forced_sum

REPO = Path(__file__).resolve().parents[1]
ENVS = ("GNN_DISABLE_MESSAGE_PASSING", "GNN_PLAN_RAW", "GNN_PLAN_RAW_SUM", "GNN_PLAN_RAW_LOCAL", "GNN_MP_NODE_EDGES",
        "GNN_MP_BIPARTITE_EDGE_CONV", "GNN_MP_PEER_EDGES")
PLACEMENT = {0: (0, 10), 1: (0, 11), 2: (1, 20), 3: (1, 21)}
CANDS = {0: [0, 2], 1: [1, 3], 2: [0, 3]}
VOCAB = ["cnn", "dnn1", "dnn2", "rf"]


@pytest.fixture(autouse=True)
def _clean_env():
    saved = {k: os.environ.get(k) for k in ENVS}
    for k in ENVS:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _graph(seed: int = 0) -> Data:
    """tests/test_plan_raw.py's graph: 3 tasks, 4 platforms on 2 nodes, 2 candidates per task."""
    torch.manual_seed(seed)
    n_tasks, n_platforms = 3, 4
    fwd = [(t, n_tasks + p) for t in range(n_tasks) for p in CANDS[t]]
    rows = fwd + [(d, s) for s, d in fwd]
    data = Data()
    data.n_tasks, data.n_platforms = n_tasks, n_platforms
    data.task_features = torch.randn(n_tasks, 3)
    data.task_type_onehot4 = torch.eye(4)[:n_tasks]
    data.platform_features = torch.randn(n_platforms, 14)
    data.edge_index = torch.tensor(rows, dtype=torch.long).t().contiguous()
    fwd_attr = torch.randn(len(fwd), 5)
    data.edge_attr = torch.cat([fwd_attr, fwd_attr])
    data.node_edge_index = torch.tensor([[3, 4, 5, 6], [4, 3, 6, 5]])
    data.peer_edge_index = torch.tensor([[0, 1], [1, 0]])
    data.peer_edge_attr = torch.ones(2, 1)
    keys, meta = {}, {}
    for t in range(n_tasks):
        keys[t] = []
        for p in CANDS[t]:
            k = f"k{t}_{p}"
            keys[t].append(k)
            meta[k] = {"platform_pos": p}
    data.task_logit_to_queue_key = keys
    data.queue_key_to_platform_meta = meta
    data.task_logit_to_placement = {t: [PLACEMENT[p] for p in CANDS[t]] for t in range(n_tasks)}
    data.dataset_id = f"syn/ds_{seed:05d}"
    return data


def _plans():
    return [tuple(PLACEMENT[p] for p in combo) for combo in itertools.product(*(CANDS[t] for t in range(3)))]


def _row(seed: int = 0, n_cands: int = 4):
    g = torch.Generator().manual_seed(100 + seed)
    plans = _plans()[:n_cands]
    cands = []
    for i, plan in enumerate(plans):
        comps = {k: float(torch.rand(1, generator=g)) * 3.0 for k in S.S_COMP_KEYS}
        s = sum(comps[k] for k in S.S_COMP_KEYS if k != "cold_dup")
        cands.append({"plan": plan, "sources": ["argmin_s"] if i == 0 else [f"branch:{i}"], "rtt": s * (1.0 + 0.3 * float(torch.rand(1, generator=g))),
                      "plan_score": float(torch.randn(1, generator=g)) * 5, "step_logit_sum": float(torch.randn(1, generator=g)) * 3,
                      "s": s, "s_comps": comps})
    cands[0]["sources"].append("branch:0")
    opt = min(c["rtt"] for c in cands)
    return {"ds": f"syn/ds_{seed:05d}", "rung": "heavy" if seed % 2 else "moderate", "n_tasks": 3, "opt": opt, "candidates": cands}


def _contract(variant: str, mp_off: bool):
    return {"kind": S.KIND, "lineage": S.LINEAGE, "variant": variant, "disable_message_passing": mp_off,
            "feature_names": list(S.PLAN_FEATURES), "dag_task_type_vocab": VOCAB, "smoke": False, "base_checkpoint_md5": None,
            "gnn": {"task_feature_dim": 3, "platform_feature_dim": 14, "edge_dim": 5, "embedding_dim": 16, "hidden_dim": 16,
                    "num_layers": 2, "dropout": 0.1, "normalize_platform_inputs": False, "task_type_onehot_dim": 4,
                    "mp_peer_edges": True, "mp_bipartite_edge_attr_zero": True, "plan_raw_sum": True, "partial_state_edge_dim": 2}}


def _trained_model(variant: str, mp_off: bool) -> S.PlanSelector:
    torch.manual_seed(7)
    m = S.build_selector(_contract(variant, mp_off))
    opt = torch.optim.SGD(m.parameters(), lr=0.05)
    m.train()
    for seed in range(3):  # a few steps, so the weights are not the init
        cost, _p, cands = m.costs(_graph(seed), _row(seed))
        loss = S.listwise_loss(cost, [c["rtt"] for c in cands], 0.05)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m


@pytest.mark.parametrize("variant", S.VARIANTS)
@pytest.mark.parametrize("mp_off", (False, True))
def test_train_and_serve_score_one_pool_identically(tmp_path, variant, mp_off):
    from src.policy.plan_rerank.selector import load_selector

    model = _trained_model(variant, mp_off)
    ck = tmp_path / "sel.pt"
    torch.save(model.state_dict(), ck)
    ck.with_suffix(".contract.json").write_text(json.dumps(_contract(variant, mp_off)))
    # serving: the loader rebuilds the model; the process's MP flag says the opposite of the sidecar and must not matter
    os.environ["GNN_DISABLE_MESSAGE_PASSING"] = "0" if mp_off else "1"
    served, _c = load_selector(ck, base_checkpoint=None)
    model.eval()
    for seed in range(4):
        graph = _graph(seed)
        live_row = _row(seed)
        live_row["candidates"] = list(reversed(live_row["candidates"]))  # live builds branches first, then argmin-S ties
        train_row = json.loads(json.dumps(_row(seed)))  # what the trainer reads from the pool jsonl
        train_graph = pickle.loads(pickle.dumps(_graph(seed)))  # what the trainer reads from graphs.pkl
        with torch.no_grad():
            c_train, p_train, k_train = model.costs(train_graph, train_row)
            c_serve, p_serve, k_serve = served.costs(graph, live_row)
        assert p_train == p_serve
        assert torch.equal(c_train, c_serve)
        assert S.select(c_train.tolist(), p_train, k_train) == S.select(c_serve.tolist(), p_serve, k_serve)
    assert bool(served.gnn._disable_mp) == mp_off


def test_the_caller_graph_is_not_written():
    model = _trained_model("a", False).eval()
    g = _graph(0)
    before = set(g.keys()) if callable(getattr(g, "keys", None)) else set(g.keys)
    with torch.no_grad():
        model.costs(g, _row(0))
    after = set(g.keys()) if callable(getattr(g, "keys", None)) else set(g.keys)
    assert before == after and getattr(g, "partial_state_edge_attr", None) is None


def test_variant_a_is_s_plus_a_residual():
    m = _trained_model("a", False).eval()
    row = _row(1)
    with torch.no_grad():
        for p in m.head.parameters():
            p.zero_()
        cost, plans, cands = m.costs(_graph(1), row)
    s = torch.tensor([c["s"] for c in cands], dtype=torch.float64)
    assert torch.allclose(cost.double(), (s - s.min()) / max(float(s.min()), S.SIGMA_FLOOR))
    assert cands[S.select(cost.tolist(), plans, cands)]["s"] == float(s.min())


def test_select_breaks_ties_to_argmin_s_then_lowest_plan():
    plans = [((0, 2),), ((0, 1),), ((0, 3),)]
    cands = [{"sources": ["branch:0"]}, {"sources": ["branch:1"]}, {"sources": ["argmin_s"]}]
    assert S.select([1.0, 1.0, 1.0], plans, cands) == 2
    assert S.select([1.0, 1.0, 2.0], plans, cands) == 1


def test_pool_inputs_refuse_a_broken_row():
    row = _row(0)
    row["candidates"][1]["s"] += 1.0
    with pytest.raises(ValueError, match="sum of its components"):
        S.pool_inputs(row)
    row = _row(0)
    row["candidates"].append(copy.deepcopy(row["candidates"][0]))
    with pytest.raises(ValueError, match="twice"):
        S.pool_inputs(row)


def test_loader_refuses_smoke_foreign_features_and_a_different_base(tmp_path):
    from src.policy.plan_rerank.selector import load_selector

    m = S.build_selector(_contract("b", False))
    ck = tmp_path / "sel.pt"
    torch.save(m.state_dict(), ck)
    base = tmp_path / "base.pt"
    base.write_bytes(b"base")
    other = tmp_path / "other.pt"
    other.write_bytes(b"other")
    c = {**_contract("b", False), "smoke": True, "base_checkpoint_md5": S.md5_of(base)}
    ck.with_suffix(".contract.json").write_text(json.dumps(c))
    with pytest.raises(RuntimeError, match="smoke"):
        load_selector(ck, base_checkpoint=base)
    load_selector(ck, base_checkpoint=base, allow_smoke=True)
    with pytest.raises(RuntimeError, match="md5"):
        load_selector(ck, base_checkpoint=other, allow_smoke=True)
    ck.with_suffix(".contract.json").write_text(json.dumps({**c, "feature_names": ["s"]}))
    with pytest.raises(RuntimeError, match="features"):
        load_selector(ck, base_checkpoint=base, allow_smoke=True)


def test_pool_functions_greedy_tie_rule_and_sums():
    tl = [[(0, 2), (0, 1)], [(1, 5), (1, 4)]]

    def score_fn(t, committed):
        if t == 0:
            return torch.tensor([1.0, 1.0])  # a tie: the lower couple (0, 1) wins
        return torch.tensor([2.0, 0.5]) if committed.get(0) == (0, 2) else torch.tensor([0.0, 3.0])

    br = step0_branches(score_fn, tl, [0, 1])
    assert br == [(((0, 2), (1, 5)), 3.0), (((0, 1), (1, 4)), 4.0)]
    assert teacher_forced_sum(score_fn, tl, [0, 1], ((0, 1), (1, 5))) == 1.0
    assert plan_score(score_fn, tl, ((0, 2), (1, 5))) == 1.0 + 2.0


def _synthetic_run(tmp: Path, seed: int) -> Path:
    cache = tmp / "cache"
    cache.mkdir(exist_ok=True)
    graphs = [_graph(s) for s in range(6)]
    with open(cache / "graphs.pkl", "wb") as fh:
        pickle.dump(graphs, fh)
    with open(cache / "dataset_ids.pkl", "wb") as fh:
        pickle.dump([g.dataset_id for g in graphs], fh)
    (cache / "metadata.json").write_text(json.dumps({"dag_task_type_vocab": VOCAB, "candidate_slate": "declared_pruning_v1",
                                                     "partial_state_contract": "partial_state_v5", "inference_feature_layout": "dim22"}))
    (tmp / "train.jsonl").write_text("".join(json.dumps(_row(s)) + "\n" for s in range(4)))
    (tmp / "val.jsonl").write_text("".join(json.dumps(_row(s)) + "\n" for s in (4, 5)))
    (tmp / "split.json").write_text(json.dumps({"train": [f"syn/ds_{s:05d}" for s in range(4)], "val": ["syn/ds_00004", "syn/ds_00005"],
                                                "test": ["syn/ds_09999"], "topologies": {"train": [1], "val": [2], "test": [3]}}))
    (tmp / "base.pt").write_bytes(b"base")
    out = tmp / f"out{seed}"
    env = {**os.environ, "WANDB_MODE": "disabled", "WANDB_RUN_NAME": "prsel-test", "PRSEL_TRAIN_SEED": str(seed),
           "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "PYTHONPATH": str(REPO)}
    r = subprocess.run([sys.executable, str(REPO / "src/policy/plan_rerank/train_selector.py"), "--cache-dir", str(cache),
                        "--pool-train", str(tmp / "train.jsonl"), "--pool-val", str(tmp / "val.jsonl"), "--split", str(tmp / "split.json"),
                        "--base-checkpoint", str(tmp / "base.pt"), "--variant", "a", "--epochs", "2", "--embedding-dim", "16",
                        "--hidden-dim", "16", "--num-layers", "2", "--pools-per-step", "2", "--out-dir", str(out)],
                       env=env, cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return out / "prsel-test.pt"


def test_trainer_is_reproducible_and_its_checkpoint_serves(tmp_path):
    from src.policy.plan_rerank.selector import load_selector

    a = _synthetic_run(tmp_path, 3)
    b = _synthetic_run(tmp_path, 3)
    wa, wb = torch.load(a), torch.load(b)
    assert wa.keys() == wb.keys() and all(torch.equal(wa[k], wb[k]) for k in wa)
    model, contract = load_selector(a, base_checkpoint=tmp_path / "base.pt")
    assert contract["selected_epoch"] in (1, 2) and contract["split"]["sha256"]
    with torch.no_grad():
        cost, plans, cands = model.costs(_graph(4), _row(4))
    assert len(plans) == 4 and torch.isfinite(cost).all()


def test_trainer_refuses_a_test_id(tmp_path):
    (tmp_path / "x").mkdir()
    sys.path.insert(0, str(REPO))
    from src.policy.plan_rerank.train_selector import check_split

    with pytest.raises(SystemExit, match="test ids"):
        check_split([{"ds": "a"}], [], {"train": ["a"], "val": [], "test": ["a"]})
    with pytest.raises(SystemExit, match="not in the split"):
        check_split([], [{"ds": "a"}], {"train": ["a"], "val": [], "test": []})
