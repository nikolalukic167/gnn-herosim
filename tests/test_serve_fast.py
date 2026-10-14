"""rp/gnn-serve-fast: the SERVEFAST_* flags change how a serving decision is computed, never what it computes.

  * every flag is off by default;
  * SERVEFAST_COLUMNS: partial_state_columns with the incremental aggregates equals the full rescan BIT FOR BIT at every step of random
    decodes (random asymmetric exchange costs, so a changed float summation order shows), across a rebuild (a second plan on the
    same context) and across a non-extension of the cached prefix;
  * SERVEFAST_BUILD: each memoised helper equals the code it replaces.
The cell-level identity (features, logits and plans of every decision, sim-identical summaries) is measured on datalab, not here.
"""
from __future__ import annotations

import math
import random
from types import SimpleNamespace

import numpy as np
import pytest

from src.placement import serve_fast
from src.placement.four_type_features import platform_extra_columns, replica_types_from_system_state
from src.policy.tabular import feature_builder as fb
from src.policy.tabular.reduced_features import (
    LOAD_SECONDS_CONTRACTS, PARTIAL_STATE_CONTRACT_V4, PARTIAL_STATE_CONTRACT_V5, PartialStateContext,
    krank_node_order, partial_state_columns,
)

ENV = {"PARTIAL_STATE_LOAD_SECONDS": "1", "PARTIAL_STATE_EXCHANGE_SECONDS": "1", "PARTIAL_STATE_PEER_MASS": "1"}


def test_flags_default_off(monkeypatch):
    for name in ("SERVEFAST_BUILD", "SERVEFAST_COLUMNS", "SERVEFAST_ROWS"):
        monkeypatch.delenv(name, raising=False)
    assert not (serve_fast.build_on() or serve_fast.columns_on() or serve_fast.rows_on())
    monkeypatch.setenv("SERVEFAST_COLUMNS", "1")
    assert serve_fast.columns_on() and not serve_fast.build_on()


def _ctx(contract, seed, n_tasks=14, n_nodes=5):
    rng = random.Random(seed)
    nodes = [f"n{i}" for i in range(n_nodes)]
    replicas = [(n, p) for n in nodes for p in (0, 1)]
    tasks = list(range(n_tasks))
    caps = {n: rng.choice([math.inf, 6.0, 9.0]) for n in nodes}
    hop = {n: float(i) for i, n in enumerate(nodes)}
    pairs = {}
    for _ in range(3 * n_tasks):
        i, j = rng.sample(tasks, 2)
        pairs[(i, j)] = rng.uniform(0.3, 40.0)
    ex = {(a, b): ((0.0, 0.0) if a == b else (rng.uniform(1e-3, 3.0), rng.uniform(0.01, 0.4))) for a in nodes for b in nodes}
    links = {(t, n): tuple(rng.sample(["l0", "l1", "l2", "c0", "c1"], rng.randint(0, 3))) for t in tasks for n in nodes}
    extra = {}
    if contract in LOAD_SECONDS_CONTRACTS:
        extra = dict(backlog_s={r: rng.uniform(0, 30) for r in replicas},
                     service_s={(t, r): rng.uniform(0.5, 9.0) for t in tasks for r in replicas})
    ctx = PartialStateContext(
        node_caps=caps, demand={(t, r): rng.uniform(0.2, 3.0) for t in tasks for r in replicas},
        node_of={r: r[0] for r in replicas}, task_type_index={t: rng.randrange(4) for t in tasks},
        parents={t: [] for t in tasks},
        route_hops_bneck={(a, b): ((0.0, math.inf) if a == b else (1.0, 100.0)) for a in nodes for b in nodes},
        payload_bytes=0.0, transfer_norm=0.0, node_rank=krank_node_order(caps, hop, contract=contract),
        ingress_links=links, core_links=frozenset({"c0", "c1"}), peer_pairs=pairs, peer_norm=1.0, node_exchange=ex,
        cand_nodes={t: list(nodes) for t in tasks}, contract=contract, **extra,
    )
    return ctx, replicas, tasks, rng


def _decode(ctx, replicas, tasks, rng, monkeypatch, steps_check):
    order = list(tasks)
    rng.shuffle(order)
    committed = {}
    for t in order:
        monkeypatch.delenv("SERVEFAST_COLUMNS", raising=False)
        want = partial_state_columns(ctx, t, replicas, committed)
        monkeypatch.setenv("SERVEFAST_COLUMNS", "1")
        got = partial_state_columns(ctx, t, replicas, committed)
        assert got.tobytes() == want.tobytes(), (t, len(committed))
        steps_check.append(1)
        committed[t] = rng.choice(replicas)


@pytest.mark.parametrize("contract", [PARTIAL_STATE_CONTRACT_V5, PARTIAL_STATE_CONTRACT_V4])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_incremental_columns_are_bit_identical(contract, seed, monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    ctx, replicas, tasks, rng = _ctx(contract, seed)
    n = []
    _decode(ctx, replicas, tasks, rng, monkeypatch, n)  # first plan: pure extension
    _decode(ctx, replicas, tasks, rng, monkeypatch, n)  # second plan on the same context: rebuild
    # a committed set that is not an extension of the cached prefix (the same keys, a changed placement)
    committed = {t: rng.choice(replicas) for t in tasks[:8]}
    monkeypatch.setenv("SERVEFAST_COLUMNS", "1")
    partial_state_columns(ctx, tasks[8], replicas, committed)
    committed[tasks[0]] = next(r for r in replicas if r != committed[tasks[0]])
    got = partial_state_columns(ctx, tasks[9], replicas, committed)
    monkeypatch.delenv("SERVEFAST_COLUMNS")
    assert got.tobytes() == partial_state_columns(ctx, tasks[9], replicas, committed).tobytes()
    assert len(n) == 2 * len(tasks)


def test_shared_fate_fast_equals_original():
    rng = random.Random(5)
    infos = []
    pos = 0
    for node in range(6):
        for _ in range(rng.randint(1, 4)):
            trig = rng.random() < 0.5
            infos.append(SimpleNamespace(node_name=f"n{node}", position=pos,
                                         platform=SimpleNamespace(initialized=SimpleNamespace(triggered=trig))))
            pos += 1
    assert fb._shared_fate_fast(infos) == fb._shared_fate_by_position(infos)


def test_platform_extra_fast_equals_original():
    rng = random.Random(6)
    class _Id:
        def __init__(self, i):
            self.id = i

    nodes = [_Id(i) for i in range(4)]
    plats = [_Id(p) for p in range(3)]
    infos = [SimpleNamespace(node_id=n.id, platform_id=p.id) for n in nodes for p in plats]
    replicas = {t: {(rng.choice(nodes), rng.choice(plats)) for _ in range(rng.randint(0, 6))} for t in ("dnn1", "dnn2", "rf", "cnn")}
    state = SimpleNamespace(replicas=replicas)
    want = platform_extra_columns(replica_types_from_system_state(state, [(i.node_id, i.platform_id) for i in infos]))
    got = fb._platform_extra_fast(state, infos)
    assert got.dtype == want.dtype and got.shape == want.shape and got.tobytes() == want.tobytes()


def test_target_concurrency_helper_and_memo():
    priors = {
        "dnn1": {"platforms": ["a", "b"], "executionTime": {"a": 2.0, "b": 4.0}},
        "dnn2": {"platforms": ["a"], "executionTime": {"a": 6.0}},
    }
    # platform a: min exec per task type 2.0 and 6.0 -> avg_min 4.0; this platform's dnn1 time 2.0 -> 5 * 4 / 2
    assert fb._target_concurrency_uncached("a", priors) == 5.0 * (4.0 / 2.0)
    assert fb._target_concurrency_uncached("z", priors) == 5.0
    assert fb._target_concurrency_uncached("a", None) == 5.0
    assert fb._target_concurrency_memo("a", priors) == fb._target_concurrency_uncached("a", priors)
    priors2 = {"dnn1": {"platforms": ["a"], "executionTime": {"a": 8.0}}}
    assert fb._target_concurrency_memo("a", priors2) == fb._target_concurrency_uncached("a", priors2)
