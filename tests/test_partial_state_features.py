"""Unit tests for the dim63crk partial-state feature block (B2 of the corrected
stage-2 registration, docs/lineages/route_b_v1/stage2-preregistration.md §2).

partial_state_columns is THE single-source definition of the 38 columns; these
tests pin each §2 column's semantics on hand-built fixtures, the krank one-hot
indexing (whose plan-level sum must reproduce the §9c/§9d krank_cols
construction — that is what puts the pooled closure inside MLP(T1)'s hypothesis
space by construction), the linkrank block, the topological-order invariant
(uncommitted parent -> loud failure, replacing the retracted constant col 32),
and the contract machinery (a sidecar-less dim63crk checkpoint is not served).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.policy.tabular.reduced_features import (  # noqa: E402
    DIM25CR_FEATURE_DIM,
    DIM63CRK_FEATURE_DIM,
    KRANK_FEATURE_DIM,
    KRANK_TYPES,
    KRANK_WIDTH,
    LINKRANK_FEATURE_DIM,
    PARTIAL_STATE_BASE_DIM,
    PARTIAL_STATE_CONTRACT_V1,
    PARTIAL_STATE_FEATURE_DIM,
    PartialStateContext,
    PartialStateContractMismatchError,
    krank_node_order,
    partial_state_columns,
    require_matching_partial_state_contract,
    resolve_partial_state_contract,
    validate_partial_state_contract,
)


def test_layout_arithmetic():
    assert PARTIAL_STATE_FEATURE_DIM == 38
    assert PARTIAL_STATE_BASE_DIM + KRANK_FEATURE_DIM + LINKRANK_FEATURE_DIM == 38
    assert KRANK_FEATURE_DIM == KRANK_WIDTH * KRANK_TYPES == 24
    assert DIM63CRK_FEATURE_DIM == DIM25CR_FEATURE_DIM + PARTIAL_STATE_FEATURE_DIM == 63


def _ctx(**overrides):
    """A two-node diamond fixture: nodes A (cap 4) and B (cap 2); candidates are
    (node, platform) tuples; tasks 0..3 with the diamond DAG 0 -> {1,2} -> 3 and
    types [0, 1, 2, 3]. Fabric: A-B route through one core link."""
    ca1, ca2, cb1 = ("A", 1), ("A", 2), ("B", 1)
    base = dict(
        node_caps={"A": 4.0, "B": 2.0},
        demand={(t, c): 1.0 for t in range(4) for c in (ca1, ca2, cb1)},
        node_of={ca1: "A", ca2: "A", cb1: "B"},
        task_type_index={0: 0, 1: 1, 2: 2, 3: 3},
        parents={0: [], 1: [0], 2: [0], 3: [1, 2]},
        route_hops_bneck={
            ("A", "A"): (0.0, math.inf),
            ("B", "B"): (0.0, math.inf),
            ("A", "B"): (2.0, 100.0),
            ("B", "A"): (2.0, 100.0),
        },
        payload_bytes=100.0,
        transfer_norm=2.0 * 100.0 / 100.0,  # the per-dataset max: hops*payload/bneck
        node_rank={"A": 1, "B": 0},  # B has smaller cap -> lower rank
        ingress_links={
            (t, node): (("A|core0", "B|core0") if node == "B" else ())
            for t in range(4)
            for node in ("A", "B")
        },
        core_links=frozenset(),  # no core-core links in this toy route
    )
    base.update(overrides)
    return PartialStateContext(**base), (ca1, ca2, cb1)


def test_empty_partial_state_is_all_zero_except_krank_and_headroom():
    ctx, (ca1, ca2, cb1) = _ctx()
    out = partial_state_columns(ctx, 0, [ca1, cb1], committed={})
    assert out.shape == (2, 38)
    # occupancy, load, violate all zero; remaining = full headroom (cap-d)/cap
    assert np.all(out[:, 0:4] == 0.0)
    assert out[0, 4] == 0.0 and out[1, 4] == 0.0
    assert out[0, 5] == pytest.approx((4.0 - 1.0) / 4.0)
    assert out[1, 5] == pytest.approx((2.0 - 1.0) / 2.0)
    assert np.all(out[:, 6] == 0.0)
    # no parents: hop/transfer columns zero (§2: 0 for a task with no parents)
    assert np.all(out[:, 7:10] == 0.0)
    # krank one-hot: task 0 has type 0; A is rank 1, B rank 0
    a_row, b_row = out[0], out[1]
    assert a_row[PARTIAL_STATE_BASE_DIM + 1 * KRANK_TYPES + 0] == 1.0
    assert b_row[PARTIAL_STATE_BASE_DIM + 0 * KRANK_TYPES + 0] == 1.0
    assert a_row[PARTIAL_STATE_BASE_DIM:PARTIAL_STATE_BASE_DIM + 24].sum() == 1.0


def test_occupancy_load_and_violate_columns():
    ctx, (ca1, ca2, cb1) = _ctx(node_caps={"A": 4.0, "B": 1.0})
    committed = {0: ca1, 1: ca2}  # two tasks on A (types 0 and 1)
    out = partial_state_columns(ctx, 2, [ca1, cb1], committed=committed)
    # candidate on A: occ = [1,1,0,0], load 2/4, remaining (4-2-1)/4, no violation
    assert list(out[0, 0:4]) == [1.0, 1.0, 0.0, 0.0]
    assert out[0, 4] == pytest.approx(2.0 / 4.0)
    assert out[0, 5] == pytest.approx(1.0 / 4.0)
    assert out[0, 6] == 0.0
    # candidate on B (cap 1.0, empty): would fit exactly -> no violation
    assert out[1, 6] == 0.0
    # now shrink B below the demand: hypothetical placement violates
    ctx2, _ = _ctx(node_caps={"A": 4.0, "B": 0.5})
    out2 = partial_state_columns(ctx2, 2, [cb1], committed=committed)
    assert out2[0, 6] == 1.0
    assert out2[0, 5] == pytest.approx((0.5 - 0.0 - 1.0) / 0.5)  # negative headroom


def test_parent_hop_and_transfer_columns():
    ctx, (ca1, ca2, cb1) = _ctx()
    committed = {0: ca1, 1: ca2, 2: cb1}
    # task 3's parents are 1 (on A) and 2 (on B)
    out = partial_state_columns(ctx, 3, [ca1, cb1], committed=committed)
    # candidate on A: parent 1 same-node (0 hops), parent 2 remote (2 hops)
    assert out[0, 7] == 0.0 and out[0, 8] == 2.0
    # transfer: only the remote parent contributes h*payload/bneck = 2*100/100 = 2,
    # normalized by transfer_norm=2 -> 1.0
    assert out[0, 9] == pytest.approx(1.0)
    # candidate on B: parent 2 same-node, parent 1 remote — same values by symmetry
    assert out[1, 7] == 0.0 and out[1, 8] == 2.0
    assert out[1, 9] == pytest.approx(1.0)


def test_uncommitted_parent_fails_loud():
    ctx, (ca1, ca2, cb1) = _ctx()
    with pytest.raises(ValueError, match="uncommitted"):
        partial_state_columns(ctx, 3, [ca1], committed={1: ca2})  # parent 2 missing


def test_krank_plan_sum_reproduces_plan_level_counts():
    """Summing the krank one-hot over a plan's edges must equal the plan-level
    krank_cols construction (rank-major, type-minor): that identity is what makes
    the §9c/§9d pooled surrogate pointwise-representable by construction."""
    ctx, (ca1, ca2, cb1) = _ctx()
    plan = {0: ca1, 1: ca2, 2: cb1, 3: cb1}  # types 0..3
    total = np.zeros(KRANK_FEATURE_DIM)
    committed = {}
    for t in [0, 1, 2, 3]:  # topological order
        row = partial_state_columns(ctx, t, [plan[t]], committed=committed)[0]
        total += row[PARTIAL_STATE_BASE_DIM:PARTIAL_STATE_BASE_DIM + KRANK_FEATURE_DIM]
        committed[t] = plan[t]
    expected = np.zeros(KRANK_FEATURE_DIM)
    for t, cand in plan.items():
        r = ctx.node_rank[ctx.node_of[cand]]
        expected[r * KRANK_TYPES + ctx.task_type_index[t]] += 1.0
    assert np.array_equal(total, expected)
    assert total.sum() == 4.0


def test_linkrank_columns_co_use_and_core_restriction():
    core = frozenset({"B|core0"})
    ctx, (ca1, ca2, cb1) = _ctx(core_links=core)
    # tasks 0 and 1 committed on B: each ingress route uses both links
    committed = {0: cb1, 1: ca1}
    # candidate on B for task 2: route links (A|core0, B|core0); committed co-use:
    # task 0 (on B) used both links; task 1 (on A) used none.
    out = partial_state_columns(ctx, 2, [cb1, ca1], committed=committed)
    base = PARTIAL_STATE_BASE_DIM + KRANK_FEATURE_DIM
    # max co-use counting this task: 1 committed + 1 = 2
    assert out[0, base + 0] == 2.0
    # links already used by >=1 committed task: both
    assert out[0, base + 1] == 2.0
    # core-restricted twins: only B|core0 is "core" here
    assert out[0, base + 2] == 2.0
    assert out[0, base + 3] == 1.0
    # co-located candidate (task source A? in this fixture ingress to A is empty):
    # all four linkrank columns must be zero (§2: 0 when co-located)
    assert np.all(out[1, base:base + 4] == 0.0)


def test_krank_node_order_key_and_width_guard():
    caps = {"n1": 4.0, "n2": 2.0, "n3": 2.0}
    hops = {"n1": 1.0, "n2": 2.0, "n3": 1.0}
    order = krank_node_order(caps, hops)
    # ascending (cap, mean_hop, name): n3 (2,1) < n2 (2,2) < n1 (4,1)
    assert order == {"n3": 0, "n2": 1, "n1": 2}
    with pytest.raises(ValueError, match="node sets differ"):
        krank_node_order({"n1": 1.0}, hops)
    many_caps = {f"n{i}": float(i) for i in range(KRANK_WIDTH + 1)}
    many_hops = {f"n{i}": 0.0 for i in range(KRANK_WIDTH + 1)}
    with pytest.raises(ValueError, match="pad width"):
        krank_node_order(many_caps, many_hops)


def test_contract_machinery():
    assert validate_partial_state_contract("partial_state_v1") == PARTIAL_STATE_CONTRACT_V1
    with pytest.raises(Exception):
        validate_partial_state_contract("nonesuch")
    assert resolve_partial_state_contract() == PARTIAL_STATE_CONTRACT_V1
    # matching passes silently
    require_matching_partial_state_contract(
        PARTIAL_STATE_CONTRACT_V1, PARTIAL_STATE_CONTRACT_V1, model_label="m"
    )
    # a sidecar-less dim63crk checkpoint is NOT evidence and is NOT served
    with pytest.raises(PartialStateContractMismatchError, match="no partial-state"):
        require_matching_partial_state_contract(
            None, PARTIAL_STATE_CONTRACT_V1, model_label="m"
        )


def test_determinism_pure_function():
    ctx, (ca1, ca2, cb1) = _ctx()
    committed = {0: ca1, 1: ca2, 2: cb1}
    a = partial_state_columns(ctx, 3, [ca1, cb1, ca2], committed=committed)
    b = partial_state_columns(ctx, 3, [ca1, cb1, ca2], committed=committed)
    assert np.array_equal(a, b)
    assert a.dtype == np.float32
