"""Closed-form fixtures for the route_b_v1 repair machinery.

Why this file exists. Two of the last three independent checks of route_b_v1 found bugs in
the VERIFIER rather than in the scorer, and the second was the bad kind: a definitional
error (`max` node-occupancy excess where the registration says `sum`) masked by a
numerical one (a normal-equations solver that never reached the true LS optimum), so each
defect hid the other for three rounds. Both were caught by luck — a corpus-scale
disagreement on one dataset — not by a test. At that point `solve_least_squares`, the
function that had been wrong twice, and `t1_cols`, the column set that produced
NO-GO-PREPROBE-T1, had zero unit coverage between them; the 13 positive controls in
test_route_b_positive_controls.py exercise only `one_integer_cols` / `k_integer_cols` on
two 2-task rigs with no DAG and no network.

So these are fixtures, not regression snapshots: every expected number below is either a
textbook closed form or hand-computed from the toy topology in this file's own comments,
and none of it was read back out of an implementation. A column-definition drift fails
here, loudly, on the next run.

Layout of the toy (used by everything except the pure solver tests):

    nodes/links     n0 --100-- n1 --50-- n2 --25-- n3      (bandwidth_mbps)
    latency         n0n1 .01  n1n2 .02  n2n3 .04
                    n0n2 .03  n1n3 .06  n0n3 .07

    dag             dnn1 -> dnn2 -> cnn      task ids 0,1,2,3 in static_order:
                    dnn1 -> rf   -> cnn      dnn1=0 dnn2=1 rf=2 cnn=3

    memory (cpu)    dnn1 4.0   dnn2 2.0   rf 1.0   cnn 3.0
    memory (gpu)    rf 0.0                       <- makes n3 an UNCAPPED node

    replicas        dnn1: n0/10 n1/11        rf:  n1/30 n2/31 n3/32(gpu)
                    dnn2: n0/20 n2/21        cnn: n0/40 n2/41 n1/42

    caps(alpha=2)   n0 8.0   n1 8.0   n2 6.0   n3 absent (max single demand 0)
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts_cosim"))

from score_route_b_contention import (  # noqa: E402
    T1_BLOCKS,
    T1_REGISTERED_BLOCKS,
    Dataset,
    k_integer_cols,
    k_integer_keys,
    marginal_surrogate_regret,
    min_marginals,
    one_integer_cols,
    score_dataset,
    t1_cols,
    t1_column_names,
)
import verify_route_b_scorer_agreement as verifier  # noqa: E402

TOL = 1e-12

# --- the toy, as data ------------------------------------------------------

TOY_TASK_TYPES = {
    "dnn1": {"memoryRequirements": {"cpu": 4.0}},
    "dnn2": {"memoryRequirements": {"cpu": 2.0}},
    "rf": {"memoryRequirements": {"cpu": 1.0, "gpu": 0.0}},
    "cnn": {"memoryRequirements": {"cpu": 3.0}},
}

TOY_REPLICAS = {
    # dnn1 carries a second replica on n0 (pid 12) purely to widen the sweep past the
    # saturation guard: 3*2*3*3 = 54 rows against the t1 fit's 23 parameters. It adds no
    # new (node, type) kint key, so every hand-computed expectation below is unaffected.
    "dnn1": [{"node_name": "n0", "platform_id": 10, "platform_type": "cpu"},
             {"node_name": "n1", "platform_id": 11, "platform_type": "cpu"},
             {"node_name": "n0", "platform_id": 12, "platform_type": "cpu"}],
    "dnn2": [{"node_name": "n0", "platform_id": 20, "platform_type": "cpu"},
             {"node_name": "n2", "platform_id": 21, "platform_type": "cpu"}],
    "rf": [{"node_name": "n1", "platform_id": 30, "platform_type": "cpu"},
           {"node_name": "n2", "platform_id": 31, "platform_type": "cpu"},
           {"node_name": "n3", "platform_id": 32, "platform_type": "gpu"}],
    "cnn": [{"node_name": "n0", "platform_id": 40, "platform_type": "cpu"},
            {"node_name": "n2", "platform_id": 41, "platform_type": "cpu"},
            {"node_name": "n1", "platform_id": 42, "platform_type": "cpu"}],
}

TOY_DAG = {"dnn1": [], "dnn2": ["dnn1"], "rf": ["dnn1"], "cnn": ["dnn2", "rf"]}

_LINE = ["n0", "n1", "n2", "n3"]
TOY_LINK_BW = {("n0", "n1"): 100.0, ("n1", "n2"): 50.0, ("n2", "n3"): 25.0}
TOY_LATENCY = {("n0", "n1"): 0.01, ("n1", "n2"): 0.02, ("n2", "n3"): 0.04,
               ("n0", "n2"): 0.03, ("n1", "n3"): 0.06, ("n0", "n3"): 0.07}

# candidate placements per task id, as they appear in the sweep
TOY_CANDIDATES = {
    0: [(0, 10), (0, 11), (0, 12)],
    1: [(0, 20), (0, 21)],
    2: [(0, 30), (0, 31), (0, 32)],
    3: [(0, 40), (0, 41), (0, 42)],
}
# the 36-row variant, narrow enough that the t1 fit trips the saturation guard
TOY_CANDIDATES_NARROW = {**TOY_CANDIDATES, 0: [(0, 10), (0, 11)]}
TOY_NODE_OF = {10: "n0", 11: "n1", 12: "n0", 20: "n0", 21: "n2",
               30: "n1", 31: "n2", 32: "n3", 40: "n0", 41: "n2", 42: "n1"}
TOY_EDGES = [(0, 1), (0, 2), (1, 3), (2, 3)]

# hand-written plans, referenced by the expectations below
PLAN_P = {0: (0, 10), 1: (0, 20), 2: (0, 30), 3: (0, 41)}  # n0,n0,n1,n2
PLAN_Q = {0: (0, 10), 1: (0, 20), 2: (0, 30), 3: (0, 42)}  # n0,n0,n1,n1
PLAN_R = {0: (0, 10), 1: (0, 20), 2: (0, 30), 3: (0, 40)}  # n0,n0,n1,n0
PLAN_S = {0: (0, 11), 1: (0, 21), 2: (0, 32), 3: (0, 41)}  # n1,n2,n3,n2


def _pair(a, b):
    return (a, b) if _LINE.index(a) < _LINE.index(b) else (b, a)


def toy_route(parent_node, child_node):
    """(hops, bottleneck, latency) on the line topology — written independently of the
    scorer's route_metrics so the two can be compared."""
    if parent_node == child_node:
        return 0, math.inf, 0.0
    i, j = sorted((_LINE.index(parent_node), _LINE.index(child_node)))
    hops = [_pair(_LINE[k], _LINE[k + 1]) for k in range(i, j)]
    return (len(hops), min(TOY_LINK_BW[h] for h in hops),
            TOY_LATENCY[_pair(parent_node, child_node)])


def toy_transfer(plan):
    """sum_edges hops/bottleneck for a plan, computed from this file's constants only."""
    total = 0.0
    for parent, child in TOY_EDGES:
        hops, bneck, _lat = toy_route(TOY_NODE_OF[plan[parent][1]],
                                      TOY_NODE_OF[plan[child][1]])
        if hops:
            total += hops / bneck
    return total


def toy_plans(candidates=None):
    """The full 3 x 2 x 3 x 3 = 54-plan sweep, in a fixed order."""
    candidates = candidates or TOY_CANDIDATES
    plans = []
    for a in candidates[0]:
        for b in candidates[1]:
            for c in candidates[2]:
                for d in candidates[3]:
                    plans.append({0: a, 1: b, 2: c, 3: d})
    return plans


def write_toy(tmp_path: Path, rtt_fn, candidates=None) -> Path:
    """Materialize the toy as a dataset directory the real loaders accept."""
    ds = tmp_path / "ds_toy"
    (ds / "placements").mkdir(parents=True)
    routes, links, maps = {}, {}, {}
    for a in _LINE:
        for b in _LINE:
            if a == b:
                continue
            i, j = _LINE.index(a), _LINE.index(b)
            step = 1 if j > i else -1
            routes.setdefault(a, {})[b] = _LINE[i:j + step:step] if step > 0 else \
                list(reversed(_LINE[j:i + 1]))
            maps.setdefault(b, {})[a] = {"latency": TOY_LATENCY[_pair(a, b)]}
    # The client attaches at n0 (an access trunk, like the generator's backbone):
    # ingress routes for the linkrank block, which walks client -> executing node.
    # route_links needs only the paths — linkrank never reads link bandwidths.
    for b in _LINE:
        routes.setdefault("client_node0", {})[b] = \
            ["client_node0"] + _LINE[:_LINE.index(b) + 1]
    for (a, b), bw in TOY_LINK_BW.items():
        links["|".join(sorted((a, b)))] = {"bandwidth_mbps": bw}
    with open(ds / "infrastructure.json", "w") as fh:
        json.dump({"replica_placements": TOY_REPLICAS,
                   "link_topology": {"routes": routes, "links": links},
                   "network_maps": maps}, fh)
    with open(ds / "workload.json", "w") as fh:
        json.dump({"events": [{"application": {"dag": TOY_DAG},
                               "node_name": "client_node0"}]}, fh)
    plans = toy_plans(candidates)
    with open(ds / "placements" / "placements.jsonl", "w") as fh:
        for plan in plans:
            fh.write(json.dumps({
                "placement_plan": {str(k): list(v) for k, v in plan.items()},
                "rtt": rtt_fn(plan),
            }) + "\n")
    with open(ds / "placement_metadata.json", "w") as fh:
        json.dump({"num_placements": len(plans), "rows_written": len(plans),
                   "worker_failed": 0, "timed_out": 0, "sweep_complete": True}, fh)
    return ds


# Separable per-(task, platform) costs, positive everywhere. These particular values are
# not arbitrary: with the coupling term below they put the toy in a FIRING cell, where the
# pointwise decode is genuinely beaten and the three repair arms separate (1int partial,
# kint none, t1 total). A first hand-picked table sat in a non-firing cell — R_exact was
# 0.000 at every coupling strength from 100 to 5000, which is not a bug but the ordinary
# case: over 600 randomized (cost table, alpha) cells only 11% fire, against the 17.2% the
# real route_b_v1 corpus fires at. A fixture that asserts "coupling breaks the pointwise
# pick" has to be placed in such a cell deliberately.
TOY_BASE_COST = {(0, 10): 51.0, (0, 11): 45.0, (0, 12): 26.0,
                 (1, 20): 28.0, (1, 21): 38.0,
                 (2, 30): 40.0, (2, 31): 24.0, (2, 32): 49.0,
                 (3, 40): 32.0, (3, 41): 33.0, (3, 42): 57.0}


def separable_rtt(plan):
    return sum(TOY_BASE_COST[(t, p[1])] for t, p in plan.items())


def coupled_rtt(plan):
    """Separable base plus the charged coupling term, at a coefficient large enough that
    it changes the argmin (500 * transfer spans ~10..75 against a ~100 base)."""
    return separable_rtt(plan) + 500.0 * toy_transfer(plan)


def toy_dataset(tmp_path, rtt_fn=coupled_rtt) -> Dataset:
    return Dataset(write_toy(tmp_path, rtt_fn), TOY_TASK_TYPES, "rtt")


# ---------------------------------------------------------------------------
# 1. solve_least_squares — the function that has been wrong twice
# ---------------------------------------------------------------------------

def test_solver_exactly_determined_system():
    """b0=1; b0+b1=3; b0+b2=7  ->  beta = [1, 2, 6]."""
    beta = verifier.solve_least_squares([[1, 0, 0], [1, 1, 0], [1, 0, 1]],
                                        [1.0, 3.0, 7.0])
    assert beta == pytest.approx([1.0, 2.0, 6.0], abs=TOL)


def test_solver_simple_regression_closed_form():
    """Textbook OLS: x=[1,2,3,4], y=[6,5,7,10].  Sxy/Sxx = 7/5 = 1.4,
    intercept = ybar - slope*xbar = 7 - 1.4*2.5 = 3.5."""
    X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]
    beta = verifier.solve_least_squares(X, [6.0, 5.0, 7.0, 10.0])
    assert beta == pytest.approx([3.5, 1.4], abs=TOL)


def test_solver_drops_dependent_column_without_moving_the_projection():
    """A dependent column gets coefficient 0 and the fitted values are unchanged — the
    property the QR rewrite relies on (any exact LS solution projects onto the same
    column space)."""
    X = [[1.0, 1.0, 2.0], [1.0, 2.0, 4.0], [1.0, 3.0, 6.0], [1.0, 5.0, 10.0]]
    y = [3.0, 5.0, 8.0, 11.0]
    beta = verifier.solve_least_squares(X, y)
    assert beta[2] == 0.0
    ref, *_ = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)
    fitted = [sum(b * v for b, v in zip(beta, row)) for row in X]
    assert fitted == pytest.approx((np.array(X) @ ref).tolist(), abs=1e-9)


def test_solver_survives_the_ill_conditioned_shape_that_broke_it():
    """Intercept + RTT-magnitude column + small integer counts, the exact mix that made
    normal equations lose the fit (10.6% vs 42.0% on ds_00008). Coefficients are compared
    against numpy loosely and FITTED VALUES tightly: fitted values are solver-independent,
    so a mismatch there is a numerical failure rather than a basis choice."""
    rng = np.random.default_rng(20260825)
    counts = rng.integers(0, 3, size=(60, 2)).astype(float)
    magnitude = 5000.0 + rng.normal(0, 300.0, size=60)
    X = np.column_stack([np.ones(60), magnitude, counts])
    y = 12.0 + 0.75 * magnitude + 40.0 * counts[:, 0] - 15.0 * counts[:, 1]
    y = y + rng.normal(0, 1e-3, size=60)
    beta = verifier.solve_least_squares(X.tolist(), y.tolist())
    ref, *_ = np.linalg.lstsq(X, y, rcond=None)
    assert (X @ np.array(beta)) == pytest.approx((X @ ref).tolist(), rel=1e-9)
    assert beta == pytest.approx(ref.tolist(), rel=1e-6)


# ---------------------------------------------------------------------------
# 2. column values, hand-computed on the toy
# ---------------------------------------------------------------------------

def test_task_ids_follow_the_documented_static_order(tmp_path):
    """Every expectation in this file is written against dnn1=0 dnn2=1 rf=2 cnn=3."""
    ds = toy_dataset(tmp_path)
    assert ds.task_type_names == ["dnn1", "dnn2", "rf", "cnn"]
    assert sorted(ds.dag_edges) == TOY_EDGES


def test_one_integer_column_is_sum_not_max(tmp_path):
    """PLAN_Q puts two tasks on n0 AND two on n1: sum(c-1) = 2, max(c-1) = 1. This is
    the plan shape that distinguishes the registered column from the verifier bug that
    survived three rounds of checking."""
    ds = toy_dataset(tmp_path)
    fn = one_integer_cols(ds)
    assert fn(PLAN_P) == [1.0]   # n0:2, n1:1, n2:1
    assert fn(PLAN_Q) == [2.0]   # n0:2, n1:2  -> max would say 1.0
    assert fn(PLAN_R) == [2.0]   # n0:3, n1:1  -> max agrees here, sum must too


def test_k_integer_vocabulary_and_counts(tmp_path):
    ds = toy_dataset(tmp_path)
    assert k_integer_keys(ds) == [
        ("n0", "cnn"), ("n0", "dnn1"), ("n0", "dnn2"),
        ("n1", "cnn"), ("n1", "dnn1"), ("n1", "rf"),
        ("n2", "cnn"), ("n2", "dnn2"), ("n2", "rf"), ("n3", "rf")]
    fn = k_integer_cols(ds)
    assert fn(PLAN_P) == [0, 1, 1, 0, 0, 1, 1, 0, 0, 0]
    assert fn(PLAN_S) == [0, 0, 0, 0, 1, 0, 1, 1, 0, 1]


def test_t1_columns_hand_computed(tmp_path):
    """PLAN_P = dnn1@n0, dnn2@n0, rf@n1, cnn@n2.

      occupancy   n0: tot 2 {dnn1,dnn2} load 6.0 | n1: tot 1 {rf} load 1.0
                  n2: tot 1 {cnn} load 3.0
      quad        cnn 1*1=1  dnn1 2*1=2  dnn2 2*1=2  rf 1*1=1
      load/cap    2*6/8 + 1*1/8 + 1*3/6 = 1.5 + 0.125 + 0.5 = 2.125
      over-cap    none (6<=8, 1<=8, 3<=6)
      edges       dnn1@n0 -> dnn2@n0 : same node                 -> same_node 1
                  dnn1@n0 -> rf@n1   : 1 hop, bw 100, lat .01
                  dnn2@n0 -> cnn@n2  : 2 hops, bw 50,  lat .03
                  rf@n1   -> cnn@n2  : 1 hop, bw 50,   lat .02
      hop sums    min 0+1+min(2,1)=2   max 0+1+max(2,1)=3
      transfer    1/100 + 2/50 + 1/50 = 0.01 + 0.04 + 0.02 = 0.07
      latency     .01 + .03 + .02 = 0.06
    """
    ds = toy_dataset(tmp_path)
    caps = ds.node_caps(2.0)
    assert caps == pytest.approx({"n0": 8.0, "n1": 8.0, "n2": 6.0})
    assert "n3" not in caps  # max single demand 0 -> uncapped, not cap 0

    expected_p = ([0, 1, 1, 0, 0, 1, 1, 0, 0, 0]      # kint
                  + [1.0, 2.0, 2.0, 1.0]              # quad: cnn dnn1 dnn2 rf
                  + [2.125, 0.0]                      # load/cap, over-cap
                  + [2.0, 3.0]                        # min/max hop sums
                  + [0.07, 0.06, 1.0])                # transfer, latency, same-node
    assert t1_cols(ds, caps)(PLAN_P) == pytest.approx(expected_p, abs=TOL)


def test_t1_columns_on_an_uncapped_node(tmp_path):
    """PLAN_S = dnn1@n1, dnn2@n2, rf@n3(gpu, 0 memory), cnn@n2.

      load/cap    1*4/8 + 2*5/6 + 1*0/inf = 0.5 + 1.6666... + 0.0
      edges       dnn1@n1 -> dnn2@n2 : 1 hop  bw 50  lat .02
                  dnn1@n1 -> rf@n3   : 2 hops bw 25  lat .06
                  dnn2@n2 -> cnn@n2  : same node
                  rf@n3   -> cnn@n2  : 1 hop  bw 25  lat .04
      transfer    1/50 + 2/25 + 1/25 = 0.02 + 0.08 + 0.04 = 0.14
    """
    ds = toy_dataset(tmp_path)
    caps = ds.node_caps(2.0)
    expected_s = ([0, 0, 0, 0, 1, 0, 1, 1, 0, 1]
                  + [2.0, 1.0, 2.0, 1.0]
                  + [0.5 + 5.0 / 3.0, 0.0]
                  + [3.0, 4.0]
                  + [0.14, 0.12, 1.0])
    assert t1_cols(ds, caps)(PLAN_S) == pytest.approx(expected_s, abs=TOL)


def test_blocks_partition_the_registered_column_set(tmp_path):
    """Every block subset is emitted in T1_BLOCKS order, the names line up with the
    values, and the per-block pieces concatenate back to the full registered set — the
    invariant the §9a report's byte-identity depends on."""
    ds = toy_dataset(tmp_path)
    caps = ds.node_caps(2.0)
    full = t1_cols(ds, caps)(PLAN_P)
    pieces = []
    for block in T1_REGISTERED_BLOCKS:
        piece = t1_cols(ds, caps, blocks=[block])(PLAN_P)
        assert len(piece) == len(t1_column_names(ds, blocks=[block]))
        pieces += piece
    assert pieces == pytest.approx(full, abs=TOL)
    # the opt-in linkrank block EXTENDS the registered set and never changes it: the
    # default emission is exactly the frozen §9a layout, and asking for every known
    # block appends linkrank's 8 columns after it.
    lnk = t1_cols(ds, caps, blocks=["linkrank"])(PLAN_P)
    assert len(lnk) == len(t1_column_names(ds, blocks=["linkrank"])) == 8
    assert t1_cols(ds, caps, blocks=T1_BLOCKS)(PLAN_P) == pytest.approx(
        full + lnk, abs=TOL)
    assert t1_column_names(ds) == (
        [f"kint[{n}|{t}]" for n, t in k_integer_keys(ds)]
        + ["quad[cnn]", "quad[dnn1]", "quad[dnn2]", "quad[rf]",
           "load_over_cap", "overcap_tasks", "min_hop_sum", "max_hop_sum",
           "transfer", "latency_sum", "same_node_edges"])
    # order is a property of T1_BLOCKS, not of the argument order
    assert t1_cols(ds, caps, blocks=["coupling", "kint"])(PLAN_P) == pytest.approx(
        t1_cols(ds, caps, blocks=["kint"])(PLAN_P)
        + t1_cols(ds, caps, blocks=["coupling"])(PLAN_P), abs=TOL)
    with pytest.raises(ValueError):
        t1_cols(ds, caps, blocks=["nonesuch"])
    with pytest.raises(ValueError):
        t1_cols(ds, caps, blocks=[])


# ---------------------------------------------------------------------------
# 3. scorer vs verifier, element-wise on the same plans
# ---------------------------------------------------------------------------

def test_scorer_and_verifier_agree_column_for_column(tmp_path):
    """The two t1 implementations differ deliberately in structure and in their cap
    convention — the scorer omits uncapped nodes, the verifier carries cap 0.0 and
    normalizes in cap_of(). PLAN_S exercises exactly that divergence."""
    ds_dir = write_toy(tmp_path, coupled_rtt)
    ds = Dataset(ds_dir, TOY_TASK_TYPES, "rtt")
    tt_path = tmp_path / "task_types.json"
    with open(tt_path, "w") as fh:
        json.dump(TOY_TASK_TYPES, fh)
    rows, ttypes, pid_map, task_db, dag_edges, net, _sources = verifier.load(
        ds_dir, tt_path)

    # the verifier's own capacity construction: peak demand per node, INCLUDING the 0.0
    peak = {}
    for plan, _v in rows:
        for t, p in plan.items():
            node, d = verifier.demand_of(t, p, ttypes, pid_map, task_db)
            peak[node] = max(peak.get(node, 0.0), d)
    v_caps = {n: 2.0 * m for n, m in peak.items()}
    assert v_caps["n3"] == 0.0  # the convention divergence, made explicit

    kint_keys = sorted({(verifier.node_of(t, p, ttypes, pid_map, task_db), ttypes[t])
                        for plan, _v in rows for t, p in plan.items()})
    assert kint_keys == k_integer_keys(ds)

    s_caps = ds.node_caps(2.0)
    s_fn = t1_cols(ds, s_caps)
    for plan in (PLAN_P, PLAN_Q, PLAN_R, PLAN_S):
        v_cols = verifier.t1_columns(plan, ttypes, pid_map, task_db, kint_keys,
                                     v_caps, dag_edges, net)
        assert s_fn(plan) == pytest.approx(v_cols, abs=TOL)
        assert verifier.repair_columns("1int", plan, ttypes, pid_map, task_db) == \
            pytest.approx(one_integer_cols(ds)(plan), abs=TOL)


def test_route_metrics_match_the_hand_written_topology(tmp_path):
    ds = toy_dataset(tmp_path)
    for a in _LINE:
        for b in _LINE:
            assert ds.route_metrics(a, b) == pytest.approx(toy_route(a, b))
    assert ds.route_metrics("n0", "n3") == (3, 25.0, 0.07)


# ---------------------------------------------------------------------------
# 4. end-to-end repair statistics
# ---------------------------------------------------------------------------

def test_separable_physics_gives_exactly_zero_regret(tmp_path):
    """The theorem-predicted zero, under a binding capacity. With rtt = sum_t c_t(p_t),
    m_t(p) = c_t(p) + const, so the marginal-sum argmin IS the constrained optimum for
    any feasibility restriction — R_exact must be 0.0 exactly, not approximately."""
    ds = toy_dataset(tmp_path, separable_rtt)
    out = score_dataset(ds, alpha=1.5)
    assert 0 < out["n_feasible"] < out["n_rows"]  # the cap actually binds
    assert out["r_exact_pct"] == 0.0
    for arm in ("1int", "kint", "t1"):
        assert out[f"r_exact_repaired_{arm}_pct"] == 0.0
        assert not out.get(f"repair_{arm}_saturated")


def test_coupled_physics_regret_matches_an_independent_brute_force(tmp_path):
    """R_exact recomputed here from the toy's own constants — marginals, feasibility and
    the argmin all re-derived without calling the scorer — and the repaired arms are
    checked against the registered min(base, repaired) clamp."""
    ds = toy_dataset(tmp_path, coupled_rtt)
    alpha = 2.0
    caps = {"n0": 8.0, "n1": 8.0, "n2": 6.0}

    def feasible(plan):
        load = {}
        for t, p in plan.items():
            node = TOY_NODE_OF[p[1]]
            mem = TOY_TASK_TYPES[["dnn1", "dnn2", "rf", "cnn"][t]]["memoryRequirements"]
            load[node] = load.get(node, 0.0) + mem[
                "gpu" if p[1] == 32 else "cpu"]
        return all(v <= caps.get(n, math.inf) + 1e-12 for n, v in load.items())

    plans = toy_plans()
    marg = {}
    for plan in plans:
        v = coupled_rtt(plan)
        for t, p in plan.items():
            cur = marg.setdefault(t, {})
            if p not in cur or v < cur[p]:
                cur[p] = v
    feas = [p for p in plans if feasible(p)]
    assert 0 < len(feas) < len(plans)
    best = min(coupled_rtt(p) for p in feas)
    pick = min(feas, key=lambda p: (sum(marg[t][q] for t, q in p.items()),
                                    tuple(sorted(p.items()))))
    expected = 100.0 * (coupled_rtt(pick) - best) / best

    out = score_dataset(ds, alpha=alpha)
    assert out["r_exact_pct"] == pytest.approx(expected, abs=TOL)
    assert out["r_exact_pct"] > 0.0  # the coupling really does break the pointwise pick
    for arm in ("1int", "kint", "t1"):
        assert out[f"r_exact_repaired_{arm}_pct"] <= out["r_exact_pct"] + TOL
    # the three arms genuinely separate on this cell: the excess-sharing column helps a
    # little, the constraint's own count vector not at all, the T1 partial-state set
    # closes it outright — the §9a shape, reproduced at a size that fits on one screen
    assert 0.0 < out["r_exact_repaired_1int_pct"] < out["r_exact_pct"]
    assert out["r_exact_repaired_kint_pct"] == pytest.approx(out["r_exact_pct"], abs=TOL)
    assert out["r_exact_repaired_t1_pct"] == 0.0


def test_saturation_guard_refuses_an_interpolating_repair(tmp_path):
    """The narrow 36-row toy against a 2 + 21 = 23-parameter t1 fit is under the 2x guard
    (36 < 46), so the t1 repair must be REFUSED, not reported as a clean zero. This is the
    P7 trap the guard exists for, on the arm that produced the NO-GO."""
    ds = Dataset(write_toy(tmp_path, coupled_rtt, TOY_CANDIDATES_NARROW),
                 TOY_TASK_TYPES, "rtt")
    caps = ds.node_caps(2.0)
    marginal = min_marginals(ds.rows)
    feas = [(p, v) for p, v in ds.rows if ds.plan_feasible(p, caps)]
    assert len(ds.rows) == 36
    assert 2 + len(t1_cols(ds, caps)(PLAN_P)) == 23
    assert marginal_surrogate_regret(ds, marginal, feas, t1_cols(ds, caps)) is None
    out = score_dataset(ds, alpha=2.0)
    assert out["repair_t1_saturated"] is True
    assert out["r_exact_repaired_t1_pct"] is None
    # ... while a narrow block stays under the guard and is reported
    coupling = t1_cols(ds, caps, blocks=["coupling"])
    assert marginal_surrogate_regret(ds, marginal, feas, coupling) is not None


def test_return_beta_is_opt_in_and_does_not_move_the_regret(tmp_path):
    ds = toy_dataset(tmp_path, coupled_rtt)
    caps = ds.node_caps(2.0)
    marginal = min_marginals(ds.rows)
    feas = [(p, v) for p, v in ds.rows if ds.plan_feasible(p, caps)]
    cols = t1_cols(ds, caps, blocks=["coupling"])
    plain = marginal_surrogate_regret(ds, marginal, feas, cols)
    regret, beta = marginal_surrogate_regret(ds, marginal, feas, cols, return_beta=True)
    assert regret == plain
    assert len(beta) == 2 + len(t1_column_names(ds, blocks=["coupling"]))
    # a saturated fit refuses under return_beta too, rather than reporting a bare None
    narrow = Dataset(write_toy(tmp_path / "narrow", coupled_rtt, TOY_CANDIDATES_NARROW),
                     TOY_TASK_TYPES, "rtt")
    n_caps = narrow.node_caps(2.0)
    n_feas = [(p, v) for p, v in narrow.rows if narrow.plan_feasible(p, n_caps)]
    saturated = marginal_surrogate_regret(narrow, min_marginals(narrow.rows), n_feas,
                                          t1_cols(narrow, n_caps), return_beta=True)
    assert saturated == (None, None)
