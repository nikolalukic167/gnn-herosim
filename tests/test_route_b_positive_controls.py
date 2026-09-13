"""route_b_v1 positive controls — the gate on the gate.

Three of four conclusions in the route A session rested on catching bugs in our own
instrumentation. "0.000% regret" and "the scorer is broken" are indistinguishable from
outside, so these rigs make the harness reproduce regrets that are non-zero BY
CONSTRUCTION, at closed-form magnitudes frozen in the route_b_v1 pre-registration
(LINEAGES.md) before any route B corpus was scored:

  Control 1 (separable costs, hot-node cap):  R_greedy == 450.000000%, R_exact == 0
  Control 2 (pairwise costs, matching shape): R_exact  == 150.000000%, and BOTH count
             repairs must fail to clean it (every feasible plan has zero co-residency,
             so no count column can separate them)

Both rigs must also score exactly 0 with the capacity removed — the regret has to flip
on the constraint alone. If any assertion here fails, route B runs are VOID, not NO-GO.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts_cosim"))

from score_route_b_contention import (  # noqa: E402
    Dataset,
    additive_argmin_plan,
    decode_regret_band,
    greedy_masked_plan,
    marginal_sum,
    marginal_surrogate_regret,
    min_marginals,
    one_integer_cols,
    k_integer_cols,
    score_dataset,
)

RIG_TASK_TYPES = {
    "rigA": {"memoryRequirements": {"rigCpu": 1.0}},
    "rigB": {"memoryRequirements": {"rigCpu": 1.0}},
}


def write_rig(tmp_path: Path, replica_placements: dict, rows: list) -> Path:
    ds = tmp_path / "ds_rig"
    (ds / "placements").mkdir(parents=True)
    with open(ds / "infrastructure.json", "w") as fh:
        json.dump({"replica_placements": replica_placements}, fh)
    with open(ds / "workload.json", "w") as fh:
        json.dump({"events": [
            {"application": {"dag": {"rigA": []}}, "node_name": "client_node0"},
            {"application": {"dag": {"rigB": []}}, "node_name": "client_node0"},
        ]}, fh)
    with open(ds / "placements" / "placements.jsonl", "w") as fh:
        for plan, rtt in rows:
            fh.write(json.dumps({
                "placement_plan": {str(k): list(v) for k, v in plan.items()},
                "rtt": rtt,
            }) + "\n")
    with open(ds / "placement_metadata.json", "w") as fh:
        json.dump({"num_placements": len(rows), "rows_written": len(rows),
                   "worker_failed": 0, "timed_out": 0, "sweep_complete": True}, fh)
    return ds


def control1_rig(tmp_path: Path) -> Dataset:
    """Separable costs. Unconstrained optimum co-locates both tasks on the hot node;
    a capacity admitting one task forces a yield, and greedy yields the WRONG task:
    X (task 0, tie-break) grabs the hot slot that Y needed, and Y's alternative is
    catastrophic.

      c_X: H1=10, ax=11      c_Y: H2=9, ay=100
      plans: (H1,H2)=19  (H1,ay)=110  (ax,H2)=20  (ax,ay)=111
      constrained optimum (cap: one task on hot) = (ax,H2) = 20
      greedy under cap = (H1,ay) = 110   ->  R_greedy = (110-20)/20 = 450% exactly
      marginal-sum decode = (ax,H2)      ->  R_exact  = 0 (separable => immune)
    """
    replicas = {
        "rigA": [
            {"node_name": "hot", "platform_id": 201, "platform_type": "rigCpu"},
            {"node_name": "alt_x", "platform_id": 203, "platform_type": "rigCpu"},
        ],
        "rigB": [
            {"node_name": "hot", "platform_id": 202, "platform_type": "rigCpu"},
            {"node_name": "alt_y", "platform_id": 204, "platform_type": "rigCpu"},
        ],
    }
    h1, h2, ax, ay = (100, 201), (100, 202), (101, 203), (102, 204)
    rows = [
        ({0: h1, 1: h2}, 19.0),
        ({0: h1, 1: ay}, 110.0),
        ({0: ax, 1: h2}, 20.0),
        ({0: ax, 1: ay}, 111.0),
    ]
    ds_dir = write_rig(tmp_path, replicas, rows)
    return Dataset(ds_dir, RIG_TASK_TYPES, "rtt")


def control2_small_rig(tmp_path: Path) -> Dataset:
    """The 4-row version of Control 2. At this size a 3-parameter repair fit can
    INTERPOLATE the full sweep (pairwise coupling included) and mechanically zero the
    regret — the saturation trap this rig originally caught in the scorer. Kept as the
    regression test that the guard refuses such repairs instead of reporting them."""
    replicas = {
        "rigA": [
            {"node_name": "z", "platform_id": 201, "platform_type": "rigCpu"},
            {"node_name": "p", "platform_id": 203, "platform_type": "rigCpu"},
        ],
        "rigB": [
            {"node_name": "z", "platform_id": 202, "platform_type": "rigCpu"},
            {"node_name": "q", "platform_id": 204, "platform_type": "rigCpu"},
        ],
    }
    a, c, b, d = (100, 201), (100, 202), (101, 203), (102, 204)
    rows = [
        ({0: a, 1: c}, 10.0),
        ({0: a, 1: d}, 30.0),
        ({0: b, 1: c}, 30.0),
        ({0: b, 1: d}, 12.0),
    ]
    ds_dir = write_rig(tmp_path, replicas, rows)
    return Dataset(ds_dir, RIG_TASK_TYPES, "rtt")


def control2_rig(tmp_path: Path) -> Dataset:
    """Pairwise (matching-shaped) costs, 3 candidates per task so the one-integer
    repair fit is over-determined (9 rows >= 2 x 3 params) and cannot interpolate.

      true costs:        c      d      f
                  a     10     30    100
                  b     30     12    100
                  e    100    100     99
      additivity check: 10 + 12 != 30 + 30 — genuinely pairwise.
      cap makes (a,c) infeasible (a and c share node z, cap one task)
      constrained optimum = (b,d) = 12
      marginals: m_X = {a:10, b:12, e:99}, m_Y = {c:10, d:12, f:99}
      marginal-sum over feasible: (a,d)=(b,c)=22 < (b,d)=24 < ... -> picks a 22-tie
      plan, true cost 30 -> R_exact = (30-12)/12 = 150% exactly.

      Why no count repair can help: every FEASIBLE plan has zero co-residency, so the
      repaired surrogate a + b*marginal_sum ranks feasible plans by marginal_sum alone
      (b>0: same 150% pick) or reversed (b<0: picks (e,f), true 99 -> 725%). With
      repaired = min(base, repaired_raw), the repaired regret is 150% in every branch.
    """
    replicas = {
        "rigA": [
            {"node_name": "z", "platform_id": 201, "platform_type": "rigCpu"},
            {"node_name": "p", "platform_id": 203, "platform_type": "rigCpu"},
            {"node_name": "r", "platform_id": 205, "platform_type": "rigCpu"},
        ],
        "rigB": [
            {"node_name": "z", "platform_id": 202, "platform_type": "rigCpu"},
            {"node_name": "q", "platform_id": 204, "platform_type": "rigCpu"},
            {"node_name": "s", "platform_id": 206, "platform_type": "rigCpu"},
        ],
    }
    a, b, e = (100, 201), (101, 203), (103, 205)
    c, d, f = (100, 202), (102, 204), (104, 206)
    costs = {
        (a, c): 10.0, (a, d): 30.0, (a, f): 100.0,
        (b, c): 30.0, (b, d): 12.0, (b, f): 100.0,
        (e, c): 100.0, (e, d): 100.0, (e, f): 99.0,
    }
    rows = [({0: x, 1: y}, v) for (x, y), v in costs.items()]
    ds_dir = write_rig(tmp_path, replicas, rows)
    return Dataset(ds_dir, RIG_TASK_TYPES, "rtt")


# ---------------------------------------------------------------------------
# Control 1
# ---------------------------------------------------------------------------

def test_control1_greedy_regret_is_exactly_450(tmp_path):
    ds = control1_rig(tmp_path)
    out = score_dataset(ds, alpha=1.0)
    assert not out.get("greedy_stuck")
    assert out["r_greedy_pct"] == pytest.approx(450.0, abs=1e-6)


def test_control1_r_exact_is_zero_on_separable_costs(tmp_path):
    # The same rig that breaks greedy must NOT fire the perfect-decode statistic:
    # its regret is pure decoder myopia, and R_exact is immune by construction.
    ds = control1_rig(tmp_path)
    out = score_dataset(ds, alpha=1.0)
    assert out["r_exact_pct"] == pytest.approx(0.0, abs=1e-9)


def test_control1_flips_on_the_constraint_alone(tmp_path):
    ds = control1_rig(tmp_path)
    out = score_dataset(ds, alpha=None)
    assert out["r_greedy_pct"] == pytest.approx(0.0, abs=1e-9)
    assert out["r_exact_pct"] == pytest.approx(0.0, abs=1e-9)


def test_control1_free_choice_plan_is_infeasible_under_cap(tmp_path):
    ds = control1_rig(tmp_path)
    out = score_dataset(ds, alpha=1.0)
    assert out["componentwise_plan_enumerated"]
    assert not out["componentwise_plan_feasible"]


# ---------------------------------------------------------------------------
# Control 2
# ---------------------------------------------------------------------------

def test_control2_r_exact_is_exactly_150(tmp_path):
    ds = control2_rig(tmp_path)
    out = score_dataset(ds, alpha=1.0)
    assert out["r_exact_pct"] == pytest.approx(150.0, abs=1e-6)


def test_control2_count_repair_cannot_clean_pairwise_structure(tmp_path):
    # All feasible plans have zero node co-residency: the count column carries no
    # signal among them, so a 1int repair that reduces this regret is a scorer bug.
    ds = control2_rig(tmp_path)
    out = score_dataset(ds, alpha=1.0)
    assert not out.get("repair_1int_saturated")
    assert out["r_exact_repaired_1int_pct"] == pytest.approx(150.0, abs=1e-6)


def test_control2_kint_repair_is_refused_as_saturated_at_rig_scale(tmp_path):
    # 9 rows vs 8 kint params: the guard must refuse the fit, not report a repaired
    # value from an interpolating model.
    ds = control2_rig(tmp_path)
    out = score_dataset(ds, alpha=1.0)
    assert out.get("repair_kint_saturated") is True
    assert out["r_exact_repaired_kint_pct"] is None


def test_saturation_guard_refuses_interpolating_repairs(tmp_path):
    # The original 4-row rig: a 3-param repair fit interpolates the sweep exactly and
    # would report the pairwise regret as fully repaired (0.0). The guard must catch
    # exactly this — it did not in the scorer's first version, and this rig is what
    # found it.
    ds = control2_small_rig(tmp_path)
    out = score_dataset(ds, alpha=1.0)
    assert out["r_exact_pct"] == pytest.approx(150.0, abs=1e-6)
    assert out.get("repair_1int_saturated") is True
    assert out["r_exact_repaired_1int_pct"] is None
    assert out.get("repair_kint_saturated") is True
    assert out["r_exact_repaired_kint_pct"] is None


def test_control2_flips_on_the_constraint_alone(tmp_path):
    # Unconstrained, the marginal-sum decode finds (a,c)=10, the true optimum.
    ds = control2_rig(tmp_path)
    out = score_dataset(ds, alpha=None)
    assert out["r_exact_pct"] == pytest.approx(0.0, abs=1e-9)


def test_control2_greedy_also_fires(tmp_path):
    ds = control2_rig(tmp_path)
    out = score_dataset(ds, alpha=1.0)
    assert out["r_greedy_pct"] == pytest.approx(150.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Fail-loud contract
# ---------------------------------------------------------------------------

def control3_tie_rig(tmp_path: Path) -> Dataset:
    """NEAR-separable costs whose min-marginal sums TIE across plans of different cost.

    Reproduces at rig scale what fired on the H0 separable control (2026-08-27). Control 1
    uses distinct integer costs and never puts two feasible plans on the same surrogate
    score, so the tie-break path in `decode_regret` was untested — and that is precisely
    the path that produced 12 of 16 firing control datasets.

    Costs are separable PLUS a small same-node coupling, exactly the shape the storage-tier
    parent-locality branch gives the real corpus (AMENDMENT 1):

      c_X: A=10, B=10, C=12     c_Y: A=10, B=10, C=10     +0.6 when both share a node

      plans (X,Y):  AA=20.6  AB=20.0  AC=20.0
                    BA=20.0  BB=20.6  BC=20.0
                    CA=22.0  CB=22.0  CC=22.6

      min-marginals over the FULL sweep: m_X = {A:20, B:20, C:22}, m_Y = {A:20, B:20, C:20}
      so every plan with X on A or B scores msum = 40.0 — a SIX-member exact tie holding
      both the optimum (20.0) and the coupled plans (20.6).

    Platform ids are assigned so the WORSE member sorts first under
    `tuple(sorted(plan.items()))`: task 0 on node A carries the lowest id, so the decode
    lands on (A,A) = 20.6 against a tie-set optimum of 20.0.

      registered  = 100*(20.6-20.0)/20.0 = 3.0%   on physics that is separable to 3%
      optimistic  = 0.0 exactly
      pessimistic = 3.0%
      mean_tied   = 100*(mean(20.6,20,20,20,20.6,20) - 20)/20 = 1.0%
    """
    replicas = {
        "rigA": [
            {"node_name": "nA", "platform_id": 201, "platform_type": "rigCpu"},
            {"node_name": "nB", "platform_id": 202, "platform_type": "rigCpu"},
            {"node_name": "nC", "platform_id": 203, "platform_type": "rigCpu"},
        ],
        "rigB": [
            {"node_name": "nA", "platform_id": 211, "platform_type": "rigCpu"},
            {"node_name": "nB", "platform_id": 212, "platform_type": "rigCpu"},
            {"node_name": "nC", "platform_id": 213, "platform_type": "rigCpu"},
        ],
    }
    x = {"nA": (100, 201), "nB": (101, 202), "nC": (102, 203)}
    y = {"nA": (100, 211), "nB": (101, 212), "nC": (102, 213)}
    base_x = {"nA": 10.0, "nB": 10.0, "nC": 12.0}
    base_y = {"nA": 10.0, "nB": 10.0, "nC": 10.0}
    rows = []
    for nx, px in x.items():
        for ny, py in y.items():
            cost = base_x[nx] + base_y[ny] + (0.6 if nx == ny else 0.0)
            rows.append(({0: px, 1: py}, cost))
    ds_dir = write_rig(tmp_path, replicas, rows)
    return Dataset(ds_dir, RIG_TASK_TYPES, "rtt")


def control3_band(tmp_path: Path) -> dict:
    ds = control3_tie_rig(tmp_path)
    # alpha=3.0: cap admits both tasks on one node, so every plan stays feasible and the
    # tie is what decides the statistic — not the constraint.
    return score_dataset(ds, alpha=3.0)


def test_control3_registered_tiebreak_fires_on_near_separable_physics(tmp_path):
    """The artifact itself: an arbitrary tie-break reads as 3% regret.

    Passes before AND after the band was added — the registered statistic is deliberately
    unchanged. It documents the defect rather than gating it.
    """
    out = control3_band(tmp_path)
    assert out["r_exact_pct"] == pytest.approx(3.0, abs=1e-9)


def test_control3_optimistic_band_is_exactly_zero(tmp_path):
    """The teeth: the surrogate CAN reach the optimum, it just isn't credited with it."""
    out = control3_band(tmp_path)
    assert out["r_exact_band"]["optimistic"] == pytest.approx(0.0, abs=1e-12)


def test_control3_band_brackets_the_registered_value(tmp_path):
    out = control3_band(tmp_path)
    band = out["r_exact_band"]
    assert band["n_tied"] == 6
    assert band["optimistic"] <= band["registered"] <= band["pessimistic"]
    assert band["optimistic"] <= band["mean_tied"] <= band["pessimistic"]


def test_control3_mean_tied_is_the_hand_computed_tie_group_mean(tmp_path):
    """Pin the arithmetic so a tolerance change cannot silently move the fair reading."""
    out = control3_band(tmp_path)
    tied = [20.6, 20.0, 20.0, 20.0, 20.6, 20.0]
    expected = 100.0 * (sum(tied) / len(tied) - 20.0) / 20.0
    assert out["r_exact_band"]["mean_tied"] == pytest.approx(expected, abs=1e-9)
    assert out["r_exact_band"]["pessimistic"] == pytest.approx(3.0, abs=1e-9)


def test_control3_band_registered_member_reproduces_r_exact_pct(tmp_path):
    """The band must reproduce the registered statistic EXACTLY, not approximate it —
    every downstream consumer (firing_set, the verifier) keys off r_exact_pct."""
    out = control3_band(tmp_path)
    assert out["r_exact_band"]["registered"] == out["r_exact_pct"]


def test_control3_scorer_band_matches_transfer_tie_band(tmp_path):
    """One tie definition in the program.

    The scorer's decode_regret_band and route_b_coefficient_transfer's Cell.tie_band must
    agree; two tolerance rules would drift apart while both looked correct.
    """
    from route_b_coefficient_transfer import Cell  # noqa: E402

    ds = control3_tie_rig(tmp_path)
    cell = Cell(ds.ds_dir, RIG_TASK_TYPES, 3.0)
    predicted = np.array([marginal_sum(cell.marginal, p) for p, _v in cell.feasible])

    optimistic, pessimistic, mean_tied, n_tied = cell.tie_band(predicted)
    band = decode_regret_band(cell.feasible, list(predicted), cell.best)

    assert optimistic == pytest.approx(band["optimistic"], abs=1e-12)
    assert pessimistic == pytest.approx(band["pessimistic"], abs=1e-12)
    assert mean_tied == pytest.approx(band["mean_tied"], abs=1e-12)
    assert n_tied == band["n_tied"]


def test_missing_placements_jsonl_raises(tmp_path):
    ds_dir = tmp_path / "ds_broken"
    (ds_dir / "placements").mkdir(parents=True)
    with open(ds_dir / "infrastructure.json", "w") as fh:
        json.dump({"replica_placements": {}}, fh)
    with open(ds_dir / "workload.json", "w") as fh:
        json.dump({"events": []}, fh)
    with pytest.raises(RuntimeError, match="placements.jsonl missing"):
        Dataset(ds_dir, RIG_TASK_TYPES, "rtt")


def test_truncated_sweep_is_refused(tmp_path):
    # Route B's Arm S smoke lost 66-72/240 rows to mid-episode replica scale-down;
    # a truncated sweep biases marginals, optima and feasible sets and must be refused,
    # not scored.
    ds_dir = control1_rig(tmp_path).ds_dir
    meta = ds_dir / "placement_metadata.json"
    meta.write_text(json.dumps({
        "num_placements": 240, "rows_written": 174, "worker_failed": 66,
        "timed_out": 0, "sweep_complete": False}))
    with pytest.raises(RuntimeError, match="TRUNCATED"):
        Dataset(ds_dir, RIG_TASK_TYPES, "rtt")


def test_unknown_platform_type_demand_raises(tmp_path):
    replicas = {"rigA": [
        {"node_name": "n", "platform_id": 201, "platform_type": "mysteryFpga"},
    ]}
    ds_dir = write_rig(tmp_path, replicas, [({0: (100, 201)}, 1.0)])
    with pytest.raises(RuntimeError, match="no memoryRequirements"):
        Dataset(ds_dir, {"rigA": {"memoryRequirements": {"rigCpu": 1.0}}}, "rtt")
