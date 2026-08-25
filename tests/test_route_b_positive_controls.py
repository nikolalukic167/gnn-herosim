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

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts_cosim"))

from score_route_b_contention import (  # noqa: E402
    Dataset,
    additive_argmin_plan,
    greedy_masked_plan,
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

def test_missing_placements_jsonl_raises(tmp_path):
    ds_dir = tmp_path / "ds_broken"
    (ds_dir / "placements").mkdir(parents=True)
    with open(ds_dir / "infrastructure.json", "w") as fh:
        json.dump({"replica_placements": {}}, fh)
    with open(ds_dir / "workload.json", "w") as fh:
        json.dump({"events": []}, fh)
    with pytest.raises(RuntimeError, match="placements.jsonl missing"):
        Dataset(ds_dir, RIG_TASK_TYPES, "rtt")


def test_unknown_platform_type_demand_raises(tmp_path):
    replicas = {"rigA": [
        {"node_name": "n", "platform_id": 201, "platform_type": "mysteryFpga"},
    ]}
    ds_dir = write_rig(tmp_path, replicas, [({0: (100, 201)}, 1.0)])
    with pytest.raises(RuntimeError, match="no memoryRequirements"):
        Dataset(ds_dir, {"rigA": {"memoryRequirements": {"rigCpu": 1.0}}}, "rtt")
