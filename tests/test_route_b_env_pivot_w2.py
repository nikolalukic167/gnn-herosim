"""route_b env pivot W2 — generator per-instance demand heterogeneity + cap_mode.

Unit-level coverage for:
  * score_route_b_contention.load_demand_scales / Dataset.demand_scales / Dataset.demand
    (demand = scale * type-table value; absent demand_scale -> scale 1.0 everywhere).
  * Dataset.node_caps's cap_mode option (alpha_max default unchanged, alpha_mean,
    {"absolute": x}), including the "uncapped node stays uncapped in every mode"
    invariant.
  * generate_gnn_datasets_fast._draw_demand_scale (seeded uniform draw, absent config
    -> 1.0 always, no rng consumption).

End-to-end generator + score-side smoke (actually running generate_gnn_datasets_fast.py
and score_route_b_contention.py against a tiny corpus) is exercised separately as a
throwaway smoke, not as a committed test — see the session report.
"""

import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts_cosim"))

from score_route_b_contention import (  # noqa: E402
    Dataset,
    load_demand_scales,
)
from generate_gnn_datasets_fast import _draw_demand_scale  # noqa: E402

RIG_TASK_TYPES = {
    "rigA": {"memoryRequirements": {"rigCpu": 2.0}},
    "rigB": {"memoryRequirements": {"rigCpu": 4.0}},
}


def _write_rig(tmp_path: Path, demand_scale=None) -> Path:
    ds = tmp_path / "ds_rig"
    (ds / "placements").mkdir(parents=True)
    replicas = {
        "rigA": [{"node_name": "n0", "platform_id": 10, "platform_type": "rigCpu"}],
        "rigB": [{"node_name": "n0", "platform_id": 20, "platform_type": "rigCpu"},
                {"node_name": "n1", "platform_id": 21, "platform_type": "rigCpu"}],
    }
    with open(ds / "infrastructure.json", "w") as fh:
        json.dump({"replica_placements": replicas}, fh)
    app = {"dag": {"rigA": [], "rigB": []}}
    if demand_scale is not None:
        app["demand_scale"] = demand_scale
    with open(ds / "workload.json", "w") as fh:
        json.dump({"events": [{"application": app, "node_name": "client_node0"}]}, fh)
    rows = [
        ({0: (0, 10), 1: (0, 20)}, 1.0),
        ({0: (0, 10), 1: (0, 21)}, 2.0),
    ]
    with open(ds / "placements" / "placements.jsonl", "w") as fh:
        for plan, rtt in rows:
            fh.write(json.dumps({
                "placement_plan": {str(k): list(v) for k, v in plan.items()},
                "rtt": rtt}) + "\n")
    with open(ds / "placement_metadata.json", "w") as fh:
        json.dump({"num_placements": len(rows), "rows_written": len(rows),
                   "worker_failed": 0, "timed_out": 0, "sweep_complete": True}, fh)
    return ds


def test_demand_scale_absent_defaults_to_one(tmp_path):
    ds_dir = _write_rig(tmp_path, demand_scale=None)
    assert load_demand_scales(ds_dir) == [1.0, 1.0]
    ds = Dataset(ds_dir, RIG_TASK_TYPES, "rtt")
    assert ds.demand[(0, (0, 10))] == pytest.approx(2.0)  # rigA table value, scale 1.0
    assert ds.demand[(1, (0, 20))] == pytest.approx(4.0)  # rigB table value, scale 1.0


def test_demand_scale_applied_when_present(tmp_path):
    ds_dir = _write_rig(tmp_path, demand_scale={"rigA": 3.0, "rigB": 0.5})
    assert load_demand_scales(ds_dir) == [3.0, 0.5]
    ds = Dataset(ds_dir, RIG_TASK_TYPES, "rtt")
    assert ds.demand[(0, (0, 10))] == pytest.approx(3.0 * 2.0)
    assert ds.demand[(1, (0, 20))] == pytest.approx(0.5 * 4.0)
    assert ds.demand[(1, (0, 21))] == pytest.approx(0.5 * 4.0)


def test_demand_scale_partial_dict_defaults_missing_types_to_one(tmp_path):
    ds_dir = _write_rig(tmp_path, demand_scale={"rigA": 5.0})  # rigB not mentioned
    assert load_demand_scales(ds_dir) == [5.0, 1.0]


def test_node_caps_alpha_max_is_the_default_and_unchanged(tmp_path):
    ds_dir = _write_rig(tmp_path, demand_scale={"rigA": 1.0, "rigB": 3.0})
    ds = Dataset(ds_dir, RIG_TASK_TYPES, "rtt")
    caps_default = ds.node_caps(2.0)
    caps_explicit = ds.node_caps(2.0, cap_mode="alpha_max")
    assert caps_default == caps_explicit
    # n0 hosts rigA (demand 2.0) and rigB (demand 12.0) candidates -> max=12.0
    assert caps_default["n0"] == pytest.approx(2.0 * 12.0)
    # n1 hosts only rigB (demand 12.0) -> max=12.0
    assert caps_default["n1"] == pytest.approx(2.0 * 12.0)


def test_node_caps_alpha_mean_differs_from_alpha_max_under_heterogeneity(tmp_path):
    ds_dir = _write_rig(tmp_path, demand_scale={"rigA": 1.0, "rigB": 3.0})
    ds = Dataset(ds_dir, RIG_TASK_TYPES, "rtt")
    caps_max = ds.node_caps(2.0, cap_mode="alpha_max")
    caps_mean = ds.node_caps(2.0, cap_mode="alpha_mean")
    # n0: demands {2.0 (rigA), 12.0 (rigB)} -> mean 7.0, max 12.0
    assert caps_mean["n0"] == pytest.approx(2.0 * 7.0)
    assert caps_max["n0"] == pytest.approx(2.0 * 12.0)
    assert caps_mean["n0"] != caps_max["n0"]
    # n1: demands {12.0} only -> mean == max
    assert caps_mean["n1"] == pytest.approx(caps_max["n1"])


def test_node_caps_absolute_ignores_demand_and_alpha_magnitude(tmp_path):
    ds_dir = _write_rig(tmp_path, demand_scale={"rigA": 1.0, "rigB": 3.0})
    ds = Dataset(ds_dir, RIG_TASK_TYPES, "rtt")
    caps = ds.node_caps(999.0, cap_mode={"absolute": 5.0})
    assert caps == {"n0": 5.0, "n1": 5.0}


def test_node_caps_uncapped_node_stays_uncapped_in_every_mode(tmp_path):
    # rigC has demand 0.0 everywhere (a GPU-style zero-cost placement) -> must stay
    # absent from caps (uncapped) under every cap_mode, matching plan_feasible's
    # caps.get(node, inf) convention.
    ds = tmp_path / "ds_zero"
    (ds / "placements").mkdir(parents=True)
    replicas = {"rigC": [{"node_name": "n2", "platform_id": 30,
                          "platform_type": "zeroCpu"}]}
    with open(ds / "infrastructure.json", "w") as fh:
        json.dump({"replica_placements": replicas}, fh)
    with open(ds / "workload.json", "w") as fh:
        json.dump({"events": [{"application": {"dag": {"rigC": []}},
                               "node_name": "client_node0"}]}, fh)
    rows = [({0: (0, 30)}, 1.0)]
    with open(ds / "placements" / "placements.jsonl", "w") as fh:
        for plan, rtt in rows:
            fh.write(json.dumps({
                "placement_plan": {str(k): list(v) for k, v in plan.items()},
                "rtt": rtt}) + "\n")
    with open(ds / "placement_metadata.json", "w") as fh:
        json.dump({"num_placements": 1, "rows_written": 1, "worker_failed": 0,
                   "timed_out": 0, "sweep_complete": True}, fh)
    task_types = {"rigC": {"memoryRequirements": {"zeroCpu": 0.0}}}
    dataset = Dataset(ds, task_types, "rtt")
    for cap_mode in ("alpha_max", "alpha_mean", {"absolute": 5.0}):
        caps = dataset.node_caps(2.0, cap_mode=cap_mode)
        assert "n2" not in caps, f"cap_mode={cap_mode} capped a zero-demand node"


# ---------------------------------------------------------------------------
# _draw_demand_scale
# ---------------------------------------------------------------------------

def test_draw_demand_scale_absent_config_returns_one_without_consuming_rng():
    rng = random.Random(42)
    state_before = rng.getstate()
    assert _draw_demand_scale(rng, None) == 1.0
    assert rng.getstate() == state_before, "absent demand_spread must not touch the rng"


def test_draw_demand_scale_uniform_is_seeded_and_in_range():
    rng1 = random.Random(7)
    rng2 = random.Random(7)
    spread = {"dist": "uniform", "params": [0.5, 2.0]}
    draws1 = [_draw_demand_scale(rng1, spread) for _ in range(20)]
    draws2 = [_draw_demand_scale(rng2, spread) for _ in range(20)]
    assert draws1 == draws2, "same seed must give the same draw sequence"
    assert all(0.5 <= d <= 2.0 for d in draws1)
    assert len(set(draws1)) > 1, "must actually vary, or the fixture is vacuous"


def test_draw_demand_scale_unknown_dist_raises():
    rng = random.Random(1)
    with pytest.raises(ValueError, match="unknown demand_spread dist"):
        _draw_demand_scale(rng, {"dist": "gaussian", "params": [1.0, 0.1]})
