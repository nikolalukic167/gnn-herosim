"""route_b_env_pivot_v1 — the verifier must apply demand_scale, like the scorer.

verify_route_b_scorer_agreement.demand_of read
`task_db[ttypes[task_id]]["memoryRequirements"][ptype]` with NO demand_scale factor,
while the scorer applies `scale * memReq` (score_route_b_contention.Dataset.demand,
built from load_demand_scales). Inert on H0 — every scale there is 1.0 — but H1's corpus
is already generated WITH demand_spread and carries values like
[1.6047, 1.5150, 1.8383, 0.8348]. Caps, feasibility and every repair would have diverged,
failing the registered 1e-9 agreement (an S0 VOID gate) for a reason that has nothing to
do with the physics under test.

Same shape as test_route_b_env_pivot_cap_mode_verifier.py, which is the file that caught
the analogous compute_caps alpha_mean dedup bug at 4614370.

Teeth: every test below except the unit-scale regression fails before the fix.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts_cosim"))

from score_route_b_contention import Dataset, score_dataset  # noqa: E402
import verify_route_b_scorer_agreement as verifier  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_route_b_repair_fixtures import (  # noqa: E402
    TOY_TASK_TYPES,
    coupled_rtt,
    write_toy,
)

TOL = 1e-9

# Deliberately non-unit, non-uniform and not a round ratio — a scale set that a bug
# multiplying by a constant, or ignoring per-task variation, cannot accidentally match.
SCALES = {"dnn1": 1.6047068212460185, "dnn2": 0.8348161072232341,
          "rf": 1.5150492311343670, "cnn": 1.8382693515572681}


def toy_files(tmp_path, scales=None):
    """The shared toy, optionally carrying an application.demand_scale block."""
    ds_dir = write_toy(tmp_path, coupled_rtt)
    if scales is not None:
        wl_path = ds_dir / "workload.json"
        workload = json.loads(wl_path.read_text())
        for event in workload["events"]:
            dag = event["application"]["dag"]
            names = list(dag)
            event["application"]["demand_scale"] = {
                name: scales[name] for name in names if name in scales}
        wl_path.write_text(json.dumps(workload))
    tt_path = tmp_path / "task_types.json"
    tt_path.write_text(json.dumps(TOY_TASK_TYPES))
    return ds_dir, tt_path


def test_the_rig_actually_carries_nonunit_scales(tmp_path):
    """Guard the premise: if the scales never reach the dataset, everything below is
    vacuously green — which is exactly how this bug survived."""
    ds_dir, tt_path = toy_files(tmp_path, SCALES)
    _rows, _tt, _pid, _db, _edges, _net, _src, scales = verifier.load(ds_dir, tt_path)
    assert len(set(scales)) > 1
    assert not all(s == 1.0 for s in scales)


def test_verifier_demand_matches_scorer_under_nonunit_demand_scale(tmp_path):
    """THE TEETH. Per (task_id, placement), the two implementations must agree."""
    ds_dir, tt_path = toy_files(tmp_path, SCALES)
    ds = Dataset(ds_dir, TOY_TASK_TYPES, "rtt")
    rows, ttypes, pid_map, task_db, _e, _n, _s, scales = verifier.load(ds_dir, tt_path)

    checked = 0
    for (task_id, placement), scorer_demand in ds.demand.items():
        _node, verifier_demand = verifier.demand_of(
            task_id, placement, ttypes, pid_map, task_db, scales)
        assert verifier_demand == pytest.approx(scorer_demand, abs=TOL), (
            f"task {task_id} at {placement}: verifier {verifier_demand} != "
            f"scorer {scorer_demand}")
        checked += 1
    assert checked > 0


@pytest.mark.parametrize("cap_mode", ["alpha_max", "alpha_mean"])
def test_verifier_caps_match_scorer_under_nonunit_demand_scale(tmp_path, cap_mode):
    """The statistic that actually fires the S0 gate."""
    ds_dir, tt_path = toy_files(tmp_path, SCALES)
    ds = Dataset(ds_dir, TOY_TASK_TYPES, "rtt")
    scorer_caps = ds.node_caps(2.0, cap_mode=cap_mode)

    rows, ttypes, pid_map, task_db, _e, _n, _s, scales = verifier.load(ds_dir, tt_path)
    verifier_caps = verifier.compute_caps(
        rows, ttypes, pid_map, task_db, verifier.demand_of, 2.0, scales, cap_mode)

    assert set(scorer_caps) == set(verifier_caps)
    for node in scorer_caps:
        assert verifier_caps[node] == pytest.approx(scorer_caps[node], abs=TOL)


def test_scaling_actually_changes_the_feasible_set(tmp_path):
    """A rig where the scale does not move feasibility would have no teeth at all."""
    plain_dir, plain_tt = toy_files(tmp_path / "plain")
    scaled_dir, _scaled_tt = toy_files(tmp_path / "scaled", SCALES)

    plain = Dataset(plain_dir, TOY_TASK_TYPES, "rtt")
    scaled = Dataset(scaled_dir, TOY_TASK_TYPES, "rtt")
    caps_plain = plain.node_caps(1.2, cap_mode="alpha_max")
    caps_scaled = scaled.node_caps(1.2, cap_mode="alpha_max")

    n_plain = sum(1 for p, _v in plain.rows if plain.plan_feasible(p, caps_plain))
    n_scaled = sum(1 for p, _v in scaled.rows if scaled.plan_feasible(p, caps_scaled))
    assert n_plain != n_scaled, (
        "demand_scale did not move the feasible set at alpha=1.2; pick a tighter alpha "
        "or a more skewed scale set, otherwise the agreement tests prove nothing")


@pytest.mark.parametrize("cap_mode", ["alpha_max", "alpha_mean"])
def test_verifier_r_exact_matches_scorer_under_nonunit_demand_scale(tmp_path, cap_mode):
    """Full independent recompute vs the scorer, end to end at 1e-9."""
    ds_dir, tt_path = toy_files(tmp_path, SCALES)
    ds = Dataset(ds_dir, TOY_TASK_TYPES, "rtt")
    scored = score_dataset(ds, alpha=2.0, cap_mode=cap_mode)

    rows, ttypes, pid_map, task_db, dag_edges, net, sources, scales = verifier.load(
        ds_dir, tt_path)
    r_exact, _r_greedy, _repairs = verifier.recompute(
        rows, ttypes, pid_map, task_db, scales, 2.0, check_repairs=True,
        dag_edges=dag_edges, net=net, sources=sources, cap_mode=cap_mode)

    assert r_exact == pytest.approx(scored["r_exact_pct"], abs=TOL)


def test_unit_demand_scale_is_identical_to_no_demand_scale(tmp_path):
    """H0's regression: an all-1.0 block and no block at all must agree exactly.

    This is what proves H0's already-recorded numbers are untouched by the fix.
    """
    ones = {name: 1.0 for name in SCALES}
    with_block, tt_a = toy_files(tmp_path / "ones", ones)
    without, tt_b = toy_files(tmp_path / "absent")

    a = verifier.load(with_block, tt_a)
    b = verifier.load(without, tt_b)
    assert a[7] == b[7] == [1.0] * len(a[7])

    ds_a = Dataset(with_block, TOY_TASK_TYPES, "rtt")
    ds_b = Dataset(without, TOY_TASK_TYPES, "rtt")
    assert ds_a.demand == ds_b.demand
    assert (score_dataset(ds_a, alpha=2.0)["r_exact_pct"]
            == score_dataset(ds_b, alpha=2.0)["r_exact_pct"])


def test_demand_of_requires_scales_explicitly():
    """No default on `scales` — a default is how a missed call site would silently
    reintroduce the bug this file exists to prevent."""
    import inspect

    sig = inspect.signature(verifier.demand_of)
    assert sig.parameters["scales"].default is inspect.Parameter.empty
    assert inspect.signature(verifier.node_of).parameters["scales"].default \
        is inspect.Parameter.empty
