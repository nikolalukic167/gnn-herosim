"""route_b_env_pivot_v1 ladder — cap_mode support across
verify_route_b_scorer_agreement.py (recompute/--check-repairs, check_blocks/
--check-blocks, check_krank/--check-krank, krank_rank_map's node ordering).

H1-H3 score under --cap-mode alpha_mean (ROUTE_B_ENV_PIVOT_SCREEN.md §3); before this
fix the verifier hardcoded alpha_max-equivalent caps everywhere. These tests lock in
that the verifier's compute_caps() reproduces the scorer's Dataset.node_caps(cap_mode=)
exactly, end to end through --check-repairs.
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


def toy_files(tmp_path):
    ds_dir = write_toy(tmp_path, coupled_rtt)
    tt_path = tmp_path / "task_types.json"
    with open(tt_path, "w") as fh:
        json.dump(TOY_TASK_TYPES, fh)
    return ds_dir, tt_path


@pytest.mark.parametrize("cap_mode", ["alpha_max", "alpha_mean", {"absolute": 10.0}])
def test_compute_caps_matches_dataset_node_caps(tmp_path, cap_mode):
    ds_dir, tt_path = toy_files(tmp_path)
    ds = Dataset(ds_dir, TOY_TASK_TYPES, "rtt")
    scorer_caps = ds.node_caps(2.0, cap_mode=cap_mode)

    rows, ttypes, pid_map, task_db, dag_edges, net, sources, scales = verifier.load(
        ds_dir, tt_path)
    verifier_caps = verifier.compute_caps(
        rows, ttypes, pid_map, task_db, verifier.demand_of, 2.0, scales, cap_mode)

    assert set(scorer_caps) == set(verifier_caps)
    for node in scorer_caps:
        assert verifier_caps[node] == pytest.approx(scorer_caps[node], abs=TOL)


@pytest.mark.parametrize("cap_mode", ["alpha_max", "alpha_mean", {"absolute": 10.0}])
def test_recompute_agrees_with_score_dataset_under_cap_mode(tmp_path, cap_mode):
    ds_dir, tt_path = toy_files(tmp_path)
    ds = Dataset(ds_dir, TOY_TASK_TYPES, "rtt")
    scored = score_dataset(ds, alpha=2.0, cap_mode=cap_mode)

    rows, ttypes, pid_map, task_db, dag_edges, net, sources, scales = verifier.load(
        ds_dir, tt_path)
    r_exact, r_greedy, repairs = verifier.recompute(
        rows, ttypes, pid_map, task_db, scales, 2.0, check_repairs=True,
        dag_edges=dag_edges, net=net, sources=sources, cap_mode=cap_mode)

    assert r_exact == pytest.approx(scored["r_exact_pct"], abs=TOL)
    for kind in ("1int", "kint", "t1"):
        v = repairs[kind]
        sat_key = f"repair_{kind}_saturated"
        val_key = f"r_exact_repaired_{kind}_pct"
        if v is None:
            assert scored.get(sat_key) is True
        else:
            assert not scored.get(sat_key)
            regret, _tied = v
            assert scored[val_key] == pytest.approx(regret, abs=1e-6)


def test_cap_mode_absent_defaults_to_alpha_max_everywhere():
    """No cap_mode argument anywhere reproduces the pre-cap_mode hardcoded formula --
    the byte-identity contract every rung below H0 depends on."""
    import inspect
    sig = inspect.signature(verifier.recompute)
    assert sig.parameters["cap_mode"].default == "alpha_max"
    sig2 = inspect.signature(verifier.compute_caps)
    assert sig2.parameters["cap_mode"].default == "alpha_max"


def test_krank_rank_map_cap_mode_changes_ordering_under_heterogeneous_caps(tmp_path):
    """The rank ordering itself (used to pool krank across datasets) must be
    cap_mode-aware, since route_b_coefficient_transfer.node_features's "cap" field
    reads from Cell.caps (cap_mode-aware)."""
    ds_dir, tt_path = toy_files(tmp_path)
    rows, ttypes, pid_map, task_db, dag_edges, net, sources, scales = verifier.load(
        ds_dir, tt_path)
    rank_max = verifier.krank_rank_map(rows, ttypes, pid_map, task_db, scales, net, 2.0,
                                       cap_mode="alpha_max")
    rank_mean = verifier.krank_rank_map(rows, ttypes, pid_map, task_db, scales, net, 2.0,
                                        cap_mode="alpha_mean")
    # Same node set either way.
    assert set(rank_max) == set(rank_mean)
    # Default (no cap_mode passed) reproduces alpha_max exactly.
    rank_default = verifier.krank_rank_map(rows, ttypes, pid_map, task_db, scales, net, 2.0)
    assert rank_default == rank_max
