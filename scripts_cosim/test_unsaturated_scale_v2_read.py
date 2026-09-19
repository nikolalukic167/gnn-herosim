"""unsaturated_scale_v2: the registered bars behave as signed, before any arm is served.

v2 exists because v1's nulls were under-powered, so the tests that matter most here are the
ones pinning that v2 cannot repeat that mistake quietly: the design-validation bar (L5), the
refusal to call an under-powered negative a tie (L4), and the selection rule that cannot be
steered by any arm's performance (M0).

Run: pipenv run python3 -m pytest scripts_cosim/test_unsaturated_scale_v2_read.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.unsaturated_scale_v2_read import (  # noqa: E402
    M_ALPHA, M_ARMS, M_BASELINES, M_DESIGN_SD_BAR, M_GNN, M_GRAPH, M_MIN_CHECKPOINTS,
    M_REACTIVE, M_SATURATION_SHARE, M_SEPARATE_PCT, M_STUDY_TOPOLOGIES, M_TARGET_SHARE,
    M_TOPOLOGY_CANDIDATES, M_TWIN, M_WINDOWS,
    V_ARM_BEATS_REACTIVE, V_DESIGN_READY, V_DESIGN_SHORT, V_GRAPH_FASTER,
    V_LEARNED_BEATS_REACTIVE, V_NOT_SEP, V_NO_LEARNED_BEATS_REACTIVE, V_POINTWISE_FASTER,
    V_POWER_DELIVERED, V_POWER_NOT_DELIVERED, V_REACTIVE_FASTER, V_SUM_COSTS, V_SUM_HELPS,
    V_SUM_NOT_SEP, V_WINDOW_CONSISTENT, V_WINDOW_FLIPS,
    checkpoint_stats, pair_checkpoint_stats, read_l1, read_l2, read_l3, read_l4, read_l5,
    read_l6, read_m0, saturated, select_topologies,
)
from scripts_cosim.unsaturated_scale_v1_read import (  # noqa: E402
    K_ALPHA, K_MIN_SEEDS, K_SATURATION_SHARE, K_SEPARATE_PCT, K_TARGET_SHARE,
)
from scripts_cosim.peer_only_v1_read import V_UNREADABLE  # noqa: E402

N = M_MIN_CHECKPOINTS
ENVS = [(9001, w) for w in M_WINDOWS]


def _stats(base, step=0.1):
    return {s: base + step * s for s in range(N)}


# --- the chain ----------------------------------------------------------------------------

def test_v2_reuses_v1s_bars_unchanged_so_the_two_are_comparable():
    """v2 is a POWER replication of v1. If a bar moved it would be a different question."""
    assert (M_SEPARATE_PCT, M_ALPHA, M_MIN_CHECKPOINTS) == (K_SEPARATE_PCT, K_ALPHA, K_MIN_SEEDS)
    assert (M_TARGET_SHARE, M_SATURATION_SHARE) == (K_TARGET_SHARE, K_SATURATION_SHARE)
    assert M_SEPARATE_PCT == 5.0 and M_ALPHA == 0.05 and M_MIN_CHECKPOINTS == 16


def test_the_design_is_sixteen_environments_and_excludes_the_hanging_topology():
    assert M_STUDY_TOPOLOGIES * len(M_WINDOWS) == 16
    assert 9004 not in M_TOPOLOGY_CANDIDATES        # hangs on every policy
    assert len(M_TOPOLOGY_CANDIDATES) == 12 and M_WINDOWS == ("w0", "w1", "w2", "w3")
    assert M_REACTIVE == "knative_network" and M_REACTIVE in M_BASELINES
    assert M_GRAPH in M_ARMS and M_TWIN in M_ARMS and M_GNN in M_ARMS


def test_knative_network_batch_is_carried_as_a_baseline_not_as_an_arm():
    """It was measured at 0.002 s of batch wait, so it is not the batching control it looks
    like. It must never appear in M_ARMS, where L4 would read it as a learned arm."""
    assert "knative_network_batch" in M_BASELINES
    assert "knative_network_batch" not in M_ARMS


# --- M0: the screen and the selection rule ------------------------------------------------

def _shares(default=0.66, overrides=None):
    out = {(t, w): default for t in M_TOPOLOGY_CANDIDATES for w in M_WINDOWS}
    out.update(overrides or {})
    return out


def test_m0_admits_only_environments_under_the_target_share():
    m0 = read_m0(_shares(overrides={(9001, "w0"): 0.85, (9002, "w0"): 0.95}))
    assert m0["environments"][(9001, "w0")]["admissible"] is False   # under 0.90 but over 0.80
    assert m0["environments"][(9001, "w0")]["saturated"] is False
    assert m0["environments"][(9002, "w0")]["saturated"] is True
    assert m0["environments"][(9003, "w0")]["admissible"] is True
    assert m0["n_screened"] == len(M_TOPOLOGY_CANDIDATES) * len(M_WINDOWS)


def test_m0_treats_a_missing_or_unknown_environment_as_inadmissible():
    """'Unknown is not a pass' -- an arm that never ran must not admit an environment."""
    s = _shares(); del s[(9001, "w2")]; s[(9002, "w1")] = None
    m0 = read_m0(s)
    assert m0["environments"][(9001, "w2")]["admissible"] is False
    assert m0["environments"][(9001, "w2")]["saturated"] is None
    assert m0["environments"][(9002, "w1")]["admissible"] is False
    assert saturated(None) is None


def test_selection_takes_the_lowest_seed_ids_not_the_best_performing():
    """The rule must be unsteerable: make the LOW seeds the slowest and it must still pick
    them, because a rule that prefers fast environments chooses the answer."""
    s = _shares()
    for t in (9001, 9002, 9003, 9005):
        for w in M_WINDOWS:
            s[(t, w)] = 0.79          # admissible, but the worst of the admissible set
    sel = select_topologies(read_m0(s))
    assert sel["verdict"] == V_DESIGN_READY
    assert sel["topologies"] == [9001, 9002, 9003, 9005]
    assert len(sel["environments"]) == 16


def test_selection_requires_all_four_windows_so_the_design_stays_crossed():
    s = _shares(overrides={(9001, "w2"): 0.93})      # 9001 fails one window only
    sel = select_topologies(read_m0(s))
    assert 9001 not in sel["topologies"]
    assert sel["topologies"] == [9002, 9003, 9005, 9101]


def test_selection_refuses_a_ragged_design_rather_than_shrinking_it():
    s = _shares(default=0.95)
    for t in (9001, 9002):
        for w in M_WINDOWS:
            s[(t, w)] = 0.66
    sel = select_topologies(read_m0(s))
    assert sel["verdict"] == V_DESIGN_SHORT and sel["qualified"] == [9001, 9002]
    assert "about the baseline" in sel["why"]


# --- the statistic ------------------------------------------------------------------------

def test_the_checkpoint_statistic_pairs_within_an_environment():
    """Reactive ranges 20-46 s across environments, so the pairing carries the leverage."""
    react = {e: v for e, v in zip(ENVS, (20.0, 30.0, 40.0, 50.0))}
    arm = {(e, 1): react[e] * 0.9 for e in ENVS}          # uniformly 10 % faster
    got = checkpoint_stats(arm, react, ENVS)
    assert got[1] == pytest.approx(-10.0)


def test_the_statistic_fails_loud_on_a_ragged_arm_or_a_missing_baseline():
    react = {e: 25.0 for e in ENVS}
    ragged = {(ENVS[0], 1): 20.0}                          # seed 1 is missing 3 environments
    with pytest.raises(ValueError, match="missing environments"):
        checkpoint_stats(ragged, react, ENVS)
    full = {(e, 1): 20.0 for e in ENVS}
    with pytest.raises(ValueError, match="no reactive baseline"):
        checkpoint_stats(full, {ENVS[0]: 25.0}, ENVS)


# --- L1 / L2 / L3: orientation is fixed ---------------------------------------------------

def test_l1_is_oriented_so_a_negative_statistic_reads_arm_beats_reactive():
    assert read_l1(_stats(-20.0, 0.0))["verdict"] == V_ARM_BEATS_REACTIVE
    assert read_l1(_stats(+20.0, 0.0))["verdict"] == V_REACTIVE_FASTER
    assert read_l1(_stats(-1.0, 0.0))["verdict"] == V_NOT_SEP
    # v1's actual readings: right size, wrong power -> must still be a tie on 16 noisy ones
    assert read_l1({s: -6.46 + 20.0 * ((-1) ** s) for s in range(N)})["verdict"] == V_NOT_SEP


def test_l2_and_l3_orientation_including_the_swap():
    assert read_l2(_stats(-20.0, 0.0))["verdict"] == V_GRAPH_FASTER
    assert read_l2(_stats(+20.0, 0.0))["verdict"] == V_POINTWISE_FASTER
    assert read_l2(_stats(-1.0, 0.0))["verdict"] == V_NOT_SEP
    # gnn SLOWER than gnnedge0 = positive median = sum costs (v1 read +26.69 %)
    assert read_l3(_stats(+26.0, 0.0))["verdict"] == V_SUM_COSTS
    assert read_l3(_stats(-26.0, 0.0))["verdict"] == V_SUM_HELPS
    assert read_l3(_stats(+1.0, 0.0))["verdict"] == V_SUM_NOT_SEP


def test_arm_vs_arm_pairs_raw_elapsed_not_two_relative_numbers():
    """A ratio of two already-relative statistics divides by a number that is near zero
    whenever an arm ties reactive -- which is exactly what v1's arms did. Caught by this
    file before any datum existed."""
    a = {(e, 1): 18.0 for e in ENVS}
    b = {(e, 1): 20.0 for e in ENVS}
    assert pair_checkpoint_stats(a, b, ENVS)[1] == pytest.approx(-10.0)
    assert pair_checkpoint_stats(b, a, ENVS)[1] == pytest.approx(+11.111, rel=1e-3)
    # a tie against reactive must not make the arm-vs-arm contrast explode
    react = {e: 20.0 for e in ENVS}
    assert checkpoint_stats(b, react, ENVS)[1] == pytest.approx(0.0)
    assert pair_checkpoint_stats(a, b, ENVS)[1] == pytest.approx(-10.0)


def test_arm_vs_arm_fails_loud_on_a_ragged_environment_set():
    a = {(e, 1): 18.0 for e in ENVS}
    b = {(ENVS[0], 1): 20.0}
    with pytest.raises(ValueError, match="missing environment"):
        pair_checkpoint_stats(a, b, ENVS)


def test_reads_refuse_a_short_checkpoint_set_rather_than_relaxing_the_bar():
    short = {s: -20.0 for s in range(5)}
    assert read_l1(short)["verdict"] == V_UNREADABLE
    assert read_l2(short)["verdict"] == V_UNREADABLE
    assert read_l3(short)["verdict"] == V_UNREADABLE


# --- L5: the design-validation bar --------------------------------------------------------

def test_l5_fires_when_the_design_delivers_the_projected_sd():
    tight = {a: {s: -8.0 + 0.5 * ((-1) ** s) for s in range(N)} for a in M_ARMS}
    r = read_l5(tight)
    assert r["verdict"] == V_POWER_DELIVERED and r["worst_sd"] <= M_DESIGN_SD_BAR


def test_l5_fails_when_the_sd_is_still_v1_sized_and_one_bad_arm_is_enough():
    loose = {a: {s: -8.0 + 0.5 * ((-1) ** s) for s in range(N)} for a in M_ARMS}
    loose[M_GNN] = {s: -8.0 + 25.0 * ((-1) ** s) for s in range(N)}
    r = read_l5(loose)
    assert r["verdict"] == V_POWER_NOT_DELIVERED and r["worst_sd"] > M_DESIGN_SD_BAR
    assert read_l5({})["verdict"] == V_UNREADABLE


# --- L4: the composite --------------------------------------------------------------------

def _l1s(verdict):
    return {a: {"verdict": verdict} for a in M_ARMS}


def test_l4_fires_on_any_registered_winner_and_names_it():
    l1 = _l1s(V_REACTIVE_FASTER); l1[M_GRAPH] = {"verdict": V_ARM_BEATS_REACTIVE}
    r = read_l4(l1, {"verdict": V_POWER_DELIVERED})
    assert r["verdict"] == V_LEARNED_BEATS_REACTIVE and r["winners"] == [M_GRAPH]


def test_l4_a_positive_stands_even_if_the_design_bar_missed():
    """A bar that fires, fires -- under-power makes negatives weak, not positives wrong."""
    l1 = _l1s(V_NOT_SEP); l1[M_GRAPH] = {"verdict": V_ARM_BEATS_REACTIVE}
    r = read_l4(l1, {"verdict": V_POWER_NOT_DELIVERED})
    assert r["verdict"] == V_LEARNED_BEATS_REACTIVE


def test_l4_refuses_to_call_an_underpowered_negative_a_tie():
    """THE point of v2: repeating v1's mistake at a bigger n would be worse, not better."""
    powered = read_l4(_l1s(V_NOT_SEP), {"verdict": V_POWER_DELIVERED})
    assert powered["verdict"] == V_NO_LEARNED_BEATS_REACTIVE and powered["interpretable"] is True
    weak = read_l4(_l1s(V_NOT_SEP), {"verdict": V_POWER_NOT_DELIVERED})
    assert weak["verdict"] == V_NO_LEARNED_BEATS_REACTIVE and weak["interpretable"] is False
    assert "UNINTERPRETABLE" in weak["why"] and "same defect v1 had" in weak["why"]


def test_l4_an_unreadable_arm_is_missing_never_a_loss():
    l1 = _l1s(V_REACTIVE_FASTER); l1[M_GNN] = {"verdict": V_UNREADABLE}
    r = read_l4(l1, {"verdict": V_POWER_DELIVERED})
    assert r["missing"] == [M_GNN] and M_GNN not in r["losers"]
    assert read_l4({a: {"verdict": V_UNREADABLE} for a in M_ARMS},
                   {"verdict": V_POWER_DELIVERED})["verdict"] == V_UNREADABLE


def test_l4_ignores_an_arm_that_was_never_registered():
    l1 = _l1s(V_REACTIVE_FASTER); l1["posthoc"] = {"verdict": V_ARM_BEATS_REACTIVE}
    assert read_l4(l1, {"verdict": V_POWER_DELIVERED})["verdict"] == V_NO_LEARNED_BEATS_REACTIVE


# --- L6: the window axis -------------------------------------------------------------------

def test_l6_reports_a_sign_flip_across_windows():
    per = {a: dict(zip(M_WINDOWS, (-8.0, -7.0, -9.0, -6.0))) for a in M_ARMS}
    per[M_GRAPH] = dict(zip(M_WINDOWS, (-8.0, +4.0, -9.0, -6.0)))
    r = read_l6(per)
    assert r["arms"][M_GRAPH]["verdict"] == V_WINDOW_FLIPS
    assert r["arms"][M_TWIN]["verdict"] == V_WINDOW_CONSISTENT
    assert r["flips"] == [M_GRAPH] and "never be quoted without this" in r["why"]


def test_l6_is_unreadable_for_an_arm_missing_a_window():
    per = {a: dict(zip(M_WINDOWS, (-8.0, -7.0, -9.0, -6.0))) for a in M_ARMS}
    per[M_GNN] = {"w0": -8.0}
    r = read_l6(per)
    assert r["arms"][M_GNN]["verdict"] == V_UNREADABLE
    assert r["arms"][M_GNN]["missing"] == ["w1", "w2", "w3"]
    assert not r["flips"]
