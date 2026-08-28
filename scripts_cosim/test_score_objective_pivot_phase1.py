"""Tests for score_objective_pivot_phase1.py — the Phase 1 registered scorer.

Each statistic is checked against a hand-computable value, and the VOID gate is checked
for teeth in both directions (a wrong commit VOIDs; the registered identity passes).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "important"))

from score_objective_pivot_phase1 import (  # noqa: E402
    GNN_ARMS,
    GNN_SEEDS,
    PIN_COMMIT,
    REQUIRED_PROVENANCE,
    _midranks,
    assert_new_arm_provenance,
    mean_margins_vs_knative,
    rank_sum_p_less,
    sign_test_p_at_least,
)
from score_gnn_draw_study import fisher_exact_greater  # noqa: E402

BLOCKS = [f"g{i}/c{i}" for i in range(6)]
CELLS = [f"cell0{i}" for i in range(1, 6)]


def make_summary(arm_rtts, kn_rtt=100.0, provenance=None):
    """A synthetic 6-block x 5-cell summary. arm_rtts: {arm: rtt or per-cell list}."""
    summary = {}
    for b, blk in enumerate(BLOCKS):
        summary[blk] = {}
        for c, cell in enumerate(CELLS):
            entry = {"knative": {"total_rtt": kn_rtt}}
            for arm, rtt in arm_rtts.items():
                val = rtt[b * 5 + c] if isinstance(rtt, list) else rtt
                rec = {"total_rtt": val}
                if provenance is not None:
                    rec["provenance"] = dict(provenance.get(arm, provenance.get("*", {})))
                entry[arm] = rec
            summary[blk][cell] = entry
    return summary


# ---- exact statistics -------------------------------------------------------------

def test_sign_test_exact_values():
    # 13/16 at p=1/2: (C(16,13)+C(16,14)+C(16,15)+C(16,16)) / 2^16 = 697/65536
    assert sign_test_p_at_least(13, 16) == pytest.approx(697 / 65536)
    assert sign_test_p_at_least(0, 16) == 1.0
    assert sign_test_p_at_least(16, 16) == pytest.approx(1 / 65536)


def test_fisher_matches_registered_power_table():
    # The three values recorded in gnn_draw_study_v1's registration.
    assert fisher_exact_greater(8, 0, 7, 9) == pytest.approx(0.0087, abs=5e-5)
    assert fisher_exact_greater(7, 1, 7, 9) == pytest.approx(0.0507, abs=5e-5)
    assert fisher_exact_greater(10, 2, 7, 9) == pytest.approx(0.0398, abs=5e-5)


def test_midranks_ties():
    assert _midranks([10, 20, 20, 30]) == [1.0, 2.5, 2.5, 4.0]
    assert _midranks([5, 5, 5]) == [2.0, 2.0, 2.0]


def test_rank_sum_null_and_separation():
    # Identical groups: p must be far from significant.
    p_null = rank_sum_p_less([1, 2, 3, 4], [1, 2, 3, 4], n_perm=4000)
    assert 0.3 < p_null <= 1.0
    # Complete separation, GNN lower: p at the MC floor.
    p_sep = rank_sum_p_less([0] * 8, [10] * 8, n_perm=4000)
    assert p_sep < 0.01
    # Reversed direction must NOT be significant for the one-sided test.
    p_rev = rank_sum_p_less([10] * 8, [0] * 8, n_perm=4000)
    assert p_rev > 0.9


def test_rank_sum_reproducible():
    a = rank_sum_p_less([0, 1, 0, 3], [5, 7, 2, 9], n_perm=2000)
    b = rank_sum_p_less([0, 1, 0, 3], [5, 7, 2, 9], n_perm=2000)
    assert a == b  # fixed seed: bit-identical


# ---- margins ----------------------------------------------------------------------

def test_mean_margins_and_cell_count_guard():
    arm = GNN_ARMS[1]
    s = make_summary({arm: 80.0})
    assert mean_margins_vs_knative(s, [arm])[arm] == pytest.approx(-20.0)
    # 29 cells must fail loud, not average silently.
    del s[BLOCKS[0]][CELLS[0]][arm]
    with pytest.raises(SystemExit):
        mean_margins_vs_knative(s, [arm])


# ---- the VOID gate ----------------------------------------------------------------

def _good_provenance():
    return dict(REQUIRED_PROVENANCE)


def test_void_gate_passes_on_registered_identity():
    arm_rtts = {GNN_ARMS[s]: 80.0 for s in GNN_SEEDS}
    s = make_summary(arm_rtts, provenance={"*": _good_provenance()})
    assert assert_new_arm_provenance(s) == []


def test_void_gate_fires_on_wrong_commit():
    arm_rtts = {GNN_ARMS[s]: 80.0 for s in GNN_SEEDS}
    prov = {"*": _good_provenance()}
    bad = _good_provenance()
    bad["code_commit"] = "f" * 40
    prov[GNN_ARMS[9]] = bad
    void = assert_new_arm_provenance(make_summary(arm_rtts, provenance=prov))
    assert [arm for arm, _ in void] == [GNN_ARMS[9]]
    assert "code_commit" in void[0][1]


def test_void_gate_fires_on_dirty_tree_and_missing_arm():
    arm_rtts = {GNN_ARMS[s]: 80.0 for s in GNN_SEEDS if s != 16}
    prov = {"*": _good_provenance()}
    dirty = _good_provenance()
    dirty["code_dirty"] = True
    prov[GNN_ARMS[10]] = dirty
    void = assert_new_arm_provenance(make_summary(arm_rtts, provenance=prov))
    names = {arm for arm, _ in void}
    assert GNN_ARMS[10] in names       # dirty tree
    assert GNN_ARMS[16] in names       # absent arm
    # Old arms (1-8) are never provenance-gated: the frozen record is compared as recorded.
    assert GNN_ARMS[1] not in names


def test_pin_commit_is_the_recorded_one():
    assert PIN_COMMIT == "c08aa7ee140fd51e3d384f97df3f31b126df96ab"
