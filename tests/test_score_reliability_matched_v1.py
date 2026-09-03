"""Dry-run the reliability_matched_v1 scorer at its registered n, before the gate runs.

`docs/lessons.md` 2026-09-03: "check the instrument can execute the n before signing it"
— Phase 3 registered n=120 against an analyzer that refused above n=22 and found out
after 615 episodes. This file is that check, done first: it drives the scorer's arithmetic
on synthetic count vectors shaped exactly like the real thing (16 vs 16 over 20 cells),
covering all four registered outcomes.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts_cosim" / "important"))

from score_reliability_matched_v1 import (  # noqa: E402
    ALPHA, N_CELLS, PRIMARY_PCT, SEEDS, ranksum_greater,
)


def test_registered_shape_is_what_the_gate_will_produce():
    assert len(SEEDS) == 16
    assert N_CELLS == 20
    assert PRIMARY_PCT == 50.0


def test_phase1_shaped_separation_clears_alpha():
    """The Phase 1 outcome, replayed: if the matched MLP collapses like the fabric-blind
    one did, the instrument resolves it comfortably."""
    mlp = [0, 0, 8, 10, 5, 3, 0, 11, 0, 0, 21, 16, 26, 0, 0, 7]
    gnn = [0] * 16
    _, p = ranksum_greater([float(x) for x in mlp], [float(x) for x in gnn])
    assert p < ALPHA, p


def test_a_single_collapsing_draw_does_not_clear_alpha():
    """One unlucky MLP draw against an all-clean GNN group must NOT read as a difference
    — the instrument has to be unable to manufacture the claim from one cell."""
    mlp = [0] * 15 + [3]
    gnn = [0] * 16
    _, p = ranksum_greater([float(x) for x in mlp], [float(x) for x in gnn])
    assert p > ALPHA, p


def test_all_zero_versus_all_zero_is_a_tie_not_a_pass():
    mlp = [0] * 16
    gnn = [0] * 16
    _, p = ranksum_greater([float(x) for x in mlp], [float(x) for x in gnn])
    assert p > 0.9, p


def test_direction_is_one_sided_and_correct_way_round():
    """H1 is 'MLP burden greater'. A GNN-worse world must not clear the bar."""
    mlp = [0] * 16
    gnn = [0, 0, 8, 10, 5, 3, 0, 11, 0, 0, 21, 16, 26, 0, 0, 7]
    _, p = ranksum_greater([float(x) for x in mlp], [float(x) for x in gnn])
    assert p > 0.9, p


def test_p_value_is_reproducible_bit_for_bit():
    """The permutation seed is part of the registration."""
    mlp = [0, 0, 8, 10, 5, 3, 0, 11, 0, 0, 21, 16, 26, 0, 0, 7]
    gnn = [0] * 16
    a = ranksum_greater([float(x) for x in mlp], [float(x) for x in gnn])
    b = ranksum_greater([float(x) for x in mlp], [float(x) for x in gnn])
    assert a == b


def test_midrank_ties_are_handled():
    """Heavily tied vectors are the expected case here (many zeros), so the rank
    statistic must use midranks rather than an arbitrary order."""
    _, p_a = ranksum_greater([0.0] * 8 + [1.0] * 8, [0.0] * 16)
    _, p_b = ranksum_greater([1.0] * 8 + [0.0] * 8, [0.0] * 16)
    assert p_a == p_b, "rank statistic must not depend on input order"


@pytest.mark.parametrize("missing", ["eval_drawgate_backbone.json"])
def test_partial_gate_refuses_to_score(tmp_path, missing):
    from score_reliability_matched_v1 import load_cells
    with pytest.raises(SystemExit) as exc:
        load_cells(tmp_path)
    assert "FAIL LOUD" in str(exc.value)
