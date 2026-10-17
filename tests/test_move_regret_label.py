"""The move-regret labelling script (scripts_cosim/uncut_move_regret_label.py): the table row and the bucket rule."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import uncut_move_regret_label as L  # noqa: E402


def test_row_is_zero_on_the_label_clipped_and_marks_unscored_candidates():
    cands = [(0, 0), (1, 0), (2, 0), (3, 0)]
    moves = {(0, 0): 0.0, (1, 0): 5.0, (2, 0): 5000.0}              # the engine never scored (3, 0)
    row, unscored = L.table_row(cands, 0, moves, S0=100.0, n=10)
    assert row == [0.0, 0.5, L.D_MAX, L.D_MAX] and unscored == 1


def test_a_move_that_improves_s_costs_zero_and_an_undecided_task_has_no_signal():
    row, u = L.table_row([(0, 0), (1, 0)], 0, {(1, 0): -3.0}, S0=40.0, n=4)
    assert row == [0.0, 0.0] and u == 0
    row, u = L.table_row([(0, 0), (1, 0), (2, 0)], 1, None, S0=40.0, n=4)
    assert row == [0.0, 0.0, 0.0] and u == 0


def test_label_index_wins_over_a_scored_key():
    row, _ = L.table_row([(0, 0), (1, 0)], 1, {(0, 0): 20.0, (1, 0): 99.0}, S0=100.0, n=10)
    assert row == [2.0, 0.0]


def test_buckets():
    assert [L.bucket(n) for n in (1, 10, 11, 50, 51, 100, 101, 256)] == ["le10", "le10", "11_50", "11_50", "51_100", "51_100", "101_256", "101_256"]


def test_unscored_flags_mark_exactly_the_candidates_the_engine_never_scored():
    cands = [(0, 0), (1, 0), (2, 0), (3, 0)]
    assert L.unscored_flags(cands, 0, {(1, 0): 1.0, (2, 0): 2.0}) == [0, 0, 0, 1]
    assert L.unscored_flags(cands, 3, {(1, 0): 1.0}) == [1, 0, 1, 0]            # the label is never "unscored"
    assert L.unscored_flags(cands, 0, None) == [0, 0, 0, 0]                    # an undecided task carries no flags


class _G:
    n_tasks = 2
    task_logit_to_placement = [[(1, 1), (2, 1)], [(3, 1), (4, 1), (5, 1)]]
    tied_optimal_logit_plans = {"inf": [[0, 0]]}


def test_label_indices_follow_the_sidecar_and_fall_back_to_the_cache(monkeypatch):
    import uncut_replay_diag as R

    monkeypatch.setattr(R, "ALPHA", "inf", raising=False)
    assert L.label_indices("g", _G(), None) == [0, 0]
    assert L.label_indices("g", _G(), {"g": {"plans": [[[2, 1], [5, 1]]]}}) == [1, 2]


def test_label_indices_fail_loud_on_a_missing_entry_a_wrong_length_or_a_non_candidate():
    with pytest.raises(RuntimeError, match="no entry"):
        L.label_indices("g", _G(), {})
    with pytest.raises(RuntimeError, match="1 tasks"):
        L.label_indices("g", _G(), {"g": {"plans": [[[2, 1]]]}})
    with pytest.raises(RuntimeError, match="not a candidate"):
        L.label_indices("g", _G(), {"g": {"plans": [[[2, 1], [9, 9]]]}})
