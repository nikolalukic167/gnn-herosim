import copy
import pickle

import numpy as np
import pytest

from src.placement.radical.block_moves import scaled_problem
from src.placement.radical.dag_reservation import DagReservation, problem
from src.placement.radical.dag_reservation_live import replay
from src.placement.radical.dag_resume import resume, snapshot, snapshot_from_result


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    return DagReservation(tmp_path_factory.mktemp("dag_resume_native"))


def events(result):
    return sorted(tuple(sorted(event.items())) for event in result["events"])


def assert_same(actual, expected):
    assert actual["objective"] == expected["objective"]
    np.testing.assert_array_equal(actual["starts"], expected["starts"])
    np.testing.assert_array_equal(actual["ends"], expected["ends"])
    assert events(actual) == events(expected)


def chain_fixture(jobs=3, hosts=3):
    b = scaled_problem(276000, jobs, 2, hosts)
    b["predecessors"] = np.tile(np.array([0, 1], dtype=np.uint64), (jobs, 1))
    b["p"][:] = np.inf
    b["p"][:, :, :2] = 2
    b["locks"][:] = 0
    b["domains"][:] = 0
    b["types"][:] = 0
    b["setup"][:] = 0
    assignment = np.zeros((jobs, 2), dtype=np.int64)
    rank = np.arange(jobs * 2).reshape(jobs, 2)
    release = np.zeros((jobs, 2))
    return b, assignment, rank, release


@pytest.mark.parametrize("seed", [276001, 276002, 276003])
@pytest.mark.parametrize("mode", [0, 1])
def test_resume_matches_independent_and_native_at_event_and_interior_cuts(engine, seed, mode):
    b = problem(seed, 3, 6, 4)
    a = b["p"].argmin(-1)
    rank = np.random.default_rng(seed).permutation(a.size).reshape(a.shape)
    cost, starts, ends = engine.plan(b, a, rank, mode)
    release = starts if mode else np.zeros(a.shape)
    expected = replay(b, a, rank, release)
    assert cost == expected["objective"]
    np.testing.assert_array_equal(ends, expected["ends"])
    cuts = {0., float(ends.max() + 1), float(starts.ravel()[2]), float(ends.ravel()[3]),
            float((starts.ravel()[1] + ends.ravel()[1]) / 2)}
    for at in cuts:
        state = snapshot(b, a, rank, release, at)
        before = pickle.dumps(state)
        assert_same(resume(b, state, a, rank, release), expected)
        assert pickle.dumps(state) == before
        assert_same(resume(b, copy.deepcopy(state), a, rank, release), expected)


def test_idle_cut_and_changed_suffix_leave_executed_prefix_intact():
    b, a, rank, release = chain_fixture()
    release[1:, 0] = 10
    expected_before = replay(b, a, rank, release)
    state = snapshot(b, a, rank, release, 7.)
    changed = rank.copy()
    changed[1], changed[2] = rank[2].copy(), rank[1].copy()
    changed_a = a.copy()
    changed_a[2] = 1
    expected = replay(b, changed_a, changed, release)
    result = resume(b, state, changed_a, changed, release)
    assert_same(result, expected)
    assert [e for e in result["events"] if e["job"] == 0] == [
        e for e in expected_before["events"] if e["job"] == 0]


def test_fork_join_and_simultaneous_completion_survive_resume(engine):
    b = scaled_problem(276010, 1, 4, 2)
    b["predecessors"] = np.array([[0, 1, 1, 6]], dtype=np.uint64)
    b["p"][:] = 3
    b["locks"][:] = 0
    b["domains"][:] = 0
    b["setup"][:] = 0
    b["types"][:] = 0
    a = np.array([[0, 0, 1, 0]])
    rank = np.arange(4).reshape(1, 4)
    release = np.zeros(a.shape)
    expected = replay(b, a, rank, release)
    assert expected["starts"].tolist() == [[0, 3, 3, 6]]
    for at in (2., 3., 4., 6.):
        assert_same(resume(b, snapshot(b, a, rank, release, at), a, rank, release), expected)
    assert engine.replay(b, a, rank, release)[0] == expected["objective"]


@pytest.mark.parametrize("resource", ["lock", "domain"])
def test_active_resource_occupancy_is_not_forgotten(resource):
    b, a, rank, release = chain_fixture(hosts=3)
    b["p"][:] = np.inf
    for j in range(3):
        b["p"][j, :, j] = 5
        b["p"][j, :, (j + 1) % 3] = 5
    a[:] = np.arange(3)[:, None]
    if resource == "lock":
        b["locks"][:] = 1
    else:
        b["domains"][:] = 1
    expected = replay(b, a, rank, release)
    assert expected["starts"][2, 0] > 0
    for at in (1., 5.):
        assert_same(resume(b, snapshot(b, a, rank, release, at), a, rank, release), expected)


def test_setup_history_and_remaining_running_duration_survive_resume():
    b, a, rank, release = chain_fixture(jobs=2, hosts=2)
    b["types"][:] = [[0, 1], [2, 3]]
    b["setup"][0, 1] = 7
    b["setup"][1, 2] = 11
    b["setup"][2, 3] = 3
    expected = replay(b, a, rank, release)
    assert expected["ends"].tolist() == [[2, 11], [24, 29]]
    for at in (1., 2., 4., 11., 18., 24.):
        assert_same(resume(b, snapshot(b, a, rank, release, at), a, rank, release), expected)


@pytest.mark.parametrize("at", [-1., np.nan, np.inf])
def test_snapshot_rejects_bad_cut(at):
    b, a, rank, release = chain_fixture()
    with pytest.raises(ValueError):
        snapshot(b, a, rank, release, at)


@pytest.mark.parametrize("bad", ["rank", "assignment", "release"])
def test_resume_rejects_invalid_plan(bad):
    b, a, rank, release = chain_fixture()
    state = snapshot(b, a, rank, release, 1.)
    if bad == "rank":
        rank[:] = 0
    elif bad == "assignment":
        a[:] = 99
    else:
        release[2, 0] = np.nan
    with pytest.raises(ValueError):
        resume(b, state, a, rank, release)


def commitment_fixture():
    b, a, rank, release = chain_fixture(hosts=2)
    release[1, 0] = 10
    release[2, 0] = 15
    b["types"][:] = [[0, 0], [1, 1], [2, 2]]
    b["setup"][2, 1] = 5
    return b, a, rank, release


def test_future_commitment_retains_full_start_end_and_setup():
    b, a, rank, release = commitment_fixture()
    expected = replay(b, a, rank, release)
    state = snapshot(b, a, rank, release, 1., committed=(2,))
    result = resume(b, state, a, rank, release)
    assert_same(result, expected)
    changed = release.copy()
    changed[2, 0] = 20
    result = resume(b, state, a, rank, changed)
    frozen = [e for e in result["events"] if (e["job"], e["operation"]) == (1, 0)]
    assert frozen == [e for e in expected["events"] if (e["job"], e["operation"]) == (1, 0)]


@pytest.mark.parametrize("kind", ["overlap", "setup", "assignment", "release"])
def test_future_commitment_conflicts_fail_loudly(kind):
    b, a, rank, release = commitment_fixture()
    state = snapshot(b, a, rank, release, 1., committed=(2,))
    if kind == "overlap":
        release[2, 0] = 9
    elif kind == "setup":
        release[2, 0] = 5
    elif kind == "assignment":
        a[1, 0] = 1
    else:
        release[1, 0] = 11
    with pytest.raises(ValueError):
        resume(b, state, a, rank, release)


@pytest.mark.parametrize("at,node", [(1., (0, 0)), (3., (0, 0)), (3., (0, 1))])
def test_running_and_completed_assignment_cannot_change(at, node):
    b, a, rank, release = chain_fixture()
    state = snapshot(b, a, rank, release, at)
    a[node] = 1
    with pytest.raises(ValueError):
        resume(b, state, a, rank, release)


def test_second_decision_keeps_first_decision_execution_history():
    b, a, rank, release = chain_fixture()
    release[1:, 0] = 10
    first_state = snapshot(b, a, rank, release, 7.)
    first_rank = rank.copy()
    first_rank[1], first_rank[2] = rank[2].copy(), rank[1].copy()
    first = resume(b, first_state, a, first_rank, release)
    second_state = snapshot_from_result(b, a, first_rank, release, first, 11.)
    second_a = a.copy()
    second_a[1] = 1
    second = resume(b, second_state, second_a, rank, release)
    immutable = [e for e in first["events"] if e["start"] <= 11]
    for event in immutable:
        assert event in second["events"]
    independent = replay(b, second_a, rank, second["starts"])
    assert_same(second, independent)


def test_snapshot_owns_arrays_and_rejects_changed_problem_and_corrupted_history():
    b, a, rank, release = chain_fixture()
    state = snapshot(b, a, rank, release, 1.)
    a[:] = 1
    rank[:] = np.flip(rank)
    release[:] = 100
    expected = replay(b, state.assignment, state.rank, state.release)
    assert_same(resume(b, state, state.assignment, state.rank, state.release), expected)
    changed_b = copy.deepcopy(b)
    changed_b["p"][0, 0, 0] += 1
    with pytest.raises(ValueError):
        resume(changed_b, state, state.assignment, state.rank, state.release)
    broken = copy.deepcopy(state)
    broken.events[0]["end"] += 1
    with pytest.raises(ValueError):
        resume(b, broken, state.assignment, state.rank, state.release)


@pytest.mark.parametrize("committed", [(-1,), (6,), ("0",)])
def test_invalid_commitment_ids_rejected(committed):
    b, a, rank, release = chain_fixture()
    with pytest.raises(ValueError):
        snapshot(b, a, rank, release, 1., committed=committed)


def test_near_simultaneous_float_completions_match_reference_tolerance():
    b, a, rank, release = chain_fixture(jobs=2, hosts=2)
    a[1] = 1
    b["p"][0, 0, 0] = 1
    b["p"][1, 0, 1] = 1 + 5e-10
    expected = replay(b, a, rank, release)
    for at in (.5, 1 - 2e-10, 1., 1 + 2e-10):
        assert_same(resume(b, snapshot(b, a, rank, release, at), a, rank, release), expected)


def test_arbitrary_idle_cut_does_not_release_work_early():
    b, a, rank, release = chain_fixture()
    release[1:, 0] = 10
    expected = replay(b, a, rank, release)
    at = 10 - 5e-10
    assert_same(resume(b, snapshot(b, a, rank, release, at), a, rank, release), expected)


@pytest.mark.parametrize("at", [0., 2., 4., 8., 12.])
def test_predispatch_cut_unchanged_resume_preserves_full_schedule(at):
    b, a, rank, release = chain_fixture()
    expected = replay(b, a, rank, release)
    state = snapshot(b, a, rank, release, at, include_starts_at_cut=False)
    assert_same(resume(b, state, a, rank, release), expected)


def test_predispatch_completion_frees_resource_and_updates_setup_before_new_choice():
    b, a, rank, release = chain_fixture()
    b["types"][:] = [[0, 1], [2, 2], [3, 3]]
    b["setup"][0, 1] = 7
    b["setup"][0, 2] = 3
    state = snapshot(b, a, rank, release, 2., include_starts_at_cut=False)
    changed = np.array([[0, 5], [1, 2], [3, 4]])
    result = resume(b, state, a, changed, release)
    expected = replay(b, a, changed, release)
    assert_same(result, expected)
    assert result["starts"][1, 0] == 2
    assert result["ends"][1, 0] == 7
    assert result["ends"][0, 0] == 2
    assert result["starts"][0, 1] > 2
    event = next(e for e in result["events"] if e["event"] == "start" and e["job"] == 1)
    assert event["setup"] == 3


def test_predispatch_can_change_next_choice_without_moving_other_running_operation():
    b, a, rank, release = chain_fixture()
    a[1] = 1
    b["p"][1, 0, 1] = 5
    original = replay(b, a, rank, release)
    state = snapshot(b, a, rank, release, 2., include_starts_at_cut=False)
    changed = np.array([[0, 5], [1, 4], [2, 3]])
    result = resume(b, state, a, changed, release)
    assert_same(result, replay(b, a, changed, release))
    assert result["starts"][2, 0] == 2
    assert result["starts"][1, 0] == original["starts"][1, 0] == 0
    assert result["ends"][1, 0] == original["ends"][1, 0] == 5
    invalid = a.copy()
    invalid[1, 0] = 0
    with pytest.raises(ValueError):
        resume(b, state, invalid, changed, release)


def test_predispatch_second_decision_changes_action_at_next_completion():
    b, a, rank, release = chain_fixture()
    first_state = snapshot(b, a, rank, release, 2., include_starts_at_cut=False)
    first_rank = np.array([[0, 5], [1, 2], [3, 4]])
    first = resume(b, first_state, a, first_rank, release)
    assert first["starts"][1, 0] == 2
    assert first["starts"][1, 1] == 4
    second_state = snapshot_from_result(
        b, a, first_rank, release, first, 4., include_starts_at_cut=False)
    second_rank = np.array([[0, 5], [1, 4], [2, 3]])
    second = resume(b, second_state, a, second_rank, release)
    assert second["starts"][2, 0] == 4
    assert second["starts"][1, 1] > 4
    for node in ((0, 0), (1, 0)):
        assert second["starts"][node] == first["starts"][node]
        assert second["ends"][node] == first["ends"][node]
    assert_same(second, replay(b, a, second_rank, second["starts"]))
