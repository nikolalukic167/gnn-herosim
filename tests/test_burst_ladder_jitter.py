"""burst_ladder_v1 Amendment 1: the perturbed-draw builder keeps groups together and the event order."""
import pytest

from scripts_cosim.burst_ladder_jitter import CAP_S, GAP_FRAC, jitter_timestamps

TS = [0.0, 0.0, 0.0, 2.0, 2.0, 10.0, 10.5, 10.5, 40.0]
GROUPS = [0, 0, 0, 1, 1, 2, 3, 3, 4]


def test_groups_stay_together_order_kept_and_shift_bounded():
    new = jitter_timestamps(TS, GROUPS, seed=1)
    assert all(b >= a for a, b in zip(new, new[1:]))
    by_group = {}
    for t, g in zip(new, GROUPS):
        by_group.setdefault(g, set()).add(t)
    assert all(len(v) == 1 for v in by_group.values())
    nxt = {0: 2.0, 1: 10.0, 2: 10.5, 3: 40.0}
    for t0, t1, g in zip(TS, new, GROUPS):
        bound = min(CAP_S, GAP_FRAC * (nxt[g] - t0)) if g in nxt else CAP_S
        assert 0.0 <= t1 - t0 <= bound


def test_draws_are_deterministic_and_differ():
    assert jitter_timestamps(TS, GROUPS, seed=3) == jitter_timestamps(TS, GROUPS, seed=3)
    assert jitter_timestamps(TS, GROUPS, seed=1) != jitter_timestamps(TS, GROUPS, seed=2)


def test_refuses_split_group_and_unsorted_groups():
    with pytest.raises(ValueError, match="more than one timestamp"):
        jitter_timestamps([0.0, 1.0], [0, 0], seed=1)
    with pytest.raises(ValueError, match="not in timestamp order"):
        jitter_timestamps([5.0, 1.0], [0, 1], seed=1)
