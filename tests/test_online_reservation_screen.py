import copy

import numpy as np

from scripts_cosim.online_reservation_screen import choose, execute, initial
from src.placement.radical.dag_reservation import problem


def test_hidden_workflow_cannot_change_initial_decision():
    b = problem(274000, jobs=4, ops=4, hosts=3)
    altered = copy.deepcopy(b)
    altered['p'][3] *= 100
    altered['locks'][3] = (1 << 12) - 1
    state = initial(b)
    visible_arrivals = np.array([0., 0., 0., np.inf])
    for mode in ('hand', 'portfolio', 'beam2', 'beam2v2'):
        assert choose(b, state, visible_arrivals, mode)[0] == choose(
            altered, state, visible_arrivals, mode)[0]


def test_arrivals_and_precedence_hold_under_online_replanning():
    b = problem(274001, jobs=4, ops=4, hosts=3)
    arrivals = np.array([0., 0., 0., 7.])
    for mode in ('hand', 'portfolio', 'beam2', 'beam2v2'):
        state = execute(b, arrivals, mode)['state']
        assert np.all(state['starts'] >= arrivals[:, None])
        for j in range(4):
            for k in range(4):
                for z in range(k):
                    if int(b['predecessors'][j, k]) & (1 << z):
                        assert state['starts'][j, k] >= state['ends'][j, z]
        for host in range(3):
            assigned = np.argwhere(state['assignment'] == host)
            for x, (j, k) in enumerate(assigned):
                for j2, k2 in assigned[x + 1:]:
                    assert (state['ends'][j, k] <= state['starts'][j2, k2] or
                            state['ends'][j2, k2] <= state['starts'][j, k])
