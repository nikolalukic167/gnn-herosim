"""Rematerialization has explicit work and agrees with actual HeROsim."""
import io

import numpy as np

from scripts_cosim.workflow_proposal_gate import configure_environment
from src.placement.radical.dag_memory import replay as cache_replay
from src.placement.radical.dag_remat import effective_lock, problem, replay
from src.placement.radical.dag_remat_live import herosim
from src.placement.radical.dag_reservation_bridge import BridgedDag, baseline


def test_zero_action_preserves_cache_contract(tmp_path):
    b = problem(164001, 4, 5, 4)
    _, a, rank, _, _ = baseline(BridgedDag(tmp_path / 'build'), b, mode=0)
    keep = np.ones(a.shape, dtype=np.int8)
    action = np.zeros(a.shape, dtype=np.int8)
    old = cache_replay(b, a, rank, keep)
    new = replay(b, a, rank, keep, action)
    assert old['objective'] == new['objective']
    np.testing.assert_array_equal(old['ends'], new['ends'])
    assert new['counts']['rematerializations'] == 0


def test_rematerialization_matches_actual_herosim(tmp_path):
    b = problem(164001, 4, 5, 4)
    _, a, rank, _, _ = baseline(BridgedDag(tmp_path / 'build'), b, mode=0)
    keep = np.zeros(a.shape, dtype=np.int8)
    action = np.zeros(a.shape, dtype=np.int8)
    for j in range(a.shape[0]):
        for k in range(a.shape[1]):
            children = [z for z in range(k + 1, a.shape[1])
                        if int(b['predecessors'][j, z]) & (1 << k)]
            if children and all(np.isfinite(b['p'][j, k, int(a[j, z])]) for z in children):
                action[j, k] = 1
                break
        if action.any():
            break
    expected = replay(b, a, rank, keep, action)
    assert expected['counts']['rematerializations'] > 0
    configure_environment()
    actual = herosim(b, a, rank, keep, action, io.StringIO())
    assert actual['mixed_execution']['counts'] == expected['counts']
    for task in actual['tasks']:
        j, k = map(int, task['taskType']['name'][2:].split('_'))
        assert abs(task['doneTime'] * 1000 - expected['ends'][j, k]) < 1e-6


def test_cached_output_does_not_acquire_remat_lock(tmp_path):
    b = problem(164002, 4, 5, 4)
    _, a, rank, _, _ = baseline(BridgedDag(tmp_path / 'build'), b, mode=0)
    b['host_memory_mb'][:] = 10000
    keep = np.ones(a.shape, dtype=np.int8)
    action = np.zeros(a.shape, dtype=np.int8)
    chosen = None
    for j in range(a.shape[0]):
        for k in range(a.shape[1]):
            children = [z for z in range(k + 1, a.shape[1])
                        if int(b['predecessors'][j, z]) & (1 << k)]
            if children and all(np.isfinite(b['p'][j, k, int(a[j, z])]) for z in children):
                chosen = (j, k, children[0])
                break
        if chosen:
            break
    j, k, child = chosen
    action[j, k] = 1
    parents = {(j, child): [(j, z) for z in range(child)
                            if int(b['predecessors'][j, child]) & (1 << z)]}
    assert effective_lock(b, (j, child), parents, action, {(j, k): 0}) == int(b['locks'][j, child])
    assert replay(b, a, rank, keep, action)['objective'] == replay(
        b, a, rank, keep, np.zeros_like(action))['objective']
