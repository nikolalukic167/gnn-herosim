import numpy as np

from src.placement.radical.dag_memory import hand_retention, problem, replay
from src.placement.radical.dag_memory_live import herosim
from src.placement.radical.dag_reservation_bridge import BridgedDag, baseline
from src.placement.radical.dag_reservation_live import replay as prior_replay
from scripts_cosim.workflow_proposal_gate import configure_environment


def test_memory_off_preserves_dag_schedule(tmp_path):
    b = problem(162091, 4, 8, 4)
    engine = BridgedDag(tmp_path)
    _, a, rank, _, _ = baseline(engine, b, mode=0)
    prior = prior_replay(b, a, rank, np.zeros(a.shape))
    b['host_memory_mb'][:] = 1e9
    b['store_read_ms_per_mb'] = 0.
    b['peer_read_ms_per_mb'] = 0.
    current = replay(b, a, rank, hand_retention(b, a, 'keep_all'))
    assert current['objective'] == prior['objective']
    assert np.array_equal(current['ends'], prior['ends'])


def test_memory_capacity_and_actual_herosim_parity(tmp_path):
    configure_environment()
    b = problem(162092, 2, 8, 4)
    engine = BridgedDag(tmp_path)
    _, a, rank, _, _ = baseline(engine, b, mode=0)
    keep = hand_retention(b, a, 'keep_all')
    expected = replay(b, a, rank, keep)
    with (tmp_path / 'live.log').open('w') as log:
        actual = herosim(b, a, rank, keep, log)
    assert abs(actual['job_completion_sum'] * 1000 - expected['objective']) < 1e-6
    assert actual['mixed_execution']['counts'] == expected['counts']
    for task in actual['tasks']:
        job, op = map(int, task['taskType']['name'][2:].split('_'))
        assert abs(task['doneTime'] * 1000 - expected['ends'][job, op]) < 1e-6
