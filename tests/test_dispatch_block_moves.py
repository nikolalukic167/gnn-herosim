import io
import numpy as np
import pytest
from src.placement.radical.block_moves import scaled_problem, baseline, graph_features, move_pool, apply_move, Move
from src.placement.radical.dispatch_adaptive import AdaptiveDispatch
from src.placement.radical.dispatch_live import replay, herosim
from scripts_cosim.workflow_proposal_gate import configure_environment


@pytest.fixture(scope='module')
def engine(tmp_path_factory):
    return AdaptiveDispatch(tmp_path_factory.mktemp('blocks'))


@pytest.mark.parametrize('shape', [(4,3,8), (16,8,8), (32,8,16), (48,8,24)])
def test_scaled_native_and_high_lock_bits(engine, shape):
    b = scaled_problem(990001, *shape)
    assert b['domains'].shape == (shape[2], (shape[2]+3)//4)
    assert np.all(b['domains'].sum(1) == 1)
    b['locks'][0, 0] = 1 << (shape[2] - 1)
    cost, a, rank = baseline(engine, b)
    expected = replay(b, a, rank)
    assert cost == expected['objective']
    assert np.array_equal(engine.score_priority(b, a, rank)[1], expected['ends'])
    x, adj, hand, _ = graph_features(b, a)
    assert x[0, 8 + shape[2] - 1] == 1
    assert x.shape[0] == a.size and hand.shape[0] == a.size
    assert np.isfinite(hand).all() and adj.shape == (4, a.size, a.size)


def test_block_pool_and_exact_moves(engine):
    b = scaled_problem(990002, 8, 6, 8)
    cost, a, rank = baseline(engine, b)
    pool, _ = move_pool(b, a, rank, per_family=16)
    assert set(m.family for m in pool) == {'critical_chain', 'lock_chain', 'setup_class', 'block_swap'}
    assert pool == move_pool(b, a, rank, per_family=16)[0]
    before = rank.copy()
    for move in pool:
        candidate = apply_move(rank, move)
        assert np.array_equal(np.sort(candidate.ravel()), np.arange(rank.size))
        c, ends = engine.score_priority(b, a, candidate)
        ref = replay(b, a, candidate)
        assert c == ref['objective'] and np.array_equal(ends, ref['ends'])
    assert np.array_equal(rank, before)
    with pytest.raises(ValueError):
        apply_move(rank, Move('critical_chain', (0, 1), 1))


def test_scaled_live_adapter_restores_original(engine):
    from src.placement.radical import coordinator
    original = coordinator.MixedExecution
    configure_environment()
    b = scaled_problem(990003, 4, 3, 8)
    cost, a, rank = baseline(engine, b)
    actual = herosim(b, a, rank, io.StringIO(), 990003)
    assert actual['total_rtt'] * 1000 == pytest.approx(cost)
    assert len(actual['tasks']) == a.size
    assert coordinator.MixedExecution is original


def test_budget_control_selection_excludes_overrun(tmp_path):
    import json
    from scripts_cosim.complete_dispatch_block_budget_gate import selection
    (tmp_path / 'protocol_before_run.json').write_text(json.dumps({'protocol': {'budget_ms': 250}}))
    (tmp_path / 'read.json').write_text(json.dumps({'capture': {'wall_fast': .3, 'wall_overrun': .8, 'count_best': .9}}))
    cell = tmp_path / 'qualification' / '1'
    cell.mkdir(parents=True)
    row = {'arms': {'wall_fast': {'repeats': [{'time_ms': 240}, {'time_ms': 245}]},
                    'wall_overrun': {'repeats': [{'time_ms': 240}, {'time_ms': 251}]}}}
    (cell / 'read.json').write_text(json.dumps(row))
    _, eligible, selected = selection(tmp_path)
    assert eligible == ['wall_fast'] and selected == 'wall_fast'
    row['arms']['wall_fast']['repeats'][1]['time_ms'] = 251
    (cell / 'read.json').write_text(json.dumps(row))
    with pytest.raises(ValueError, match='no budget-valid hand arm'):
        selection(tmp_path)
