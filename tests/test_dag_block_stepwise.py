import numpy as np

from src.placement.radical.dag_block_moves import apply_proposal, candidate_pool
from src.placement.radical.dag_reservation import problem
from src.placement.radical.dag_reservation_bridge import BridgedDag, baseline, chronological


def test_decoder_translation_and_all_block_families(tmp_path):
    b = problem(161299, 8, 6, 8)
    engine = BridgedDag(tmp_path)
    cost, a, rank, decoder, _ = baseline(engine, b)
    _, starts, _ = engine.plan(b, a, rank, decoder)
    translated = chronological(rank, starts) if decoder == 0 else rank
    translated_cost, translated_starts, _ = engine.plan(b, a, translated, 1)
    assert translated_cost <= cost
    original_assignment = a.copy()
    original_rank = translated.copy()
    pool = candidate_pool(b, a, translated, translated_starts, 24)
    assert {p.family for p in pool} == {'critical_chain', 'lock_chain', 'setup_class', 'block_swap'}
    assert any(p.flip_hosts for p in pool)
    assert pool == candidate_pool(b, a, translated, translated_starts, 24)
    for proposal in pool:
        moved_assignment, moved_rank = apply_proposal(b, a, translated, proposal)
        moved_cost, _, _ = engine.plan(b, moved_assignment, moved_rank, 1)
        assert np.isfinite(moved_cost)
        assert sorted(moved_rank.ravel().tolist()) == list(range(a.size))
    assert np.array_equal(a, original_assignment)
    assert np.array_equal(translated, original_rank)
