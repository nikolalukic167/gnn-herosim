import numpy as np
import pytest
import torch
from src.placement.radical.dag_reservation import problem
from src.placement.radical.dag_reservation_bridge import BridgedDag, baseline
from src.policy.dag_reservation.model import ReservationProposalNet, decode, features
from src.policy.dag_reservation.train import seed_everything, train_epoch


def test_graph_contains_actual_fork_edges_and_decodes_feasible_plan(tmp_path):
    b = problem(160501, 2, 8, 4)
    engine = BridgedDag(tmp_path)
    _, a, rank, _, _ = baseline(engine, b)
    x, graph, eligible = features(b, a, rank)
    for j in range(2):
        for k in range(8):
            for z in range(k):
                edge = bool(int(b['predecessors'][j, k]) & (1 << z))
                assert (graph[0, j * 8 + k, j * 8 + z] > 0) == edge
                assert (graph[1, j * 8 + z, j * 8 + k] > 0) == edge
    seed_everything(9)
    model = ReservationProposalNet(x.shape[-1], 4, 16, 2, True)
    proposed_a, proposed_rank = decode(model, x, graph, eligible, 2, 8)
    assert np.isfinite(engine.plan(b, proposed_a, proposed_rank, 0)[0])
    assert np.isfinite(engine.plan(b, proposed_a, proposed_rank, 1)[0])
    assert sorted(proposed_rank.ravel().tolist()) == list(range(16))


@pytest.mark.parametrize('mp', [False, True])
def test_trainer_same_seed_same_weights(mp):
    b = problem(160502, 2, 8, 4)
    engine = BridgedDag('/tmp/herosim_dag_learning_test_build')
    _, a, rank, _, _ = baseline(engine, b)
    x, graph, eligible = features(b, a, rank)
    tensors = (torch.tensor(np.stack([x, x])), torch.tensor(np.stack([graph, graph])),
               torch.tensor(np.stack([eligible, eligible])), torch.tensor(np.stack([a.ravel(), a.ravel()])),
               torch.tensor(np.stack([rank.ravel() / 15, rank.ravel() / 15]), dtype=torch.float32))
    weights = []
    for _ in range(2):
        seed_everything(103)
        model = ReservationProposalNet(x.shape[-1], 4, 16, 2, mp)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        train_epoch(model, optimizer, tensors, 2, torch.Generator().manual_seed(103))
        weights.append({k: v.detach().clone() for k, v in model.state_dict().items()})
    assert all(torch.equal(weights[0][k], weights[1][k]) for k in weights[0])
