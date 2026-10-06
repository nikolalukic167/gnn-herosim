import numpy as np
import pytest
import torch

from src.placement.radical.dispatch import DispatchNative
from src.placement.radical.environment import initial, problem
from src.policy.dispatch_priority.model import PriorityNet, features
from src.policy.dispatch_priority.train import ranks, train_epoch
from src.policy.workflow.train import seed_everything


def test_rank_conversion_is_stable_permutation():
    scores = np.array([[2., 1., 1., 4.]])
    assert ranks(scores).tolist() == [[2, 0, 1, 3]]


def test_batched_flat_ranks_are_reshaped_for_native_cost(tmp_path):
    from src.policy.dispatch_priority.train import costs

    physical = problem(43, jobs=3, ops=3)
    assignment = initial(physical, "affinity")
    priority = np.arange(9)[None]
    value = costs(DispatchNative(tmp_path), [physical], assignment[None], priority)
    assert value.shape == (1,) and np.isfinite(value[0])


@pytest.mark.parametrize("arm", ["gnn", "mpoff", "mlp_hand"])
def test_training_is_deterministic_and_ignores_labels(arm):
    torch.set_num_threads(1)
    physical = problem(43, jobs=3, ops=3)
    assignment = initial(physical, "affinity")
    x, adjacency = features(physical, assignment, arm == "mlp_hand")
    tx = torch.from_numpy(np.stack([x, x]))
    ta = torch.from_numpy(np.stack([adjacency, adjacency]))
    target = torch.arange(9).float()[None].repeat(2, 1) / 8
    states = []
    for _ in range(2):
        seed_everything(55)
        model = PriorityNet(x.shape[-1], mp=arm == "gnn")
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        train_epoch(model, optimizer, tx, ta, target, torch.Generator().manual_seed(55))
        states.append({key: value.clone() for key, value in model.state_dict().items()})
    assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])
    physical["teacher_priority"] = np.arange(9)
    physical["teacher_cost"] = 0
    again = features(physical, assignment, arm == "mlp_hand")
    assert all(np.array_equal(left, right) for left, right in zip((x, adjacency), again))


def test_mp_off_is_adjacency_invariant_and_gnn_is_sensitive():
    seed_everything(55)
    physical = problem(43, jobs=3, ops=3)
    x, adjacency = features(physical, initial(physical, "affinity"))
    x = torch.from_numpy(x)[None]
    adjacency = torch.from_numpy(adjacency)[None]
    for mp in (False, True):
        model = PriorityNet(x.shape[-1], mp=mp).eval()
        with torch.inference_mode():
            left = model(x, adjacency)
            right = model(x, adjacency * 0)
        assert torch.equal(left, right) == (not mp)


def test_model_priority_has_native_objective(tmp_path):
    physical = problem(43, jobs=3, ops=3)
    assignment = initial(physical, "affinity")
    x, adjacency = features(physical, assignment)
    model = PriorityNet(x.shape[-1]).eval()
    with torch.inference_mode():
        priority = ranks(model(torch.from_numpy(x)[None], torch.from_numpy(adjacency)[None]).numpy())[0].reshape(3, 3)
    cost, _ = DispatchNative(tmp_path).score_priority(physical, assignment, priority)
    assert np.isfinite(cost)
