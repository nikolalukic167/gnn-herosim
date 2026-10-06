import numpy as np
import torch

from src.policy.dispatch_value.model import ReadySetNet
from src.policy.dispatch_value.train import listwise_loss, numpy_head


def test_listwise_loss_rewards_low_cost_action():
    feasible = torch.tensor([[[True, True, False]]])
    costs = torch.tensor([[[100.0, 120.0, float("nan")]]])
    rule = torch.tensor([120.0])
    good = listwise_loss(torch.tensor([[[4.0, -4.0, -1e9]]]), feasible, costs, rule)
    bad = listwise_loss(torch.tensor([[[-4.0, 4.0, -1e9]]]), feasible, costs, rule)
    assert good < bad


def test_numpy_head_matches_torch_argmax():
    torch.manual_seed(9)
    model = ReadySetNet(5, hidden=8, layers=2, mp=True)
    x = torch.randn(1, 6, 5)
    adjacency = torch.randn(1, 4, 6, 6)
    encoded = model.encode(x, adjacency)
    dynamic = np.random.default_rng(9).normal(size=(6, 13)).astype(np.float32)
    feasible = np.array([True, False, True, True, False, True])
    with torch.inference_mode():
        expected = int(model.score(encoded, torch.from_numpy(dynamic)[None, None],
                                   torch.from_numpy(feasible)[None, None])[0, 0].argmax())
    assert numpy_head(model, encoded[0].detach().numpy())(dynamic, feasible) == expected
