import torch

from src.policy.dispatch_state.model import ReadySetNet


def test_ready_set_model_masks_and_message_passing():
    torch.manual_seed(7)
    model = ReadySetNet(5, hidden=8, layers=2, mp=True)
    x = torch.randn(2, 6, 5)
    adjacency = torch.zeros(2, 4, 6, 6)
    adjacency[:, 0, 1, 0] = 1
    dynamic = torch.randn(2, 3, 6, 13)
    feasible = torch.zeros(2, 3, 6, dtype=torch.bool)
    feasible[:, :, :2] = True
    logits = model(x, adjacency, dynamic, feasible)
    assert logits.shape == (2, 3, 6)
    assert torch.all(logits[:, :, 2:] < -1e8)
    changed = model(x, adjacency.roll(1, -1), dynamic, feasible)
    assert not torch.equal(logits[:, :, :2], changed[:, :, :2])
