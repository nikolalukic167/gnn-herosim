"""Operation-priority encoder with matched MP-OFF and hand-graph controls."""
import numpy as np
import torch
from torch import nn
from src.policy.mixed.model import MixedNet

CONTRACT = "mixed_dispatch_priority_v1"


def features(problem, assignment, hand=False):
    from src.policy.mixed.model import features as mixed_features

    x, _, adjacency = mixed_features(problem)
    hosts = np.asarray(assignment).reshape(-1)
    assigned = np.take_along_axis(
        problem["p"], np.asarray(assignment)[..., None], axis=-1
    )[..., 0].reshape(-1, 1) / 20
    x = np.concatenate(
        [x, np.eye(problem["p"].shape[-1], dtype=np.float32)[hosts], assigned],
        axis=-1,
    ).astype(np.float32)
    if hand:
        one_hop = adjacency @ x
        two_hop = adjacency @ one_hop
        x = np.concatenate([x, *one_hop, *two_hop], axis=-1)
    return np.ascontiguousarray(x), adjacency

class PriorityNet(MixedNet):
    def __init__(self,feature_dim,hidden=16,layers=2,mp=True):
        super().__init__(feature_dim,hidden,layers,mp);self.output=nn.Linear(hidden,1)
    def forward(self,x,adj):
        return super().forward(x,torch.ones((*x.shape[:2],1),device=x.device,dtype=x.dtype),adj).squeeze(-1)
