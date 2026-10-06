"""Bipartite request-host scorer and two separately trained controls."""
import numpy as np
import torch
from torch import nn

from src.placement.residency import features


class ResidencyNet(nn.Module):
    def __init__(self, arm="gnn", hidden=48, layers=2):
        super().__init__()
        self.arm = arm
        if arm == "hand_mlp":
            # Full graph information is available to this fixed-size control.
            self.global_net = nn.Sequential(nn.Linear(8 * 3 * 20, 128), nn.ReLU(),
                                            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 24))
        else:
            self.embed = nn.Sequential(nn.Linear(20, hidden), nn.ReLU())
            self.updates = nn.ModuleList([nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(),
                                                        nn.Linear(hidden, hidden)) for _ in range(layers)])
            self.output = nn.Linear(hidden, 1)

    def forward(self, x):
        if self.arm == "hand_mlp":
            return self.global_net(x.flatten(1)).reshape(-1, 8, 3)
        edge = self.embed(x)
        for update in self.updates:
            if self.arm == "gnn":
                task = edge.mean(2, keepdim=True).expand_as(edge)
                host = edge.mean(1, keepdim=True).expand_as(edge)
            else:
                task = torch.zeros_like(edge)
                host = torch.zeros_like(edge)
            edge = torch.relu(edge + update(torch.cat([edge, task, host], -1)))
        return self.output(edge).squeeze(-1)


def decode(model, case):
    plan = []
    model.eval()
    with torch.no_grad():
        for step in range(len(case["requests"])):
            x = torch.from_numpy(features(case, plan)[None])
            scores = model(x)[0, step]
            plan.append(int(scores.argmax()))
    return plan


def decode_many(model, cases):
    plans = [[] for _ in cases]
    model.eval()
    with torch.no_grad():
        for step in range(8):
            x = torch.from_numpy(np.stack([features(c, p) for c, p in zip(cases, plans)]))
            chosen = model(x)[:, step].argmax(-1).tolist()
            for plan, host in zip(plans, chosen):
                plan.append(host)
    return plans
