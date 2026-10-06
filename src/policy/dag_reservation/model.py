"""Matched graph and pointwise schedulers for irregular DAG workflows."""
import numpy as np
import torch
from torch import nn

CONTRACT = 'dag_reservation_proposal_v1'


def features(b, assignment, rank):
    p = np.asarray(b['p']); jobs, ops, hosts = p.shape; n = jobs * ops
    eligible = np.isfinite(p)
    duration = np.where(eligible, p, 0.) / 30.
    a = np.asarray(assignment).reshape(-1)
    r = np.asarray(rank).reshape(-1)
    lock = ((np.asarray(b['locks']).reshape(-1, 1) >> np.arange(hosts)) & 1).astype(np.float32)
    types = np.eye(len(b['setup']), dtype=np.float32)[np.asarray(b['types']).ravel()]
    host = np.eye(hosts, dtype=np.float32)[a]
    row = np.stack([np.repeat(np.arange(jobs) / jobs, ops), np.tile(np.arange(ops) / ops, jobs),
                    r / max(n - 1, 1), np.min(p, axis=2).ravel() / 30.,
                    np.repeat(np.min(p, axis=2).sum(1) / (30. * ops), ops)], axis=-1)
    base = np.concatenate([duration.reshape(n, hosts), eligible.reshape(n, hosts).astype(float),
                           host, lock, types, row], axis=-1).astype(np.float32)
    graph = np.zeros((4, n, n), dtype=np.float32)
    for j in range(jobs):
        for k in range(ops):
            i = j * ops + k
            for z in range(k):
                if int(b['predecessors'][j, k]) & (1 << z):
                    graph[0, i, j * ops + z] = 1.
                    graph[1, j * ops + z, i] = 1.
    graph[2] = (lock @ lock.T > 0).astype(np.float32)
    graph[3] = (host @ host.T > 0).astype(np.float32)
    for relation in graph: np.fill_diagonal(relation, 0.)
    graph /= np.maximum(graph.sum(-1, keepdims=True), 1.)
    hand = graph.mean(0)
    one = hand @ base; two = hand @ one; four = hand @ (hand @ two)
    x = np.concatenate([base, one, two, four], axis=-1).astype(np.float32)
    return x, graph, eligible.reshape(n, hosts).astype(np.float32)


class ReservationProposalNet(nn.Module):
    def __init__(self, feature_dim, hosts=16, hidden=64, layers=2, mp=True):
        super().__init__(); self.mp = mp
        self.input = nn.Linear(feature_dim, hidden)
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(hidden * 5, hidden), nn.ReLU(),
                                                   nn.Linear(hidden, hidden)) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.host = nn.Linear(hidden, hosts); self.order = nn.Linear(hidden, 1)

    def forward(self, x, graph, eligible):
        h = torch.relu(self.input(x))
        for layer, norm in zip(self.layers, self.norms):
            messages = [torch.bmm(graph[:, z], h) for z in range(4)] if self.mp else [torch.zeros_like(h) for _ in range(4)]
            h = norm(h + layer(torch.cat([h, *messages], dim=-1)))
        return self.host(h).masked_fill(eligible == 0, -1e9), self.order(h).squeeze(-1)


def decode(model, x, graph, eligible, jobs, ops, device='cpu'):
    model.eval()
    with torch.no_grad():
        logits, order = model(torch.as_tensor(x[None], device=device),
                              torch.as_tensor(graph[None], device=device),
                              torch.as_tensor(eligible[None], device=device))
    a = logits.argmax(-1).cpu().numpy().reshape(jobs, ops)
    values = order[0].cpu().numpy()
    ranking = np.empty(len(values), dtype=np.int64)
    ranking[np.argsort(values, kind='stable')] = np.arange(len(values))
    return a, ranking.reshape(jobs, ops)
