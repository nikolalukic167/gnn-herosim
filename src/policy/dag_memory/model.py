"""Graph and pointwise value models for exact-scored output-retention flips."""
import time

import numpy as np
import torch
from torch import nn

from src.placement.radical.dag_memory import replay
from src.policy.dag_reservation.model import features as dag_features

CONTRACT = 'dag_memory_flip_value_v1'


def features(b, assignment, rank, retain):
    retain = np.asarray(retain)
    x, graph, _ = dag_features(b, assignment, rank)
    jobs, ops = retain.shape
    n = jobs * ops
    consumers = np.zeros((jobs, ops), dtype=np.float32)
    local = np.zeros_like(consumers)
    remote = np.zeros_like(consumers)
    for job in range(jobs):
        for op in range(ops):
            for parent in range(op):
                if int(b['predecessors'][job, op]) & (1 << parent):
                    consumers[job, parent] += 1
                    if assignment[job, parent] == assignment[job, op]:
                        local[job, parent] += 1
                    else:
                        remote[job, parent] += 1
    cap = b['host_memory_mb'][assignment]
    size = b['output_size_mb']
    raw = np.column_stack([size.ravel() / 32., cap.ravel() / 32.,
                           np.asarray(retain).ravel(), consumers.ravel() / ops,
                           local.ravel() / ops, remote.ravel() / ops,
                           (size * local * b['store_read_ms_per_mb']).ravel() / 30.,
                           (size * remote * (b['store_read_ms_per_mb'] - b['peer_read_ms_per_mb'])).ravel() / 30.]
                          ).astype(np.float32)
    mixed = graph.mean(0)
    one = mixed @ raw
    two = mixed @ one
    four = mixed @ (mixed @ two)
    x = np.concatenate([x, raw, one, two, four], axis=-1).astype(np.float32)
    return x, graph, (consumers.ravel() > 0).astype(np.float32)


class MemoryValueNet(nn.Module):
    def __init__(self, feature_dim, hidden=64, layers=2, arm='gnn'):
        super().__init__()
        if arm not in ('gnn', 'mpoff', 'hand_mlp'):
            raise ValueError('unknown memory model arm')
        self.arm = arm
        self.input = nn.Linear(feature_dim, hidden)
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(hidden * 5, hidden), nn.ReLU(),
                                                   nn.Linear(hidden, hidden)) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.hand = nn.Sequential(nn.Linear(feature_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.output = nn.Linear(hidden, 1)

    def forward(self, x, graph, mask):
        if self.arm == 'hand_mlp':
            value = self.hand(x).squeeze(-1)
        else:
            h = torch.relu(self.input(x))
            for layer, norm in zip(self.layers, self.norms):
                messages = [torch.bmm(graph[:, z], h) for z in range(4)] if self.arm == 'gnn' else [torch.zeros_like(h) for _ in range(4)]
                h = norm(h + layer(torch.cat([h, *messages], dim=-1)))
            value = self.output(h).squeeze(-1)
        return value.masked_fill(mask == 0, -1e6)


def serve(model, b, assignment, rank, initial, prep_ms, budget_ms=250):
    tick = time.perf_counter()
    current = np.asarray(initial['retain'], dtype=np.int8).copy()
    cost = float(initial['cost'])
    scored = accepted = 0
    model.eval()
    while prep_ms + (time.perf_counter() - tick) * 1000 < budget_ms - 10:
        x, graph, mask = features(b, assignment, rank, current)
        with torch.no_grad():
            values = model(torch.as_tensor(x[None]), torch.as_tensor(graph[None]),
                           torch.as_tensor(mask[None]))[0].cpu().numpy()
        changed = False
        for node in np.argsort(-values, kind='stable'):
            if not mask[node]:
                continue
            if prep_ms + (time.perf_counter() - tick) * 1000 >= budget_ms - 10:
                break
            trial = current.copy()
            trial.flat[node] ^= 1
            trial_cost = replay(b, assignment, rank, trial)['objective']
            scored += 1
            if trial_cost < cost:
                current, cost = trial, trial_cost
                accepted += 1
                changed = True
                break
        if not changed:
            break
    total_ms = prep_ms + (time.perf_counter() - tick) * 1000
    return {'cost': cost, 'retain': current.tolist(), 'scores': scored,
            'accepted': accepted, 'time_ms': total_ms}
