"""Static graph encoder and cheap dynamic ready-set scoring head."""
import torch
from torch import nn

CONTRACT = "mixed_dispatch_ready_set_v1"


class ReadySetNet(nn.Module):
    def __init__(self, feature_dim, dynamic_dim=13, hidden=16, layers=2, mp=True):
        super().__init__()
        self.mp = mp
        self.input = nn.Linear(feature_dim, hidden)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden * 5, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(hidden + dynamic_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def encode(self, x, adjacency):
        hidden = torch.relu(self.input(x))
        for layer, norm in zip(self.layers, self.norms):
            messages = (torch.matmul(adjacency, hidden[:, None]) if self.mp else
                        hidden.new_zeros((hidden.shape[0], 4, hidden.shape[1], hidden.shape[2])))
            hidden = norm(hidden + layer(torch.cat([hidden, *messages.unbind(1)], dim=-1)))
        return hidden

    def score(self, encoded, dynamic, feasible):
        states = dynamic.shape[1]
        expanded = encoded[:, None].expand(-1, states, -1, -1)
        logits = self.head(torch.cat([expanded, dynamic], dim=-1)).squeeze(-1)
        return logits.masked_fill(~feasible, -1e9)

    def forward(self, x, adjacency, dynamic, feasible):
        return self.score(self.encode(x, adjacency), dynamic, feasible)
