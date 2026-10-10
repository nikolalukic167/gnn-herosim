#!/usr/bin/env python3
"""r1_attribution_v1: one-line architecture summary of each learned arm, built exactly as the trainer builds it (read-only; no data, no checkpoint).

  r1a_arch_summary.py [--width 64] [--layers 3]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts_cosim"))

import joint_burst_v2_sidecheck as sc  # noqa: E402


def build(arm: str, width: int, layers: int):
    from src.policy.gnn.arm_models import ARM_MLP_SAME, ARM_SET_TRANSFORMER, build_graph_free_arm
    from src.policy.gnn.gnn_model import TaskPlacementGNN

    kind, mp_off, raw, conv, phys = sc.RA_ARMS[f"ra_{arm}"]
    os.environ["GNN_DISABLE_MESSAGE_PASSING"] = "1" if mp_off else "0"
    ps = 2 if raw else 27
    if kind != "gnn":
        return build_graph_free_arm(kind, task_feature_dim=5, platform_feature_dim=16, embedding_dim=width, hidden_dim=width, num_layers=layers,
                                    task_type_onehot_dim=4, partial_state_edge_dim=ps, heads=4, inducing=8)
    return TaskPlacementGNN(task_feature_dim=5, platform_feature_dim=16, embedding_dim=width, hidden_dim=width, num_layers=layers,
                            task_type_onehot_dim=4, partial_state_edge_dim=ps, mp_peer_edges=True, mp_bipartite_edge_conv=conv or raw,
                            mp_bipartite_edge_attr_zero=(conv or raw) and not phys, plan_raw=raw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    a = ap.parse_args()
    for arm in sc.RA_ARMS:
        arm = arm[3:]
        m = build(arm, a.width, a.layers)
        n = sum(p.numel() for p in m.parameters())
        used = sum(p.numel() for name, p in m.named_parameters() if not (getattr(m, "_disable_mp", False) and (name.startswith("bip_convs") or name.startswith("peer_conv") or name.startswith("gin"))))
        print(f"{arm:16s} params {n:>8,} (trained/used {used:>8,})  {type(m).__name__}  mp_off={getattr(m, '_disable_mp', None)} "
              f"bip_conv={getattr(m, 'mp_bipartite_edge_conv', None)} aggr={getattr(m, 'mp_bipartite_aggr', 'attention(softmax-mean)' if arm == 'set_transformer' else 'none')} "
              f"edge_attr_zero={getattr(m, 'mp_bipartite_edge_attr_zero', None)} peer={getattr(m, 'mp_peer_edges', None)} plan_raw={getattr(m, 'plan_raw', None)} "
              f"scorer_in={getattr(getattr(m, 'edge_scorer', None), 'fc1', None) and m.edge_scorer.fc1.in_features}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
