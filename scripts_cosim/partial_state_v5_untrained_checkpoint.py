#!/usr/bin/env python3
"""An UNTRAINED partial_state_v5 checkpoint with a real sidecar, for train/serve parity checks only (not evidence).

Parity does not need fitted weights: it needs the live serving path and the cache path to build the same graph and the
same decode inputs. The architecture is the template sidecar's (a gnnedge0-style checkpoint); only the widths follow the
v5 cache (task 5, platform 16, 27 partial-state columns) and the contract key says partial_state_v5.

  python3 scripts_cosim/partial_state_v5_untrained_checkpoint.py --cache-dir <v5 cache> --template <v4 .contract.json> \\
      --out models/ps_v5_untrained_seed0.pt
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.policy.gnn.gnn_model import TaskPlacementGNN  # noqa: E402
from src.policy.tabular.reduced_features import (  # noqa: E402
    PARTIAL_STATE_CONTRACT_V5, partial_state_feature_dim,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    side = json.loads(a.template.read_text())
    g = pickle.load(open(a.cache_dir / "graphs.pkl", "rb"))[0]
    meta = json.loads((a.cache_dir / "metadata.json").read_text())
    torch.manual_seed(a.seed)
    onehot = int(side["task_type_onehot_dim"])
    model = TaskPlacementGNN(
        task_feature_dim=int(g.task_features.size(-1)), platform_feature_dim=int(g.platform_features.size(-1)),
        embedding_dim=64, hidden_dim=64, num_layers=3, dropout=0.0, post_gin_dropout=0.0,
        mp_residual=bool(side.get("mp_residual")), mp_node_edges=bool(side.get("mp_node_edges")),
        mp_node_edges_candidates_only=bool(side.get("mp_node_edges_candidates_only", True)),
        mp_network_entities=False, mp_dag_edges=bool(side.get("mp_dag_edges")),
        mp_peer_edges=bool(side.get("mp_peer_edges")), mp_platform_edges=bool(side.get("mp_platform_edges", True)),
        mp_bipartite_edge_conv=bool(side.get("mp_bipartite_edge_conv")),
        mp_bipartite_edge_attr_zero=bool(side.get("mp_bipartite_edge_attr_zero")),
        mp_bipartite_aggr=str(side.get("mp_bipartite_aggr") or "mean"), mp_bipartite_hetero=False,
        task_type_onehot_dim=onehot, partial_state_edge_dim=partial_state_feature_dim(PARTIAL_STATE_CONTRACT_V5),
    )
    side.update(
        partial_state_contract=PARTIAL_STATE_CONTRACT_V5,
        partial_state_feature_dim=partial_state_feature_dim(PARTIAL_STATE_CONTRACT_V5),
        cache_dir=str(a.cache_dir), train_seed=a.seed, label_objective="rtt", split_artifact=None,
        untrained_for_parity_only=True,
        physics_env=meta.get("physics_env"), candidate_slate=meta.get("candidate_slate"),
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), a.out)
    a.out.with_suffix(".contract.json").write_text(json.dumps(side, indent=1))
    print(f"wrote {a.out} (task {g.task_features.size(-1)}, platform {g.platform_features.size(-1)}, "
          f"partial {side['partial_state_feature_dim']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
