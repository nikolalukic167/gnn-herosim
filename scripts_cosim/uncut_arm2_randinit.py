"""Random-initialised Arm 2 checkpoints (gnn and MP-OFF twin) for a wall-feasibility smoke only.

Copies a trained flags-off sidecar for the corpus-side fields, sets the Arm 2 flags, and writes untrained weights. Stem and sidecar mark them
`random_init` so they cannot be mistaken for a trained arm; nothing read from a smoke built on them is a result.

    python3 scripts_cosim/uncut_arm2_randinit.py --like <trained.pt> --stem arm2-randinit --out <models_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.policy.gnn.gnn_model import TaskPlacementGNN  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--like", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    like = Path(args.like)
    state = torch.load(like, map_location="cpu")
    side = json.loads(like.with_suffix(".contract.json").read_text())
    task_w, plat_w = state["task_encoder.net.0.weight"], state["platform_encoder.net.0.weight"]
    onehot = int(side.get("task_type_onehot_dim") or 0)
    hidden, emb = int(task_w.shape[0]), int(state["task_encoder.net.4.weight"].shape[0])
    layers = sum(1 for k in state if k.startswith("gin.convs.") and k.endswith(".nn.lins.0.weight"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for arm, mp_off in (("gnn-eng", False), ("twin-eng", True)):
        torch.manual_seed(args.seed)
        model = TaskPlacementGNN(
            task_feature_dim=int(task_w.shape[1]) - onehot, platform_feature_dim=int(plat_w.shape[1]), embedding_dim=emb,
            hidden_dim=hidden, num_layers=layers, mp_peer_edges=True, mp_platform_edges=True,
            mp_bipartite_edge_conv=not mp_off, mp_bipartite_edge_attr_zero=not mp_off, task_type_onehot_dim=onehot,
            partial_state_edge_dim=int(side["partial_state_feature_dim"]), reencode_every="auto", group_anchor=True,
            normalize_platform_inputs=side.get("feature_dim") == 21,
        )
        path = out / f"{args.stem}-{arm}-seed{args.seed}.pt"
        torch.save(model.state_dict(), path)
        sc = dict(side)
        sc.update(disable_message_passing=mp_off, mp_bipartite_edge_conv=not mp_off, mp_bipartite_edge_attr_zero=not mp_off,
                  reencode_every="auto", group_anchor=True, random_prefix_k=2, anchor_ce_weight=0.5, random_init=True)
        path.with_suffix(".contract.json").write_text(json.dumps(sc, indent=1))
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
