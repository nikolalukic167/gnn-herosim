#!/usr/bin/env python3
"""ZERO-EXTENDED partial_state_v7 checkpoints of the four live_link_features_v1 arms, with real sidecars, for the decision-cost w calibration cells only (not evidence).

A decision-time w is seconds of scoring per plan evaluation: it depends on the architecture, the feature widths and the code path, not on fitted weights. But a cell needs sane dynamics
to run in a calibration's time: a randomly initialised scorer spreads tasks badly and the cell runs 5x+ longer than the filed arm's (measured, job 901738/901772). So each arm is its filed v6
counterpart's weights with the new input columns' weights set to ZERO: the same architecture at the v7 widths (platform + LINK_PLATFORM_DIM, partial-state + LINK_EDGE_DIM), the same decisions
as the v6 arm until trained, the template's sidecar with the contract and widths changed. The shapes are asserted against the template (every tensor equal except the three columns that grew). Output names are the gate's: <stem>-seed1.pt and .contract.json.

  python3 scripts_cosim/v7_untrained_arm_checkpoints.py --models ~/gnn-herosim/models --out /share/.../in_link/models
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.placement.link_features import LINK_EDGE_DIM, LINK_PLATFORM_DIM  # noqa: E402
from src.policy.gnn.arm_models import build_graph_free_arm, infer_graph_free_dims  # noqa: E402
from src.policy.gnn.gnn_model import TaskPlacementGNN  # noqa: E402
from src.policy.tabular.reduced_features import PARTIAL_STATE_CONTRACT_V7, partial_state_feature_dim  # noqa: E402

# arm kind in the gate -> (template stem of the filed v6 arm, output stem, MP-OFF)
ARMS = {
    "ra_gnn_eng_combfl_link": ("uncut-comb-fl-gnn-eng", "uncut-link-gnn-eng", False),
    "ra_twin_eng_combfl_link": ("uncut-comb-fl-twin", "uncut-link-twin", True),
    "ra_mlp_same_combfl_link": ("uncut-comb-fl-mlp-same", "uncut-link-mlp-same", False),
    "ra_mlp_wide_combfl_link": ("uncut-comb-fl-mlp-wide", "uncut-link-mlp-wide", False),
}
EMB = HID = 128
LAYERS = 3


def build(kind: str, side: dict, template_sd: dict, mp_off: bool, seed: int):
    psd = partial_state_feature_dim(PARTIAL_STATE_CONTRACT_V7)
    plat = int(side["platform_feature_dim"]) + LINK_PLATFORM_DIM
    onehot = int(side["task_type_onehot_dim"])
    torch.manual_seed(seed)
    if side["arm_kind"] == "mlp_same":
        dims = infer_graph_free_dims("mlp_same", template_sd, onehot, side)
        dims.update(platform_feature_dim=plat)
        return build_graph_free_arm("mlp_same", **dims, partial_state_edge_dim=psd, task_type_onehot_dim=onehot, dropout=0.0)
    import os

    if mp_off:
        os.environ["GNN_DISABLE_MESSAGE_PASSING"] = "1"
    else:
        os.environ.pop("GNN_DISABLE_MESSAGE_PASSING", None)
    return TaskPlacementGNN(
        task_feature_dim=int(side["task_feature_dim"]), platform_feature_dim=plat, embedding_dim=EMB, hidden_dim=HID, num_layers=LAYERS, edge_dim=int(side["edge_dim"]),
        dropout=0.0, post_gin_dropout=0.0, mp_residual=bool(side.get("mp_residual")), mp_node_edges=bool(side.get("mp_node_edges")),
        mp_node_edges_candidates_only=bool(side.get("mp_node_edges_candidates_only", True)), mp_network_entities=False, mp_dag_edges=bool(side.get("mp_dag_edges")),
        mp_peer_edges=bool(side.get("mp_peer_edges")), mp_platform_edges=bool(side.get("mp_platform_edges", True)),
        mp_bipartite_edge_conv=bool(side.get("mp_bipartite_edge_conv")), mp_bipartite_edge_attr_zero=bool(side.get("mp_bipartite_edge_attr_zero")),
        mp_bipartite_aggr=str(side.get("mp_bipartite_aggr") or "mean"), mp_bipartite_hetero=bool(side.get("mp_bipartite_hetero")),
        task_type_onehot_dim=onehot, partial_state_edge_dim=psd)


def zero_extend(kind: str, new_sd: dict, old_sd: dict, old_plat: int, old_ps: int, edge_dim: int) -> dict:
    """The filed arm's weights in the v7 shapes: every old input keeps its weight and the added inputs get 0.0, so v7 and v6 score identically until trained.
    GNN: platform_encoder's and the edge scorer's last columns are the added ones. Same-input MLP: the first layer reads [task + one-hot | platform | edge | partial-state],
    so the added platform columns sit after the platform block and the added partial-state columns at the end."""
    out = {}
    for k, new in new_sd.items():
        old = old_sd[k]
        if tuple(new.shape) == tuple(old.shape):
            out[k] = old.clone()
            continue
        w = torch.zeros_like(new)
        if "mlp" in kind:
            n_task = int(old.shape[1]) - old_plat - edge_dim - old_ps
            if n_task <= 0:
                raise SystemExit(f"FAIL LOUD: {kind}: first layer takes {old.shape[1]} columns, not task + platform {old_plat} + edge {edge_dim} + partial {old_ps}")
            head = n_task + old_plat
            w[:, :head] = old[:, :head]
            w[:, head + LINK_PLATFORM_DIM:head + LINK_PLATFORM_DIM + edge_dim + old_ps] = old[:, head:]
        else:
            w[:, :old.shape[1]] = old
        out[k] = w
    return out


def check_shapes(kind: str, new_sd: dict, old_sd: dict) -> None:
    """Every tensor of the filed arm is present with its shape, except the inputs that read the new columns, each wider by exactly their columns."""
    if set(new_sd) != set(old_sd):
        raise SystemExit(f"FAIL LOUD: {kind}: tensor names differ from the filed arm: only new {sorted(set(new_sd) - set(old_sd))[:5]}, only old {sorted(set(old_sd) - set(new_sd))[:5]}")
    grown = {k: (tuple(old_sd[k].shape), tuple(new_sd[k].shape)) for k in new_sd if tuple(new_sd[k].shape) != tuple(old_sd[k].shape)}
    want = ({"net.0.weight": LINK_PLATFORM_DIM + LINK_EDGE_DIM} if "mlp" in kind
            else {"platform_encoder.net.0.weight": LINK_PLATFORM_DIM, "edge_scorer.fc1.weight": LINK_EDGE_DIM})
    got = {k: (n[-1] - o[-1]) for k, (o, n) in grown.items() if o[:-1] == n[:-1]}
    if set(grown) != set(want) or got != want:
        raise SystemExit(f"FAIL LOUD: {kind}: grown tensors {grown}, expected {want}")
    print(f"  {kind}: grown inputs {want}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True, help="directory of the filed v6 checkpoints (-seed1-final.pt + .contract.json)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for kind, (tmpl, stem, mp_off) in ARMS.items():
        side = json.loads((a.models / f"{tmpl}-seed1-final.contract.json").read_text())
        old_sd = torch.load(a.models / f"{tmpl}-seed1-final.pt", map_location="cpu")
        model = build(kind, side, old_sd, mp_off, a.seed)
        check_shapes(kind, model.state_dict(), old_sd)
        sd = zero_extend(kind, model.state_dict(), old_sd, int(side["platform_feature_dim"]), int(side["partial_state_feature_dim"]), int(side["edge_dim"]))
        model.load_state_dict(sd)
        out_pt = a.out / f"{stem}-seed{a.seed}.pt"
        if out_pt.exists():
            raise SystemExit(f"FAIL LOUD: {out_pt} exists; refusing to overwrite")
        side.update(partial_state_contract=PARTIAL_STATE_CONTRACT_V7, partial_state_feature_dim=partial_state_feature_dim(PARTIAL_STATE_CONTRACT_V7),
                    platform_feature_dim=int(side["platform_feature_dim"]) + LINK_PLATFORM_DIM, train_seed=a.seed, zero_extended_from_v6_for_calibration_only=True,
                    disable_message_passing=mp_off)
        torch.save(sd, out_pt)
        out_pt.with_suffix(".contract.json").write_text(json.dumps(side, indent=1))
        print(f"wrote {out_pt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
