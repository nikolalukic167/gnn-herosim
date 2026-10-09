#!/usr/bin/env python3
"""joint_burst_v2 gate helper: verify a checkpoint's sidecar before serving it in the gate.
Called as: joint_burst_v2_sidecheck.py <contract.json> <arm: ra_gnn_eng|ra_twin_eng|ra_mlp_same|ra_gnn_raw|ra_twin_raw|ra_gnn_eng_physmp|ra_set_transformer|gnnedge0|mpoff|v4load|v4twin|bc1load|bc1mpoff|fc1load|xs1load|xs1mpoff|rawgnn|rawmlp|rawE|rawS|rawES|rawStwin|lf1gnn|lf1twin|lf1mlp|sb1sum|lf1sum|lf1het> <split.json> <want_alpha>
FAIL LOUD (exit 1) on any mismatch. Kept as a real file, not an inline heredoc, because the
gate sbatch nests other heredocs and a `PY` terminator line collides across nesting levels.
"""
from __future__ import annotations

import hashlib
import json
import sys


RAW_V2 = ("rawE", "rawS", "rawES", "rawStwin")
# local_features_v1: each arm is a raw-plan arm plus the static per-candidate columns
LF1_BASE = {"lf1gnn": "rawS", "lf1twin": "rawStwin", "lf1mlp": "rawmlp"}
# sum aggregation: the same recipe with mp_bipartite_aggr = "sum"
AGG_BASE = {"sb1sum": "xs1load", "lf1sum": "lf1gnn"}
# hetero_conv_v1: lf1gnn's recipe with per-relation / per-node-type bipartite weights
HET_BASE = {"lf1het": "lf1gnn"}


# r1_attribution_v1: partial_state_v5 (27 columns), the so1load recipe, and an explicit architecture per arm.
# arm -> (arm_kind, mp off, raw plan, bipartite conv, conv sees physics edge_attr)
RA_ARMS = {
    "ra_gnn_eng": ("gnn", False, False, True, False),
    "ra_twin_eng": ("gnn", True, False, False, False),
    "ra_mlp_same": ("mlp_same", False, False, False, False),
    "ra_gnn_raw": ("gnn", False, True, True, False),
    "ra_twin_raw": ("gnn", True, True, False, False),
    "ra_gnn_eng_physmp": ("gnn", False, False, True, True),
    "ra_set_transformer": ("set_transformer", False, False, False, False),
}


def check_ra(sc: dict, arm: str, split: str, want_alpha: str) -> int:
    kind, mp_off, raw, conv, phys = RA_ARMS[arm]
    graph_free = kind != "gnn"
    want = {
        "arm_kind": kind, "disable_message_passing": mp_off, "partial_state_contract": "partial_state_v5",
        "load_seconds": True, "exchange_seconds": True, "full_context_ce_weight": 1.0, "dag_alpha_key": want_alpha,
        "mp_residual": False, "plan_raw": raw, "plan_raw_sum": False, "plan_raw_local": False,
        "mp_bipartite_edge_conv": conv, "mp_bipartite_edge_attr_zero": conv and not phys,
        "mp_peer_edges": not graph_free, "partial_state_feature_dim": 2 if raw else 27,
    }
    if graph_free:
        want.update(mp_bipartite_edge_conv=False, mp_bipartite_edge_attr_zero=False)
    if kind == "set_transformer" and not sc.get("set_heads"):
        print("FAIL LOUD: a set_transformer sidecar must record set_heads", file=sys.stderr)
        return 1
    for k, v in want.items():
        if sc.get(k) != v:
            print(f"FAIL LOUD: sidecar {k}={sc.get(k)!r}, arm {arm} expects {v!r}", file=sys.stderr)
            return 1
    if sc.get("candidate_slate") != "declared_pruning_v1" or not sc.get("physics_env"):
        print("FAIL LOUD: an r1_attribution_v1 checkpoint must record candidate_slate=declared_pruning_v1 and its physics_env",
              file=sys.stderr)
        return 1
    sha = hashlib.sha256(open(split, "rb").read()).hexdigest()
    if (sc.get("split_artifact") or {}).get("sha256") != sha:
        print(f"FAIL LOUD: sidecar split sha != {split}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    contract_path, arm, split, want_alpha = sys.argv[1:5]
    sc = json.load(open(contract_path))
    if arm in RA_ARMS:
        return check_ra(sc, arm, split, want_alpha)
    aggr = "sum" if arm in AGG_BASE else "mean"
    arm = AGG_BASE.get(arm, arm)
    hetero = arm in HET_BASE
    arm = HET_BASE.get(arm, arm)
    local = arm in LF1_BASE
    arm = LF1_BASE.get(arm, arm)
    raw = ("rawgnn", "rawmlp") + RAW_V2
    want = {
        "disable_message_passing": arm in ("mpoff", "bc1mpoff", "xs1mpoff", "rawmlp", "rawStwin"),
        "mp_peer_edges": True,
        "partial_state_contract": "partial_state_v3",
        "partial_state_feature_dim": 22,
        "mp_residual": False,
        "dag_alpha_key": want_alpha,
    }
    if arm in ("v4load", "v4twin"):
        # load_repr_v1: the gnnedge0 architecture under partial_state_v4, load columns on / zeroed
        want.update(partial_state_contract="partial_state_v4", partial_state_feature_dim=25,
                    load_seconds=(arm == "v4load"))
    if arm in ("bc1load", "bc1mpoff", "fc1load", "xs1load", "xs1mpoff") + raw:
        # backlog_corpus_v1: v4load's recipe (and jb2 mpoff's swap) on the synthetic-backlog corpus
        want.update(partial_state_contract="partial_state_v4", partial_state_feature_dim=25, load_seconds=True)
    if arm in raw:
        # raw_plan_v1: the engineered block is replaced by the 2 raw plan columns (+5 local ones, local_features_v1)
        want.update(partial_state_feature_dim=7 if local else 2, plan_raw=True)
    if local:
        want["plan_raw_local"] = True
    elif sc.get("plan_raw_local"):
        want["plan_raw_local"] = False
    elif sc.get("plan_raw"):
        want["plan_raw"] = False  # a raw-plan checkpoint served as an engineered-context arm
    # raw_plan_v2: the committed-load channel on rawS / rawES / rawStwin, absent everywhere else
    if arm in ("rawS", "rawES", "rawStwin"):
        want["plan_raw_sum"] = True
    elif sc.get("plan_raw_sum"):
        want["plan_raw_sum"] = False
    if arm in ("fc1load", "xs1load", "xs1mpoff") + raw:
        want["full_context_ce_weight"] = 1.0
    if arm in ("xs1load", "xs1mpoff") + raw:
        want["exchange_seconds"] = True
    if arm in ("gnnedge0", "v4load", "v4twin", "bc1load", "fc1load", "xs1load", "rawgnn", "rawS"):
        want.update(mp_bipartite_edge_conv=True, mp_bipartite_edge_attr_zero=True)
        if "mp_bipartite_aggr" in sc or aggr != "mean":
            want["mp_bipartite_aggr"] = aggr
    if arm in ("rawE", "rawES"):
        # raw_plan_v2: the conv sees the physics edge_attr
        want.update(mp_bipartite_edge_conv=True, mp_bipartite_edge_attr_zero=False)
        if "mp_bipartite_aggr" in sc:
            want["mp_bipartite_aggr"] = "mean"
    if hetero or sc.get("mp_bipartite_hetero"):
        want["mp_bipartite_hetero"] = hetero
    for k, v in want.items():
        if sc.get(k) != v:
            print(f"FAIL LOUD: sidecar {k}={sc.get(k)!r}, arm {arm} expects {v!r}", file=sys.stderr)
            return 1
    sha = hashlib.sha256(open(split, "rb").read()).hexdigest()
    if (sc.get("split_artifact") or {}).get("sha256") != sha:
        print(f"FAIL LOUD: sidecar split sha != {split}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
