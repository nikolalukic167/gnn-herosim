#!/usr/bin/env python3
"""joint_burst_v2 gate helper: verify a checkpoint's sidecar before serving it in the gate.
Called as: joint_burst_v2_sidecheck.py <contract.json> <arm: gnnedge0|mpoff|v4load|v4twin|bc1load|bc1mpoff|fc1load|xs1load|xs1mpoff|rawgnn|rawmlp|rawE|rawS|rawES|rawStwin|lf1gnn|lf1twin|lf1mlp> <split.json> <want_alpha>
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


def main() -> int:
    contract_path, arm, split, want_alpha = sys.argv[1:5]
    sc = json.load(open(contract_path))
    aggr = "sum" if arm in AGG_BASE else "mean"
    arm = AGG_BASE.get(arm, arm)
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
