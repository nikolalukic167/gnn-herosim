#!/usr/bin/env python3
"""joint_burst_v2 gate helper: verify a checkpoint's sidecar before serving it in the gate.
Called as: joint_burst_v2_sidecheck.py <contract.json> <arm: gnnedge0|mpoff|v4load|v4twin|bc1load|bc1mpoff|fc1load> <split.json> <want_alpha>
FAIL LOUD (exit 1) on any mismatch. Kept as a real file, not an inline heredoc, because the
gate sbatch nests other heredocs and a `PY` terminator line collides across nesting levels.
"""
from __future__ import annotations

import hashlib
import json
import sys


def main() -> int:
    contract_path, arm, split, want_alpha = sys.argv[1:5]
    sc = json.load(open(contract_path))
    want = {
        "disable_message_passing": arm in ("mpoff", "bc1mpoff"),
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
    if arm in ("bc1load", "bc1mpoff", "fc1load"):
        # backlog_corpus_v1: v4load's recipe (and jb2 mpoff's swap) on the synthetic-backlog corpus
        want.update(partial_state_contract="partial_state_v4", partial_state_feature_dim=25, load_seconds=True)
    if arm == "fc1load":
        want["full_context_ce_weight"] = 1.0
    if arm in ("gnnedge0", "v4load", "v4twin", "bc1load", "fc1load"):
        want.update(mp_bipartite_edge_conv=True, mp_bipartite_edge_attr_zero=True)
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
