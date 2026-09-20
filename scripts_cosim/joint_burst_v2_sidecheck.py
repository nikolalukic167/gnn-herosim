#!/usr/bin/env python3
"""joint_burst_v2 gate helper: verify a checkpoint's sidecar before serving it in the gate.
Called as: joint_burst_v2_sidecheck.py <contract.json> <arm: gnnedge0|mpoff> <split.json> <want_alpha>
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
        "disable_message_passing": arm == "mpoff",
        "mp_peer_edges": True,
        "partial_state_contract": "partial_state_v3",
        "partial_state_feature_dim": 22,
        "mp_residual": False,
        "dag_alpha_key": want_alpha,
    }
    if arm == "gnnedge0":
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
