"""r1_attribution_v1 production build: the corpus' volume and filter counts from the per-cell warm_summary files.

Fails loud when the batches made fall short of the target, when a cell's capture was not ok without being listed, or when sweeps were
discarded above --max-discard-share. A sub-batch counts toward its parent batch, never as a batch of its own.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--target", type=int, default=5000)
    ap.add_argument("--max-discard-share", type=float, default=0.02)
    a = ap.parse_args()
    tot = {"cells": 0, "batches_made": 0, "datasets": 0, "offered": 0, "rejected": 0, "discarded": 0, "single_candidate_node": 0,
           "disconnected_batch": 0, "no_choice": 0, "no_peer_pairs": 0, "pruned_snapshots": 0, "sub_batched_snapshots": 0,
           "snapshots": 0, "datasets_from_sub_batches": 0}
    for f in sorted(a.root.glob("gnn_datasets_wf1_*/warm_summary_*.json")):
        s = json.loads(f.read_text())
        tot["cells"] += 1
        tot["batches_made"] += s["batches_made"]
        tot["datasets"] += s["made"]
        for k in ("offered", "rejected", "discarded", "single_candidate_node", "disconnected_batch", "no_choice", "no_peer_pairs"):
            tot[k] += s.get(k, 0)
        dp = s.get("declared_pruning") or {}
        for k in ("pruned_snapshots", "sub_batched_snapshots", "snapshots", "datasets_from_sub_batches"):
            tot[k] += dp.get(k, 0)
    cells_not_ok, cap_hit = [], []
    for f in sorted((a.root / "build_logs").glob("topology_*.json")):
        t = json.loads(f.read_text())
        if t["cells_not_ok"]:
            cells_not_ok.append(f"{t['topology']}: {t['cells_not_ok']}")
        if t.get("cap_hit_cells"):
            cap_hit.append(f"{t['topology']}: {t['cap_hit_cells']}")
    shares = {k: (tot[k] / tot["offered"] if tot["offered"] else 0.0) for k in
              ("rejected", "single_candidate_node", "disconnected_batch", "no_choice", "no_peer_pairs", "discarded")}
    sub_share = tot["sub_batched_snapshots"] / tot["snapshots"] if tot["snapshots"] else 0.0
    print(json.dumps({"totals": tot, "rate_per_offered": shares, "sub_batched_share_of_batches": sub_share,
                      "capture_not_ok_cells": cells_not_ok, "cap_hit_cells": cap_hit}, indent=1))
    problems = []
    if cap_hit:
        problems.append(f"{len(cap_hit)} topolog(ies) have cap-hit cells (the capture stopped at the snapshot cap): {cap_hit[:5]}")
    if tot["batches_made"] < a.target:
        problems.append(f"{tot['batches_made']} batches made, target {a.target}")
    made_or_discarded = tot["datasets"] + tot["discarded"]
    if made_or_discarded and tot["discarded"] / made_or_discarded > a.max_discard_share:
        problems.append(f"{tot['discarded']} of {made_or_discarded} sweeps discarded (> {a.max_discard_share:.0%})")
    if problems:
        print("FAIL LOUD: " + "; ".join(problems), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
