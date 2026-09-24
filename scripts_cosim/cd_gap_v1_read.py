#!/usr/bin/env python3
"""cd_gap_v1 -- readers for the live probes (docs/lineages/cd_gap_v1.md).

  cd_gap_v1_read.py d1 --gate <fresh gate dir> --probe <d1 dir> --selection selected.json [--out read.json]

The statistic is `fresh_topo_burst_v1`'s: per environment the paired %, per topology the median over
its 4 windows, exact two-sided Wilcoxon over topologies. A topology missing any run of either arm is
dropped by name. Reference-arm runs may come from the fresh gate's commit; the probe's own commit
must reproduce them with the knob off, which is checked on the witness cell named in the node.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import median
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fresh_topo_burst_v1_read as F  # noqa: E402

WINDOWS = F.WINDOWS


def _decomp(rows: List[dict]) -> Dict[str, float]:
    return {
        "elapsed": median(float(r["averageElapsedTime"]) for r in rows),
        "queue": median(float(r["averageQueueTime"]) for r in rows),
        "exchange_per_task": median(float(r.get("totalPeerExchangeTime") or 0) / int(r["num_tasks"]) for r in rows),
    }


def rule_vs_rule(ref_dir: str, probe_dir: str, selection: dict, arm: str, ref: str) -> dict:
    s = F._summaries(ref_dir)
    s.update(F._summaries(probe_dir))
    complete, dropped = [], {}
    for t in selection["topologies"]:
        missing = [(w, k) for w in WINDOWS for k in (arm, ref) if (t, w, k, 0) not in s]
        (dropped.__setitem__(str(t), missing) if missing else complete.append(t))
    codes = {k: sorted({s[(t, w, k, 0)]["code"]["commit"][:7] for t in complete for w in WINDOWS}) for k in (arm, ref)}
    dirty = any(s[(t, w, k, 0)]["code"]["dirty"] for t in complete for w in WINDOWS for k in (arm, ref))
    out = {"topologies": complete, "dropped": dropped, "code_by_arm": codes}
    if dirty:
        out["verdict"] = "INVALID-CODE-STATE"
        return out
    if len(complete) < F.MIN_TOPOLOGIES:
        out["verdict"] = "DESIGN-SHORT"
        return out
    vals = {t: median(F._pct(F._el(s[(t, w, arm, 0)]), F._el(s[(t, w, ref, 0)])) for w in WINDOWS) for t in complete}
    out["read"] = F._read(vals, arm.upper(), ref.upper())
    out["decomposition"] = {k: _decomp([s[(t, w, k, 0)] for t in complete for w in WINDOWS]) for k in (arm, ref)}
    out["blinded_per_run_median"] = median(
        int((s[(t, w, arm, 0)].get("schedulerCounters") or {}).get("pg_partners_blinded") or 0)
        for t in complete for w in WINDOWS)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("probe", choices=("d1",))
    ap.add_argument("--gate", required=True)
    ap.add_argument("--probe-dir", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = rule_vs_rule(a.gate, a.probe_dir, json.load(open(a.selection)), "cd_blind", "cd")
    r = res.get("read")
    if r:
        v = r["verdict"]
        res["verdict"] = ("OUT-OF-BATCH-MATTERS" if v == "CD-FASTER" else
                          "OUT-OF-BATCH-MATTERS (direction only)" if v.startswith("CD-FASTER") else
                          "BLIND-FASTER" if v.startswith("CD_BLIND-FASTER") else "NOT-SEPARATED")
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
