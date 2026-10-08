"""physics_audit_v1 pass 2: run I1-I7, I9, I10 on every trace in a directory (<topo>_<rung>_<arm>.trace.jsonl with its
.json result beside it), I12 across rungs per (topology, arm), and print the table. Writes <out>/checks.json."""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_invariants import Trace, i12_rung_config, run_trace_checks  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = {}
    for tp in sorted(glob.glob(os.path.join(a.dir, "*.trace.jsonl"))):
        name = os.path.basename(tp)[: -len(".trace.jsonl")]
        rp = tp.replace(".trace.jsonl", ".json")
        runs[name] = run_trace_checks(tp, rp if os.path.exists(rp) else None)
    status = defaultdict(lambda: defaultdict(int))
    for name, res in runs.items():
        for r in res:
            status[r["id"]][r["status"]] += 1
    print(f"{len(runs)} runs")
    for i in sorted(status, key=lambda s: int(s[1:])):
        print(f"  {i}: " + ", ".join(f"{k} {v}" for k, v in sorted(status[i].items())))
    for name, res in runs.items():
        for r in res:
            if r["status"] not in ("PASS",) and r["id"] != "I5":
                print(f"  ! {name} {r['id']} {r['status']} {r['detail']}")
    i5 = {n: next(r for r in res if r["id"] == "I5")["numbers"] for n, res in runs.items()}
    out = {"runs": runs}
    for arm in ("cd", "reactive"):
        rows = {n: v for n, v in i5.items() if n.endswith("_" + arm)}
        shares = [v.get("load_share") for v in rows.values() if v.get("load_share") is not None]
        inst = [v.get("spearman_binned") for v in rows.values() if v.get("spearman_binned") is not None]
        stab = [v.get("info_spearman_vs_kpa_stable_window") for v in rows.values() if v.get("info_spearman_vs_kpa_stable_window") is not None]
        if shares:
            print(f"I5 {arm}: n={len(rows)} load share median {statistics.median(shares):.3f} (min {min(shares):.3f} max {max(shares):.3f}, "
                  f">0.5 in {sum(s > 0.5 for s in shares)}/{len(shares)}); rho instantaneous median {statistics.median(inst):.2f}; "
                  f"rho KPA stable-window median {statistics.median(stab):.2f}")
            out[f"i5_{arm}"] = {"n": len(rows), "load_share_median": statistics.median(shares), "load_shares": shares,
                                "rho_inflight_median": statistics.median(inst), "rho_stable_median": statistics.median(stab),
                                "rho_inflight": inst, "rho_stable": stab}
    by = defaultdict(dict)
    for n in runs:
        topo, rung, arm = n.split("_")
        by[(topo, arm)][rung] = Trace(os.path.join(a.dir, n + ".trace.jsonl"))
    out["i12"] = {}
    for (topo, arm), rungs in sorted(by.items()):
        r = i12_rung_config(rungs)
        out["i12"][f"{topo}_{arm}"] = r
        print(f"I12 {topo} {arm}: {r['status']} {r['detail']} cold/task " +
              " ".join(f"{k}={v['cold_starts_per_task']:.3f}" for k, v in sorted(r["numbers"]["per_rung"].items())))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
