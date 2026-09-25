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


def _load(dirs: List[str]) -> Dict:
    s: Dict = {}
    for d in dirs:
        s.update(F._summaries(d))
    return s


def _registered_seeds(kind: str) -> List[int]:
    """The seeds the design registers for an arm -- never inferred from which files exist, so a seed
    that never ran is a missing run, not a smaller sample."""
    if kind.endswith("_cdshadow"):
        return [1, 2, 3]
    if kind.split("_")[0] in ("cdimit", "v4load", "v4twin", "bc1load", "bc1mpoff", "fc1load"):  # v4*: load_repr_v1; bc1*: backlog_corpus_v1
        return [1, 2, 3, 4]
    if kind.split("_")[0] in ("gnnedge0", "mpoff"):
        return list(F.SEEDS)
    return [0]


def contrast(s: Dict, topos: List[int], arm: str, ref: str) -> dict:
    """arm vs ref on the study. A learned arm (13 seeds) against a rule: per environment the median over
    seeds of the paired %. Two learned arms: the same seed paired. Two rules: the single run."""
    a_seeds, r_seeds = _registered_seeds(arm), _registered_seeds(ref)
    same_seed = a_seeds != [0] and r_seeds != [0]

    def env(t, w):
        if same_seed:
            return median(F._pct(F._el(s[(t, w, arm, sd)]), F._el(s[(t, w, ref, sd)])) for sd in a_seeds)
        return median(F._pct(F._el(s[(t, w, arm, sd)]), F._el(s[(t, w, ref, r_seeds[0])])) for sd in a_seeds)

    complete, dropped = [], {}
    for t in topos:
        missing = [(w, k, sd) for w in WINDOWS for k, seeds in ((arm, a_seeds), (ref, r_seeds)) for sd in seeds
                   if (t, w, k, sd) not in s]
        (dropped.__setitem__(str(t), missing[:6]) if missing else complete.append(t))
    codes = {k: sorted({s[(t, w, k, sd)]["code"]["commit"][:7] for t in complete for w in WINDOWS
                        for sd in (a_seeds if k == arm else r_seeds)}) for k in (arm, ref)}
    dirty = [f"{t}/{w}/{k}/{sd}" for t in complete for w in WINDOWS for k, seeds in ((arm, a_seeds), (ref, r_seeds))
             for sd in seeds if s[(t, w, k, sd)]["code"]["dirty"]]
    out = {"arm": arm, "ref": ref, "seeds": {arm: a_seeds, ref: r_seeds}, "topologies": complete,
           "dropped": dropped, "code_by_arm": codes}
    if dirty:
        return dict(out, verdict="INVALID-CODE-STATE", dirty=dirty[:10])
    if len(complete) < F.MIN_TOPOLOGIES:
        return dict(out, verdict="DESIGN-SHORT")
    vals = {t: median(env(t, w) for w in WINDOWS) for t in complete}
    out["read"] = F._read(vals, arm.upper(), ref.upper())
    out["decomposition"] = {k: _decomp([s[(t, w, k, sd)] for t in complete for w in WINDOWS
                                        for sd in (a_seeds if k == arm else r_seeds)]) for k in (arm, ref)}
    return out


def recovered_share(s: Dict, topos: List[int], base: str, partial: str, full: str) -> dict:
    """D6-3: per topology, the median over windows and seeds of (base - partial) elapsed divided by the
    median of (base - full); then the median over topologies."""
    seeds = _registered_seeds(base)
    per = {}
    for t in topos:
        try:
            gp = median(F._el(s[(t, w, base, sd)]) - F._el(s[(t, w, partial, sd)]) for w in WINDOWS for sd in seeds)
            gf = median(F._el(s[(t, w, base, sd)]) - F._el(s[(t, w, full, sd)]) for w in WINDOWS for sd in seeds)
        except KeyError:
            continue
        per[str(t)] = {"partial_gain_s": gp, "full_gain_s": gf, "share": gp / gf if gf > 0 else None}
    shares = [v["share"] for v in per.values() if v["share"] is not None]
    return {"per_topology": per, "median_share": median(shares) if shares else None, "n": len(shares)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("probe", choices=("d1", "d5", "d6", "a"))
    ap.add_argument("--gate", required=True)
    ap.add_argument("--probe-dir", required=True, nargs="+")
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    sel = json.load(open(a.selection))
    if a.probe == "d1":
        res = rule_vs_rule(a.gate, a.probe_dir[0], sel, "cd_blind", "cd")
        r = res.get("read")
        if r:
            v = r["verdict"]
            res["verdict"] = ("OUT-OF-BATCH-MATTERS" if v == "CD-FASTER" else
                              "OUT-OF-BATCH-MATTERS (direction only)" if v.startswith("CD-FASTER") else
                              "BLIND-FASTER" if v.startswith("CD_BLIND-FASTER") else "NOT-SEPARATED")
    elif a.probe == "d5":
        s = _load([a.gate] + a.probe_dir)
        topos = sel["topologies"]
        res = {"D5b-1": contrast(s, topos, "gnnedge0_cdapply", "cd"),
               "D5b-2": contrast(s, topos, "gnnedge0_cdapply", "gnnedge0"),
               "D5b-3": contrast(s, topos, "gnnedge0_cdapply", "mpoff_cdapply"),
               "disclosed_mpoff_cdapply_vs_cd": contrast(s, topos, "mpoff_cdapply", "cd")}
    elif a.probe == "a":
        s = _load([a.gate] + a.probe_dir)
        topos = sel["topologies"]
        res = {"A2": contrast(s, topos, "cdimit", "cd"),
               "A3": contrast(s, topos, "cdimit", "gnnedge0")}
    else:
        s = _load([a.gate] + a.probe_dir)
        topos = sel["topologies"]
        res = {"D6-1": contrast(s, topos, "gnnedge0_selfref", "gnnedge0"),
               "D6-2": contrast(s, topos, "gnnedge0_selfref", "cd"),
               "D6-3": recovered_share(s, topos, "gnnedge0", "gnnedge0_selfref", "gnnedge0_cdapply")}
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
