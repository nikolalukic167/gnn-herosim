"""accel_nosplit_v1 live-gate reader (docs/lineages/accel_nosplit_v1.md).

The scale_160 reader with the no-split arms: per topology the median over (window, seed) of the paired % of an arm's mean latency against CD
(seed 0), then the median over the 24 topologies and an exact two-sided Wilcoxon. Primary family {ra_gnn_eng_nosplit, ra_gnn_eng_physmp_nosplit}
vs CD x {moderate, heavy}, Holm over 4. Arm label: CD-FASTER (CD Holm-confirmed faster at some rung), else WIN (median <= -5 % with Holm p < .05
at BOTH rungs), else DIRECTION-ONLY (median < 0 at both), else NOT-SEPARATED. Per-rung labels: CONFIRMED / CD-FASTER / DIRECTION-ONLY /
NOT-SEPARATED. Descriptive, outside the family and never the bar: the same GNN arms with the flag off, cd_pull (CD + node pull hold, a hand rule
with information the GNN lacks), self-predict, and CD<-GNN (ra_gnn_eng_cdapply, served FLAG-OFF: one kind, as registered). The bar is plain CD.
Amendment fdedbb52: cd_expand (seed 0) is a descriptive arm; the read adds the no-split arms vs cd_expand, paired, descriptive.

  accel_nosplit_v1_read.py --gate <dir> [<dir> ...] [--out read.json]
"""
import argparse
import json
import signal
import sys

import scale_160_v1_read as S

ARMS = ("ra_gnn_eng_nosplit", "ra_gnn_eng_physmp_nosplit")
DESCRIPTIVE = ("ra_gnn_eng", "ra_gnn_eng_physmp", "cd_pull", "selfpredict", "ra_gnn_eng_cdapply")


def rung_label(c: dict) -> str:
    if c["median_pct"] is None:
        return "NOT-SEPARATED"
    if c["median_pct"] > 0 and c["holm_p"] < S.ALPHA:
        return "CD-FASTER"
    if c["median_pct"] <= S.BAR_PCT and c["holm_p"] < S.ALPHA:
        return "CONFIRMED"
    return "DIRECTION-ONLY" if c["median_pct"] < 0 else "NOT-SEPARATED"


def read(gate) -> dict:
    saved = S.ARMS, S.DESCRIPTIVE
    S.ARMS, S.DESCRIPTIVE = ARMS, DESCRIPTIVE
    try:
        r = S.read(gate)
    finally:
        S.ARMS, S.DESCRIPTIVE = saved
    for fam in (r["main"], r["sensitivity"]["family"]):
        fam["rung_labels"] = {k: rung_label(c) for k, c in fam["tests"].items()}
    r["vs_cd_expand"] = vs_cd_expand(gate, r["topologies"])
    return r


def vs_cd_expand(gate, topos) -> dict:
    """fdedbb52 amendment: the no-split arms against cd_expand (seed 0), paired like every contrast; descriptive, outside the family."""
    cells, _ = S.R.load(gate)
    out = {}
    for a in ARMS:
        for rung in S.RUNGS:
            c = S.R.contrast(cells, a, "cd_expand", rung, topos)
            out[f"{a}|{rung}"] = {k: c[k] for k in ("n_topologies", "median_pct", "p", "wins")}
    return out


def main() -> int:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = read(a.gate)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(r, fh, indent=1)
    S.print_report(r)
    print("-- per-rung labels (main): " + ", ".join(f"{k} {v}" for k, v in r["main"]["rung_labels"].items()))
    print("-- no-split GNN vs cd_expand (paired, descriptive; negative = the GNN is faster)")
    for k, c in r["vs_cd_expand"].items():
        m = "-" if c["median_pct"] is None else f"{c['median_pct']:+.2f} %"
        print(f"  {k:40s} median {m:>10s} n={c['n_topologies']} wins={c['wins']} p={'-' if c['p'] is None else format(c['p'], '.4f')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
