"""accel_moderate_confirm_v1 live-gate reader (docs/lineages/accel_moderate_confirm_v1.md).

The scale_160_v1 reader on the moderate rung alone: per topology the median over (window, seed) of the paired % of an arm's mean latency
against CD, then the median over the 24 topologies and an exact two-sided Wilcoxon. Primary family {gnn_eng, gnn_eng_physmp} vs CD at
moderate, Holm over 2. Labels per arm: CONFIRMED (median <= -5 % and Holm p < .05), CD-FASTER (CD Holm-confirmed faster), DIRECTION-ONLY
(median < 0), NOT-SEPARATED. Self-predict and CD<-GNN are descriptive. Same cell-drop and sensitivity rules as scale_160_v1_read.py.

  accel_moderate_confirm_v1_read.py --gate <dir> [<dir> ...] [--out read.json]
"""
import argparse
import json
import signal
import sys

import scale_160_v1_read as S

RUNGS = ("moderate",)  # the scale_160 reader's functions look S.RUNGS up at call time; read() scopes the override
RENAME = {"WIN": "CONFIRMED"}


def read(gate) -> dict:
    saved, S.RUNGS = S.RUNGS, RUNGS
    try:
        r = S.read(gate)
    finally:
        S.RUNGS = saved
    for fam in (r["main"], r["sensitivity"]["family"]):
        fam["labels"] = {a: RENAME.get(v, v) for a, v in fam["labels"].items()}
    return r


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
    return 0


if __name__ == "__main__":
    sys.exit(main())
