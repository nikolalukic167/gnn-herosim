#!/usr/bin/env python3
"""Score a route_b live replay gate (`route_b_live_replay_gate.py` output).

Per arm: regret % vs the stored sweep optimum (may be NEGATIVE for a reactive arm — it is
not bound by replica uniqueness or the alpha cap, so it can reach plans the sweep never
enumerated; those cases are counted and reported, never clipped).

Paired reading, per forced arm against a reactive reference (default `knative_network`):
per-parent d = 100 * ln(rtt_arm / rtt_ref) (negative = arm faster), median and mean over
the parents, exact/scipy Wilcoxon on the per-parent d, wins/ties/losses. Forced arms named
`<family>_s<k>` are grouped by family and the per-seed medians are summarised (median of
medians, min/max, seeds with p<0.05 in each direction) so the seed spread is visible.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts_cosim"))
from analyze_route_b_fit_p1 import wilcoxon_signed_rank  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gate_json", type=Path)
    ap.add_argument("--reference", default="knative_network@replay",
                    help="the reactive arm's chosen plan replayed through the forced path, so "
                         "the pairing is engine-identical (raw reactive rtts carry no 0.1 s "
                         "batch_timeout wait per task)")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    d = json.loads(args.gate_json.read_text())
    per = d["per_dataset"]
    parents = sorted(per)
    arms = sorted({a for v in per.values() for a in v["rtt"]})
    if args.reference not in arms:
        raise SystemExit(f"FAIL LOUD: reference arm {args.reference!r} not in {arms}")

    print(f"{len(parents)} parents; arms: {arms}\n")
    print(f"{'arm':<22}{'mean regret%':>13}{'median%':>9}{'p90%':>8}{'<opt':>6}{'=opt':>6}  mean rtt")
    summary: Dict[str, Dict[str, float]] = {}
    for a in arms:
        reg = [100.0 * (per[p]["rtt"][a] / per[p]["optimum_rtt"] - 1.0) for p in parents]
        below = sum(1 for r in reg if r < -1e-9)
        eq = sum(1 for r in reg if abs(r) <= 1e-9)
        q = st.quantiles(reg, n=10)
        summary[a] = {"mean_regret_pct": st.mean(reg), "median_regret_pct": st.median(reg),
                      "p90_regret_pct": q[8], "n_below_optimum": below, "n_at_optimum": eq,
                      "mean_rtt": st.mean(per[p]["rtt"][a] for p in parents)}
        print(f"{a:<22}{st.mean(reg):>13.2f}{st.median(reg):>9.2f}{q[8]:>8.1f}{below:>6}{eq:>6}  "
              f"{summary[a]['mean_rtt']:.2f}")

    ref = args.reference
    print(f"\nPaired vs {ref}: d = 100*ln(rtt_arm/rtt_ref) per parent (negative = arm faster)")
    print(f"{'arm':<22}{'median d':>9}{'mean d':>8}{'p_wilcoxon':>11}{'wins':>6}{'ties':>6}{'losses':>7}")
    paired: Dict[str, Dict[str, float]] = {}
    for a in arms:
        if a == ref:
            continue
        dd = [100.0 * math.log(per[p]["rtt"][a] / per[p]["rtt"][ref]) for p in parents]
        _, pw = wilcoxon_signed_rank(dd)
        wins = sum(1 for x in dd if x < -1e-9)
        ties = sum(1 for x in dd if abs(x) <= 1e-9)
        paired[a] = {"median_d": st.median(dd), "mean_d": st.mean(dd), "p_wilcoxon": pw,
                     "wins": wins, "ties": ties, "losses": len(dd) - wins - ties}
        print(f"{a:<22}{st.median(dd):>9.2f}{st.mean(dd):>8.2f}{pw:>11.4f}{wins:>6}{ties:>6}"
              f"{len(dd) - wins - ties:>7}")

    fam: Dict[str, List[str]] = defaultdict(list)
    for a in paired:
        m = re.match(r"^(.*)_s(\d+)$", a)
        if m:
            fam[m.group(1)].append(a)
    families: Dict[str, Dict[str, float]] = {}
    if fam:
        print(f"\nPer-family seed summary vs {ref} (median d per seed):")
        for f, members in sorted(fam.items()):
            meds = [paired[a]["median_d"] for a in members]
            means = [paired[a]["mean_d"] for a in members]
            sig_faster = sum(1 for a in members if paired[a]["p_wilcoxon"] < 0.05 and paired[a]["median_d"] < 0)
            sig_slower = sum(1 for a in members if paired[a]["p_wilcoxon"] < 0.05 and paired[a]["median_d"] > 0)
            families[f] = {"n_seeds": len(members), "median_of_median_d": st.median(meds),
                           "min_median_d": min(meds), "max_median_d": max(meds),
                           "mean_of_mean_d": st.mean(means),
                           "seeds_sig_faster": sig_faster, "seeds_sig_slower": sig_slower}
            print(f"  {f:<10} n={len(members)} median-of-medians {st.median(meds):+.2f} "
                  f"[{min(meds):+.2f}, {max(meds):+.2f}]  mean-of-means {st.mean(means):+.2f}  "
                  f"seeds sig faster/slower: {sig_faster}/{sig_slower}")

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"gate": str(args.gate_json), "reference": ref, "n_parents": len(parents),
             "per_arm": summary, "paired": paired, "families": families}, indent=1))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
