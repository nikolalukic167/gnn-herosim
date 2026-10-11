#!/usr/bin/env python3
"""plan_rerank_v1 hand control S_alpha (offline): S with its queue (drain) term scaled by alpha, from fidelity_cd_plans --s-comps.

Variants (each with alpha in {1, .75, .5, .25, 0}; alpha = 1 without credit is argmin-S itself):
  drain      alpha * (drain_platform + drain_batch)   [the registered one]
  batch      alpha * drain_batch only                 [descriptive: only the in-batch stacking part]
  +cold      the same, also crediting avoided cold starts (cold - cold_dup)
The plan of a variant = its argmin over the table's full sweep; ties (|dS| <= 1e-12) take the lowest label, as argmin-S's 6.1 % does
(the tie-mean is reported too). alpha is chosen per variant on the TRAIN tables only, by pooled regret (sum(rtt - opt) / sum(opt));
held-out is then read once per rung: pooled, pooled without its top-3 tables, median and mean relative regret, and per-table
better / equal / worse counts against argmin-S.

  plan_rerank_salpha.py --train-comps DIR --heldout-comps DIR --heldout-table table_<ckpt>.jsonl --train-meta train.jsonl --out read.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

ALPHAS = (1.0, 0.75, 0.5, 0.25, 0.0)
VARIANTS = ("drain", "batch", "drain+cold", "batch+cold")
TIE = 1e-12


def load(d: Path, keep=None) -> dict:
    out = {}
    for f in sorted(d.glob("ds_*.json")):
        if keep is not None and f.stem not in keep:
            continue
        j = json.load(open(f))
        if tuple(j["keys"]) != ("rtt", "drain_platform", "drain_batch", "cold", "cold_dup", "exec_lat", "exch"):
            raise SystemExit(f"FAIL LOUD: {f} keys {j['keys']}")
        out[f.stem] = j["rows"]
    return out


def pick(rows, variant, a):
    def s(r):
        _, dp, db, c, cd, el, ex = r
        drain = a * (dp + db) if variant.startswith("drain") else dp + a * db
        cold = c - cd if variant.endswith("+cold") else c
        return drain + cold + el + ex
    sv = [s(r) for r in rows]
    m = min(sv)
    tied = [rows[i][0] for i in range(len(rows)) if sv[i] - m <= TIE]
    return min(tied), st.mean(tied)


def table_regret(tables: dict, variant, a):
    return {ds: (pick(rows, variant, a), min(r[0] for r in rows)) for ds, rows in tables.items()}


def pooled(res):
    return sum(v[0][0] - v[1] for v in res.values()) / sum(v[1] for v in res.values())


def summary(res, base):
    ex = sorted(((v[0][0] - v[1]) for v in res.values()), reverse=True)
    so = sum(v[1] for v in res.values())
    rel = [(v[0][0] - v[1]) / v[1] for v in res.values()]
    b = {ds: base[ds][0][0] for ds in res}
    better = sum(res[ds][0][0] < b[ds] - 1e-9 * b[ds] for ds in res)
    worse = sum(res[ds][0][0] > b[ds] + 1e-9 * b[ds] for ds in res)
    return dict(n=len(res), pooled=sum(ex) / so, pooled_tiemean=sum(v[0][1] - v[1] for v in res.values()) / so,
                pooled_trim3=sum(ex[3:]) / so, median=st.median(rel), mean=st.mean(rel),
                misses=sum(e > 1e-9 for e in ex), better_than_argmin_s=better, equal=len(res) - better - worse, worse=worse)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-comps", required=True)
    ap.add_argument("--heldout-comps", required=True)
    ap.add_argument("--heldout-table", required=True, help="plan_rerank_table.py output: rung and queued per held-out table")
    ap.add_argument("--train-ids", required=True, help="file with the train-split dataset names (one per line)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    keep = {l.strip().rsplit("/", 1)[-1] for l in open(a.train_ids) if l.strip()}
    train = load(Path(a.train_comps), keep)
    meta = {json.loads(l)["ds"].rsplit("/", 1)[-1]: json.loads(l) for l in open(a.heldout_table)}
    held = load(Path(a.heldout_comps), set(meta))
    print(f"train tables {len(train)} (of {len(keep)} train ids), held-out {len(held)} (of {len(meta)} in the table)")
    if len(held) != len(meta):
        raise SystemExit(f"FAIL LOUD: held-out comps for {len(held)} of {len(meta)} tables")
    # identity: alpha = 1 without credit must be argmin-S's label (the table's argmin_s_min) on every held-out table
    off = [ds for ds, rows in held.items() if abs(pick(rows, "drain", 1.0)[0] - meta[ds]["argmin_s_min"]) > 1e-9 * meta[ds]["argmin_s_min"]]
    print(f"identity alpha=1 vs argmin-S on held-out: {len(held) - len(off)}/{len(held)} equal")
    if off:
        raise SystemExit(f"FAIL LOUD: alpha=1 differs from argmin-S on {off[:5]}")
    out = {"train": {}, "chosen": {}, "heldout": {}}
    for v in VARIANTS:
        out["train"][v] = {str(al): pooled(table_regret(train, v, al)) for al in ALPHAS}
        best = min(ALPHAS, key=lambda al: (out["train"][v][str(al)], -al))
        out["chosen"][v] = best
        print(f"train {v:11s} " + "  ".join(f"a={al:g}: {100 * x:5.2f}%" for al, x in zip(ALPHAS, out["train"][v].values())) + f"  -> alpha {best:g}")
    base = table_regret(held, "drain", 1.0)
    subsets = {"moderate": lambda m: m["rung"] == "moderate", "heavy": lambda m: m["rung"] == "heavy", "all": lambda m: True,
               "loaded": lambda m: m["queued"] > 0}
    arms = [("argmin_s", "drain", 1.0)] + [(f"{v}@{out['chosen'][v]:g}", v, out["chosen"][v]) for v in VARIANTS] + \
           [(f"{v}@{al:g}", v, al) for v in ("drain", "drain+cold") for al in ALPHAS if al not in (1.0, out["chosen"][v])]
    for name, v, al in arms:
        res = table_regret(held, v, al)
        out["heldout"][name] = {}
        for sn, f in subsets.items():
            r = {ds: x for ds, x in res.items() if f(meta[ds])}
            out["heldout"][name][sn] = summary(r, {ds: base[ds] for ds in r})
    pct = lambda x: f"{100 * x:6.2f}%"
    for name, by in out["heldout"].items():
        for sn, m in by.items():
            print(f"held-out {name:16s} {sn:8s} n {m['n']:3d} | pooled {pct(m['pooled'])} (tie-mean {pct(m['pooled_tiemean'])}) trim-3 {pct(m['pooled_trim3'])} "
                  f"| median {pct(m['median'])} mean {pct(m['mean'])} | misses {m['misses']} | vs argmin-S better {m['better_than_argmin_s']} "
                  f"equal {m['equal']} worse {m['worse']}")
    json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
