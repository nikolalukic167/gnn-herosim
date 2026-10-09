"""Per replay variant (A..D): median / p95 / max |error| against the isolated live truth over every state of every cell, per rung and pooled,
with the worst states and their absolute misses. usage: summarize_variants.py OUTDIR"""
import glob, json, math, os, statistics as st, sys

O = sys.argv[1]
NAMES = {"A": "A d28d1bb0 (poll, name order)", "B": "B f7711324 (exact_batch, name order)", "C": "C poll + task-id order/timers", "D": "D exact_batch + task-id order/timers (8ada0140)", "E": "E D + due time = pop + latency"}


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, math.ceil(p * len(v)) - 1)] if v else float("nan")


rows = {}
for f in glob.glob(f"{O}/*_*/replay_[ABCDE]_*.jsonl"):
    cell = os.path.basename(os.path.dirname(f)); var = os.path.basename(f).split("_")[1]
    for l in open(f):
        r = json.loads(l)
        if r.get("rel_err") is None:
            r["_fail"] = True
        r["_cell"] = cell
        rows.setdefault(var, []).append(r)
for var in "ABCDE":
    rs = rows.get(var, [])
    ok = [r for r in rs if not r.get("_fail")]
    e = [abs(r["rel_err"]) for r in ok]
    print(f"== {NAMES[var]}: states {len(rs)}, scored {len(ok)}, unscored {len(rs) - len(ok)}: median {100 * st.median(e):.3f}% p95 {100 * q(e, .95):.3f}% max {100 * max(e):.3f}%")
    for rung in "zlmh":
        e2 = [abs(r["rel_err"]) for r in ok if r["_cell"].endswith("_" + rung)]
        if e2:
            print(f"     rung {rung}: n {len(e2)} median {100 * st.median(e2):.3f}% p95 {100 * q(e2, .95):.3f}% max {100 * max(e2):.3f}%")
    for r in sorted(ok, key=lambda r: -abs(r["rel_err"]))[:4]:
        print(f"     worst {r['_cell']} t={r['t']:.1f} err={100 * r['rel_err']:.3f}% live={r['live']:.4f} replay={r['replay']:.4f} abs miss {abs(r['replay'] - r['live']):.4f}")
