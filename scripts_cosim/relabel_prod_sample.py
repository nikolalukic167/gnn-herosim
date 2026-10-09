"""Registered measurement: relabel a stratified sample of production datasets at the current replay (F) and compare with the stored (E) labels.
  relabel_prod_sample.py sample  OUTLIST  --roots TRAIN,HELDOUT [--per-rung 200 --seed N]   writes the dataset list (written BEFORE the relabel)
  relabel_prod_sample.py report  RELABEL.json LIST                                          argmin changes and the regret of E's argmin under F's labels, per rung
Rungs come from the dataset's cell config path (inputs/wf1_<rung>); train and held-out are sampled in proportion to their share of the rung's complete sweeps."""
import argparse, json, math, os, random, statistics as st, sys
from pathlib import Path

sys.path.insert(0, os.getcwd())


def rung_of(ds: Path) -> str:
    p = json.load(open(ds / "warm_snapshot.json"))["provenance"]["cell_config"]
    return p.split("/inputs/")[1].split("/")[0].replace("wf1_", "")


def sample(a):
    from src.placement.sweep_status import sweep_complete
    pools = {}
    for root in a.roots.split(","):
        split = "heldout" if "heldout" in root else "train"
        for d in sorted(Path(root).glob("ds_*")):
            if not sweep_complete(d):
                continue
            pools.setdefault(rung_of(d), {}).setdefault(split, []).append(str(d))
    rng = random.Random(a.seed)
    out = []; meta = {}
    for rung in sorted(pools):
        tr, he = pools[rung].get("train", []), pools[rung].get("heldout", [])
        n_he = round(a.per_rung * len(he) / (len(tr) + len(he)))
        n_tr = a.per_rung - n_he
        pick = rng.sample(tr, min(n_tr, len(tr))) + rng.sample(he, min(n_he, len(he)))
        out += sorted(pick)
        meta[rung] = dict(complete_train=len(tr), complete_heldout=len(he), picked_train=min(n_tr, len(tr)), picked_heldout=min(n_he, len(he)))
    open(a.outlist, "w").write("\n".join(out) + "\n")
    json.dump(dict(seed=a.seed, per_rung=a.per_rung, rungs=meta), open(a.outlist + ".meta.json", "w"), indent=1)
    print(json.dumps(meta))


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, math.ceil(p * len(v)) - 1)] if v else float("nan")


def report(a):
    res = [x for x in json.load(open(a.relabel)) if "error" not in x]
    errs = [x for x in json.load(open(a.relabel)) if "error" in x]
    byr = {}
    for x in res:
        byr.setdefault(rung_of(Path(x["ds"])), []).append(x)
    print(f"datasets relabelled {len(res)}, errors {len(errs)}", [(Path(e['ds']).name, e['error'][:80]) for e in errs[:5]])
    for rung in sorted(byr):
        xs = byr[rung]; nch = 0; arg = 0; regs = []; plans = 0; pch = 0; big = 0; mx = 0.0; am_abs = []
        for x in xs:
            rows = x["rows"]; plans += len(rows)
            for r in rows:
                d = abs(r["new"] - r["old"]); pch += d / r["old"] > 1e-6; big += d / r["old"] > 0.01; mx = max(mx, d)
            nch += any(abs(r["new"] - r["old"]) / r["old"] > 1e-6 for r in rows)
            e_arg = min(rows, key=lambda r: (r["old"], r["plan_idx"]))     # E's argmin: always in the sample (best by stored label)
            f_min = min(rows, key=lambda r: (r["new"], r["plan_idx"]))
            regret = (e_arg["new"] - f_min["new"]) / f_min["new"]
            regs.append(regret)
            if regret > 1e-9:
                arg += 1; am_abs.append(e_arg["new"] - f_min["new"])
        n = len(xs)
        print(f"== {rung}: datasets {n}; any plan label changed {nch} ({100 * nch / n:.1f}%); plans {plans}, changed {pch}, >1% {big}, max abs change {mx:.4f} s")
        print(f"   argmin changes (sampled plans): {arg}/{n} = {100 * arg / n:.2f}%; regret of E's argmin under F: median {100 * st.median(regs):.4f}% p90 {100 * q(regs, .9):.4f}% max {100 * max(regs):.4f}%; datasets with regret >1%: {sum(r > .01 for r in regs)}, >5%: {sum(r > .05 for r in regs)}; abs regret max {max(am_abs) if am_abs else 0:.4f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample"); s.add_argument("outlist"); s.add_argument("--roots", required=True); s.add_argument("--per-rung", type=int, default=200); s.add_argument("--seed", type=int, default=20261009)
    r = sub.add_parser("report"); r.add_argument("relabel"); r.add_argument("list")
    a = ap.parse_args()
    sample(a) if a.cmd == "sample" else report(a)
