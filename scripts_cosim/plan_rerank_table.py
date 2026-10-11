#!/usr/bin/env python3
"""plan_rerank_v1 data side: the held-out candidate table and the reference reproduction (offline; no simulation).

Per held-out table with one decision (argmin-S row decisions == 1), from existing outputs only:
  * the GNN plan score of every sweep plan (NEAR_RTT_EVAL_PLAN_SCORES dump, 856206 at a07c9fd0; gnn_selfsearch's convention);
    the GNN top-k is the k best-scored sweep plans (score desc, sweep order breaks ties; unscored plans = outside the slate, last);
  * each plan's true RTT (placements.jsonl via the dump's rtt_map);
  * the argmin-S tie set and labels (fidelity_cd_plans --argmin-s, 7f25baca) and the plain CD plan's label (heldout2/cd.jsonl);
  * rung (source tag), queued at decision, sweep completeness (placement_metadata.json).
Pooled regret = sum(rtt - opt) / sum(opt) over the tables of a subset.

  plan_rerank_table.py --scores <dump.json> [...] --argmin argmins2/cd_expand.jsonl --cd heldout2/cd.jsonl --out-dir DIR
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
from pathlib import Path

KS = (1, 2, 5, 10)
EXACT_CAP = 625
EPS = 1e-9


def key(plan):
    return tuple(tuple(int(v) for v in c) for c in plan)


def tag_of(ds: Path) -> str:
    a = json.load(open(ds / "generation_provenance.json"))["argv"]
    return a[a.index("--source-tag") + 1]


def rung_of(tag: str) -> str:
    return next((r for r in ("heavy", "moderate") if f"_{r}_" in f"_{tag}_"), "other")


def build(dump: dict, argm: dict, cd: dict, view: Path) -> list:
    rows = []
    for rec in dump["records"]:
        ds = rec["dataset_id"]
        if ds not in argm or "plans" not in rec:
            continue
        a, c = argm[ds], cd.get(ds)
        P = [(key(p), sc, float(rtt)) for p, sc, rtt in rec["plans"]]
        rtts = [p[2] for p in P]
        opt = min(rtts)
        if abs(opt - rec["opt_rtt"]) > 1e-6 * max(1.0, opt):
            raise SystemExit(f"FAIL LOUD: {ds} sweep min {opt} != opt_rtt {rec['opt_rtt']}")
        order = sorted(range(len(P)), key=lambda i: (P[i][1] is None, -(P[i][1] or 0.0), i))
        idx = {p[0]: i for i, p in enumerate(P)}
        ties = [(key(t["plan"]), float(t["label"])) for t in a["argmin_s"]["tied"]]
        meta = json.load(open(view / ds / "placement_metadata.json"))
        srt = sorted(rtts)
        second_distinct = next((x for x in srt if x > opt * (1 + EPS)), None)
        rows.append(dict(
            ds=ds, rung=rung_of(tag_of(view / ds)), queued=int(a.get("queued") or 0), n_tasks=a["n_tasks"],
            n_sweep=len(P), n_slate=sum(p[1] is not None for p in P), sweep_complete=bool(meta.get("sweep_complete")),
            num_placements=meta.get("num_placements"), opt=opt, opt_plan=[list(x) for x in P[rtts.index(opt)][0]],
            top10=[dict(plan=[list(x) for x in P[i][0]], score=P[i][1], rtt=P[i][2]) for i in order[:10]],
            served_regret=rec.get("regret"),
            argmin_s=[dict(plan=[list(x) for x in p], label=l, in_sweep=p in idx) for p, l in ties],
            argmin_s_min=min(l for _, l in ties), argmin_s_mean=st.mean(l for _, l in ties), argmin_s_max=max(l for _, l in ties),
            argmin_s_in_top10=any(p in {key(P[i][0]) for i in order[:10]} for p, _ in ties),
            cd_rtt=(float(c["label_of_plan"]) if c and "error" not in c and c.get("matched_rows") and c.get("label_of_plan") is not None else None),
            second=srt[1] if len(srt) > 1 else None, second_distinct=second_distinct))
    return rows


def pooled(rows, f):
    v = [(f(r), r["opt"]) for r in rows]
    v = [(x, o) for x, o in v if x is not None]
    return (sum(x - o for x, o in v) / sum(o for _, o in v), len(v)) if v else (None, 0)


def measures(rows: list) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    top = lambda r, k: min(t["rtt"] for t in r["top10"][:k])
    m = {"n": n}
    m["argmin_s_min"], _ = pooled(rows, lambda r: r["argmin_s_min"])
    m["argmin_s_mean_tie"], _ = pooled(rows, lambda r: r["argmin_s_mean"])
    m["argmin_s_max"], _ = pooled(rows, lambda r: r["argmin_s_max"])
    m["plan_score_top1"], _ = pooled(rows, lambda r: r["top10"][0]["rtt"])
    for k in KS:
        m[f"oracle_top{k}"], _ = pooled(rows, lambda r, k=k: top(r, k))
    m["oracle_union"], _ = pooled(rows, lambda r: min(x for x in (top(r, 10), r["argmin_s_min"], r["cd_rtt"]) if x is not None))
    m["cd"], m["cd_n"] = pooled(rows, lambda r: r["cd_rtt"])
    m["served_decode"], m["served_n"] = pooled(rows, lambda r: None if r["served_regret"] is None else r["opt"] + r["served_regret"])
    m["argmin_s_miss_share"] = sum(r["argmin_s_min"] > r["opt"] * (1 + EPS) for r in rows) / n
    m["argmin_s_miss_share_any_tie"] = sum(r["argmin_s_max"] > r["opt"] * (1 + EPS) for r in rows) / n
    m["argmin_s_in_top10"] = sum(r["argmin_s_in_top10"] for r in rows) / n
    m["slate_le_625"] = sum(r["n_slate"] <= EXACT_CAP for r in rows) / n
    m["sweep_le_625"] = sum(r["n_sweep"] <= EXACT_CAP for r in rows) / n
    m["sweep_incomplete"] = sum(not r["sweep_complete"] for r in rows)
    m["unscored_in_top10"] = sum(any(t["score"] is None for t in r["top10"][:min(10, r["n_sweep"])]) and r["n_slate"] >= 10 for r in rows)
    m["argmin_s_plan_not_in_sweep"] = sum(not any(t["in_sweep"] for t in r["argmin_s"]) for r in rows)
    m["near_tie_top2_1pct"] = sum(r["second"] is not None and r["second"] <= r["opt"] * 1.01 for r in rows) / n
    m["near_tie_distinct_1pct"] = sum(r["second_distinct"] is not None and r["second_distinct"] <= r["opt"] * 1.01 for r in rows) / n
    m["single_rtt_tables"] = sum(r["second_distinct"] is None for r in rows)
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--argmin", required=True)
    ap.add_argument("--cd", required=True)
    ap.add_argument("--view", default="/share/nikola.lukic/accel_prod/20261010_061132/cache_run/view")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    view = Path(a.view)
    pre = "gnn_datasets_wf1_heldout/"
    argm, cd = {}, {}
    for l in open(a.argmin):
        r = json.loads(l)
        if "error" not in r and r.get("decisions") == 1:
            argm[pre + r["ds"].rsplit("/", 1)[-1]] = r
    for l in open(a.cd):
        r = json.loads(l)
        cd[pre + r["ds"].rsplit("/", 1)[-1]] = r
    print(f"argmin-S tables with decisions == 1: {len(argm)}; CD rows {len(cd)}")
    os.makedirs(a.out_dir, exist_ok=True)
    summary = {}
    for path in a.scores:
        dump = json.load(open(path))
        name = Path(path).stem
        rows = build(dump, argm, cd, view)
        with open(Path(a.out_dir) / f"table_{name}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        subsets = {
            "all": rows,
            "moderate": [r for r in rows if r["rung"] == "moderate"],
            "heavy": [r for r in rows if r["rung"] == "heavy"],
            "loaded": [r for r in rows if r["queued"] > 0],
            "heavy_loaded": [r for r in rows if r["rung"] == "heavy" and r["queued"] > 0],
            "moderate_loaded": [r for r in rows if r["rung"] == "moderate" and r["queued"] > 0],
            "other_rung": [r for r in rows if r["rung"] == "other"],
        }
        summary[name] = {s: measures(v) for s, v in subsets.items()}
        print(f"== {name}  (checkpoint {dump.get('checkpoint')})")
        pct = lambda x: "   -  " if x is None else f"{100 * x:5.1f}%"
        for s, m in summary[name].items():
            if not m["n"]:
                print(f"  {s:16s} n 0")
                continue
            print(f"  {s:16s} n {m['n']:3d} | argmin-S {pct(m['argmin_s_min'])} (tie mean {pct(m['argmin_s_mean_tie'])}, max {pct(m['argmin_s_max'])}) "
                  f"| top-1 {pct(m['plan_score_top1'])} | oracle top-2/5/10 {pct(m['oracle_top2'])} {pct(m['oracle_top5'])} {pct(m['oracle_top10'])} "
                  f"| union {pct(m['oracle_union'])} | CD {pct(m['cd'])} (n {m['cd_n']}) | served decode {pct(m['served_decode'])} (n {m['served_n']}) "
                  f"| argmin-S misses {pct(m['argmin_s_miss_share'])}, argmin-S in top-10 {pct(m['argmin_s_in_top10'])} "
                  f"| slate<=625 {pct(m['slate_le_625'])}, sweep<=625 {pct(m['sweep_le_625'])}, sweep incomplete {m['sweep_incomplete']}, "
                  f"argmin-S plan not in sweep {m['argmin_s_plan_not_in_sweep']}, unscored in top-10 {m['unscored_in_top10']} "
                  f"| top-2 within 1 % {pct(m['near_tie_top2_1pct'])} (distinct {pct(m['near_tie_distinct_1pct'])}, single-RTT tables {m['single_rtt_tables']})")
    with open(Path(a.out_dir) / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
