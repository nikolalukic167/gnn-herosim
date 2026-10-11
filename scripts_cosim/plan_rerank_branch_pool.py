#!/usr/bin/env python3
"""plan_rerank_v1 Phase 1 candidate pool (data prep): step-0 branch completions U the argmin-S plan(s), per single-decision table, train and
held-out, per checkpoint. Offline; no simulation.

Per candidate: plan, sources (branch:<first-task candidate index> / argmin_s), true RTT (sweep), GNN plan score, summed step logits along
the plan in the masked_topo order (both from NEAR_RTT_EVAL_VARIANTS + NEAR_RTT_EVAL_EXTRA_PLANS), S and its components
(fidelity_cd_plans --s-comps: drain_platform, drain_batch, cold, cold_dup, exec_lat, exch; S = all but cold_dup). Identical plans from
several sources are one candidate with every source listed. argmin-S enters as its whole tie set (usually one plan); the reported
argmin-S arm is the lowest-label tie, the 6.1 % convention.

  --prep   write the extra-plans file (argmin-S ties per table) that the eval job scores
  build    write pool_<ckpt>_<split>.jsonl and print the union-oracle regret per rung

Fail loud: a candidate outside the sweep or the S-components file, an RTT that disagrees with the components row, a table missing
from the eval dump.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

P = Path("/share/nikola.lukic/plan_rerank")
VIEW = Path("/share/nikola.lukic/accel_prod/20261010_061132/cache_run/view")
CKS = {"g4": "accel-replica-v1-gnn-eng-g4-seed1", "physmp_g3": "accel-replica-v1-gnn-eng-physmp-g3-seed1"}
TABLES = {"train": P / "traintable_0cac2d2a", "val": P / "data_2ebd3ba8"}
COMPS = {"train": P / "salpha_10c9081a/comps_train", "val": P / "salpha_10c9081a/comps_heldout"}
KEYS = ("rtt", "drain_platform", "drain_batch", "cold", "cold_dup", "exec_lat", "exch")
key = lambda plan: tuple(tuple(int(v) for v in c) for c in plan)


def table_rows(split, ck):
    return [json.loads(l) for l in open(TABLES[split] / f"table_{CKS[ck]}-best.jsonl")]


def prep(out: Path) -> None:
    extra = {}
    for split in TABLES:
        for r in table_rows(split, "g4"):
            extra[r["ds"]] = [t["plan"] for t in r["argmin_s"]]
    json.dump(extra, open(out, "w"))
    print(f"extra plans for {len(extra)} tables -> {out}")


def comps_of(split, ds):
    j = json.load(open(COMPS[split] / f"{ds.rsplit('/', 1)[-1]}.json"))
    if tuple(j["keys"]) != KEYS:
        raise SystemExit(f"FAIL LOUD: {ds} comps keys {j['keys']}")
    plans = [r["placement_plan"] for r in (json.loads(l) for l in open(VIEW / ds / "placements" / "placements.jsonl") if l.strip())
             if "placement_plan" in r]
    if len(plans) != len(j["rows"]):
        raise SystemExit(f"FAIL LOUD: {ds}: {len(plans)} sweep rows, {len(j['rows'])} comps rows")
    return {key([p[str(i)] for i in range(len(p))]): row for p, row in zip(plans, j["rows"])}


def build(dump_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for ck, stem in CKS.items():
        for split in ("train", "val"):
            recs = {r["dataset_id"]: r for r in json.load(open(dump_dir / f"{stem}-{split}.json"))["records"]}
            tabs = []
            with open(out_dir / f"pool_{ck}_{split}.jsonl", "w") as fh:
                for tr in table_rows(split, ck):
                    rec = recs.get(tr["ds"])
                    if rec is None or "extra" not in rec.get("variants", {}):
                        raise SystemExit(f"FAIL LOUD: {tr['ds']} has no variants/extra record in {stem}-{split}")
                    comps = comps_of(split, tr["ds"])
                    v = rec["variants"]
                    if len(v["extra"]) != len(tr["argmin_s"]):
                        raise SystemExit(f"FAIL LOUD: {tr['ds']} argmin-S ties {len(tr['argmin_s'])}, scored {len(v['extra'])}")
                    cands = {}
                    srcs = [(f"branch:{i}", b) for i, b in enumerate(v["branches"])] + [("argmin_s", e) for e in v["extra"]]
                    for src, c in srcs:
                        k = key(c["plan"])
                        if k not in comps or c["rtt"] is None:
                            raise SystemExit(f"FAIL LOUD: {tr['ds']} candidate {src} outside the sweep / comps")
                        row = comps[k]
                        if abs(row[0] - c["rtt"]) > 1e-6 * max(1.0, c["rtt"]):
                            raise SystemExit(f"FAIL LOUD: {tr['ds']} {src} rtt {c['rtt']} != comps rtt {row[0]}")
                        if k in cands:
                            cands[k]["sources"].append(src)
                            continue
                        sc = dict(zip(KEYS[1:], row[1:]))
                        cands[k] = {"plan": [list(x) for x in k], "sources": [src], "rtt": c["rtt"], "plan_score": c["plan_score"],
                                    "step_logit_sum": c["step_logit_sum"],
                                    "s": sc["drain_platform"] + sc["drain_batch"] + sc["cold"] + sc["exec_lat"] + sc["exch"], "s_comps": sc}
                    lab = {key(t["plan"]): t["label"] for t in tr["argmin_s"]}
                    a_min = min(lab, key=lambda p: (lab[p], p))
                    row = {"ds": tr["ds"], "rung": tr["rung"], "queued": tr["queued"], "n_tasks": tr["n_tasks"], "n_sweep": tr["n_sweep"],
                           "opt": tr["opt"], "argmin_s_plan": [list(x) for x in a_min], "argmin_s_ties": len(lab),
                           "served_plan": rec["combo"], "candidates": list(cands.values())}
                    fh.write(json.dumps(row) + "\n")
                    tabs.append({"rung": tr["rung"], "opt": tr["opt"], "n": len(cands),
                                 "union": min(c["rtt"] for c in cands.values()),
                                 "branches": min(c["rtt"] for c in cands.values() if any(s.startswith("branch") for s in c["sources"])),
                                 "argmin_s": tr["argmin_s_min"], "served": rec["opt_rtt"] + rec["regret"]})
            res = {}
            for rung in ("moderate", "heavy", "all"):
                sub = [t for t in tabs if rung == "all" or t["rung"] == rung]
                so = sum(t["opt"] for t in sub)
                res[rung] = {"n": len(sub), "pool_size_mean": st.mean(t["n"] for t in sub), "pool_size_max": max(t["n"] for t in sub)}
                for arm in ("union", "branches", "argmin_s", "served"):
                    ex = sorted((t[arm] - t["opt"] for t in sub), reverse=True)
                    res[rung][arm] = {"pooled": sum(ex) / so, "trim3": sum(ex[3:]) / so,
                                      "median": st.median((t[arm] - t["opt"]) / t["opt"] for t in sub),
                                      "hit_opt": sum(e <= 1e-9 for e in ex) / len(sub)}
                print(f"{ck:9s} {split:5s} {rung:8s} n {len(sub):4d} pool {res[rung]['pool_size_mean']:.2f} (max {res[rung]['pool_size_max']}) | "
                      + " | ".join(f"{arm} {100 * res[rung][arm]['pooled']:6.2f}% trim3 {100 * res[rung][arm]['trim3']:5.2f}% "
                                   f"hit {100 * res[rung][arm]['hit_opt']:5.1f}%" for arm in ("union", "branches", "argmin_s", "served")), flush=True)
            summary[f"{ck}/{split}"] = res
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=1)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prep", default=None, help="write the extra-plans json here and exit")
    ap.add_argument("--dump-dir")
    ap.add_argument("--out-dir")
    a = ap.parse_args()
    if a.prep:
        prep(Path(a.prep))
        return 0
    build(Path(a.dump_dir), Path(a.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
