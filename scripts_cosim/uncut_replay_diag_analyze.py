#!/usr/bin/env python3
"""Reader for the Arm 1b replay diagnostic (spec: docs/lineages/uncut_groups_s0_v1.md, "Arm 1b replay diagnostic" + amendments). INTERIM-DESCRIPTIVE.

  uncut_replay_diag_analyze.py --replay-dir R --ils-dir I --draw DRAW.json --out OUT.json [--md OUT.md]

Every table is per bucket (le10 / 11_50 / 51_100 / 101_256) and per label type (exact / ils / hand), group-bootstrap 95 % CIs (2000 resamples, seed 0), never pooled-only.
MYOPIC-REPLAY: the replay sees only the group's own tasks at t = 0; this is offline regret, not live latency.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

BUCKETS = ("le10", "11_50", "51_100", "101_256")
LARGE = ("11_50", "51_100", "101_256")
FINALS = ("gnn_eng_s1_final", "gnn_eng_s2_final", "twin_eng_s1_final", "twin_eng_s2_final")
BV = {r: r.replace("_final", "_bv") for r in FINALS}
CAVEAT = "MYOPIC-REPLAY: offline regret at t = 0 on the group's own tasks, not live latency."
B = 2000


def boot(x: Sequence[float], stat=np.mean, seed: int = 0):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    s = x[idx].mean(axis=1) if stat is np.mean else np.array([stat(x[i]) for i in idx])
    return {"n": int(len(x)), "est": float(stat(x)), "lo": float(np.percentile(s, 2.5)), "hi": float(np.percentile(s, 97.5))}


def boot_pair(a: Sequence[float], b: Sequence[float], seed: int = 0):
    """CI of mean(a) - mean(b) over the same groups."""
    return boot(np.asarray(a, float) - np.asarray(b, float), seed=seed)


def fmt(c, pct=True):
    if c is None:
        return "n/a"
    k = 100 if pct else 1
    return f"{k * c['est']:.2f} [{k * c['lo']:.2f}, {k * c['hi']:.2f}] (n={c['n']})"


def load(d: str, pat: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for f in sorted(Path(d).glob(pat)):
        out.update(json.load(open(f)))
    return out


def reg(r: Dict[str, Any], plan: str) -> float:
    return (r["rtt"][plan] - r["rtt"]["label"]) / r["rtt"]["label"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-dir", required=True)
    ap.add_argument("--ils-dir", default="")
    ap.add_argument("--draw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default="")
    ap.add_argument("--corpus", default="/share/nikola.lukic/uncut_arm1/corpus")
    a = ap.parse_args(argv)
    draw = json.load(open(a.draw))
    rep = load(a.replay_dir, "r_*.json")
    ils = load(a.ils_dir, "i_*.json") if a.ils_dir else {}
    meta = draw["meta"]
    val = [g for g in draw["val"] if g in rep]
    train = [g for g in draw["train"] if g in rep]
    topup = {g for v in draw.get("ils_topup_train", {}).values() for g in v}
    out: Dict[str, Any] = {"caveat": CAVEAT, "n_val": len(val), "n_train": len(train), "classes": {}}
    md: List[str] = [f"# Arm 1b replay diagnostic (INTERIM-DESCRIPTIVE)\n\n{CAVEAT}\n"]

    def sel(groups, bucket=None, label=None):
        return [g for g in groups if (bucket is None or meta[g]["bucket"] == bucket) and (label is None or meta[g]["label"] == label)]

    # ---- composition ---------------------------------------------------------------------------------------------------------------------------
    comp = {b: {k: len(sel(val, b, k)) for k in ("exact", "ils", "hand")} for b in BUCKETS}
    out["val_composition"] = comp
    # ---- headroom, hand, random, models --------------------------------------------------------------------------------------------------------
    tables: Dict[str, Any] = {}
    cells = [(b, k) for b in BUCKETS for k in ("exact", "ils", "hand") if sel(val, b, k)] + [(b, "all") for b in BUCKETS] + [("pooled", "all")]
    for b, k in cells:
        gs = sel(val, None if b == "pooled" else b, None if k == "all" else k)
        if not gs:
            continue
        cell: Dict[str, Any] = {"n_groups": len(gs), "n_tasks": int(sum(rep[g]["n"] for g in gs))}
        cell["headroom_hand_vs_label_replay"] = boot([(rep[g]["rtt"]["hand"] - rep[g]["rtt"]["label"]) / rep[g]["rtt"]["label"] for g in gs])
        cell["random_regret"] = boot([reg(rep[g], "rand") for g in gs])
        cell["share_hand_better_than_label"] = boot([float(rep[g]["rtt"]["hand"] < rep[g]["rtt"]["label"] - 1e-9) for g in gs])
        cell["hand_regret"] = boot([reg(rep[g], "hand") for g in gs])
        for run in FINALS:
            d = [rep[g]["decode"][run] for g in gs]
            ex = [float(x["exact"]) for x in d]
            nc = np.array([x["n_choice"] for x in d], float)
            ag = np.array([x["agree"] for x in d], float)
            tf = np.array([x["tf_hit"] for x in d], float)
            mr = [reg(rep[g], run) for g in gs]
            miss = [m for m, e in zip(mr, ex) if not e]
            hit = [m for m, e in zip(mr, ex) if e]
            cell[run] = {"plan_exact": boot(ex), "per_task_match_free": boot(list(ag / np.maximum(nc, 1))), "per_task_acc_teacher_forced": boot(list(tf / np.maximum(nc, 1))),
                         "task_weighted_match_free": float(ag.sum() / max(nc.sum(), 1)), "task_weighted_acc_tf": float(tf.sum() / max(nc.sum(), 1)),
                         "regret": boot(mr), "regret_exact_match_groups": boot(hit), "regret_miss_groups": boot(miss),
                         "share_model_beats_hand": boot([float(rep[g]["rtt"][run] < rep[g]["rtt"]["hand"] - 1e-9) for g in gs]),
                         "regret_minus_hand_regret": boot_pair(mr, [reg(rep[g], "hand") for g in gs])}
        tables[f"{b}|{k}"] = cell
    out["tables"] = tables
    md.append("## Val tables (mean % [95 % group-bootstrap CI])\n")
    for key, c in tables.items():
        md.append(f"### {key}  (groups {c['n_groups']}, tasks {c['n_tasks']})\n")
        md.append(f"- headroom (hand - label)/label: {fmt(c['headroom_hand_vs_label_replay'])}; hand regret {fmt(c['hand_regret'])}; random regret {fmt(c['random_regret'])}; hand better than label: {fmt(c['share_hand_better_than_label'])}")
        for run in FINALS:
            r = c[run]
            md.append(f"- {run}: plan_exact {fmt(r['plan_exact'])}; per-task match (free) {fmt(r['per_task_match_free'])}; TF acc {fmt(r['per_task_acc_teacher_forced'])}; regret {fmt(r['regret'])} "
                      f"(exact groups {fmt(r['regret_exact_match_groups'])}; miss groups {fmt(r['regret_miss_groups'])}); model-minus-hand regret {fmt(r['regret_minus_hand_regret'])}; beats hand {fmt(r['share_model_beats_hand'])}")
        md.append("")
    # ---- gnn vs twin ----------------------------------------------------------------------------------------------------------------------------
    gv = {}
    for b in list(BUCKETS) + ["pooled"]:
        gs = sel(val, None if b == "pooled" else b)
        for s in (1, 2):
            gv[f"{b}|seed{s}"] = boot_pair([reg(rep[g], f"gnn_eng_s{s}_final") for g in gs], [reg(rep[g], f"twin_eng_s{s}_final") for g in gs])
    out["gnn_minus_twin_regret"] = gv
    md.append("## gnn minus twin replay regret (pp, same seed)\n")
    md += [f"- {k}: {fmt(v)}" for k, v in gv.items()]
    # ---- train vs val -----------------------------------------------------------------------------------------------------------------------------
    tv = {}
    for b in list(BUCKETS) + ["pooled"]:
        for run in FINALS:
            gt, gvv = sel(train, None if b == "pooled" else b), sel(val, None if b == "pooled" else b)
            if not gt:
                continue
            rt, rv = [reg(rep[g], run) for g in gt], [reg(rep[g], run) for g in gvv]
            ht, hv = [reg(rep[g], "hand") for g in gt], [reg(rep[g], "hand") for g in gvv]

            def acc(gs):
                return sum(rep[g]["decode"][run]["agree"] for g in gs) / max(1, sum(rep[g]["decode"][run]["n_choice"] for g in gs))
            tv[f"{b}|{run}"] = {"train_regret": boot(rt), "val_regret": boot(rv), "val_minus_train_regret_pp": float(100 * (np.mean(rv) - np.mean(rt))),
                                "train_hand_regret": boot(ht), "train_model_minus_hand": boot_pair(rt, ht), "train_acc_free": acc(gt), "val_acc_free": acc(gvv),
                                "train_minus_val_acc_pp": float(100 * (acc(gt) - acc(gvv))), "n_train": len(gt), "n_val": len(gvv)}
    out["train_vs_val"] = tv
    # ---- classes ---------------------------------------------------------------------------------------------------------------------------------
    cl = out["classes"]
    # S-MISMATCH (<= 10 groups)
    le = sel(val, "le10")
    sm = [float(rep[g]["rtt"]["label"] > rep[g]["rtt"]["hand"] + 1e-9) for g in le]
    cl["S-MISMATCH"] = {"share_label_worse_than_hand_le10": boot(sm), "fires": bool(sm and np.mean(sm) >= 0.10)}
    # BUG
    ilsval = sel(val, label="ils") + sel(sorted(topup), label="ils")
    bug = [g for g in ilsval if g in rep and rep[g]["rtt"]["label"] > rep[g]["rtt"]["hand"] + 1e-9 * max(1.0, rep[g]["rtt"]["hand"])]
    cl["BUG"] = {"n_ils_groups_checked": len(ilsval), "n_label_worse_than_hand": len(bug), "fires": bool(bug)}
    # TOO-EASY: H = share_ILS x mean r, task-weighted
    h = {}
    for b in LARGE:
        gs = sel(val, b)
        tw = sum(meta[g]["n_tasks"] for g in gs)
        ig = sel(gs, label="ils")
        share = sum(meta[g]["n_tasks"] for g in ig) / max(tw, 1)
        mr = sum(meta[g]["n_tasks"] * (meta[g]["r"] or 0.0) for g in ig) / max(sum(meta[g]["n_tasks"] for g in ig), 1)
        h[b] = {"share_ILS_task_weighted": share, "mean_r_task_weighted": mr, "H": share * mr, "n_groups": len(gs), "n_ils": len(ig)}
    cl["TOO-EASY"] = {"H_by_bucket": h, "fires": bool(h["51_100"]["H"] < 0.01 and h["101_256"]["H"] < 0.01)}
    # UNDERFIT / OVERFIT / EXPOSURE / AMBIGUOUS per run and bucket
    uf, of, ex, am = {}, {}, {}, {}
    for run in FINALS:
        for b in list(LARGE) + ["pooled"]:
            t = tv.get(f"{b}|{run}")
            if not t:
                continue
            gap_tm = t["train_model_minus_hand"]
            uf[f"{b}|{run}"] = bool(gap_tm["est"] >= 0 and abs(t["val_minus_train_regret_pp"]) < 0.5)
            of[f"{b}|{run}"] = bool(t["val_minus_train_regret_pp"] > 1.0 and t["train_minus_val_acc_pp"] > 5.0)
            gs = sel(val, None if b == "pooled" else b)
            tf = sum(rep[g]["decode"][run]["tf_hit"] for g in gs) / max(1, sum(rep[g]["decode"][run]["n_choice"] for g in gs))
            fr = sum(rep[g]["decode"][run]["agree"] for g in gs) / max(1, sum(rep[g]["decode"][run]["n_choice"] for g in gs))
            ex[f"{b}|{run}"] = {"tf_minus_free_pp": 100 * (tf - fr), "fires": bool(b in LARGE and 100 * (tf - fr) > 10)}
            miss = [g for g in gs if not rep[g]["decode"][run]["exact"]]
            amb = np.mean([reg(rep[g], run) <= 0.005 for g in miss]) if miss else None
            am[f"{b}|{run}"] = {"share_noncopied_within_0.5pct": None if amb is None else float(amb), "n_noncopied": len(miss), "fires": bool(amb is not None and amb >= 0.5)}
    cl["UNDERFIT"] = uf
    cl["OVERFIT_TOPO_SHIFT"] = of
    cl["EXPOSURE"] = ex
    cl["AMBIGUOUS"] = am
    cl["UNDERFIT_note"] = "val-only in the 101_256 bucket (n = 21); the model-based classes never use the train top-up"
    # OBJECTIVE (ep_bv -> ep100): val, and the train 300 as the confirmation set
    obj = {}
    for run in FINALS:
        for name, groups in (("val", val), ("train300", train)):
            gs = sel(groups)
            acc_f = np.mean([rep[g]["decode"][run]["agree"] / max(1, rep[g]["decode"][run]["n_choice"]) for g in gs])
            acc_b = np.mean([rep[g]["decode"][BV[run]]["agree"] / max(1, rep[g]["decode"][BV[run]]["n_choice"]) for g in gs])
            d_reg = boot_pair([reg(rep[g], run) for g in gs], [reg(rep[g], BV[run]) for g in gs])
            obj[f"{run}|{name}"] = {"acc_rise_pp": float(100 * (acc_f - acc_b)), "testable": bool(100 * (acc_f - acc_b) >= 3.0), "regret_final_minus_bv": d_reg,
                                    "regret_not_improved": bool(d_reg["hi"] >= 0)}
    for run in FINALS:
        v, t = obj[f"{run}|val"], obj[f"{run}|train300"]
        v["fires_on_val"] = bool(v["testable"] and v["regret_not_improved"])
        v["label"] = ("OBJECTIVE (selection-biased)" if v["fires_on_val"] else "not fired") if not v["fires_on_val"] else \
            ("OBJECTIVE confirmed on train300" if (t["testable"] and t["regret_not_improved"]) else "OBJECTIVE (selection-biased), unconfirmed on train300")
    cl["OBJECTIVE (ep_bv -> ep100)"] = obj
    cl["SHIFT"] = "not testable: looks 1 and 2 kept no live batch snapshots (S6)"
    # ---- LABEL-SEARCH ---------------------------------------------------------------------------------------------------------------------------
    ls: Dict[str, Any] = {"note": "needs the ILS array"} if not ils else {}
    if ils:
        for b in LARGE:
            for kind in ("ils", "hand"):
                gs = [g for g in ils if meta[g]["bucket"] == b and meta[g]["label"] == kind]
                vgs = [g for g in gs if g not in topup]
                for name, s in (("val_only", vgs), ("val_plus_topup", gs)):
                    gaps = [(ils[g]["rtt_label"] - ils[g]["best_of_4"]) / ils[g]["rtt_label"] for g in s]
                    if not gaps:
                        continue
                    ls[f"{b}|{kind}|{name}"] = {"n": len(gaps), "gap_mean": boot(gaps), "share_gap_gt_1pct": boot([float(x > 0.01) for x in gaps]),
                                                "fires": bool(kind == "ils" and np.mean([x > 0.01 for x in gaps]) >= 0.20),
                                                "capped_runs": int(sum(r["capped"] for g in s for r in ils[g]["runs"])), "runs": int(4 * len(s))}
        ls["fires_rule"] = "ILS-labelled groups only; fires if the best of 4 beats the label by > 1 % on >= 20 %; in 101_256 on val-only and val+topup agreeing, else val-only. Hand-fallback is reported, never fires."
    cl["LABEL-SEARCH"] = ls
    # label-side cap disclosure: the sampled ILS-labelled groups whose label search hit the labeller's 600 s cap
    capped = 0
    sampled = sorted({g for v in draw["ils_subsample"].values() for g in v} | topup)
    for g in sampled:
        kind, ds = g.split("/")
        with open(f"{a.corpus}/{'train' if 'train' in kind else 'heldout'}/{ds}/placements/placements.jsonl") as fh:
            hd = json.loads(fh.readline())
        capped += int(hd.get("stopped") == "seconds")
    cl["label_side_capped"] = {"sampled_ils_labelled": len(sampled), "label_search_stopped_by_seconds": capped}
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    md.append("\n## Classes\n")
    md.append("```json\n" + json.dumps(cl, indent=1, default=float) + "\n```\n")
    if a.md:
        open(a.md, "w").write("\n".join(md))
    print(json.dumps({k: cl[k] for k in ("BUG", "S-MISMATCH", "TOO-EASY", "label_side_capped")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
