#!/usr/bin/env python3
"""Expected move-regret lever, stage 1 (uncut_groups_s0_v1 Record, "Offline signal before any live gate"): the lambda runs against Arm 1b at the same seed on the val groups.

  uncut_lever_stage1.py prep    --out SHARDS_DIR [--per-shard 25]                       graph shards of the 1108 val groups (one pass over the Arm 1b cache)
  uncut_lever_stage1.py decode  --shard SHARD.pkl --out OUT.json --mask MASK.pkl        free masked_topo decode + teacher-forced accuracy of every run (Arm 1b gnn and the lambda runs) per group,
                                                                                         plus, on the ILS-labelled val groups, the unscored mass and the unscored pick
  uncut_lever_stage1.py replay  ...                                                     = uncut_replay_diag.py replay with this script's RUNS (engine replays: label, hand, random, every run)
  uncut_lever_stage1.py analyze --replay-dir D --decode-dir D --out OUT.json --md OUT.md

The decode, the teacher-forced accuracy, the replay and the regret definition ((R(plan) - R(label)) / R(label)) are `uncut_replay_diag.py`'s, unchanged. MYOPIC-REPLAY: offline regret at t = 0 on the
group's own tasks, not live latency. Descriptive until every registered run is present.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts_cosim"))
import uncut_replay_diag as RD  # noqa: E402

DRAW = REPO / "docs/lineages/uncut_groups_s0_v1/replay_diag_draw.json"
# name -> checkpoint stem; every name starts with gnn_eng (one process, one arm)
RUNS = {
    "gnn_eng_s1_base": "uncut-arm1b-gnn-eng-seed1-final",
    "gnn_eng_s2_base": "uncut-arm1b-gnn-eng-seed2-final",
    "gnn_eng_s1_l0p5": "uncut-arm3-gnn-eng-l0p5-seed1-final",
    "gnn_eng_s2_l0p5": "uncut-arm3-gnn-eng-l0p5-seed2-final",
    "gnn_eng_s1_l2p0": "uncut-arm3-gnn-eng-l2p0-seed1-final",
    "gnn_eng_s2_l2p0": "uncut-arm3-gnn-eng-l2p0-seed2-final",
}
BUCKETS = ("le10", "11_50", "51_100", "101_256")
B = 2000


def present_runs() -> List[str]:
    return [r for r, stem in RUNS.items() if (Path(RD.MODELS) / (stem + ".pt")).exists()]


def cmd_prep(a):
    draw = json.load(open(DRAW))
    val_only = {"train": [], "val": draw["val"], "ils_topup_train": {}}
    tmp = Path(a.out).with_suffix(".draw.json")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(val_only, open(tmp, "w"))
    RD.cmd_prep(argparse.Namespace(draw=str(tmp), out=a.out, per_shard=a.per_shard))


def unscored_stats(model, g, u, dec) -> Dict[str, float]:
    """Mean-probability numerator and step count on the engine's never-scored candidates under the label's prefix (teacher-forced), and how many of the free decode's choices are such candidates."""
    import torch
    from src.policy.gnn.partial_state_edges import make_partial_state_score_fn
    from src.policy.gnn.seq_decode import topological_task_order
    from src.policy.tabular.reduced_features import build_partial_state_context_from_graph

    n = int(g.n_tasks)
    ctx = build_partial_state_context_from_graph(g)
    ctx.node_caps = g.partial_state_ctx["node_caps_by_alpha"][RD.ALPHA]
    choice = [t for t in range(n) if len(g.task_logit_to_placement[t]) > 1]
    lp = [tuple(int(v) for v in g.task_logit_to_placement[t][int(g.tied_optimal_logit_plans[RD.ALPHA][0][t])]) for t in range(n)]
    mass = 0.0
    with torch.no_grad():
        sf = make_partial_state_score_fn(model, g, ctx)
        try:
            committed: Dict[int, Any] = {}
            for t in topological_task_order(n, g.dag_parents):
                lg = sf(t, committed)
                if t in choice:
                    p = torch.softmax(lg.double(), dim=0)
                    mass += float((p * torch.as_tensor(u[t], dtype=torch.double)).sum())
                committed[t] = lp[t]
        finally:
            if hasattr(sf, "close"):
                sf.close()
    pick = None
    if dec is not None:
        pick = 0
        for t in choice:
            cands = [tuple(int(v) for v in c) for c in g.task_logit_to_placement[t]]
            pick += int(u[t][cands.index(tuple(dec[t]))])
    return {"tf_mass": mass, "tf_steps": len(choice), "pick_unscored": pick, "pick_steps": len(choice) if pick is not None else 0}


def cmd_decode(a):
    shard = pickle.load(open(a.shard, "rb"))
    names = present_runs()
    RD.RUNS.update({n: RUNS[n] for n in names})
    models = RD._load_models(names)
    mask = pickle.load(open(a.mask, "rb"))["groups"]
    res: Dict[str, Any] = {}
    t0 = time.time()
    for g in shard:
        gid = str(g.dataset_id)
        res[gid] = {}
        for n, m in models.items():
            r = RD.decode_group(m, g)
            if gid in mask:
                u = mask[gid]["u"]
                if mask[gid]["n"] != int(g.n_tasks) or any(len(u[t]) != len(g.task_logit_to_placement[t]) for t in range(int(g.n_tasks))):
                    raise SystemExit(f"FAIL LOUD: unscored mask of {gid} does not match the graph's candidate sets")
                r.update(unscored_stats(m, g, u, r["decoded"]))
            res[gid][n] = r
    json.dump({"runs": names, "groups": res, "wall": time.time() - t0}, open(a.out, "w"))
    print(f"decode: {len(res)} groups x {len(names)} checkpoints in {time.time() - t0:.0f} s")


def cmd_replay(a):
    names = present_runs()
    RD.RUNS.clear()
    RD.RUNS.update({n: RUNS[n] for n in names})
    RD.cmd_replay(a)


def boot_mean(x, seed=20261016):
    x = np.asarray(x, float)
    if len(x) == 0:
        return None
    s = x[np.random.default_rng(seed).integers(0, len(x), size=(B, len(x)))].mean(axis=1)
    return {"n": int(len(x)), "est": float(x.mean()), "lo": float(np.percentile(s, 2.5)), "hi": float(np.percentile(s, 97.5))}


def f(c, k=100.0):
    return "n/a" if c is None else f"{k * c['est']:+.2f} [{k * c['lo']:+.2f}, {k * c['hi']:+.2f}] (n={c['n']})"


def cmd_analyze(a):
    draw = json.load(open(DRAW))
    meta = draw["meta"]
    rep: Dict[str, Any] = {}
    for fl in sorted(Path(a.replay_dir).glob("r_*.json")):
        rep.update(json.load(open(fl)))
    dec = RD.load_decodes(a.decode_dir)
    val = [g for g in draw["val"] if g in rep]
    if len(val) != len(draw["val"]):
        raise SystemExit(f"FAIL LOUD: replay covers {len(val)} of {len(draw['val'])} val groups")
    runs = [r for r in RUNS if any(r in rep[g]["rtt"] for g in val)]
    for r in runs:
        miss = [g for g in val if r not in rep[g]["rtt"]]
        if miss:
            raise SystemExit(f"FAIL LOUD: {r} has no decoded plan on {len(miss)} val groups, e.g. {miss[:3]}")
    ident_bad = [g for g in val if rep[g]["stored_label_rtt"] is None or abs(rep[g]["rtt"]["label"] - rep[g]["stored_label_rtt"]) > 1e-9 * max(1.0, abs(rep[g]["stored_label_rtt"]))]
    if ident_bad:
        raise SystemExit(f"FAIL LOUD: label replay differs from the stored RTT on {len(ident_bad)} groups, e.g. {ident_bad[:3]}")

    def reg(g, r):
        return (rep[g]["rtt"][r] - rep[g]["rtt"]["label"]) / rep[g]["rtt"]["label"]

    def tf(g, r):
        d = rep[g]["decode"][r]
        return d["tf_hit"] / max(d["n_choice"], 1)

    def sel(buckets, label=None):
        return [g for g in val if meta[g]["bucket"] in buckets and (label is None or meta[g]["label"] == label)]

    sets = {"51_256": sel(("51_100", "101_256")), "51_256_ils": sel(("51_100", "101_256"), "ils"), "le50": sel(("le10", "11_50")), **{b: sel((b,)) for b in BUCKETS}, "all_val": val}
    out: Dict[str, Any] = {"runs": runs, "n_val": len(val), "sets": {k: len(v) for k, v in sets.items()}, "pairs": {}, "descriptive": {}}
    pairs = [(r, r.replace(l, "base")) for r in runs for l in ("l0p5", "l2p0") if r.endswith(l)]
    for r, base in pairs:
        if base not in runs:
            continue
        cell: Dict[str, Any] = {}
        for k, gs in sets.items():
            if not gs:
                continue
            m, b0 = [reg(g, r) for g in gs], [reg(g, base) for g in gs]
            d = np.asarray(m) - np.asarray(b0)
            cell[k] = {"n": len(gs), "regret_model": boot_mean(m), "regret_base": boot_mean(b0), "diff_pp": boot_mean(d),
                       "median_model": float(np.median(m)), "median_base": float(np.median(b0)),
                       "share_model_better": float(np.mean(d < -1e-12)), "share_model_worse": float(np.mean(d > 1e-12)),
                       "tf_match_diff": boot_mean([tf(g, r) - tf(g, base) for g in gs]),
                       "free_match_diff": boot_mean([rep[g]["decode"][r]["agree"] / max(rep[g]["decode"][r]["n_choice"], 1) - rep[g]["decode"][base]["agree"] / max(rep[g]["decode"][base]["n_choice"], 1) for g in gs]),
                       "plan_exact_model": float(np.mean([float(rep[g]["decode"][r]["exact"]) for g in gs])), "plan_exact_base": float(np.mean([float(rep[g]["decode"][base]["exact"]) for g in gs]))}
        out["pairs"][f"{r} vs {base}"] = cell
    # descriptive unscored diagnostics on the ILS-labelled val groups that carry the mask
    for r in runs:
        gs = [g for g in val if "tf_mass" in dec[g][r]]
        if not gs:
            continue
        mass = sum(dec[g][r]["tf_mass"] for g in gs) / max(sum(dec[g][r]["tf_steps"] for g in gs), 1)
        pk = [g for g in gs if dec[g][r].get("pick_unscored") is not None]
        pick = sum(dec[g][r]["pick_unscored"] for g in pk) / max(sum(dec[g][r]["pick_steps"] for g in pk), 1)
        out["descriptive"][r] = {"groups": len(gs), "tf_unscored_mass": mass, "free_decode_unscored_pick_rate": pick,
                                 "n_choice_steps": int(sum(dec[g][r]["tf_steps"] for g in gs))}
    json.dump(out, open(a.out, "w"), indent=1)
    md = ["# Lever stage 1 (PARTIAL, DESCRIPTIVE: the lambda 2.0 seed 1 run is not trained yet)" if len(runs) < 6 else "# Lever stage 1",
          "", "MYOPIC-REPLAY: offline regret at t = 0 on the group's own tasks, not live latency. Regret = (R(plan) - R(label)) / R(label); difference = model minus Arm 1b at the same seed (negative = better); "
          f"95 % paired group-bootstrap CI ({B} resamples).", "", f"Runs present: {', '.join(runs)}. Val groups: {len(val)}; subsets {out['sets']}.", ""]
    for name, cell in out["pairs"].items():
        md.append(f"## {name}\n")
        md.append("| subset | n | regret model % | regret Arm 1b % | diff pp [95 % CI] | median model / base % | better / worse share | TF match diff pp | free match diff pp |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for k, c in cell.items():
            md.append(f"| {k} | {c['n']} | {100 * c['regret_model']['est']:.2f} | {100 * c['regret_base']['est']:.2f} | {f(c['diff_pp'])} | {100 * c['median_model']:.2f} / {100 * c['median_base']:.2f} | "
                      f"{c['share_model_better']:.2f} / {c['share_model_worse']:.2f} | {f(c['tf_match_diff'])} | {f(c['free_match_diff'])} |")
        md.append("")
    md += ["## Unscored candidates (ILS-labelled val groups with a mask; descriptive)", "", "| run | groups | choice steps | teacher-forced mass on never-scored | free-decode pick rate of never-scored |", "|---|---|---|---|---|"]
    for r, d in out["descriptive"].items():
        md.append(f"| {r} | {d['groups']} | {d['n_choice_steps']} | {100 * d['tf_unscored_mass']:.3f} % | {100 * d['free_decode_unscored_pick_rate']:.3f} % |")
    open(a.md, "w").write("\n".join(md) + "\n")
    print("\n".join(md))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep"); p.add_argument("--out", required=True); p.add_argument("--per-shard", type=int, default=25)
    d = sub.add_parser("decode"); d.add_argument("--shard", required=True); d.add_argument("--out", required=True); d.add_argument("--mask", required=True)
    rp = sub.add_parser("replay")
    rp.add_argument("--shards", required=True); rp.add_argument("--n-shards", type=int, required=True); rp.add_argument("--task", type=int, required=True)
    rp.add_argument("--stride", type=int, default=8); rp.add_argument("--decode-dir", required=True); rp.add_argument("--draw", required=True); rp.add_argument("--out", required=True)
    an = sub.add_parser("analyze"); an.add_argument("--replay-dir", required=True); an.add_argument("--decode-dir", required=True); an.add_argument("--out", required=True); an.add_argument("--md", required=True)
    a = ap.parse_args(argv)
    {"prep": cmd_prep, "decode": cmd_decode, "replay": cmd_replay, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
