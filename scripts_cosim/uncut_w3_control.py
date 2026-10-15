#!/usr/bin/env python3
"""W3 label-prefix control (spec: docs/lineages/uncut_groups_s0_v1.md, Record 2026-10-15 "W3 label-prefix control"). Offline diagnostic, MYOPIC-REPLAY, INTERIM-DESCRIPTIVE.

W3 (`uncut_w3_prefix.py`) froze the MODEL's own prefix and let the ILS complete it. This control freezes the LABEL plan's prefix at the same tasks and points, lets the model's free decode complete
the rest from that prefix, then runs the same ILS with the label prefix frozen. The paired difference (W3 minus control) says whether the room W3 found comes from the model's own prefix.

  uncut_w3_control.py draw    --src w3_draw.json --out w3c_draw.json [--classes ils]     the control groups (search-labelled val groups of 51-100 and 101-256)
  uncut_w3_control.py points  --shards D --draw w3c_draw.json --out points.json         forced-prefix decodes, with the identity checks
  (ILS stage: `uncut_w3_prefix.py ils` on that points.json; rows have `model_key` = "<run>@<frac>")
  uncut_w3_control.py analyze --rows-dir D --w3-rows w3/rows_all.json --out out.json [--md out.md]

Decode, prep and replay are S5's `uncut_replay_diag.py`, unchanged. The only new decode logic is `force_prefix_score_fn`: a wrapper around the served scorer that returns, for a forced task, the
model's own scores with the label candidate lifted above the rest. The wrapper still calls the scorer at every step (its incremental state stays in step) and the committed state it shows later tasks
holds the label placements, as the decoder records the lifted candidate. Nothing in `seq_decode` is edited."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Set, Tuple

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts_cosim")):
    if p not in sys.path:
        sys.path.insert(0, p)

RUNS = ("gnn_eng_s1_final", "gnn_eng_s2_final")
FRACS = (0.25, 0.5)
BUCKETS = ("51_100", "101_256")
LIFT = 1e9
MARGIN_PP = 0.10                    # EXPOSURE-ROOM: the paired share difference (W3 minus control) is at least this ...


def force_prefix_score_fn(score_fn: Callable[[int, Mapping[int, Any]], Sequence[float]], cands: Mapping[int, Sequence[Tuple[int, int]]], label: Sequence[Sequence[int]],
                          forced: Set[int]) -> Callable[[int, Mapping[int, Any]], List[float]]:
    """score_fn(task, committed) -> scores, with the label candidate lifted to the top for every task in `forced`. A forced task whose label placement is not among its candidates is a
    FAIL LOUD error (the label plan was produced over the same candidate lists)."""
    def wrapped(task: int, committed: Mapping[int, Any]) -> List[float]:
        sc = [float(x) for x in score_fn(task, committed)]
        if task in forced:
            want = tuple(int(v) for v in label[task])
            idx = [i for i, c in enumerate(cands[task]) if tuple(int(v) for v in c) == want]
            if len(idx) != 1:
                raise RuntimeError(f"FAIL LOUD: task {task}: the label placement {want} is not exactly one of its {len(cands[task])} candidates")
            sc[idx[0]] = max(sc) + LIFT
        return sc
    if hasattr(score_fn, "close"):
        wrapped.close = score_fn.close      # type: ignore[attr-defined]
    return wrapped


def forced_decode(model, g, label: Sequence[Sequence[int]], forced: Set[int]):
    """S5's `decode_group` decode call (the served masked_topo decode), with `forced` tasks pinned to their label placement through `force_prefix_score_fn`."""
    import torch
    import uncut_replay_diag as R
    from src.policy.gnn.partial_state_edges import make_partial_state_score_fn
    from src.policy.gnn.seq_decode import decode_masked_topo_placement
    from src.policy.tabular.reduced_features import build_partial_state_context_from_graph

    n = int(g.n_tasks)
    ctx = build_partial_state_context_from_graph(g)
    ctx.node_caps = g.partial_state_ctx["node_caps_by_alpha"][R.ALPHA]
    demands = {t: [float(ctx.demand[(t, tuple(int(v) for v in c))]) for c in g.task_logit_to_placement[t]] for t in range(n)}
    with torch.no_grad():
        sf = make_partial_state_score_fn(model, g, ctx)
        try:
            fsf = force_prefix_score_fn(sf, g.task_logit_to_placement, label, forced)
            combo = decode_masked_topo_placement([torch.empty(0)] * n, g.task_logit_to_placement, n, dag_parents=g.dag_parents, node_caps=ctx.node_caps, demands=demands, score_fn=fsf,
                                                 allow_replica_reuse=True, relax_on_stuck=True)
        finally:
            if hasattr(sf, "close"):
                sf.close()
    return None if combo is None else [[int(v) for v in combo[t]] for t in range(n)]


def cmd_draw(a):
    src = json.load(open(a.src))
    keep = sorted(g for g in src["val"] if src["meta"][g]["label"] in a.classes.split(","))
    out = {**{k: v for k, v in src.items() if k not in ("val", "meta", "source")}, "source": a.src, "train": [], "val": keep, "meta": {g: src["meta"][g] for g in keep}}
    json.dump(out, open(a.out, "w"), indent=1)
    from collections import Counter
    print(dict(Counter(f"{src['meta'][g]['bucket']}/{src['meta'][g]['label']}" for g in keep)))


def cmd_points(a):
    import uncut_replay_diag as R
    from src.policy.gnn.seq_decode import topological_task_order
    from uncut_w3_prefix import prefix_len

    draw = json.load(open(a.draw))
    models = R._load_models(list(RUNS))
    out: Dict[str, Any] = {}
    bad = {"identity_t0": [], "label_all": [], "prefix_not_kept": []}
    for f in sorted(Path(a.shards).glob("g_*.pkl")):
        for g in pickle.load(open(f, "rb")):
            gid, n = str(g.dataset_id), int(g.n_tasks)
            order = [int(x) for x in topological_task_order(n, g.dag_parents)]
            label = [[int(v) for v in g.task_logit_to_placement[t][int(g.tied_optimal_logit_plans[R.ALPHA][0][t])]] for t in range(n)]
            runs = {}
            for run, m in models.items():
                s5 = R.decode_group(m, g)["decoded"]
                free = forced_decode(m, g, label, set())
                if free != s5:
                    bad["identity_t0"].append((gid, run))
                allf = forced_decode(m, g, label, set(order))
                if allf != label:
                    bad["label_all"].append((gid, run))
                by, pts = {}, {}
                for fr in FRACS:
                    pre = order[:prefix_len(n, fr)]
                    plan = forced_decode(m, g, label, set(pre))
                    if plan is not None and any(plan[t] != label[t] for t in pre):
                        bad["prefix_not_kept"].append((gid, run, fr))
                        plan = None
                    by[str(fr)], pts[str(fr)] = plan, pre
                runs[run] = {"decoded": s5, "decoded_by_frac": by, "points": pts}
            out[gid] = {"n": n, "bucket": draw["meta"][gid]["bucket"], "label_type": draw["meta"][gid]["label"], "order": order, "label": label, "runs": runs}
    miss = sorted(set(draw["val"]) - set(out))
    if miss:
        raise SystemExit(f"FAIL LOUD: {len(miss)} drawn groups missing from the shards, e.g. {miss[:3]}")
    json.dump(out, open(a.out, "w"))
    json.dump(bad, open(str(a.out) + ".checks.json", "w"), indent=1)
    print(f"points: {len(out)} groups; checks failed: identity(no forcing == S5 decode) {len(bad['identity_t0'])}, label(all forced == label plan) {len(bad['label_all'])}, "
          f"prefix not kept (those points get no run) {len(bad['prefix_not_kept'])}; decode None {sum(1 for v in out.values() for r in v['runs'].values() for x in r['decoded_by_frac'].values() if x is None)}")
    if bad["identity_t0"]:
        raise SystemExit(f"FAIL LOUD: the unforced wrapper differs from S5's decode on {bad['identity_t0'][:5]}")


def cmd_analyze(a):
    from uncut_w3_prefix import BUCKETS as _B, _boot, gains_of, summarize

    rows: Dict[str, Any] = {}
    for f in sorted(Path(a.rows_dir).glob("i_*.json")):
        rows.update(json.load(open(f)))
    w3 = json.load(open(a.w3_rows))
    ctrl = {(p["group"], p["run"], p["frac"]): p for p in gains_of(rows)}
    selfp = {(p["group"], p["run"], p["frac"]): p for p in gains_of({g: w3[g] for g in rows})}
    id_err = [abs(r["rtt"]["label"] - r["stored_label_rtt"]) / max(1.0, abs(r["stored_label_rtt"])) for r in rows.values() if r["label_in_stored_rows"]]
    keys = sorted(set(ctrl) & set(selfp))
    res: Dict[str, Any] = {"groups": len(rows), "control_points": len(ctrl), "paired_points": len(keys), "unpaired_control_points": len(set(ctrl) - set(selfp)),
                           "decode_failed_runs": sum(1 for r in rows.values() for c in r["records"] if c.get("decode_failed")), "capped_runs": sum(1 for r in rows.values() for c in r["records"] if c.get("capped")),
                           "over2x_runs": sum(1 for r in rows.values() for c in r["records"] if c.get("over2x")), "runs": sum(len(r["records"]) for r in rows.values()),
                           "label_replay_identity": {"checked": len(id_err), "max_rel_err": max(id_err) if id_err else None, "fail_1e-9": sum(1 for e in id_err if e > 1e-9)}, "cells": {}, "paired": {}}
    mean = lambda xs: sum(xs) / len(xs)
    for scope in list(BUCKETS) + ["both"]:
        sel = [k for k in keys if scope == "both" or ctrl[k]["bucket"] == scope]
        for run in (None,) + RUNS:
            for fr in (None,) + FRACS:
                ks = [k for k in sel if (run is None or k[1] == run) and (fr is None or k[2] == fr)]
                if not ks:
                    continue
                tag = f"{scope}|{run or 'both_seeds'}|{fr if fr is not None else 'both_points'}"
                res["cells"][tag + "|control"] = summarize([ctrl[k] for k in ks])
                res["cells"][tag + "|w3_model_prefix"] = summarize([selfp[k] for k in ks])
                by_g: Dict[str, List[Tuple[float, float]]] = {}
                for k in ks:
                    by_g.setdefault(k[0], []).append((1.0 if selfp[k]["gain"] >= 0.05 else 0.0, 1.0 if ctrl[k]["gain"] >= 0.05 else 0.0))
                dsh = [[x - y for x, y in v] for v in by_g.values()]
                dmean = [[selfp[k]["gain"] - ctrl[k]["gain"] for k in ks if k[0] == g] for g in by_g]
                res["paired"][tag] = {"n_points": len(ks), "n_groups": len(by_g), "share_ge5_diff": mean([x for v in dsh for x in v]), "share_ge5_diff_ci": _boot(dsh, mean),
                                      "mean_gain_diff": mean([x for v in dmean for x in v]), "mean_gain_diff_ci": _boot(dmean, mean)}
    v: Dict[str, Any] = {}
    for b in BUCKETS:
        c = res["paired"].get(f"{b}|both_seeds|both_points")
        if c:
            v[b] = {"diff": c["share_ge5_diff"], "ci": c["share_ge5_diff_ci"], "exposure_room": c["share_ge5_diff"] >= MARGIN_PP and c["share_ge5_diff_ci"][0] > 0}
    pooled = res["paired"].get("both|both_seeds|both_points")
    res["verdict_by_bucket"] = v
    if any(x["exposure_room"] for x in v.values()):
        res["VERDICT"] = "EXPOSURE-ROOM"
    elif pooled and pooled["share_ge5_diff_ci"][1] < MARGIN_PP:
        res["VERDICT"] = "NO-EXPOSURE-ROOM"
    else:
        res["VERDICT"] = "UNCLEAR"
    res["pooled_paired"] = pooled
    json.dump(res, open(a.out, "w"), indent=1)
    if a.md:
        f = lambda x, ci=None: f"{100 * x:.1f}%" + (f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]" if ci else "")
        L = ["# W3 label-prefix control (offline, MYOPIC-REPLAY, INTERIM-DESCRIPTIVE)", "", f"VERDICT: {res['VERDICT']}  groups {res['groups']}  paired points {res['paired_points']}  runs {res['runs']}  decode-failed {res['decode_failed_runs']}  capped {res['capped_runs']}  OVER2X {res['over2x_runs']}",
             f"label replay identity: {res['label_replay_identity']}", "", "| scope | seed | point | prefix | n pts | share >=5% [CI] | share <=-5% [CI] | median gain [CI] | mean gain [CI] | ILS S-better / replay-worse |", "|---|---|---|---|---|---|---|---|---|---|"]
        for k, c in res["cells"].items():
            sc, run, fr, which = k.split("|")
            L.append(f"| {sc} | {run} | {fr} | {which} | {c['n_points']} | {f(c['share_gain_ge5'], c['share_gain_ge5_ci'])} | {f(c['share_gain_le_m5'], c['share_gain_le_m5_ci'])} | {f(c['median_gain'], c['median_gain_ci'])} | {f(c['mean_gain'], c['mean_gain_ci'])} | {c['ils_s_better']} / {c['ils_s_better_replay_worse']} |")
        L += ["", "| scope | seed | point | n pts | paired share>=5% diff (W3 minus control) [CI] | paired mean-gain diff [CI] |", "|---|---|---|---|---|---|"]
        for k, c in res["paired"].items():
            sc, run, fr = k.split("|")
            L.append(f"| {sc} | {run} | {fr} | {c['n_points']} | {f(c['share_ge5_diff'], c['share_ge5_diff_ci'])} | {f(c['mean_gain_diff'], c['mean_gain_diff_ci'])} |")
        Path(a.md).write_text("\n".join(L) + "\n")
    print(json.dumps({"VERDICT": res["VERDICT"], "verdict_by_bucket": v, "pooled": pooled}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw"); d.add_argument("--src", required=True); d.add_argument("--out", required=True); d.add_argument("--classes", default="ils")
    p = sub.add_parser("points"); p.add_argument("--shards", required=True); p.add_argument("--draw", required=True); p.add_argument("--out", required=True)
    an = sub.add_parser("analyze"); an.add_argument("--rows-dir", required=True); an.add_argument("--w3-rows", required=True); an.add_argument("--out", required=True); an.add_argument("--md", default="")
    a = ap.parse_args(argv)
    {"draw": cmd_draw, "points": cmd_points, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
