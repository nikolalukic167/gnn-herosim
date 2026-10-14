#!/usr/bin/env python3
"""Arm 1b replay diagnostic (spec: docs/lineages/uncut_groups_s0_v1.md, "Arm 1b replay diagnostic", 0c5a00a1 + the amendment). Diagnostic only; reads no gate state.

  uncut_replay_diag.py prep   --draw DRAW.json --out SHARDS_DIR [--per-shard 25]     one pass over the Arm 1b cache -> graph shards of the drawn groups
  uncut_replay_diag.py decode --shard SHARD.pkl --out OUT.json                       every checkpoint's free masked_topo decode + teacher-forced accuracy per group
  uncut_replay_diag.py replay --shard SHARD.pkl --decode DECODE.json --draw DRAW.json --out OUT.json [--ils]   engine replays (label, hand, models, random[, ILS])
  uncut_replay_diag.py identity --decode-dir D --wandb-root W                         the decode identity against the trainer's logged final val metrics

The decode is the trainer's `_masked_topo_regret_for_graph` (train_near_rtt.py:430) and `_plan_agreement_with_label` (:517) restated, because the trainer trains at import;
the decode identity (all 1108 val groups reproduce each run's logged final val/mt_plan_exact and val/mt_task_acc_choice) is the check that the restatement is the trainer's.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CACHE = "/share/nikola.lukic/uncut_arm1/cache_run/cache/graphs.pkl"
CORPUS = "/share/nikola.lukic/uncut_arm1/corpus"
MODELS = os.environ.get("REPLAY_DIAG_MODELS", "/home/nikola.lukic/gnn-herosim/models")
ALPHA = "inf"
ENV = {"PARTIAL_STATE_CONTRACT": "partial_state_v5", "PARTIAL_STATE_LOAD_SECONDS": "1", "PARTIAL_STATE_EXCHANGE_SECONDS": "1", "PARTIAL_STATE_PEER_MASS": "1"}
# name -> (checkpoint file stem). `final` = -final.pt (epoch 100, the served one); `bv` = the best-val checkpoint (ep_bv).
RUNS = {f"{arm}_s{s}_{kind}": f"uncut-arm1b-{arm.replace('_', '-')}-seed{s}" + ("-final" if kind == "final" else "")
        for arm in ("gnn_eng", "twin_eng") for s in (1, 2) for kind in ("final", "bv")}


def dataset_dir(gid: str) -> str:
    kind, ds = gid.split("/")
    return f"{CORPUS}/{'train' if 'train' in kind else 'heldout'}/{ds}"


def needed_groups(draw: Dict[str, Any]) -> List[str]:
    ids = set(draw["train"]) | set(draw["val"])
    for v in draw.get("ils_topup_train", {}).values():
        ids |= set(v)
    return sorted(ids)


# ---- prep ------------------------------------------------------------------------------------------------------------------------------------

def cmd_prep(a):
    draw = json.load(open(a.draw))
    need = set(needed_groups(draw))
    t0 = time.time()
    graphs = pickle.load(open(CACHE, "rb"))
    keep = [g for g in graphs if str(g.dataset_id) in need]
    got = {str(g.dataset_id) for g in keep}
    if got != need:
        raise SystemExit(f"FAIL LOUD: {len(need - got)} drawn groups are not in the cache, e.g. {sorted(need - got)[:3]}")
    keep.sort(key=lambda g: (int(g.n_tasks), str(g.dataset_id)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # cost-balanced shards: deal the groups out by descending size
    n_sh = max(1, -(-len(keep) // a.per_shard))
    shards: List[List[Any]] = [[] for _ in range(n_sh)]
    load = [0.0] * n_sh
    for g in sorted(keep, key=lambda g: -int(g.n_tasks) ** 2):
        k = min(range(n_sh), key=lambda i: load[i])
        shards[k].append(g)
        load[k] += int(g.n_tasks) ** 2
    man = {}
    for k, sh in enumerate(shards):
        p = out / f"g_{k:03d}.pkl"
        pickle.dump(sh, open(p, "wb"))
        man[p.name] = [str(g.dataset_id) for g in sh]
    json.dump({"cache": CACHE, "n_groups": len(keep), "shards": man}, open(out / "manifest.json", "w"))
    print(f"prep: {len(keep)} groups in {n_sh} shards, {time.time() - t0:.0f} s")


# ---- decode ----------------------------------------------------------------------------------------------------------------------------------

def _load_models(names: Sequence[str]):
    for k, v in {**ENV, "HEROSIM_TRANSFER_MODEL": "pipelined", "HEROSIM_REPLICA_RELEASE": "1", "HEROSIM_SCALEOUT": "kpa", "HEROSIM_WARMTH_PHYSICS": "node_disk_v2",
                                    "GNN_SERVE_CANDIDATE_SLATE": "declared_pruning_v1_uncapped", "GNN_DECODE_MODE": "masked_topo",
                                    "GNN_BATCH_BY_PEER_GROUP": "1"}.items():
        os.environ[k] = v
    from src.placement import replica_rule
    from src.executesimulation import load_gnn_model

    arms = {n.split("_s")[0] for n in names}
    if len(arms) != 1:
        raise SystemExit(f"FAIL LOUD: one process decodes one arm (the twin's GNN_DISABLE_MESSAGE_PASSING is a process-wide flag); got {sorted(arms)}")
    if arms == {"twin_eng"}:
        os.environ["GNN_DISABLE_MESSAGE_PASSING"] = "1"
    else:
        os.environ.pop("GNN_DISABLE_MESSAGE_PASSING", None)
    replica_rule.set_rule("fastest_compatible")      # the Arm 1 cache's physics (checkpoint sidecar physics_env), which load_gnn_model checks

    models = {}
    for n in names:
        m, _device = load_gnn_model(Path(MODELS) / (RUNS[n] + ".pt"))
        m.eval()
        models[n] = m
    return models


def decode_group(model, g) -> Dict[str, Any]:
    """Free masked_topo decode of one graph (the trainer's `_masked_topo_regret_for_graph` + `_plan_agreement_with_label`) and the teacher-forced
    per-task accuracy: at each step, under the LABEL's committed prefix, is the argmax over the step's candidates the label's candidate."""
    import torch
    from src.policy.gnn.partial_state_edges import make_partial_state_score_fn
    from src.policy.gnn.seq_decode import decode_masked_topo_placement, topological_task_order
    from src.policy.tabular.reduced_features import build_partial_state_context_from_graph

    n = int(g.n_tasks)
    ctx = build_partial_state_context_from_graph(g)
    ctx.node_caps = g.partial_state_ctx["node_caps_by_alpha"][ALPHA]
    demands = {t: [float(ctx.demand[(t, tuple(int(v) for v in c))]) for c in g.task_logit_to_placement[t]] for t in range(n)}
    plans = g.tied_optimal_logit_plans[ALPHA]
    label_plans = [[tuple(int(v) for v in g.task_logit_to_placement[t][int(p[t])]) for t in range(n)] for p in plans]
    choice = [t for t in range(n) if len(g.task_logit_to_placement[t]) > 1]
    with torch.no_grad():
        sf = make_partial_state_score_fn(model, g, ctx)
        try:
            combo = decode_masked_topo_placement([torch.empty(0)] * n, g.task_logit_to_placement, n, dag_parents=g.dag_parents, node_caps=ctx.node_caps,
                                                 demands=demands, score_fn=sf, allow_replica_reuse=True, relax_on_stuck=True)
        finally:
            if hasattr(sf, "close"):
                sf.close()
        out: Dict[str, Any] = {"n": n, "n_choice": len(choice), "decoded": None}
        if combo is not None:
            dec = [tuple(int(v) for v in combo[t]) for t in range(n)]
            out["decoded"] = dec
            out["agree"] = max(sum(dec[t] == lp[t] for t in choice) for lp in label_plans)
            out["exact"] = any(dec == lp for lp in label_plans)
        # teacher-forced accuracy under the first label plan's prefix
        sf = make_partial_state_score_fn(model, g, ctx)
        try:
            lp, committed, hit = label_plans[0], {}, 0
            for t in topological_task_order(n, g.dag_parents):
                lg = sf(t, committed)
                if t in choice:
                    cands = [tuple(int(v) for v in c) for c in g.task_logit_to_placement[t]]
                    hit += int(cands[int(torch.argmax(lg))] == lp[t])
                committed[t] = lp[t]
        finally:
            if hasattr(sf, "close"):
                sf.close()
        out["tf_hit"] = hit
    return out


def cmd_decode(a):
    shard = pickle.load(open(a.shard, "rb"))
    names = a.runs.split(",") if a.runs else [n for n in RUNS if n.startswith(a.arm)]
    models = _load_models(names)
    res: Dict[str, Any] = {}
    t0 = time.time()
    for g in shard:
        res[str(g.dataset_id)] = {n: decode_group(m, g) for n, m in models.items()}
    json.dump({"runs": names, "groups": res, "wall": time.time() - t0}, open(a.out, "w"))
    print(f"decode: {len(res)} groups x {len(names)} checkpoints in {time.time() - t0:.0f} s")


# ---- probe: replay order and timing on a few groups (stored-RTT identity only; no model output) -------------------------------------------------------

def _label_rows(ds: str):
    rows = [json.loads(l) for l in open(f"{ds}/placements/placements.jsonl")]
    head, plans = rows[0], [r for r in rows[1:] if r.get("rtt") is not None]
    return head, plans


def cmd_probe(a):
    sys.path.insert(0, str(REPO / "scripts_cosim"))
    import uncut_ils as U

    for gid in a.groups.split(","):
        ds = dataset_dir(gid)
        head, plans = _label_rows(ds)
        trace = [int(x) for x in head["trace_task_ids"]]
        order = sorted(range(len(trace)), key=lambda i: trace[i])        # dataset index of each gid-ascending slot
        t0 = time.time()
        d = U.decision_pass({"dataset": ds, "sim_input": a.sim_input}, "cd_ib0", 0, 1, 1, None, check_arm=False)
        t_dec = time.time() - t0
        n = len(trace)
        for r in plans:
            pl = {int(k): v for k, v in r["placement_plan"].items()}
            variants = {"dataset_order": [list(pl[i]) for i in range(n)], "gid_order": [list(pl[i]) for i in order]}
            for name, plan in variants.items():
                for tname, types in (("types_ds", d["types"]), ("types_gid", [d["types"][i] for i in order])):
                    t1 = time.time()
                    try:
                        rt, setup, runs = U.replay_rtts(d["spec"], types, [plan])
                    except Exception as e:  # noqa: BLE001 - a wrong order is expected to fail for one of the four combinations
                        print(json.dumps({"group": gid, "plan_order": name, "types": tname, "error": str(e)[:160]}))
                        continue
                    print(json.dumps({"group": gid, "n": n, "plan_order": name, "types": tname, "stored_rtt": r["rtt"], "replay_rtt": rt[0], "rel_err": abs(rt[0] - r["rtt"]) / max(1.0, abs(r["rtt"])),
                                      "decision_s": t_dec, "replay_s": time.time() - t1, "hand_len": len(d["hand"]), "hand_eq_label_dsorder": [list(map(int, x)) for x in d["hand"]] == variants["dataset_order"]}))
        print("hand", d["hand"][:3], "types", d["types"][:4], "trace", trace[:4], flush=True)


# ---- replay stages ---------------------------------------------------------------------------------------------------------------------------

SIM_INPUT = "/home/nikola.lukic/gnn-herosim/data/nofs-ids"


def _ds_order(plan_by_gid: Sequence[Sequence[int]], trace: Sequence[int]) -> List[List[int]]:
    """A plan in the engine's batch order (gid ascending) -> dataset order (task i has gid trace[i])."""
    rank = {g: j for j, g in enumerate(sorted(trace))}
    return [[int(x) for x in plan_by_gid[rank[g]]] for g in trace]


def _batch_order(plan_ds: Sequence[Sequence[int]], trace: Sequence[int]) -> List[List[int]]:
    order = sorted(range(len(trace)), key=lambda i: trace[i])
    return [[int(x) for x in plan_ds[i]] for i in order]


def cmd_replay(a):
    sys.path.insert(0, str(REPO / "scripts_cosim"))
    import uncut_ils as U

    draw = json.load(open(a.draw))
    dec = load_decodes(a.decode_dir)
    res: Dict[str, Any] = {}
    t0 = time.time()
    for k in range(a.task, a.n_shards, a.stride):
        for g in pickle.load(open(f"{a.shards}/g_{k:03d}.pkl", "rb")):
            gid = str(g.dataset_id)
            ds = dataset_dir(gid)
            head, rows = _label_rows(ds)
            trace = [int(x) for x in head["trace_task_ids"]]
            n = len(trace)
            label = [[int(v) for v in g.task_logit_to_placement[t][int(g.tied_optimal_logit_plans[ALPHA][0][t])]] for t in range(n)]
            stored = {json.dumps([list(map(int, r["placement_plan"][str(i)])) for i in range(n)]): float(r["rtt"]) for r in rows}
            lkey = json.dumps(label)
            d = U.decision_pass({"dataset": ds, "sim_input": SIM_INPUT}, "cd_ib0", 0, 1, 1, None, check_arm=False)
            hand = _ds_order(d["hand"], trace)
            rng = random.Random(f"uncut-replay-diag:v1:rand:{gid}")
            rand = [[int(v) for v in rng.choice(g.task_logit_to_placement[t])] for t in range(n)]
            plans = {"label": label, "hand": hand, "rand": rand}
            for run in RUNS:
                dc = dec[gid][run]["decoded"]
                if dc is not None:
                    plans[run] = [[int(v) for v in c] for c in dc]
            names = list(plans)
            rt, _setup, _runs = U.replay_rtts(d["spec"], d["types"], [plans[x] for x in names])
            r = dict(zip(names, rt))
            res[gid] = {"n": n, "bucket": draw["meta"][gid]["bucket"] if gid in draw["meta"] else None, "label_type": draw["meta"].get(gid, {}).get("label"),
                        "rtt": r, "plans": plans, "stored_label_rtt": stored.get(lkey), "label_in_stored_rows": lkey in stored, "stored_rows": len(stored),
                        "hand_equals_label": hand == label, "trace": trace, "decode": {run: {k2: dec[gid][run].get(k2) for k2 in ("agree", "exact", "n_choice", "tf_hit")} for run in RUNS}}
    json.dump(res, open(a.out, "w"))
    print(f"replay: {len(res)} groups, {time.time() - t0:.0f} s")


def cmd_ils(a):
    """4 seeded ILS runs from the label plan on the ILS and hand-fallback subsamples; each run's final best replayed. Budget = evaluations (200 x n x mean candidates);
    the 1800 s wall cap is a safety only and a run it stops is CAPPED."""
    sys.path.insert(0, str(REPO / "scripts_cosim"))
    import uncut_ils as U

    draw = json.load(open(a.draw))
    rep: Dict[str, Any] = {}
    for f in sorted(Path(a.replay_dir).glob("r_*.json")):
        rep.update(json.load(open(f)))
    groups = sorted({g for v in draw["ils_subsample"].values() for g in v} | {g for v in draw.get("ils_topup_train", {}).values() for g in v} |
                    {g for v in draw["fallback_subsample"].values() for g in v}, key=lambda g: (-draw["meta"][g]["n_tasks"], g))
    mine = groups[a.task::a.stride]
    orig = U._search_at_decision
    out: Dict[str, Any] = {}
    for gid in mine:
        r0 = rep[gid]
        ds, trace, label = dataset_dir(gid), r0["trace"], r0["plans"]["label"]
        label_b = _batch_order(label, trace)

        def wrapped(sched, batch_tasks, system_state, res, local_of, *args, **kw):
            res2 = dict(res)
            for i, t in enumerate(batch_tasks):
                if int(t.id) in local_of:
                    res2[i] = tuple(label_b[local_of[int(t.id)]])
            return orig(sched, batch_tasks, system_state, res2, local_of, *args, **kw)

        runs, bests = [], []
        U._search_at_decision = wrapped
        try:
            for seed in range(4):
                t1 = time.time()
                d = U.decision_pass({"dataset": ds, "sim_input": SIM_INPUT}, "cd_ib0", seed, None, 200, 1800.0, check_arm=False)
                runs.append({"seed": seed, "n_evals": d["n_evals"], "budget": d["budget"], "stopped": d["stopped"], "capped": d["stopped"] == "seconds",
                             "search_wall": d["search_wall"], "wall": time.time() - t1, "best_S": d["best_S"], "start_S": d["start_S"],
                             "start_is_label": [list(map(int, x)) for x in d["hand"]] == label_b})
                bests.append(_ds_order(d["best"], trace))
        finally:
            U._search_at_decision = orig
        rt, _s, _r = U.replay_rtts(d["spec"], d["types"], [label] + bests)
        out[gid] = {"n": r0["n"], "bucket": draw["meta"][gid]["bucket"], "label_type": draw["meta"][gid]["label"], "rtt_label": rt[0], "rtt_ils": rt[1:], "runs": runs,
                    "best_of_4": min(rt[1:]), "topup_train": any(gid in v for v in draw.get("ils_topup_train", {}).values())}
        json.dump(out, open(a.out, "w"))
    print(f"ils: {len(out)} groups")


# ---- decode identity -------------------------------------------------------------------------------------------------------------------------

WANDB = {"gnn_eng_s1": "run-20261014_100127-kebglex7", "gnn_eng_s2": "run-20261014_100126-kjteh12l",
         "twin_eng_s1": "run-20261014_100123-rgnx535g", "twin_eng_s2": "run-20261014_100123-4ysjvmvz"}


def load_decodes(decode_dir: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for f in sorted(Path(decode_dir).glob("d_*_*.json")):
        for gid, runs in json.load(open(f))["groups"].items():
            out.setdefault(gid, {}).update(runs)
    return out


def cmd_identity(a):
    sys.path.insert(0, str(REPO / "scripts_cosim"))
    import read_training_curves as R

    draw = json.load(open(a.draw))
    dec = load_decodes(a.decode_dir)
    val = draw["val"]
    missing = [g for g in val if g not in dec]
    if missing:
        raise SystemExit(f"FAIL LOUD: {len(missing)} val groups have no decode, e.g. {missing[:3]}")
    bad = 0
    report = {}
    for run in RUNS:
        stem = run.rsplit("_", 1)[0]
        ds = [dec[g][run] for g in val]
        tot = sum(1 for d in ds if d["decoded"] is not None)
        ex = sum(int(d.get("exact", False)) for d in ds)
        ag, nc = sum(d.get("agree", 0) for d in ds), sum(d["n_choice"] for d in ds)
        mine = {"mt_plan_exact": ex / max(1, tot), "mt_task_acc_choice": ag / max(1, nc), "n_decoded": tot}
        rows, _summary, _cfg = R._read_wandb_dir(Path(a.wandb_root) / WANDB[stem])
        vrows = [r for r in rows if "val/mt_plan_exact" in r]
        if run.endswith("final"):
            ref = vrows[-1]
            ep = len(vrows)
        else:
            ep = min(range(len(vrows)), key=lambda i: vrows[i]["val/regret_masked_topo"]) + 1
            ref = vrows[ep - 1]
        logged = {"mt_plan_exact": ref["val/mt_plan_exact"], "mt_task_acc_choice": ref["val/mt_task_acc_choice"]}
        ok = all(abs(mine[k] - logged[k]) <= 1e-9 for k in logged)
        bad += 0 if ok else 1
        report[run] = {"epoch": ep, "mine": mine, "logged": logged, "identical": ok}
        print(f"{run:16s} ep {ep:3d}  plan_exact {mine['mt_plan_exact']:.6f} vs {logged['mt_plan_exact']:.6f}  task_acc_choice {mine['mt_task_acc_choice']:.6f} vs {logged['mt_task_acc_choice']:.6f}  {'IDENTICAL' if ok else 'MISMATCH'}")
    json.dump(report, open(a.out, "w"), indent=1)
    if bad:
        raise SystemExit(f"DECODE IDENTITY FAILED for {bad} of {len(RUNS)} checkpoints: stop and report")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--draw", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per-shard", type=int, default=25)
    d = sub.add_parser("decode")
    d.add_argument("--shard", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--runs", default="")
    d.add_argument("--arm", default="gnn_eng", choices=("gnn_eng", "twin_eng"))
    rp = sub.add_parser("replay")
    rp.add_argument("--shards", required=True)
    rp.add_argument("--n-shards", type=int, required=True)
    rp.add_argument("--task", type=int, required=True)
    rp.add_argument("--stride", type=int, default=8)
    rp.add_argument("--decode-dir", required=True)
    rp.add_argument("--draw", required=True)
    rp.add_argument("--out", required=True)
    il = sub.add_parser("ils")
    il.add_argument("--replay-dir", required=True)
    il.add_argument("--task", type=int, required=True)
    il.add_argument("--stride", type=int, default=8)
    il.add_argument("--draw", required=True)
    il.add_argument("--out", required=True)
    ide = sub.add_parser("identity")
    ide.add_argument("--decode-dir", required=True)
    ide.add_argument("--wandb-root", required=True)
    ide.add_argument("--draw", required=True)
    ide.add_argument("--out", required=True)
    pr = sub.add_parser("probe")
    pr.add_argument("--groups", required=True)
    pr.add_argument("--sim-input", default="/home/nikola.lukic/gnn-herosim/data/nofs-ids")
    a = ap.parse_args(argv)
    return {"prep": cmd_prep, "decode": cmd_decode, "probe": cmd_probe, "identity": cmd_identity, "replay": cmd_replay, "ils": cmd_ils}[a.cmd](a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
