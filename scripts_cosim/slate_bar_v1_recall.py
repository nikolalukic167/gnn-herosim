#!/usr/bin/env python3
"""slate_bar_v1 recall audit (docs/lineages/slate_bar_v1.md, S6 section): how often each candidate-slate rule holds the couple full CD and
cd_exactS actually pick, at the snapshot's own state. CPU only; the picks come from the rules' decision replay (rollout_replay, batch-only).

  slate_bar_v1_recall.py sample JOBS.jsonl --corpus <accel_replica_corpus> --split <accel_replica_v1_split.json> --states <states/top5>
                                [--per-stratum 60] [--workers 16]
  slate_bar_v1_recall.py run ROWS.jsonl JOBS.jsonl [--workers 16] [--repeat 20]
  slate_bar_v1_recall.py read ROWS.jsonl [--json OUT]

Rules (fixed in the node before any read): top5 (declared), union (top-3 standalone + the 2 cheapest of the rest by standalone + X),
top6, partner (top-3 standalone + the cheapest couple on up to 2 partners' anchor nodes, filled in standalone order).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
import re
import statistics as st
import sys
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts_cosim"))

SEED = 20261012
RULES = ("top5", "union", "top6", "partner")
CONTINUATIONS = ("cd", "cd_exacts")
CAP = 100_000


# ---- slate rules (pure: ranked couples and partner facts in, slates out) ---------------------------------------------------------

def slates_for(ranked, exch, anchors):
    """ranked: the task's couples [(node_id, platform_id, node_name)] in standalone order; exch: {couple: X seconds} (rule 2);
    anchors: partner anchor node names in descending payload order (rule 4). Returns {rule: set of (node_id, platform_id)}."""
    key = lambda c: (int(c[0]), int(c[1]))
    top3 = [key(c) for c in ranked[:3]]
    rest = [c for c in ranked[3:]]
    pos = {key(c): i for i, c in enumerate(ranked)}
    union = list(top3) + [key(c) for c in sorted(rest, key=lambda c: (exch[key(c)], pos[key(c)]))[:2]]
    partner = list(top3)
    for node_name in anchors:
        if len(partner) >= 5:
            break
        hit = next((key(c) for c in ranked if c[2] == node_name and key(c) not in partner), None)
        if hit is not None and len(partner) < 5:
            partner.append(hit)
    for c in ranked:
        if len(partner) >= 5:
            break
        if key(c) not in partner:
            partner.append(key(c))
    return {"top5": {key(c) for c in ranked[:5]}, "union": set(union), "top6": {key(c) for c in ranked[:6]}, "partner": set(partner)}


# ---- the decision hook (runs inside the replay, at the batch decision) ------------------------------------------------------------

def hook(sched, batch_tasks, chosen, system_state, ctx):
    from src.placement import declared_slate
    from src.placement.live_audit import _candidate_payload

    orch = sched._pg_orchestrator()
    memo = {}
    nodes = sched.nodes.items
    queued = ctx["queued"]
    free_ids = {int(t.id) for t in batch_tasks if int(t.id) not in queued}
    ranked_of, valid_of = {}, {}
    for t in batch_tasks:
        valid = sched._get_valid_replicas(system_state.replicas.get(t.type["name"], set()), t)
        r = sorted(valid, key=lambda c: (sched._pg_standalone(t, c[0], c[1], orch, memo, nodes), int(c[0].id), int(c[1].id)))
        ranked_of[int(t.id)] = r
        valid_of[int(t.id)] = valid
    table = getattr(orch, "peer_exchange", None) or {}

    def known_node(pid):
        peer = orch.task_by_id.get(int(pid))
        if peer is None:
            return None
        plat = getattr(peer, "platform", None)
        return plat.node.node_name if plat is not None else getattr(peer, "planned_node_name", None)

    tasks_out = []
    decl_mismatch = 0
    for i, t in enumerate(batch_tasks):
        tid = int(t.id)
        if tid not in free_ids:
            continue
        ranked = ranked_of[tid]
        rc = [(int(n.id), int(p.id), n.node_name) for n, p in ranked]
        decl = declared_slate.prune_candidates([_candidate_payload(sched, t, n, p) for n, p in valid_of[tid]])
        if {(int(c["node_id"]), int(c["platform_id"])) for c in decl} != {(c[0], c[1]) for c in rc[:5]}:
            decl_mismatch += 1
        partners = []  # (payload, anchor node, candidate node set M_j)
        unknown = 0
        for pid, payload in sorted((table.get(tid) or {}).items(), key=lambda kv: (-float(kv[1]), int(kv[0]))):
            if int(pid) in free_ids:
                pr = ranked_of[int(pid)]
                partners.append((float(payload), pr[0][0].node_name, {n.node_name for n, _p in pr[:3]}))
            else:
                name = known_node(pid)
                if name is None:
                    unknown += 1
                    continue
                partners.append((float(payload), name, {name}))
        exch = {}
        for (n, p) in ranked:
            x = 0.0
            for payload, _a, M in partners:
                x += min(sched._pg_exchange_seconds(n, p, [(m, payload)]) for m in M)
            exch[(int(n.id), int(p.id))] = x
        anchors = []
        for _payload, a, _M in partners:
            if a not in anchors:
                anchors.append(a)
        sl = slates_for(rc, exch, anchors)
        pick = (int(chosen[i][0].id), int(chosen[i][1].id))
        ids = [(c[0], c[1]) for c in rc]
        cd_cands = sched._pg_candidates(t, system_state)
        tasks_out.append({
            "tid": tid, "n_valid": len(rc), "n_cd_cands": len(cd_cands), "init_only": len(cd_cands) < len(rc),
            "pick": list(pick), "rank": ids.index(pick) + 1 if pick in ids else 0,
            "in": {r: pick in sl[r] for r in RULES}, "size": {r: len(sl[r]) for r in RULES},
            "partners": len(partners), "partners_unknown": unknown,
        })
    return {"tasks": tasks_out, "n_batch": len(batch_tasks), "n_queued_in_batch": len(batch_tasks) - len(free_ids),
            "decl_mismatch": decl_mismatch}


# ---- sample ------------------------------------------------------------------------------------------------------------------------

_FILE_RE = re.compile(r"^cc40s(\d+)_(moderate|heavy)_g(\d)\.jsonl$")
_TAG_RE = re.compile(r"cc40s(\d+)_(moderate|heavy)_g\d$")


def _scan(path):
    out = []
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        fid = r.get("fidelity") or {}
        if not fid.get("batch"):
            continue
        out.append((int(r["snapshot_id"]), float(r["time"]), len(fid["batch"]), len(fid.get("queued") or [])))
    return path, out


def sample(a):
    split = json.load(open(a.split))
    train = set(split["topologies"]["train"])
    if set(split["topologies"].get("test") or []) & train:
        raise SystemExit("FAIL LOUD: split's test topologies overlap train")
    corpus = Path(a.corpus)
    phys = {}
    for ws in sorted(glob.glob(str(corpus / "gnn_datasets_wf1_train" / "ds_*" / "warm_snapshot.json"))):
        tag = json.load(open(ws))["provenance"]["source_tag"]  # wf1p_cc40s<topo>_<rung>_g<w>
        m = _TAG_RE.search(tag)
        if m is None:
            raise SystemExit(f"FAIL LOUD: unparsed source_tag {tag!r} in {ws}")
        phys.setdefault((m.group(1), m.group(2)), str(Path(ws).parent))
    files = []
    for p in sorted((corpus / "snapshots").iterdir()):
        m = _FILE_RE.match(p.name)
        if m and m.group(1) in train:
            files.append((p, m.group(1), m.group(2)))
    missing = sorted({(t, r) for _p, t, r in files} - set(phys))
    if missing:
        raise SystemExit(f"FAIL LOUD: no train dataset for {missing[:5]}")
    strata = defaultdict(list)
    meta = {str(p): (t, r) for p, t, r in files}
    with get_context("spawn").Pool(a.workers) as pool:
        for path, recs in pool.imap_unordered(_scan, [str(p) for p, _t, _r in files]):
            topo, rung = meta[path]
            for sid, time_, n, q in recs:
                strata[(rung, n)].append(dict(src="accel", rung=rung, n=n, topo=topo, snapfile=path, snapshot_id=sid, time=time_,
                                              queued=q, physics_from=phys[(topo, rung)]))
    jobs, pop = [], {}
    for (rung, n), recs in sorted(strata.items()):
        recs.sort(key=lambda d: (d["snapfile"], d["snapshot_id"], d["time"]))
        pop[f"accel|{rung}|{n}"] = len(recs)
        jobs += random.Random(f"{SEED}:{rung}:{n}").sample(recs, min(a.per_stratum, len(recs)))
    g8 = 0
    for d in sorted(Path(a.states).iterdir()):
        if not d.name.startswith("ds_"):
            continue
        prov = json.load(open(d / "warm_snapshot.json"))["provenance"]
        _topo, rung, g = prov["source_tag"].split("_")
        if g != "G8":
            continue
        n = len(json.load(open(d / "infrastructure.json"))["live_snapshot_seed"]["fidelity_replay"]["snapshot"]["fidelity"]["batch"])
        jobs.append(dict(src="g8", rung=rung, n=n, ds=str(d), time=float(prov["snapshot_time"])))
        g8 += 1
    for k in ("moderate", "heavy"):
        pop[f"g8|{k}|8"] = sum(1 for j in jobs if j["src"] == "g8" and j["rung"] == k)
    with open(a.out, "w") as fh:
        fh.write(json.dumps({"population": pop, "files": len(files), "train_topologies": sorted(train), "seed": SEED,
                             "per_stratum": a.per_stratum}) + "\n")
        for j in jobs:
            fh.write(json.dumps(j) + "\n")
    print(f"sample: {len(files)} snapshot files, {sum(len(v) for v in strata.values())} batches, {len(jobs) - g8} accel + {g8} G8 jobs -> {a.out}")


# ---- run ---------------------------------------------------------------------------------------------------------------------------

def _job(args):
    job, cont, rep = args
    import time

    import rollout_replay

    t0 = time.time()
    rj = dict(plan="policy", H=-1.0, continuation=cont, table=None, eps=0.0, decision_hook="slate_bar_v1_recall:hook")
    if job["src"] == "accel":
        rj.update(ds=job["physics_from"], group=dict(snapfile=job["snapfile"], snapshot_id=job["snapshot_id"], time=job["time"],
                                                     physics_from=job["physics_from"]))
    else:
        rj.update(ds=job["ds"])
    r = rollout_replay._one(rj)
    key = {k: job[k] for k in ("src", "rung", "n") if k in job}
    ident = job.get("ds") if job["src"] == "g8" else f"{job['snapfile']}#{job['snapshot_id']}@{job['time']}"
    base = dict(key, id=ident, cont=cont, repeat=rep, wall=time.time() - t0)
    if "error" in r:
        return dict(base, error=r["error"], tb=r.get("tb", "")[-1200:])
    if not r.get("hook"):
        return dict(base, error="decision hook never ran (the batch decision did not hold every dataset task)")
    return dict(base, plan=r["plan"], hook=r["hook"], exact_batches=r["exact_batches"], exact_fallbacks=r["exact_fallbacks"],
                decisions=r["decisions"])


def run(a):
    lines = [json.loads(l) for l in open(a.jobs) if l.strip()]
    jobs = lines[1:]
    work = [(j, c, False) for j in jobs for c in CONTINUATIONS]
    for src in ("accel", "g8"):
        first = [j for j in jobs if j["src"] == src][:a.repeat]
        work += [(j, c, True) for j in first for c in CONTINUATIONS]
    if a.limit:  # smoke: evenly spaced over the job list, so every source and rung is touched
        step = max(1, len(work) // a.limit)
        work = work[::step][:a.limit]
    print(f"run: {len(work)} replays", flush=True)
    with get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool, open(a.out, "w") as fh:
        fh.write(json.dumps({"header": lines[0]}) + "\n")
        for r in pool.imap_unordered(_job, work):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
    print(f"run: done -> {a.out}")


# ---- read --------------------------------------------------------------------------------------------------------------------------

def _gmean(v):
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else None


def summarise(rows, population):
    """Per (src, rung, n, cont): task recall (all and >5 candidates), batch recall, plan ratio vs top5; pooled per (src, rung, cont)
    with population weights over n."""
    cells = defaultdict(list)
    for r in rows:
        if "error" in r or r.get("repeat"):
            continue
        cells[(r["src"], r["rung"], int(r["n"]), r["cont"])].append(r)
    out = {}
    for (src, rung, n, cont), rs in sorted(cells.items()):
        tasks = [t for r in rs for t in r["hook"]["tasks"]]
        big = [t for t in tasks if t["n_valid"] > 5]
        d = {"batches": len(rs), "tasks": len(tasks), "tasks_gt5": len(big),
             "over_cap": sum(1 for r in rs if math.prod(min(5, t["n_valid"]) for t in r["hook"]["tasks"]) > CAP) / len(rs),
             "exact_fallback": sum(1 for r in rs if r["exact_fallbacks"]) / len(rs) if cont == "cd_exacts" else None,
             "rank_gt5": sum(1 for t in tasks if t["rank"] > 5) / len(tasks) if tasks else None}
        for rule in RULES:
            d[f"task_{rule}"] = sum(t["in"][rule] for t in tasks) / len(tasks) if tasks else None
            d[f"task_gt5_{rule}"] = sum(t["in"][rule] for t in big) / len(big) if big else None
            d[f"batch_{rule}"] = sum(all(t["in"][rule] for t in r["hook"]["tasks"]) for r in rs) / len(rs)
            ratios = [math.prod(t["size"][rule] for t in r["hook"]["tasks"]) / math.prod(t["size"]["top5"] for t in r["hook"]["tasks"])
                      for r in rs if r["hook"]["tasks"]]
            d[f"ratio_{rule}_median"] = st.median(ratios) if ratios else None
            d[f"ratio_{rule}_gmean"] = _gmean(ratios)
        out[f"{src}|{rung}|{n}|{cont}"] = d
    pooled = {}
    for src, rung, cont in sorted({(k.split("|")[0], k.split("|")[1], k.split("|")[3]) for k in out}):
        ks = [k for k in out if k.startswith(f"{src}|{rung}|") and k.endswith(f"|{cont}")]
        w = {k: population.get(f"{src}|{rung}|{k.split('|')[2]}", 0) for k in ks}
        tw = sum(w.values())
        p = {"population": tw, "batches": sum(out[k]["batches"] for k in ks)}
        for rule in RULES:
            for m in ("batch", "task"):
                p[f"{m}_{rule}"] = sum(w[k] * out[k][f"{m}_{rule}"] for k in ks if out[k][f"{m}_{rule}"] is not None) / tw if tw else None
            p[f"ratio_{rule}_gmean"] = math.exp(sum(w[k] * math.log(out[k][f"ratio_{rule}_gmean"]) for k in ks) / tw) if tw else None
        pooled[f"{src}|{rung}|{cont}"] = p
    return out, pooled


def checks(rows):
    errs = [r for r in rows if "error" in r]
    by = defaultdict(list)
    for r in rows:
        if "error" not in r:
            by[(r["id"], r["cont"])].append(r)
    repeat_pairs = [(v[0]["plan"], v[1]["plan"]) for v in by.values() if len(v) == 2]
    return {"rows": len(rows), "errors": len(errs), "error_examples": [e["error"][:200] for e in errs[:5]],
            "repeat_pairs": len(repeat_pairs), "repeat_identical": sum(1 for x, y in repeat_pairs if x == y),
            "decl_mismatch_tasks": sum(r["hook"]["decl_mismatch"] for r in rows if "error" not in r),
            "tasks": sum(len(r["hook"]["tasks"]) for r in rows if "error" not in r),
            "rank0_picks": sum(1 for r in rows if "error" not in r for t in r["hook"]["tasks"] if t["rank"] == 0),
            "multi_decision": sum(1 for r in rows if "error" not in r and r["decisions"] != 1)}


def recommend(pooled):
    """The node's fixed rule: highest pooled G8 cd_exactS batch recall among ratio <= 1 rules (both rungs averaged); top5 stands unless
    the winner beats it by >= 2 pp; top6 only if it beats every ratio <= 1 rule by >= 5 pp."""
    def g8(rule):
        v = [pooled.get(f"g8|{r}|cd_exacts", {}).get(f"batch_{rule}") for r in ("moderate", "heavy")]
        return st.mean(v) if all(x is not None for x in v) else None
    cheap = {r: g8(r) for r in ("top5", "union", "partner")}
    if any(v is None for v in cheap.values()):
        return {"pick": None, "why": "G8 cd_exactS rows missing", "g8": cheap}
    best = max(cheap, key=lambda r: (cheap[r], r == "top5"))
    pick = best if cheap[best] >= cheap["top5"] + 0.02 else "top5"
    t6 = g8("top6")
    if t6 is not None and t6 >= max(cheap.values()) + 0.05:
        pick = "top6"
    return {"pick": pick, "g8_batch_recall_cd_exacts": dict(cheap, top6=t6)}


def read(a):
    lines = [json.loads(l) for l in open(a.rows) if l.strip()]
    population = lines[0]["header"]["population"]
    rows = lines[1:]
    out, pooled = summarise(rows, population)
    res = {"checks": checks(rows), "pooled": pooled, "cells": out, "recommend": recommend(pooled)}
    print(json.dumps(res["checks"]))
    for k, v in pooled.items():
        print("POOLED", k, json.dumps({x: (round(y, 4) if isinstance(y, float) else y) for x, y in v.items()}))
    for k, v in out.items():
        print("CELL", k, json.dumps({x: (round(y, 4) if isinstance(y, float) else y) for x, y in v.items()}))
    print("RECOMMEND", json.dumps(res["recommend"]))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("out")
    s.add_argument("--corpus", required=True)
    s.add_argument("--split", required=True)
    s.add_argument("--states", required=True)
    s.add_argument("--per-stratum", type=int, default=60)
    s.add_argument("--workers", type=int, default=16)
    s.set_defaults(fn=sample)
    r = sub.add_parser("run")
    r.add_argument("out")
    r.add_argument("jobs")
    r.add_argument("--workers", type=int, default=16)
    r.add_argument("--repeat", type=int, default=20)
    r.add_argument("--limit", type=int, default=None)
    r.set_defaults(fn=run)
    d = sub.add_parser("read")
    d.add_argument("rows")
    d.add_argument("--json", default=None)
    d.set_defaults(fn=read)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
