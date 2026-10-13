"""uncut_groups_s0_v1 Arm 1 label build (SCIENTIST's corpus spec, 2026-10-13): pool planning, state drawing, label driver.

  plan   : from the minted corpus workloads, choose the capture pool per (size bucket x time stratum) = POOL_MULT x its draw, as (window set, window, rung, group, topology)
           pairs, seeded; write one R1A_SNAPSHOT_PLAN per window set and the pool file. Prints the pool census and stops (exit 3) when any cell has < 3 x its draw.
  openings: write an inputs tree whose workloads keep only the first OPEN_S seconds of arrival (opening-only capture cells: a cell's opening does not depend on what follows).
Buckets are the served batch size: <= 10, 11-50, 51-100, 101-256 (larger batches are counted and not labelled). Time stratum: batch arrival t < 360 s (30 % of every bucket's draw) or later."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

U_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(U_REPO))
BUCKETS = (("le10", 1, 10), ("11_50", 11, 50), ("51_100", 51, 100), ("101_256", 101, 256))
DRAWS = {"le10": 4800, "11_50": 4600, "51_100": 2000, "101_256": 600}       # 12,000 labelled states (SCIENTIST, as corrected)
EARLY_SHARE, EARLY_T, POOL_MULT, MIN_MULT = 0.30, 360.0, 5, 3
WINDOWS, RUNGS = ("g0", "g1", "g2", "g3"), ("moderate", "heavy")


def bucket_of(size: int):
    return next((n for n, lo, hi in BUCKETS if lo <= size <= hi), None)


def groups_of(wl: Dict[str, Any]) -> List[Tuple[int, int, float]]:
    """(trigger task id = the group's first event index, size, first arrival time) per peer group, in arrival order."""
    first: Dict[Any, int] = {}
    size: collections.Counter = collections.Counter()
    for i, e in enumerate(wl["events"]):
        g = e["peer_group"]
        size[g] += 1
        first.setdefault(g, i)
    return [(first[g], size[g], float(wl["events"][first[g]]["timestamp"])) for g in sorted(first, key=lambda g: first[g])]


def cell_draw(bucket: str, early: bool) -> int:
    n_early = round(DRAWS[bucket] * EARLY_SHARE)
    return n_early if early else DRAWS[bucket] - n_early


def plan_pool(wls: Dict[Tuple[str, str, str], Dict[str, Any]], topos: Sequence[int], seed: int, mult: int = POOL_MULT, thin_le10: float = 0.05,
              open_sets: Sequence[str] = ()) -> Dict[str, Any]:
    """wls[(window set, window, rung)] -> workload. Returns the pool, per-cell census and the plans per window set. A set in `open_sets` is run opening-only
    (its cells stop at the end of the opening), so only its groups with t < EARLY_T enter the pool."""
    cells: Dict[Tuple[str, bool], List[Tuple[Tuple[str, str, str], int, int, float]]] = collections.defaultdict(list)
    over256 = collections.Counter()
    for key, wl in sorted(wls.items()):
        for trig, size, t in groups_of(wl):
            b = bucket_of(size)
            if b is None:
                if size > 256:
                    over256[key[2]] += 1
                continue
            if key[0] in open_sets and t >= EARLY_T:
                continue
            cells[(b, t < EARLY_T)].append((key, trig, size, t))
    rng = random.Random(f"arm1_pool:{seed}")
    pool, census = [], []
    T = len(topos)
    for (b, early), items in sorted(cells.items()):
        draw = cell_draw(b, early)
        n_pairs = len(items) * T
        avail = int(n_pairs * thin_le10) if b == "le10" else n_pairs        # <= 10: a seeded 5 % of batches
        take = min(mult * draw, avail)
        idx = rng.sample(range(n_pairs), take)
        for i in sorted(idx):
            key, trig, size, t = items[i // T]
            pool.append({"set": key[0], "window": key[1], "rung": key[2], "topology": int(topos[i % T]), "trigger": trig, "size": size, "t": t, "bucket": b, "early": early})
        census.append({"bucket": b, "early": early, "draw": draw, "available": avail, "pool": take, "ratio_available_to_draw": avail / draw if draw else None, "ok": avail >= MIN_MULT * draw})
    plans: Dict[str, Dict[str, List[int]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in pool:
        plans[p["set"]][f"{p['topology']}__{p['window']}{p['rung']}"].append(p["trigger"])
    plans = {s: {k: sorted(set(v)) for k, v in d.items()} for s, d in plans.items()}
    return {"pool": pool, "census": census, "plans": plans, "groups_over_256": dict(over256)}


GATE_SHARES = {"11_50": 0.39, "51_100": 0.44, "101_256": 0.17}      # the gate windows' task-weighted shares within batches > 10 (node § 4)


def _median(xs: Sequence[float]) -> float:
    ys = sorted(xs)
    n = len(ys)
    return float(ys[n // 2]) if n % 2 else 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def draw_states(rows: Sequence[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    """rows: the captured pool, each with bucket, early (snapshot time < 360 s), cross (cross-node in-batch partner pairs of the served cd_ib0 plan), key (unique id).
    Per bucket x time stratum: the draw (30 / 70 by time); for batches > 10, 2/3 above the stratum's median cross and 1/3 at or below it (ties go to at-or-below), the median taken
    on the capture pool. 101-256 opening fallback (SCIENTIST / expert): when that stratum's pool is < 3 x its draw, draw floor(pool / 3) and move the rest to late.
    Returns the drawn rows (tagged side), the census and the stop flags (the corpus-wide opening floor of 25 % fails loud)."""
    rng = random.Random(f"arm1_draw:{seed}")
    by: Dict[Tuple[str, bool], List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by[(r["bucket"], bool(r["early"]))].append(r)
    plan_draw = {}
    for b in DRAWS:
        d_early = cell_draw(b, True)
        pool_early = len(by[(b, True)])
        fallback = False
        if pool_early < MIN_MULT * d_early:
            if b != "101_256":
                raise SystemExit(f"FAIL LOUD: {b} opening pool {pool_early} < {MIN_MULT} x its draw {d_early}; only 101-256 has the registered fallback")
            d_early, fallback = pool_early // MIN_MULT, True
        plan_draw[(b, True)] = d_early
        plan_draw[(b, False)] = DRAWS[b] - d_early
        plan_draw[(b, "fallback")] = fallback
    drawn, census, problems = [], [], []
    for b in DRAWS:
        for early in (True, False):
            pool = by[(b, early)]
            n = plan_draw[(b, early)]
            if b == "le10":
                sides = {"all": (pool, n)}
            else:
                med = _median([r["cross"] for r in pool]) if pool else 0.0
                above = [r for r in pool if r["cross"] > med]
                below = [r for r in pool if r["cross"] <= med]
                n_above = round(n * 2 / 3)
                sides = {"above": (above, n_above), "below": (below, n - n_above)}
            for side, (cand, k) in sides.items():
                if len(cand) < k:
                    problems.append(f"{b} {'early' if early else 'late'} {side}: pool {len(cand)} < draw {k}")
                take = rng.sample(sorted(cand, key=lambda r: r["key"]), min(k, len(cand)))
                for r in take:
                    drawn.append({**r, "side": side})
                census.append({"bucket": b, "early": early, "side": side, "pool": len(cand), "draw": k, "drawn": len(take), "ratio": len(cand) / k if k else None,
                               "fallback": bool(plan_draw[(b, "fallback")] and early)})
    n_open = sum(1 for r in drawn if r["early"])
    open_share = n_open / len(drawn) if drawn else 0.0
    if open_share < 0.25:
        problems.append(f"corpus-wide opening share {open_share:.3f} < 0.25")
    task_w = {b: sum(r["size"] for r in drawn if r["bucket"] == b) for b in GATE_SHARES}
    tot = sum(task_w.values())
    shares = {b: task_w[b] / tot for b in task_w} if tot else {}
    for b, sh in shares.items():
        if not 0.5 <= sh / GATE_SHARES[b] <= 2.0:
            problems.append(f"task-weighted share of {b} within batches > 10 is {sh:.3f} against the gate's {GATE_SHARES[b]} (outside 2x)")
    return {"drawn": drawn, "census": census, "problems": problems, "opening_share": open_share,
            "opening_share_by_bucket": {b: sum(1 for r in drawn if r["bucket"] == b and r["early"]) / max(1, sum(1 for r in drawn if r["bucket"] == b)) for b in DRAWS},
            "counts": {b: sum(1 for r in drawn if r["bucket"] == b) for b in DRAWS}, "task_weighted_shares_gt10": shares}


def served_cross_pairs(snap: Dict[str, Any], chosen: Dict[str, Any]) -> Tuple[int, int]:
    """(in-batch partner pairs on different nodes under the served plan, tasks the served decision deferred)."""
    node = {int(t): int(n) for t, n, _p in chosen["chosen"]}
    cross = sum(1 for a, b, _pay in snap["fidelity"]["pairs"] if int(a) in node and int(b) in node and node[int(a)] != node[int(b)])
    return cross, len(chosen.get("deferred") or [])


def capture_rows(pool: Sequence[Dict[str, Any]], cap_root: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Join the pool to the captured snapshots (+ chosen sidecars). Returns (rows for draw_states, pool pairs that were not captured)."""
    cells: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in pool:
        cells[f"{r['set']}/cc40s{r['topology']}__{r['window']}{r['rung']}__cd_ib0_s0"].append(r)
    rows, missing = [], []
    for cell, prs in sorted(cells.items()):
        stem = f"{cap_root}/{cell}"
        snapf = stem + ".snapshots.jsonl"
        if not os.path.exists(snapf):
            missing += prs
            continue
        want = {r["trigger"]: r for r in prs}
        chosen = {}
        if os.path.exists(snapf + ".chosen.jsonl"):
            for line in open(snapf + ".chosen.jsonl"):
                if line.strip():
                    c = json.loads(line)
                    chosen[int(c["trigger_task_id"])] = c
        seen = set()
        for line in open(snapf):
            if not line.strip():
                continue
            snap = json.loads(line)
            trig = int(snap["trigger_task_id"])
            if trig not in want or trig in seen or trig not in chosen:
                continue
            seen.add(trig)
            r = want[trig]
            cross, n_def = served_cross_pairs(snap, chosen[trig])
            n_batch = len(snap["fidelity"]["batch"])
            b = bucket_of(n_batch)
            if b is None:
                continue
            rows.append({**r, "key": f"{cell}#{trig}", "snapfile": snapf, "time": float(snap["time"]), "early": float(snap["time"]) < EARLY_T, "n_batch": n_batch, "bucket": b,
                         "size": n_batch, "cross": cross, "n_deferred": n_def, "cell": cell})
        missing += [want[t] for t in want if t not in seen]
    return rows, missing


R_MIN = 0.10             # SCIENTIST: the label is the ILS plan only if (RTT_hand - RTT_ILS) / RTT_hand >= 0.10 at N = 0
EXACT_CAP = 100_000      # <= 10-task states: exact optimum on S over the full candidate product when it has at most this many plans, else the > 10 path
ILS_SECONDS = 600.0


def job_of(state: Dict[str, Any], root: str) -> Dict[str, Any]:
    """The decision-pass job for a drawn state. The cell env comes from the full-length set-A summary of the same cell (opening-only cells write none: the runner's
    N_TASKS check marks them failed); the config from the set's own whole-group inputs."""
    topo, w, rung, s = state["topology"], state["window"], state["rung"], state["set"]
    return {"state_id": state["key"], "bucket": state["bucket"], "group_size": state["n_batch"], "snapfile": state["snapfile"], "trigger_task_id": int(state["trigger"]),
            "time": state["time"], "cell_summary": f"{root}/cap/A/cc40s{topo}__{w}{rung}__cd_ib0_s0.summary.json",
            "cell_config": f"{root}/inputs{s}/inputs_group/wf1_{rung}/cfg/cc40s{topo}.json", "sim_input": str(U_REPO / "data" / "nofs-ids"), "role": "first"}



def lock_wait_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Descriptive: mean lock wait (compute_start - io_end) per task at the W horizon, ILS plan minus hand plan, for batch tasks and later
    arrivals, split by the state's label source. Pooled over states by task counts (sum of sums / sum of tasks)."""
    out: Dict[str, Any] = {}
    for src in sorted({r.get("source") for r in rows if r.get("lock_wait")}):
        sub = [r for r in rows if r.get("source") == src and r.get("lock_wait") and r["lock_wait"].get("ils") and r["lock_wait"].get("hand")]
        cell: Dict[str, Any] = {"states": len(sub)}
        for who in ("batch", "later"):
            acc = {}
            for plan in ("hand", "ils"):
                tot = sum(r["lock_wait"][plan]["W"][who]["sum"] for r in sub)
                n = sum(r["lock_wait"][plan]["W"][who]["n"] for r in sub)
                acc[plan] = {"mean_s": tot / n if n else None, "tasks": n}
            acc["ils_minus_hand_s"] = None if None in (acc["ils"]["mean_s"], acc["hand"]["mean_s"]) else acc["ils"]["mean_s"] - acc["hand"]["mean_s"]
            diffs = [r["lock_wait"]["ils"]["W"][who]["sum"] - r["lock_wait"]["hand"]["W"][who]["sum"] for r in sub]
            acc["states_ils_higher"] = sum(d > 1e-9 for d in diffs); acc["states_ils_lower"] = sum(d < -1e-9 for d in diffs)
            cell[who] = acc
        out[str(src)] = cell
    return out


def label_state(job: Dict[str, Any], seed: int = 0, evals_dir: Optional[str] = None) -> Dict[str, Any]:
    """The Arm 1 label of one state. <= 10 tasks: the exact S-optimum when the candidate product has <= EXACT_CAP plans. Otherwise ILS from the hand (cd_ib0) plan at the K1(b)
    budget, one replay each of the hand and ILS plans, label = ILS iff r >= R_MIN else hand. A state whose hand plan cannot be re-decided whole (SPLIT / UNPLACED) takes the served
    plan from the capture's chosen sidecar (hand label); one with deferrals in the served plan is dropped and reported."""
    from scripts_cosim import uncut_ils as U

    base = {"state_id": job["state_id"], "bucket": job["bucket"], "n_batch": job["group_size"], "trigger_task_id": job["trigger_task_id"], "snapfile": job["snapfile"]}
    try:
        if job["group_size"] <= 10:
            d = U.decision_pass(dict(job, exhaustive_cap=EXACT_CAP), "cd_ib0", seed, 5, 200, 30.0)
            base.update(plan_space=d["plan_space"], slate_log10=d["slate_log10"], hand=d["hand"], start_S=d["start_S"])
            if d["exhaustive"]:
                return {**base, "source": "exact", "label": d["exhaustive"]["argmin_batch"], "label_S": d["exhaustive"]["s_opt"], "exact_plans": d["exhaustive"]["plans"],
                        "hand_equals_label": d["hand"] == d["exhaustive"]["argmin_batch"]}
        row = U.run_state(dict(job, keep_plans=evals_dir is not None), "cd_ib0", seed, None, 200, ILS_SECONDS, "final")
    except RuntimeError as exc:
        from src.placement import live_audit

        rec = live_audit.load_chosen(job["snapfile"]).get(int(job["trigger_task_id"]))
        reason = "SPLIT" if "decided whole" in str(exc) else "UNPLACED" if "partner the live run" in str(exc) else "OTHER"
        if reason == "OTHER" or rec is None or rec.get("deferred"):
            return {**base, "source": "dropped", "reason": reason, "error": str(exc)[:300], "n_deferred": None if rec is None else len(rec.get("deferred") or [])}
        snap = U._load_snapshot(job)
        gids = [int(r["gid"]) for r in sorted(snap["fidelity"]["batch"], key=lambda r: int(r["gid"]))]
        served = {int(t): [int(n), int(p)] for t, n, p in rec["chosen"]}
        return {**base, "source": "hand_served", "reason": reason, "label": [served[g] for g in gids]}
    r = (row["rtt_hand"] - row["rtt_ils"]) / row["rtt_hand"] if row["rtt_hand"] > 0 else 0.0
    use_ils = r >= R_MIN
    evals_file = None
    if evals_dir is not None and row.get("evals_full"):       # every ILS-evaluated plan with its S and iteration, snapshot gid order (the header of the dataset's placements.jsonl points here)
        import gzip

        Path(evals_dir).mkdir(parents=True, exist_ok=True)
        evals_file = str(Path(evals_dir) / (hashlib.sha1(str(job["state_id"]).encode()).hexdigest()[:16] + ".jsonl.gz"))
        with gzip.open(evals_file + ".partial", "wt") as fh:
            for ev, sv, it, plan in row["evals_full"]:
                fh.write(json.dumps({"eval": ev, "iter": it, "S": sv, "plan": plan}, separators=(",", ":")) + "\n")
        os.replace(evals_file + ".partial", evals_file)
    return {**base, "source": "ils" if use_ils else "hand", "label": row["best"] if use_ils else row["hand"], "r": r, "rtt_hand": row["rtt_hand"], "rtt_ils": row["rtt_ils"],
            "hand": row["hand"], "ils": row["best"], "start_S": row["start_S"], "best_S": row["best_S"], "n_evals": row["n_evals"], "budget": row["budget"], "stopped": row["stopped"],
            "time_bound": row["time_bound"], "evals_file": evals_file, "iters": row["iters"], "plan_space": row["plan_space"], "slate_log10": row["slate_log10"], "cands_mean": row["cands_mean"], "search_wall": row["search_wall"],
            "chosen_match": row.get("chosen_match")}


def draw_extra(rows: Sequence[Dict[str, Any]], exclude_keys: Sequence[str], seed: int, quotas: Dict[str, int]) -> List[Dict[str, Any]]:
    """A separate seeded draw of > 10 states by bucket (SCIENTIST: convergence 200 = 80 / 60 / 60, seed 20261015; fresh-100 = 50 / 30 / 20, seed 20261014), disjoint from the
    labelled and held-out states (`exclude_keys`) and from any other check's draw."""
    rng = random.Random(f"arm1_extra:{seed}")
    ex = set(exclude_keys)
    out = []
    for b, n in quotas.items():
        pool = sorted((r for r in rows if r["bucket"] == b and r["key"] not in ex and r["n_deferred"] == 0), key=lambda r: r["key"])
        if len(pool) < n:
            raise SystemExit(f"FAIL LOUD: extra draw {b}: pool {len(pool)} < {n}")
        out += rng.sample(pool, n)
    return out


def convergence_state(job: Dict[str, Any]) -> Dict[str, Any]:
    """ILS at the K1(b) budget and at twice it, same seed: the S the longer run gains (relative to the longer run's S)."""
    from scripts_cosim import uncut_ils as U

    a = U.run_state(dict(job), "cd_ib0", 0, None, 200, ILS_SECONDS, "none")
    b = U.run_state(dict(job), "cd_ib0", 0, 2 * a["budget"], 200, ILS_SECONDS, "none")
    return {"state_id": job["state_id"], "bucket": job["bucket"], "n_batch": a["n_batch"], "budget_1x": a["budget"], "S_1x": a["best_S"], "S_2x": b["best_S"], "evals_1x": a["n_evals"],
            "evals_2x": b["n_evals"], "stopped_1x": a["stopped"], "stopped_2x": b["stopped"], "gain": (a["best_S"] - b["best_S"]) / b["best_S"] if b["best_S"] else 0.0}


def truncate_workload(wl: Dict[str, Any], open_s: float) -> Dict[str, Any]:
    """The workload cut to events arriving by open_s (ids unchanged: events are in arrival order); pairs touching a dropped event go."""
    keep = sum(1 for e in wl["events"] if float(e["timestamp"]) <= open_s)
    out = {k: v for k, v in wl.items() if k not in ("events", "peer_exchange")}
    out["events"] = wl["events"][:keep]
    out["peer_exchange"] = [p for p in wl.get("peer_exchange", []) if p[0] < keep and p[1] < keep]
    out["rps"] = keep
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--sets", nargs="+", required=True, help="NAME=inputs_group_dir ... (the inputs tree whose wf1_<rung>/wl windows are the corpus windows)")
    p.add_argument("--open-sets", nargs="*", default=[], help="names of the sets whose cells are opening-only")
    p.add_argument("--topos", nargs="+", type=int, required=True); p.add_argument("--seed", type=int, required=True); p.add_argument("--out", required=True)
    d = sub.add_parser("draw")
    d.add_argument("--pool", required=True); d.add_argument("--cap-root", required=True); d.add_argument("--seed", type=int, required=True); d.add_argument("--out", required=True)
    d.add_argument("--exclude-topos", nargs="*", type=int, default=[16226], help="never drawn: 16226 is accel_replica_v1's TEST topology (S6) until the expert rules")
    lb = sub.add_parser("label")
    lb.add_argument("ids"); lb.add_argument("--root", default="/share/nikola.lukic/uncut_arm1"); lb.add_argument("--shard", default="0/1"); lb.add_argument("--out", required=True)
    lb.add_argument("--seed", type=int, default=0); lb.add_argument("--k1-read-commit", required=True, help="the commit that records the K1(b) read (label_gate refuses without it)")
    lb.add_argument("--sealed", default=str(U_REPO / "docs/lineages/uncut_groups_s0_v1/k1_sealed_states.json"), help="the sealed K1(b) states: never labelled")
    ck = sub.add_parser("check")
    ck.add_argument("which", choices=("convergence", "determinism", "fresh100", "lockwait")); ck.add_argument("ids"); ck.add_argument("--root", default="/share/nikola.lukic/uncut_arm1")
    ck.add_argument("--pool"); ck.add_argument("--shard", default="0/1"); ck.add_argument("--out"); ck.add_argument("--labels", nargs="*", default=[])
    o = sub.add_parser("openings")
    o.add_argument("--src", required=True); o.add_argument("--dst", required=True); o.add_argument("--open-s", type=float, default=420.0)
    a = ap.parse_args()
    if a.cmd == "openings":
        if os.path.exists(a.dst):
            raise SystemExit(f"FAIL LOUD: {a.dst} exists")
        shutil.copytree(a.src, a.dst, ignore=shutil.ignore_patterns("wl"))
        for rung in RUNGS:
            os.makedirs(f"{a.dst}/wf1_{rung}/wl")
            for w in WINDOWS:
                f = f"wf1_{rung}/wl/grounded_{w}_n50000.json"
                wl = truncate_workload(json.load(open(f"{a.src}/{f}")), a.open_s)
                json.dump(wl, open(f"{a.dst}/{f}", "w"), separators=(",", ":"))
                print(f"[open] {rung} {w}: {wl['rps']} events kept (arrival <= {a.open_s} s)")
        return 0
    if a.cmd == "label":
        states = json.load(open(a.ids))["states"]
        from scripts_cosim import uncut_ils as U

        gate_jobs = [{**job_of(st_, a.root), "topology": st_["topology"], "window": st_["window"], "rung": st_["rung"]} for st_ in states]
        U.label_gate(gate_jobs, json.load(open(a.sealed)), a.k1_read_commit, False, U_REPO)
        k, m = (int(x) for x in a.shard.split("/"))
        done = set()
        if os.path.exists(a.out):
            done = {json.loads(l)["state_id"] for l in open(a.out) if l.strip()}
        with open(a.out, "a") as fh:
            for st_ in states[k::m]:
                if st_["key"] in done:
                    continue
                row = label_state(job_of(st_, a.root), a.seed, f"{a.root}/evals")
                row["set"], row["topology"], row["window"], row["rung"], row["early"], row["side"] = st_["set"], st_["topology"], st_["window"], st_["rung"], st_["early"], st_["side"]
                fh.write(json.dumps(row, separators=(",", ":")) + "\n"); fh.flush()
                print(row["state_id"], row["source"], flush=True)
        return 0
    if a.cmd == "check" and a.which == "lockwait":
        print(json.dumps(lock_wait_summary([json.loads(l) for l in open(a.ids)]), indent=1))
        return 0
    if a.cmd == "check" and not (a.pool and a.out):
        raise SystemExit("FAIL LOUD: check %s needs --pool and --out" % a.which)
    if a.cmd == "check":
        ids = json.load(open(a.ids))
        drawn = {st_["key"] for st_ in ids["states"]}
        k, m = (int(x) for x in a.shard.split("/"))
        if a.which == "determinism":
            lab = {}
            for f in a.labels:
                for l in open(f):
                    if l.strip():
                        r = json.loads(l)
                        lab[r["state_id"]] = r
            rng = random.Random("arm1_determinism:20261013")
            pick = rng.sample(sorted(k_ for k_, r in lab.items() if r["source"] in ("ils", "hand", "exact")), 100)
            by_key = {st_["key"]: st_ for st_ in ids["states"]}
            with open(a.out, "a") as fh:
                for key in pick[k::m]:
                    again = label_state(job_of(by_key[key], a.root), 0, None)
                    drop = ("search_wall", "evals_file")
                    same = {x: y for x, y in lab[key].items() if x not in drop and x in again} == {x: y for x, y in again.items() if x not in drop}
                    fh.write(json.dumps({"state_id": key, "source": again["source"], "byte_identical_label_row": same, "label_equal": lab[key].get("label") == again.get("label")}) + "\n"); fh.flush()
            return 0
        pool = json.load(open(a.pool))
        rows, _missing = capture_rows([r for r in pool if r["topology"] != 16226 and r["bucket"] != "le10"], a.root + "/cap")
        quotas = {"11_50": 80, "51_100": 60, "101_256": 60} if a.which == "convergence" else {"11_50": 50, "51_100": 30, "101_256": 20}
        seed = 20261015 if a.which == "convergence" else 20261014
        pickd = draw_extra(rows, sorted(drawn), seed, quotas)
        Path(a.out + ".ids.json").write_text(json.dumps({"seed": seed, "quotas": quotas, "states": pickd}, indent=1))
        with open(a.out, "a") as fh:
            for st_ in pickd[k::m]:
                job = job_of(st_, a.root)
                if a.which == "convergence":
                    row = convergence_state(job)
                else:
                    from scripts_cosim import uncut_horizon as H

                    lab = label_state(job, 0, None)
                    row = {"state_id": st_["key"], "bucket": st_["bucket"], "source": lab["source"], "r": lab.get("r")}
                    if lab.get("r") is not None:
                        H.HORIZON_SPECS = ("W",)
                        hz = H.run_state(dict(job), {"hand": lab["hand"], "ils": lab["ils"]})
                        row["horizon"] = {n: {kk: vv["total"] for kk, vv in p["by_n"].items()} for n, p in hz["plans"].items()}
                        row["lock_wait"] = {n: {kk: vv["lock_wait"] for kk, vv in p["by_n"].items()} for n, p in hz["plans"].items() if p}
                fh.write(json.dumps(row, separators=(",", ":")) + "\n"); fh.flush()
                print(row["state_id"], flush=True)
        return 0
    if a.cmd == "draw":
        pool = [r for r in json.load(open(a.pool)) if r["topology"] not in set(a.exclude_topos)]
        rows, missing = capture_rows(pool, a.cap_root)
        print(f"pool {len(pool)} pairs; captured {len(rows)}; not captured {len(missing)}")
        if missing:
            print("  first missing:", [(m["set"], m["topology"], m["window"], m["rung"], m["trigger"]) for m in missing[:5]])
        res = draw_states(rows, a.seed)
        bad = [r["key"] for r in res["drawn"] if r["topology"] in set(a.exclude_topos)]
        if bad:
            raise SystemExit(f"FAIL LOUD: excluded topologies {a.exclude_topos} appear in drawn states {bad[:3]}")
        assert not any(r["topology"] == 16226 for r in res["drawn"]), "16226 (accel_replica_v1 test topology) is never drawn"
        out = Path(a.out)
        text = json.dumps({"seed": a.seed, "states": res["drawn"], "census": res["census"], "counts": res["counts"], "opening_share": res["opening_share"],
                           "opening_share_by_bucket": res["opening_share_by_bucket"], "task_weighted_shares_gt10": res["task_weighted_shares_gt10"], "problems": res["problems"],
                           "n_not_captured": len(missing)}, indent=1)
        out.write_text(text)
        for c in res["census"]:
            print(f"{c['bucket']:8s} {'early' if c['early'] else 'late ':5s} {c['side']:5s} pool {c['pool']:6d} draw {c['draw']:5d} drawn {c['drawn']:5d}{' (fallback)' if c['fallback'] else ''}")
        print("counts", res["counts"], "opening share %.3f" % res["opening_share"], {k: round(v, 3) for k, v in res["opening_share_by_bucket"].items()})
        print("task-weighted shares within > 10:", {k: round(v, 3) for k, v in res["task_weighted_shares_gt10"].items()}, "gate", GATE_SHARES)
        print("states with a deferral in the served plan:", sum(1 for r in res["drawn"] if r["n_deferred"]))
        print("ids file sha256", hashlib.sha256(text.encode()).hexdigest())
        for p_ in res["problems"]:
            print("PROBLEM:", p_)
        return 3 if res["problems"] else 0
    wls = {}
    for spec in a.sets:
        name, d = spec.split("=", 1)
        for rung in RUNGS:
            for w in WINDOWS:
                wls[(name, w, rung)] = json.load(open(f"{d}/wf1_{rung}/wl/grounded_{w}_n50000.json"))
    res = plan_pool(wls, a.topos, a.seed, open_sets=a.open_sets)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for s, d in res["plans"].items():
        (out / f"snapshot_plan_{s}.json").write_text(json.dumps(d, sort_keys=True))
    (out / "pool.json").write_text(json.dumps(res["pool"], separators=(",", ":")))
    (out / "census.json").write_text(json.dumps({"census": res["census"], "groups_over_256": res["groups_over_256"], "seed": a.seed, "topologies": a.topos}, indent=1))
    for c in res["census"]:
        print(f"{c['bucket']:8s} {'early' if c['early'] else 'late ':5s} draw {c['draw']:5d} available {c['available']:9d} pool {c['pool']:6d} {'' if c['ok'] else '  <-- under 3x the draw'}")
    print("groups over 256 (counted, not labelled):", res["groups_over_256"])
    print("pool sha256", hashlib.sha256((out / "pool.json").read_bytes()).hexdigest())
    return 0 if all(c["ok"] for c in res["census"]) else 3


if __name__ == "__main__":
    sys.exit(main())
