"""cost_to_go_v1 hand V: the in-flight term, one definition for replay and live.

`inflight_remaining_seconds` (live_audit) reads `platform.inflight_service_end`, which only the live `_serve_task` sets; a replayed
platform holds ghosts and returns None, so the first feature pass logged 0 in-flight everywhere. The snapshot's own in-flight list
(`fidelity.ghosts`, built by `snapshot_fidelity._ghost_record` from the live tasks' stage markers) is the common source:

  replay / offline: ghost records read from the snapshot
  live:             the same records built by `_ghost_record` from `platform.admitted` + `platform.inflight` (needs the fidelity
                    stage markers, HEROSIM_SNAPSHOT_FIDELITY=1; a task without them raises)

`ghost_remaining_seconds(rec)` is the only place a record becomes seconds, so the two paths cannot read differently.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

VARIANTS = ("b1", "b2")


def ghost_remaining_seconds(rec):
    """Seconds until this task's service ends on its platform, from one ghost record. Transfers it still owes (ingress, input I/O),
    a cold start still running, then execution (or its remainder) and output. A rendezvous wait and an ingress cold start are not in
    the record and count 0."""
    stage = rec["stage"]
    exe, out = float(rec["exec"]), float(rec["output"])
    if stage == "compute":
        return float(rec["compute_remaining"]) + out
    if stage == "lock_wait":
        return exe + out
    if stage == "ingress":
        link = float(rec.get("net_remaining", 0.0)) + float(rec.get("hold_remaining", 0.0))
        return link + float(rec.get("io_remaining", 0.0)) + exe + out
    if stage == "cold":
        return float(rec["cold_remaining"]) + float(rec.get("io_remaining", 0.0)) + exe + out
    if stage in ("input_io", "rendezvous"):
        return float(rec.get("io_remaining", 0.0)) + exe + out
    raise ValueError(f"FAIL LOUD: unknown ghost stage {stage!r} (task {rec.get('tid')})")


def platform_inflight(recs, variant):
    """{platform key: seconds}. b1: sum over the platform's in-flight tasks; b2: the largest single task's remaining."""
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r}; expected one of {VARIANTS}")
    per = defaultdict(list)
    for r in recs:
        per[r["q"]].append(ghost_remaining_seconds(r))
    return {q: (sum(v) if variant == "b1" else max(v)) for q, v in per.items()}


def live_ghost_recs(platform, now, fn=None):
    """Ghost records of one live platform, the way the snapshot capture builds them."""
    from src.placement.snapshot_fidelity import _ghost_record

    rv = getattr(platform, "rendezvous_procs", None) or {}
    recs = []
    if platform.admitted is not None:
        recs.append(_ghost_record(platform, platform.admitted, "admitted", now, fn, False))
    for t in list(platform.inflight):
        recs.append(_ghost_record(platform, t, "inflight", now, fn, t in rv))
    return recs


def live_inflight_remaining(platform, now, variant):
    """The in-flight term of one live platform; equals `platform_inflight` over the snapshot ghosts taken at the same `now`."""
    recs = live_ghost_recs(platform, now)
    return platform_inflight(recs, variant).get(recs[0]["q"], 0.0) if recs else 0.0


def _platform_id(q):
    return q.rsplit(":", 1)[1]


def add_inflight(feat, ghosts, queued_gids=()):
    """feat with `inflight_b1/b2` and `load_after_b1/b2` added. Feature keys are 'nodeId:platformId', ghost keys 'nodeName:platformId';
    platform ids are unique across the topology, so the match is on the platform id and checked to be one-to-one."""
    suffix = defaultdict(list)
    for typ in feat["type_platforms"].values():
        for k in typ:
            if k not in suffix[_platform_id(k)]:
                suffix[_platform_id(k)].append(k)
    out = dict(feat)
    for v in VARIANTS:
        fly = {k: 0.0 for k in feat["load_after"]}
        for q, sec in platform_inflight(ghosts, v).items():
            keys = suffix.get(_platform_id(q), [])
            if len(keys) != 1:
                raise ValueError(f"FAIL LOUD: ghost platform {q} matches feature keys {keys}")
            fly[keys[0]] = sec
        out[f"inflight_{v}"] = fly
        out[f"load_after_{v}"] = {k: float(feat["load_after"][k]) + fly[k] for k in feat["load_after"]}
    out["ghost_queue_overlap"] = len({int(g["tid"]) for g in ghosts} & {int(x) for x in queued_gids})
    return out


def assign_tasks(plan_keys, service, own):
    """Which platform each task's service ran on. The feature rows log S's per-task service in the hook's batch order but not the
    plan's platform per task, and the hook order is not the S0 row's dataset order. Recover it as the assignment of tasks to the plan's
    platform slots whose per-platform sums equal `own` (committed service per platform, load_after - drain); raises unless it is unique
    up to tasks with equal (service, platform). -> [(service, platform key)]."""
    from collections import Counter

    n = len(service)
    slots = Counter(plan_keys)
    sols = set()
    got = defaultdict(float)
    asg = []

    def bt(i):
        if len(sols) > 1:
            return
        if i == n:
            if all(abs(got.get(q, 0.0) - own.get(q, 0.0)) < 1e-6 for q in set(got) | set(own)):
                sols.add(tuple(sorted((round(sv, 9), q) for sv, q in asg)))
            return
        for q in list(slots):
            if slots[q] == 0 or got.get(q, 0.0) + service[i] > own.get(q, 0.0) + 1e-6:
                continue
            slots[q] -= 1
            got[q] += service[i]
            asg.append((service[i], q))
            bt(i + 1)
            asg.pop()
            got[q] -= service[i]
            slots[q] += 1

    bt(0)
    if len(sols) != 1:
        raise ValueError(f"FAIL LOUD: task->platform assignment has {len(sols)} consistent solutions")
    return [(sv, q) for sv, q in next(iter(sols))]


def backlog_forms(feat, plan_keys):
    """Backlog-seconds forms of one plan, registered 2026-10-10 (cost_to_go_v1), B[p] = drain + in-flight b1:
    v1 = sum over platforms the plan touches of (B[p] + this plan's committed service on p)
    v2 = sum over the plan's tasks of own service x (B[p] + the other committed work on its platform)."""
    own = {k: float(feat["load_after"][k]) - float(feat["drain"][k]) for k in feat["load_after"]}
    asg = assign_tasks(plan_keys, [float(x) for x in feat["s_service"]], {k: v for k, v in own.items() if v != 0.0})
    base = {k: float(feat["drain"][k]) + float(feat["inflight_b1"][k]) for k in feat["load_after"]}
    touched = {q for _sv, q in asg}
    v1 = sum(base[q] + own[q] for q in touched)
    v2 = sum(sv * (base[q] + own[q] - sv) for sv, q in asg)
    return {"backlog_v1": v1, "backlog_v2": v2}


def forms(a):
    plans = {}
    for l in open(a.s0):
        r = json.loads(l)
        if "error" not in r and float(r["H"]) == 5.0:
            plans[(r["ds"], r["tag"].split("|", 1)[1])] = [f"{n}:{p}" for n, p in r["plan"]]
    n = 0
    with open(a.out, "w") as fo:
        for l in open(a.features):
            f = json.loads(l)
            f.update(backlog_forms(f, plans[(f["ds"], f["slot"])]))
            fo.write(json.dumps(f) + "\n")
            n += 1
    print(f"{n} feature rows with backlog_v1 / backlog_v2 written to {a.out}")


def recompute(a):
    cache = {}
    n = 0
    stages = defaultdict(int)
    overlap = no_ghost = 0
    with open(a.out, "w") as fo:
        for l in open(a.features):
            f = json.loads(l)
            ds = f["ds"]
            if ds not in cache:
                fid = json.load(open(f"{ds}/infrastructure.json"))["live_snapshot_seed"]["fidelity_replay"]["snapshot"]["fidelity"]
                cache[ds] = (fid.get("ghosts") or [], [int(x["gid"]) for x in fid.get("queued") or []])
            ghosts, queued = cache[ds]
            if f["slot"] == "policy":
                for g in ghosts:
                    stages[g["stage"]] += 1
                no_ghost += not ghosts
            g = add_inflight(f, ghosts, queued)
            if f["slot"] == "policy":
                overlap += g["ghost_queue_overlap"]
            fo.write(json.dumps(g) + "\n")
            n += 1
    print(f"{n} feature rows written to {a.out}")
    print(f"ghost stages over policy-slot states: {dict(stages)}; states with no ghosts: {no_ghost}")
    print(f"ghost tasks also in the snapshot's queued list (double counting with queue drain), policy-slot states: {overlap}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("recompute")
    r.add_argument("--features", required=True)
    r.add_argument("--out", required=True)
    f = sub.add_parser("forms")
    f.add_argument("--s0", required=True)
    f.add_argument("--features", required=True)
    f.add_argument("--out", required=True)
    a = ap.parse_args()
    {"recompute": recompute, "forms": forms}[a.cmd](a)


if __name__ == "__main__":
    main()
