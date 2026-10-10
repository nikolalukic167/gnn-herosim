"""cost_to_go_v1 rung (b): the post-plan features of EVERY plan of a state, from ONE policy rollout.

The batch decision's state does not depend on the plan forced afterwards, so one cd_exactS policy rollout per state reaches the
same decision every forced-plan rollout of that state reached. At that decision this hook evaluates S7's `post_commit_features`
(cost_to_go_handv_features) for the policy plan and for each given plan of the job (`job["plans"]`: {slot: [[node_id,
platform_id], ...] in dataset task order}), plus S's own totals. Batch tasks outside the dataset (queued tasks forced at the same
decision) keep the decision's couple in every plan, as in the forced-plan rollouts.

  job: {..., "decision_hook": "scripts_cosim.cost_to_go_mlpv_features:decision_hook", "plans": {slot: plan}}
  row["hook"] = {"features": [feature row per slot, the cost_to_go_handv_features schema plus s_total/s_service/slot]}
"""
from __future__ import annotations

import argparse
import json


def decision_hook(sched, batch_tasks, plan, system_state, ctx):
    from scripts_cosim.cost_to_go_handv_features import post_commit_features  # simulator imports: only inside a replay

    job = ctx["job"]
    local_of = ctx["local_of"]
    couple = {(int(n.id), int(p.id)): (n, p) for n in sched.nodes.items for p in n.platforms.items}
    plans = {"policy": list(plan)}
    for slot, pl in (job.get("plans") or {}).items():
        objs = []
        for i, t in enumerate(batch_tasks):
            d = local_of.get(int(t.id))
            objs.append(plan[i] if d is None else couple[(int(pl[d][0]), int(pl[d][1]))])
        plans[slot] = objs
    orch = sched._pg_orchestrator()
    type_defs = orch.data.task_types
    stable = float(sched.autoscaler.kpa.config.stable_window)
    out = []
    for slot, objs in plans.items():
        feat = post_commit_features(sched, batch_tasks, objs, system_state, float(sched.env.now), type_defs, stable,
                                    ds=job["ds"], slot=slot, t0=float(ctx["t0"]), workload=job["workload"])
        total, service, scores = sched._pg_plan_cost(batch_tasks, objs, orch, {}, sched.nodes.items)
        feat["s_total"], feat["s_service"], feat["s_scores"] = float(total), [float(x) for x in service], [float(x) for x in scores]
        feat["pg_inflight"] = bool(sched.pg_inflight)
        feat["plan"] = [[int(objs[i][0].id), int(objs[i][1].id)] for i in range(len(objs))]
        feat["local_batch"] = [local_of.get(int(t.id)) for t in batch_tasks]
        out.append(feat)
    return {"features": out}


def jobs(a):
    """One H=5 policy job per state of the rollout job list, carrying every other slot's plan."""
    by = {}
    for l in open(a.jobs):
        j = json.loads(l)
        if float(j["H"]) != 5.0 or float(j.get("eps") or 0.0) != 0.0:
            continue
        by.setdefault(j["ds"], {})[j["tag"].split("|", 1)[1]] = j
    n = 0
    with open(a.out, "w") as fh:
        for ds, slots in sorted(by.items()):
            pol = dict(slots["policy"])
            pol["plans"] = {s: j["plan"] for s, j in slots.items() if s != "policy" and isinstance(j["plan"], list)}
            pol["decision_hook"] = "scripts_cosim.cost_to_go_mlpv_features:decision_hook"
            pol["tag"] = "feat|policy"
            fh.write(json.dumps(pol) + "\n")
            n += 1
    print(f"{n} feature jobs (one per state) -> {a.out}")


def extract(a):
    """Feature rows per (ds, slot) from the hook rows; a state whose policy rollout does not reproduce its rollout row's Q (bit for
    bit) or whose plans differ from the rows' plans is excluded and counted."""
    ref = {}
    for l in open(a.rows):
        r = json.loads(l)
        if "error" not in r and float(r["H"]) == 5.0:
            ref[(r["ds"], r["tag"].split("|", 1)[1])] = r
    kept = bad_q = bad_plan = err = 0
    with open(a.out, "w") as fh:
        for l in open(a.hook_rows):
            r = json.loads(l)
            if "error" in r or not r.get("hook"):
                err += 1
                continue
            pol = ref.get((r["ds"], "policy"))
            if pol is None or r["q"] != pol["q"] or r["plan"] != pol["plan"]:
                bad_q += 1
                continue
            feats = r["hook"]["features"]
            ok = True
            for f in feats:
                row = ref.get((r["ds"], f["slot"]))
                if row is None:
                    ok = False
                    break
                ds_plan = [None] * len(row["plan"])
                for i, d in enumerate(f["local_batch"]):
                    if d is not None:
                        ds_plan[d] = f["plan"][i]
                if ds_plan != row["plan"]:
                    ok = False
                    break
                f["q"] = row["q"]
            if not ok:
                bad_plan += 1
                continue
            for f in feats:
                fh.write(json.dumps(f) + "\n")
            kept += 1
    print(f"states kept {kept}; excluded: policy Q or plan not bit-identical to its row {bad_q}, a slot's plan differs {bad_plan}, "
          f"errors / no hook {err}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("jobs"); j.add_argument("jobs"); j.add_argument("out")
    e = sub.add_parser("extract"); e.add_argument("hook_rows"); e.add_argument("rows"); e.add_argument("out")
    a = ap.parse_args()
    {"jobs": jobs, "extract": extract}[a.cmd](a)


if __name__ == "__main__":
    main()
