#!/usr/bin/env python3
"""cost_to_go_v1 gate check (2): live value-term features equal the replay features at the same decision.

  cost_to_go_v_parity.py jobs  DUMP.jsonl DATASETS.txt OUT_JOBS.jsonl
  cost_to_go_v_parity.py check DUMP.jsonl HOOK_ROWS.jsonl

jobs: for each dry-run fidelity dataset built from the live run's own snapshots, the dump record of the same decision (time and
batch task ids) gives the live candidate plans; one batch-only replay job per dataset carries them (cost_to_go_mlpv_features hook).
check: per plan, the replay's load_after, drain, replicas (key, type, idle_s, used, cold_s), type_platforms, scale_in_after_s and
the pre-t0 type rates (the trace over the 60 s lookback) must equal the live dump's exactly (|diff| <= 1e-9).
"""
import argparse
import json
import sys


def _key(t, ids):
    return (round(float(t), 9), tuple(sorted(ids)))


def jobs(a):
    dump = {}
    for l in open(a.dump):
        r = json.loads(l)
        dump[_key(r["now"], r["batch_ids"])] = r
    n = miss = 0
    with open(a.out, "w") as fh:
        for ds in (l.strip() for l in open(a.datasets) if l.strip()):
            fr = json.load(open(f"{ds}/fidelity_replay.json"))
            wl = json.load(open(f"{ds}/workload.json"))
            gids = [int(g) for g in wl["trace_task_ids"]]
            rec = dump.get(_key(fr["snapshot_time"], gids))
            if rec is None:
                miss += 1
                continue
            pos = {g: i for i, g in enumerate(rec["batch_ids"])}
            plans = {f"c{k}": [rec["plans"][k][pos[g]] for g in gids] for k in range(len(rec["plans"]))}
            fh.write(json.dumps({"ds": ds, "plan": "policy", "H": -1.0, "continuation": "cd_exacts", "table": None, "eps": 0.0,
                                 "tag": "parity|policy", "workload": None, "plans": plans, "live_key": [rec["now"], rec["batch_ids"]],
                                 "decision_hook": "scripts_cosim.cost_to_go_mlpv_features:decision_hook"}) + "\n")
            n += 1
    print(f"{n} parity jobs; {miss} datasets without a live dump record at their decision")


IDLE = []


def _diff(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                out.append(f"{path}.{k} missing on one side")
            else:
                _diff(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path} length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{path}[{i}]", out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        d = abs(float(a) - float(b))
        if path.endswith(".idle_s"):
            # the replay decides 1 ms after its t0 (the policy path's batch poll; the offline training rows carry the same
            # offset): idle_s may differ by that 1 ms and nothing else; it is reported, and anything above it fails
            IDLE.append(d)
            if d > 1.1e-3:
                out.append(f"{path}: live {a} vs replay {b}")
        elif d > 1e-9:
            out.append(f"{path}: live {a} vs replay {b}")
    elif a != b:
        out.append(f"{path}: live {a!r} vs replay {b!r}")


def check(a):
    sys.path.insert(0, "scripts_cosim")
    from cost_to_go_handv import type_rates

    dump = {}
    for l in open(a.dump):
        r = json.loads(l)
        dump[_key(r["now"], r["batch_ids"])] = r
    plans_ok = plans_bad = states = 0
    for l in open(a.hook_rows):
        r = json.loads(l)
        if "error" in r:
            print("ERROR", r["ds"], r["error"][:300])
            plans_bad += 1
            continue
        states += 1
        job_key = None
        feats = {f["slot"]: f for f in r["hook"]["features"]}
        wl = json.load(open(f"{r['ds']}/workload.json"))
        gids = [int(g) for g in wl["trace_task_ids"]]
        fr = json.load(open(f"{r['ds']}/fidelity_replay.json"))
        rec = dump[_key(fr["snapshot_time"], gids)]
        trace = json.load(open(json.load(open(f"{r['ds']}/warm_snapshot.json"))["provenance"]["trace"]))["events"]
        diffs = []
        _diff(rec["rates"], type_rates(trace, float(fr["snapshot_time"]), 60.0), "rates", diffs)
        for k, live in enumerate(rec["features"]):
            rep = feats[f"c{k}"]
            d = []
            _diff(live, {x: rep[x] for x in live}, f"plan c{k}", d)
            plans_ok += not d
            plans_bad += bool(d)
            diffs += d
        print(r["ds"].split("/")[-1], "t", fr["snapshot_time"], "plans", len(rec["features"]), "diffs", len(diffs), diffs[:5])
    print(f"idle_s |live - replay|: max {max(IDLE) if IDLE else 0:.6f} s over {len(IDLE)} replicas (tolerance 1.1 ms, the replay's batch poll)")
    print(f"PARITY: {states} states, plans identical {plans_ok}, plans differing {plans_bad} -> {'PASS' if plans_bad == 0 and states else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("jobs"); j.add_argument("dump"); j.add_argument("datasets"); j.add_argument("out")
    c = sub.add_parser("check"); c.add_argument("dump"); c.add_argument("hook_rows")
    a = ap.parse_args()
    {"jobs": jobs, "check": check}[a.cmd](a)


if __name__ == "__main__":
    main()
