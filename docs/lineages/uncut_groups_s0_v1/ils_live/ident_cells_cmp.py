#!/usr/bin/env python3
"""identity check: rerun cell vs filed cell. PASS = every summary leaf equal except provenance (run_provenance, code, host, python_env), wallclock_s, decisionTiming and total_rtt_plus_inference (= total_rtt + decisionTiming.total_s); rerun-only keys must be the
additive zero-valued ILS/exact-G/S-charge counters; the three .npy arrays byte-equal."""
import json, sys, hashlib
filed, rerun = sys.argv[1], sys.argv[2]   # path prefixes without extension
# provenance (code, host, python_env, run_provenance), wall clock, and the fields derived from the decision wall (decisionTiming; total_rtt_plus_inference = total_rtt + decisionTiming.total_s)
ALLOW_PREFIX = ("/run_provenance", "/code", "/host", "/python_env", "/wallclock_s", "/decisionTiming", "/total_rtt_plus_inference")
NEW_OK = ("pg_ils_", "pg_exact_g_skipped", "sched_plan_evals_s")
a = json.load(open(filed + ".summary.json")); b = json.load(open(rerun + ".summary.json"))
diff, new, gone = [], [], []
def walk(x, y, p):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if k not in x: new.append((f"{p}/{k}", y[k]))
            elif k not in y: gone.append(f"{p}/{k}")
            else: walk(x[k], y[k], f"{p}/{k}")
    elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        for i, (u, v) in enumerate(zip(x, y)): walk(u, v, f"{p}[{i}]")
    elif x != y: diff.append((p, x, y))
walk(a, b, "")
bad = [d for d in diff if not d[0].startswith(ALLOW_PREFIX)]
badnew = [n for n in new if not (n[0].startswith(ALLOW_PREFIX) or (n[0].split("/")[-1].startswith(NEW_OK) and n[1] in (0, 0.0)))]
badgone = [g for g in gone if not g.startswith(ALLOW_PREFIX)]
md = lambda f: hashlib.md5(open(f, "rb").read()).hexdigest()
arr = {s: (md(f"{filed}.{s}.npy") == md(f"{rerun}.{s}.npy")) for s in ("tasks", "tasks_bsz", "tasks_lock")}
print(f"leaves differing {len(diff)} (allowed-prefix {len(diff)-len(bad)}); rerun-only {len(new)} (unexpected {len(badnew)}); filed-only {len(gone)} (unexpected {len(badgone)}); arrays {arr}")
for d in bad[:20]: print("  DIFF", d[0], str(d[1])[:60], "->", str(d[2])[:60])
for n in badnew[:20]: print("  NEW ", n)
for g in badgone[:20]: print("  GONE", g)
print("PASS" if not bad and not badnew and not badgone and all(arr.values()) else "FAIL")
sys.exit(0 if not bad and not badnew and not badgone and all(arr.values()) else 1)
