#!/usr/bin/env python3
"""scale_d_v1 identity check (v3 item 3): a re-run cell must reproduce its filed summary. Pass = every .npy sidecar byte-equal and every summary field equal outside
provenance and timing (the same rule as the parent's identity cells).
usage: scale_d_ident_cmp.py <new .summary.json> <filed .summary.json>
"""
import filecmp
import glob
import json
import sys

new, old = sys.argv[1], sys.argv[2]
SKIP = {"wallclock_s", "host", "env", "env_fingerprint", "code", "python_env", "decisionTiming", "total_rtt_plus_inference", "cell_started", "provenance", "run_provenance"}
a, b = json.load(open(new)), json.load(open(old))
diff = [k for k in sorted(set(a) | set(b)) if k not in SKIP and a.get(k) != b.get(k)]
nb, ob = new[: -len(".summary.json")], old[: -len(".summary.json")]
npy = []
for f in sorted(glob.glob(ob + ".*.npy") + glob.glob(ob + "_*.npy") + glob.glob(ob + ".npy")):
    g = nb + f[len(ob):]
    npy.append((f.split("/")[-1], filecmp.cmp(f, g, shallow=False)))
print(json.dumps({"new": new, "filed": old, "fields_differing": diff, "npy": npy,
                  "avg_elapsed": [a.get("averageElapsedTime"), b.get("averageElapsedTime")], "total_rtt": [a.get("total_rtt"), b.get("total_rtt")],
                  "pin": [a.get("code"), b.get("code")] if False else None}, indent=1))
ok = not diff and npy and all(x for _, x in npy)
print("IDENTITY", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
