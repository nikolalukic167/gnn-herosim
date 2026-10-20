#!/usr/bin/env python3
"""Field-by-field diff of two cell summaries (recursive). Prints every differing leaf path; exit 0 always (the caller judges).
Paths that carry run provenance or wall-clock measurements are listed separately as VOLATILE."""
import json, sys
VOL = ("wallclock_s", "code", "python_env", "host", "env_fingerprint", "decisionTiming", "total_rtt_plus_inference")
def walk(a, b, p, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a: out.append((p + [k], "<absent>", b[k]))
            elif k not in b: out.append((p + [k], a[k], "<absent>"))
            else: walk(a[k], b[k], p + [k], out)
    elif a != b:
        out.append((p, a, b))
a, b = (json.load(open(x)) for x in sys.argv[1:3])
out = []; walk(a, b, [], out)
vol = [d for d in out if d[0][0] in VOL]; real = [d for d in out if d[0][0] not in VOL]
print(f"leaves differing: {len(out)}; volatile (provenance / wall clock): {len(vol)}; other: {len(real)}")
for p, x, y in real[:40]: print("  DIFF", ".".join(map(str, p)), str(x)[:80], "->", str(y)[:80])
for p, x, y in vol[:40]: print("  vol ", ".".join(map(str, p)), str(x)[:60], "->", str(y)[:60])
