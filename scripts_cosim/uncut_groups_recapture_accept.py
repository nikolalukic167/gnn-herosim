"""uncut_groups_s0_v1 K1 recapture acceptance: old vs new summaries and snapshots of the 3 recaptured cells.  usage: OLD_DIR NEW_DIR"""
import json, sys
OLD, NEW = sys.argv[1], sys.argv[2]
CELLS = ["cc40s19463__g1moderate__cd_ib0_k100_s0", "cc40s19465__g1moderate__cd_ib0_k100_s0", "cc40s19463__g3moderate__cd_k100_s0"]
# provenance (code, python_env), the capture book itself, and timing keys: total_rtt_plus_inference adds measured inference WALL seconds to the simulated RTT
# (ruled a timing key by the expert after the recapture was seen, same class as the oracle_terms identity's excluded timing keys); total_rtt is compared.
SKIP = {"run_provenance", "wallclock_s", "sealed_snapshots", "decisionTiming", "env", "code", "python_env", "total_rtt_plus_inference"}
def diff(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if not path and k in SKIP: continue
            if k not in a or k not in b: out.append(f"{path}/{k}: only in {'old' if k in a else 'new'}")
            else: out += diff(a[k], b[k], f"{path}/{k}")
    elif a != b: out.append(f"{path}: {str(a)[:40]} != {str(b)[:40]}")
    return out
ok_all = True
for c in CELLS:
    o, n = json.load(open(f"{OLD}/{c}.summary.json")), json.load(open(f"{NEW}/{c}.summary.json"))
    d = diff(o, n)
    ol = [l for l in open(f"{OLD}/{c}.snapshots.jsonl")]; nl = [l for l in open(f"{NEW}/{c}.snapshots.jsonl")]
    first = {}
    for l in ol: first.setdefault(json.loads(l)["trigger_task_id"], l)
    nt = {}
    for l in nl: nt.setdefault(json.loads(l)["trigger_task_id"], []).append(l)
    byte_eq, mod_id, bad = 0, 0, []
    for t, l in first.items():
        m = nt.get(t, [None])[0]
        if m == l: byte_eq += 1
        elif m and json.loads(m) | {"snapshot_id": 0} == json.loads(l) | {"snapshot_id": 0}: mod_id += 1
        else: bad.append(t)
    sealed = n["sealed_snapshots"]
    ok = not d and not bad and not sealed["missing"] and all(len(v) == 1 for v in nt.values())
    ok_all &= ok
    print(c[5:-3], "summary diffs", d[:6], "| old ids", len(first), "byte-equal", byte_eq, "equal-modulo-snapshot_id", mod_id, "differ", bad,
          "| new sealed", sealed, "new captures per id", {t: len(v) for t, v in nt.items()}, "PASS" if ok else "FAIL")
print("ACCEPTANCE", "PASS (ruled by the expert: total_rtt_plus_inference is a timing key)" if ok_all else "FAIL")
