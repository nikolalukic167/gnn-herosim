#!/usr/bin/env python3
"""S4. Identity of the three request-level re-runs (bar, FL gnn s1, FL MLP s1; 19538 g2 heavy; each at its cells' own pin) against the filed cells:
the three .npy sidecars byte-equal and every summary field equal except provenance (code, host, python_env, env_fingerprint), wallclock_s, decisionTiming and
total_rtt_plus_inference. Writes anomaly/req_ident_check.txt once."""
import glob, hashlib, json, os, sys
U = "/share/nikola.lukic/uncut_groups"; A = U + "/anomaly"
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
ALLOWED = ("/code", "/host", "/python_env", "/wallclock_s", "/decisionTiming", "/total_rtt_plus_inference", "/env_fingerprint")
CASES = (("bar", U + "/comb/cells_bar_g8", "a459dd98"), ("gnn", U + "/comb/cells_comb", "a8681537"), ("mlp", U + "/comb/cells_comb", "a8681537"))
def walk(x, y, p, out):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if k not in x: out["new"].append(f"{p}/{k}")
            elif k not in y: out["gone"].append(f"{p}/{k}")
            else: walk(x[k], y[k], f"{p}/{k}", out)
    elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        for i, (u, v) in enumerate(zip(x, y)): walk(u, v, f"{p}[{i}]", out)
    elif x != y: out["diff"].append(p)
L = []; rc = 0
for k, filed_root, pin in CASES:
    d = f"{A}/req_ident_{k}"; sm = glob.glob(d + "/*.summary.json"); fl = glob.glob(d + "/*.failed.json")
    if len(sm) != 1 or fl: L.append(f"{k}: FAIL summaries {len(sm)} failed {len(fl)}"); rc = 1; continue
    stem = os.path.basename(sm[0]).replace(".summary.json", ""); filed = f"{filed_root}/{stem}"
    f = json.load(open(filed + ".summary.json")); s = json.load(open(sm[0])); out = {"new": [], "gone": [], "diff": []}; walk(f, s, "", out)
    bad = [p for p in out["diff"] if not p.startswith(ALLOWED)]
    npy = {e: md5(f"{filed}.{e}.npy") == md5(f"{d}/{stem}.{e}.npy") for e in ("tasks", "tasks_bsz", "tasks_lock")}
    pin_ok = s["code"]["commit"][:8] == pin and not s["code"]["dirty"] and f["code"]["commit"][:8] == pin
    ok = not (bad or out["new"] or out["gone"]) and all(npy.values()) and pin_ok
    L.append(f"{k}: {stem} identity {'OK' if ok else 'FAIL'} | pin {pin} rerun+filed {pin_ok} | fields differing outside provenance/timing: {bad} | new {out['new']} gone {out['gone']} | npy byte-equal {npy} | allowed diffs {len(out['diff'])}")
    rc |= 0 if ok else 1
L.append("IDENTITY PASS" if not rc else "IDENTITY FAIL")
dst = A + "/req_ident_check.txt"; assert not os.path.exists(dst)
open(dst, "w").write("\n".join(L) + "\n"); print("\n".join(L)); sys.exit(rc)
