"""Descriptive read of the fixed bar (cd_ib0_inflightfull_sym_k100) on the llf_cd_confirm_v1 fresh 24, with the frozen reader unchanged.

The frozen `llf_confirm_read.py` (blob cbf7fb08) is imported as is; this wrapper only registers a second single-seed bar kind ("sym") with its own
code pin, and adds its checks (bar charge, flag on; flag off on the old bar). Every statistic, runaway rule, Wilcoxon and fail-loud loader check is
the reader's. No label: the output drops `labels` and `gnn_blocked_by_failure`; `primary` (gnn vs sym, Holm over the 2 rungs) is kept only as rows.
usage: sym_bar_read.py --cells <d_cd>,<d_cd_s456>,<d_sym> --sym-pin <prefix> --allow-missing ... --ids ... --ids-sha256 ... --rulings ... --pin static=P,bar=P --out read.json
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("llf_confirm_read", os.path.join(HERE, "llf_confirm_read.py"))
R = importlib.util.module_from_spec(spec)
sys.modules["llf_confirm_read"] = R
spec.loader.exec_module(R)

SYM_KIND = "cd_ib0_inflightfull_sym_k100"
SYM_FLAG = "HEROSIM_PG_EXCHANGE_SYMMETRIC"
R.CD_ARM_REGISTRY["sym"] = SYM_KIND
R.S0_ARMS = R.S0_ARMS + ("sym",)

_orig_load, _orig_cd_check = R.load, R.cd_check
SYM_PIN = None


def load(root, topos, pin, arms=None, **kw):
    roots = [root] if isinstance(root, str) else list(root)
    sym_roots = [r for r in roots if os.path.basename(r.rstrip("/")) == "d_sym"]
    other = [r for r in roots if r not in sym_roots]
    if len(sym_roots) != 1 or "sym" not in arms:
        raise SystemExit("FAIL LOUD: exactly one d_sym root and the sym arm are required")
    V1, p1, g1 = _orig_load(other, topos, pin, {a: k for a, k in arms.items() if a != "sym"}, **kw)
    kw2 = dict(kw, allow_missing=(), flat=None, failed_learned=())
    V2, p2, g2 = _orig_load(sym_roots, topos, f"bar={SYM_PIN}", {"sym": arms["sym"]}, **kw2)
    if set(V1) & set(V2):
        raise SystemExit("FAIL LOUD: a cell key is in both loads")
    return {**V1, **V2}, sorted(set(p1) | set(p2)), {**g1, "sym": g2["bar"]}


def cd_check(key, s, f, arms=None, crung=None):
    _orig_cd_check(key, s, f, arms, crung)
    env = s.get("env") or {}
    rung = key[4] if crung is None else crung
    if key[0] == SYM_KIND:
        if float(env.get("HEROSIM_DECISION_COST_MS", "nan")) != R.CD_BAR_CHARGE[rung]:
            raise SystemExit(f"FAIL LOUD: {f} sym decision charge {env.get('HEROSIM_DECISION_COST_MS')!r}, not the bar's {R.CD_BAR_CHARGE[rung]}")
        if env.get(SYM_FLAG) != "1":
            raise SystemExit(f"FAIL LOUD: {f} is a sym cell without {SYM_FLAG}=1")
    elif env.get(SYM_FLAG) not in (None, "", "0"):
        raise SystemExit(f"FAIL LOUD: {f} ({key[0]}) carries {SYM_FLAG}={env.get(SYM_FLAG)!r}")


R.load, R.cd_check = load, cd_check


def main():
    global SYM_PIN
    ap = argparse.ArgumentParser()
    for k in ("--cells", "--sym-pin", "--ids", "--ids-sha256", "--rulings", "--pin", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--allow-missing", default="")
    a = ap.parse_args()
    SYM_PIN = a.sym_pin
    if os.path.exists(a.out):
        raise SystemExit(f"FAIL LOUD: {a.out} exists (read once)")
    ids = tuple(int(x) for x in a.ids.split(","))
    sha = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
    if sha != a.ids_sha256 or len(set(ids)) != len(ids):
        raise SystemExit("FAIL LOUD: the id list does not match the filed sha256 (or repeats an id)")
    allow = []
    for c in filter(None, a.allow_missing.split(",")):
        m = R.NAME.match(c + ".summary.json")
        allow.append((m.group(4), int(m.group(5)), int(m.group(1)), m.group(2), m.group(3)))
    res = R.run_cd(a.cells.split(","), json.load(open(a.rulings)), ids, a.pin, None, sha, allow_missing=allow,
                   arms=["gnn", "wide", "mlp", "sym", "bar", "random"], primary_pairs=[("gnn", "sym")])
    res.pop("labels", None)
    res.pop("gnn_blocked_by_failure", None)
    res["family"] = "llf_cd_confirm_v1 fixed-bar descriptive read (not registered, no label)"
    res["wrapper_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    res["sym_pin"] = SYM_PIN
    with open(a.out + ".tmp", "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    os.replace(a.out + ".tmp", a.out)
    fm = lambda x: "n/a" if x is None else f"{x:+.2f}"
    for rd in ("no_runaway", "all"):
        for pair in ("sym_vs_bar", "gnn_vs_sym", "wide_vs_sym", "mlp_vs_sym", "gnn_vs_bar"):
            for r in ("moderate", "heavy"):
                v = res["secondary"][f"{rd}|whole|{pair}|{r}"]
                print(f"{rd:10s} {pair:12s} {r:8s} {fm(v['median_pct'])} % {v['faster']}/{v['n']} p {v['p']}")


if __name__ == "__main__":
    main()
