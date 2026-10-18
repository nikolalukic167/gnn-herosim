#!/usr/bin/env python3
"""uncut_groups_s0_v1 seed extension (registered 2026-10-17, B6, post-look): the four extension families on the new seeds only.

Every arm's new-seed cells are first read by uncut_arm_gate_read.py --seeds <its seeds> (its assertions: checkpoint md5s, w at the pin,
host, warmth); this tool takes those reads (--read, one per arm) and the same cell directories (--gate), and computes:
  E1 (a): Arm 3 gnn vs Arm 1b gnn, both rungs x both statistics (k = 4), seeds 11-18.
  E2 (b): WITHDRAWN 2026-10-17 by the user's direction ("stop all twin trainings") before any extension twin finished; no twin is read.
  E3 (c): heavy whole and heavy after-360 vs cd_ib0 for Arm 3, Arm 1b and K0 (k = 6); Arm 3 / 1b on 11-18, K0 on 11-16.
  E4 (d): heavy Arm 2 gnn vs K0 gnn (K0 the claimed faster), k = 2, seeds 11-16. E4b (K0 gnn vs Arm 1 gnn) WITHDRAWN 2026-10-17: the user
  cancelled the Arm 1 gnn extension runs before any finished, so Arm 1 has no new-seed read (amendment: "E4 is k = 2").
Statistic: the reader's exactly (cell = mean over the seeds; per topology the median over windows of the paired %; exact two-sided Wilcoxon over
topologies), Holm within each family, separately for the frozen-w cells (_k100) and the co-tenancy cells (_kr).
Labels per test (claimed direction: the first arm faster, except E4's Arm 2 - K0, where K0 is the faster):
  CLAIM: Holm p < .05 and median in the claimed direction. REVERSED: Holm p < .05, median against it.
  CLEAN-NEGATIVE: neither, and the sign-test 95 % interval for the median over topologies (n = 12: [x(3), x(10)], coverage 96.1 %) excludes a
  claimed-direction effect of 1 % or more (post-look margin). UNRESOLVED: otherwise.
Combined per test: the frozen and co-tenancy labels if they agree, else UNRESOLVED. A rung's finding is a claim (clean negative) only if both of
its statistics are. --pooled adds the same contrasts on seeds 1, 2 and the new seeds together: descriptive, no label, no Holm.
"""
import argparse
import hashlib
import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uncut_arm_gate_read as G  # noqa: E402

A3G = "ra_gnn_eng_arm3_nosplit"
A1BG = "ra_gnn_eng_arm1b_nosplit"
K0G, A2G = "ra_gnn_eng_arm2k0_nosplit", "ra_gnn_eng_arm2_nosplit"
P1, P2 = tuple(range(11, 19)), tuple(range(11, 17))
SEEDS = {A3G: P1, A1BG: P1, K0G: P2, A2G: P2}
READS = {A3G: "", A1BG: "", K0G: "", A2G: ""}  # primary -> twin in its own read (no extension twin: E2 withdrawn)
MARGIN, ALPHA = 1.0, 0.05
# (family, first arm, second arm, rungs, claimed sign): sign -1 = first arm faster (negative paired %), +1 = second arm faster
FAMILIES = {
    "E1": [(A3G, A1BG, G.RUNGS, -1)],
    "E3": [(A3G, G.BAR, ("heavy",), -1), (A1BG, G.BAR, ("heavy",), -1), (K0G, G.BAR, ("heavy",), -1)],
    "E4": [(A2G, K0G, ("heavy",), +1)],
}


def ci_index(n):
    """0-based index k of the sign-test interval [x(k+1), x(n-k)] with coverage >= 95 %: the largest k with 1 - 2 P(Bin(n, .5) <= k) >= .95."""
    k = -1
    while 1 - 2 * sum(math.comb(n, i) for i in range(k + 2)) / 2 ** n >= 0.95:
        k += 1
    if k < 0:
        raise SystemExit(f"FAIL LOUD: {n} topologies cannot give a 95 % sign-test interval")
    return k


def label(vals, p_holm, sign):
    med = st.median(vals)
    if p_holm < ALPHA:
        return "CLAIM" if sign * med > 0 else "REVERSED"
    e = sorted(-v if sign < 0 else v for v in vals)  # the claimed-direction effect, positive = as claimed
    k = ci_index(len(e))
    return "CLEAN-NEGATIVE" if e[len(e) - 1 - k] < MARGIN else "UNRESOLVED"


def seeds_of(arm, pooled):
    if arm == G.BAR:
        return (0,)
    return ((1, 2) + SEEDS[arm]) if pooled else SEEDS[arm]


def raw_values(cells, a, b, rung, stat, topos, pooled):
    out = {}
    for t in topos:
        v = []
        for w in G.WINDOWS:
            ma, mb = G.cell_mean(cells, a, seeds_of(a, pooled), t, w, rung), G.cell_mean(cells, b, seeds_of(b, pooled), t, w, rung)
            if ma is not None and mb is not None:
                v.append(100.0 * (ma[stat] - mb[stat]) / mb[stat])
        if v:
            out[t] = st.median(v)
    return out


def family_tests(cells, fam, topos, pooled=False):
    tests = []
    for a, b, rungs, sign in FAMILIES[fam]:
        for r in rungs:
            for s in G.STATS:
                per = raw_values(cells, a, b, r, s, topos, pooled)
                vals = list(per.values())
                tests.append({"a": a, "b": b, "rung": r, "stat": s, "claimed_sign": sign, "n_topologies": len(vals),
                              "median_pct": st.median(vals) if vals else None, "faster": sum(1 for x in vals if x < 0),
                              "p": G.exact_wilcoxon(vals) if vals else None, "per_topology": {str(t): round(x, 3) for t, x in sorted(per.items())},
                              "_vals": vals})
    if not pooled:
        adj = G.holm({str(i): e["p"] for i, e in enumerate(tests)})
        for i, e in enumerate(tests):
            e["p_holm"] = 1.0 if adj[str(i)] is None else adj[str(i)]
            if e["n_topologies"] != len(topos):
                raise SystemExit(f"FAIL LOUD: {fam} {e['a']} vs {e['b']} {e['rung']} {e['stat']}: {e['n_topologies']} of {len(topos)} topologies")
            e["label"] = label(e["_vals"], e["p_holm"], e["claimed_sign"])
            k = ci_index(len(e["_vals"]))
            srt = sorted(e["_vals"])
            e["sign_ci_95"] = [srt[k], srt[len(srt) - 1 - k]]
    for e in tests:
        del e["_vals"]
    return tests


def combine(frozen, cot):
    out = []
    for f, c in zip(frozen, cot):
        assert (f["a"], f["b"], f["rung"], f["stat"]) == (c["a"], c["b"], c["rung"], c["stat"])
        out.append({"a": f["a"], "b": f["b"], "rung": f["rung"], "stat": f["stat"], "frozen": f["label"], "cotenancy": c["label"],
                    "combined": f["label"] if f["label"] == c["label"] else "UNRESOLVED"})
    rungs = {}
    for e in out:
        rungs.setdefault((e["a"], e["b"], e["rung"]), []).append(e["combined"])
    findings = {f"{a} vs {b} {r}": (ls[0] if len(set(ls)) == 1 and ls[0] in ("CLAIM", "CLEAN-NEGATIVE", "REVERSED") else "UNRESOLVED")
                for (a, b, r), ls in rungs.items()}
    return out, findings


def check_reads(paths):
    seen = {}
    for p in paths:
        with open(p) as fh:
            r = json.load(fh)
        pr = r.get("primary")
        if pr not in READS:
            raise SystemExit(f"FAIL LOUD: {p} reads {pr!r}, not an extension arm")
        if r.get("twin", "") != READS[pr]:
            raise SystemExit(f"FAIL LOUD: {p}: twin {r.get('twin')!r}, the extension reads {pr} with twin {READS[pr]!r}")
        if tuple(r.get("seeds") or ()) != SEEDS[pr]:
            raise SystemExit(f"FAIL LOUD: {p} was read at seeds {r.get('seeds')}, the extension registered {list(SEEDS[pr])} for {pr}")
        if not isinstance(r.get("combined"), dict):
            raise SystemExit(f"FAIL LOUD: {p} is not a read under both verdicts (--cotenancy required)")
        seen[pr] = {"path": os.path.abspath(p), "md5": G.md5(p)}
    missing = sorted(set(READS) - set(seen))
    if missing:
        raise SystemExit(f"FAIL LOUD: no extension read for {missing}")
    return seen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", required=True, help="every cell directory the reads used (bar and new-seed cells, _k100 and _kr)")
    ap.add_argument("--read", required=True, help="comma-separated: the reader's --seeds output for Arm 3, Arm 1b, K0 and Arm 2 gnn")
    ap.add_argument("--topos", required=True)
    ap.add_argument("--topos-sha256", required=True)
    ap.add_argument("--pooled-gate", default="", help="cell directories holding seeds 1, 2 too: adds the pooled descriptive line")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if hashlib.sha256(a.topos.encode()).hexdigest() != a.topos_sha256:
        raise SystemExit("FAIL LOUD: the gate topology list hash differs from the filed one")
    topos = [int(x) for x in a.topos.split(",")]
    reads = check_reads([x for x in a.read.split(",") if x])
    cells, _ = G.load(a.gate, "_k100")
    frozen = {k: v for k, v in cells.items() if not k[0].endswith(G.KR)}
    cot = {(k[0][:-len(G.KR)],) + k[1:]: v for k, v in cells.items() if k[0].endswith(G.KR)}
    cot.update({k: v for k, v in frozen.items() if k[0] == G.BAR})
    stray = sorted({(k[0], k[1]) for k in frozen if k[0] in SEEDS and k[1] not in SEEDS[k[0]]})
    if stray:
        raise SystemExit(f"FAIL LOUD: new-seed cell set holds other seeds: {stray[:6]}")
    res = {"registration": "uncut_groups_s0_v1 seed extension, 2026-10-17 (B6, post-look); filed verdicts unchanged", "reads": reads,
           "seeds": {k: list(v) for k, v in SEEDS.items()}, "margin_pct": MARGIN, "families": {}, "findings": {}}
    for fam in FAMILIES:
        f, c = family_tests(frozen, fam, topos), family_tests(cot, fam, topos)
        comb, findings = combine(f, c)
        res["families"][fam] = {"holm_family": len(f), "frozen": f, "cotenancy": c, "combined": comb}
        res["findings"][fam] = findings
    if a.pooled_gate:
        pc, _ = G.load(a.pooled_gate, "_k100")
        pf = {k: v for k, v in pc.items() if not k[0].endswith(G.KR)}
        res["pooled_descriptive"] = {fam: family_tests(pf, fam, topos, pooled=True) for fam in FAMILIES}
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    for fam, fs in res["findings"].items():
        print(fam, json.dumps(fs))
    return res


if __name__ == "__main__":
    main()
