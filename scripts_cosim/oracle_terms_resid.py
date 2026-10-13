#!/usr/bin/env python3
"""Type-aware residual model for S. Which part of (label - S) is predictable from S's own terms, split by task type?  Reads oracle_diag/<cell>/*.jsonl written with ORACLE_DIAG_TERMS=1
(scheduler `_pg_terms_rec`); no simulation.

Row = one logged plan p of a within-cap batch, against the hand plan h (= s_plan, the S-argmin).  Target rho = (label_p - label_h) - (S_p - S_h).  Features: the per-task-type
term deltas (queue = drain+inflight, stack_raw, cold, exec, lat+in_wait, exch) summed over tasks of a type, the totals, moved-task counts by type, node-count change, batch size, t<360.
`agnostic` = totals only (no type), `typed` = totals + per type.  Held out by topology (leave-one-heavy-topology-out; and train all heavy -> moderate 19405).
Reported: R2 of rho (all / t<360 / steady), grouped permutation importance (a term group's columns permuted together), and the regret of argmin(S + rho_hat) over two pools:
A = hand + the 4 lowest-S logged plans (no label information in the pool), B = A + the logged random plans.  R2 < 0.05 => noise.

  oracle_terms_resid.py <view dir with oracle_diag/> [--json OUT]
"""
import argparse, glob, json, os, re
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

GROUP = {"queue": ["drain", "inflight"], "stack_raw": ["stack_raw"], "cold": ["cold"], "exec": ["exec"], "lat": ["lat", "in_wait"], "exch": ["exch"]}
TYPES = ["cnn", "dnn1", "dnn2", "rf"]
T0 = 360.0


def cell_of(p):
    m = re.search(r"cc40s(\d+)__g0(\w+?)__", p)
    return int(m.group(1)), m.group(2)


def rows_of(d, topo, rung):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            if "terms" not in r or r.get("search"):
                continue
            ci = {c: i for i, c in enumerate(r["terms_cols"])}
            tm = {tuple(map(tuple, k)): np.array(v, float) for k, v in r["terms"]}
            lab = {tuple(map(tuple, k)): (s, l) for k, s, l in r["table"]}
            h = tuple(map(tuple, r["s_plan"]))
            if h not in tm or h not in lab:
                continue
            types = r["ctx"]["task_types"]
            tidx = {t: [i for i, x in enumerate(types) if x == t] for t in TYPES}
            nsel = r["terms_selected"]
            S_all = np.array(sorted(s for s, _ in lab.values()))
            s_cut = S_all[min(4, len(S_all) - 1)]
            hn = len({x[0] for x in h})
            cand = []
            for j, (k, a) in enumerate(tm.items()):
                if k == h or k not in lab:
                    continue
                dl = a - tm[h]
                g = {n: dl[:, [ci[c] for c in cs]].sum(1) for n, cs in GROUP.items()}      # per task
                feat = [g[n].sum() for n in GROUP]
                for t in TYPES:
                    feat += [g[n][tidx[t]].sum() for n in GROUP]
                moved = [i for i in range(len(types)) if k[i] != h[i]]
                feat += [len(moved)] + [sum(types[i] == t for i in moved) for t in TYPES]
                feat += [len({x[0] for x in k}) - hn, r["n"], 1.0 if r["t"] < T0 else 0.0]
                s_p, l_p = lab[k]
                cand.append({"feat": feat, "rho": (l_p - lab[h][1]) - (s_p - lab[h][0]),
                             "dS": s_p - lab[h][0], "dL": l_p - lab[h][1], "rand": j >= nsel, "lowS": s_p <= s_cut})
            if cand:
                out.append({"topo": topo, "rung": rung, "t": r["t"], "hand_label": lab[h][1], "cand": cand})
    return out


def names():
    n = [f"tot_{g}" for g in GROUP]
    for t in TYPES:
        n += [f"{t}_{g}" for g in GROUP]
    return n + ["n_moved"] + [f"moved_{t}" for t in TYPES] + ["d_nodes", "n", "early"]


def flat(batches):
    X, y, dS, bi = [], [], [], []
    for b, bt in enumerate(batches):
        for c in bt["cand"]:
            X.append(c["feat"]); y.append(c["rho"]); dS.append(c["dS"]); bi.append(b)
    return np.array(X), np.array(y), np.array(dS), np.array(bi)


def fit(X, y, cols, seed=0):
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, max_leaf_nodes=24, min_samples_leaf=60, random_state=seed)
    return m.fit(X[:, cols], y)


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / max(1e-12, ((y - y.mean()) ** 2).sum()))


def regret(batches, pred, bi, pools):
    """mean over batches of (label of argmin(S+rho_hat) - hand label); hand (d=0) is always in the pool.  Also the oracle-in-pool floor."""
    res = {}
    for pool in pools:
        ch, orc, base, hand = [], [], [], 0.0
        k = 0
        for b, bt in enumerate(batches):
            idx = [i for i, c in enumerate(bt["cand"]) if (pool == "B" or c["lowS"])]
            sub = [bt["cand"][i] for i in idx]
            p = pred[k:k + len(bt["cand"])][idx] if idx else np.array([])
            k += len(bt["cand"])
            hand += bt["hand_label"]
            if not sub:
                ch.append(0.0); orc.append(0.0); continue
            dS = np.array([c["dS"] for c in sub]); dL = np.array([c["dL"] for c in sub])
            j = int(np.argmin(dS + p)); ch.append(float(dL[j]) if dS[j] + p[j] < 0 else 0.0)   # hand stays unless the corrected score is strictly lower
            orc.append(min(0.0, float(dL.min())))
        res[pool] = {"model_gain_pct_neg_is_loss": 100 * -sum(ch) / hand, "oracle_in_pool_gain_pct": 100 * -sum(orc) / hand, "n_batches": len(batches)}
    return res


def evaluate(train, test, label):
    X, y, dS, bi = flat(train); Xt, yt, dSt, bit = flat(test)
    nm = names(); typed = list(range(len(nm)))
    agn = [i for i, n in enumerate(nm) if n.startswith("tot_") or n in ("n_moved", "d_nodes", "n", "early")]
    out = {"label": label, "train_rows": len(y), "test_rows": len(yt), "test_batches": len(test), "rho_std": float(yt.std())}
    early = Xt[:, nm.index("early")] > 0.5
    rng = np.random.default_rng(0)
    preds = {}
    for nmod, cols in (("typed", typed), ("agnostic", agn)):
        m = fit(X, y, cols); p = m.predict(Xt[:, cols]); preds[nmod] = p
        out[nmod] = {"r2": r2(yt, p), "r2_early": r2(yt[early], p[early]) if early.sum() > 50 else None, "r2_steady": r2(yt[~early], p[~early]) if (~early).sum() > 50 else None}
        if nmod == "typed":
            base = out[nmod]["r2"]; imp = {}
            for g in GROUP:
                gc = [i for i, n in enumerate(nm) if n.endswith("_" + g)]
                Xp = Xt.copy()
                perm = rng.permutation(len(yt))
                Xp[:, gc] = Xt[perm][:, gc]
                imp[g] = base - r2(yt, m.predict(Xp[:, cols]))
            out["typed"]["perm_importance_r2_drop"] = imp
            for t in TYPES:
                tc = [i for i, n in enumerate(nm) if n.startswith(t + "_") or n == "moved_" + t]
                Xp = Xt.copy(); perm = rng.permutation(len(yt)); Xp[:, tc] = Xt[perm][:, tc]
                imp["type:" + t] = base - r2(yt, m.predict(Xp[:, cols]))
    out["regret"] = {k: regret(test, preds[k], bit, ("A", "B")) for k in preds}
    out["regret"]["S_only"] = regret(test, np.zeros(len(yt)), bit, ("A", "B"))
    for tag, want_early in (("early", True), ("steady", False)):          # the same regret on batches with t < 360 s / t >= 360 s
        keep = [b for b, bt in enumerate(test) if (bt["t"] < T0) == want_early]
        sub = [test[b] for b in keep]
        sel = np.isin(bit, keep)
        out["regret_" + tag] = {k: regret(sub, preds[k][sel], None, ("A", "B")) for k in preds}
        if sel.sum() > 50:                                                # permutation importance inside the bin
            m = fit(X, y, typed); base = r2(yt[sel], m.predict(Xt[sel]))
            imp2 = {}
            for g in GROUP:
                gc = [i for i, n in enumerate(nm) if n.endswith("_" + g)]
                Xp = Xt[sel].copy(); Xp[:, gc] = Xp[rng.permutation(sel.sum())][:, gc]
                imp2[g] = base - r2(yt[sel], m.predict(Xp))
            out["perm_importance_" + tag] = {"base_r2": base, **imp2}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("view"); ap.add_argument("--json", default=None)
    a = ap.parse_args()
    data = {}
    for d in sorted(glob.glob(os.path.join(a.view, "oracle_diag", "cc40s*__g0*__*"))):
        t, g = cell_of(d)
        data[(t, g)] = rows_of(d, t, g)
        print(f"loaded {t} {g}: {len(data[(t, g)])} batches", flush=True)
    heavy = [k for k in data if k[1] == "heavy"]; mod = [k for k in data if k[1] == "moderate"]
    res = []
    for k in heavy:
        res.append(evaluate([b for kk in heavy if kk != k for b in data[kk]], data[k], f"heavy: hold out {k[0]}"))
    if mod:
        res.append(evaluate([b for kk in heavy for b in data[kk]], [b for kk in mod for b in data[kk]], "heavy -> moderate 19405"))
    for r in res:
        print(json.dumps(r, default=float))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
