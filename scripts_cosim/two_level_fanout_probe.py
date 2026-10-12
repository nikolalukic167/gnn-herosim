#!/usr/bin/env python3
"""Offline probe: does the realised first-level fan-out of an Alibaba MSCallGraph request predict the second-level fan-out?

Rows: the same shard rows and the same first --n-traces requests (arrival order) as grounded_workload_v1_extract.py (file md5 checked against the committed library).
Unit: a call `m` that is a member of a first-level fan-out F1 (calls sharing an rpcid parent, |F1| >= 2). Its second-level fan-out is the calls whose parent is m's rpcid:
width w2 (0 = leaf) and the set S2 of their callee services (`dm`). Features known when F1 is revealed: callee service of m, caller (parent) service `um`, rpctype, |F1|,
the multiset of sibling callee services, m's dispatch rank. Baseline: the per-callee-service table.

  two_level_fanout_probe.py --csv MSCallGraph_0.csv --lib fanout_lib_0.json.gz --out probe_0.json
"""
import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

NMIN, LAM = 5, 2.0


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def build(csv, n_traces):
    df = pd.read_csv(csv, usecols=["traceid", "timestamp", "rpcid", "um", "rpctype", "dm"], dtype=str)
    n_rows = len(df)
    df["timestamp"] = df["timestamp"].astype("int64")
    start = df.groupby("traceid", sort=False)["timestamp"].min().sort_values(kind="stable")
    keep = start.iloc[:n_traces]
    sub = df[df["traceid"].isin(set(keep.index))].dropna(subset=["rpcid"])
    sub = sub[sub["rpcid"].str.fullmatch(r"\d+(\.\d+)+")].copy()
    sub["parent"] = sub["rpcid"].str.rsplit(".", n=1).str[0]
    order = {t: i for i, t in enumerate(keep.index)}
    sub["req"] = sub["traceid"].map(order)
    n_dup = int(sub.duplicated(["traceid", "rpcid"]).sum())
    sub = sub.drop_duplicates(["traceid", "rpcid"], keep="first")
    return df, n_rows, keep, sub, n_dup


def members(sub):
    g = sub.groupby(["traceid", "parent"])
    sub = sub.assign(w1=g["rpcid"].transform("size"))
    f1 = sub[sub["w1"] >= 2].copy()
    sib = f1.groupby(["traceid", "parent"])["dm"].agg(lambda s: "|".join(sorted(s)))
    f1 = f1.join(sib.rename("sibset"), on=["traceid", "parent"])
    f1["rank"] = f1.groupby(["traceid", "parent"])["timestamp"].rank(method="first")
    f1["same_svc_sibs"] = f1.groupby(["traceid", "parent", "dm"])["rpcid"].transform("size") - 1
    ch = sub.groupby(["traceid", "parent"])["dm"].agg(w2="size", s2=lambda s: "|".join(sorted(set(s))))
    f1 = f1.join(ch, on=["traceid", "rpcid"]) if False else f1.merge(ch.reset_index().rename(columns={"parent": "rpcid"}), on=["traceid", "rpcid"], how="left")
    f1["w2"] = f1["w2"].fillna(0).astype(int)
    f1["s2"] = f1["s2"].fillna("")
    return f1.reset_index(drop=True)


def base_rates(sub, f1, n_traces):
    tr_f1 = set(f1["traceid"])
    tr_l2 = set(f1.loc[f1["w2"] > 0, "traceid"])
    has_kids = f1[f1["w2"] > 0]
    wd = has_kids["w2"].clip(upper=11)
    hist = {("11+" if k == 11 else str(k)): int(v) for k, v in wd.value_counts().sort_index().items()}
    svc = f1.groupby("dm")
    big = [(k, g) for k, g in svc if len(g) >= 20]
    det_w = [(len(g), g["w2"].nunique() == 1) for k, g in big]
    det_s = [(len(g), g["s2"].nunique() == 1) for k, g in big]
    modal = lambda col: float(f1.groupby("dm")[col].agg(lambda s: s.value_counts().iloc[0]).sum() / len(f1))
    return {"traces": n_traces, "traces_with_first_level_fanout": len(tr_f1), "traces_with_second_level": len(tr_l2),
            "share_traces_first_level": len(tr_f1) / n_traces, "share_traces_second_level": len(tr_l2) / n_traces,
            "share_of_first_level_traces_with_second_level": len(tr_l2) / max(1, len(tr_f1)),
            "members": len(f1), "member_leaf_share": float((f1["w2"] == 0).mean()), "members_with_children": int((f1["w2"] > 0).sum()),
            "second_level_width_hist_members_with_children": hist, "second_level_width_mean": float(has_kids["w2"].mean()),
            "second_level_width_median": float(has_kids["w2"].median()), "second_level_width_p90": float(has_kids["w2"].quantile(0.9)),
            "services": int(f1["dm"].nunique()), "services_ge20_members": len(big),
            "deterministic_width_services_ge20": [sum(1 for _, d in det_w if d), len(det_w)],
            "deterministic_width_member_share_of_services_ge20": sum(n for n, d in det_w if d) / max(1, sum(n for n, _ in det_w)),
            "deterministic_set_services_ge20": [sum(1 for _, d in det_s if d), len(det_s)],
            "deterministic_set_member_share_of_services_ge20": sum(n for n, d in det_s if d) / max(1, sum(n for n, _ in det_s)),
            "table_modal_width_accuracy_in_sample": modal("w2"), "table_modal_set_accuracy_in_sample": modal("s2"),
            "members_in_services_ge20": sum(n for n, _ in det_w) / len(f1)}


CTX = [("svc", ["dm"]), ("svc+rpctype+w1", ["dm", "rpctype", "w1"]), ("svc+parent", ["dm", "um"]), ("svc+parent+rpctype+w1", ["dm", "um", "rpctype", "w1"]),
       ("svc+sibset", ["dm", "sibset"]), ("svc+parent+sibset", ["dm", "um", "sibset"])]


def fit_ctx(train, cols, target):
    d = defaultdict(Counter)
    for key, y in zip(zip(*[train[c] for c in cols]), train[target]):
        d[key][y] += 1
    return d


def backoff_cross_entropy(train, test, cols, target):
    """Held-out cross-entropy (bits/member) of p(target | cols) backed off to p(target | dm), then to the global distribution; unseen outcomes share one OTHER symbol."""
    glob = Counter(train[target])
    n_g = sum(glob.values())
    vocab = len(glob) + 1
    p_glob = lambda y: ((glob.get(y, 0) + 0.0) + 1.0 / vocab) / (n_g + 1.0)
    svc = fit_ctx(train, ["dm"], target)
    ctx = fit_ctx(train, cols, target)
    tot = 0.0
    for key, y in zip(zip(*[test[c] for c in cols]), test[target]):
        ps = svc.get((key[cols.index("dm")],), None) if "dm" in cols else None
        n_s = sum(ps.values()) if ps else 0
        p_s = ((ps.get(y, 0) if ps else 0) + LAM * p_glob(y)) / (n_s + LAM)
        cc = ctx.get(key)
        n_c = sum(cc.values()) if cc else 0
        p = ((cc.get(y, 0) if cc else 0) + LAM * p_s) / (n_c + LAM)
        tot += -math.log2(p)
    return tot / len(test)


def in_sample_entropy(df, cols, target):
    n = len(df)
    j = df.groupby(cols + [target]).size()
    c = df.groupby(cols).size()
    h = -(j * np.log2(j / c.reindex(j.index.droplevel(-1)).to_numpy())).sum() / n
    return float(h)


def splits(f1, n_folds, seed):
    rng = np.random.default_rng(seed)
    reqs = f1["req"].unique()
    fold_of = dict(zip(reqs, rng.integers(0, n_folds, len(reqs))))
    f = f1["req"].map(fold_of).to_numpy()
    out = [(np.where(f != k)[0], np.where(f == k)[0], f"fold{k}") for k in range(n_folds)]
    cut = np.quantile(f1["req"], 0.7)
    out.append((np.where(f1["req"] <= cut)[0], np.where(f1["req"] > cut)[0], "temporal"))
    return out


def width_models(f1, tr, te):
    from sklearn.ensemble import HistGradientBoostingRegressor
    train, test = f1.iloc[tr], f1.iloc[te]
    g_mean, g_med = train["w2"].mean(), train["w2"].median()
    s = train.groupby("dm")["w2"].agg(["sum", "count", "median"])
    tab_mean = test["dm"].map(s["sum"] / s["count"]).fillna(g_mean).to_numpy()
    tab_med = test["dm"].map(s["median"]).fillna(g_med).to_numpy()
    y = test["w2"].to_numpy()
    # features
    top = {k: i for i, k in enumerate(train["dm"].value_counts().index[:254])}
    topu = {k: i for i, k in enumerate(train["um"].value_counts().index[:254])}
    rt = {k: i for i, k in enumerate(train["rpctype"].unique())}
    sm = (s["sum"] / s["count"]).to_dict()

    def feats(d, loo):
        own = d["dm"].map(sm).fillna(g_mean).to_numpy()
        if loo:
            c, sm_ = d["dm"].map(s["count"]).to_numpy(float), d["dm"].map(s["sum"]).to_numpy(float)
            own = np.where(c > 1, (sm_ - d["w2"].to_numpy()) / np.maximum(c - 1, 1), g_mean)
        sibmean = (d["sibset"].map(lambda z: np.mean([sm.get(k, g_mean) for k in z.split("|")]))).to_numpy()
        X = np.column_stack([d["dm"].map(top).fillna(254).to_numpy(), d["um"].map(topu).fillna(254).to_numpy(), d["rpctype"].map(rt).fillna(len(rt)).to_numpy(),
                             d["w1"].to_numpy(), d["same_svc_sibs"].to_numpy(), d["rank"].to_numpy(), d["rank"].to_numpy() / d["w1"].to_numpy(), own, sibmean])
        return X
    Xtr, Xte = feats(train, True), feats(test, False)
    cat = [True, True, True, False, False, False, False, False, False]
    res = {"y": y, "tab_mean": tab_mean, "tab_med": tab_med}
    for loss, key in (("squared_error", "gbm_mean"), ("absolute_error", "gbm_med")):
        m = HistGradientBoostingRegressor(loss=loss, max_iter=200, learning_rate=0.1, categorical_features=cat, random_state=0)
        m.fit(Xtr, train["w2"].to_numpy())
        res[key] = m.predict(Xte)
    # same learner without the first-level features (service + own table only) isolates what the joint features add
    sel = [0, 7]
    for loss, key in (("squared_error", "gbm_svc_mean"), ("absolute_error", "gbm_svc_med")):
        m = HistGradientBoostingRegressor(loss=loss, max_iter=200, learning_rate=0.1, categorical_features=[True, False], random_state=0)
        m.fit(Xtr[:, sel], train["w2"].to_numpy())
        res[key] = m.predict(Xte[:, sel])
    return res


def set_models(f1, tr, te):
    train, test = f1.iloc[tr], f1.iloc[te]
    out = {}
    levels = [CTX[0]] + [CTX[i] for i in (1, 2, 3, 4, 5)]
    maps = {}
    for name, cols in levels:
        d = fit_ctx(train, cols, "s2")
        maps[name] = (cols, d)
    svc_mode = {k[0]: c.most_common(1)[0][0] for k, c in maps["svc"][1].items()}
    glob_mode = Counter(train["s2"]).most_common(1)[0][0]

    def joint_pred(row):
        for name in ("svc+parent+sibset", "svc+sibset", "svc+parent+rpctype+w1", "svc+parent", "svc+rpctype+w1"):
            cols, d = maps[name]
            c = d.get(tuple(row[k] for k in cols))
            if c and sum(c.values()) >= NMIN:
                return c.most_common(1)[0][0]
        return svc_mode.get(row["dm"], glob_mode)
    rows = test.to_dict("records")
    tab = [svc_mode.get(r["dm"], glob_mode) for r in rows]
    jnt = [joint_pred(r) for r in rows]
    act = test["s2"].tolist()

    def jac(a, b):
        A, B = set(a.split("|")) - {""}, set(b.split("|")) - {""}
        return 1.0 if not A and not B else len(A & B) / len(A | B)
    out["table_exact"] = float(np.mean([a == b for a, b in zip(tab, act)]))
    out["joint_exact"] = float(np.mean([a == b for a, b in zip(jnt, act)]))
    out["table_jaccard"] = float(np.mean([jac(a, b) for a, b in zip(tab, act)]))
    out["joint_jaccard"] = float(np.mean([jac(a, b) for a, b in zip(jnt, act)]))
    kid = np.array([a != "" for a in act])
    for k, v in (("table", tab), ("joint", jnt)):
        out[f"{k}_exact_kids"] = float(np.mean([a == b for a, b, m in zip(v, act, kid) if m])) if kid.any() else None
        out[f"{k}_jaccard_kids"] = float(np.mean([jac(a, b) for a, b, m in zip(v, act, kid) if m])) if kid.any() else None
    out["n"] = len(act)
    out["n_kids"] = int(kid.sum())
    return out


def rel(a, b):
    return (a - b) / a if a else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True); ap.add_argument("--lib", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-traces", type=int, default=40000); ap.add_argument("--folds", type=int, default=5); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    lib = json.load(gzip.open(a.lib))["source"]
    got = md5(a.csv)
    if got != lib["md5"]:
        raise SystemExit(f"FAIL LOUD: {a.csv} md5 {got} != library {lib['md5']}")
    df, n_rows, keep, sub, n_dup = build(a.csv, a.n_traces)
    if n_rows != lib["rows"]:
        raise SystemExit(f"FAIL LOUD: {n_rows} rows != library {lib['rows']}")
    f1 = members(sub)
    res = {"file": a.csv, "md5": got, "rows": n_rows, "library": a.lib, "library_rows": lib["rows"], "traces_in_shard": int(df["traceid"].nunique()), "traces_used": len(keep),
           "calls_in_used_traces": len(sub), "duplicate_rpcid_rows_dropped": n_dup, "first_level_member_rows": len(f1)}
    res["base_rates"] = base_rates(sub, f1, len(keep))
    res["entropy_in_sample_bits"] = {t: {"H(.|none)": in_sample_entropy(f1.assign(_=0), ["_"], t), **{n: in_sample_entropy(f1, c, t) for n, c in CTX}} for t in ("w2", "s2")}
    sp = splits(f1, a.folds, a.seed)
    ce = {t: defaultdict(list) for t in ("w2", "s2")}
    wm, sm = defaultdict(list), defaultdict(list)
    for tr, te, name in sp:
        train, test = f1.iloc[tr], f1.iloc[te]
        for t in ("w2", "s2"):
            ce[t][name] = {n: backoff_cross_entropy(train, test, c, t) for n, c in CTX}
        r = width_models(f1, tr, te)
        y = r["y"]
        wm[name] = {"n": len(y), **{k: {"rmse": float(np.sqrt(np.mean((r[k] - y) ** 2))), "mae": float(np.mean(np.abs(r[k] - y)))} for k in r if k != "y"}}
        sm[name] = set_models(f1, tr, te)
        print(name, "done", flush=True)
    res["heldout_cross_entropy_bits"] = {t: dict(ce[t]) for t in ce}
    res["width_models"] = dict(wm)
    res["set_models"] = dict(sm)
    cv = [n for _, _, n in sp if n.startswith("fold")]
    def mean_over(d, f):
        return float(np.mean([f(d[n]) for n in cv]))
    agg = {"width": {}, "set": {}, "cross_entropy": {}}
    for k in ("tab_mean", "gbm_svc_mean", "gbm_mean"):
        agg["width"][k + "_rmse"] = mean_over(wm, lambda x: x[k]["rmse"])
    for k in ("tab_med", "gbm_svc_med", "gbm_med"):
        agg["width"][k + "_mae"] = mean_over(wm, lambda x: x[k]["mae"])
    w = agg["width"]
    w["rmse_reduction_gbm_vs_table"] = rel(w["tab_mean_rmse"], w["gbm_mean_rmse"])
    w["rmse_reduction_gbm_svc_only_vs_table"] = rel(w["tab_mean_rmse"], w["gbm_svc_mean_rmse"])
    w["mae_reduction_gbm_vs_table"] = rel(w["tab_med_mae"], w["gbm_med_mae"])
    w["mae_reduction_gbm_svc_only_vs_table"] = rel(w["tab_med_mae"], w["gbm_svc_med_mae"])
    wt = wm["temporal"]
    w["temporal"] = {"rmse_table": wt["tab_mean"]["rmse"], "rmse_gbm": wt["gbm_mean"]["rmse"], "rmse_reduction": rel(wt["tab_mean"]["rmse"], wt["gbm_mean"]["rmse"]),
                     "mae_table": wt["tab_med"]["mae"], "mae_gbm": wt["gbm_med"]["mae"], "mae_reduction": rel(wt["tab_med"]["mae"], wt["gbm_med"]["mae"])}
    for t in ("w2", "s2"):
        agg["cross_entropy"][t] = {n: mean_over(ce[t], lambda x: x[n]) for n, _ in CTX}
        base = agg["cross_entropy"][t]["svc"]
        agg["cross_entropy"][t]["reduction_vs_svc"] = {n: rel(base, agg["cross_entropy"][t][n]) for n, _ in CTX[1:]}
        agg["cross_entropy"][t]["temporal"] = {n: ce[t]["temporal"][n] for n, _ in CTX}
    for k in ("table_exact", "joint_exact", "table_jaccard", "joint_jaccard", "table_exact_kids", "joint_exact_kids", "table_jaccard_kids", "joint_jaccard_kids"):
        agg["set"][k] = mean_over(sm, lambda x: x[k])
    agg["set"]["temporal"] = sm["temporal"]
    res["summary"] = agg
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps({"base_rates": res["base_rates"], "summary": agg}, indent=1, default=float))


if __name__ == "__main__":
    main()
