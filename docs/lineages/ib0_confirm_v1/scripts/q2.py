import pandas as pd, numpy as np, sys
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
b = pd.read_csv(sys.argv[1]+"/batches.tsv", sep="\t"); b["bid"]=np.arange(len(b))
tmax=b.groupby(["cell","rung"]).t.transform("max"); b["tf"]=b.t/tmax
p = pd.read_csv(sys.argv[1]+"/plans.tsv", sep="\t")
p = p.merge(b[["bid","tf","hand_label","hand_S","plans"]], on="bid")
p = p[p.groupby("bid").is_hand.transform("max")==1]
h = p[p.is_hand==1].drop_duplicates("bid").set_index("bid")
for c in ["nodes","plats","maxstack","coloc","stack"]:
    p["d_"+c] = p[c].values - h.loc[p.bid, c].values
scale = p.hand_label.clip(lower=1e-3)
p["y"] = (p.label - p.hand_label)/scale
p["dS"] = (p.S - p.hand_S)/scale
p["Srank"] = p.groupby("bid").S.rank(method="min")
feats = ["dS","Srank","n","plans","tf","nodes","plats","maxstack","coloc","stack","d_nodes","d_plats","d_maxstack","d_coloc","d_stack"]
cells = sorted(p.cell.unique()); test=cells[-2:]
for rung,g in p.groupby("rung"):
  for name,sub in [("all",g),("tf>=0.05",g[g.tf>=.05])]:
    tr, te = sub[~sub.cell.isin(test)], sub[sub.cell.isin(test)]
    lin = LinearRegression().fit(tr[["dS"]], tr.y)
    r_tr = tr.y - lin.predict(tr[["dS"]]); r_te = te.y - lin.predict(te[["dS"]])
    def r2(y,f): return 1-((y-f)**2).sum()/((y-y.mean())**2).sum()
    gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, random_state=0).fit(tr[feats], r_tr)
    fte = gb.predict(te[feats])
    gby = HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, random_state=0).fit(tr[feats], tr.y)
    yte = gby.predict(te[feats])
    print(f"== {rung} {name}: train rows {len(tr)}, test rows {len(te)} (test cells {test})")
    print(f"   R2 of y from dS alone (test) {r2(te.y, lin.predict(te[['dS']])):.3f};  residual R2 from features (test) {r2(r_te, fte):.3f}")
    # importance via permutation on residual model
    base = r2(r_te, fte); imp={}
    rng=np.random.default_rng(0)
    for f in feats:
        X=te[feats].copy(); X[f]=rng.permutation(X[f].values); imp[f]=base-r2(r_te, gb.predict(X))
    print("   top features (drop in residual R2):", {k:round(v,3) for k,v in sorted(imp.items(), key=lambda kv:-kv[1])[:5]})
    # decision value on held-out batches
    te = te.assign(pred_gb=yte, S_=te.S)
    reg_S=[]; reg_gb=[]; w=[]
    for bid, gg in te.groupby("bid"):
        if len(gg)<2: continue
        mn = gg.label.min(); hl=gg.hand_label.iloc[0]
        reg_S.append(gg.label[gg.S_.idxmin()]-mn); reg_gb.append(gg.label[gg.pred_gb.idxmin()]-mn); w.append(hl)
    W=np.sum(w)
    print(f"   held-out decision regret (sampled plans, % of hand label s): argmin S {np.sum(reg_S)/W*100:.2f}%   argmin GBM {np.sum(reg_gb)/W*100:.2f}%   batches {len(w)}")
