"""transfer_physics_v1 read: per condition x rung, each arm's latency / queue share / exchange (median over topologies of
per-topology means) and the paired % vs CD (median over topologies of the per-topology median over windows x seeds)."""
import glob, json, os, statistics as st, sys
from scipy.stats import wilcoxon
B = "/home/nikola.lukic/gnn-herosim/simulation_data/"
EXCLUDE = {int(x) for x in os.environ.get("TP1_EXCLUDE", "").split(",") if x}  # sensitivity: e.g. TP1_EXCLUDE=9485
topos = [t for t in json.load(open(B + "client_local_v1/inputs_so_server/selected.json"))["topologies"] if t not in EXCLUDE]
COND = {"sf_held": ["client_local_v1/gate_so_server", "small_batch_so_v1/gate"],
        "pipe": ["transfer_physics_v1/gate_pipe"], "release": ["transfer_physics_v1/gate_release"],
        "pipe_release": ["transfer_physics_v1/gate_pipe_release"]}
ARMS = ["reactive", "selfpredict", "locality", "batched", "cd", "so1load_selfref"]
SEEDS = {"so1load_selfref": (1, 2)}
def load(dirs):
    s = {}
    for d in dirs:
        for f in glob.glob(B + d + "/*.summary.json"):
            r = json.load(open(f)); k = r["arm"].split("__")[2].rsplit("_s", 1)[0]
            s[(int(r["topology"]), r["window"], k, int(r["checkpoint_seed"]))] = r
    return s
out = {}
for cond, dirs in COND.items():
    s = load(dirs); out[cond] = {}
    for rung in ("20", "30", "50"):
        ws = [f"g{i}x{rung}" for i in range(4)]
        rows = {}
        for a in ARMS:
            seeds = SEEDS.get(a, (0,))
            lat, qs, ex, pc = [], [], [], []
            for t in topos:
                rs = [s[(t, w, a, sd)] for w in ws for sd in seeds if (t, w, a, sd) in s]
                if rs:
                    lat.append(st.mean(r["averageElapsedTime"] for r in rs))
                    qs.append(st.mean(r["averageQueueTime"] / r["averageElapsedTime"] for r in rs))
                    ex.append(st.mean(r["totalPeerExchangeTime"] / r["num_tasks"] for r in rs))
                p = [100 * (s[(t, w, a, sd)]["averageElapsedTime"] - s[(t, w, "cd", 0)]["averageElapsedTime"]) / s[(t, w, "cd", 0)]["averageElapsedTime"]
                     for w in ws for sd in seeds if (t, w, a, sd) in s and (t, w, "cd", 0) in s]
                if p: pc.append(st.median(p))
            p_w = wilcoxon(pc).pvalue if a != "cd" and len(pc) > 5 and any(pc) else None
            rows[a] = dict(n=len(lat), lat=st.median(lat), qshare=st.median(qs), exch=st.median(ex),
                           vs_cd=st.median(pc) if pc else None, faster=sum(x < 0 for x in pc), n_pc=len(pc), p=p_w)
        out[cond][rung] = rows
        print(f"\n== {cond} x{rung}")
        for a, r in sorted(rows.items(), key=lambda kv: kv[1]["lat"]):
            vs = "" if a == "cd" else f"vs CD {r['vs_cd']:+6.1f}% ({r['faster']}/{r['n_pc']}, p={r['p']:.4f})"
            print(f"  {a:16s} n={r['n']:2d} lat {r['lat']:8.2f}s  qshare {r['qshare']:.2f}  exch {r['exch']:.2f}s  {vs}")
json.dump(out, open(B + "transfer_physics_v1/tp1_read" + ("_excl" if EXCLUDE else "") + ".json", "w"), indent=1)
