"""Read-only feature extraction over ib0_confirm oracle_diag per-batch files (runs on datalab login node, stdlib only).
Writes two TSVs to stdout-specified paths under $OUT (a local-to-home tmp dir)."""
import glob, json, os, random, sys
from collections import Counter

ROOT = "/share/nikola.lukic/ib0_confirm/oracle_8cc890fa/oracle_diag"
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
rng = random.Random(0)


def struct(plan):
    nodes = Counter(p[0] for p in plan)
    plats = Counter((p[0], p[1]) for p in plan)
    n = len(plan)
    coloc = sum(c * (c - 1) // 2 for c in nodes.values())
    stack = sum(c * (c - 1) // 2 for c in plats.values())
    return len(nodes), len(plats), max(plats.values()), coloc, stack


bf = open(f"{OUT}/batches.tsv", "w")
pf = open(f"{OUT}/plans.tsv", "w")
bf.write("cell\trung\tt\tn\tplans\thand_label\tbest_label\thand_S\tbest_S\tbest_Srank\t"
         "h_nodes\th_plats\th_maxstack\th_coloc\th_stack\tb_nodes\tb_plats\tb_maxstack\tb_coloc\tb_stack\tnew_nodes_vs_hand\n")
pf.write("cell\trung\tbid\tn\tS\tlabel\tnodes\tplats\tmaxstack\tcoloc\tstack\tis_hand\n")
bid = 0
for d in sorted(glob.glob(f"{ROOT}/*")):
    cell = os.path.basename(d).split("__")[0]
    rung = "heavy" if "heavy" in d else "moderate"
    for fn in sorted(glob.glob(f"{d}/*.jsonl")):
        with open(fn) as fh:
            for line in fh:
                r = json.loads(line)
                tab = r.get("table") or []
                if not tab:
                    continue
                hand = [tuple(x) for x in r["s_plan"]]
                best = [tuple(x) for x in r["best"]]
                Ss = {tuple(map(tuple, p)): s for p, s, _ in tab}
                hS = Ss.get(tuple(hand))
                bS = Ss.get(tuple(best))
                order = sorted(s for _, s, _ in tab)
                brank = 1 + sum(1 for s in order if bS is not None and s < bS - 1e-12)
                hs, bs = struct(hand), struct(best)
                new_nodes = len({p[0] for p in best} - {p[0] for p in hand})
                bf.write("\t".join(map(str, [cell, rung, round(r["t"], 3), r["n"], r["plans"], r["s_label"], r["best_label"],
                                             hS, bS, brank, *hs, *bs, new_nodes])) + "\n")
                if len(tab) >= 2 and rng.random() < 0.25:
                    rows = tab if len(tab) <= 40 else rng.sample(tab, 39) + [t for t in tab if [tuple(x) for x in t[0]] == hand][:1]
                    for p, s, lab in rows:
                        pl = [tuple(x) for x in p]
                        pf.write("\t".join(map(str, [cell, rung, bid, r["n"], s, lab, *struct(pl), int(pl == hand)])) + "\n")
                bid += 1
bf.close(); pf.close()
print("batches", bid)
