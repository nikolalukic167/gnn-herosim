#!/usr/bin/env python3
"""FL gnn seeds 3-6 at 64 servers: DESCRIPTIVE read, no label, no verdict. Reuses the frozen reader's (reference-physics 2b571f99, blob 549643c8) `load`, `four`
and `family` unchanged. Cells: size64/cells_bar (bar cd_ib0_inflightfull_k100, seed 0), size64/cells_learned (FL gnn s1-2), size64/cells_learned_s36 (s3-6),
both charges (_k100 and _kr). Per seed, seeds 3-6 pooled, seeds 1-6 pooled, each vs the bar: the four tests (whole, after 360 s x moderate, heavy), median % of the
per-topology median over g0-g3, faster/12, raw exact Wilcoxon p, Holm within each four. usage: size64_s36_read.py <outdir>"""
import hashlib, json, os, subprocess, sys

RD = os.path.expanduser("~/gnn-herosim-wt/s6_reader_2b571f99")
want = subprocess.run(["git", "-C", RD, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
blob = subprocess.run(["git", "-C", RD, "rev-parse", "HEAD:scripts_cosim/uncut_arm_gate_read.py"], capture_output=True, text=True).stdout.strip()
if not want.startswith("2b571f99") or not blob.startswith("549643c8"):
    raise SystemExit(f"FAIL LOUD: reader {want} blob {blob}")
sys.path.insert(0, RD + "/scripts_cosim")
import uncut_arm_gate_read as G  # noqa: E402

U = "/share/nikola.lukic/uncut_groups"; OUT = sys.argv[1]
PRIM, BARK, KR = "ra_gnn_eng_combfl_nosplit", "cd_ib0_inflightfull", "_kr"
TOPOS = list(range(30129, 30141))
LIST = f"{U}/size64_s36/cells_learned_s36_list.txt"
gates = f"{U}/size64/cells_bar,{U}/size64/cells_learned,{U}/size64/cells_learned_s36"
cells, failed = G.load(gates, "_k100")
if failed:
    raise SystemExit(f"FAIL LOUD: {len(failed)} failed cells, e.g. {sorted(failed)[:3]}")
listed = {l.strip() for l in open(LIST) if l.strip()}
# the stripped loader dropped _k100, so compare on (arm, seed, topology, window, rung)
want_keys = set()
for n in listed:
    t, wr, arm, sd = n.split("__")[0][len("cc40s"):], n.split("__")[1], n.split("__")[2], int(n.rsplit("_s", 1)[1])
    arm = arm.rsplit("_s", 1)[0]
    arm = arm[: -len("_k100")] if arm.endswith("_k100") else arm
    w, r = wr[:2], wr[2:]
    want_keys.add((arm, sd, int(t), w, r))
got_keys = {k for k in cells if k[1] in (3, 4, 5, 6)}
if got_keys != want_keys or len(want_keys) != 768:
    raise SystemExit(f"FAIL LOUD: s3-6 cells differ from the filed list: missing {len(want_keys - got_keys)}, extra {len(got_keys - want_keys)}")
for k, s in cells.items():
    if s.get("servers") != 64:
        raise SystemExit(f"FAIL LOUD: {k} servers = {s.get('servers')}")
k100 = {k: v for k, v in cells.items() if not k[0].endswith(KR)}
kr = {(k[0][: -len(KR)],) + k[1:]: v for k, v in cells.items() if k[0].endswith(KR)}
kr.update({k: v for k, v in k100.items() if k[0] == BARK})
sets = {"seed3": (3,), "seed4": (4,), "seed5": (5,), "seed6": (6,), "seeds3-6": (3, 4, 5, 6), "seeds1-2": (1, 2), "seeds1-6": (1, 2, 3, 4, 5, 6)}
res, L = {}, ["=== FL gnn at 64 servers (topologies 30129-30140) vs cd_ib0_inflightfull_k100; DESCRIPTIVE, no verdict. median % of per-topology median over g0-g3; faster/12; raw exact Wilcoxon p; Holm within each four"]
for charge, base in (("_k100", k100), ("_kr", kr)):
    res[charge] = {}
    for name, sds in sets.items():
        es = G.family(G.four(base, PRIM, sds, BARK, (0,), TOPOS))
        res[charge][name] = es
        L.append(f"--- {charge} {name} (seeds {list(sds)})")
        for e in es:
            L.append(f"    {e['rung']:8s} {e['stat']:8s} median {e['median_pct']:+7.2f} %  faster {e['faster']:2d}/{e['n_topologies']}  p {e['p']:.4f}  p_holm {e['p_holm']:.4f}  cells {e['n_cells']}")
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(f"{OUT}/size64_s36_read.json", "w"), indent=1)
open(f"{OUT}/size64_s36_read.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
