#!/usr/bin/env python3
"""Cell list of amendment A1 (TCP-on arm study with fair-share links, uncut_groups_s0_v1): prints the 2,016 grid cells and the 12 identity cells.
Grid: sizes 24 (19537-19548), 64 (30129-30140), 128 (30241-30252); bar cd_ib0_inflightfull_k100 s0 and FL gnn, FL MLP, twin (combfl, _k100) s1-2;
moderate and heavy; windows g0-g3; roots fairshare_arms/fs<size>; flags TCP=1 FAIRSHARE=1. Identity: both switches off at the new pin against the filed
switch-off cell (g0 heavy, first topology of the size), one per arm per size, roots fairshare_arms/ident<size>. Same tab-separated format as
tcp_arms_cells.txt."""
TOPOS = {24: range(19537, 19549), 64: range(30129, 30141), 128: range(30241, 30253)}
ARMS = (("cd_ib0_inflightfull_k100", (0,)), ("ra_gnn_eng_combfl_nosplit_k100", (1, 2)), ("ra_mlp_same_combfl_nosplit_k100", (1, 2)),
        ("ra_twin_eng_combfl_nosplit_k100", (1, 2)))
FILED = {  # size -> arm -> filed switch-off root (relative to the share's uncut_groups/ or anomaly/ as written)
    24: {"bar": "anomaly/dose_7cb483c5/cells_ref_heavy070", "gnn": "anomaly/dose_7cb483c5/cells_gnn_heavy070", "mlp": "comb/cells_comb",
         "twin": "anomaly/tcp_arms/off24"},
    64: {"bar": "size64/cells_bar", "gnn": "size64/cells_learned", "mlp": "size64/cells_desc", "twin": "size64/cells_twin"},
    128: {"bar": "size128/cells_bar", "gnn": "size128/cells_learned", "mlp": "size128/mlp_cells", "twin": "anomaly/tcp_arms/off128"},
}
SHORT = {"cd_ib0_inflightfull_k100": "bar", "ra_gnn_eng_combfl_nosplit_k100": "gnn", "ra_mlp_same_combfl_nosplit_k100": "mlp",
         "ra_twin_eng_combfl_nosplit_k100": "twin"}


def lines():
    out = []
    for size, topos in TOPOS.items():
        for rung in ("moderate", "heavy"):
            for t in topos:
                for w in ("g0", "g1", "g2", "g3"):
                    for arm, seeds in ARMS:
                        for sd in seeds:
                            out.append(f"fairshare_arms/fs{size}\tcc40s{t}__{w}{rung}__{arm}_s{sd}\tTCP=1\tFAIRSHARE=1")
    for size, topos in TOPOS.items():
        for arm, seeds in ARMS:
            out.append(f"fairshare_arms/ident{size}\tcc40s{topos[0]}__g0heavy__{arm}_s{seeds[0]}\tTCP=0\tFAIRSHARE=0\tvs {FILED[size][SHORT[arm]]}")
    return out


if __name__ == "__main__":
    print("\n".join(lines()))
