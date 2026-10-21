#!/usr/bin/env python3
"""The ILS-live / exact-live cell lists (uncut_groups_s0_v1, "Label search and exact search under the decision charge", with amendment A2): one line per cell,
<root relative to the uncut_groups dir> TAB <cell name> TAB <arm kind>. One list per arm root: 12 topologies x g0-g3 x moderate / heavy = 96 cells, then the identity
entries (never opened by the read) for the exact list. Printed to stdout; the file's md5 is filed with the pin before any cell.
  ils_live_cells.py exact                       > ils_live_cells_exact.txt      root ils_live/cells_exact, kind cd_exactS_ib0_live_k100
  ils_live_cells.py ils <budget 200|50|10>       > ils_live_cells_ils_b<B>.txt   root ils_live/cells_ils_b<B>, kind cd_ib0_ils[_p<B>]_k100   (A2: the budget ladder)"""
import sys

TOPOS = range(19537, 19549)
ILS_KIND = {"200": "cd_ib0_ils_k100", "50": "cd_ib0_ils_p50_k100", "10": "cd_ib0_ils_p10_k100"}


def root_kind(which, budget=None):
    if which == "exact":
        return "cells_exact", "cd_exactS_ib0_live_k100"
    return f"cells_ils_b{budget}", ILS_KIND[budget]


def lines(which, budget=None):
    root, arm = root_kind(which, budget)
    out = [f"ils_live/{root}\tcc40s{t}__{w}{r}__{arm}_s0\t{arm}" for r in ("moderate", "heavy") for t in TOPOS for w in ("g0", "g1", "g2", "g3")]
    if which == "exact":
        out.append("ils_live/ident_exact_bar\tcc40s19537__g0heavy__cd_ib0_inflightfull_k100_s0\tcd_ib0_inflightfull_k100")
        out.append("ils_live/ident_exact_gnn\tcc40s19537__g0heavy__ra_gnn_eng_combfl_nosplit_k100_s1\tra_gnn_eng_combfl_nosplit_k100")
    return out


if __name__ == "__main__":
    sys.stdout.write("\n".join(lines(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)) + "\n")
