#!/usr/bin/env python3
"""The ILS-live / exact-live cell list (uncut_groups_s0_v1, "Label search and exact search under the decision charge"): one line per cell,
<root relative to the uncut_groups dir> TAB <cell name> TAB <arm kind>. 2 arms x 12 topologies x g0-g3 x moderate / heavy = 192 cells, then one bar identity
cell (never opened by the read). Printed to stdout; the file's md5 is filed with the pin before any cell.
  ils_live_cells.py > ils_live_cells.txt ; md5sum ils_live_cells.txt"""
import sys

TOPOS = range(19537, 19549)
ARMS = (("cells_ils", "cd_ib0_ils_k100"), ("cells_exact", "cd_exactS_ib0_live_k100"))


def lines():
    out = []
    for root, arm in ARMS:
        for r in ("moderate", "heavy"):
            for t in TOPOS:
                for w in ("g0", "g1", "g2", "g3"):
                    out.append(f"ils_live/{root}\tcc40s{t}__{w}{r}__{arm}_s0\t{arm}")
    out.append("ils_live/ident_bar\tcc40s19537__g0heavy__cd_ib0_inflightfull_k100_s0\tcd_ib0_inflightfull_k100")
    return out


if __name__ == "__main__":
    sys.stdout.write("\n".join(lines()) + "\n")
