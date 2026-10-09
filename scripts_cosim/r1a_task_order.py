#!/usr/bin/env python3
"""r1_attribution_v1 production training: array index -> (arm, config), one definition for the training sbatch.

Indices 0-20 (first half): the 21 slowest (arm, config) pairs, longest first -- cost = measured CPU epoch seconds of the arm (dry run, 859 train
datasets) x 1.8 for the width-128 configs g3-g5 (assumed). Indices 21-41 (second half, submitted later): the remaining 21 pairs with the engineered trio
(gnn_eng, twin_eng, mlp_same) first so they finish first, then the rest by the same cost order.

  r1a_task_order.py <index>      prints "<arm> <config>"
  r1a_task_order.py --table      prints the whole map
"""
from __future__ import annotations

import sys

EPOCH_S = {"gnn_eng": 10.6, "twin_eng": 8.9, "mlp_same": 9.5, "gnn_raw": 42.0, "twin_raw": 13.5, "gnn_eng_physmp": 10.8, "set_transformer": 16.5}
TRIO = ("gnn_eng", "twin_eng", "mlp_same")
FIRST_HALF = 21


def _cost(ag):
    return -EPOCH_S[ag[0]] * (1.8 if ag[1] >= 3 else 1.0)


def default_order():
    return sorted(((a, g) for a in EPOCH_S for g in range(6)), key=lambda ag: (_cost(ag), list(EPOCH_S).index(ag[0]), ag[1]))


def order():
    d = default_order()
    first, rest = d[:FIRST_HALF], d[FIRST_HALF:]
    trio = sorted((t for t in rest if t[0] in TRIO), key=lambda ag: (TRIO.index(ag[0]), ag[1]))
    return first + trio + [t for t in rest if t[0] not in TRIO]


def task_for(index: int):
    o = order()
    if not 0 <= index < len(o):
        raise IndexError(f"array index {index} beyond {len(o) - 1}")
    return o[index]


if __name__ == "__main__":
    if sys.argv[1:] == ["--table"]:
        for i, (a, g) in enumerate(order()):
            print(i, a, g)
    else:
        print(*task_for(int(sys.argv[1])))
