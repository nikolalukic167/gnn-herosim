"""Compare two serve_fast_probe pickles (flags A vs flags B) of the same cell.

    python3 scripts_cosim/serve_fast_compare.py A.pkl B.pkl [--bit]

Always reports: decisions, plans differing, steps, the max |logit| difference and every step whose unmasked argmax differs (with the margin
between A's top two logits). With --bit it also fails unless the graph hash, every column hash and every logit are byte-equal.
Exit 0 = the bar asked for passes (plans identical; with --bit also bit-identical).
"""
from __future__ import annotations

import argparse
import pickle
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--bit", action="store_true")
    args = ap.parse_args()
    A, B = pickle.load(open(args.a, "rb")), pickle.load(open(args.b, "rb"))
    if len(A) != len(B):
        print(f"FAIL: {len(A)} decisions vs {len(B)}")
        return 1
    n_plan = n_graph = n_cols = n_bytes = steps = flips = 0
    maxd = 0.0
    flip_info = []
    for i, (x, y) in enumerate(zip(A, B)):
        n_plan += x["plan"] != y["plan"]
        n_graph += x["g"] != y["g"]
        n_cols += x["cols"] != y["cols"]
        if x["task"] != y["task"]:
            print(f"decision {i}: different step order")
            return 1
        for la, lb in zip(x["logits"], y["logits"]):
            steps += 1
            n_bytes += la.tobytes() != lb.tobytes()
            if la.shape == lb.shape:
                maxd = max(maxd, float(np.max(np.abs(la.astype(np.float64) - lb.astype(np.float64)))) if la.size else 0.0)
                if la.size and int(np.argmax(la)) != int(np.argmax(lb)):
                    flips += 1
                    top = np.sort(la)[::-1]
                    flip_info.append((i, float(top[0] - top[1]) if top.size > 1 else float("nan")))
    print(f"decisions {len(A)}  steps {steps}  plans differing {n_plan}  graph hashes differing {n_graph}  column-hash lists differing {n_cols}  "
          f"logit blocks not byte-equal {n_bytes}  max|dlogit| {maxd:.3e}  unmasked-argmax flips {flips}")
    for i, m in flip_info[:10]:
        print(f"  flip at decision {i}: A's top-1 margin {m:.3e}")
    ok = n_plan == 0 and (not args.bit or (n_graph == 0 and n_cols == 0 and n_bytes == 0))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
