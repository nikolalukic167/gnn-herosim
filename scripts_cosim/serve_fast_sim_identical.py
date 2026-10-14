"""SIM-IDENTICAL check for rp/gnn-serve-fast: two gate OUT dirs (flags A, flags B) over the same cells.

    python3 scripts_cosim/serve_fast_sim_identical.py OUT_A OUT_B

For every <cell>.summary.json present in both: all keys equal except the timing / provenance ones (decisionTiming, total_rtt_plus_inference =
total_rtt + the measured decision wall, run_provenance); and the per-task arrays (<cell>.tasks.npy, .tasks_bsz.npy, .tasks_lock.npy) byte-equal.
Exit 0 when every cell is identical.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

SKIP = {"decisionTiming", "total_rtt_plus_inference", "run_provenance", "code", "python_env", "wallclock_s", "host"}


def flat(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if k in SKIP:
                continue
            yield from flat(v, p + "/" + str(k))
    else:
        yield p, x


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def main(a: str, b: str) -> int:
    cells = sorted(os.path.basename(f)[: -len(".summary.json")] for f in glob.glob(os.path.join(a, "*.summary.json")))
    bad = 0
    for c in cells:
        fb = os.path.join(b, c + ".summary.json")
        if not os.path.exists(fb):
            print(f"{c}: missing in B")
            bad += 1
            continue
        da = dict(flat(json.load(open(os.path.join(a, c + ".summary.json")))))
        db = dict(flat(json.load(open(fb))))
        diff = [k for k in sorted(set(da) | set(db)) if da.get(k) != db.get(k)]
        arrs = [s for s in (".tasks.npy", ".tasks_bsz.npy", ".tasks_lock.npy")
                if os.path.exists(os.path.join(a, c + s)) and md5(os.path.join(a, c + s)) != md5(os.path.join(b, c + s))]
        if diff or arrs:
            bad += 1
            print(f"{c}: {len(diff)} keys differ {diff[:5]}  arrays differ {arrs}")
    print(f"{len(cells)} cells compared, {bad} not identical -> {'SIM-IDENTICAL' if bad == 0 and cells else 'DIFFERENT'}")
    return 0 if bad == 0 and cells else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
