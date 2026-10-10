"""scale_160_v1 vs accel_replica_v1: each arm's paired % against CD in its own gate, side by side (median and IQR over the topologies, per rung).
UNPAIRED, DESCRIPTIVE: the two gates use different test topologies, rungs and replica rules, so no contrast between them is computed.

  scale_vs_accel_side_by_side.py <scale_160 read.json> <accel read.json>      (outputs of scale_160_v1_read.py --out)
"""
import json
import statistics as st
import sys

ARMS = ("ra_gnn_eng", "ra_gnn_eng_physmp")
RUNGS = ("moderate", "heavy")


def cell(read: dict, arm: str, rung: str) -> str:
    t = read["main"]["tests"].get(f"{arm}|{rung}")
    v = sorted((t or {}).get("per_topology", {}).values())
    if len(v) < 2:
        return "-"
    q = st.quantiles(v, n=4, method="inclusive")
    return f"{st.median(v):+.2f} % [{q[0]:+.2f}, {q[2]:+.2f}] n={len(v)}"


def table(a: dict, b: dict) -> str:
    rows = ["unpaired, descriptive: paired % vs CD within each gate; median [IQR] over topologies",
            f"{'arm':20s} {'rung':9s} {'scale_160_v1 (first_compatible)':36s} accel_replica_v1 (fastest_compatible)"]
    for arm in ARMS:
        for rung in RUNGS:
            rows.append(f"{arm:20s} {rung:9s} {cell(a, arm, rung):36s} {cell(b, arm, rung)}")
    return "\n".join(rows)


if __name__ == "__main__":
    print(table(json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))))
