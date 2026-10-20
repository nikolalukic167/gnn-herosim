#!/usr/bin/env python3
"""HEROSIM_EXCHANGE_FAIRSHARE against the filed shaped-link check (netns_check/netns_result.json, md5 9de512ea1f91f292edc86f56420fb0b6).
For each measured n = 2 and n = 4 point (n equal concurrent flows over one route, 50 and 250 MB) the fair-share engine is run on the case's own
route (hop capacities, route latency) and the model makespan is set beside the measured median makespan: plain fair share (size / share
+ latency) and fair share with the TCP start-up (HEROSIM_EXCHANGE_TCP: handshake and slow-start rounds per transfer, only the remaining bytes
share the links). Nothing is fitted. The wired-to-wired case is not quotable (the sender probe reaches 758 MB/s against a 117 MB/s limit) and is
printed but excluded from the summary, as in the filed check. A check of the code against measurements, not a quotable result."""
import hashlib
import json
import os
import statistics as st
import sys

import simpy

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from src.placement.network_fabric import FairShare, TCP_FRAME, TCP_MSS, tcp_startup_and_rest  # noqa: E402

NETNS = os.path.join(REPO, "docs/lineages/uncut_groups_s0_v1/netns_check/netns_result.json")
NETNS_MD5 = "9de512ea1f91f292edc86f56420fb0b6"
NOT_QUOTABLE = ("wired_wired",)


def model_makespan(case, size_mb, n, tcp):
    env = simpy.Environment()
    fs = FairShare(env)
    hops = [(f"l{i}", bw) for i, bw in enumerate(case["hop_mbs"])]
    lat, size = case["latency_s"], size_mb * 1024 ** 2
    if tcp:
        b, rest = tcp_startup_and_rest(size, case["bottleneck_mbs"], lat)
        scale = TCP_MSS / TCP_FRAME
    else:
        b, rest, scale = 0.0, size, 1.0
    done = []

    def one():
        if b > 0:
            yield env.timeout(b)
        if rest > 0:
            yield fs.start(hops, rest, scale)
        yield env.timeout(lat)
        done.append(env.now)

    for _ in range(n):
        env.process(one())
    env.run()
    return max(done)


def rows():
    raw = open(NETNS, "rb").read()
    if hashlib.md5(raw).hexdigest() != NETNS_MD5:
        raise SystemExit("FAIL LOUD: netns_result.json md5")
    d = json.loads(raw)
    cases = {c["case"]: c for c in d["cases"]}
    out = []
    for t in d["table"]:
        if t["flows"] < 2:
            continue
        meas = t["makespan"]["median"]
        p, q = (model_makespan(cases[t["case"]], t["size_mb"], t["flows"], tcp) for tcp in (False, True))
        out.append({"case": t["case"], "size_mb": t["size_mb"], "n": t["flows"], "measured_s": meas, "plain_fs_s": p, "tcp_fs_s": q,
                    "plain_err": p / meas - 1.0, "tcp_err": q / meas - 1.0, "quotable": t["case"] not in NOT_QUOTABLE})
    return out


def main():
    rs = rows()
    print(__doc__.strip(), "\n")
    print(f"{'case':20s} {'MB':>4s} {'n':>2s} {'measured s':>10s} {'plain FS s':>10s} {'err':>7s} {'TCP+FS s':>9s} {'err':>7s}")
    for r in rs:
        print(f"{r['case']:20s} {r['size_mb']:4d} {r['n']:2d} {r['measured_s']:10.2f} {r['plain_fs_s']:10.2f} {100 * r['plain_err']:+6.1f}% "
              f"{r['tcp_fs_s']:9.2f} {100 * r['tcp_err']:+6.1f}%" + ("" if r["quotable"] else "   (not quotable)"))
    qs = [r for r in rs if r["quotable"]]
    for key, lab in (("plain_err", "plain fair share"), ("tcp_err", "TCP + fair share")):
        e = [abs(r[key]) for r in qs]
        print(f"{lab}: {len(qs)} quotable points, median |error| {100 * st.median(e):.1f} %, max {100 * max(e):.1f} %, within 10 %: {sum(x <= 0.10 for x in e)}")


if __name__ == "__main__":
    main()
