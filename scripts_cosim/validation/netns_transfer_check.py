#!/usr/bin/env python3
"""Hardware-component check of the simulator's transfer model: real TCP over a shaped veth link vs the code's charge.

What this checks
    That TCP over a tc-shaped link behaves like ``size / bottleneck + route latency`` (the simulator's charge) and
    how concurrent flows on one uplink compare with the two sharing rules the simulator has. It does NOT show that real
    cellular or Wi-Fi links have these rates; the rates are the simulator's own (ACCESS_CLASSES).

Code under check: reference-physics 02abe7b7 (this branch is cut from it; the sha, base and file md5s are in the output).
    rates       src/placement/network_fabric.py:80-87 ACCESS_CLASSES (MB/s, 1 MB = 1024**2 bytes), directional via
                directed_bandwidth; wired 117/117, wifi 7..14 symmetric, cellular out 4 / in 75.
    charge      pipelined model (HEROSIM_TRANSFER_MODEL=pipelined, what every uncut_groups cell runs):
                transmission_hops(n)=1, bottleneck = min over EVERY hop of fabric.hops(src, dst) (backbone links included),
                charge = size / (bottleneck * 1024**2) + fabric.route_latency(src, dst)   [infrastructure.py:1121-1128,
                :1314-1316 for peer exchange]. Computed in-process from one real gate cfg (topology seed 19537) by
                prepare_infrastructure_for_real_simulation + NetworkFabric; never from a table.
    ingress     every link is a simpy Resource(capacity=1); a flow holds its route's pipes for the whole ``hold`` (:1541-1566),
                so n simultaneous flows on one uplink are served one at a time: completions at k*hold + latency.
                PASS criterion: makespan within 10 % of n*hold + latency. The per-flow split (FIFO) vs TCP fairness is a
                disclosed non-match; the mean against the fair-share prediction (all flows at n*hold + latency) is reported.
    peer        exchange is additive and uncontended (:1314-1316, no pipe): every flow at hold + latency. n = 1: PASS if the
                median is within 10 % of the charge for 50, 250, 500 MB. 4 MB: deviation from the charge and from
                charge + 1 RTT, no verdict. n >= 2: no verdict, measured / charged per flow and for the makespan.
    store_forward is NOT tested; no uncut_groups cell uses it.

Topology: two network namespaces joined by one veth pair created directly inside them (no root-namespace interface is
touched). The sender's veth carries netem (one-way route latency) as root with tbf (bottleneck rate) as its child; the receiver's
veth carries netem with the same one-way delay, so RTT = 2 * latency. tc and sysctl run only inside the namespaces. Each case is
one real (src, dst) server pair of the cfg, the pair's actual bottleneck and latency; a class combination with no pair in the
cfg is reported and dropped, never synthesised.

Timing: t0 = in the sender just before connect(), t1 = the receiver has read the last byte; both CLOCK_MONOTONIC on one host,
so the handshake and slow start are inside it (hence the charge + 1 RTT row). Payload digests are checked AFTER t1. TCP
RetransSegs of the sender namespace are read before and after every run; a run with retransmits is flagged, not dropped.
Unshaped probe first (500 MB, no tbf/netem): goodput under 10 * 117 MB/s means the Python sender limits the wired case and
the wired rows are marked NOT QUOTABLE.

Run (root, Linux, iproute2; use the repo's python so simpy imports):
    PYTHONPATH=<this worktree> HEROSIM_TRANSFER_MODEL=pipelined pipenv run python3 netns_transfer_check.py --cfg cc40s19537.json --out result.json
    --plan: print the cases and the planned grid (no namespaces, no transfers).   --cleanup-stale: remove leftover namespaces.
Cleanup: namespaces herosim_val_<pid>_a/_b are removed in a finally block, by atexit and on SIGINT/SIGTERM/SIGHUP.
Grid: n = 1 at 4/50/250/500 MB; n = 2 and 4 at 50 and 250 MB; 3 repetitions each.
"""
import argparse, atexit, hashlib, json, os, platform, signal, socket, statistics, subprocess, sys, tempfile, time

MB = 1024 * 1024
PORT = 5201
SIZES_N1 = (4, 50, 250, 500)
SIZES_NN = (50, 250)
FLOWS_NN = (2, 4)
TOL = 0.10
CLASS_PAIRS = (("wired_wired", "wired", "wired"), ("cellular_to_wired", "cellular", "wired"), ("wired_to_cellular", "wired", "cellular"),
               ("wifi_wifi_low", "wifi", "wifi"), ("wifi_wifi_high", "wifi", "wifi"))
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def sh(cmd, check=True):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"{cmd!r} failed ({r.returncode}): {r.stderr.strip()}")
    return r


def charge_cases(cfg_path):
    """Pick one real server pair per case from the cfg and compute its bottleneck / hops / latency from the fabric."""
    sys.path.insert(0, ROOT)
    if os.environ.get("HEROSIM_TRANSFER_MODEL") != "pipelined":
        raise SystemExit("HEROSIM_TRANSFER_MODEL=pipelined is required (the model every uncut_groups cell runs)")
    import simpy
    from src.executesimulation import prepare_infrastructure_for_real_simulation
    from src.placement.network_fabric import NetworkFabric, transfer_model, transmission_hops
    assert transfer_model() == "pipelined"
    cfg = json.load(open(cfg_path)); seed = cfg["network"]["topology"]["seed"]
    inf = prepare_infrastructure_for_real_simulation(cfg, seed=seed)
    lt = inf["link_topology"]; fab = NetworkFabric(simpy.Environment(), lt)
    ac = {n: v["class"] for n, v in lt["access_classes"].items()}; servers = sorted((n for n in ac if not n.startswith("client")), key=lambda s: int(s[4:]))
    nmap = {n["node_name"]: n["network_map"] for n in inf["nodes"]}
    pairs = {}
    for a in servers:
        for b in servers:
            if a != b and fab.has_route(a, b):
                hops = fab.hops(a, b); bw = min(x for _k, x in hops); lat = fab.route_latency(a, b)
                nm = nmap[b].get(a)
                if nm is not None and abs(float(nm) - lat) > 1e-9:
                    raise SystemExit(f"network_map latency {nm} != route_latency {lat} for {a}->{b}")
                pairs.setdefault((ac[a], ac[b]), []).append({"src": a, "dst": b, "hops": len(hops), "route": [k for k, _x in hops], "hop_mbs": [x for _k, x in hops], "bottleneck_mbs": bw,
                                                             "latency_s": lat, "transmission_hops": transmission_hops(len(hops))})
    cases, dropped = [], []
    for label, cs, cd in CLASS_PAIRS:
        cand = pairs.get((cs, cd), [])
        if not cand:
            dropped.append({"case": label, "reason": f"no ({cs} -> {cd}) server pair in {os.path.basename(cfg_path)}"}); continue
        if label.startswith("wifi_wifi"):
            target = (min if label.endswith("low") else max)(c["bottleneck_mbs"] for c in cand); cand = [c for c in cand if c["bottleneck_mbs"] == target]
        cand = sorted(cand, key=lambda c: (c["latency_s"], c["src"], c["dst"])); pick = cand[(len(cand) - 1) // 2]   # median latency among the candidates
        cases.append({"case": label, "src_class": cs, "dst_class": cd, "candidates": len(cand), **pick})
    return {"seed": seed, "n_servers": len(servers), "class_counts": {c: sum(1 for s in servers if ac[s] == c) for c in ("wired", "wifi", "cellular")}}, cases, dropped


def predictions(case, size_mb, n):
    hold = case["transmission_hops"] * size_mb / case["bottleneck_mbs"]; lat = case["latency_s"]
    return {"hold_s": hold, "latency_s": lat, "single_charge_s": hold + lat, "single_charge_plus_rtt_s": hold + 3 * lat,
            "fifo_s": [k * hold + lat for k in range(1, n + 1)], "additive_s": [hold + lat] * n, "fair_share_s": [n * hold + lat] * n}


def grid():
    return [(s, 1) for s in SIZES_N1] + [(s, n) for n in FLOWS_NN for s in SIZES_NN]


def print_plan(cfg_path, reps):
    meta, cases, dropped = charge_cases(cfg_path)
    print(meta, "dropped:", dropped)
    print(f"{'case':18s}{'pair':>14s}{'hops':>5s}{'MB/s':>7s}{'lat ms':>8s}{'size':>5s}{'n':>3s}{'charge s':>10s}{'n*hold+lat':>12s}{'wall s':>8s}")
    tot = 0.0
    for c in cases:
        for s, n in grid():
            p = predictions(c, s, n); w = reps * (n * p["hold_s"] + 3 * p["latency_s"] + 2.0 + 0.2 * n); tot += w
            print(f"{c['case']:18s}{c['src'] + '>' + c['dst']:>14s}{c['hops']:5d}{c['bottleneck_mbs']:7.2f}{1000 * c['latency_s']:8.2f}{s:5d}{n:3d}{p['single_charge_s']:10.3f}{p['fifo_s'][-1]:12.3f}{w:8.0f}")
    print(f"estimated wall time {tot / 3600:.2f} h at {reps} reps (+ probe)")


class Lab:
    def __init__(self):
        self.tag = f"herosim_val_{os.getpid()}"; self.a, self.b = self.tag + "_a", self.tag + "_b"; self.live = False
        atexit.register(self.down)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, lambda s, f: (self.down(), sys.exit(128 + s)))

    def up(self):
        self.live = True   # set first: a partial setup must still be torn down
        sh(["ip", "netns", "add", self.a]); sh(["ip", "netns", "add", self.b])
        sh(f"ip -n {self.a} link add va type veth peer name vb netns {self.b}")   # created in the namespaces, never in the root one
        for ns, dev, ip in ((self.a, "va", "10.77.0.1/24"), (self.b, "vb", "10.77.0.2/24")):
            sh(f"ip -n {ns} addr add {ip} dev {dev}"); sh(f"ip -n {ns} link set {dev} up"); sh(f"ip -n {ns} link set lo up")
            sh(f"ip netns exec {ns} sysctl -qw net.ipv4.tcp_rmem='4096 131072 67108864' net.ipv4.tcp_wmem='4096 131072 67108864' net.core.rmem_max=67108864 net.core.wmem_max=67108864")

    def sysctls(self):
        return {ns: {k: sh(f"ip netns exec {ns} sysctl -n {k}").stdout.strip() for k in ("net.ipv4.tcp_rmem", "net.ipv4.tcp_wmem", "net.core.rmem_max", "net.core.wmem_max", "net.ipv4.tcp_congestion_control")} for ns in (self.a, self.b)}

    def unshape(self):
        for ns, dev in ((self.a, "va"), (self.b, "vb")): sh(f"ip netns exec {ns} tc qdisc del dev {dev} root", check=False)

    def shape(self, rate_mbs, latency_s):
        """netem one-way delay on both ends (RTT = 2 * latency); tbf (rate) as netem's child on the sender end only."""
        ms = f"{latency_s * 1000:.4f}ms"; rate_bit = int(rate_mbs * MB * 8); burst = max(65536, int(rate_mbs * MB / 100))   # burst >= rate/HZ
        self.unshape()
        sh(f"ip netns exec {self.a} tc qdisc add dev va root handle 1: netem delay {ms} limit 200000")
        sh(f"ip netns exec {self.a} tc qdisc add dev va parent 1:1 handle 10: tbf rate {rate_bit}bit burst {burst} limit 64mb")
        sh(f"ip netns exec {self.b} tc qdisc add dev vb root handle 1: netem delay {ms} limit 200000")

    def retrans(self):
        t = sh(f"ip netns exec {self.a} cat /proc/net/snmp").stdout.splitlines(); tc = [l.split() for l in t if l.startswith("Tcp:")]
        return int(dict(zip(tc[0], tc[1]))["RetransSegs"])

    def down(self):
        if not self.live: return
        self.live = False
        for ns in (self.a, self.b): subprocess.run(["ip", "netns", "del", ns], capture_output=True)


def cleanup_stale():
    for line in sh("ip netns list", check=False).stdout.splitlines():
        name = line.split()[0] if line.split() else ""
        if name.startswith("herosim_val_"): subprocess.run(["ip", "netns", "del", name], capture_output=True); print("removed", name)


def block(size_mb, flow):
    seed = f"herosim_val_v1:{size_mb}:{flow}".encode(); out = bytearray(); i = 0
    while len(out) < MB:
        out += hashlib.sha256(seed + i.to_bytes(8, "big")).digest(); i += 1
    return bytes(out[:MB])


def digest(size_mb, flow):
    h = hashlib.sha256(); b = block(size_mb, flow)
    for _ in range(size_mb): h.update(b)
    return h.hexdigest()


def recv_worker(n, size_mb, ready_path, out_path):
    import threading
    want = {k: digest(size_mb, k) for k in range(n)}
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); srv.bind(("10.77.0.2", PORT)); srv.listen(16)
    open(ready_path, "w").write("1"); res = [None] * n; ths = []

    def one(c, k):
        hdr = b""
        while len(hdr) < 4: hdr += c.recv(4 - len(hdr))
        flow = int.from_bytes(hdr, "big"); buf = bytearray(size_mb * MB); mv = memoryview(buf); got = 0
        while got < len(buf):
            r = c.recv_into(mv[got:], min(1 << 20, len(buf) - got))
            if not r: break
            got += r
        t1 = time.clock_gettime(time.CLOCK_MONOTONIC)   # digest is checked after the timestamp
        res[k] = {"flow": flow, "t1": t1, "bytes": got, "ok": got == len(buf) and hashlib.sha256(buf).hexdigest() == want[flow]}
        c.send(b"k"); c.close()
    for k in range(n):
        c = srv.accept()[0]; t = threading.Thread(target=one, args=(c, k)); t.start(); ths.append(t)
    [t.join() for t in ths]; json.dump(res, open(out_path, "w"))


def send_worker(flow, size_mb, start_at, out_path):
    b = memoryview(block(size_mb, flow))
    while time.clock_gettime(time.CLOCK_MONOTONIC) < start_at: time.sleep(0.0003)
    t0 = time.clock_gettime(time.CLOCK_MONOTONIC)
    s = socket.socket(); s.connect(("10.77.0.2", PORT)); s.sendall(flow.to_bytes(4, "big"))
    for _ in range(size_mb): s.sendall(b)
    s.recv(1); s.close(); json.dump({"flow": flow, "t0": t0}, open(out_path, "w"))


def run_point(lab, size_mb, n, tmp, timeout):
    rp, op = f"{tmp}/ready", f"{tmp}/recv.json"
    for p in (rp, op):
        if os.path.exists(p): os.remove(p)
    me = [sys.executable, os.path.abspath(__file__)]
    rcv = subprocess.Popen(["ip", "netns", "exec", lab.b] + me + ["--_recv", str(n), str(size_mb), rp, op])
    while not os.path.exists(rp):
        if rcv.poll() is not None: raise RuntimeError("receiver died")
        time.sleep(0.02)
    r0 = lab.retrans(); start_at = time.clock_gettime(time.CLOCK_MONOTONIC) + 1.0 + 0.2 * n
    snd = [subprocess.Popen(["ip", "netns", "exec", lab.a] + me + ["--_send", str(k), str(size_mb), repr(start_at), f"{tmp}/send{k}.json"]) for k in range(n)]
    for p in snd + [rcv]:
        if p.wait(timeout=timeout): raise RuntimeError("worker failed")
    retrans = lab.retrans() - r0
    recv = json.load(open(op)); t0 = {k: json.load(open(f"{tmp}/send{k}.json"))["t0"] for k in range(n)}
    fl = sorted(({"flow": r["flow"], "completion_s": r["t1"] - t0[r["flow"]], "ok": r["ok"]} for r in recv), key=lambda r: r["completion_s"])
    return fl, retrans


def med_range(xs):
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs)}


def summarise(results, wired_quotable):
    rows = []; keys = sorted({(r["case"], r["size_mb"], r["flows"]) for r in results}, key=lambda k: (k[0], k[2], k[1]))
    for case, s, n in keys:
        rs = [r for r in results if (r["case"], r["size_mb"], r["flows"]) == (case, s, n)]; p = rs[0]["pred"]
        comp = [r["completions_s"] for r in rs]; make = med_range([c[-1] for c in comp]); first = med_range([c[0] for c in comp])
        sorted_med = [statistics.median(c[i] for c in comp) for i in range(n)]
        row = {"case": case, "size_mb": s, "flows": n, "reps": len(rs), "flagged_retransmit_runs": sum(1 for r in rs if r["retransmits"] > 0), "digests_ok": all(r["all_digests_ok"] for r in rs),
               "charge_s": p["single_charge_s"], "charge_plus_rtt_s": p["single_charge_plus_rtt_s"], "makespan": make, "per_flow_sorted_median_s": sorted_med,
               "ingress_fifo_makespan_s": p["fifo_s"][-1], "peer_additive_s": p["additive_s"][0], "fair_share_s": p["fair_share_s"][0],
               "makespan_over_ingress_pred": make["median"] / p["fifo_s"][-1], "makespan_over_peer_charge": make["median"] / p["additive_s"][0],
               "per_flow_over_peer_charge": [c / p["additive_s"][0] for c in sorted_med], "per_flow_over_fifo": [c / f for c, f in zip(sorted_med, p["fifo_s"])],
               "mean_over_fair_share": statistics.mean(sorted_med) / p["fair_share_s"][0]}
        quotable = wired_quotable or not case.startswith("wired_wired")
        if n == 1:
            row["dev_vs_charge_pct"] = 100 * (first["median"] / p["single_charge_s"] - 1); row["dev_vs_charge_plus_rtt_pct"] = 100 * (first["median"] / p["single_charge_plus_rtt_s"] - 1)
            row["peer_n1"] = "NOT QUOTABLE (sender limit)" if not quotable else ("DISCLOSE (4 MB, small-transfer handshake bias)" if s == 4 else ("PASS" if abs(row["dev_vs_charge_pct"]) <= 100 * TOL else "FAIL"))
        else:
            dev = 100 * (make["median"] / p["fifo_s"][-1] - 1)
            row["ingress_makespan_dev_pct"] = dev
            row["ingress_makespan"] = "NOT QUOTABLE (sender limit)" if not quotable else ("PASS" if abs(dev) <= 100 * TOL else "FAIL")
            row["peer_nn"] = "DISCLOSE (measured/charged, no verdict)"; row["ingress_per_flow_split"] = "DISCLOSE (FIFO vs TCP fairness)"
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cfg"); ap.add_argument("--out"); ap.add_argument("--plan", action="store_true"); ap.add_argument("--cleanup-stale", action="store_true")
    ap.add_argument("--reps", type=int, default=3); ap.add_argument("--cases"); ap.add_argument("--_recv", nargs=4); ap.add_argument("--_send", nargs=4)
    a = ap.parse_args()
    if a._recv: return recv_worker(int(a._recv[0]), int(a._recv[1]), a._recv[2], a._recv[3])
    if a._send: return send_worker(int(a._send[0]), int(a._send[1]), float(a._send[2]), a._send[3])
    if a.cleanup_stale: return cleanup_stale()
    if not a.cfg: raise SystemExit("--cfg required")
    if a.plan: return print_plan(a.cfg, a.reps)
    if os.geteuid() != 0 or not a.out: raise SystemExit("needs root and --out")
    git = lambda *x: sh(["git", "-C", ROOT, *x]).stdout.strip()
    if git("status", "--porcelain", "--", "scripts_cosim/validation"): raise SystemExit("FAIL LOUD: the harness has uncommitted changes; commit it first (provenance)")
    meta, cases, dropped = charge_cases(a.cfg)
    if a.cases: cases = [c for c in cases if c["case"] in a.cases.split(",")]
    prov = {"harness_head": git("rev-parse", "HEAD"), "harness_base": "02abe7b7", "base_is_ancestor": sh(["git", "-C", ROOT, "merge-base", "--is-ancestor", "02abe7b7", "HEAD"], check=False).returncode == 0,
            "script_md5": md5(os.path.abspath(__file__)), "network_fabric_md5": md5(f"{ROOT}/src/placement/network_fabric.py"), "infrastructure_md5": md5(f"{ROOT}/src/placement/infrastructure.py"),
            "cfg": os.path.abspath(a.cfg), "cfg_md5": md5(a.cfg), "kernel": platform.release(), "iproute2": sh("ip -V").stdout.strip(), "tc": sh("tc -V").stdout.strip(), "host": platform.node(),
            "python": sys.version.split()[0], "transfer_model": os.environ.get("HEROSIM_TRANSFER_MODEL"), "store_forward_tested": False}
    lab = Lab(); results = []; probe = None
    def save(rows=None):
        json.dump({"provenance": prov, "cfg_meta": meta, "cases": cases, "dropped_cases": dropped, "sysctls": sysctls, "unshaped_probe": probe, "grid": {"n1_sizes_mb": SIZES_N1, "nn_sizes_mb": SIZES_NN, "nn_flows": FLOWS_NN, "reps": a.reps},
                   "tolerance": TOL, "runs": results, "table": rows}, open(a.out, "w"), indent=1)
    try:
        lab.up(); sysctls = lab.sysctls()
        with tempfile.TemporaryDirectory() as tmp:
            fl, rt = run_point(lab, 500, 1, tmp, 120); probe = {"size_mb": 500, "goodput_mbs": 500 / fl[0]["completion_s"], "retransmits": rt, "digest_ok": fl[0]["ok"], "limit_mbs": 10 * 117.0}
            wired_ok = probe["goodput_mbs"] >= probe["limit_mbs"]; probe["wired_quotable"] = wired_ok; print("probe", probe, flush=True)
            for c in cases:
                lab.shape(c["bottleneck_mbs"], c["latency_s"])
                for s, n in grid():
                    pr = predictions(c, s, n)
                    for rep in range(a.reps):
                        fl, rt = run_point(lab, s, n, tmp, 120 + 3 * pr["fifo_s"][-1])
                        comp = [f["completion_s"] for f in fl]
                        results.append({"case": c["case"], "size_mb": s, "flows": n, "rep": rep, "completions_s": comp, "all_digests_ok": all(f["ok"] for f in fl), "retransmits": rt, "pred": pr})
                        print(c["case"], s, "MB x", n, "rep", rep, [round(x, 2) for x in comp], "retrans", rt, flush=True); save()
        rows = summarise(results, wired_ok); save(rows)
        for r in rows: print(r["case"], r["size_mb"], r["flows"], r.get("peer_n1") or r.get("ingress_makespan"), flush=True)
    finally:
        lab.down()


if __name__ == "__main__":
    main()
