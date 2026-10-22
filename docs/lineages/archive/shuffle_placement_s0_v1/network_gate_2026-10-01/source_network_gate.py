"""Run a real TCP shuffle in isolated Linux network namespaces.

Launch through unshare --user --map-root-user --net. No host networking is changed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import threading
import time

import numpy as np

from scripts_cosim.shuffle_placement_s0 import digest, maxmin_rates, recv_exact, replay, route, topology, write_json


def command(argv):
    return subprocess.run([str(x) for x in argv], check=True, capture_output=True, text=True).stdout


def rpc(path, message):
    with socket.socket(socket.AF_UNIX) as sock:
        sock.settimeout(120)
        sock.connect(str(path))
        sock.sendall(json.dumps(message).encode() + b"\n")
        with sock.makefile("rb") as stream:
            response = json.loads(stream.readline())
    if "error" in response:
        raise RuntimeError(response["error"])
    return response


def worker(host, directory, port):
    directory = Path(directory)
    progress, completed, blocks, errors = {}, {}, {}, []
    mutex, compute = threading.Lock(), threading.Lock()
    stop = threading.Event()

    def receive(conn):
        try:
            with conn:
                conn.settimeout(90)
                header_size = struct.unpack("!I", recv_exact(conn, 4))[0]
                header = json.loads(recv_exact(conn, header_size))
                key = header["key"]
                payload = bytearray(header["bytes"])
                view = memoryview(payload)
                offset = 0
                start = time.perf_counter()
                with mutex:
                    progress[key] = {"received": 0, "bytes": len(payload), "start": start, "end": None}
                while offset < len(payload):
                    count = conn.recv_into(view[offset:offset + 262144])
                    if not count:
                        raise RuntimeError("truncated flow")
                    offset += count
                    with mutex:
                        progress[key]["received"] = offset
                end = time.perf_counter()
                if hashlib.sha256(payload).hexdigest() != header["sha256"]:
                    raise RuntimeError("payload SHA mismatch")
                with mutex:
                    progress[key]["end"] = end
                if header.get("aggregate"):
                    group = header["group"]
                    with mutex:
                        blocks.setdefault(group, []).append(payload)
                        ready = len(blocks[group]) == 4
                        inputs = blocks.pop(group) if ready else None
                    if ready:
                        waiting = time.perf_counter()
                        with compute:
                            began = time.perf_counter()
                            records = np.concatenate([np.frombuffer(b, dtype=np.uint32).reshape(-1, 2) for b in inputs])
                            totals = np.bincount(records[:, 0], weights=records[:, 1], minlength=4096)
                            record = {"ready": waiting, "compute_start": began, "end": time.perf_counter(),
                                      "rows": len(records), "value_sum": int(totals.sum()),
                                      "key_sum": int(records[:, 0].sum())}
                        with mutex:
                            completed[group] = record
                conn.sendall(b"K")
        except Exception as exc:
            with mutex:
                errors.append(repr(exc))

    def send(message):
        try:
            payload = Path(message["file"]).read_bytes()
            delay = message["release"] - time.perf_counter()
            if delay > 0:
                stop.wait(delay)
            header = {k: v for k, v in message.items() if k not in ("op", "file", "release", "destination")}
            encoded = json.dumps(header).encode()
            address = f"10.201.{message['destination']}.2"
            with socket.create_connection((address, port), timeout=90) as conn:
                conn.sendall(struct.pack("!I", len(encoded)) + encoded)
                conn.sendall(payload)
                if recv_exact(conn, 1) != b"K":
                    raise RuntimeError("missing flow acknowledgement")
        except Exception as exc:
            with mutex:
                errors.append(repr(exc))

    def network_loop():
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((f"10.201.{host}.2", port))
            server.listen(512)
            server.settimeout(0.2)
            while not stop.is_set():
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    continue
                threading.Thread(target=receive, args=(conn,), daemon=True).start()

    threading.Thread(target=network_loop, daemon=True).start()
    path = directory / f"worker{host}.sock"
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(str(path))
        server.listen(64)
        while not stop.is_set():
            conn, _ = server.accept()
            with conn, conn.makefile("rb") as stream:
                message = json.loads(stream.readline())
                if message["op"] == "send":
                    threading.Thread(target=send, args=(message,), daemon=True).start()
                    response = {"ok": True}
                elif message["op"] == "stats":
                    with mutex:
                        prefix = message.get("prefix", "")
                        response = json.loads(json.dumps({"progress": {k: v for k, v in progress.items() if k.startswith(prefix)},
                            "completed": {k: v for k, v in completed.items() if k.startswith(prefix)}, "errors": errors}))
                elif message["op"] == "stop":
                    stop.set()
                    response = {"ok": True}
                else:
                    response = {"error": "unknown operation"}
                conn.sendall(json.dumps(response).encode() + b"\n")
    path.unlink(missing_ok=True)


class Network:
    def __init__(self, topo, protocol, directory):
        self.topo, self.protocol, self.directory = topo, protocol, Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.namespaces, self.workers, self.logs = [], [], []

    def inside(self, node, *args):
        return command(["nsenter", "-t", self.namespaces[node].pid, "-n", *args])

    def shape(self, node, device, capacity):
        self.inside(node, "tc", "qdisc", "add", "dev", device, "root", "handle", "1:", "tbf",
                    "rate", f"{int(capacity * 8)}bit", "burst", self.protocol["tbf_burst_bytes"],
                    "latency", f"{self.protocol['tbf_latency_ms']}ms")
        self.inside(node, "tc", "qdisc", "add", "dev", device, "parent", "1:1", "handle", "10:",
                    "netem", "delay", f"{self.protocol['netem_delay_us_per_egress']}us")

    def __enter__(self):
        try:
            for _ in range(6):
                self.namespaces.append(subprocess.Popen(["unshare", "--net", "sleep", "3600"]))
            # Wait for each child to finish unshare before moving devices into it.
            parent_net = os.readlink("/proc/self/ns/net")
            for process in self.namespaces:
                limit = time.monotonic() + 5
                while os.readlink(f"/proc/{process.pid}/ns/net") == parent_net:
                    if process.poll() is not None or time.monotonic() > limit:
                        raise RuntimeError("namespace startup failed")
                    time.sleep(0.01)
            for i in range(6):
                self.inside(i, "ip", "link", "set", "lo", "up")
            for host in range(4):
                rack = 4 + host // 2
                a, b = f"h{host}", f"r{host}"
                command(["ip", "link", "add", a, "type", "veth", "peer", "name", b])
                command(["ip", "link", "set", a, "netns", self.namespaces[host].pid])
                command(["ip", "link", "set", b, "netns", self.namespaces[rack].pid])
                for node, dev, address in [(host, a, f"10.201.{host}.2/24"), (rack, b, f"10.201.{host}.1/24")]:
                    self.inside(node, "ip", "addr", "add", address, "dev", dev)
                    self.inside(node, "ip", "link", "set", dev, "up")
                self.inside(host, "ip", "route", "add", "default", "via", f"10.201.{host}.1")
                self.shape(host, a, self.topo["capacities"][f"out:{host}"])
                self.shape(rack, b, self.topo["capacities"][f"in:{host}"])
            command(["ip", "link", "add", "core0", "type", "veth", "peer", "name", "core1"])
            for rack in range(2):
                node, dev = 4 + rack, f"core{rack}"
                command(["ip", "link", "set", dev, "netns", self.namespaces[node].pid])
                self.inside(node, "ip", "addr", "add", f"10.202.0.{rack+1}/24", "dev", dev)
                self.inside(node, "ip", "link", "set", dev, "up")
                self.inside(node, "sysctl", "-qw", "net.ipv4.ip_forward=1")
                for host in range(4):
                    if host // 2 != rack:
                        self.inside(node, "ip", "route", "add", f"10.201.{host}.0/24", "via", f"10.202.0.{2-rack}")
                self.shape(node, dev, self.topo["capacities"][f"core:{rack}"])
            for host in range(4):
                log = (self.directory / f"worker{host}.log").open("w")
                self.logs.append(log)
                process = subprocess.Popen(["nsenter", "-t", str(self.namespaces[host].pid), "-n", sys.executable,
                    str(Path(__file__).resolve()), "worker", "--host", str(host), "--out", str(self.directory),
                    "--port", str(self.protocol["tcp_port"])], stdout=log, stderr=log)
                self.workers.append(process)
            limit = time.monotonic() + 30
            while not all((self.directory / f"worker{h}.sock").exists() for h in range(4)):
                if any(p.poll() is not None for p in self.workers) or time.monotonic() > limit:
                    raise RuntimeError(f"worker startup failed; inspect {self.directory}")
                time.sleep(0.02)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def stats(self, prefix=""):
        with ThreadPoolExecutor(max_workers=4) as pool:
            values = list(pool.map(lambda h: rpc(self.directory / f"worker{h}.sock", {"op": "stats", "prefix": prefix}), range(4)))
        errors = [error for value in values for error in value["errors"]]
        if errors:
            raise RuntimeError(str(errors))
        return {"observed_at": time.perf_counter(), "progress": {k: v for value in values for k, v in value["progress"].items()},
                "completed": {k: v for value in values for k, v in value["completed"].items()}}

    def send(self, source, message):
        rpc(self.directory / f"worker{source}.sock", {"op": "send", **message})

    def __exit__(self, *_):
        for process in self.workers + self.namespaces:
            if process.poll() is None:
                process.terminate()
        for process in self.workers + self.namespaces:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in self.logs:
            log.close()
        for host in range(4):
            (self.directory / f"worker{host}.sock").unlink(missing_ok=True)


def materialize(parent, directory):
    config = parent["workload"]
    files = {}
    for seed in config["seeds"]:
        for mapper in range(4):
            rng = np.random.default_rng(seed + mapper * 100)
            records = np.empty((config["rows_per_mapper"], 2), dtype=np.uint32)
            records[:, 0] = (rng.zipf(1.15 + 0.1 * mapper, len(records)) % config["keys"] + mapper) % config["keys"]
            records[:, 1] = rng.integers(1, 100, len(records), dtype=np.uint32)
            for reducer in range(4):
                selected = records[records[:, 0] % 4 == reducer]
                path = directory / f"{seed}_{mapper}_{reducer}.bin"
                path.write_bytes(selected.tobytes())
                files[seed, mapper, reducer] = {"file": str(path), "bytes": path.stat().st_size,
                    "sha256": digest(path), "rows": len(selected), "key_sum": int(selected[:, 0].sum()),
                    "value_sum": int(selected[:, 1].sum())}
    return files


def flow_seconds(paths, sizes, capacities):
    active = list(zip(paths, map(float, sizes)))
    elapsed = 0.0
    while active:
        rates = maxmin_rates([p for p, _ in active], capacities)
        step = min(size / rate for (_, size), rate in zip(active, rates))
        elapsed += step
        active = [(path, size - step * rate) for (path, size), rate in zip(active, rates) if size - step * rate > 1e-6]
    return elapsed


def calibrate(net, protocol, directory):
    size = protocol["calibration_bytes_per_flow"]
    path = directory / "calibration.bin"
    path.write_bytes(bytes(size))
    cases = {"single_core": [(0, 2)], "shared_core": [(0, 2), (1, 3)],
             "disjoint_access": [(0, 1), (2, 3)], "shared_receiver": [(0, 2), (1, 2)],
             "single_access": [(0, 1)], "local_tcp": [(0, 0)]}
    results = []
    for name, pairs in cases.items():
        paths = [route(a, b) for a, b in pairs]
        predicted = flow_seconds(paths, [size] * len(pairs), net.topo["capacities"])
        for rep in range(protocol["calibration_repeats"]):
            release = time.perf_counter() + 0.15
            keys = []
            for i, (source, destination) in enumerate(pairs):
                key = f"cal:{name}:{rep}:{i}"
                keys.append(key)
                net.send(source, {"destination": destination, "key": key, "file": str(path), "bytes": size,
                                  "sha256": digest(path), "release": release, "aggregate": False})
            limit = time.monotonic() + 60
            while True:
                stats = net.stats()
                if all(stats["progress"].get(key, {}).get("end") is not None for key in keys):
                    break
                if time.monotonic() > limit:
                    raise RuntimeError("calibration timeout")
                time.sleep(0.01)
            measured = max(stats["progress"][key]["end"] for key in keys) - release
            results.append({"case": name, "repeat": rep, "predicted_s": predicted,
                            "measured_s": measured, "absolute_relative_error": abs(predicted - measured) / measured,
                            "flow_observations": {key: stats["progress"][key] for key in keys}})
    write_json(directory / "calibration.json", results)
    return results


def select(trace, topo, history, snapshot, arm, budget):
    background = {link: 0.0 for link in topo["capacities"]}
    backlog_compute = [0.0] * 4
    old_jobs = []
    for item in history:
        remaining = []
        durations = list(item["trace"]["reduce_s"])
        for m, row in enumerate(item["trace"]["bytes"]):
            values = []
            for r, size in enumerate(row):
                key = f"job:{item['id']}:{m}:{r}"
                left = max(0, size - snapshot["progress"].get(key, {}).get("received", 0))
                values.append(left)
                for link in route(topo["mapper_hosts"][m], item["plan"][r]):
                    background[link] += left
                if f"job:{item['id']}:{r}" in snapshot["completed"]:
                    durations[r] = 0.0
            remaining.append(values)
        if any(map(sum, remaining)) or any(durations):
            for r, host in enumerate(item["plan"]):
                backlog_compute[host] += durations[r]
            ready = [max(0.0, t - snapshot["observed_at"]) for t in item["data_ready"]]
            old_jobs.append((0, {"bytes": remaining, "ready_s": ready, "reduce_s": durations}, item["plan"]))
    if arm == "static_route":
        background = {link: 0.0 for link in background}
        backlog_compute = [0.0] * 4
    plans = list(itertools.product(range(4), repeat=4))
    ranked = []
    for plan in plans:
        load = dict(background)
        compute = list(backlog_compute)
        for r, host in enumerate(plan):
            compute[host] += trace["reduce_s"][r]
            for m, row in enumerate(trace["bytes"]):
                for link in route(topo["mapper_hosts"][m], host):
                    load[link] += row[r]
        score = max(load[l] / topo["capacities"][l] for l in load) + max(compute)
        ranked.append((score, plan))
    ranked.sort()
    if arm != "adaptive_search":
        return ranked[0][1], len(old_jobs)
    scored = [(sum(replay(old_jobs + [(0, trace, plan)], topo)), plan) for _, plan in ranked[:budget]]
    return min(scored)[1], len(old_jobs)


def live_stream(net, protocol, traces, files, arm, interval, prefix):
    history, decisions = [], []
    start = time.perf_counter() + 0.05
    for j in range(protocol["jobs_per_stream"]):
        arrival = start + j * interval
        delay = arrival - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        began = time.perf_counter()
        snapshot = net.stats(f"job:{prefix}:")
        trace = traces[j % len(traces)]
        plan, active_jobs = select(trace, net.topo, history, snapshot, arm, protocol["exact_forecast_budget"])
        selected = time.perf_counter()
        job_id = f"{prefix}:{j}"
        for m in range(4):
            for r in range(4):
                data = files[trace["seed"], m, r]
                net.send(net.topo["mapper_hosts"][m], {"destination": plan[r], "key": f"job:{job_id}:{m}:{r}",
                    "group": f"job:{job_id}:{r}", "file": data["file"], "bytes": data["bytes"],
                    "sha256": data["sha256"], "aggregate": True,
                    "release": max(selected, arrival + trace["ready_s"][m])})
        dispatched = time.perf_counter()
        history.append({"id": job_id, "trace": trace, "plan": plan, "arrival": arrival,
                        "data_ready": [max(selected, arrival + t) for t in trace["ready_s"]]})
        decisions.append({"job": j, "id": job_id, "arrival": arrival, "plan": plan,
                          "decision_s": selected - began, "dispatch_s": dispatched - selected,
                          "controller_lag_s": began - arrival, "observed_active_jobs": active_jobs,
                          "snapshot": snapshot})
    limit = time.monotonic() + 120
    keys = [f"job:{item['id']}:{r}" for item in history for r in range(4)]
    while True:
        stats = net.stats(f"job:{prefix}:")
        if all(key in stats["completed"] for key in keys):
            break
        if time.monotonic() > limit:
            raise RuntimeError("live stream timeout")
        time.sleep(0.01)
    durations = []
    for item in history:
        for r in range(4):
            actual = stats["completed"][f"job:{item['id']}:{r}"]
            for field in ("rows", "key_sum", "value_sum"):
                expected = sum(files[item["trace"]["seed"], m, r][field] for m in range(4))
                if actual[field] != expected:
                    raise RuntimeError(f"aggregation mismatch: {field}")
        durations.append(max(stats["completed"][f"job:{item['id']}:{r}"]["end"] for r in range(4)) - item["arrival"])
    selected_keys = {f"job:{item['id']}:{m}:{r}" for item in history for m in range(4) for r in range(4)}
    return {"arm": arm, "interval_s": interval, "mean_job_s": float(np.mean(durations)), "durations_s": durations,
            "median_decision_s": float(np.median([d["decision_s"] for d in decisions])), "decisions": decisions,
            "flows": {k: v for k, v in stats["progress"].items() if k in selected_keys},
            "completed": {k: stats["completed"][k] for k in keys}, "byte_checks_and_aggregation_pass": True}


def run(args):
    protocol = json.loads(args.protocol.read_text())
    parent_dir = Path("docs/lineages/shuffle_placement_s0_v1/run_2026-10-01")
    parent = json.loads((parent_dir / "protocol.json").read_text())
    raw = json.loads((parent_dir / "measurements.json").read_text())["traces"]
    traces = [sorted([t for t in raw if t["seed"] == s], key=lambda x: x["wall_s"])[1] for s in parent["workload"]["seeds"]]
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "protocol.json").exists():
        raise RuntimeError("use a fresh output directory")
    write_json(args.out / "protocol.json", protocol)
    write_json(args.out / "provenance.json", {"source_sha256": digest(__file__), "parent_source_sha256": digest("scripts_cosim/shuffle_placement_s0.py"),
        "protocol_sha256": digest(args.protocol), "measurements_sha256": digest(parent_dir / "measurements.json"),
        "python": sys.version, "kernel": os.uname().release, "network_namespace": os.readlink("/proc/self/ns/net")})
    data = args.out / "payloads"
    data.mkdir()
    files = materialize(parent, data)
    for trace in traces:
        assert trace["bytes"] == [[files[trace["seed"], m, r]["bytes"] for r in range(4)] for m in range(4)]
    results = []
    for seed in protocol["configuration_seeds"]:
        topo = topology(seed, parent)
        topo["speed"] = [1.0] * 4
        topo_dir = args.out / str(seed)
        write_json(topo_dir / "topology.json", topo)
        with Network(topo, protocol, topo_dir) as net:
            calibration = calibrate(net, protocol, topo_dir)
            print(f"seed={seed} calibration median error={np.median([c['absolute_relative_error'] for c in calibration]):.3f}", flush=True)
            for rep in range(protocol["repeats"]):
                arms = protocol["arms"] if rep % 2 == 0 else list(reversed(protocol["arms"]))
                for interval in protocol["intervals_seconds"]:
                    for arm in arms:
                        prefix = f"{seed}:{rep}:{interval}:{arm}"
                        result = live_stream(net, protocol, traces, files, arm, interval, prefix)
                        result.update({"seed": seed, "repeat": rep})
                        write_json(topo_dir / f"{rep}_{interval}_{arm}.json", result)
                        results.append({k: result[k] for k in ("seed", "repeat", "arm", "interval_s", "mean_job_s", "median_decision_s")})
                        write_json(args.out / "summary.json", results)
                        print(f"seed={seed} rep={rep} interval={interval} arm={arm} mean={result['mean_job_s']:.3f}s decision={result['median_decision_s']:.4f}s", flush=True)
    print("network gate complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["run", "worker"])
    parser.add_argument("--protocol", type=Path, default=Path("experiments/shuffle_network_gate_v1.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--host", type=int)
    parser.add_argument("--port", type=int, default=19381)
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args.host, args.out, args.port)
    else:
        run(args)


if __name__ == "__main__":
    main()
