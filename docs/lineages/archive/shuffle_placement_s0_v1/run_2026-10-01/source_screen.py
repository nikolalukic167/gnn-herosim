"""Measure a TCP hash-aggregation shuffle and screen joint reducer placement.

This standalone event model does not modify HeROsim or validate its network physics.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import socket
import struct
import time

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def recv_exact(sock, size):
    data = bytearray(size)
    view = memoryview(data)
    offset = 0
    while offset < size:
        count = sock.recv_into(view[offset:])
        if not count:
            raise RuntimeError("truncated shuffle transfer")
        offset += count
    return data


def receiver(index, mappers, ready, output):
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(mappers)
        server.settimeout(60)
        ready.put((index, server.getsockname()[1]))

        def receive(conn):
            with conn:
                conn.settimeout(60)
                mapper, size = struct.unpack("!IQ", recv_exact(conn, 12))
                started = time.perf_counter()
                records = np.frombuffer(recv_exact(conn, size), dtype=np.uint32).reshape(-1, 2)
                ended = time.perf_counter()
                return mapper, records, started, ended

        with ThreadPoolExecutor(max_workers=mappers) as pool:
            futures = [pool.submit(receive, server.accept()[0]) for _ in range(mappers)]
            blocks = [future.result() for future in futures]
        started = time.perf_counter()
        records = np.concatenate([block[1] for block in blocks])
        totals = np.bincount(records[:, 0], weights=records[:, 1], minlength=4096)
        ended = time.perf_counter()
        output.put({"kind": "reducer", "id": index, "compute_s": ended - started,
                    "compute_start": started, "end": ended, "rows": len(records),
                    "value_sum": int(totals.sum()), "key_sum": int(records[:, 0].sum()),
                    "receives": [{"mapper": m, "bytes": b.nbytes, "start": s, "end": e}
                                 for m, b, s, e in blocks]})


def producer(index, seed, rows, keys, ports, start, output):
    start.wait(60)
    started = time.perf_counter()
    rng = np.random.default_rng(seed + index * 100)
    records = np.empty((rows, 2), dtype=np.uint32)
    records[:, 0] = (rng.zipf(1.15 + 0.1 * index, rows) % keys + index) % keys
    records[:, 1] = rng.integers(1, 100, rows, dtype=np.uint32)
    buckets = records[:, 0] % len(ports)
    blocks = [records[buckets == r].tobytes() for r in range(len(ports))]
    available = time.perf_counter()

    def send(reducer):
        with socket.create_connection(("127.0.0.1", ports[reducer]), timeout=60) as conn:
            conn.sendall(struct.pack("!IQ", index, len(blocks[reducer])))
            sent = time.perf_counter()
            conn.sendall(blocks[reducer])
            return {"reducer": reducer, "bytes": len(blocks[reducer]),
                    "send_start": sent, "send_end": time.perf_counter()}

    with ThreadPoolExecutor(max_workers=len(ports)) as pool:
        flows = list(pool.map(send, range(len(ports))))
    output.put({"kind": "mapper", "id": index, "start": started, "ready": available,
                "value_sum": int(records[:, 1].sum()), "key_sum": int(records[:, 0].sum()),
                "rows": rows, "flows": flows})


def measure_one(config, seed):
    ctx = mp.get_context("spawn")
    ready, output, start = ctx.Queue(), ctx.Queue(), ctx.Event()
    receivers = [ctx.Process(target=receiver, args=(r, config["mappers"], ready, output))
                 for r in range(config["reducers"])]
    processes = receivers.copy()
    try:
        for process in receivers:
            process.start()
        ports = dict(ready.get(timeout=90) for _ in receivers)
        producers = [ctx.Process(target=producer, args=(m, seed, config["rows_per_mapper"],
                     config["keys"], ports, start, output)) for m in range(config["mappers"])]
        processes += producers
        for process in producers:
            process.start()
        start.set()
        events = [output.get(timeout=120) for _ in processes]
        for process in processes:
            process.join(timeout=10)
            if process.exitcode != 0:
                raise RuntimeError(f"worker failed: {process.pid}, {process.exitcode}")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
    maps = sorted((e for e in events if e["kind"] == "mapper"), key=lambda e: e["id"])
    reducers = sorted((e for e in events if e["kind"] == "reducer"), key=lambda e: e["id"])
    for field in ("rows", "value_sum", "key_sum"):
        if sum(e[field] for e in maps) != sum(e[field] for e in reducers):
            raise RuntimeError(f"shuffle conservation failed: {field}")
    for reducer in reducers:
        for transfer in reducer["receives"]:
            if transfer["bytes"] != maps[transfer["mapper"]]["flows"][reducer["id"]]["bytes"]:
                raise RuntimeError("sender/receiver byte mismatch")
    origin = min(e["start"] for e in maps)
    return {"seed": seed, "origin": origin, "events": events,
            "wall_s": max(e["end"] for e in reducers) - origin,
            "ready_s": [e["ready"] - origin for e in maps],
            "reduce_s": [e["compute_s"] for e in reducers],
            "bytes": [[f["bytes"] for f in e["flows"]] for e in maps]}


def maxmin_rates(paths, capacities):
    rates = np.zeros(len(paths))
    free = set(range(len(paths)))
    residual = dict(capacities)
    while free:
        users = {link: [i for i in sorted(free) if link in paths[i]] for link in capacities}
        used = {link: ids for link, ids in users.items() if ids}
        if not used:
            raise ValueError("every flow needs a resource")
        delta = min(residual[link] / len(ids) for link, ids in used.items())
        for i in free:
            rates[i] += max(delta, 0)
        for link, ids in used.items():
            residual[link] -= max(delta, 0) * len(ids)
        frozen = {i for link, ids in used.items() if residual[link] < 1e-6 for i in ids}
        if not frozen:
            raise RuntimeError("bandwidth allocation stalled")
        free -= frozen
    return rates


def topology(seed, protocol):
    rng = np.random.default_rng(seed)
    net = protocol["network"]
    caps = {f"memory:{h}": net["local_bytes_per_second"] for h in range(4)}
    for h in range(4):
        for direction in ("out", "in"):
            caps[f"{direction}:{h}"] = float(rng.choice(net["access_bytes_per_second"]))
    for rack in range(2):
        caps[f"core:{rack}"] = float(rng.choice(net["core_bytes_per_second"]))
    return {"seed": seed, "capacities": caps, "speed": rng.permutation(protocol["compute_speed_factors"]).tolist(),
            "mapper_hosts": rng.permutation(4).tolist(), "latency": net["propagation_seconds"]}


def route(source, destination):
    if source == destination:
        return (f"memory:{source}",)
    links = [f"out:{source}", f"in:{destination}"]
    if source // 2 != destination // 2:
        links.append(f"core:{source // 2}")
    return tuple(links)


def replay(jobs, topo):
    """Joint event replay: flow arrivals/completions and per-host reducer service."""
    pending, needs, durations, assigned = [], {}, {}, {}
    for j, (arrival, trace, plan) in enumerate(jobs):
        for r, host in enumerate(plan):
            needs[j, r] = len(trace["bytes"])
            durations[j, r] = trace["reduce_s"][r] / topo["speed"][host]
            assigned[j, r] = host
        for m, row in enumerate(trace["bytes"]):
            for r, size in enumerate(row):
                path = route(topo["mapper_hosts"][m], plan[r])
                latency = 0 if len(path) == 1 else topo["latency"] * len(path)
                pending.append((arrival + trace["ready_s"][m] + latency, j, r, float(size), path))
    pending.sort(key=lambda x: x[:3])
    active, finish, host_free = [], {}, [0.0] * 4
    time_now, cursor = 0.0, 0
    while active or cursor < len(pending):
        next_arrival = pending[cursor][0] if cursor < len(pending) else float("inf")
        if not active:
            time_now = next_arrival
        while cursor < len(pending) and pending[cursor][0] <= time_now + 1e-12:
            _, j, r, size, path = pending[cursor]
            active.append([j, r, size, path])
            cursor += 1
        rates = maxmin_rates([flow[3] for flow in active], topo["capacities"])
        next_arrival = pending[cursor][0] if cursor < len(pending) else float("inf")
        step = min(min(flow[2] / rate for flow, rate in zip(active, rates)), next_arrival - time_now)
        if step < -1e-10:
            raise RuntimeError("event time reversal")
        time_now += max(step, 0)
        remaining = []
        completed = []
        for flow, rate in zip(active, rates):
            flow[2] -= rate * step
            if flow[2] <= 1e-5:
                j, r = flow[:2]
                needs[j, r] -= 1
                if needs[j, r] == 0:
                    completed.append((j, r))
            else:
                remaining.append(flow)
        for j, r in sorted(completed):
            host = assigned[j, r]
            end = max(time_now, host_free[host]) + durations[j, r]
            host_free[host] = end
            finish[j, r] = end
        active = remaining
    if any(needs.values()):
        raise RuntimeError("incomplete job")
    return [max(finish[j, r] for r in range(len(plan))) - arrival
            for j, (arrival, trace, plan) in enumerate(jobs)]


def features(trace, topo, plan):
    caps = topo["capacities"]
    loads = {link: 0.0 for link in caps}
    host_compute = [0.0] * 4
    isolated = []
    for r, host in enumerate(plan):
        host_compute[host] += trace["reduce_s"][r] / topo["speed"][host]
        for m, row in enumerate(trace["bytes"]):
            path = route(topo["mapper_hosts"][m], host)
            for link in path:
                loads[link] += row[r]
            isolated.append(trace["ready_s"][m] + row[r] / min(caps[link] for link in path))
    load_times = [loads[link] / caps[link] for link in sorted(caps)]
    raw = list(np.asarray(trace["bytes"], dtype=float).ravel() / 1e6)
    raw += trace["ready_s"] + trace["reduce_s"] + topo["speed"]
    raw += list(np.eye(4)[list(plan)].ravel()) + list(np.eye(4)[topo["mapper_hosts"]].ravel())
    raw += [caps[link] / 1e8 for link in sorted(caps)]
    return raw + load_times + sorted(load_times) + host_compute + sorted(host_compute) + sorted(isolated)


def proxy(trace, topo, plan):
    loads = {link: 0.0 for link in topo["capacities"]}
    compute = [0.0] * 4
    for r, host in enumerate(plan):
        compute[host] += trace["reduce_s"][r] / topo["speed"][host]
        for m, row in enumerate(trace["bytes"]):
            for link in route(topo["mapper_hosts"][m], host):
                loads[link] += row[r]
    return max(trace["ready_s"]) + max(loads[l] / topo["capacities"][l] for l in loads) + max(compute)


def measure(protocol, out):
    traces = []
    for seed in protocol["workload"]["seeds"]:
        for repeat in range(protocol["workload"]["repeats"]):
            result = measure_one(protocol["workload"], seed)
            result["repeat"] = repeat
            traces.append(result)
            write_json(out / "measurements.json", {"scope": protocol["scope"], "traces": traces})
            print(f"measured seed={seed} repeat={repeat}: {result['wall_s']:.3f}s", flush=True)


def screen(protocol, out, checkpoints):
    import joblib
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import wandb

    raw = json.loads((out / "measurements.json").read_text())["traces"]
    # Freeze a measured repetition by median observed duration, never by placement outcome.
    traces = [sorted([t for t in raw if t["seed"] == seed], key=lambda t: t["wall_s"])[len(raw) // len(protocol["workload"]["seeds"]) // 2]
              for seed in protocol["workload"]["seeds"]]
    plans = list(itertools.product(range(4), repeat=4))
    budget = protocol["online_plan_evaluations"]
    records = []
    for seed in protocol["development_topologies"] + protocol["holdout_topologies"]:
        topo = topology(seed, protocol)
        for trace in traces:
            begun = time.perf_counter()
            costs = [replay([(0, trace, plan)], topo)[0] for plan in plans]
            row = {"topology": seed, "trace_seed": trace["seed"], "costs": costs,
                   "features": [features(trace, topo, plan) for plan in plans],
                   "proxy": [proxy(trace, topo, plan) for plan in plans],
                   "enumeration_s": time.perf_counter() - begun}
            records.append(row)
        print(f"enumerated topology={seed}: {len(plans)} complete plans per trace", flush=True)
    write_json(out / "exact_scores.json", {"plans": plans, "records": records})
    dev = [row for row in records if row["topology"] in protocol["development_topologies"]]
    models = {"ridge": make_pipeline(StandardScaler(), Ridge(alpha=protocol["ridge_alpha"])),
              "extra_trees": ExtraTreesRegressor(**protocol["extra_trees"])}
    checkpoints.mkdir(parents=True, exist_ok=True)
    with wandb.init(project="herosim-shuffle-screen", mode="offline", dir=str(out), config=protocol) as run:
        x = np.asarray([f for row in dev for f in row["features"]])
        y = np.asarray([cost for row in dev for cost in row["costs"]])
        for name, model in models.items():
            begun = time.perf_counter()
            model.fit(x, y)
            run.log({f"{name}/fit_s": time.perf_counter() - begun,
                     f"{name}/train_mse": float(np.mean((model.predict(x) - y) ** 2))})
            target = checkpoints / f"{name}.joblib"
            joblib.dump(model, target)
            write_json(target.with_suffix(".contract.json"), {"schema": "shuffle_plan_v1", "features": x.shape[1],
                       "protocol_sha256": digest(out / "protocol.json"), "measurement_sha256": digest(out / "measurements.json"),
                       "training_topologies": protocol["development_topologies"], "model_sha256": digest(target)})
    results, streams = [], []
    for seed in protocol["holdout_topologies"]:
        topo = topology(seed, protocol)
        selected = {name: [] for name in ("route_proxy", "route_search", "ridge_search", "extra_trees_search", "bounded_reference")}
        for trace in traces:
            row = next(r for r in records if r["topology"] == seed and r["trace_seed"] == trace["seed"])
            costs = np.asarray(row["costs"])
            proxy_order = np.argsort(row["proxy"], kind="stable")
            baseline, oracle = int(proxy_order[0]), int(costs.argmin())
            choices = {"route_proxy": baseline, "route_search": int(min(proxy_order[:budget], key=lambda i: costs[i])),
                       "bounded_reference": oracle}
            timing = {}
            for name, model in models.items():
                begun = time.perf_counter()
                order = np.argsort(model.predict(row["features"]), kind="stable")
                timing[name] = time.perf_counter() - begun
                candidates = [baseline] + [int(i) for i in order if i != baseline][:budget - 1]
                choices[name + "_search"] = min(candidates, key=lambda i: costs[i])
            gain = float(costs[baseline] - costs[oracle])
            result = {"topology": seed, "trace_seed": trace["seed"], "oracle_gain_percent": 100 * gain / costs[baseline],
                      "cost_s": {name: float(costs[i]) for name, i in choices.items()},
                      "capture": {name: float((costs[baseline] - costs[i]) / gain) if gain > 1e-12 else None
                                  for name, i in choices.items()}, "inference_s": timing}
            results.append(result)
            for name, i in choices.items():
                selected[name].append(plans[i])
        for name, choices in selected.items():
            jobs = [(j * protocol["stream_gate"]["interval_seconds"], traces[j % len(traces)], choices[j % len(traces)])
                    for j in range(protocol["stream_gate"]["jobs"])]
            durations = replay(jobs, topo)
            streams.append({"topology": seed, "arm": name, "mean_job_s": float(np.mean(durations)),
                            "p95_job_s": float(np.percentile(durations, 95)), "durations_s": durations})
    per_topology = []
    for seed in protocol["holdout_topologies"]:
        rows = [r for r in results if r["topology"] == seed]
        captures = {name: float(np.median([r["capture"][name] for r in rows if r["capture"][name] is not None]))
                    for name in ("route_search", "ridge_search", "extra_trees_search")
                    if any(r["capture"][name] is not None for r in rows)}
        per_topology.append({"topology": seed, "oracle_gain_percent": float(np.median([r["oracle_gain_percent"] for r in rows])),
                             "capture": captures})
    write_json(out / "results.json", {"scope": protocol["scope"], "independent_unit": "network configuration conditional on four shared workload traces; all configurations share the same two-rack graph shape",
               "training_qualified": False, "reason": "multi-host validation and adaptive closed-loop policy gate still required",
               "per_topology": per_topology, "paired": results, "stream_replay": streams,
               "timing_caveat": "reported inference_s is predictor-only; excludes feature construction, ranking and exact evaluation",
               "stream_caveat": "fixed plans selected in isolation; ongoing link state is not observed by the selector; not an adaptive live-policy gate",
               "reference_caveat": "exhaustive for 4^4 static reducer assignments only, not an upper bound on stream performance"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["measure", "screen"])
    parser.add_argument("--protocol", type=Path, default=Path("experiments/shuffle_placement_s0_v1.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save-checkpoints", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    frozen = args.out / "protocol.json"
    if frozen.exists() and json.loads(frozen.read_text()) != protocol:
        raise RuntimeError("protocol differs from frozen output protocol")
    write_json(frozen, protocol)
    write_json(args.out / f"provenance_{args.phase}.json", {"script_sha256": digest(__file__), "protocol_sha256": digest(frozen),
               "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
               "pid": os.getpid(), "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    if args.phase == "measure":
        if (args.out / "measurements.json").exists():
            raise RuntimeError("measurement file already exists; use a fresh output directory")
        measure(protocol, args.out)
    else:
        if args.save_checkpoints is None:
            parser.error("screen requires --save-checkpoints")
        screen(protocol, args.out, args.save_checkpoints)


if __name__ == "__main__":
    main()
