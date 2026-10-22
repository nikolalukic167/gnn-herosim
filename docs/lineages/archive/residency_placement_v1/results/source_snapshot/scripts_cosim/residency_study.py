"""Measured container profiles, exhaustive batch labels, and causal physical gates."""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import time

import numpy as np
from src.placement.residency import CONTRACT, all_plans, evaluate, evict_list, features, greedy, matching, priority, search

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / os.environ.get("RESIDENCY_DATA_DIR", "simulation_data/residency_placement_v1")
PROTOCOL = ROOT / os.environ.get("RESIDENCY_PROTOCOL", "experiments/residency_placement_v1.json")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def receive(process, timeout=30):
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        if not selector.select(timeout):
            raise TimeoutError("container failed to respond")
    line = process.stdout.readline()
    if not line:
        raise RuntimeError("container exited: " + process.stderr.read())
    return json.loads(line)


class Container:
    def __init__(self, function, cpu, memory=256):
        self.name = f"herosim-residency-{os.getpid()}-{time.monotonic_ns()}"
        self.started = time.perf_counter()
        image = json.loads(PROTOCOL.read_text())["image"]
        command = ["docker", "run", "--rm", "-i", "--name", self.name,
                   "--label", "herosim-study=residency_placement_v1", "--network", "none",
                   "--cpus", str(cpu), "--memory", f"{int(memory)}m", "--memory-swap", f"{int(memory)}m",
                   "--mount", f"type=bind,src={DATA / 'fixtures'},dst=/study,readonly",
                   "--mount", f"type=bind,src={ROOT / 'scripts_cosim/residency_worker.py'},dst=/worker.py,readonly",
                   image, "python3", "-u", "/worker.py", str(function)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, bufsize=1)
        try:
            self.ready = receive(self.process)
            if not self.ready.get("ready"):
                raise RuntimeError("missing ready event")
            self.cold_s = time.perf_counter() - self.started
        except BaseException:
            self.close(force=True)
            raise

    def call(self):
        started = time.perf_counter()
        self.process.stdin.write("{}\n")
        self.process.stdin.flush()
        result = receive(self.process)
        result["rpc_s"] = time.perf_counter() - started
        return result

    def close(self, force=False):
        started = time.perf_counter()
        if self.process.poll() is None and not force:
            self.process.stdin.write('{"stop": true}\n')
            self.process.stdin.flush()
            self.process.wait(timeout=15)
        elif self.process.poll() is None:
            subprocess.run(["docker", "rm", "-f", self.name], check=True, capture_output=True)
            self.process.wait(timeout=15)
        code = self.process.returncode
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close()
        if code and not force:
            raise RuntimeError(f"container exit {code}")
        return time.perf_counter() - started


def profile():
    protocol = json.loads(PROTOCOL.read_text())
    if (DATA / "profiles.json").exists():
        raise ValueError("preserve existing measurements")
    (DATA / "fixtures").mkdir(parents=True, exist_ok=True)
    for size in [24000, 96000]:
        write(DATA / f"fixtures/records_{size}.json",
              [{"key": f"key-{i}", "value": i % 997, "weight": i % 13,
                "text": f"event-{i % 1024}-payload"} for i in range(size)])
    measurements = []
    for repetition in range(protocol["profile_repetitions"]):
        for cpu in [.5, 1., 2.]:
            for function in range(6):
                worker = Container(function, cpu)
                try:
                    samples = [worker.call() for _ in range(protocol["profile_warm_requests"])]
                    if len({str(s["result"]) for s in samples}) != 1:
                        raise ValueError("non-deterministic function result")
                    row = {"repetition": repetition, "cpu": cpu, "function": function,
                           "cold_s": worker.cold_s, "ready": worker.ready, "samples": samples}
                finally:
                    row_stop = worker.close()
                row["stop_s"] = row_stop
                measurements.append(row)
        print("profile repetition", repetition, flush=True)
    summary = {}
    for cpu in [.5, 1., 2.]:
        for f in range(6):
            selected = [r for r in measurements if r["cpu"] == cpu and r["function"] == f]
            rss = max(s["rss_kib"] for r in selected for s in [r["ready"], *r["samples"]])
            summary[f"{f}:{cpu}"] = {
                "cold_s": float(np.median([r["cold_s"] for r in selected])),
                "exec_s": float(np.median([s["rpc_s"] for r in selected for s in r["samples"]])),
                "stop_s": float(np.median([r["stop_s"] for r in selected])),
                "memory_mib": max(16, math.ceil(rss / 1024 * 1.5)),
                "expected_result": selected[0]["samples"][0]["result"]}
    write(DATA / "profiles.json", {"protocol_sha256": sha(PROTOCOL), "summary": summary,
                                   "measurements": measurements, "worker_sha256": sha(ROOT / "scripts_cosim/residency_worker.py"),
                                   "image": protocol["image"]})


def trace_rows():
    source = DATA / "source/invocations_per_function_md.anon.d01.csv"
    selected, seen = [], set()
    with source.open() as stream:
        for row in csv.DictReader(stream):
            app = (row["HashOwner"], row["HashApp"])
            counts = [int(row[str(i)]) for i in range(1, 1441)]
            if app in seen or not sum(counts):
                continue
            seen.add(app)
            selected.append({"app": list(app), "function": row["HashFunction"], "counts": counts})
            if len(selected) >= 648:
                break
    if len(selected) != 648:
        raise ValueError("not enough disjoint applications")
    return selected


def make_case(rows, index, attempt, profiles):
    rng = np.random.default_rng(74000 + index)
    block = rows[index * 6:(index + 1) * 6]
    counts = np.asarray([r["counts"] for r in block])
    minute_weights = counts.sum(axis=0)
    minute = int(rng.choice(1440, p=minute_weights / minute_weights.sum()))
    cpu = attempt["cpu"]
    cold = np.array([[profiles[f"{f}:{c}"]["cold_s"] for c in cpu] for f in range(6)])
    execution = np.array([[profiles[f"{f}:{c}"]["exec_s"] for c in cpu] for f in range(6)])
    memory = np.array([max(profiles[f"{f}:{c}"]["memory_mib"] for c in [.5, 1., 2.]) for f in range(6)])
    # A reproducible memory sweep, not a claim about production node sizes.
    capacity = np.full(3, max(memory.max(), np.median(memory) * attempt["slots"]))
    resident = np.zeros((3, 6), bool)
    for h in range(3):
        for f in rng.permutation(6):
            if resident[h] @ memory + memory[f] <= capacity[h]:
                resident[h, f] = True
    windows = []
    for wave in range(2):
        target = (minute + wave * (1 if attempt["trace_mode"] == "adjacent" else 720)) % 1440
        available = np.flatnonzero(minute_weights)
        actual = int(available[np.abs(available - target).argmin()])
        weights = counts[:, actual].astype(float)
        windows.append({"minute": actual + 1, "original_counts": weights.astype(int).tolist(),
                        "requests": rng.choice(6, 8, p=weights / weights.sum()).tolist()})
        if attempt.get("mixed_probe", False):
            mixed = np.r_[np.arange(6), rng.choice(6, 2, p=weights / weights.sum())]
            rng.shuffle(mixed)
            windows[-1]["requests"] = mixed.tolist()
            windows[-1]["synthetic_mechanism_probe"] = True
    return {"id": index, "attempt": attempt["name"], "contract": CONTRACT, "cpu": cpu,
            "apps": [r["app"] for r in block], "trace_functions": [r["function"] for r in block],
            "requests": windows[0]["requests"], "windows": windows,
            "frequency": (counts[:, :minute].sum(axis=1) + np.bincount(windows[0]["requests"], minlength=6) + 1).astype(float).tolist(),
            "cold_s": cold.tolist(), "exec_s": execution.tolist(), "memory_mib": memory.tolist(),
            "capacity_mib": capacity.tolist(), "resident": resident.tolist(),
            "stop_s": [float(np.median([profiles[f"{f}:{c}"]["stop_s"] for f in range(6)])) for c in cpu]}


def generate():
    if (DATA / "METADATA.json").exists():
        raise ValueError("preserve existing corpus")
    protocol = json.loads(PROTOCOL.read_text())
    profiles = json.loads((DATA / "profiles.json").read_text())["summary"]
    rows = trace_rows()
    write(DATA / "trace_selection.json", {"source_sha256": sha(DATA / "source/invocations_per_function_md.anon.d01.csv"),
                                          "rows": rows})
    plans = all_plans()
    (DATA / "placements").mkdir(exist_ok=True)
    sweep_path = DATA / "placements/placements.jsonl"
    sweep = sweep_path.open("x")
    files, ids = {}, {}
    for split, bounds in {"train": (0, 64), "validation": (64, 80), "test": (80, 104), "physical": (104, 108)}.items():
        cases, tensors, targets = [], [], []
        for attempt in protocol["attempts"]:
            for index in range(*bounds):
                case = make_case(rows, index, attempt, profiles)
                if split in ("train", "validation"):
                    costs = evaluate(case, plans)
                    for plan, cost in zip(plans, costs):
                        sweep.write(json.dumps({"split": split, "attempt": attempt["name"], "case": index,
                                               "placement_plan": plan.tolist(), "sum_completion_s": float(cost)}, separators=(",", ":")) + "\n")
                    case["optimal_cost"] = float(costs.min())
                    case["optimal_plan"] = plans[costs.argmin()].tolist()
                    if split == "train":
                        rng = np.random.default_rng(95000 + index)
                        behavior = [case["optimal_plan"], greedy(case), rng.integers(0, 3, 8).tolist()]
                        for path in behavior:
                            compatible = np.ones(len(plans), bool)
                            for step in range(8):
                                prefix = path[:step]
                                q = [float(costs[compatible & (plans[:, step] == h)].min()) for h in range(3)]
                                scale = max(float(np.median(case["cold_s"])), .001)
                                tensors.append(features(case, prefix))
                                targets.append((np.asarray(q) - min(q)) / scale)
                                compatible &= plans[:, step] == path[step]
                cases.append(case)
            print("generated", split, attempt["name"], flush=True)
        write(DATA / f"{split}.json", cases)
        files[f"{split}.json"] = sha(DATA / f"{split}.json")
        ids[split] = sorted(set(c["id"] for c in cases))
        if tensors:
            np.savez_compressed(DATA / "train.npz", x=np.asarray(tensors), q=np.asarray(targets, np.float32))
            files["train.npz"] = sha(DATA / "train.npz")
    sweep.close()
    files["placements/placements.jsonl"] = sha(sweep_path)
    write(DATA / "METADATA.json", {"contract": CONTRACT, "files": files, "ids": ids,
                                   "profiles_sha256": sha(DATA / "profiles.json"), "protocol_sha256": sha(PROTOCOL),
                                   "physics": "measured Docker startup/stop/RPC; admitted resident memory; host FIFO",
                                   "queue_feature_contract": "residency_raw_and_summaries_v1",
                                   "warmth_physics": CONTRACT, "status": "active",
                                   "label_scope": "exhaustive assignments at fixed FIFO and fixed eviction policy"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["profile", "generate"])
    args = parser.parse_args()
    {"profile": profile, "generate": generate}[args.phase]()


if __name__ == "__main__":
    main()
