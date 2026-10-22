"""Persistent, deterministic microbenchmarks used inside study containers."""
import hashlib
import json
import resource
import sys
import time

kind = int(sys.argv[1])
size = 24000 if kind % 2 == 0 else 96000
started = time.perf_counter()
with open(f"/study/records_{size}.json") as stream:
    records = json.load(stream)
index = {row["key"]: row for row in records}
blob = json.dumps(records, separators=(",", ":")).encode() if kind // 2 == 2 else None

def execute():
    if kind // 2 == 0:
        return sum(index[f"key-{i % size}"]["value"] for i in range(20000))
    if kind // 2 == 1:
        return sum(row["value"] * row["weight"] for row in records)
    return hashlib.sha256(blob).hexdigest()

print(json.dumps({"ready": True, "init_s": time.perf_counter() - started,
                  "rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)
for line in sys.stdin:
    command = json.loads(line)
    if command.get("stop"):
        break
    started = time.perf_counter()
    result = execute()
    print(json.dumps({"result": result, "exec_s": time.perf_counter() - started,
                      "rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)
