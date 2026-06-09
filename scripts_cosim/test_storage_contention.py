"""
Proof of storage FilterStore serialization — N-task sweep.

Tests 5 different concurrent cold-task counts (N = 1..5).
For each N, runs two scenarios:

  Contended  — all N tasks routed to platforms on the SAME physical node
               → all N image pulls must serialize through one FilterStore
               → task-(N-1) waits (N-1) × T_pull before its pull even starts

  Parallel   — each task routed to a platform on a DIFFERENT node
               → all N image pulls run concurrently
               → every task finishes in T_pull regardless of N

The multiplier (contended_RTT / parallel_RTT) should grow linearly with N,
directly proving that a scheduler blind to co-located cold replicas causes
N× latency inflation that no queue-depth heuristic can see or prevent.

Pull-time formula (exact match to knative/autoscaler.py):
  T_pull = imageSize_GB / (min(storage_write_mbps, node_network_bw_mbps) / 1024)
         + storage_write_latency

Parameters from data/nofs-ids/ (rpiCpu + flashCard):
  imageSize   = 3.057 GB
  speed       = min(171, 100) = 100 MB/s  (network is the bottleneck)
  latency     = 0.00012 s
  cold_start  = 0.33 s
  exec_time   = 0.78 s
  → T_pull ≈ 31.30 s
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import simpy
from simpy.resources.store import FilterStore

# ---------------------------------------------------------------------------
# Parameters — all taken directly from data/nofs-ids/ JSON files
# ---------------------------------------------------------------------------
IMAGE_SIZE_GB      = 3.057    # task-types.json → dnn1 → imageSize → rpiCpu
STORAGE_WRITE_MBPS = 171.0    # storage-types.json → flashCard → throughput.write
NETWORK_BWIDTH_MBS = 100.0    # infrastructure.json → network.bandwidth
STORAGE_LATENCY    = 0.00012  # storage-types.json → flashCard → latency.write
COLD_START_S       = 0.33     # task-types.json → dnn1 → coldStartDuration → rpiCpu
EXEC_TIME_S        = 0.78     # task-types.json → dnn1 → executionTime → rpiCpu

# Derived — matches the exact formula in knative/autoscaler.py
PULL_SPEED = min(STORAGE_WRITE_MBPS, NETWORK_BWIDTH_MBS)
T_PULL     = IMAGE_SIZE_GB / (PULL_SPEED / 1024.0) + STORAGE_LATENCY
T_BASELINE = T_PULL + COLD_START_S + EXEC_TIME_S   # best-case RTT for one task


# ---------------------------------------------------------------------------
# Core SimPy processes — mirror infrastructure.py + knative/autoscaler.py
# ---------------------------------------------------------------------------
class StorageDevice:
    """Mirrors infrastructure.py Storage: sits inside the node-level FilterStore."""
    def __init__(self):
        self.write_mbps = STORAGE_WRITE_MBPS
        self.latency    = STORAGE_LATENCY


def _pull(env, node_storage, task_id, results):
    """
    Mirrors autoscaler.initialize_replica():
      yield node.storage.get(...)          ← blocks if another pull is in progress
      yield env.timeout(retrieval_duration) ← holds the device for the full pull
      node.storage.put(...)                ← releases for the next waiter
    """
    storage = yield node_storage.get(lambda s: True)

    pull_start     = env.now
    effective_spd  = min(storage.write_mbps, NETWORK_BWIDTH_MBS)
    pull_duration  = IMAGE_SIZE_GB / (effective_spd / 1024.0) + storage.latency
    yield env.timeout(pull_duration)
    node_storage.put(storage)

    results[task_id] = {
        "pull_start":    pull_start,
        "pull_wait":     pull_start,         # time spent blocked before pull began
        "pull_duration": pull_duration,
        "pull_end":      env.now,
    }


def _task(env, node_storage, task_id, results):
    """Full cold-start lifecycle: pull → cold-start → execute."""
    arrival = env.now
    yield env.process(_pull(env, node_storage, task_id, results))
    yield env.timeout(COLD_START_S + EXEC_TIME_S)
    results[task_id]["rtt"]         = env.now - arrival
    results[task_id]["done_time"]   = env.now


# ---------------------------------------------------------------------------
# Scenario runners
# ---------------------------------------------------------------------------
def run_contended(n):
    """All N tasks → same node → one shared FilterStore."""
    env          = simpy.Environment()
    node_storage = FilterStore(env)
    node_storage.put(StorageDevice())   # single storage device — the bottleneck

    results = {}
    for i in range(n):
        env.process(_task(env, node_storage, i, results))
    env.run()
    return results


def run_parallel(n):
    """Each task → its own node → independent FilterStore."""
    env     = simpy.Environment()
    results = {}
    for i in range(n):
        store = FilterStore(env)
        store.put(StorageDevice())
        env.process(_task(env, store, i, results))
    env.run()
    return results


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
SWEEP_N = [1, 2, 3, 4, 5]

if __name__ == "__main__":
    print("=" * 72)
    print("Storage FilterStore Contention — N-task sweep")
    print(f"T_pull = {T_PULL:.2f}s  |  cold_start = {COLD_START_S}s  "
          f"|  exec = {EXEC_TIME_S}s  |  baseline RTT = {T_BASELINE:.2f}s")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # Sweep header
    # -----------------------------------------------------------------------
    header = (f"{'N':>3}  {'Last-task RTT':>14}  {'Last-task RTT':>14}  "
              f"{'RTT penalty':>12}  {'Predicted':>10}  {'Match':>6}")
    subhdr = (f"{'':>3}  {'Contended (A)':>14}  {'Parallel  (B)':>14}  "
              f"{'abs / %':>12}  {'N×T_pull+base':>10}  {'':>6}")
    print("\n" + header)
    print(subhdr)
    print("-" * 72)

    all_pass = True

    for n in SWEEP_N:
        res_c = run_contended(n)
        res_p = run_parallel(n)

        last = n - 1

        rtt_c    = res_c[last]["rtt"]
        rtt_p    = res_p[last]["rtt"]
        penalty  = rtt_c - rtt_p
        pct      = (penalty / rtt_p) * 100

        # Theoretical prediction: last task's pull starts only after (N-1) prior
        # pulls complete → pull_wait = (N-1) × T_pull → RTT = N×T_pull + CST + exec
        predicted_rtt = n * T_PULL + COLD_START_S + EXEC_TIME_S
        error_s       = abs(rtt_c - predicted_rtt)
        match         = "OK" if error_s < 0.01 else f"ERR {error_s:.3f}s"
        if error_s >= 0.01:
            all_pass = False

        print(f"  {n:>1}  {rtt_c:>13.2f}s  {rtt_p:>13.2f}s  "
              f"+{penalty:>8.2f}s / {pct:>4.0f}%  {predicted_rtt:>9.2f}s  {match:>6}")

    print("-" * 72)

    # -----------------------------------------------------------------------
    # Per-task pull timeline for N=4 (most illustrative)
    # -----------------------------------------------------------------------
    n_detail = 4
    res_detail = run_contended(n_detail)
    print(f"\nDetailed pull timeline — Contended, N={n_detail}")
    print(f"{'Task':>6}  {'Pull starts':>12}  {'Pull ends':>10}  "
          f"{'Wait before pull':>17}  {'RTT':>8}")
    print("-" * 60)
    for i in range(n_detail):
        r = res_detail[i]
        wait = r["pull_start"]   # all arrive at t=0, so wait = pull_start
        print(f"  {i:>4}  {r['pull_start']:>11.2f}s  {r['pull_end']:>9.2f}s  "
              f"{wait:>16.2f}s  {r['rtt']:>7.2f}s")
    print(f"\n  → Each task waits exactly (task_index × {T_PULL:.2f}s) "
          f"before its pull can start.")

    # -----------------------------------------------------------------------
    # What a baseline scheduler cannot see
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("What this means for scheduler design")
    print("=" * 72)
    print(f"  A shortest-queue heuristic (Knative) sees:")
    print(f"    node_A:platform_0  queue=0  ← routes task 0 here")
    print(f"    node_A:platform_1  queue=0  ← routes task 1 here  (looks free!)")
    print(f"    node_A:platform_2  queue=0  ← routes task 2 here  (looks free!)")
    print(f"    node_A:platform_3  queue=0  ← routes task 3 here  (looks free!)")
    print(f"  All four queue depths are 0 — the heuristic is maximally confident.")
    print(f"  Actual RTT for task 3: {res_detail[3]['rtt']:.2f}s  "
          f"(expected by heuristic: {T_BASELINE:.2f}s)")
    print(f"  Heuristic underestimation: {res_detail[3]['rtt'] - T_BASELINE:.2f}s "
          f"({(res_detail[3]['rtt'] - T_BASELINE) / T_BASELINE * 100:.0f}% error)")
    print()
    print(f"  node.storage has no public API to inspect pending pull count.")
    print(f"  The serialization queue lives only inside SimPy's FilterStore internals.")
    print(f"  A GNN with node_cold_replicas feature aggregates this density across")
    print(f"  the physical node neighbourhood and can predict the N× multiplier.")

    print()
    if all_pass:
        print("All N predictions match theory within 0.01s tolerance. SWEEP PASSED.")
    else:
        print("WARNING: One or more predictions deviated from theory.")
