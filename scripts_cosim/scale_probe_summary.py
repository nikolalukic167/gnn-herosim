"""Scale probe: per captured cell, the load health of the capture run from its raw result (plain and effective queue share,
latency percentiles, request failures, run end / last arrival). Uses the gate reader's helpers where they import.
usage: scale_probe_summary.py RAW_DIR INPUTS_DIR  -> JSON on stdout"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from scripts_cosim.fresh_topo_burst_v1_gate import lock_wait_profile, latency_percentiles, placement_wait
except Exception as e:  # noqa: BLE001 - the probe still reports the plain figures
    lock_wait_profile = latency_percentiles = placement_wait = None
    print(f"helpers unavailable: {e!r}", file=sys.stderr)

raw_dir, inputs = Path(sys.argv[1]), Path(sys.argv[2])
out = {}
for f in sorted(raw_dir.glob("*.raw.json")):
    cell, rung, window = f.name.replace(".raw.json", "").rsplit("_", 2)
    try:
        st = json.load(open(f))["stats"]
    except Exception as e:  # noqa: BLE001
        out[f.name] = {"error": repr(e)[:200]}
        continue
    wl = json.load(open(inputs / f"wf1_{rung}" / "wl" / f"grounded_{window}_n50000.json"))
    last_arrival = max(ev["timestamp"] for ev in wl["events"])
    e = float(st.get("averageElapsedTime") or 0.0); q = float(st.get("averageQueueTime") or 0.0)
    row = {"num_tasks": st.get("num_tasks"), "elapsed_mean": e, "queue_share": (q / e if e else None),
           "request_failures": st.get("requestFailures"), "end_time": st.get("endTime"), "last_arrival": last_arrival,
           "run_end_over_last_arrival": (float(st["endTime"]) / last_arrival if st.get("endTime") and last_arrival else None),
           "wallclock_s": st.get("wallclock_s")}
    tr = st.get("taskResults")
    if tr and lock_wait_profile:
        try:
            row["lock_wait"] = lock_wait_profile(tr); row["latency_percentiles"] = latency_percentiles(tr); row["placement_wait"] = placement_wait(tr)
        except Exception as ex:  # noqa: BLE001
            row["helper_error"] = repr(ex)[:200]
    out[f"{rung}_{window}"] = row
print(json.dumps(out, indent=1))
