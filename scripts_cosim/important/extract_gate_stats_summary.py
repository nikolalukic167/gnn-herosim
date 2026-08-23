"""Pull the scalar `stats` block out of every live-gate result into one small summary.

A gate result JSON is ~80MB, almost all of it per-task records, and there are 120 of them
across the three contention gates x four arms. Loading them to read `averageOccupation` is
both slow and (per the datalab pitfalls) not something to do on a login node.

The `stats` object sits in the first few KB of the file, before the bulk, so this reads a
bounded prefix and brace-matches the object out of it instead of parsing the document. That
turns "parse 9.6GB" into "read 120 x 64KB".

Writes one JSON keyed by (gate, condition, arm, cell). Pairs each result with its
`.decode_stats.json` sidecar when present, so decode behaviour and outcome sit together.

Usage:
  extract_gate_stats_summary.py --root simulation_data/normal_sim_sweeps \\
      --out simulation_data/gate_stats_summary.json
"""

import argparse
import json
import sys
from pathlib import Path

# (gate prefix, condition) -> the sweep dirs are <prefix>_<cond>_<arm>.
GATES = [
    ("drawgate", "backbone"),
    ("drawgate", "nobackbone"),
    ("promo175", "backbone"),
    ("promo175", "nobackbone"),
    ("bbrob", "bb_core8_bw1p5"),
    ("bbrob", "bb_core4_bw0p5"),
]
ARMS = ["knative", "deployed", "tempfix", "mlp", "mlptempfix"]
ARM_SUFFIX = {"knative": "knative", "mlp": "mlp_dim22", "mlptempfix": "mlp_dim22"}

# Scalars worth carrying; the two response-time distributions are kept separately because
# they are 100-element percentile curves and dominate the output size otherwise.
SCALARS = [
    "endTime", "unusedPlatforms", "unusedNodes", "averageOccupation",
    "averageElapsedTime", "averagePullTime", "averageColdStartTime",
    "averageExecutionTime", "averageWaitTime", "averageQueueTime",
    "averageInitializationTime", "averageComputeTime", "averageCommunicationsTime",
    "penaltyProportion", "localDependenciesProportion", "localCommunicationsProportion",
    "nodeCacheHitsProportion", "taskCacheHitsProportion", "coldStartProportion",
]

PREFIX_BYTES = 262144  # 256KB: stats has been observed to close well inside 64KB


def extract_object(text: str, key: str) -> dict | None:
    """Brace-match the JSON object stored under `"<key>":` inside a prefix of a document."""
    marker = f'"{key}"'
    i = text.find(marker)
    if i < 0:
        return None
    i = text.find("{", i + len(marker))
    if i < 0:
        return None
    depth, j, in_str, esc = 0, i, False, False
    while j < len(text):
        c = text[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[i:j + 1])
        j += 1
    return None  # did not close inside the prefix


def read_result(path: Path) -> dict:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        head = fh.read(PREFIX_BYTES)
    stats = extract_object(head, "stats")
    if stats is None:
        raise SystemExit(
            f"FAIL LOUD: no complete `stats` object in the first {PREFIX_BYTES}B of {path}. "
            "Raise PREFIX_BYTES rather than silently reporting a partial record."
        )
    prov = extract_object(head, "run_provenance") or {}
    total_rtt = None
    for line in head.splitlines():
        if '"total_rtt"' in line:
            total_rtt = json.loads("{" + line.rstrip(",") + "}")["total_rtt"]
            break
    if not total_rtt:
        raise SystemExit(f"FAIL LOUD: {path} has no usable total_rtt")

    out = {k: stats.get(k) for k in SCALARS}
    out["total_rtt"] = total_rtt
    out["taskResponseTimeDistribution"] = stats.get("taskResponseTimeDistribution")
    env = prov.get("env", {}) or {}
    out["provenance"] = {
        "INFERENCE_FEATURE_LAYOUT": env.get("INFERENCE_FEATURE_LAYOUT"),
        "QUEUE_FEATURE_CONTRACT": env.get("QUEUE_FEATURE_CONTRACT"),
        "MLP_MODEL_PATH": env.get("MLP_MODEL_PATH"),
        "GNN_MODEL_PATH": env.get("GNN_MODEL_PATH"),
        "warmth_physics": prov.get("warmth_physics"),
    }
    ds = path.parent / path.name.replace(".json", ".decode_stats.json")
    if ds.is_file():
        out["decode_stats"] = json.loads(ds.read_text())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("simulation_data/normal_sim_sweeps"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    summary, missing = {}, []

    for prefix, cond in GATES:
        for arm in arms:
            d = args.root / f"{prefix}_{cond}_{arm}" / "results"
            if not d.is_dir():
                missing.append(f"{prefix}/{cond}/{arm} (no results dir)")
                continue
            suffix = ARM_SUFFIX.get(arm, "gnn")
            for p in sorted(d.glob(f"*_s0_{suffix}.json")):
                cell = p.name.replace(f"_s0_{suffix}.json", "")
                summary.setdefault(f"{prefix}/{cond}", {}).setdefault(cell, {})[arm] = \
                    read_result(p)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    n = sum(len(a) for c in summary.values() for a in c.values())
    print(f"wrote {args.out}: {n} results over {len(summary)} gate/condition blocks")
    for m in missing:
        print(f"  (absent) {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
