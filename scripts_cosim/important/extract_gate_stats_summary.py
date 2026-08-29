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
import re
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
ARMS = [
    "knative", "deployed", "tempfix", "mlp", "mlptempfix", "mlpcandrel", "mlpcandreltf",
]

# p5b_draw_study: {cache} x {layout} x seed, one arm (and one sweep dir) per checkpoint.
DRAW_STUDY_ARMS = [
    f"ds{cache}{layout}s{seed}"
    for cache in ("dim14", "tempfix")
    for layout in ("dim22", "dim25cr")
    for seed in (1, 2, 3, 4)
]
ARMS += DRAW_STUDY_ARMS

# gnn_draw_study_v1: 8 seeded draws of the deployed GNN config, same 30 cells.
# objective_pivot_v1 Phase 1 extends the family to seeds 9-16 (trained at the pinned
# commit c08aa7e; see docs/lineages/objective_pivot_v1/phase1-registration-draft.md).
GNN_DRAW_ARMS = [f"gnndraws{seed}" for seed in range(1, 17)]
ARMS += GNN_DRAW_ARMS

# mp_ablation_v1: the message-passing-OFF pair of each gnndraws arm, same seeds, same
# pinned commit, same 30 cells. Same `gnn` policy, so ARM_SUFFIX defaults to "gnn"; only
# the sweep dir separates them.
MP_OFF_ARMS = [f"gnnmpoff{seed}" for seed in range(1, 17)]
ARMS += MP_OFF_ARMS

ARM_SUFFIX = {
    "knative": "knative",
    "mlp": "mlp_dim22",
    "mlptempfix": "mlp_dim22",
    # P5b candidate-relative arms: same policy, dim25cr checkpoints (program_verdict_v1).
    "mlpcandrel": "mlp_dim22",
    "mlpcandreltf": "mlp_dim22",
}
# Every draw-study arm is the same `mlp` policy under a different checkpoint, so the runner
# writes them all as `<cell>_s0_mlp_dim22.json`; only the sweep dir keeps them apart.
ARM_SUFFIX.update({a: "mlp_dim22" for a in DRAW_STUDY_ARMS})
# The GNN draw arms need no entry: they are the `gnn` policy, and ARM_SUFFIX.get defaults
# to "gnn". Listed here only so the asymmetry is not read as an omission.

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

PREFIX_BYTES = 262144  # 256KB; every field read here lands in the first ~10KB

# `stats` cannot be brace-matched: the ~80MB of per-task records are nested INSIDE it
# (`stats.tasks` opens at byte ~2400 and the object closes at ~79.87MB). So the scalars are
# pulled out by name from a bounded prefix instead, and the two response-time curves by
# bracket-matching just their own arrays. `run_provenance` sits before `stats` and is small,
# so that one really can be brace-matched.
SCALAR_RE = r'"{}"\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'


def extract_object(text: str, key: str, open_c: str = "{", close_c: str = "}"):
    """Bracket-match the JSON value stored under `"<key>":` inside a prefix of a document."""
    marker = f'"{key}"'
    i = text.find(marker)
    if i < 0:
        return None
    i = text.find(open_c, i + len(marker))
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
        elif c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                return json.loads(text[i:j + 1])
        j += 1
    return None  # did not close inside the prefix


def read_result(path: Path) -> dict:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        head = fh.read(PREFIX_BYTES)
    s = head.find('"stats"')
    if s < 0:
        raise SystemExit(f"FAIL LOUD: no `stats` key in the first {PREFIX_BYTES}B of {path}")
    tail = head[s:]
    prov = extract_object(head[:s], "run_provenance") or {}

    m = re.search(SCALAR_RE.format("total_rtt"), head[:s])
    if not m or not float(m.group(1)):
        raise SystemExit(f"FAIL LOUD: {path} has no usable total_rtt")
    total_rtt = float(m.group(1))

    out = {}
    for k in SCALARS:
        m = re.search(SCALAR_RE.format(k), tail)
        if m is None:
            raise SystemExit(
                f"FAIL LOUD: {path} has no `{k}` in the first {PREFIX_BYTES}B after `stats`. "
                "Raise PREFIX_BYTES rather than silently reporting a partial record."
            )
        out[k] = float(m.group(1))
    out["total_rtt"] = total_rtt
    out["taskResponseTimeDistribution"] = extract_object(
        tail, "taskResponseTimeDistribution", "[", "]")
    env = prov.get("env", {}) or {}
    code = prov.get("code", {}) or {}
    out["provenance"] = {
        "INFERENCE_FEATURE_LAYOUT": env.get("INFERENCE_FEATURE_LAYOUT"),
        "QUEUE_FEATURE_CONTRACT": env.get("QUEUE_FEATURE_CONTRACT"),
        "MLP_MODEL_PATH": env.get("MLP_MODEL_PATH"),
        "GNN_MODEL_PATH": env.get("GNN_MODEL_PATH"),
        "warmth_physics": prov.get("warmth_physics"),
        # objective_pivot_v1 Phase 1: the scorer asserts venue/code identity on the new
        # arms from the summary alone, so the identity axes must survive extraction.
        "code_commit": code.get("commit"),
        "code_dirty": code.get("dirty"),
        "code_diff_sha256": code.get("diff_sha256"),
        "GNN_DECODE_MODE": env.get("GNN_DECODE_MODE"),
        "GNN_BATCH_SIZE": env.get("GNN_BATCH_SIZE"),
        "GNN_BATCH_TIMEOUT": env.get("GNN_BATCH_TIMEOUT"),
        "HEROSIM_GNN_DEVICE": env.get("HEROSIM_GNN_DEVICE"),
        "TOPOLOGY_FEATURE_CONTRACT": env.get("TOPOLOGY_FEATURE_CONTRACT"),
        "GNN_MP_NODE_EDGES": env.get("GNN_MP_NODE_EDGES"),
        "GNN_DISABLE_MESSAGE_PASSING": env.get("GNN_DISABLE_MESSAGE_PASSING"),
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
