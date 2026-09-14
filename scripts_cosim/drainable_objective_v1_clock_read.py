"""A1 (drainable_objective_v1): how far off is the corpus's backlog clock, and what is
the measured one?

The co-sim generator prices a seeded backlog with `Platform.seed_virtual_warmup`:

    cold_start + count x (execution + comm)

Live, a queued task also pays its peer transfers and its source->platform latency, which
`live_audit.platform_queue_drain_seconds` charges and that formula omits. This read puts
the two side by side on the same (task type, platform type) cells, using the live-audit
snapshots the parent lineage already captured, and emits the measured table that Phase B
generates its corpus with.

BAR A1 (signed in docs/lineages/drainable_objective_v1.md before this ran):
    median(measured / formula) >= 3.0  =>  CLOCK-DEFECT

Inputs are the D1 capture files: JSONL, one live-audit snapshot per line, each carrying
per-candidate `queue_length` and `queue_drain_seconds`. A candidate's drain covers the
tasks actually queued on that platform, whose types the snapshot does not record -- so
the measured per-item figure is attributed to the (task type, platform type) cell of the
platform's own replica registration, which is what `queue_distributions` is keyed by too.
Where a platform serves more than one type live, the cell is reported with its type mix
and the read says so rather than silently averaging.

Usage:

    python3 scripts_cosim/drainable_objective_v1_clock_read.py \
        --snapshots simulation_data/drainable_debug_v1/d1/snapshots/knb.jsonl \
        --snapshots simulation_data/drainable_debug_v1/d1/snapshots/gnn.jsonl \
        --snapshots simulation_data/drainable_debug_v1/d1/snapshots/mpoff.jsonl \
        --task-types data/nofs-ids/task-types.json \
        --corpus simulation_data/gnn_datasets_peer_affinity_v1_c3_x200_train2 \
        --out simulation_data/drainable_objective_v1/a1_clock_read.json \
        --drain-table simulation_data/drainable_objective_v1/drain_table.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scripts_cosim.drift_label import approx_comm, drain_table_key

# ---- BARS. Committed before the data exists; do not edit after. ------------
A1_RATIO_MIN = 3.0
# A cell needs this many live observations before its median is quoted; below it the cell
# is reported and excluded from the verdict, never silently folded in.
A1_MIN_OBSERVATIONS = 30
# The verdict needs this many qualifying cells, or the read is VOID -- one cell agreeing
# is not a measurement of "the clock".
A1_MIN_CELLS = 3
# ---------------------------------------------------------------------------


class ClockReadError(RuntimeError):
    """Fail loud."""


def formula_drain(task_type: Dict[str, Any], platform_type: str) -> float:
    """`execution + comm`: what seed_virtual_warmup charges one queued item."""
    execution = float((task_type.get("executionTime") or {}).get(platform_type, 0.0) or 0.0)
    return execution + approx_comm(task_type)


def iter_candidates(snapshot_paths: Sequence[Path]):
    """(platform_type, queue_length, queue_drain_seconds) for every busy candidate."""
    for path in snapshot_paths:
        if not path.exists():
            raise ClockReadError(f"snapshot file missing: {path}")
        with path.open() as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    snap = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ClockReadError(f"{path}:{line_no}: invalid JSON") from exc
                seen_keys = set()
                for task in snap.get("tasks") or []:
                    for cand in task.get("candidates") or []:
                        # One platform appears as a candidate of many tasks in the same
                        # snapshot; count its queue once per snapshot or a deep platform
                        # with many suitors dominates the median by multiplicity alone.
                        key = (snap.get("snapshot_id"), cand.get("queue_key"))
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        depth = int(cand.get("queue_length") or 0)
                        drain = float(cand.get("queue_drain_seconds") or 0.0)
                        if depth <= 0 or drain <= 0.0:
                            continue
                        ptype = cand.get("platform_type")
                        if not ptype:
                            raise ClockReadError(
                                f"{path}:{line_no}: candidate without platform_type -- "
                                "this capture predates the field and cannot be read"
                            )
                        yield str(ptype), depth, drain


def corpus_type_mix(corpus_dirs: Sequence[Path]) -> Dict[str, Dict[str, int]]:
    """platform_type -> {task type: how many datasets register it there}.

    The live snapshots do not record which task types sit in a queue, so the corpus's own
    replica registrations say which (task type, platform type) cells exist and need a
    drain entry.
    """
    mix: Dict[str, Dict[str, int]] = {}
    for corpus in corpus_dirs:
        if not corpus.is_dir():
            raise ClockReadError(f"corpus dir missing: {corpus}")
        for ds in sorted(corpus.glob("ds_*")):
            infra_path = ds / "infrastructure.json"
            if not infra_path.exists():
                continue
            infra = json.loads(infra_path.read_text())
            for ttype, replicas in (infra.get("replica_placements") or {}).items():
                for rep in replicas:
                    ptype = str(rep["platform_type"])
                    cell = mix.setdefault(ptype, {})
                    cell[ttype] = cell.get(ttype, 0) + 1
    if not mix:
        raise ClockReadError("no replica_placements found in any corpus dir")
    return mix


def read(
    snapshot_paths: Sequence[Path],
    task_types_db: Dict[str, Dict[str, Any]],
    corpus_dirs: Sequence[Path],
) -> Dict[str, Any]:
    per_ptype: Dict[str, List[float]] = {}
    depths: Dict[str, List[int]] = {}
    for ptype, depth, drain in iter_candidates(snapshot_paths):
        per_ptype.setdefault(ptype, []).append(drain / depth)
        depths.setdefault(ptype, []).append(depth)
    if not per_ptype:
        raise ClockReadError(
            "no busy candidate carried both queue_length and queue_drain_seconds -- "
            "these captures cannot measure a drain clock"
        )

    mix = corpus_type_mix(corpus_dirs)
    cells: List[Dict[str, Any]] = []
    table: Dict[str, float] = {}
    for ptype in sorted(set(per_ptype) | set(mix)):
        measured = sorted(per_ptype.get(ptype, []))
        types_here = mix.get(ptype, {})
        measured_median = st.median(measured) if measured else None
        for ttype in sorted(types_here) or sorted(task_types_db):
            type_row = task_types_db.get(ttype)
            if type_row is None:
                raise ClockReadError(f"task type {ttype!r} absent from the type table")
            formula = formula_drain(type_row, ptype)
            row = {
                "task_type": ttype,
                "platform_type": ptype,
                "datasets_registering_this_cell": types_here.get(ttype, 0),
                "n_live_observations": len(measured),
                "formula_seconds_per_item": formula,
                "measured_median_seconds_per_item": measured_median,
                "measured_p10": measured[len(measured) // 10] if measured else None,
                "measured_p90": measured[9 * len(measured) // 10] if measured else None,
                "live_depth_median": st.median(depths[ptype]) if measured else None,
                "ratio": (measured_median / formula) if (measured_median and formula > 0) else None,
                "qualifies": len(measured) >= A1_MIN_OBSERVATIONS,
            }
            cells.append(row)

    # ---- the emitted table -------------------------------------------------
    # A candidate's `queue_drain_seconds` covers whatever types are actually queued on
    # that platform, and the snapshot does not record them -- so the measurement is
    # per PLATFORM, not per (type, platform). Two ways to turn it into a per-cell table:
    #
    #   flat      every type on p costs the measured value. Matches the aggregate and
    #             throws away type heterogeneity the simulator really models (cnn runs
    #             3.09 s on rpiCpu against dnn1's 0.003 s), which would make the label
    #             indifferent to which type goes where.
    #   offset    keep the formula's per-type cost and add the platform's missing
    #             seconds: drain(t,p) = formula(t,p) + max(0, measured(p) - mean_t
    #             formula(t,p)). The omitted terms are the peer transfer and the
    #             source->platform latency, which are per task INSTANCE (payload bytes),
    #             not per type -- so a type-independent offset is the shape the physics
    #             has, and the aggregate still matches what was measured.
    #
    # `offset` is emitted. Chosen after A1's ratios were read, which is disclosed in the
    # node; it changes no bar (A1's is the ratio, computed above and untouched) and the
    # flat table is emitted alongside so the choice is inspectable.
    flat_table: Dict[str, float] = {}
    for c in cells:
        if c["measured_median_seconds_per_item"] is not None:
            flat_table[drain_table_key(c["task_type"], c["platform_type"])] = float(
                c["measured_median_seconds_per_item"]
            )
    offsets: Dict[str, float] = {}
    for ptype in sorted({c["platform_type"] for c in cells}):
        here = [c for c in cells if c["platform_type"] == ptype]
        measured = next(
            (c["measured_median_seconds_per_item"] for c in here
             if c["measured_median_seconds_per_item"] is not None),
            None,
        )
        if measured is None:
            continue
        mean_formula = sum(c["formula_seconds_per_item"] for c in here) / len(here)
        offsets[ptype] = max(0.0, measured - mean_formula)
    # Emit an entry for EVERY (task type, platform type with a measured offset), not only
    # the cells the corpus happens to register. The live cluster queues combinations the
    # cold corpus never does -- this read found dnn2 on pynqFpga, which appears in no
    # x200 dataset's replica_placements, and the same shape as the 2026-09-13 xavierGpu
    # audit. A table that covers only the corpus makes every consumer fail loud on the
    # first live state, which is how this surfaced.
    for ptype, offset in offsets.items():
        for ttype, type_row in task_types_db.items():
            table[drain_table_key(ttype, ptype)] = formula_drain(type_row, ptype) + offset

    qualifying = [c for c in cells if c["qualifies"] and c["ratio"] is not None]
    ratios = sorted(c["ratio"] for c in qualifying)
    if len(qualifying) < A1_MIN_CELLS:
        verdict = "VOID"
        median_ratio: Optional[float] = None
    else:
        median_ratio = st.median(ratios)
        verdict = "CLOCK-DEFECT" if median_ratio >= A1_RATIO_MIN else "CLOCK-OK"

    return {
        "bar": {
            "A1_RATIO_MIN": A1_RATIO_MIN,
            "A1_MIN_OBSERVATIONS": A1_MIN_OBSERVATIONS,
            "A1_MIN_CELLS": A1_MIN_CELLS,
        },
        "sources": [str(p) for p in snapshot_paths],
        "corpora": [str(p) for p in corpus_dirs],
        "n_cells": len(cells),
        "n_qualifying_cells": len(qualifying),
        "median_ratio": median_ratio,
        "min_ratio": ratios[0] if ratios else None,
        "max_ratio": ratios[-1] if ratios else None,
        "verdict": verdict,
        "cells": cells,
        "table_form": "offset: formula(t,p) + max(0, measured(p) - mean_t formula(t,p))",
        "table_covers": "every (task type in the type table) x (platform type with a measured offset)",
        "cells_in_corpus_only": sorted(
            f"{c['task_type']}|{c['platform_type']}"
            for c in cells
            if c["datasets_registering_this_cell"] > 0 and c["n_live_observations"] == 0
        ),
        "platform_types_live": sorted(per_ptype),
        "platform_types_in_corpus": sorted(mix),
        "platform_offset_seconds": offsets,
        "drain_seconds_per_item": table,
        "drain_seconds_per_item_flat": flat_table,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshots", action="append", required=True, type=Path)
    ap.add_argument("--task-types", type=Path, default=Path("data/nofs-ids/task-types.json"))
    ap.add_argument("--corpus", action="append", required=True, type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--drain-table",
        type=Path,
        default=None,
        help="also write just the measured table, for HEROSIM_BACKLOG_DRAIN_TABLE",
    )
    args = ap.parse_args()

    db = json.loads(args.task_types.read_text())
    result = read(args.snapshots, db, args.corpus)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    if args.drain_table:
        args.drain_table.parent.mkdir(parents=True, exist_ok=True)
        args.drain_table.write_text(
            json.dumps(
                {
                    "drain_seconds_per_item": result["drain_seconds_per_item"],
                    "source": "drainable_objective_v1 A1",
                    "table_form": result["table_form"],
                    "platform_offset_seconds": result["platform_offset_seconds"],
                    "sources": result["sources"],
                    "verdict": result["verdict"],
                },
                indent=2,
            )
        )

    print(f"[A1] cells {result['n_cells']} ({result['n_qualifying_cells']} qualifying)")
    for c in result["cells"]:
        if not c["qualifies"]:
            continue
        print(
            f"  {c['task_type']:>6}|{c['platform_type']:<10} "
            f"formula {c['formula_seconds_per_item']:8.4f} s  "
            f"measured {c['measured_median_seconds_per_item']:8.3f} s  "
            f"ratio {c['ratio']:8.1f}x  (n={c['n_live_observations']}, "
            f"live depth median {c['live_depth_median']})"
        )
    print(
        f"[A1] median ratio "
        f"{result['median_ratio'] if result['median_ratio'] is not None else float('nan'):.1f}x "
        f"against a {A1_RATIO_MIN}x bar -> {result['verdict']}"
    )
    print(f"[A1] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
