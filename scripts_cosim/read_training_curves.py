#!/usr/bin/env python3
"""Read a finished training run's curves the way an analyst should, not the way
W&B lays them out.

Motivation (audit of run xnjb91ic, peer_affinity_warm_v1 W1, 2026-09-14): a run
logs ~34 per-step scalars, of which 25 were structurally dead or constant for the
objective that was actually running, and the 9 live ones had no floor printed
next to them. Three numbers looked like findings and were not:

  * ``task_acc`` starting at 43% -- that is chance. The corpus averages 2.79
    candidates per task, so a uniform scorer gets 40.4% and the best constant
    index gets 45.2%.
  * ``ce`` climbing to 11 -- the uniform-scorer plan NLL is 10.26, so this is the
    validation likelihood falling back THROUGH chance, not a divergence.
  * ``regret_greedy`` between 57k and 9k -- raw seconds on a corpus whose optimum
    averages 146,060 s. As a fraction of the random-plan regret it is 72% -> 11%.

This script classifies every logged metric (dead / constant / live), prints the
live ones against their chance floor where the run recorded one, names the
selected epoch and what was thrown away after it, and flags overfitting.

Usage:
    python3 scripts_cosim/read_training_curves.py <wandb-run-dir>
    python3 scripts_cosim/read_training_curves.py wandb/run-20260914_043905-xnjb91ic
    python3 scripts_cosim/read_training_curves.py --csv history.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Metrics whose job is to be a fixed number (a guard), not to move.
GUARD_SUFFIXES = {
    "count_regret_greedy",
    "count_regret_topk",
    "count_regret_masked_topo",
    "count_regret_seq_reforward",
    "greedy_sidecar_coverage",
    "topk_sidecar_coverage",
    "seq_reforward_sidecar_coverage",
    "masked_topo_mapped_rate",
    "masked_topo_decoded",
    "greedy_unmapped",
    "seq_reforward_unmapped",
}

# live metric -> summary key holding the floor it should be read against.
CHANCE_FLOOR = {
    "val/task_acc": ("baseline/val_chance_task_acc", "above"),
    "val/task_acc_choice": ("baseline/val_chance_task_acc_choice", "above"),
    "val/mt_task_acc_choice": ("baseline/val_chance_task_acc_choice", "above"),
    "val/mt_plan_exact": ("baseline/val_chance_graph_acc", "above"),
    "val/acc": ("baseline/val_chance_graph_acc", "above"),
    "val/ce": ("baseline/val_chance_ce", "below"),
    "train/ce": ("baseline/val_chance_ce", "below"),
}

# Summary prefixes that are reference lines rather than curves. `untrained/` is the
# model before any gradient step (train_near_rtt.py logs it since 2026-09-16); the
# first HISTORY row is already after epoch 1, so without it a chart has no true start.
REFERENCE_PREFIXES = ("baseline/", "scale/", "untrained/")


def reference_floors(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in summary.items() if k.startswith(REFERENCE_PREFIXES)}


def untrained_key(metric: str) -> str:
    """The summary key holding a curve's pre-training value: val/task_acc -> untrained/val_task_acc.
    Train-side curves have no untrained read (the trainer evaluates the val split only)."""
    split, _, name = metric.partition("/")
    return f"untrained/{split}_{name}"


def _read_wandb_dir(run_dir: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """History rows, summary, config from an on-disk wandb run directory."""
    from wandb.proto import wandb_internal_pb2 as pb  # noqa: PLC0415
    from wandb.sdk.internal import datastore  # noqa: PLC0415

    wandb_files = sorted(run_dir.glob("*.wandb"))
    if not wandb_files:
        raise SystemExit(f"FAIL LOUD: no *.wandb transaction log under {run_dir}")
    ds = datastore.DataStore()
    ds.open_for_scan(str(wandb_files[0]))
    rows: List[Dict[str, Any]] = []
    # An OFFLINE run has no files/wandb-summary.json until it is synced, so the
    # summary is read out of the transaction log as well. Without this the
    # reference lines silently read as "NONE RECORDED" on every unsynced run.
    summary: Dict[str, Any] = {}

    def _items(items: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for item in items:
            key = "/".join(item.nested_key) if item.nested_key else item.key
            try:
                out[key] = json.loads(item.value_json)
            except Exception:  # noqa: BLE001
                out[key] = item.value_json
        return out

    while True:
        try:
            raw = ds.scan_data()
        except Exception:  # noqa: BLE001 - torn tail on a killed run is expected
            break
        if raw is None:
            break
        rec = pb.Record()
        rec.ParseFromString(raw)
        kind = rec.WhichOneof("record_type")
        if kind == "history":
            rows.append(_items(rec.history.item))
        elif kind == "summary":
            summary.update(_items(rec.summary.update))
    summary_path = run_dir / "files" / "wandb-summary.json"
    if summary_path.exists():
        summary.update(json.loads(summary_path.read_text()))
    config: Dict[str, Any] = {}
    config_path = run_dir / "files" / "config.yaml"
    if config_path.exists():
        try:
            import yaml  # noqa: PLC0415

            raw_cfg = yaml.safe_load(config_path.read_text()) or {}
            config = {
                k: v.get("value") if isinstance(v, dict) and "value" in v else v
                for k, v in raw_cfg.items()
                if not k.startswith("_")
            }
        except Exception:  # noqa: BLE001
            config = {}
    return rows, summary, config


def _read_csv(path: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle)), {}, {}


def _series(rows: List[Dict[str, Any]], key: str) -> List[float]:
    out: List[float] = []
    for row in rows:
        val = row.get(key)
        if val is None or val == "":
            continue
        try:
            out.append(float(val))
        except (TypeError, ValueError):
            continue
    return out


def _fmt(x: float) -> str:
    if x == 0 or 1e-3 <= abs(x) < 1e6:
        return f"{x:,.4f}".rstrip("0").rstrip(".") or "0"
    return f"{x:.4g}"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", help="wandb/run-<ts>-<id> directory")
    ap.add_argument("--csv", help="a history CSV instead of a wandb run dir")
    args = ap.parse_args(argv)

    if args.csv:
        rows, summary, config = _read_csv(Path(args.csv))
    elif args.run_dir:
        rows, summary, config = _read_wandb_dir(Path(args.run_dir))
    else:
        ap.error("give a wandb run directory or --csv")

    def _has(row: Dict[str, Any], prefix_: str) -> bool:
        return any(k.startswith(prefix_) and row[k] not in (None, "") for k in row)

    # A row can carry BOTH the last epoch and the final/* evaluation: wandb merges
    # a step-less log into the current step. Keep it and ignore its final/* keys,
    # or the last epoch silently disappears from every curve.
    step_rows = [r for r in rows if _has(r, "train/") or _has(r, "val/")]
    if not step_rows:
        raise SystemExit("FAIL LOUD: no per-step history rows found")
    keys = sorted({k for r in step_rows for k in r if not k.startswith(("_", "final/"))})

    dead: List[str] = []
    constant: List[tuple[str, float]] = []
    guards: List[tuple[str, float, float]] = []
    live: List[str] = []
    for key in keys:
        xs = _series(step_rows, key)
        if not xs:
            continue
        lo, hi = min(xs), max(xs)
        if key.rsplit("/", 1)[-1] in GUARD_SUFFIXES:
            guards.append((key, lo, hi))
        elif lo == hi == 0.0:
            dead.append(key)
        elif lo == hi:
            constant.append((key, lo))
        else:
            live.append(key)

    n_epochs = len(step_rows)
    print("=" * 78)
    print(f"TRAINING CURVE READ -- {n_epochs} epochs, {len(keys)} logged per-step metrics")
    if config:
        bits = [f"{k}={config[k]}" for k in ("loss_type", "lr", "epochs", "num_train", "num_val", "top_k_decode") if k in config]
        print("  " + "  ".join(bits))
    print("=" * 78)

    print(f"\nDEAD (identically zero every epoch) -- {len(dead)}")
    print("  these cannot move under this run's objective; they are chart noise")
    for key in dead:
        print(f"    {key}")
    print(f"\nCONSTANT -- {len(constant)}")
    for key, val in constant:
        print(f"    {key:42s} = {_fmt(val)}")
    print(f"\nGUARDS (expected fixed; drift here invalidates the regret curves) -- {len(guards)}")
    for key, lo, hi in guards:
        flag = ""
        suffix = key.rsplit("/", 1)[-1]
        family = suffix.split("_")[0] if suffix.endswith(("_coverage", "_mapped_rate")) else ""
        decoded = max(
            _series(step_rows, k) or [0.0]
            for k in (f"{key.rsplit('/', 1)[0]}/count_regret_{family}", key)
        ) if family else [1.0]
        never_ran = family and max(decoded) == 0.0
        if (suffix.endswith("_coverage") or suffix.endswith("_mapped_rate")) and lo < 1.0:
            # 0.0 with nothing decoded means the decode never ran, not that it missed.
            flag = "  (decode never ran)" if never_ran else "  <-- SOME DECODES FELL TO THE worst_regret FLOOR"
        elif suffix.endswith("_unmapped") and hi > 0:
            flag = "  <-- UNMAPPED DECODES"
        elif lo != hi:
            flag = "  <-- MOVED; the scored-dataset count should not change"
        print(f"    {key:42s} min={_fmt(lo)} max={_fmt(hi)}{flag}")

    print(f"\nLIVE -- {len(live)}")
    print(f"    {'metric':40s} {'first':>11s} {'min@ep':>15s} {'max@ep':>15s} {'last':>11s}")
    for key in live:
        xs = _series(step_rows, key)
        imin, imax = xs.index(min(xs)), xs.index(max(xs))
        print(
            f"    {key:40s} {_fmt(xs[0]):>11s} "
            f"{_fmt(min(xs)) + '@' + str(imin):>15s} {_fmt(max(xs)) + '@' + str(imax):>15s} "
            f"{_fmt(xs[-1]):>11s}"
        )

    # --- reference lines -----------------------------------------------------
    floors = reference_floors(summary)
    print("\nREFERENCE LINES (from the run summary)")
    if not floors:
        print("    NONE RECORDED. This run predates baseline/* and scale/* logging, so")
        print("    every number above is unanchored: an accuracy near chance and an")
        print("    accuracy far above it look identical. Re-read with the corpus's")
        print("    candidate counts in hand before quoting anything.")
    else:
        for key in sorted(floors):
            print(f"    {key:42s} = {_fmt(float(floors[key])) if isinstance(floors[key], (int, float)) else floors[key]}")
        for metric, (floor_key, direction) in CHANCE_FLOOR.items():
            if metric not in live or floor_key not in floors:
                continue
            xs = _series(step_rows, metric)
            floor = float(floors[floor_key])
            beats = (max(xs) > floor) if direction == "above" else (min(xs) < floor)
            ends_beating = (xs[-1] > floor) if direction == "above" else (xs[-1] < floor)
            verdict = "beats chance" if beats else "NEVER BEATS CHANCE"
            tail = "" if ends_beating else "  (and ENDS on the wrong side of it)"
            # `start` is the first LOGGED row, i.e. after epoch 1's gradient steps;
            # `untrained` is the model before any step, when the trainer recorded it.
            u = floors.get(untrained_key(metric))
            u_txt = f"untrained={_fmt(float(u))} " if isinstance(u, (int, float)) else ""
            print(f"    {metric:30s} {u_txt}after-ep1={_fmt(xs[0])} best={_fmt(max(xs) if direction == 'above' else min(xs))} chance={_fmt(floor)} -> {verdict}{tail}")

    # --- selection and overfit ----------------------------------------------
    print("\nSELECTION")
    sel_key = None
    constant_keys = {k for k, _ in constant}
    for cand in ("val/regret_masked_topo", "val/regret_topk", "val/regret_greedy"):
        # A selector that never moved is still the selector -- look past `live`.
        if cand in live or cand in constant_keys:
            sel_key = cand
            break
    if sel_key:
        xs = _series(step_rows, sel_key)
        sel = xs.index(min(xs))
        print(f"    checkpoint metric looks like {sel_key}; best at epoch {sel} of {n_epochs}")
        print(f"    {n_epochs - sel - 1} epochs ({(n_epochs - sel - 1) / n_epochs * 100:.0f}%) ran after the selected one")
        for key in live:
            if not key.startswith("val/") or key.startswith("val/regret"):
                continue
            ys = _series(step_rows, key)
            if len(ys) != len(xs):
                continue
            better = max if key.endswith(("acc",)) else min
            peak = ys.index(better(ys))
            if abs(peak - sel) >= max(5, n_epochs // 20):
                print(f"    NOTE {key} peaks at epoch {peak}, not {sel} -- the selector is not optimising it")
    else:
        print("    could not identify a checkpoint metric from the logged keys")

    if "train/ce" in live and "val/ce" in live:
        tr, va = _series(step_rows, "train/ce"), _series(step_rows, "val/ce")
        vmin = va.index(min(va))
        print("\nOVERFIT")
        print(f"    train/ce {_fmt(tr[0])} -> {_fmt(tr[-1])}   val/ce {_fmt(va[0])} -> {_fmt(va[-1])} (min {_fmt(min(va))} @ epoch {vmin})")
        if va[-1] > min(va) * 1.05:
            print(f"    val/ce rose {(va[-1] / min(va) - 1) * 100:.0f}% off its floor after epoch {vmin}: the run memorised.")
        floor_key = CHANCE_FLOOR["val/ce"][0]
        if floor_key in summary and va[-1] > float(summary[floor_key]):
            print(f"    val/ce ENDS ABOVE the uniform-scorer floor {_fmt(float(summary[floor_key]))}: the last-epoch weights")
            print("    assign the optimal plan less probability than a coin would. Never serve them.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
