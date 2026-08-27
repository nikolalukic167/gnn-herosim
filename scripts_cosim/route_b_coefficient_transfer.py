#!/usr/bin/env python3
"""route_b_v1 §9b — does the §9a expressiveness bound survive ONE coefficient set?

§9a's T1 repair fits fresh coefficients on every dataset's own sweep; a trained
cross-dataset model gets one set. So §9a bounds what a T1-expressible surrogate can do
PER DATASET, which is strictly more than a single model can do, and NO-GO-PREPROBE-T1 is
only as strong as that gap is small. This measures the gap, under the reading registered
in docs/lineages/route_b_v1/stage2-preregistration.md §9b before the number existed.

Three cells, so that "cost of dropping kint" is never confused with "cost of pooling":

  A  per-dataset, full T1        reproduces §9a (median 1.000) or the harness is wrong
  B  per-dataset, T1 - kint      the cost of dropping the un-poolable block
  C  pooled,      T1 - kint      THE REGISTERED STATISTIC

kint cannot be pooled: its columns are one per (node, task_type) pair in that dataset's own
demand, so both the vocabulary and the width differ across datasets (K in 8..13 here).
Dropping it moves the pooled surrogate CLOSER to the registered T1 feature layout — §2 cols
25-28 are candidate-relative per-type occupancy, whose plan-level rendering is the `quad`
block, and node-identified kint is strictly more than a model ever sees.

Also produced, on the same pass over the same 35 firing datasets:
  * per-dataset coefficient dispersion, with condition numbers and rank deficiency, against
    the two physical predictions registered in §9b (transfer 762.939453125, latency 1.0);
  * the block-attribution ablation (§9a's "parent-coupling 1.000 / occupancy 0.892" prose,
    which no code in this repo could reproduce until now), including the kint+quad arm that
    separates "nonlinear in the same counts" from "needs parent state";
  * a descriptive characterization of the residual datasets the T1 repair leaves above 5%.

Fail-loud: a firing set that disagrees with the frozen report, a saturated fit, or a
non-positive optimum raises rather than being skipped.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from score_route_b_contention import (
    Dataset,
    T1_EXTENDED_BLOCKS,
    decode_regret,
    k_integer_keys,
    load_task_types,
    marginal_sum,
    marginal_surrogate_regret,
    min_marginals,
    summarize,
    t1_cols,
    t1_column_names,
    tie_set_indices,
)

# The T1 blocks that carry a dataset-independent meaning, hence the pooled column set.
# The registered §9b cells always use exactly this tuple; `--add-linkrank` extends only
# the exploratory krank arms with the ingress-route co-use block (route-C screen), so
# re-running without the flag reproduces the frozen §9c/§9d artifacts unchanged.
POOLED_BLOCKS = ("quad", "cap", "hop", "coupling")

MATERIAL_PCT = 5.0        # stage-1's materiality bar, unchanged
REPAIR_MAX = 0.5          # §9a / §9b reading threshold, unchanged
TIGHT_ALPHA = 2.0

# §9b physical predictions, fixed before the fit. Platform._payload_transfer_time charges
# n_hops * payload / (bottleneck_mbps * 1024**2) when a fabric is configured, and
# _dependency_transfer_time adds latency in raw seconds.
PAYLOAD_BYTES = 800_000_000.0
PREDICTED_COEF = {"transfer": PAYLOAD_BYTES / (1024.0 * 1024.0), "latency_sum": 1.0}


def registered_median(values: Sequence[float]) -> float:
    """The median convention the registered gate uses (score_route_b_gate.py:98) — the
    upper of the two middles at even n. Reproduced rather than improved, so §9b's numbers
    are comparable to §9a's digit for digit."""
    return sorted(values)[len(values) // 2] if values else 0.0


def dispersion(values: Sequence[float]) -> dict:
    """Descriptive spread. sd/mean is withheld where the mean is within one sd of zero —
    there the ratio is an artifact of the mean's proximity to zero, not a spread."""
    arr = np.array(values, dtype=float)
    mean, sd = float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    q1, q3 = (float(np.percentile(arr, 25)), float(np.percentile(arr, 75)))
    out = {"n": len(arr), "mean": mean, "sd": sd, "median": float(np.median(arr)),
           "q1": q1, "q3": q3, "iqr": q3 - q1,
           "min": float(arr.min()), "max": float(arr.max())}
    out["sd_over_mean"] = (sd / abs(mean) if abs(mean) > sd and abs(mean) > 0 else None)
    return out


# ---------------------------------------------------------------------------
# Corpus / firing set
# ---------------------------------------------------------------------------

def firing_set(report_path: Path, alpha_key: str) -> Tuple[List[int], List[float]]:
    """Positional indices of the firing datasets, selected exactly as the registered gate
    does (score_route_b_gate.py:49-59): Arm S, tight alpha, R_exact > 5.0, strict."""
    with open(report_path) as fh:
        report = json.load(fh)
    if isinstance(report, list):
        if len(report) != 1:
            raise RuntimeError(f"{report_path}: expected one corpus record, got "
                               f"{len(report)} — pass the Arm S report")
        report = report[0]
    rows = report["per_dataset"][alpha_key]
    idx = [i for i, r in enumerate(rows)
           if not r.get("no_feasible_rows") and r["r_exact_pct"] > MATERIAL_PCT]
    return idx, [rows[i]["r_exact_pct"] for i in idx]


class Cell:
    """One firing dataset, loaded once and reused by every cell and every ablation arm."""

    def __init__(self, ds_dir: Path, task_types_db: dict, alpha: float,
                cap_mode="alpha_max"):
        self.ds_dir = ds_dir
        self.ds = Dataset(ds_dir, task_types_db, "rtt")
        # route_b_env_pivot_v1 (2026-08-27): H1+ rungs score under cap_mode=alpha_mean
        # (score_route_b_contention.py --cap-mode); the transfer step must use the SAME
        # cap_mode or its caps (hence feasibility, r_base, every repair) silently
        # diverge from what the frozen report scored. Default "alpha_max" is the
        # pre-existing behavior, unchanged for every caller that doesn't pass this.
        self.cap_mode = cap_mode
        self.caps = self.ds.node_caps(alpha, cap_mode=cap_mode)
        self.marginal = min_marginals(self.ds.rows)
        self.feasible = [(p, v) for p, v in self.ds.rows
                         if self.ds.plan_feasible(p, self.caps)]
        if not self.feasible:
            raise RuntimeError(f"{ds_dir}: no feasible rows at alpha={alpha}")
        self.best = min(v for _p, v in self.feasible)
        if self.best <= 0:
            raise RuntimeError(f"{ds_dir}: non-positive optimum {self.best}")
        self.r_base = marginal_surrogate_regret(self.ds, self.marginal, self.feasible)

    def repair(self, blocks) -> Tuple[float, np.ndarray]:
        """Per-dataset repaired regret and its fitted coefficients, for a block subset."""
        cols = t1_cols(self.ds, self.caps, blocks=blocks)
        repaired, beta = marginal_surrogate_regret(
            self.ds, self.marginal, self.feasible, cols, return_beta=True)
        if repaired is None:
            raise RuntimeError(
                f"{self.ds_dir}: repair over blocks {list(blocks)} hit the saturation "
                "guard — refusing to report an interpolated zero")
        return min(self.r_base, repaired), beta

    def tie_band(self, predicted: np.ndarray) -> Tuple[float, float, float, int]:
        """(optimistic, pessimistic, mean_tied, n_tied) regret over the plans a surrogate's
        scores cannot separate.

        A coarse column set can predict many feasible plans EQUAL to machine precision; the
        registered decode then picks among them by sorted plan key, which is deterministic
        but arbitrary, and two correct implementations summing in different orders land on
        different plans. That is not a bug in either — it means the statistic is only as
        determined as the surrogate's resolving power. Reporting the band makes an
        indeterminate median visible instead of letting a tie-break decide a verdict.

        `mean_tied` is the §9c-registered fair reading: a real masked decoder must pick one
        tied plan and cannot pick the best by oracle, so what it achieves under a fixed but
        uninformative tie rule is the tie group's MEAN. `optimistic` credits the surrogate
        with plans it cannot distinguish and is an upper bound only, never a verdict.
        """
        # Shared with the scorer's decode_regret_band — one tie definition in the program,
        # so the scorer's R_exact band and this one stay comparable by construction.
        tied = [float(self.feasible[i][1]) for i in tie_set_indices(predicted)]
        pct = lambda v: 100.0 * (v - self.best) / self.best  # noqa: E731
        return (pct(min(tied)), pct(max(tied)),
                pct(float(np.mean(tied))), len(tied))

    def repair_band(self, blocks) -> dict:
        """The registered repair fraction plus the band the prediction ties leave open."""
        cols = t1_cols(self.ds, self.caps, blocks=blocks)
        repaired, beta = marginal_surrogate_regret(
            self.ds, self.marginal, self.feasible, cols, return_beta=True)
        if repaired is None:
            raise RuntimeError(
                f"{self.ds_dir}: repair over blocks {list(blocks)} hit the saturation "
                "guard — refusing to report an interpolated zero")
        Xf = np.array([[1.0, marginal_sum(self.marginal, p)] + cols(p)
                       for p, _v in self.feasible])
        best_tied, worst_tied, mean_tied, n_tied = self.tie_band(Xf @ beta)
        return {"registered": self.fraction(min(self.r_base, repaired)),
                "optimistic": self.fraction(min(self.r_base, best_tied)),
                "pessimistic": self.fraction(min(self.r_base, worst_tied)),
                "mean_tied": self.fraction(min(self.r_base, mean_tied)),
                "n_tied": n_tied, "n_feasible": len(self.feasible),
                "beta": beta}

    def fraction(self, repaired: float) -> float:
        return 1.0 - repaired / self.r_base

    def design(self, blocks) -> Tuple[np.ndarray, np.ndarray]:
        """(X_fit, y_fit) shared columns only: [marginal_sum] + blocks. The intercept is
        supplied per-dataset by the caller (pooled) or by the fit itself (per-dataset)."""
        cols = t1_cols(self.ds, self.caps, blocks=blocks)
        X = np.array([[marginal_sum(self.marginal, p)] + cols(p) for p, _v in self.ds.rows])
        y = np.array([v for _p, v in self.ds.rows], dtype=float)
        return X, y

    def feasible_design(self, blocks) -> np.ndarray:
        cols = t1_cols(self.ds, self.caps, blocks=blocks)
        return np.array([[marginal_sum(self.marginal, p)] + cols(p)
                         for p, _v in self.feasible])


# ---------------------------------------------------------------------------
# Cell C: the pooled fit
# ---------------------------------------------------------------------------

def pooled_fit(cells: List[Cell], blocks, equal_dataset_weight: bool
               ) -> Tuple[np.ndarray, List[float], dict]:
    """One global LS over every firing dataset's full sweep.

    Columns: one intercept indicator per dataset, then the shared [marginal_sum] + blocks.
    A per-dataset intercept is free — an additive constant cannot change a within-dataset
    argmin — so it costs the pooled surrogate nothing and removes RTT-level differences
    that would otherwise be absorbed by the shared coefficients.

    equal_dataset_weight is the declared non-decisive sensitivity: scaling each dataset's
    rows (X and y alike) by 1/sd(y_d) is weighted least squares, giving every dataset the
    same influence instead of letting the high-RTT ones dominate the squared error. Both
    sides are scaled, so the coefficients stay in physical units either way.
    """
    n_shared = None
    blocks_X, blocks_y, owner = [], [], []
    for i, cell in enumerate(cells):
        X, y = cell.design(blocks)
        n_shared = X.shape[1] if n_shared is None else n_shared
        if X.shape[1] != n_shared:
            raise RuntimeError(f"{cell.ds_dir}: shared column width {X.shape[1]} != "
                               f"{n_shared} — the pooled block set is not dataset-"
                               "independent, which is the whole premise")
        weight = 1.0 / float(y.std(ddof=1)) if equal_dataset_weight else 1.0
        blocks_X.append(X * weight)
        blocks_y.append(y * weight)
        owner.append(np.full(len(y), i))
    owner = np.concatenate(owner)
    shared = np.vstack(blocks_X)
    y_all = np.concatenate(blocks_y)
    intercepts = np.zeros((len(y_all), len(cells)))
    for i, cell in enumerate(cells):
        mask = owner == i
        w = 1.0 / float(cell.design(blocks)[1].std(ddof=1)) if equal_dataset_weight else 1.0
        intercepts[mask, i] = w
    X_all = np.hstack([intercepts, shared])
    beta, *_ = np.linalg.lstsq(X_all, y_all, rcond=None)
    shared_beta = beta[len(cells):]

    fractions, bands = [], []
    for i, cell in enumerate(cells):
        Xf = cell.feasible_design(blocks)
        predicted = Xf @ shared_beta          # intercept is constant within a dataset
        repaired = min(cell.r_base, decode_regret(cell.feasible, predicted, cell.best))
        fractions.append(cell.fraction(repaired))
        best_tied, worst_tied, mean_tied, n_tied = cell.tie_band(predicted)
        bands.append({"optimistic": cell.fraction(min(cell.r_base, best_tied)),
                      "pessimistic": cell.fraction(min(cell.r_base, worst_tied)),
                      "mean_tied": cell.fraction(min(cell.r_base, mean_tied)),
                      "n_tied": n_tied})
    info = {"n_rows": int(X_all.shape[0]), "n_params": int(X_all.shape[1]),
            "rank": int(np.linalg.matrix_rank(X_all)),
            "equal_dataset_weight": equal_dataset_weight,
            "median_optimistic": registered_median([b["optimistic"] for b in bands]),
            "median_pessimistic": registered_median([b["pessimistic"] for b in bands]),
            "median_mean_tied": registered_median([b["mean_tied"] for b in bands]),
            "datasets_with_prediction_ties": sum(1 for b in bands if b["n_tied"] > 1),
            "max_tie_group": max(b["n_tied"] for b in bands)}
    return shared_beta, fractions, info


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def run(corpus: Path, report: Path, task_types: Path, alpha: float,
       cap_mode="alpha_max") -> dict:
    task_types_db = load_task_types(task_types)
    idx, reported_r = firing_set(report, str(alpha))
    ds_dirs = sorted(d for d in corpus.glob("ds_*") if d.is_dir())
    print(f"firing datasets: {len(idx)}/{len(ds_dirs)} "
          f"({len(idx) / len(ds_dirs):.4f}) at R_exact > {MATERIAL_PCT}%")

    cells: List[Cell] = []
    for position, r_reported in zip(idx, reported_r):
        cell = Cell(ds_dirs[position], task_types_db, alpha, cap_mode=cap_mode)
        if abs(cell.r_base - r_reported) > 1e-9:
            raise RuntimeError(
                f"{ds_dirs[position]}: R_exact {cell.r_base} != frozen report's "
                f"{r_reported} — the corpus and the report disagree, refusing to fit "
                f"(cap_mode={cap_mode!r} -- confirm this matches what --cap-mode the "
                "report was scored with)")
        cells.append(cell)
    names = t1_column_names(cells[0].ds, blocks=POOLED_BLOCKS)
    shared_names = ["marginal_sum"] + names

    out: dict = {"corpus": str(corpus), "report": str(report), "alpha": alpha,
                 "cap_mode": cap_mode,
                 "n_datasets": len(ds_dirs), "firing": len(cells),
                 "firing_indices": idx,
                 "pooled_blocks": list(POOLED_BLOCKS),
                 "shared_columns": shared_names,
                 "predicted_coefficients": PREDICTED_COEF}

    # --- cells A and B, per dataset -------------------------------------
    per_ds = []
    frac_a, frac_b, betas_b, betas_a = [], [], [], []
    band_a, band_b = [], []
    # route_b_env_pivot_v1 screen S2 (the kill bar): the per-dataset t1x tie band
    # (T1_EXTENDED_BLOCKS = kint+quad+cap+hop+coupling+linkrank+hetdem+futureint), on
    # the SAME firing-set cells this function already loaded -- no separate corpus
    # pass. score_route_b_contention.py's t1x arm reports only a flat regret value
    # (no tie band); this is the band the registration's "registered and pessimistic
    # required to agree in direction" bar needs.
    band_t1x = []
    frac_t1x_saturated = []
    for cell in cells:
        ba = cell.repair_band(("kint",) + POOLED_BLOCKS)
        bb = cell.repair_band(POOLED_BLOCKS)
        beta_a, beta_b = ba.pop("beta"), bb.pop("beta")
        rep_a = cell.r_base * (1 - ba["registered"])
        rep_b = cell.r_base * (1 - bb["registered"])
        band_a.append(ba)
        band_b.append(bb)
        frac_a.append(ba["registered"])
        frac_b.append(bb["registered"])
        betas_a.append(beta_a[-len(names):])
        betas_b.append(beta_b[1:])           # drop the fitted intercept, keep [msum]+cols
        X, _y = cell.design(POOLED_BLOCKS)
        Xi = np.hstack([np.ones((len(X), 1)), X])
        try:
            bt = cell.repair_band(T1_EXTENDED_BLOCKS)
            bt.pop("beta")
            frac_t1x_saturated.append(False)
        except RuntimeError:
            # saturation guard refusal (repair_band raises rather than reporting an
            # interpolated value) -- recorded, never silently a pass or a zero.
            bt = None
            frac_t1x_saturated.append(True)
        band_t1x.append(bt)
        per_ds.append({
            "ds": cell.ds_dir.name,
            "r_exact_pct": cell.r_base,
            "cell_a_repaired_pct": rep_a, "cell_a_fraction": cell.fraction(rep_a),
            "cell_b_repaired_pct": rep_b, "cell_b_fraction": cell.fraction(rep_b),
            "t1x_band": bt, "t1x_saturated": bt is None,
            "n_rows": len(cell.ds.rows), "n_feasible": len(cell.feasible),
            "kint_width": len(k_integer_keys(cell.ds)),
            "cond": float(np.linalg.cond(Xi)),
            "rank": int(np.linalg.matrix_rank(Xi)), "n_params": int(Xi.shape[1]),
            "coefficients": dict(zip(shared_names, [float(v) for v in beta_b[1:]])),
        })
    out["per_dataset"] = per_ds

    valid_t1x_bands = [b for b in band_t1x if b is not None]
    out["t1x_per_dataset"] = {
        "n_cells": len(cells),
        "n_saturated": sum(frac_t1x_saturated),
        "median_registered": (registered_median([b["registered"] for b in valid_t1x_bands])
                              if valid_t1x_bands else None),
        "median_pessimistic": (registered_median([b["pessimistic"] for b in valid_t1x_bands])
                               if valid_t1x_bands else None),
        "median_mean_tied": (registered_median([b["mean_tied"] for b in valid_t1x_bands])
                             if valid_t1x_bands else None),
        "median_optimistic": (registered_median([b["optimistic"] for b in valid_t1x_bands])
                              if valid_t1x_bands else None),
    }

    # --- cell C, pooled --------------------------------------------------
    beta_c, frac_c, info_c = pooled_fit(cells, POOLED_BLOCKS, equal_dataset_weight=False)
    beta_w, frac_w, info_w = pooled_fit(cells, POOLED_BLOCKS, equal_dataset_weight=True)
    out["cells"] = {
        "A_per_dataset_full_t1": {
            "median_fraction": registered_median(frac_a),
            "mean_fraction": float(np.mean(frac_a)),
            "n_closed_ge_half": sum(1 for f in frac_a if f >= REPAIR_MAX),
            "residual_gt_material": sum(1 for c, f in zip(cells, frac_a)
                                        if c.r_base * (1 - f) > MATERIAL_PCT),
            "median_optimistic": registered_median([b["optimistic"] for b in band_a]),
            "median_pessimistic": registered_median([b["pessimistic"] for b in band_a]),
            "median_mean_tied": registered_median([b["mean_tied"] for b in band_a]),
            "datasets_with_prediction_ties": sum(1 for b in band_a if b["n_tied"] > 1),
            "max_tie_group": max(b["n_tied"] for b in band_a)},
        "B_per_dataset_no_kint": {
            "median_fraction": registered_median(frac_b),
            "mean_fraction": float(np.mean(frac_b)),
            "n_closed_ge_half": sum(1 for f in frac_b if f >= REPAIR_MAX),
            "median_optimistic": registered_median([b["optimistic"] for b in band_b]),
            "median_pessimistic": registered_median([b["pessimistic"] for b in band_b]),
            "median_mean_tied": registered_median([b["mean_tied"] for b in band_b]),
            "datasets_with_prediction_ties": sum(1 for b in band_b if b["n_tied"] > 1),
            "max_tie_group": max(b["n_tied"] for b in band_b)},
        "C_pooled_no_kint": {
            "median_fraction": registered_median(frac_c),
            "mean_fraction": float(np.mean(frac_c)),
            "n_closed_ge_half": sum(1 for f in frac_c if f >= REPAIR_MAX),
            "residual_gt_material": sum(1 for c, f in zip(cells, frac_c)
                                        if c.r_base * (1 - f) > MATERIAL_PCT),
            "coefficients": dict(zip(shared_names, [float(v) for v in beta_c])),
            "fit": info_c},
        "C_sensitivity_equal_dataset_weight": {
            "median_fraction": registered_median(frac_w),
            "mean_fraction": float(np.mean(frac_w)),
            "n_closed_ge_half": sum(1 for f in frac_w if f >= REPAIR_MAX),
            "coefficients": dict(zip(shared_names, [float(v) for v in beta_w])),
            "fit": info_w},
    }
    for cell_row, fa, fb, fc, fw in zip(per_ds, frac_a, frac_b, frac_c, frac_w):
        cell_row["cell_c_fraction"] = fc
        cell_row["cell_c_sensitivity_fraction"] = fw

    # --- dispersion ------------------------------------------------------
    arr_b = np.array(betas_b)   # [msum] + shared columns, per dataset
    out["dispersion"] = {
        name: dispersion(arr_b[:, j].tolist())
        for j, name in enumerate(shared_names)}
    out["dispersion_full_t1_shared_columns"] = {
        name: dispersion(np.array(betas_a)[:, j].tolist())
        for j, name in enumerate(names)}
    out["rank_deficient_datasets"] = [r["ds"] for r in per_ds
                                      if r["rank"] < r["n_params"]]

    # --- the verdict, applied as registered ------------------------------
    med_b, med_c, med_w = (out["cells"]["B_per_dataset_no_kint"]["median_fraction"],
                           out["cells"]["C_pooled_no_kint"]["median_fraction"],
                           out["cells"]["C_sensitivity_equal_dataset_weight"]
                           ["median_fraction"])
    if med_b < REPAIR_MAX:
        verdict = "VOID-KINT-CONFOUNDED"
    elif med_c >= REPAIR_MAX:
        verdict = "BOUND-TRANSFERS"
    else:
        verdict = "BOUND-DOES-NOT-TRANSFER"
    straddle = (med_c >= REPAIR_MAX) != (med_w >= REPAIR_MAX)
    out["verdict"] = verdict
    out["sensitivity_straddles_threshold"] = bool(straddle)
    # Whether a cell's verdict survives the plans its own surrogate cannot separate. Cell
    # A (the §9a statistic) must be checked too — a NO-GO resting on a tie-break would be
    # no better than the ds_00008 "genuine tie" that turned out to be two verifier bugs.
    out["tie_indeterminate"] = {
        key: bool((out["cells"][key]["median_optimistic"] >= REPAIR_MAX)
                  != (out["cells"][key]["median_fraction"] >= REPAIR_MAX))
        for key in ("A_per_dataset_full_t1", "B_per_dataset_no_kint")}

    # §9c(b): cell B IS the anonymous (dim36crk-expressible) closure, because `quad` is the
    # plan-level rendering of cols 25-28, `load_over_cap` of col 29, `overcap_tasks` of 31,
    # and min/max_hop_sum + transfer of 33-35. kint is the only block with no col in §2.
    cell_b = out["cells"]["B_per_dataset_no_kint"]
    readings = {k: cell_b[key] for k, key in
                (("mean_tied", "median_mean_tied"),
                 ("registered", "median_fraction"),
                 ("pessimistic", "median_pessimistic"))}
    directions = {k: v >= REPAIR_MAX for k, v in readings.items()}
    if len(set(directions.values())) > 1:
        anon_verdict = "VOID-TIE-INDETERMINATE"
    elif directions["mean_tied"]:
        anon_verdict = "NO-GO-PREPROBE-T1-STANDS"
    else:
        anon_verdict = "VOID-T1-MISSPECIFIED"
    out["anonymous_closure"] = {
        "note": "cell B = the dim36crk-expressible plan-level column set; kint is the only "
                "T1 block with no column in the §2 feature table",
        "readings": readings, "directions": directions,
        "optimistic_upper_bound_not_a_verdict": cell_b["median_optimistic"],
        "verdict": anon_verdict}
    return out


def node_features(cell: Cell) -> Dict[str, Dict[str, float]]:
    """Identity-free descriptors of each node, i.e. what a cross-dataset model could key on
    instead of a node's name. Nothing here uses the node's label."""
    ds = cell.ds
    with open(ds.ds_dir / "infrastructure.json") as fh:
        infra = json.load(fh)
    queues = infra.get("queue_distributions") or {}
    nodes = sorted({ds.node_of(p) for _t, p in ds.demand})
    types = sorted(set(ds.task_type_names))

    q_by_node: Dict[str, Dict[str, float]] = {n: {k: 0.0 for k in types} for n in nodes}
    for ttype, slots in queues.items():
        for key, depth in slots.items():
            node = key.split(":")[0]
            if node in q_by_node and ttype in q_by_node[node]:
                q_by_node[node][ttype] += float(depth)

    reps: Dict[str, Dict[str, int]] = {n: {k: 0 for k in types} for n in nodes}
    for (task_id, placement) in ds.demand:
        node = ds.node_of(placement)
        reps[node][ds.task_type_names[task_id]] += 1

    degree = {n: 0 for n in nodes}
    for key in ds.links:
        a, b = key.split("|")
        for n in (a, b):
            if n in degree:
                degree[n] += 1

    out: Dict[str, Dict[str, float]] = {}
    for node in nodes:
        hops, bnecks = [], []
        for other in nodes:
            if other == node:
                continue
            h, bw, _lat = ds.route_metrics(other, node)
            hops.append(float(h))
            bnecks.append(bw if math.isfinite(bw) else 0.0)
        out[node] = {
            "cap": float(cell.caps.get(node, 0.0)),
            "max_demand": float(cell.caps.get(node, 0.0)) / TIGHT_ALPHA,
            "n_replicas": float(sum(reps[node].values())),
            "queue_total": float(sum(q_by_node[node].values())),
            "mean_hop": float(np.mean(hops)) if hops else 0.0,
            "min_hop": float(min(hops)) if hops else 0.0,
            "degree": float(degree[node]),
            "mean_bottleneck": float(np.mean(bnecks)) if bnecks else 0.0,
            **{f"n_replicas_{k}": float(reps[node][k]) for k in types},
            **{f"queue_{k}": float(q_by_node[node][k]) for k in types},
        }
    return out


def identity_or_features(cells: List[Cell]) -> dict:
    """§9c(a): is the fitted kint block a function of node FEATURES, or of node IDENTITY?

    Gauge fix, and it is not optional. Within a dataset the kint columns for a given task
    type sum, on every row, to that type's fixed task count n_type (every valid plan places
    every task, so Σ_node kint(node, type) = n_type identically across all candidate plans).
    That row-wise identity — not the specific value of n_type — is what makes the design
    rank-deficient by exactly one dimension per type: the fitted coefficients are defined
    only up to a per-type additive shift, and `lstsq` returns the minimum-norm
    representative, which is a convention, not a fact. Re-derived 2026-08-25 for the 8-task
    probe (two diamond4 instances, n_type=2 for every type, kint no longer one-hot per row):
    the argument depends only on the row sum being constant, so it is unchanged by n_type
    going from 1 to 2 — confirmed against the `centre = mean(coefs[same type])` code below,
    which was already written for a general group size, not hardcoded to pairs of 1.
    Regressing a convention on node features would be meaningless. So both the target and
    the score are taken on coefficients CENTERED within
    (dataset, task_type): the gauge-invariant content is a node's coefficient RELATIVE to the
    other nodes' for that type, which is exactly the quantity a model would have to recover.
    """
    rows, targets, groups = [], [], []
    feat_names: Optional[List[str]] = None
    for gi, cell in enumerate(cells):
        _rep, beta = cell.repair(("kint",) + POOLED_BLOCKS)
        keys = k_integer_keys(cell.ds)
        coefs = {key: float(beta[2 + i]) for i, key in enumerate(keys)}
        feats = node_features(cell)
        types = sorted(set(cell.ds.task_type_names))
        for ttype in types:
            same = [(n, t) for (n, t) in keys if t == ttype]
            if len(same) < 2:
                continue
            centre = float(np.mean([coefs[k] for k in same]))
            for key in same:
                node = key[0]
                fv = feats[node]
                if feat_names is None:
                    feat_names = sorted(fv) + [f"is_{t}" for t in types]
                row = [fv[name] for name in sorted(fv)]
                row += [1.0 if ttype == t else 0.0 for t in types]
                rows.append(row)
                targets.append(coefs[key] - centre)
                groups.append(gi)
    X = np.array(rows)
    y = np.array(targets)
    g = np.array(groups)
    X = np.hstack([np.ones((len(X), 1)), X])

    preds = np.zeros_like(y)
    for gi in np.unique(g):
        train, test = g != gi, g == gi
        beta, *_ = np.linalg.lstsq(X[train], y[train], rcond=None)
        preds[test] = X[test] @ beta
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2_out = 1.0 - ss_res / ss_tot
    beta_all, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2_in = 1.0 - float(((y - X @ beta_all) ** 2).sum()) / ss_tot
    return {"n_coefficients": int(len(y)), "n_datasets": int(len(np.unique(g))),
            "features": ["intercept"] + (feat_names or []),
            "r2_heldout_by_dataset": r2_out, "r2_in_sample": r2_in,
            "target": "kint coefficient centered within (dataset, task_type)",
            "target_sd": float(y.std(ddof=1))}


def krank_cols(cell: Cell, n_ranks: Optional[int] = None):
    """Exploratory, NOT registered and no verdict is read from it (§9c): occupancy indexed
    by node RANK under an identity-free canonical ordering (capacity, then mean hop), rather
    than by node name. Anonymous and fixed-width like dim36crk, but preserving the cross-node
    distribution that `quad` sums away — it separates "needs identity" from "needs per-node
    resolution keyed by something".

    `n_ranks` pads to a common width so the block can be POOLED across datasets whose node
    counts differ (5 vs 6 here); a dataset with fewer nodes simply leaves its top rank slots
    at zero. Padding at the top is the natural choice because the ordering is ascending by
    capacity."""
    ds = cell.ds
    feats = node_features(cell)
    order = sorted(feats, key=lambda n: (feats[n]["cap"], feats[n]["mean_hop"], n))
    if n_ranks is not None and len(order) > n_ranks:
        raise RuntimeError(f"{ds.ds_dir}: {len(order)} nodes exceeds pad width {n_ranks}")
    width = n_ranks if n_ranks is not None else len(order)
    rank = {n: i for i, n in enumerate(order)}
    types = sorted(set(ds.task_type_names))

    def fn(plan):
        cols = [0.0] * (width * len(types))
        for task_id, placement in plan.items():
            r = rank[ds.node_of(placement)]
            k = types.index(ds.task_type_names[task_id])
            cols[r * len(types) + k] += 1.0
        return cols
    return fn


def krank_demand_cols(cell: Cell, n_ranks: Optional[int] = None):
    """route_b env pivot (2026-08-27), --extended-blocks: krank_cols's exact rank x type
    structure, but summing each placement's REAL per-instance demand instead of a unit
    count — the demand-weighted analog the W1 plan item 2 calls for, padded/pooled
    identically to krank_cols so the two stay directly comparable (same rank order, same
    pad width, same call signature)."""
    ds = cell.ds
    feats = node_features(cell)
    order = sorted(feats, key=lambda n: (feats[n]["cap"], feats[n]["mean_hop"], n))
    if n_ranks is not None and len(order) > n_ranks:
        raise RuntimeError(f"{ds.ds_dir}: {len(order)} nodes exceeds pad width {n_ranks}")
    width = n_ranks if n_ranks is not None else len(order)
    rank = {n: i for i, n in enumerate(order)}
    types = sorted(set(ds.task_type_names))

    def fn(plan):
        cols = [0.0] * (width * len(types))
        for task_id, placement in plan.items():
            r = rank[ds.node_of(placement)]
            k = types.index(ds.task_type_names[task_id])
            cols[r * len(types) + k] += ds.demand[(task_id, placement)]
        return cols
    return fn


def ablation(cells: List[Cell]) -> dict:
    """§9a's block attribution, which no committed code could reproduce until now.

    The prose in LINEAGES/PREREG names a 'parent-coupling block' and an 'occupancy block'
    without fixing their membership; PREREG:406 parenthesizes the former as
    '(kint + cols 33-35 analogues)', i.e. INCLUDING kint. Both readings are reported, plus
    the kint+quad arm that separates 'nonlinear in the same counts' from 'needs parent
    state' — the distinction stage-1 finding #2 turns on.
    """
    arms = {
        "kint": ("kint",),
        "quad": ("quad",),
        "kint+quad": ("kint", "quad"),
        "occupancy (kint+quad+cap)": ("kint", "quad", "cap"),
        "parent-coupling (hop+coupling)": ("hop", "coupling"),
        "parent-coupling incl kint": ("kint", "hop", "coupling"),
        "full T1": ("kint", "quad", "cap", "hop", "coupling"),
        # route_b_env_pivot_v1 screen, B1 (registered build item, §4): single-block
        # arms for the new extended-competitor columns, so S4's attribution table can
        # show whether hetdem/futureint alone carry a rung's closure — the same
        # question the parent-coupling arm answers for hop+coupling.
        "hetdem": ("hetdem",),
        "futureint": ("futureint",),
    }
    out = {}
    for name, blocks in arms.items():
        fractions, residual = [], 0
        for cell in cells:
            repaired, _beta = cell.repair(blocks)
            fractions.append(cell.fraction(repaired))
            residual += 1 if repaired > MATERIAL_PCT else 0
        out[name] = {"blocks": list(blocks),
                     "median_fraction": registered_median(fractions),
                     "mean_fraction": float(np.mean(fractions)),
                     "n_closed_ge_half": sum(1 for f in fractions if f >= REPAIR_MAX),
                     "residual_gt_material": residual,
                     "fractions": fractions}
    return out


def characterize(cells: List[Cell], per_ds: List[dict]) -> dict:
    """Descriptive only, and declared so before it ran: the residual stratum is 5.4% of the
    corpus, under the program's own 10% materiality bar, so nothing here is a claim. It is
    the only place non-pointwise-expressible structure could still live, and describing it
    costs one pass over data already on disk."""
    rows = []
    for cell, row in zip(cells, per_ds):
        cols = t1_cols(cell.ds, cell.caps, blocks=("hop", "coupling"))
        opt_plan = min(cell.feasible, key=lambda pv: pv[1])[0]
        min_hop, max_hop, transfer, latency, same_node = cols(opt_plan)
        nodes = [cell.ds.node_of(p) for p in opt_plan.values()]
        loads: Dict[str, float] = {}
        for t, p in opt_plan.items():
            loads[cell.ds.node_of(p)] = (loads.get(cell.ds.node_of(p), 0.0)
                                         + cell.ds.demand[(t, p)])
        util = [loads[n] / cell.caps[n] for n in loads if n in cell.caps]
        spread = np.array([v for _p, v in cell.feasible])
        rows.append({
            "ds": row["ds"],
            "residual": row["r_exact_pct"] * (1 - row["cell_a_fraction"]) > MATERIAL_PCT,
            "r_exact_pct": row["r_exact_pct"],
            "feasible_frac": len(cell.feasible) / len(cell.ds.rows),
            "distinct_nodes_in_optimum": len(set(nodes)),
            "same_node_edges_in_optimum": same_node,
            "transfer_in_optimum": transfer,
            "max_hop_sum_in_optimum": max_hop,
            "max_load_over_cap_in_optimum": max(util) if util else 0.0,
            "rtt_spread_cv": float(spread.std(ddof=1) / spread.mean()),
            "kint_width": row["kint_width"],
            "cond": row["cond"],
        })
    keys = [k for k in rows[0] if k not in ("ds", "residual")]
    groups = {"residual": [r for r in rows if r["residual"]],
              "closed": [r for r in rows if not r["residual"]]}
    contrast = {k: {g: float(np.median([r[k] for r in rs])) for g, rs in groups.items()}
                for k in keys}
    return {"per_dataset": rows, "n_residual": len(groups["residual"]),
            "n_closed": len(groups["closed"]), "median_contrast": contrast}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True,
                    help="the frozen §9a Arm S report (defines the firing set)")
    ap.add_argument("--task-types", type=Path,
                    default=Path("data/nofs-ids/task-types.json"))
    ap.add_argument("--alpha", type=float, default=TIGHT_ALPHA)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--add-linkrank", action="store_true",
                    help="extend the exploratory krank arms with the linkrank block "
                         "(ingress-route link co-use; route-C screen competitor). "
                         "Registered §9b cells are never affected.")
    ap.add_argument("--extended-blocks", action="store_true",
                    help="route_b env pivot (2026-08-27), W1 item 2: extend the "
                         "exploratory krank pooled/per-dataset arms with hetdem+"
                         "futureint (score_route_b_contention.t1_cols) AND add a "
                         "second krank-shaped block of demand-weighted rank x type "
                         "occupancy (krank_demand_cols), padded/pooled exactly as "
                         "krank_cols. Registered §9a/§9b cells (A/B/C) are hardcoded "
                         "and never affected by this flag.")
    ap.add_argument("--cap-mode", default="alpha_max",
                    help="route_b_env_pivot_v1: must match whatever --cap-mode the "
                         "--report was scored with (score_route_b_contention.py), or "
                         "R_exact disagrees and Cell.__init__ refuses to fit. "
                         "'alpha_max' (default, unchanged) | 'alpha_mean' | a bare "
                         "number (interpreted as {'absolute': x}).")
    args = ap.parse_args()
    extra_t1_blocks = ()
    if args.add_linkrank:
        extra_t1_blocks += ("linkrank",)
    if args.extended_blocks:
        extra_t1_blocks += ("hetdem", "futureint")
    krank_pool_blocks = POOLED_BLOCKS + extra_t1_blocks

    if args.cap_mode in ("alpha_max", "alpha_mean"):
        cap_mode = args.cap_mode
    else:
        try:
            cap_mode = {"absolute": float(args.cap_mode)}
        except ValueError:
            raise SystemExit(
                f"--cap-mode: unrecognised value {args.cap_mode!r}; expected "
                "'alpha_max', 'alpha_mean', or a number")

    out = run(args.corpus, args.report, args.task_types, args.alpha, cap_mode=cap_mode)
    task_types_db = load_task_types(args.task_types)
    ds_dirs = sorted(d for d in args.corpus.glob("ds_*") if d.is_dir())
    cells = [Cell(ds_dirs[i], task_types_db, args.alpha, cap_mode=cap_mode)
            for i in out["firing_indices"]]
    out["ablation"] = ablation(cells)
    out["residual_characterization"] = characterize(cells, out["per_dataset"])
    out["identity_or_features"] = identity_or_features(cells)

    # exploratory krank arm (§9c, no verdict read from it). Per-dataset fractions and
    # tie bands are report fields since PP0′ (stage-2 corrected registration §10): the
    # independent verifier must agree on EVERY (dataset, arm) fraction, band included,
    # so the aggregates alone stopped being enough. Purely additive — every
    # pre-existing key is computed exactly as before.
    kr, kr_bands = [], []
    for cell in cells:
        cols = krank_cols(cell)
        combined = t1_cols(cell.ds, cell.caps, blocks=krank_pool_blocks)
        if args.extended_blocks:
            demand_cols = krank_demand_cols(cell)
            merged = (lambda p, a=cols, b=combined, c=demand_cols: a(p) + b(p) + c(p))
        else:
            merged = (lambda p, a=cols, b=combined: a(p) + b(p))
        repaired, beta_k = marginal_surrogate_regret(
            cell.ds, cell.marginal, cell.feasible, merged, return_beta=True)
        if repaired is None:
            kr.append(None)
            kr_bands.append(None)
            continue
        Xf = np.array([[1.0, marginal_sum(cell.marginal, p)] + merged(p)
                       for p, _v in cell.feasible])
        best_tied, worst_tied, mean_tied, n_tied = cell.tie_band(Xf @ beta_k)
        kr.append(cell.fraction(min(cell.r_base, repaired)))
        kr_bands.append({
            "registered": kr[-1],
            "optimistic": cell.fraction(min(cell.r_base, best_tied)),
            "pessimistic": cell.fraction(min(cell.r_base, worst_tied)),
            "mean_tied": cell.fraction(min(cell.r_base, mean_tied)),
            "n_tied": n_tied, "n_feasible": len(cell.feasible)})
    if all(v is not None for v in kr):
        out["krank_exploratory"] = {
            "note": "occupancy by identity-free node rank + the dim36crk set; NOT "
                    "registered, no verdict read from it",
            "blocks": list(krank_pool_blocks),
            "ds": [cell.ds_dir.name for cell in cells],
            "fractions": kr,
            "bands": kr_bands,
            "median_fraction": registered_median(kr),
            "mean_fraction": float(np.mean(kr)),
            "n_closed_ge_half": sum(1 for f in kr if f >= REPAIR_MAX)}

        # Unlike kint, krank IS poolable: identity-free index, fixed width across datasets
        # with the same node count. This is the follow-up the §9b VOID named. Still
        # exploratory — no verdict is read from it.
        n_ranks = max(len(node_features(c)) for c in cells)

        def _pooled_row_fn(cell):
            kc = krank_cols(cell, n_ranks)
            bc = t1_cols(cell.ds, cell.caps, blocks=krank_pool_blocks)
            if not args.extended_blocks:
                return lambda p: kc(p) + bc(p)
            dc = krank_demand_cols(cell, n_ranks)
            # order MUST match pooled_names below: krank, krank_demand, then t1 blocks.
            return lambda p: kc(p) + dc(p) + bc(p)

        if True:
            pooled_kr, bands_kr = [], []
            n = len(cells)
            X_parts, y_parts, owner = [], [], []
            for i, cell in enumerate(cells):
                row_fn = _pooled_row_fn(cell)
                X_parts.append(np.array(
                    [[marginal_sum(cell.marginal, p)] + row_fn(p)
                     for p, _v in cell.ds.rows]))
                y_parts.append(np.array([v for _p, v in cell.ds.rows]))
                owner.append(np.full(len(cell.ds.rows), i))
            owner = np.concatenate(owner)
            shared = np.vstack(X_parts)
            inter = np.zeros((len(shared), n))
            for i in range(n):
                inter[owner == i, i] = 1.0
            beta, *_ = np.linalg.lstsq(np.hstack([inter, shared]),
                                       np.concatenate(y_parts), rcond=None)
            sb = beta[n:]
            full_bands_kr = []
            for cell in cells:
                row_fn = _pooled_row_fn(cell)
                Xf = np.array([[marginal_sum(cell.marginal, p)] + row_fn(p)
                               for p, _v in cell.feasible])
                pred = Xf @ sb
                rep = min(cell.r_base, decode_regret(cell.feasible, pred, cell.best))
                pooled_kr.append(cell.fraction(rep))
                bo, bp, mt, nt = cell.tie_band(pred)
                bands_kr.append(cell.fraction(min(cell.r_base, mt)))
                full_bands_kr.append({
                    "registered": pooled_kr[-1],
                    "optimistic": cell.fraction(min(cell.r_base, bo)),
                    "pessimistic": cell.fraction(min(cell.r_base, bp)),
                    "mean_tied": bands_kr[-1],
                    "n_tied": nt, "n_feasible": len(cell.feasible)})
            types0 = sorted(set(cells[0].ds.task_type_names))
            krank_names = [f"krank[r{r}|{k}]"
                           for r in range(n_ranks) for k in types0]
            pooled_names = ["marginal_sum"] + krank_names
            if args.extended_blocks:
                pooled_names += [f"krank_demand[r{r}|{k}]"
                                 for r in range(n_ranks) for k in types0]
            pooled_names += t1_column_names(cells[0].ds, blocks=krank_pool_blocks)
            out["krank_pooled_exploratory"] = {
                "note": "ONE coefficient set over identity-free rank-indexed occupancy + "
                        "the dim36crk set — the follow-up the §9b VOID named. Exploratory.",
                "blocks": list(krank_pool_blocks),
                "extended_blocks": bool(args.extended_blocks),
                "n_ranks": int(n_ranks),
                "ds": [cell.ds_dir.name for cell in cells],
                "fractions": pooled_kr,
                "bands": full_bands_kr,
                "coefficients": dict(zip(pooled_names, [float(v) for v in sb])),
                "median_fraction": registered_median(pooled_kr),
                "median_mean_tied": registered_median(bands_kr),
                "mean_fraction": float(np.mean(pooled_kr)),
                "n_closed_ge_half": sum(1 for f in pooled_kr if f >= REPAIR_MAX)}

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)

    c = out["cells"]
    print("\n=== §9b cells (median repair fraction over the firing datasets) ===")
    for key in ("A_per_dataset_full_t1", "B_per_dataset_no_kint", "C_pooled_no_kint",
                "C_sensitivity_equal_dataset_weight"):
        row = c[key]
        band = ""
        if "median_optimistic" in row:
            band = (f"  tie-band [{row['median_pessimistic']:.4f}, "
                    f"{row['median_optimistic']:.4f}] "
                    f"({row['datasets_with_prediction_ties']}/{out['firing']} tied, "
                    f"max group {row['max_tie_group']})")
        elif "median_optimistic" in row.get("fit", {}):
            band = (f"  tie-band [{row['fit']['median_pessimistic']:.4f}, "
                    f"{row['fit']['median_optimistic']:.4f}]")
        print(f"  {key:38s} median={row['median_fraction']:.4f} "
              f"mean={row['mean_fraction']:.4f} "
              f">=0.5: {row['n_closed_ge_half']:2d}/{out['firing']}{band}")
    print(f"\n  VERDICT: {out['verdict']}"
          + ("   [sensitivity straddles the threshold]"
             if out["sensitivity_straddles_threshold"] else ""))
    for key, bad in out["tie_indeterminate"].items():
        if bad:
            print(f"  !! {key}: prediction ties straddle the {REPAIR_MAX} threshold — "
                  "the median is not determined by the surrogate, only by the tie-break")

    print("\n=== pooled coefficients vs the registered physical predictions ===")
    pooled = c["C_pooled_no_kint"]["coefficients"]
    for name, value in pooled.items():
        pred = PREDICTED_COEF.get(name)
        note = f"   predicted {pred:.6f}  ratio {value / pred:+.4f}" if pred else ""
        print(f"  {name:18s} {value:16.6f}{note}")

    print("\n=== per-dataset coefficient dispersion (descriptive) ===")
    for name, d in out["dispersion"].items():
        ratio = f"{d['sd_over_mean']:.3f}" if d["sd_over_mean"] is not None else "  n/a"
        print(f"  {name:18s} mean={d['mean']:14.4f} sd={d['sd']:14.4f} "
              f"sd/mean={ratio}  IQR={d['iqr']:.4f}")
    if out["rank_deficient_datasets"]:
        print(f"  rank-deficient designs: {out['rank_deficient_datasets']}")

    print("\n=== block attribution (per-dataset fits) ===")
    for name, row in out["ablation"].items():
        print(f"  {name:32s} median={row['median_fraction']:.4f} "
              f"mean={row['mean_fraction']:.4f} "
              f">=0.5: {row['n_closed_ge_half']:2d}/{out['firing']} "
              f"residual>5%: {row['residual_gt_material']}")

    rc = out["residual_characterization"]
    print(f"\n=== residual stratum ({rc['n_residual']} residual vs {rc['n_closed']} "
          "closed; DESCRIPTIVE, below materiality) ===")
    for key, row in rc["median_contrast"].items():
        print(f"  {key:32s} residual={row['residual']:12.4f} "
              f"closed={row['closed']:12.4f}")
    ac = out["anonymous_closure"]
    print("\n=== §9c(b) ANONYMOUS CLOSURE — cell B is the dim36crk-expressible set ===")
    for name, value in ac["readings"].items():
        print(f"  {name:14s} median={value:.4f}   "
              f"{'>=' if value >= REPAIR_MAX else '< '} {REPAIR_MAX}")
    print(f"  {'optimistic':14s} median="
          f"{ac['optimistic_upper_bound_not_a_verdict']:.4f}   "
          "(upper bound only, NOT a decoder reading)")
    print(f"\n  §9c(b) VERDICT: {ac['verdict']}")

    iof = out["identity_or_features"]
    print("\n=== §9c(a) is kint identity or features? ===")
    print(f"  held-out-by-dataset R² = {iof['r2_heldout_by_dataset']:.4f}  "
          f"(in-sample {iof['r2_in_sample']:.4f}) over {iof['n_coefficients']} "
          f"coefficients, {iof['n_datasets']} datasets")
    print(f"  reading: R² >= 0.5 -> kint is feature-representable, §9a stands; "
          f"< 0.5 -> identity-memorized")
    if "krank_exploratory" in out:
        kre = out["krank_exploratory"]
        print(f"\n  [exploratory, no verdict] krank+dim36crk per-dataset median="
              f"{kre['median_fraction']:.4f} >=0.5: {kre['n_closed_ge_half']}/"
              f"{out['firing']}")
    if "krank_pooled_exploratory" in out:
        kpe = out["krank_pooled_exploratory"]
        print(f"  [exploratory, no verdict] krank+dim36crk POOLED  median="
              f"{kpe['median_fraction']:.4f} (mean_tied {kpe['median_mean_tied']:.4f}) "
              f">=0.5: {kpe['n_closed_ge_half']}/{out['firing']}")

    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
