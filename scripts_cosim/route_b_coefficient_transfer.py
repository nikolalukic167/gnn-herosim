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

Fail-loud: a firing set that disagrees with the frozen report or a non-positive optimum
raises rather than being skipped.

A **saturated fit is a refusal, not an abort** (route_b_env_pivot_v1, 2026-08-27). The
saturation guard fires per (dataset, block set): a fit with fewer than 2x as many rows as
parameters could interpolate the sweep, so its regret is refused. Until this was fixed,
ONE refused arm aborted the whole run, which made arms that fit perfectly well unreadable
— on the pivot's H0/H1 the S4 bar block `hop+coupling` (7 params, needs 14 rows) fits in
BOTH the 16-row and 64-row arms, yet no S4 table could be produced because the unrelated
full-T1 arm (21 params, needs 42) refused on the 16-row arm. Every per-dataset arm now
records its refusals, reports `n_fitted` / `n_saturated` and a `by_arm` breakdown keyed on
the unconstrained sweep size, and computes its median over the fitted subset only. A
statistic whose denominator lost an arm says so in its own block; an arm with nothing left
to fit reports `median: null` and a named VOID, never a zero and never a silent pass.
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


class SaturationRefusal(RuntimeError):
    """A per-(dataset, block set) fit refused by the saturation guard.

    Its own class, not a bare RuntimeError, so a caller can tolerate exactly this and
    nothing else: `except RuntimeError` at a call site would also swallow "no feasible
    rows", "non-positive optimum" and the firing-set disagreement, every one of which
    must stay fatal. Subclasses RuntimeError so any existing handler keeps working.
    """


def _fitted_by_arm(cells: Sequence["Cell"], flags: Sequence[bool]) -> Dict[str, dict]:
    """`{n_rows: {fitted, saturated}}` — the preflight's step-3 rule, in the artifact.

    Keyed on the UNCONSTRAINED sweep size, which is the replica-config arm's signature and
    survives the refusal, so a reader can see at a glance when a median's denominator has
    collapsed onto one arm. `flags[i]` is True where cell i was fitted.
    """
    out: Dict[str, dict] = {}
    for cell, ok in zip(cells, flags):
        row = out.setdefault(str(len(cell.ds.rows)),
                             {"fitted": 0, "saturated": 0})
        row["fitted" if ok else "saturated"] += 1
    return out


def _fmt(value: Optional[float], spec: str = ".4f", width: int = 6) -> str:
    """Format a statistic that may be absent. A refusal prints as `n/a`, never as 0."""
    return format(value, spec) if value is not None else "n/a".rjust(width)


def _median_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return registered_median(vals) if vals else None


def _per_dataset_cell(cells: Sequence["Cell"],
                      fracs: Sequence[Optional[float]],
                      bands: Sequence[Optional[dict]],
                      residual_gt_material: bool = False) -> dict:
    """One per-dataset cell's block, computed over the FITTED subset only.

    `fracs` / `bands` are index-aligned with `cells` and carry None where the saturation
    guard refused. Every pre-existing key keeps its name, order and value on a corpus
    where nothing is refused; `n_fitted` / `n_saturated` / `by_arm` are appended so a
    reader can never mistake a collapsed denominator for a full one.
    """
    ok = [b is not None for b in bands]
    fit_fracs = [f for f in fracs if f is not None]
    fit_bands = [b for b in bands if b is not None]
    out: dict = {
        "median_fraction": _median_or_none(fracs),
        "mean_fraction": float(np.mean(fit_fracs)) if fit_fracs else None,
        "n_closed_ge_half": sum(1 for f in fit_fracs if f >= REPAIR_MAX),
    }
    if residual_gt_material:
        out["residual_gt_material"] = sum(
            1 for c, f in zip(cells, fracs)
            if f is not None and c.r_base * (1 - f) > MATERIAL_PCT)
    out.update({
        "median_optimistic": _median_or_none([b["optimistic"] for b in fit_bands]),
        "median_pessimistic": _median_or_none([b["pessimistic"] for b in fit_bands]),
        "median_mean_tied": _median_or_none([b["mean_tied"] for b in fit_bands]),
        "datasets_with_prediction_ties": sum(1 for b in fit_bands if b["n_tied"] > 1),
        "max_tie_group": max((b["n_tied"] for b in fit_bands), default=None),
        "n_fitted": len(fit_bands),
        "n_saturated": len(bands) - len(fit_bands),
        "by_arm": _fitted_by_arm(cells, ok),
    })
    return out

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
            raise SaturationRefusal(
                f"{self.ds_dir}: repair over blocks {list(blocks)} hit the saturation "
                f"guard ({len(self.ds.rows)} sweep rows) — refusing to report an "
                "interpolated zero")
        return min(self.r_base, repaired), beta

    def try_repair(self, blocks) -> Optional[Tuple[float, np.ndarray]]:
        """`repair`, returning None on a saturation refusal. Every other error stays fatal."""
        try:
            return self.repair(blocks)
        except SaturationRefusal:
            return None

    def try_repair_band(self, blocks) -> Optional[dict]:
        """`repair_band`, returning None on a saturation refusal. Others stay fatal."""
        try:
            return self.repair_band(blocks)
        except SaturationRefusal:
            return None

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
            raise SaturationRefusal(
                f"{self.ds_dir}: repair over blocks {list(blocks)} hit the saturation "
                f"guard ({len(self.ds.rows)} sweep rows) — refusing to report an "
                "interpolated zero")
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
        # A saturated arm is a REFUSAL, not an abort (see the module docstring). The lists
        # below stay index-aligned with `cells` and carry None for a refusal: dropping the
        # entry instead would let a later zip() truncate silently against per_ds.
        ba = cell.try_repair_band(("kint",) + POOLED_BLOCKS)
        bb = cell.try_repair_band(POOLED_BLOCKS)
        beta_a = ba.pop("beta") if ba is not None else None
        beta_b = bb.pop("beta") if bb is not None else None
        rep_a = cell.r_base * (1 - ba["registered"]) if ba is not None else None
        rep_b = cell.r_base * (1 - bb["registered"]) if bb is not None else None
        band_a.append(ba)
        band_b.append(bb)
        frac_a.append(ba["registered"] if ba is not None else None)
        frac_b.append(bb["registered"] if bb is not None else None)
        betas_a.append(beta_a[-len(names):] if beta_a is not None else None)
        # drop the fitted intercept, keep [msum]+cols
        betas_b.append(beta_b[1:] if beta_b is not None else None)
        X, _y = cell.design(POOLED_BLOCKS)
        Xi = np.hstack([np.ones((len(X), 1)), X])
        # saturation guard refusal -- recorded, never silently a pass or a zero.
        bt = cell.try_repair_band(T1_EXTENDED_BLOCKS)
        if bt is not None:
            bt.pop("beta")
        frac_t1x_saturated.append(bt is None)
        band_t1x.append(bt)
        per_ds.append({
            "ds": cell.ds_dir.name,
            "r_exact_pct": cell.r_base,
            "cell_a_repaired_pct": rep_a,
            "cell_a_fraction": cell.fraction(rep_a) if rep_a is not None else None,
            "cell_a_saturated": ba is None,
            "cell_b_repaired_pct": rep_b,
            "cell_b_fraction": cell.fraction(rep_b) if rep_b is not None else None,
            "cell_b_saturated": bb is None,
            "t1x_band": bt, "t1x_saturated": bt is None,
            "n_rows": len(cell.ds.rows), "n_feasible": len(cell.feasible),
            "kint_width": len(k_integer_keys(cell.ds)),
            "cond": float(np.linalg.cond(Xi)),
            "rank": int(np.linalg.matrix_rank(Xi)), "n_params": int(Xi.shape[1]),
            # No cell-B fit means no cell-B coefficients. `null` rather than an empty
            # dict, so a reader cannot mistake "refused" for "fitted, all zero".
            "coefficients": (dict(zip(shared_names, [float(v) for v in beta_b[1:]]))
                             if beta_b is not None else None),
        })
    out["per_dataset"] = per_ds

    valid_t1x_bands = [b for b in band_t1x if b is not None]
    out["t1x_per_dataset"] = {
        "n_cells": len(cells),
        "n_saturated": sum(frac_t1x_saturated),
        "n_fitted": len(valid_t1x_bands),
        "by_arm": _fitted_by_arm(cells, [b is not None for b in band_t1x]),
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
        "A_per_dataset_full_t1": _per_dataset_cell(cells, frac_a, band_a,
                                                   residual_gt_material=True),
        "B_per_dataset_no_kint": _per_dataset_cell(cells, frac_b, band_b),
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
    # Over the FITTED coefficient vectors only; a refused arm contributes no coefficients
    # to disperse. The counts are reported so a narrow dispersion over a collapsed
    # denominator cannot read as agreement across the corpus.
    fit_betas_b = [b for b in betas_b if b is not None]
    fit_betas_a = [b for b in betas_a if b is not None]
    out["dispersion_n_datasets"] = {"cell_b": len(fit_betas_b),
                                    "cell_a": len(fit_betas_a),
                                    "n_cells": len(cells)}
    arr_b = np.array(fit_betas_b)   # [msum] + shared columns, per dataset
    out["dispersion"] = ({name: dispersion(arr_b[:, j].tolist())
                          for j, name in enumerate(shared_names)}
                         if fit_betas_b else None)
    out["dispersion_full_t1_shared_columns"] = (
        {name: dispersion(np.array(fit_betas_a)[:, j].tolist())
         for j, name in enumerate(names)}
        if fit_betas_a else None)
    out["rank_deficient_datasets"] = [r["ds"] for r in per_ds
                                      if r["rank"] < r["n_params"]]

    # --- the verdict, applied as registered ------------------------------
    med_b, med_c, med_w = (out["cells"]["B_per_dataset_no_kint"]["median_fraction"],
                           out["cells"]["C_pooled_no_kint"]["median_fraction"],
                           out["cells"]["C_sensitivity_equal_dataset_weight"]
                           ["median_fraction"])
    # A refused cell B has no median, so the registered comparison has nothing to test.
    # That is a VOID with its own name — never a pass, never a silent zero, and never a
    # comparison against None. No threshold moves: the registered branches below are
    # reached with exactly their original values whenever the statistic exists.
    if med_b is None:
        verdict = "VOID-CELL-B-UNFITTABLE"
    elif med_b < REPAIR_MAX:
        verdict = "VOID-KINT-CONFOUNDED"
    elif med_c >= REPAIR_MAX:
        verdict = "BOUND-TRANSFERS"
    else:
        verdict = "BOUND-DOES-NOT-TRANSFER"
    straddle = (None if med_c is None or med_w is None
                else (med_c >= REPAIR_MAX) != (med_w >= REPAIR_MAX))
    out["verdict"] = verdict
    out["sensitivity_straddles_threshold"] = (None if straddle is None
                                              else bool(straddle))
    # Whether a cell's verdict survives the plans its own surrogate cannot separate. Cell
    # A (the §9a statistic) must be checked too — a NO-GO resting on a tie-break would be
    # no better than the ds_00008 "genuine tie" that turned out to be two verifier bugs.
    def _tie_indeterminate(key: str) -> Optional[bool]:
        opt = out["cells"][key]["median_optimistic"]
        reg = out["cells"][key]["median_fraction"]
        if opt is None or reg is None:
            return None      # nothing fitted — unknown, which is not "determinate"
        return bool((opt >= REPAIR_MAX) != (reg >= REPAIR_MAX))

    out["tie_indeterminate"] = {
        key: _tie_indeterminate(key)
        for key in ("A_per_dataset_full_t1", "B_per_dataset_no_kint")}

    # §9c(b): cell B IS the anonymous (dim36crk-expressible) closure, because `quad` is the
    # plan-level rendering of cols 25-28, `load_over_cap` of col 29, `overcap_tasks` of 31,
    # and min/max_hop_sum + transfer of 33-35. kint is the only block with no col in §2.
    cell_b = out["cells"]["B_per_dataset_no_kint"]
    readings = {k: cell_b[key] for k, key in
                (("mean_tied", "median_mean_tied"),
                 ("registered", "median_fraction"),
                 ("pessimistic", "median_pessimistic"))}
    directions = {k: (None if v is None else v >= REPAIR_MAX)
                  for k, v in readings.items()}
    if any(v is None for v in directions.values()):
        # Cell B was refused on every dataset the guard saw: there is no closure reading
        # to be indeterminate ABOUT. Distinct from VOID-TIE-INDETERMINATE, which means a
        # reading exists and its band straddles the threshold.
        anon_verdict = "VOID-CELL-B-UNFITTABLE"
    elif len(set(directions.values())) > 1:
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
    n_saturated = 0
    for gi, cell in enumerate(cells):
        # A refused kint fit yields no coefficients to regress on node features; that
        # dataset contributes nothing rather than aborting the §9c(a) reading.
        got = cell.try_repair(("kint",) + POOLED_BLOCKS)
        if got is None:
            n_saturated += 1
            continue
        _rep, beta = got
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
    if not rows:
        return {"n_coefficients": 0, "n_datasets": 0,
                "n_saturated": n_saturated,
                "features": ["intercept"] + (feat_names or []),
                "r2_heldout_by_dataset": None, "r2_in_sample": None,
                "target": "kint coefficient centered within (dataset, task_type)",
                "target_sd": None,
                "note": "every kint fit was REFUSED by the saturation guard — §9c(a) "
                        "has no coefficients to regress; not a zero, not a reading"}
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
            "n_saturated": n_saturated,
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
        # Arms differ in width, so they saturate independently: on the pivot's 16-row arm
        # `hop+coupling` (7 params) fits and `full T1` (21) does not. One refused arm must
        # not cost the others their table — it is recorded per arm instead.
        fractions: List[Optional[float]] = []
        fitted, residual = [], 0
        for cell in cells:
            got = cell.try_repair(blocks)
            if got is None:
                fractions.append(None)
                fitted.append(False)
                continue
            repaired, _beta = got
            fractions.append(cell.fraction(repaired))
            fitted.append(True)
            residual += 1 if repaired > MATERIAL_PCT else 0
        fit_fracs = [f for f in fractions if f is not None]
        out[name] = {"blocks": list(blocks),
                     "median_fraction": _median_or_none(fractions),
                     "mean_fraction": float(np.mean(fit_fracs)) if fit_fracs else None,
                     "n_closed_ge_half": sum(1 for f in fit_fracs if f >= REPAIR_MAX),
                     "residual_gt_material": residual,
                     "fractions": fractions,
                     "n_fitted": len(fit_fracs),
                     "n_saturated": len(fractions) - len(fit_fracs),
                     "by_arm": _fitted_by_arm(cells, fitted)}
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
        # A refused cell A leaves the residual undecidable for that dataset: it is neither
        # "residual" nor "closed", so it joins neither group rather than defaulting into
        # one. The group sizes are reported, so a shrunken contrast is visible.
        rows.append({
            "ds": row["ds"],
            "residual": (None if row["cell_a_fraction"] is None else
                         row["r_exact_pct"] * (1 - row["cell_a_fraction"]) > MATERIAL_PCT),
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
    groups = {"residual": [r for r in rows if r["residual"] is True],
              "closed": [r for r in rows if r["residual"] is False]}
    contrast = {k: {g: (float(np.median([r[k] for r in rs])) if rs else None)
                    for g, rs in groups.items()}
                for k in keys}
    return {"per_dataset": rows, "n_residual": len(groups["residual"]),
            "n_closed": len(groups["closed"]),
            "n_undecidable_cell_a_saturated": sum(1 for r in rows
                                                  if r["residual"] is None),
            "median_contrast": contrast}


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
    # Emitted ALWAYS, with its refusals accounted for, rather than vanishing when any one
    # dataset is refused: a silently absent block cannot be told apart from "not
    # applicable", and this one is what the pivot screen's S3 bar points at.
    _kr_ok = [v is not None for v in kr]
    _kr_fit = [v for v in kr if v is not None]
    out["krank_exploratory"] = {
        "note": "occupancy by identity-free node rank + the dim36crk set; NOT "
                "registered, no verdict read from it",
        "blocks": list(krank_pool_blocks),
        "ds": [cell.ds_dir.name for cell in cells],
        "fractions": kr,
        "bands": kr_bands,
        "median_fraction": _median_or_none(kr),
        "mean_fraction": float(np.mean(_kr_fit)) if _kr_fit else None,
        "n_closed_ge_half": sum(1 for f in _kr_fit if f >= REPAIR_MAX),
        "n_fitted": len(_kr_fit),
        "n_saturated": len(kr) - len(_kr_fit),
        "by_arm": _fitted_by_arm(cells, _kr_ok)}

    if True:
        # Unlike kint, krank IS poolable: identity-free index, fixed width across datasets
        # with the same node count. This is the follow-up the §9b VOID named. Still
        # exploratory — no verdict is read from it.
        #
        # De-nested from the per-dataset `all(fitted)` condition above (2026-08-27): this
        # fit stacks EVERY row of EVERY dataset into one design with per-dataset
        # intercepts and solves for ONE shared coefficient set, so a per-dataset
        # saturation cannot make it interpolate — the row count it fits against is the
        # corpus's, not any single dataset's. Gating it on a per-dataset refusal withheld
        # the pooled statistic for a reason that does not apply to it. The
        # rows-vs-parameters ratio is reported below so the reader can see the headroom
        # rather than take it on trust.
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
                "n_closed_ge_half": sum(1 for f in pooled_kr if f >= REPAIR_MAX),
                # Headroom against the SAME 2x criterion the per-dataset guard applies,
                # reported not enforced: the pooled design's rows are the corpus's, so
                # this is the number that says whether pooling actually bought the
                # room a per-dataset fit lacked.
                "pooled_fit_rows": int(len(shared)),
                "pooled_fit_params": int(shared.shape[1] + n),
                "pooled_fit_clears_2x": bool(len(shared)
                                             >= 2 * (shared.shape[1] + n))}

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)

    c = out["cells"]
    print("\n=== §9b cells (median repair fraction over the firing datasets) ===")
    for key in ("A_per_dataset_full_t1", "B_per_dataset_no_kint", "C_pooled_no_kint",
                "C_sensitivity_equal_dataset_weight"):
        row = c[key]
        band = ""
        if "median_optimistic" in row:
            band = (f"  tie-band [{_fmt(row['median_pessimistic'])}, "
                    f"{_fmt(row['median_optimistic'])}] "
                    f"({row['datasets_with_prediction_ties']}/{out['firing']} tied, "
                    f"max group {row['max_tie_group']})")
        elif "median_optimistic" in row.get("fit", {}):
            band = (f"  tie-band [{_fmt(row['fit']['median_pessimistic'])}, "
                    f"{_fmt(row['fit']['median_optimistic'])}]")
        sat = ""
        if row.get("n_saturated"):
            sat = (f"  [{row['n_fitted']}/{row['n_fitted'] + row['n_saturated']} fitted, "
                   f"{row['n_saturated']} REFUSED by the saturation guard; by arm "
                   f"{row['by_arm']}]")
        print(f"  {key:38s} median={_fmt(row['median_fraction'])} "
              f"mean={_fmt(row['mean_fraction'])} "
              f">=0.5: {row['n_closed_ge_half']:2d}/{out['firing']}{band}{sat}")
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
    dn = out["dispersion_n_datasets"]
    if dn["cell_b"] != dn["n_cells"]:
        print(f"  !! over {dn['cell_b']}/{dn['n_cells']} datasets — the rest were "
              "REFUSED by the saturation guard and contribute no coefficients")
    for name, d in (out["dispersion"] or {}).items():
        ratio = f"{d['sd_over_mean']:.3f}" if d["sd_over_mean"] is not None else "  n/a"
        print(f"  {name:18s} mean={d['mean']:14.4f} sd={d['sd']:14.4f} "
              f"sd/mean={ratio}  IQR={d['iqr']:.4f}")
    if out["rank_deficient_datasets"]:
        print(f"  rank-deficient designs: {out['rank_deficient_datasets']}")

    print("\n=== block attribution (per-dataset fits) ===")
    for name, row in out["ablation"].items():
        sat = (f"  [{row['n_fitted']}/{out['firing']} fitted, {row['n_saturated']} "
               f"REFUSED; by arm {row['by_arm']}]") if row["n_saturated"] else ""
        print(f"  {name:32s} median={_fmt(row['median_fraction'])} "
              f"mean={_fmt(row['mean_fraction'])} "
              f">=0.5: {row['n_closed_ge_half']:2d}/{out['firing']} "
              f"residual>5%: {row['residual_gt_material']}{sat}")

    rc = out["residual_characterization"]
    print(f"\n=== residual stratum ({rc['n_residual']} residual vs {rc['n_closed']} "
          "closed; DESCRIPTIVE, below materiality) ===")
    if rc.get("n_undecidable_cell_a_saturated"):
        print(f"  !! {rc['n_undecidable_cell_a_saturated']} dataset(s) undecidable — "
              "cell A was REFUSED by the saturation guard, so they join neither group")
    for key, row in rc["median_contrast"].items():
        print(f"  {key:32s} residual={_fmt(row['residual'], '12.4f', 12)} "
              f"closed={_fmt(row['closed'], '12.4f', 12)}")
    ac = out["anonymous_closure"]
    print("\n=== §9c(b) ANONYMOUS CLOSURE — cell B is the dim36crk-expressible set ===")
    for name, value in ac["readings"].items():
        side = "n/a" if value is None else ('>=' if value >= REPAIR_MAX else '< ')
        print(f"  {name:14s} median={_fmt(value)}   {side} {REPAIR_MAX}")
    print(f"  {'optimistic':14s} median="
          f"{_fmt(ac['optimistic_upper_bound_not_a_verdict'])}   "
          "(upper bound only, NOT a decoder reading)")
    print(f"\n  §9c(b) VERDICT: {ac['verdict']}")

    iof = out["identity_or_features"]
    print("\n=== §9c(a) is kint identity or features? ===")
    print(f"  held-out-by-dataset R² = {_fmt(iof['r2_heldout_by_dataset'])}  "
          f"(in-sample {_fmt(iof['r2_in_sample'])}) over {iof['n_coefficients']} "
          f"coefficients, {iof['n_datasets']} datasets"
          + (f"  [{iof['n_saturated']} REFUSED]" if iof.get("n_saturated") else ""))
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
