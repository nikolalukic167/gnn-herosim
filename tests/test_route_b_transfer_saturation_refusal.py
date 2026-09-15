"""route_b_coefficient_transfer: a saturated arm is a REFUSAL, not an abort.

Why this file exists (route_b_env_pivot_v1, 2026-08-27). The saturation guard fires per
(dataset, block set) — a fit with fewer than 2x as many rows as parameters could
interpolate the sweep, so its regret is refused rather than reported as a clean zero.
That part was right. What was wrong is what the transfer tool did with a refusal: ONE
refused arm raised and aborted the whole run, so arms that fit perfectly well became
unreadable.

That is not hypothetical. On the pivot's H0/H1 the S4 bar block `hop+coupling` is 7
parameters and needs 14 rows, so it fits in BOTH the 16-row and the 64-row arm — yet no
S4 attribution table could be produced on either rung, because the unrelated full-T1 arm
(21 parameters, needs 42) refused on the 16-row arm and took the run down with it.

Every test below fails against the pre-fix code, most of them with the RuntimeError the
fix removes. The teeth are the point: a "tolerate the refusal" change is exactly the kind
that can silently become "tolerate everything", so `test_only_saturation_is_tolerated`
pins the boundary.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts_cosim"))

import route_b_coefficient_transfer as tr  # noqa: E402

from test_route_b_repair_fixtures import (  # noqa: E402
    TOY_CANDIDATES,
    TOY_TASK_TYPES,
    coupled_rtt,
    write_toy,
)

ALPHA = 2.0

# The narrow rig drops cnn's third replica (platform 42 on n1), taking the sweep from 54
# rows to 36. Full T1 is 2 + 21 = 23 parameters and needs 46, so it is REFUSED; the S4 bar
# block `hop+coupling` is 2 + 5 = 7 parameters and needs 14, so it FITS. One rig, both
# sides of the guard — the same shape as the pivot's 16-row arm.
#
# Not `TOY_CANDIDATES_NARROW` from the fixtures module, deliberately: that one drops a
# task-0 replica and lands in a NON-firing cell (r_base = 0), which the transfer tool can
# never be handed — `run()` builds cells from the firing set only, and `fraction()` divides
# by r_base. The drop below leaves r_base at the wide rig's own 7.2848, so wide and narrow
# are the SAME firing cell differing only in sweep width, which is what makes the
# fitted-vs-refused comparison below controlled rather than confounded.
NARROW_CANDIDATES = {**TOY_CANDIDATES, 3: [(0, 40), (0, 41)]}

WIDE_ROWS = 54
NARROW_ROWS = 36


def _cell(tmp_path, name, candidates):
    ds_dir = write_toy(tmp_path / name, coupled_rtt, candidates)
    return tr.Cell(ds_dir, TOY_TASK_TYPES, ALPHA)


def _wide(tmp_path, name="wide"):
    return _cell(tmp_path, name, TOY_CANDIDATES)


def _narrow(tmp_path, name="narrow"):
    return _cell(tmp_path, name, NARROW_CANDIDATES)


# ---------------------------------------------------------------------------
# the rig itself — if these drift, every expectation below is meaningless
# ---------------------------------------------------------------------------

def test_the_rig_straddles_the_guard(tmp_path):
    narrow, wide = _narrow(tmp_path), _wide(tmp_path)
    assert len(narrow.ds.rows) == NARROW_ROWS
    assert len(wide.ds.rows) == WIDE_ROWS
    assert narrow.try_repair(("kint", "quad", "cap", "hop", "coupling")) is None, \
        "full T1 must be REFUSED on the narrow toy or this file tests nothing"
    assert narrow.try_repair(("hop", "coupling")) is not None, \
        "hop+coupling must FIT on the narrow toy — it is the S4 bar block"
    # both rigs are the SAME firing cell: only the sweep width differs, so a difference
    # in what fits cannot be a difference in the physics
    assert narrow.r_base > 0 and wide.r_base > 0
    assert narrow.r_base == pytest.approx(wide.r_base)


# ---------------------------------------------------------------------------
# 1. the defect: one refused arm must not cost the others their table
# ---------------------------------------------------------------------------

def test_a_refused_arm_does_not_abort_the_fittable_arms(tmp_path):
    """Pre-fix this raised SaturationRefusal out of `ablation()` and produced nothing."""
    out = tr.ablation([_narrow(tmp_path)])

    refused = out["full T1"]
    assert refused["median_fraction"] is None
    assert refused["n_fitted"] == 0 and refused["n_saturated"] == 1

    fitted = out["parent-coupling (hop+coupling)"]
    assert fitted["n_fitted"] == 1 and fitted["n_saturated"] == 0
    assert isinstance(fitted["median_fraction"], float)


def test_a_refused_median_is_none_never_zero(tmp_path):
    """A zero here would read as 'the competitor closes nothing' — the opposite of
    'we refused to let it interpolate'. The guard exists to prevent exactly that."""
    out = tr.ablation([_narrow(tmp_path)])
    refused = [name for name, row in out.items() if row["n_saturated"]]
    fitted = [name for name, row in out.items() if not row["n_saturated"]]
    # On this rig exactly `full T1` (23 params, needs 46 > 36) is over the guard; the
    # narrower arms all fit. Both lists must be non-empty or the test is vacuous.
    assert refused == ["full T1"], refused
    assert fitted, "every arm refused — the rig no longer separates the two cases"
    for arm in refused:
        assert out[arm]["median_fraction"] is None
        assert out[arm]["mean_fraction"] is None
        assert out[arm]["n_closed_ge_half"] == 0
        assert out[arm]["fractions"] == [None]
    # and a genuine 0.0 stays 0.0 — the refusal must not be confused with "closes
    # nothing", which is a real and different reading that other arms do produce here
    assert out["occupancy (kint+quad+cap)"]["median_fraction"] == 0.0
    assert out["occupancy (kint+quad+cap)"]["n_saturated"] == 0


def test_mixed_corpus_medians_are_over_the_fitted_subset_only(tmp_path):
    """Two cells, one fittable and one refused, on the arm that straddles."""
    cells = [_wide(tmp_path), _narrow(tmp_path)]
    out = tr.ablation(cells)

    full = out["full T1"]
    assert (full["n_fitted"], full["n_saturated"]) == (1, 1)
    assert full["fractions"][1] is None
    # the median is the single fitted value, not a median taken over a None
    assert full["median_fraction"] == pytest.approx(full["fractions"][0])

    both = out["parent-coupling (hop+coupling)"]
    assert (both["n_fitted"], both["n_saturated"]) == (2, 0)


# ---------------------------------------------------------------------------
# 2. the denominator must be visible, keyed on the arm (preflight step 3)
# ---------------------------------------------------------------------------

def test_by_arm_breakdown_keys_on_the_unconstrained_sweep_size(tmp_path):
    """A median whose denominator has collapsed onto one arm must say so in its own
    block — this is the confound that has already cost this lineage two sessions."""
    cells = [_wide(tmp_path), _narrow(tmp_path)]
    out = tr.ablation(cells)

    assert out["full T1"]["by_arm"] == {
        str(WIDE_ROWS): {"fitted": 1, "saturated": 0},
        str(NARROW_ROWS): {"fitted": 0, "saturated": 1},
    }
    assert out["parent-coupling (hop+coupling)"]["by_arm"] == {
        str(WIDE_ROWS): {"fitted": 1, "saturated": 0},
        str(NARROW_ROWS): {"fitted": 1, "saturated": 0},
    }


def test_per_dataset_cell_reports_its_own_refusals(tmp_path):
    cells = [_wide(tmp_path), _narrow(tmp_path)]
    bands = [c.try_repair_band(("kint",) + tr.POOLED_BLOCKS) for c in cells]
    fracs = [b["registered"] if b is not None else None for b in bands]

    block = tr._per_dataset_cell(cells, fracs, bands, residual_gt_material=True)
    assert (block["n_fitted"], block["n_saturated"]) == (1, 1)
    assert block["by_arm"][str(NARROW_ROWS)] == {"fitted": 0, "saturated": 1}
    assert block["median_fraction"] == pytest.approx(fracs[0])
    # a refused dataset cannot contribute to the residual count either
    assert block["residual_gt_material"] <= 1


def test_a_cell_with_nothing_fitted_reports_none_not_an_empty_median(tmp_path):
    cells = [_narrow(tmp_path)]
    bands = [c.try_repair_band(("kint",) + tr.POOLED_BLOCKS) for c in cells]
    block = tr._per_dataset_cell(cells, bands and [None], bands)
    assert block["median_fraction"] is None
    assert block["mean_fraction"] is None
    assert block["max_tie_group"] is None
    assert block["n_fitted"] == 0


# ---------------------------------------------------------------------------
# 3. the boundary: tolerate the refusal and NOTHING else
# ---------------------------------------------------------------------------

def test_only_saturation_is_tolerated(tmp_path, monkeypatch):
    """`except RuntimeError` at the call site would also swallow 'no feasible rows',
    'non-positive optimum' and the firing-set disagreement — every one of which must
    stay fatal. SaturationRefusal is its own class so it cannot."""
    cell = _wide(tmp_path)

    def boom(_blocks):
        raise RuntimeError("a real bug, not a saturation refusal")

    monkeypatch.setattr(cell, "repair", boom)
    with pytest.raises(RuntimeError, match="a real bug"):
        cell.try_repair(("hop", "coupling"))

    monkeypatch.setattr(cell, "repair_band", boom)
    with pytest.raises(RuntimeError, match="a real bug"):
        cell.try_repair_band(("hop", "coupling"))


def test_saturation_refusal_is_a_runtimeerror_subclass():
    """Any pre-existing `except RuntimeError` handler keeps working unchanged."""
    assert issubclass(tr.SaturationRefusal, RuntimeError)


def test_repair_still_raises_by_default(tmp_path):
    """The refusal is opt-in via try_*; the plain call stays fail-loud."""
    narrow = _narrow(tmp_path)
    with pytest.raises(tr.SaturationRefusal, match="saturation guard"):
        narrow.repair(("kint", "quad", "cap", "hop", "coupling"))
    with pytest.raises(tr.SaturationRefusal, match="saturation guard"):
        narrow.repair_band(("kint", "quad", "cap", "hop", "coupling"))


def test_the_refusal_message_names_the_sweep_size(tmp_path):
    """Whoever reads the message needs the number that decides the fix, not just 'it
    saturated' — the pivot's answer turned on 64 rows against a needed 82."""
    with pytest.raises(tr.SaturationRefusal, match=f"{NARROW_ROWS} sweep rows"):
        _narrow(tmp_path).repair(("kint", "quad", "cap", "hop", "coupling"))


# ---------------------------------------------------------------------------
# 4. inertness where the fix claims to be inert
# ---------------------------------------------------------------------------

def test_nothing_changes_when_no_arm_saturates(tmp_path):
    """On a corpus where every arm fits, the refusal path never runs and every
    statistic is what it was. (The corpus-scale version of this is the byte-identity
    diff against the frozen pilot artifact: 0 pre-existing values moved.)"""
    cells = [_wide(tmp_path, "w1"), _wide(tmp_path, "w2")]
    out = tr.ablation(cells)
    for arm in out.values():
        assert arm["n_saturated"] == 0
        assert arm["n_fitted"] == 2
        assert None not in arm["fractions"]
        assert arm["median_fraction"] == pytest.approx(
            tr.registered_median(arm["fractions"]))
        assert arm["mean_fraction"] == pytest.approx(float(np.mean(arm["fractions"])))
