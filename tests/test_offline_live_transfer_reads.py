"""Tests for offline_live_transfer_v1's R0/R1 read.

The bars are constants in the module and were committed before the registered families
were read; these tests pin the machinery that applies them, not the bars' values.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts_cosim import offline_live_transfer_v1_pairing_read as R  # noqa: E402

scipy_stats = pytest.importorskip("scipy.stats")


# --------------------------------------------------------------------------- stats
def test_spearman_matches_scipy_on_a_monotone_series():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    ys = [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0]
    rho, p = R.spearman(xs, ys)
    ref = scipy_stats.spearmanr(xs, ys)
    assert rho == pytest.approx(ref.statistic, abs=1e-12)
    assert p == pytest.approx(ref.pvalue, rel=1e-6)


def test_spearman_matches_scipy_with_ties():
    xs = [1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    ys = [5.0, 4.0, 4.0, 3.0, 9.0, 1.0, 2.0, 8.0, 7.0, 6.0]
    rho, p = R.spearman(xs, ys)
    ref = scipy_stats.spearmanr(xs, ys)
    assert rho == pytest.approx(ref.statistic, abs=1e-12)
    assert p == pytest.approx(ref.pvalue, rel=1e-6)


def test_spearman_matches_scipy_on_an_anticorrelated_series():
    xs = [float(i) for i in range(16)]
    ys = [float(-i) + (i % 3) for i in range(16)]
    rho, p = R.spearman(xs, ys)
    ref = scipy_stats.spearmanr(xs, ys)
    assert rho < 0
    assert rho == pytest.approx(ref.statistic, abs=1e-12)
    assert p == pytest.approx(ref.pvalue, rel=1e-6)


def test_spearman_is_exactly_one_on_a_perfect_ranking():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    rho, p = R.spearman(xs, [10.0, 20.0, 30.0, 40.0, 50.0])
    assert rho == pytest.approx(1.0)
    assert p == 0.0


def test_pearson_refuses_a_constant_series_instead_of_returning_zero():
    with pytest.raises(R.TransferReadError, match="constant"):
        R.pearson([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0])


def test_correlating_mismatched_lengths_fails_loud():
    with pytest.raises(R.TransferReadError, match="differ in length"):
        R.pearson([1.0, 2.0, 3.0], [1.0, 2.0])


def test_zscore_centres_and_scales():
    zs = R.zscore([1.0, 2.0, 3.0, 4.0])
    assert sum(zs) == pytest.approx(0.0, abs=1e-12)
    assert sum(z * z for z in zs) / len(zs) == pytest.approx(1.0)


def test_zscore_refuses_a_constant_cell():
    with pytest.raises(R.TransferReadError, match="constant"):
        R.zscore([3.0, 3.0, 3.0])


def test_zscore_removes_a_cell_offset_but_keeps_its_ordering():
    a = R.zscore([10.0, 11.0, 12.0, 13.0])
    b = R.zscore([110.0, 111.0, 112.0, 113.0])
    assert a == pytest.approx(b)


# --------------------------------------------------------------------- burned families
@pytest.mark.parametrize("burned", sorted(R.BURNED_RESULT_DIRS))
def test_the_burned_live_families_are_refused(tmp_path, burned):
    (tmp_path / burned).mkdir()
    with pytest.raises(R.TransferReadError, match="BURNED"):
        R.load_live(tmp_path, burned)


@pytest.mark.parametrize(
    "run_re",
    [r"peer-affinity-v1-t1b-(gnn|mpoff)-lr2e3-seed(\d+)",
     r"drainable-objective-v1-v1-(gnn|mpoff)-seed(\d+)"],
)
def test_the_burned_offline_families_are_refused(tmp_path, run_re):
    with pytest.raises(R.TransferReadError, match="BURNED"):
        R.load_offline(tmp_path, run_re)


def test_the_registered_families_are_not_burned():
    for spec in R.FAMILIES.values():
        assert spec["results"] not in R.BURNED_RESULT_DIRS
        assert not any(b in spec["run_re"] for b in R.BURNED_RUN_PATTERNS)


def test_three_families_are_registered():
    assert len(R.FAMILIES) == 3


# ------------------------------------------------------------------------- loading
def _write_live(root: Path, results: str, rows):
    d = root / results
    d.mkdir(parents=True, exist_ok=True)
    for arm, seed, latency, queue in rows:
        (d / f"{arm}_s{seed}.summary.json").write_text(json.dumps({
            "arm": f"{arm}_s{seed}",
            R.LIVE_KEY: latency,
            R.LIVE_CONTROL_KEY: queue,
        }))
    return d


def _write_runs(root: Path, prefix: str, rows):
    for i, (arm, seed, offline) in enumerate(rows):
        d = root / f"run-2026_{prefix}_{i:03d}" / "files"
        d.mkdir(parents=True, exist_ok=True)
        (d / "output.log").write_text(f"Syncing run {prefix}-{arm}-lr2e3-seed{seed}\n")
        (d / "wandb-summary.json").write_text(json.dumps({
            R.OFFLINE_PRIMARY_KEY: offline,
            R.OFFLINE_SECONDARY_KEY: offline + 20.0,
        }))


def test_reactive_baseline_summaries_are_skipped_not_paired(tmp_path):
    d = tmp_path / "some_gate"
    d.mkdir()
    (d / "knative_network.summary.json").write_text(json.dumps(
        {"arm": "knative_network", R.LIVE_KEY: 25.9, R.LIVE_CONTROL_KEY: 10.0}))
    _write_live(tmp_path, "some_gate", [("gnn", 1, 50.0, 40.0)])
    live = R.load_live(tmp_path, "some_gate")
    assert set(live) == {("gnn", 1)}


def test_a_live_summary_missing_the_statistic_fails_loud(tmp_path):
    d = tmp_path / "g"
    d.mkdir()
    (d / "gnn_s1.summary.json").write_text(json.dumps({"arm": "gnn_s1", R.LIVE_KEY: 1.0}))
    with pytest.raises(R.TransferReadError, match=R.LIVE_CONTROL_KEY):
        R.load_live(tmp_path, "g")


def test_an_unfinished_wandb_run_fails_loud_rather_than_reshaping_the_sample(tmp_path):
    d = tmp_path / "run-x" / "files"
    d.mkdir(parents=True)
    (d / "output.log").write_text("Syncing run fam-gnn-lr2e3-seed1\n")
    (d / "wandb-summary.json").write_text(json.dumps({"_step": 12}))
    with pytest.raises(R.TransferReadError, match="cannot be paired"):
        R.load_offline(tmp_path, r"fam-(gnn|mpoff)-lr2e3-seed(\d+)")


def test_two_runs_matching_one_seed_is_an_ambiguous_pairing(tmp_path):
    _write_runs(tmp_path, "fam", [("gnn", 1, 50.0), ("gnn", 1, 51.0)])
    with pytest.raises(R.TransferReadError, match="ambiguous"):
        R.load_offline(tmp_path, r"fam-(gnn|mpoff)-lr2e3-seed(\d+)")


def test_a_short_cell_is_refused_rather_than_read(tmp_path):
    gate, wb = tmp_path / "gate", tmp_path / "wandb"
    rows_live, rows_off = [], []
    for arm in R.ARMS:
        n = 16 if arm == "mpoff" else R.MIN_SEEDS_PER_CELL - 1
        for s in range(1, n + 1):
            rows_live.append((arm, s, 50.0 + s, 40.0 + s))
            rows_off.append((arm, s, 100.0 + s))
    _write_live(gate, "fam_gate", rows_live)
    _write_runs(wb, "fam", rows_off)
    with pytest.raises(R.TransferReadError, match="paired seeds"):
        R.pair_family(gate, wb, "FX", {"results": "fam_gate",
                                       "run_re": r"fam-(gnn|mpoff)-lr2e3-seed(\d+)"})


# --------------------------------------------------------------------------- reads
def _family(name, *, offline_to_live_sign, queue_tracks_latency=True, n=16):
    """A synthetic family. sign=+1 makes offline predict live, -1 anti-predicts, 0 noise."""
    cells = {}
    noise = [0.31, -0.22, 0.07, 0.44, -0.38, 0.12, -0.05, 0.27,
             -0.41, 0.19, 0.02, -0.33, 0.36, -0.14, 0.23, -0.29]
    for arm in R.ARMS:
        offline = [100.0 + i for i in range(n)]
        if offline_to_live_sign == 0:
            live = [50.0 + 10.0 * noise[i % len(noise)] for i in range(n)]
        else:
            live = [50.0 + offline_to_live_sign * float(i) for i in range(n)]
        queue = list(live) if queue_tracks_latency else [float(n - i) for i in range(n)]
        cells[arm] = {"seeds": [float(i) for i in range(1, n + 1)],
                      "offline": offline, "offline_test": offline,
                      "live": live, "live_queue": queue}
    return {"family": name, "results_dir": name.lower(), "cells": cells}


def test_r0_passes_when_queue_tracks_latency():
    fams = [_family(n, offline_to_live_sign=0) for n in ("F1", "F2", "F3")]
    r0 = R.read_r0(fams)
    assert r0["passes"] and r0["verdict"] == "CONTROL-PASSES"


def test_r0_failing_in_one_family_voids_everything():
    fams = [_family("F1", offline_to_live_sign=1),
            _family("F2", offline_to_live_sign=1),
            _family("F3", offline_to_live_sign=1, queue_tracks_latency=False)]
    out = R.read(fams)
    assert out["R0"]["verdict"] == "CONTROL-FAILS-EVERYTHING-VOID"
    assert out["outcome"] == "VOID-R0-FAILED"
    assert out["R1"]["fires"] is None


def test_r1_fires_when_offline_perfectly_predicts_live():
    fams = [_family(n, offline_to_live_sign=1) for n in ("F1", "F2", "F3")]
    out = R.read(fams)
    assert out["R1"]["verdict"] == "OFFLINE-INFORMATIVE"
    assert out["R1"]["within_arm_bar"]["fires"]
    assert out["R1"]["pooled_z"]["rho"] == pytest.approx(1.0)


def test_r1_fires_on_a_consistent_anticorrelation_too():
    """The bar is on |rho|: a metric that reliably points the wrong way is informative."""
    fams = [_family(n, offline_to_live_sign=-1) for n in ("F1", "F2", "F3")]
    out = R.read(fams)
    assert out["R1"]["verdict"] == "OFFLINE-INFORMATIVE"
    assert out["R1"]["pooled_z"]["rho"] < 0


def test_r1_does_not_fire_on_noise():
    fams = [_family(n, offline_to_live_sign=0) for n in ("F1", "F2", "F3")]
    out = R.read(fams)
    assert out["R1"]["verdict"] == "OFFLINE-UNINFORMATIVE"
    assert not out["R1"]["within_arm_bar"]["fires"]
    assert not out["R1"]["pooled_z"]["fires"]


def test_within_arm_bar_needs_the_same_arm_in_two_families():
    """One family firing is not enough, however strong it is."""
    fams = [_family("F1", offline_to_live_sign=1),
            _family("F2", offline_to_live_sign=0),
            _family("F3", offline_to_live_sign=0)]
    r1 = R.read_r1(fams)
    assert r1["within_arm"]["F1/gnn"]["fires"]
    assert not r1["within_arm_bar"]["fires"]


def test_within_arm_bar_needs_one_direction_not_two():
    """Two families firing in opposite directions is noise, not a relationship."""
    fams = [_family("F1", offline_to_live_sign=1),
            _family("F2", offline_to_live_sign=-1),
            _family("F3", offline_to_live_sign=0)]
    r1 = R.read_r1(fams)
    assert r1["within_arm"]["F1/gnn"]["fires"] and r1["within_arm"]["F2/gnn"]["fires"]
    assert not r1["within_arm_bar"]["fires"]


def test_pooled_z_strips_an_arm_offset_that_the_raw_pooled_read_keeps():
    """The registered reason pooled-z is primary and pooled-raw decides nothing."""
    fams = []
    for name in ("F1", "F2", "F3"):
        fam = _family(name, offline_to_live_sign=0)
        # gnn scores better offline and worse live -- the shape of the reversal.
        fam["cells"]["gnn"]["offline"] = [x - 50.0 for x in fam["cells"]["gnn"]["offline"]]
        fam["cells"]["gnn"]["live"] = [x + 20.0 for x in fam["cells"]["gnn"]["live"]]
        fam["cells"]["gnn"]["live_queue"] = list(fam["cells"]["gnn"]["live"])
        fams.append(fam)
    r1 = R.read_r1(fams)
    assert r1["pooled_raw_decides_nothing"]["rho"] < -0.5   # the "reversal"
    assert abs(r1["pooled_z"]["rho"]) < R.R1_POOLED_Z_MIN_ABS_RHO
    assert r1["verdict"] == "OFFLINE-UNINFORMATIVE"


def test_reading_a_subset_of_the_registered_families_is_refused():
    fams = [_family(n, offline_to_live_sign=0) for n in ("F1", "F2")]
    with pytest.raises(R.TransferReadError, match="registered"):
        R.read(fams)


def test_the_registered_expectation_is_recorded_in_the_output():
    fams = [_family(n, offline_to_live_sign=0) for n in ("F1", "F2", "F3")]
    out = R.read(fams)
    assert out["R1"]["registered_expectation"] == "OFFLINE-UNINFORMATIVE"


def test_bars_are_the_values_signed_in_the_node():
    assert R.R0_CONTROL_MIN_RHO == 0.90
    assert R.R1_WITHIN_MIN_ABS_RHO == 0.50
    assert R.R1_WITHIN_MIN_FAMILIES == 2
    assert R.R1_POOLED_Z_MIN_ABS_RHO == 0.30
    assert R.R1_ALPHA == 0.05
    assert R.MIN_SEEDS_PER_CELL == 12


def test_end_to_end_through_the_loaders(tmp_path):
    gate, wb = tmp_path / "gate", tmp_path / "wandb"
    rows_live, rows_off = [], []
    for arm in R.ARMS:
        for s in range(1, 17):
            rows_live.append((arm, s, 50.0 + s, 40.0 + s))
            rows_off.append((arm, s, 100.0 + s))
    _write_live(gate, "fam_gate", rows_live)
    _write_runs(wb, "fam", rows_off)
    fam = R.pair_family(gate, wb, "FX",
                        {"results": "fam_gate", "run_re": r"fam-(gnn|mpoff)-lr2e3-seed(\d+)"})
    assert len(fam["cells"]["gnn"]["seeds"]) == 16
    out = R.read([fam, dict(fam, family="FY"), dict(fam, family="FZ")])
    assert out["R0"]["passes"]
    assert out["R1"]["verdict"] == "OFFLINE-INFORMATIVE"
