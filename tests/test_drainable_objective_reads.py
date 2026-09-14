"""Tests for the drainable_objective_v1 Phase A read tools.

Committed before the reads run, per the lineage's own discipline: a bar whose tool was
debugged against its data is not a bar. Every test here is on synthetic input whose
answer is known by construction.

    PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH=. pipenv run python3 \
        -m pytest tests/test_drainable_objective_reads.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts_cosim import drainable_objective_v1_clock_read as a1
from scripts_cosim import drainable_objective_v1_externality_read as a3
from scripts_cosim import drainable_objective_v1_label_reversal_read as a2


# ==========================================================================
# A1 -- the clock gap
# ==========================================================================


def _snapshot_line(snapshot_id, candidates):
    return json.dumps(
        {
            "snapshot_id": snapshot_id,
            "tasks": [{"task_id": 0, "candidates": candidates}],
        }
    )


def _cand(ptype, queue_length, drain, key=None):
    return {
        "platform_type": ptype,
        "queue_length": queue_length,
        "queue_drain_seconds": drain,
        "queue_key": key or f"n:{ptype}",
    }


def _write_snapshots(tmp_path, lines) -> Path:
    p = tmp_path / "snap.jsonl"
    p.write_text("\n".join(lines) + "\n")
    return p


def _write_corpus(tmp_path, cells) -> Path:
    """cells: {task_type: [platform_type, ...]} -> a one-dataset corpus dir."""
    corpus = tmp_path / "corpus"
    ds = corpus / "ds_00000"
    ds.mkdir(parents=True)
    replicas = {
        ttype: [
            {"node_name": "n", "platform_id": 100 + i, "platform_type": p}
            for i, p in enumerate(ptypes)
        ]
        for ttype, ptypes in cells.items()
    }
    (ds / "infrastructure.json").write_text(json.dumps({"replica_placements": replicas}))
    return corpus


TYPE_DB = {
    "dnn1": {
        "executionTime": {"rpiCpu": 0.003},
        "stateSize": {"app": {"input": 0, "output": 0}},
    }
}


def test_a1_ratio_is_measured_over_formula(tmp_path):
    # formula = 0.003 + 2 x 0.001 (two storage latencies) = 0.005 s/item.
    # measured = 5.0 s / 1 queued = 5.0 s/item -> ratio 1000x.
    lines = [_snapshot_line(i, [_cand("rpiCpu", 1, 5.0)]) for i in range(40)]
    snaps = _write_snapshots(tmp_path, lines)
    corpus = _write_corpus(tmp_path, {"dnn1": ["rpiCpu"]})
    res = a1.read([snaps], TYPE_DB, [corpus])
    cell = res["cells"][0]
    assert cell["formula_seconds_per_item"] == pytest.approx(0.005)
    assert cell["measured_median_seconds_per_item"] == pytest.approx(5.0)
    assert cell["ratio"] == pytest.approx(1000.0)
    # One qualifying cell is under A1_MIN_CELLS: the verdict must be VOID, not a claim.
    assert res["verdict"] == "VOID"


def test_a1_needs_three_cells_before_it_will_say_clock_defect(tmp_path):
    lines = []
    for i in range(40):
        lines.append(
            _snapshot_line(
                i,
                [
                    _cand("rpiCpu", 1, 5.0, key="n:1"),
                    _cand("xavierCpu", 1, 5.0, key="n:2"),
                    _cand("pynqFpga", 1, 5.0, key="n:3"),
                ],
            )
        )
    snaps = _write_snapshots(tmp_path, lines)
    db = {
        "dnn1": {
            "executionTime": {"rpiCpu": 0.003, "xavierCpu": 0.001, "pynqFpga": 0.0005},
            "stateSize": {"app": {"input": 0, "output": 0}},
        }
    }
    corpus = _write_corpus(tmp_path, {"dnn1": ["rpiCpu", "xavierCpu", "pynqFpga"]})
    res = a1.read([snaps], db, [corpus])
    assert res["n_qualifying_cells"] == 3
    assert res["verdict"] == "CLOCK-DEFECT"
    assert res["median_ratio"] > a1.A1_RATIO_MIN


def test_a1_says_clock_ok_when_the_formula_already_matches(tmp_path):
    db = {
        "dnn1": {
            "executionTime": {"rpiCpu": 1.0, "xavierCpu": 1.0, "pynqFpga": 1.0},
            "stateSize": {"app": {"input": 0, "output": 0}},
        }
    }
    # formula ~1.002 s/item; measured 1.0 s/item -> ratio ~1.0, under the 3x bar.
    lines = [
        _snapshot_line(
            i,
            [
                _cand("rpiCpu", 2, 2.0, key="n:1"),
                _cand("xavierCpu", 2, 2.0, key="n:2"),
                _cand("pynqFpga", 2, 2.0, key="n:3"),
            ],
        )
        for i in range(40)
    ]
    snaps = _write_snapshots(tmp_path, lines)
    corpus = _write_corpus(tmp_path, {"dnn1": ["rpiCpu", "xavierCpu", "pynqFpga"]})
    res = a1.read([snaps], db, [corpus])
    assert res["verdict"] == "CLOCK-OK"


def test_a1_counts_one_platform_once_per_snapshot(tmp_path):
    """A deep platform is a candidate of every task in a batch; counting it once per
    suitor would let one queue dominate the median by multiplicity."""
    line = json.dumps(
        {
            "snapshot_id": 0,
            "tasks": [
                {"task_id": t, "candidates": [_cand("rpiCpu", 1, 5.0, key="n:1")]}
                for t in range(10)
            ],
        }
    )
    snaps = _write_snapshots(tmp_path, [line])
    corpus = _write_corpus(tmp_path, {"dnn1": ["rpiCpu"]})
    res = a1.read([snaps], TYPE_DB, [corpus])
    assert res["cells"][0]["n_live_observations"] == 1


def test_a1_refuses_a_capture_without_platform_type(tmp_path):
    line = json.dumps(
        {
            "snapshot_id": 0,
            "tasks": [{"task_id": 0, "candidates": [{"queue_length": 1, "queue_drain_seconds": 5.0}]}],
        }
    )
    snaps = _write_snapshots(tmp_path, [line])
    corpus = _write_corpus(tmp_path, {"dnn1": ["rpiCpu"]})
    with pytest.raises(a1.ClockReadError, match="platform_type"):
        a1.read([snaps], TYPE_DB, [corpus])


def test_a1_refuses_captures_with_no_busy_queue_at_all(tmp_path):
    lines = [_snapshot_line(i, [_cand("rpiCpu", 0, 0.0)]) for i in range(10)]
    snaps = _write_snapshots(tmp_path, lines)
    corpus = _write_corpus(tmp_path, {"dnn1": ["rpiCpu"]})
    with pytest.raises(a1.ClockReadError, match="cannot measure a drain clock"):
        a1.read([snaps], TYPE_DB, [corpus])


def test_a1_emits_a_drain_table_the_label_can_read(tmp_path):
    from scripts_cosim.drift_label import drain_table_key, load_drain_table

    lines = [_snapshot_line(i, [_cand("rpiCpu", 2, 9.0)]) for i in range(40)]
    snaps = _write_snapshots(tmp_path, lines)
    corpus = _write_corpus(tmp_path, {"dnn1": ["rpiCpu"]})
    res = a1.read([snaps], TYPE_DB, [corpus])
    table_path = tmp_path / "drain.json"
    table_path.write_text(json.dumps({"drain_seconds_per_item": res["drain_seconds_per_item"]}))
    loaded = load_drain_table(table_path)
    assert loaded[drain_table_key("dnn1", "rpiCpu")] == pytest.approx(4.5)


# ==========================================================================
# A2 -- the label reversal, and the inherited rank-stability control
# ==========================================================================


def test_spearman_is_one_for_an_order_preserving_change():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert a2.spearman(a, b) == pytest.approx(1.0)


def test_spearman_is_minus_one_for_a_reversal():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [4.0, 3.0, 2.0, 1.0]
    assert a2.spearman(a, b) == pytest.approx(-1.0)


def test_spearman_is_none_when_a_side_is_constant():
    assert a2.spearman([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]) is None


def test_spearman_handles_ties_with_average_ranks():
    # Two ties on each side, same order -> still perfectly correlated.
    assert a2.spearman([1.0, 1.0, 2.0, 3.0], [5.0, 5.0, 6.0, 7.0]) == pytest.approx(1.0)


def _ds(name, rows, rank_stability=None, has_choice=True):
    """A per-dataset record shaped like read_dataset's output."""
    return {
        "dataset": name,
        "n_rows": 500 if has_choice else 8,
        "has_real_choice": has_choice,
        "per_v": {v: {"optimum": 1.0, "regret_pct": dict(r)} for v, r in rows.items()},
        "rank_stability": rank_stability
        or {"v1_vs_v2": 0.95, "v1_vs_v05": 0.95, "v1_vs_v0": 0.5},
    }


def _uniform(n, sq, gnn):
    rows = {v: {"shortest_queue": sq, "gnn": gnn} for v in ("0", "0.5", "1", "2")}
    return [_ds(f"ds_{i:05d}", rows) for i in range(n)]


def test_a2_fires_when_the_shaped_label_prefers_the_reactive_plan():
    # Under the shaped label the reactive plan is better on every state.
    per_ds = _uniform(50, sq=2.0, gnn=40.0)
    s = a2.summarise(per_ds, "gnn")
    assert s["reversal_rate_pct"] == pytest.approx(100.0)
    assert s["verdict"] == "LABEL-AGREES-WITH-STREAM"
    assert s["rank_stability_verdict"] == "STABLE"


def test_a2_does_not_fire_when_the_shaped_label_still_prefers_the_checkpoint():
    per_ds = _uniform(50, sq=40.0, gnn=2.0)
    s = a2.summarise(per_ds, "gnn")
    assert s["reversal_rate_pct"] == pytest.approx(0.0)
    assert s["verdict"] == "LABEL-DOES-NOT-AGREE"


def test_a2_bar_is_at_sixty_percent_not_a_majority():
    # 55 % reversal: a majority, and still under the registered bar.
    per_ds = _uniform(55, sq=2.0, gnn=40.0) + _uniform(45, sq=40.0, gnn=2.0)
    for i, d in enumerate(per_ds):
        d["dataset"] = f"ds_{i:05d}"
    s = a2.summarise(per_ds, "gnn")
    assert s["reversal_rate_pct"] == pytest.approx(55.0)
    assert s["verdict"] == "LABEL-DOES-NOT-AGREE"


def test_a2_voids_below_the_minimum_dataset_count():
    s = a2.summarise(_uniform(10, sq=2.0, gnn=40.0), "gnn")
    assert s["verdict"] == "VOID"


def test_a2_reads_the_bar_only_on_states_with_real_choice():
    """Degenerate states score every plan the same; D1 measured the reactive source's
    median plan space at 8 rows, so this is the difference between a reading and noise."""
    choice = _uniform(50, sq=2.0, gnn=40.0)
    degenerate = [
        _ds(f"deg_{i}", {v: {"shortest_queue": 0.0, "gnn": 0.0} for v in ("0", "0.5", "1", "2")},
            has_choice=False)
        for i in range(500)
    ]
    s = a2.summarise(choice + degenerate, "gnn")
    assert s["n_datasets"] == 550
    assert s["n_with_real_choice"] == 50
    assert s["reversal_rate_pct"] == pytest.approx(100.0)


def test_a2_rank_control_marks_an_unstable_label_unstable():
    per_ds = _uniform(50, sq=2.0, gnn=40.0)
    for d in per_ds:
        d["rank_stability"] = {"v1_vs_v2": 0.1, "v1_vs_v05": 0.9, "v1_vs_v0": 0.0}
    s = a2.summarise(per_ds, "gnn")
    # The reversal can still fire; the control is separate and must say UNSTABLE, which
    # is what keeps that V out of training.
    assert s["verdict"] == "LABEL-AGREES-WITH-STREAM"
    assert s["rank_stability_verdict"] == "UNSTABLE"


def test_a2_rank_control_needs_both_comparisons():
    per_ds = _uniform(50, sq=2.0, gnn=40.0)
    for d in per_ds:
        d["rank_stability"] = {"v1_vs_v2": 0.95, "v1_vs_v05": 0.5, "v1_vs_v0": 0.0}
    assert a2.summarise(per_ds, "gnn")["rank_stability_verdict"] == "UNSTABLE"


def test_a2_regret_pct_refuses_a_non_positive_optimum():
    with pytest.raises(a2.LabelReversalError):
        a2.regret_pct(1.0, 0.0)


# ==========================================================================
# A3 -- the realized externality
# ==========================================================================


def _rec(task_id, platform, scheduled, arrived, done):
    return {
        "taskId": task_id,
        "executionNode": "n0",
        "executionPlatform": platform,
        "scheduledTime": scheduled,
        "arrivedTime": arrived,
        "doneTime": done,
    }


def test_a3_measures_the_wait_a_batch_inflicts_on_later_tasks():
    # Task 0 is committed at t=0 and occupies the platform over [0, 10].
    # Task 1 is committed at t=1 and waits behind it until 10 -> 9 s inflicted.
    recs = [_rec(0, 1, 0.0, 0.0, 10.0), _rec(1, 1, 1.0, 10.0, 12.0)]
    out = a3.analyse(recs)
    assert out["n_batches"] >= 1
    assert out["realized_median_nonzero"] == pytest.approx(9.0)


def test_a3_charges_nothing_when_the_platform_was_free():
    # Two tasks that never overlap: the first finishes before the second is committed.
    recs = [_rec(0, 1, 0.0, 0.0, 5.0), _rec(1, 1, 6.0, 6.0, 8.0)]
    out = a3.analyse(recs)
    assert out["realized_median"] == pytest.approx(0.0)


def test_a3_predicted_grows_superlinearly_with_the_work_added():
    """The predicted column is the closed form's shape, so a longer batch on the same
    platform must be charged more than proportionally."""
    short = a3.analyse([_rec(0, 1, 0.0, 0.0, 1.0), _rec(1, 1, 0.5, 1.0, 2.0)])
    long = a3.analyse([_rec(0, 1, 0.0, 0.0, 10.0), _rec(1, 1, 0.5, 10.0, 11.0)])
    assert long["predicted_median"] > 10 * short["predicted_median"]


def test_a3_voids_below_the_batch_floor():
    recs = [_rec(0, 1, 0.0, 0.0, 10.0), _rec(1, 1, 1.0, 10.0, 12.0)]
    assert a3.analyse(recs)["verdict"] == "VOID"


def test_a3_fires_when_prediction_and_reality_agree_in_rank():
    """Many platforms, each loaded differently: the deeper the platform the longer the
    follower waits, which is exactly what the closed form orders on."""
    recs = []
    tid = 0
    for p in range(1, a3.A3_MIN_BATCHES // 2 + 2):
        span = 1.0 + (p % 50)
        recs.append(_rec(tid, p, 0.0, 0.0, span))
        tid += 1
        recs.append(_rec(tid, p, 0.1, span, span + 1.0))
        tid += 1
    out = a3.analyse(recs)
    assert out["n_batches"] >= a3.A3_MIN_BATCHES
    assert out["spearman_nonzero"] == pytest.approx(1.0, abs=0.05)
    assert out["verdict"] == "CLOSED-FORM-TRACKS"


def test_a3_refuses_a_result_with_no_task_records(tmp_path):
    p = tmp_path / "raw.json"
    p.write_text(json.dumps({"stats": {"taskResults": []}}))
    with pytest.raises(a3.ExternalityReadError, match="KEEP_RAW"):
        a3.load_task_results(p)


def test_a3_num_treats_a_zero_timestamp_as_a_timestamp():
    """`float(rec.get(k) or 0.0)` is wrong: a batch committed at simulated time 0 is
    legitimate and reads as falsy. The D2 tests caught this once already."""
    assert a3._num({"scheduledTime": 0.0}, "scheduledTime", default=-1.0) == 0.0
    assert a3._num({}, "scheduledTime", default=-1.0) == -1.0


def test_a2_unwraps_the_warm_snapshot_wrapper(tmp_path, monkeypatch):
    """make_warm_corpus writes {"snapshot": {...}, "provenance": {...}}; handing the
    wrapper to shortest_queue_plan produces an empty plan, which is then correctly but
    uselessly reported as "not in the sweep". This is how it surfaced on the first run."""
    captured = {}

    def fake_shortest_queue_plan(snapshot, cosim_id_of):
        captured["keys"] = sorted(snapshot.keys())
        return {0: (1, 10)}

    monkeypatch.setattr(a2, "shortest_queue_plan", fake_shortest_queue_plan)
    monkeypatch.setattr(
        a2, "build_state_context", lambda *a, **k: object()
    )
    monkeypatch.setattr(a2, "externality_seconds", lambda plan, ctx: 0.0)

    ds = tmp_path / "ds_00000"
    ds.mkdir()
    (ds / "warm_snapshot.json").write_text(
        json.dumps({"snapshot": {"tasks": [], "time": 1.0}, "provenance": {"source_tag": "knb"}})
    )
    (ds / "workload.json").write_text(json.dumps({"trace_task_ids": [7]}))

    import scripts_cosim.score_route_b_contention as scorer

    monkeypatch.setattr(scorer, "load_rows", lambda d, o: [({0: (1, 10)}, 5.0)])
    a2.read_dataset(ds, {}, {}, 0.46)
    assert captured["keys"] == ["tasks", "time"], "the wrapper reached shortest_queue_plan"


def test_a1_offset_table_keeps_type_heterogeneity(tmp_path):
    """The emitted table adds the platform's MISSING seconds rather than replacing the
    per-type cost, because the omitted terms (peer transfer, source->platform latency)
    are per task instance, not per type. A flat table would make the label indifferent to
    which type goes where, which the simulator is not."""
    db = {
        "cheap": {"executionTime": {"rpiCpu": 0.0}, "stateSize": {"a": {"input": 0, "output": 0}}},
        "dear": {"executionTime": {"rpiCpu": 3.0}, "stateSize": {"a": {"input": 0, "output": 0}}},
    }
    lines = [_snapshot_line(i, [_cand("rpiCpu", 1, 8.0)]) for i in range(40)]
    snaps = _write_snapshots(tmp_path, lines)
    corpus = _write_corpus(tmp_path, {"cheap": ["rpiCpu"], "dear": ["rpiCpu"]})
    res = a1.read([snaps], db, [corpus])
    t = res["drain_seconds_per_item"]
    cheap, dear = t["cheap|rpiCpu"], t["dear|rpiCpu"]
    # The 3 s execution gap survives.
    assert dear - cheap == pytest.approx(3.0)
    # And the platform's mean still lands on what was measured.
    assert (cheap + dear) / 2 == pytest.approx(8.0, abs=1e-6)
    # The flat table is emitted too, so the choice stays inspectable.
    assert res["drain_seconds_per_item_flat"]["cheap|rpiCpu"] == pytest.approx(8.0)


def test_a1_offset_is_never_negative(tmp_path):
    """A platform whose formula already exceeds the measured drain must not get a negative
    offset -- that would price a backlog below the simulator's own floor."""
    db = {
        "dear": {"executionTime": {"rpiCpu": 10.0}, "stateSize": {"a": {"input": 0, "output": 0}}}
    }
    lines = [_snapshot_line(i, [_cand("rpiCpu", 2, 1.0)]) for i in range(40)]  # 0.5 s/item
    snaps = _write_snapshots(tmp_path, lines)
    corpus = _write_corpus(tmp_path, {"dear": ["rpiCpu"]})
    res = a1.read([snaps], db, [corpus])
    assert res["platform_offset_seconds"]["rpiCpu"] == 0.0
    assert res["drain_seconds_per_item"]["dear|rpiCpu"] == pytest.approx(10.002, abs=1e-3)
