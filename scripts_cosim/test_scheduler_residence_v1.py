#!/usr/bin/env python3
"""scheduler_residence_v1 -- the instrument and the bars.

The decomposition's whole claim is that its three parts add back up to `averageWaitTime`.
These tests pin that, the bars as signed, and the two ways the experiment could silently
become a different one: a task scheduled without passing the collector, and counters the
orchestrator's scalar whitelist drops on the way out.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.scheduler_residence_v1_read import (  # noqa: E402
    CELLS,
    R0_HOL_MIN_S,
    R0_MIN_CELLS,
    R0_MIN_SEEDS,
    R0_RECONSTRUCTION_TOL,
    R1_GOOD_CELL,
    R1_MIN_SEPARATING,
    ResidenceReadError,
    cell_structure,
    read_r0,
    read_r1,
    residence_summary,
)


def test_bars_are_the_registered_constants():
    assert (R0_RECONSTRUCTION_TOL, R0_HOL_MIN_S, R0_MIN_CELLS, R0_MIN_SEEDS) == (
        0.01, 2.0, 2, 12)
    assert R1_MIN_SEPARATING == 1
    assert R1_GOOD_CELL == "cell_s9001_f4000_pg16"


# ------------------------------------------------------------------ the summary arithmetic
def _tasks(n, hol, coll, place, t0=0.0, step=1.0):
    return [[t0 + i * step, hol, coll, place] for i in range(n)]


def test_parts_reconstruct_the_average_wait_time():
    tasks = _tasks(100, 4.0, 2.0, 0.5)
    got = residence_summary(tasks, [[0.0, 1.0, 5.0]], [], 6.5, 0)
    assert got["sum_of_parts_s"] == pytest.approx(6.5)
    assert got["reconstruction_error"] == pytest.approx(0.0)


def test_reconstruction_error_is_reported_not_corrected():
    """If the parts disagree with the total, the read must SEE the disagreement."""
    got = residence_summary(_tasks(10, 4.0, 2.0, 0.5), [], [], 10.0, 0)
    assert got["sum_of_parts_s"] == pytest.approx(6.5)
    assert got["reconstruction_error"] == pytest.approx(0.35)
    assert got["average_wait_time_s"] == 10.0     # left as measured


def test_bucketing_uses_the_task_decile_edges():
    tasks = [[1.0, 1, 1, 1], [9.9, 1, 1, 1], [10.0, 1, 1, 1], [99.0, 1, 1, 1]]
    got = residence_summary(tasks, [], [10.0, 20.0], 3.0, 0)
    assert [r["n"] for r in got["deciles"]] == [2, 1, 1]


def test_empty_decile_invents_no_share():
    got = residence_summary([[1.0, 1, 1, 1]], [], [10.0, 20.0], 3.0, 0)
    assert got["deciles"][1]["n"] == 0
    assert "head_of_line_s" not in got["deciles"][1]


def test_no_rows_fails_loud():
    with pytest.raises(ResidenceReadError):
        residence_summary([], [], [10.0], 3.0, 0)


def test_mean_batch_size_comes_from_the_batch_rows():
    got = residence_summary(_tasks(4, 1, 1, 1), [[0, 1, 4.0], [1, 2, 6.0]], [], 3.0, 0)
    assert got["mean_batch_size"] == pytest.approx(5.0)


# --------------------------------------------------------------------------------- R0
def _arm(cell, seed, hol, coll, place, aw=None, unstamped=0):
    aw = (hol + coll + place) if aw is None else aw
    return {
        "arm": f"{cell}__residence_gnn_s{seed}",
        "residence_summary": {
            "head_of_line_s": hol, "collection_s": coll, "placement_s": place,
            "sum_of_parts_s": hol + coll + place, "average_wait_time_s": aw,
            "reconstruction_error": abs(hol + coll + place - aw) / aw if aw else None,
            "unstamped": unstamped, "n_tasks": 1000, "n_batches": 100, "deciles": [],
        },
    }


def _corpus(per_cell, unstamped=0):
    arms = {}
    for cell, (hol, coll, place) in per_cell.items():
        for seed in range(1, 17):
            if seed == 3 or (cell == CELLS[0] and seed in (8, 14)):
                continue
            d = _arm(cell, seed, hol, coll, place, unstamped=unstamped)
            arms[d["arm"]] = d
    return arms


def test_r0_fires_when_head_of_line_dominates_on_two_cells():
    got = read_r0(_corpus({CELLS[0]: (5.0, 1.5, 0.4),
                           CELLS[1]: (5.0, 1.5, 0.4),
                           CELLS[2]: (0.5, 6.0, 0.4)}))
    assert got["verdict"] == "HEAD-OF-LINE-DOMINATES"
    assert got["cells_firing"] == 2


def test_r0_does_not_fire_on_one_cell():
    got = read_r0(_corpus({CELLS[0]: (5.0, 1.5, 0.4),
                           CELLS[1]: (0.5, 6.0, 0.4),
                           CELLS[2]: (0.5, 6.0, 0.4)}))
    assert got["verdict"] == "COLLECTION-DOMINATES"


def test_r0_needs_head_of_line_to_EXCEED_collection_not_just_clear_the_floor():
    """3 s of head-of-line behind 9 s of collection is a collection problem."""
    got = read_r0(_corpus({c: (3.0, 9.0, 0.4) for c in CELLS}))
    assert got["verdict"] == "COLLECTION-DOMINATES"
    assert got["cells_firing"] == 0


def test_r0_voids_a_cell_whose_parts_do_not_reconstruct():
    arms = _corpus({c: (5.0, 1.5, 0.4) for c in CELLS})
    for name in [k for k in arms if k.startswith(CELLS[1])]:
        arms[name]["residence_summary"]["average_wait_time_s"] = 20.0
        arms[name]["residence_summary"]["reconstruction_error"] = 0.6
    got = read_r0(arms)
    assert got["per_cell"][CELLS[1]]["verdict"] == "VOID-DOES-NOT-RECONSTRUCT"
    assert got["verdict"] == "HEAD-OF-LINE-DOMINATES"      # the other two still fire


def test_r0_voids_a_cell_with_unstamped_tasks():
    got = read_r0(_corpus({c: (5.0, 1.5, 0.4) for c in CELLS}, unstamped=1))
    for cell in CELLS:
        assert got["per_cell"][cell]["verdict"] == "VOID-DOES-NOT-RECONSTRUCT"


def test_r0_excludes_the_burned_and_unservable_seeds():
    got = read_r0(_corpus({c: (5.0, 1.5, 0.4) for c in CELLS}))
    assert got["per_cell"][CELLS[0]]["n"] == 13      # 16 - seed 3 - seeds 8 and 14
    assert got["per_cell"][CELLS[1]]["n"] == 15


def test_r0_rejects_an_arm_without_the_instrument():
    from scripts_cosim.scheduler_residence_v1_read import _res
    with pytest.raises(ResidenceReadError):
        _res({"arm": "x"})


# --------------------------------------------------------------------------------- R1
def _cfg(servers, clients, reach):
    nodes = [{"node_name": f"s{i}", "platforms": [{}, {}]} for i in range(servers)]
    for i in range(clients):
        nodes.append({"node_name": f"c{i}", "is_client": True,
                      "network_map": {f"s{j}": 1 for j in range(reach)}})
    return {"infrastructure": {"nodes": nodes}}


def test_r1_separates_on_a_statistic_outside_the_others_range():
    structures = {
        CELLS[0]: cell_structure(_cfg(6, 20, 6)),
        R1_GOOD_CELL: cell_structure(_cfg(12, 20, 12)),     # strictly outside
        CELLS[2]: cell_structure(_cfg(6, 20, 6)),
    }
    got = read_r1(structures)
    assert got["verdict"] == "STRUCTURE-SEPARATES"
    assert "server_nodes" in got["separating"]


def test_r1_does_not_separate_when_the_good_cell_is_between_the_others():
    structures = {
        CELLS[0]: cell_structure(_cfg(4, 20, 4)),
        R1_GOOD_CELL: cell_structure(_cfg(6, 20, 6)),
        CELLS[2]: cell_structure(_cfg(8, 20, 8)),
    }
    got = read_r1(structures)
    assert got["verdict"] == "STRUCTURE-DOES-NOT-SEPARATE"


def test_r1_does_not_separate_when_all_three_are_identical():
    structures = {c: cell_structure(_cfg(6, 20, 6)) for c in CELLS}
    got = read_r1(structures)
    assert got["separating"] == []


def test_r1_skips_a_statistic_it_cannot_measure_rather_than_imputing_it():
    structures = {c: {k: None for k in ("server_nodes",)} for c in CELLS}
    got = read_r1(structures)
    assert got["per_stat"]["server_nodes"]["verdict"] == "NOT-MEASURABLE"
    assert got["verdict"] == "STRUCTURE-DOES-NOT-SEPARATE"


def test_cell_structure_reads_an_empty_config_as_unknown_not_zero():
    got = cell_structure({"infrastructure": {"nodes": []}})
    assert all(v is None for v in got.values())


# ----------------------------------------------------------------------- the instrument
def test_counters_survive_the_orchestrator_whitelist():
    src = (REPO_ROOT / "src/placement/orchestrator.py").read_text()
    assert '"residence_unstamped"' in src
    assert '"residence_tasks"' in src and '"residence_batches"' in src
    assert "isinstance(value, list)" in src


def test_scheduler_exposes_the_instrument():
    from src.policy.gnn.scheduler import GNNScheduler
    for name in ("residence_tasks", "residence_batches", "residence_unstamped"):
        assert isinstance(getattr(GNNScheduler, name), property), name
    src = (REPO_ROOT / "src/policy/gnn/scheduler.py").read_text()
    # Both scheduling points must close the decomposition, or a whole decode mode is missing
    # from it while still contributing to averageWaitTime.
    assert src.count("self._record_residence_placed(task)") == 2
    # The collector is wrapped, not edited at its two return points.
    assert "yield from self._collect_task_batch_inner()" in src


def test_recorder_arithmetic():
    """Head-of-line + collection + placement == waitTime, on the real method."""
    from src.policy.gnn.scheduler import GNNScheduler

    class S:
        pass

    s = S()
    s._residence_tasks, s._residence_unstamped = [], 0
    s.env = type("E", (), {"now": 30.0})()
    rec = GNNScheduler._record_residence_placed.__get__(s)

    task = type("T", (), {})()
    task.dispatched_time = 10.0          # entered the store at 10
    task._res_loop_start = 18.0          # the batch that placed it began at 18  -> HOL 8
    task._res_batch_return = 26.0        # that batch returned at 26             -> COLL 8
    rec(task)                            # scheduled now, at 30                  -> PLACE 4
    row = s._residence_tasks[0]
    assert row == [10.0, 8.0, 8.0, 4.0]
    assert sum(row[1:]) == 30.0 - 10.0   # == waitTime


def test_recorder_handles_a_task_that_arrived_after_the_loop_started():
    from src.policy.gnn.scheduler import GNNScheduler

    class S:
        pass

    s = S()
    s._residence_tasks, s._residence_unstamped = [], 0
    s.env = type("E", (), {"now": 30.0})()
    task = type("T", (), {})()
    task.dispatched_time = 22.0          # arrived mid-collection: no head-of-line time
    task._res_loop_start = 18.0
    task._res_batch_return = 26.0
    GNNScheduler._record_residence_placed.__get__(s)(task)
    row = s._residence_tasks[0]
    assert row == [22.0, 0.0, 4.0, 4.0]
    assert sum(row[1:]) == 30.0 - 22.0   # still reconstructs waitTime exactly


def test_recorder_counts_an_unstamped_task_instead_of_dropping_it():
    from src.policy.gnn.scheduler import GNNScheduler

    class S:
        pass

    s = S()
    s._residence_tasks, s._residence_unstamped = [], 0
    s.env = type("E", (), {"now": 30.0})()
    task = type("T", (), {})()
    task.dispatched_time = 10.0          # never went through the collector
    GNNScheduler._record_residence_placed.__get__(s)(task)
    assert s._residence_tasks == [] and s._residence_unstamped == 1
