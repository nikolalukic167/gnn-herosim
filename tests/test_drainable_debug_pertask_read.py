"""Tests for the drainable_debug_v1 D2 per-task read, written before any arm was re-run.

The read has to be trustworthy in one specific way: it must not quietly report zeros when the
records it needs are absent. Every gate in this program before 2026-09-14 wrote
`taskResults: []` (the streaming stats path above 10,000 events), so a read that treated an
empty list as "nothing to see" would have produced a clean, entirely fictional table.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts_cosim.drainable_debug_pertask_read import (  # noqa: E402
    D2_MIN_EXCESS_PCT_OF_REF,
    D2_MIN_EXCESS_S,
    D2_SERIALIZATION_MIN_PCT,
    batches,
    chosen_queue_vs_min,
    concentration,
    d2_verdict,
    decomposition,
    load_task_results,
    peer_colocation,
    peer_group_spread,
    serialization,
)
from src.placement.orchestrator import build_peer_exchange_table  # noqa: E402


def _rec(tid, node, plat, scheduled, arrived, done, snap=None, **kw):
    """One task record in the live schema (`Task.result()`)."""
    out = {
        "taskId": tid,
        "taskType": {"name": "dnn1"},
        "executionNode": node,
        "executionPlatform": str(plat),
        "scheduledTime": scheduled,
        "arrivedTime": arrived,
        "doneTime": done,
        "queueTime": arrived - scheduled,
        "waitTime": 0.0,
        "initializationTime": 0.0,
        "computeTime": done - arrived,
        "elapsedTime": done - scheduled,
        "peerExchangeTime": 0.0,
        "peerRendezvousWait": 0.0,
        "pullTime": 0.0,
        "coldStartTime": 0.0,
        "queueSnapshotAtScheduling": snap,
    }
    out.update(kw)
    return out


# Two tasks decoded at t=0 onto the SAME platform: the second waits out the first.
# One task decoded at t=0 onto its own platform: no wait.
SAME_PLATFORM = [
    _rec(0, "node0", 10, scheduled=0.0, arrived=0.0, done=10.0),
    _rec(1, "node0", 10, scheduled=0.0, arrived=10.0, done=20.0),
    _rec(2, "node1", 11, scheduled=0.0, arrived=0.0, done=10.0),
]


def test_empty_task_results_fails_loud_rather_than_reporting_zeros(tmp_path):
    """The streaming stats path's exact output. A silent zero here would have been the whole
    read: every gate before the KEEP_RAW hook wrote this."""
    p = tmp_path / "streaming.json"
    p.write_text(json.dumps({"stats": {"statsSchemaVersion": "v2_streaming", "taskResults": []}}))
    with pytest.raises(SystemExit) as exc:
        load_task_results(p)
    assert "KEEP_RAW" in str(exc.value)


def test_serialization_attributes_the_second_task_wait_to_the_first():
    ser = serialization(SAME_PLATFORM)
    # Task 1 waited 10 s, and every second of it was behind task 0 on the same platform,
    # placed by the same decode (identical scheduledTime).
    assert ser["total_queue_time"] == pytest.approx(10.0)
    assert ser["behind_any_predecessor"] == pytest.approx(10.0)
    assert ser["behind_same_batch"] == pytest.approx(10.0)
    assert ser["share_same_batch_pct"] == pytest.approx(100.0)


def test_serialization_does_not_credit_a_predecessor_from_another_batch():
    """A task already running when this decode happened is load, not self-inflicted."""
    recs = [
        _rec(0, "node0", 10, scheduled=-5.0, arrived=-5.0, done=10.0),
        _rec(1, "node0", 10, scheduled=0.0, arrived=10.0, done=20.0),
    ]
    ser = serialization(recs)
    assert ser["behind_any_predecessor"] == pytest.approx(10.0)
    assert ser["behind_same_batch"] == pytest.approx(0.0)


def test_serialization_ignores_a_predecessor_that_finished_before_the_wait_began():
    recs = [
        _rec(0, "node0", 10, scheduled=0.0, arrived=0.0, done=1.0),
        _rec(1, "node0", 10, scheduled=5.0, arrived=6.0, done=7.0),
    ]
    assert serialization(recs)["behind_any_predecessor"] == pytest.approx(0.0)


def test_chosen_queue_vs_min_excludes_single_candidate_tasks():
    """A task with one legal replica had no choice; averaging a forced 0 in would make any
    scheduler look perfectly load-aware on a rung where placement does not exist."""
    recs = [
        _rec(0, "node0", 10, 0, 0, 1, snap={"node0:10": 7, "node1:11": 2}),
        _rec(1, "node1", 11, 0, 0, 1, snap={"node1:11": 3}),
    ]
    out = chosen_queue_vs_min(recs)
    assert out["n_with_choice"] == 1
    assert out["mean"] == pytest.approx(5.0)
    assert out["no_choice_pct"] == pytest.approx(50.0)


def test_chosen_queue_vs_min_fails_loud_when_the_chosen_replica_is_not_a_candidate():
    recs = [_rec(0, "node0", 10, 0, 0, 1, snap={"node1:11": 2, "node2:12": 3})]
    with pytest.raises(SystemExit) as exc:
        chosen_queue_vs_min(recs)
    assert "absent from" in str(exc.value)


def test_chosen_queue_vs_min_reports_when_nothing_had_a_choice():
    recs = [_rec(0, "node0", 10, 0, 0, 1, snap={"node0:10": 3})]
    out = chosen_queue_vs_min(recs)
    assert out["n_with_choice"] == 0
    assert "note" in out and "cannot differ by placement" in out["note"]


def test_a_missing_snapshot_says_so_instead_of_claiming_no_choice():
    """Until 2026-09-14 neither the masked_topo prefix path nor the Knative batch subclass
    wrote this snapshot, so the arms that needed the statistic most reported nothing. The two
    cases -- "every task had one candidate" and "no task recorded its candidates" -- must not
    render the same way."""
    recs = [_rec(0, "node0", 10, 0, 0, 1, snap=None)]
    out = chosen_queue_vs_min(recs)
    assert out["missing_snapshot_pct"] == pytest.approx(100.0)
    assert "KEEP_RAW_QSNAP" in out["note"]


def test_concentration_counts_effective_platforms():
    con = concentration(SAME_PLATFORM)
    assert con["platforms_used"] == 2
    # 2/3 and 1/3 shares -> HHI 5/9 -> 1.8 effective platforms
    assert con["effective_platforms"] == pytest.approx(1.8)
    assert con["busiest_platform_share_pct"] == pytest.approx(200.0 / 3)


def test_batches_group_by_the_instant_the_decode_committed():
    out = batches(SAME_PLATFORM)
    assert out["n_batches"] == 1
    assert out["mean_batch_size"] == pytest.approx(3.0)
    assert out["mean_distinct_platforms_per_batch"] == pytest.approx(2.0)


def test_peer_colocation_uses_the_simulators_own_pairing_and_scores_bytes():
    table = build_peer_exchange_table([[0, 1, 100.0], [0, 2, 300.0]])
    out = peer_colocation(SAME_PLATFORM, table)
    assert out["pairs_scored"] == 2
    # (0,1) share node0:10; (0,2) are on different nodes.
    assert out["same_platform_pct"] == pytest.approx(50.0)
    assert out["remote_pct"] == pytest.approx(50.0)
    assert out["bytes_free_pct"] == pytest.approx(25.0)


def test_peer_group_spread_reports_platforms_per_group():
    out = peer_group_spread(SAME_PLATFORM, {0: 0, 1: 0, 2: 0})
    assert out["groups_scored"] == 1
    assert out["mean_platforms_per_group"] == pytest.approx(2.0)
    assert out["mean_nodes_per_group"] == pytest.approx(2.0)
    assert out["groups_on_one_node_pct"] == pytest.approx(0.0)


def test_decomposition_means_are_per_task():
    dec = decomposition(SAME_PLATFORM)
    assert dec["n_tasks"] == 3
    assert dec["queue"] == pytest.approx(10.0 / 3)


def _arm(queue_per_task, same_batch_overlap, n=2000):
    return {
        "decomposition": {"n_tasks": n, "queue": queue_per_task},
        "serialization": {"behind_same_batch": same_batch_overlap * n},
    }


def test_d2_fires_when_the_excess_queue_is_the_arms_own_colocation():
    arms = {
        "knative_network": _arm(10.0, 1.0),
        "gnn": _arm(30.0, 17.0),  # +20 s excess, +16 s of it behind its own batch = 80 %
    }
    out = d2_verdict(arms, "knative_network")
    assert out["arms"]["gnn"]["share_pct"] == pytest.approx(80.0)
    assert out["arms"]["gnn"]["verdict"] == "SELF-INFLICTED-COLOCATION"


def test_d2_does_not_fire_when_the_excess_is_standing_load():
    arms = {"knative_network": _arm(10.0, 1.0), "gnn": _arm(30.0, 3.0)}
    out = d2_verdict(arms, "knative_network")
    assert out["arms"]["gnn"]["share_pct"] == pytest.approx(10.0)
    assert out["arms"]["gnn"]["verdict"] == "NOT-SELF-INFLICTED"


def test_d2_is_void_for_an_arm_with_too_few_records():
    arms = {"knative_network": _arm(10.0, 1.0), "gnn": dict(_arm(30.0, 17.0), void="only 5 records")}
    assert d2_verdict(arms, "knative_network")["arms"]["gnn"]["verdict"] == "VOID"


def test_a_share_of_a_negligible_excess_is_not_a_finding():
    """Dividing by an excess of a few milliseconds produces a confident-looking percentage
    from noise; the smoke run that first exercised this read printed 167.6 %."""
    arms = {"knative_network": _arm(10.0, 1.0), "gnn": _arm(10.2, 9.0)}
    out = d2_verdict(arms, "knative_network")
    assert out["arms"]["gnn"]["verdict"] == "NO-EXCESS"


def test_the_excess_floor_is_relative_not_only_absolute():
    """On the landed x200 gate the queue is ~1,334 s per task, so a 2.5 s difference is 0.19 %
    and decomposing it printed 167.6 % on the first smoke. Seconds alone cannot gate this."""
    arms = {"knative_network": _arm(1334.0, 1.0), "gnn": _arm(1336.5, 5.0)}
    out = d2_verdict(arms, "knative_network")
    assert out["arms"]["gnn"]["excess_queue_s_per_task"] == pytest.approx(2.5)
    assert out["arms"]["gnn"]["verdict"] == "NO-EXCESS"


def test_the_drainable_rung_shape_is_still_decomposed():
    """The case the bar exists for: 39.6 s against a 17.8 s reference."""
    arms = {"knative_network": _arm(17.8, 1.0), "gnn": _arm(39.6, 18.0)}
    out = d2_verdict(arms, "knative_network")
    assert out["arms"]["gnn"]["verdict"] == "SELF-INFLICTED-COLOCATION"


def test_the_bars_are_the_registered_ones():
    """If these constants move, the node's registration moved with them."""
    assert D2_SERIALIZATION_MIN_PCT == 50.0
    assert D2_MIN_EXCESS_S == 1.0
    assert D2_MIN_EXCESS_PCT_OF_REF == 5.0


def test_both_new_capture_sites_are_guarded_by_the_env_flag():
    """The prefix decode path and the Knative batch subclass gained a per-task snapshot capture
    on 2026-09-14. Both must stay behind GNN_CAPTURE_DATASET_STATE, or every historical gate
    result becomes incomparable with every future one."""
    root = Path(__file__).resolve().parents[1]
    for rel, marker in (
        ("src/policy/gnn/scheduler.py", "valid_now = self._get_valid_replicas(task_replicas, task)"),
        (
            "src/policy/knative_network_batch/scheduler.py",
            "task.queue_snapshot_at_scheduling = self._capture_queue_snapshot_for_replicas(valid_replicas)",
        ),
    ):
        src = (root / rel).read_text()
        assert marker in src, f"{rel}: the D2 capture is gone"
        before = src[: src.index(marker)]
        guard = before.rindex('os.environ.get("GNN_CAPTURE_DATASET_STATE", "0") == "1"')
        assert before[guard:].count("\n") <= 3, (
            f"{rel}: the capture is no longer directly under its GNN_CAPTURE_DATASET_STATE guard"
        )
