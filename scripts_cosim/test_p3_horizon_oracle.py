"""Guards for the P3 horizon extension of the live-snapshot co-sim oracle.

What can silently corrupt the pilot, in order of blast radius:
  1. the horizon window off by one boundary (a snapshot's own batch re-arriving, or
     the last in-window arrival dropped),
  2. the task-id -> forced-placement mapping drifting (the determined scheduler
     hard-exits on any unforced task, but an id collision would *not* be loud — a
     horizon task would inherit a batch task's forced placement),
  3. h=0 not being bit-identical to the pre-P3 t=0 behaviour.
"""

import math

import pytest

from scripts_cosim.live_snapshot_cosim_oracle import (
    AUTO_RESOLVE,
    HORIZON_ARRIVAL_OFFSET,
    build_workload_from_snapshot,
    horizon_forced_placements,
    slice_horizon_events,
)


def _ev(ts, task_type="dnn1", node="client_node3"):
    return {
        "timestamp": ts,
        "application": {"name": f"nofs-{task_type}", "dag": {task_type: []}},
        "qos": {"name": "medium", "maxDurationDeviation": 15},
        "node_name": node,
    }


TRACE = [_ev(t) for t in (9.0, 10.0, 10.5, 12.0, 15.0, 15.0001, 20.0)]


class TestSliceHorizonEvents:
    def test_window_is_left_exclusive_right_inclusive(self):
        # snapshot at t=10.0, h=5: the t=10.0 event is the batch itself (exclude),
        # t=15.0 is the boundary (include), t=15.0001 is past it (exclude).
        out = slice_horizon_events(TRACE, 10.0, 5.0)
        original = [10.5, 12.0, 15.0]
        assert [round(e["timestamp"] - HORIZON_ARRIVAL_OFFSET + 10.0, 6) for e in out] == original

    def test_timestamps_shifted_to_offset(self):
        out = slice_horizon_events(TRACE, 10.0, 5.0)
        assert out[0]["timestamp"] == pytest.approx(HORIZON_ARRIVAL_OFFSET + 0.5)
        assert all(e["timestamp"] >= HORIZON_ARRIVAL_OFFSET for e in out)
        assert out == sorted(out, key=lambda e: e["timestamp"])

    def test_source_events_not_mutated(self):
        before = [e["timestamp"] for e in TRACE]
        slice_horizon_events(TRACE, 10.0, 5.0)
        assert [e["timestamp"] for e in TRACE] == before

    def test_zero_horizon_fails_loud(self):
        with pytest.raises(ValueError, match="horizon_seconds"):
            slice_horizon_events(TRACE, 10.0, 0.0)


class TestHorizonForcedPlacements:
    def test_ids_start_after_batch_and_are_auto_resolve(self):
        events = slice_horizon_events(TRACE, 10.0, 5.0)
        forced = horizon_forced_placements(4, events)
        assert forced == {4: AUTO_RESOLVE, 5: AUTO_RESOLVE, 6: AUTO_RESOLVE}

    def test_no_overlap_with_batch_ids(self):
        events = slice_horizon_events(TRACE, 10.0, 5.0)
        forced = horizon_forced_placements(4, events)
        assert all(task_id >= 4 for task_id in forced)

    def test_multi_task_dag_consumes_multiple_ids(self):
        ev = _ev(11.0)
        ev["application"]["dag"] = {"dnn1": [], "dnn2": []}
        forced = horizon_forced_placements(2, [ev, _ev(12.0)])
        assert forced == {2: AUTO_RESOLVE, 3: AUTO_RESOLVE, 4: AUTO_RESOLVE}


class TestBuildWorkload:
    TASKS = [
        {"task_type": "dnn1", "source_node": "client_node0"},
        {"task_type": "dnn2", "source_node": "client_node1"},
    ]

    def test_no_horizon_is_pre_p3_behaviour(self):
        wl = build_workload_from_snapshot(self.TASKS)
        assert wl["duration"] == 1
        assert len(wl["events"]) == 2
        assert [e["timestamp"] for e in wl["events"]] == [0.0, 0.0001]

    def test_horizon_events_appended_after_batch(self):
        horizon = slice_horizon_events(TRACE, 10.0, 5.0)
        wl = build_workload_from_snapshot(self.TASKS, horizon_events=horizon)
        assert len(wl["events"]) == 2 + 3
        ts = [e["timestamp"] for e in wl["events"]]
        assert ts == sorted(ts)
        assert wl["duration"] >= math.ceil(ts[-1]) + 1

    def test_unshifted_horizon_events_fail_loud(self):
        with pytest.raises(ValueError, match="pre-shifted"):
            build_workload_from_snapshot(self.TASKS, horizon_events=[_ev(0.001)])

    def test_batch_reaching_offset_fails_loud(self):
        huge_batch = [{"task_type": "dnn1", "source_node": "c"} for _ in range(101)]
        horizon = slice_horizon_events(TRACE, 10.0, 5.0)
        with pytest.raises(ValueError, match="interleave"):
            build_workload_from_snapshot(huge_batch, horizon_events=horizon)


class TestPreflightReachability:
    INFRA = {
        "nodes": [
            {"node_name": "client_node3", "network_map": {"node0": 0.01}},
            {"node_name": "client_node7", "network_map": {}},
            {"node_name": "node0", "network_map": {"client_node3": 0.01}},
            {"node_name": "node1", "network_map": {}},
        ]
    }
    SNAP = {
        "replicas_by_type": {
            "dnn1": [{"node_name": "node0", "platform_id": 5}],
            "dnn2": [{"node_name": "client_node3", "platform_id": 9}],
        }
    }

    def test_reachable_via_server_map(self):
        from scripts_cosim.live_snapshot_cosim_oracle import preflight_horizon_reachability
        evs = [_ev(1.0, "dnn1", "client_node3")]
        evs[0]["timestamp"] = HORIZON_ARRIVAL_OFFSET + 1.0
        assert preflight_horizon_reachability(self.INFRA, self.SNAP, evs) == []

    def test_local_replica_counts(self):
        from scripts_cosim.live_snapshot_cosim_oracle import preflight_horizon_reachability
        evs = [_ev(1.0, "dnn2", "client_node3")]
        assert preflight_horizon_reachability(self.INFRA, self.SNAP, evs) == []

    def test_unreachable_source_reported(self):
        from scripts_cosim.live_snapshot_cosim_oracle import preflight_horizon_reachability
        # client_node7: no local replica, and node0's map does not contain it.
        evs = [_ev(1.0, "dnn1", "client_node7"), _ev(2.0, "dnn2", "client_node7")]
        out = preflight_horizon_reachability(self.INFRA, self.SNAP, evs)
        assert out == [("dnn1", "client_node7"), ("dnn2", "client_node7")]

    def test_pre_p3_snapshot_fails_loud(self):
        from scripts_cosim.live_snapshot_cosim_oracle import preflight_horizon_reachability
        with pytest.raises(ValueError, match="replicas_by_type"):
            preflight_horizon_reachability(self.INFRA, {"tasks": []}, [_ev(1.0)])
