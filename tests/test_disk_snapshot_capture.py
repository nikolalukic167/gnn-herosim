"""Tests for node disk snapshot capture (B1 feature plumbing)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.policy.state_capture import StateCaptureHelper


class _FakeStorage:
    def __init__(self, functions_cache: set):
        self.type = {"remote": False}
        self.functions_cache = functions_cache

    def has_function(self, platform: str, task_type) -> bool:
        name = task_type["name"] if isinstance(task_type, dict) else task_type.type["name"]
        return (platform, name) in self.functions_cache


class _FakePlatform:
    def __init__(self, plat_id: int, short_name: str):
        self.id = plat_id
        self.type = {"shortName": short_name}
        self.current_task = None
        self.initialized = SimpleNamespace(triggered=True)

    def queue_length(self) -> int:
        return 0


class _FakeNode:
    def __init__(self, name: str, plat_id: int = 0, short_name: str = "rpiCpu"):
        self.node_name = name
        self.platforms = SimpleNamespace(items=[_FakePlatform(plat_id, short_name)])
        self.storage = SimpleNamespace(
            items=[_FakeStorage({("rpiCpu", "dnn1")})]
        )


class TestDiskSnapshotCapture(unittest.TestCase):
    def test_capture_disk_snapshot_by_task_type(self):
        node = _FakeNode("server_node0")
        nodes = SimpleNamespace(items=[node])
        helper = StateCaptureHelper(env=MagicMock(), nodes=nodes)

        snapshot = helper.capture_disk_snapshot()

        self.assertIn("dnn1", snapshot)
        self.assertIn("dnn2", snapshot)
        key = "server_node0:0"
        self.assertEqual(snapshot["dnn1"][key], 1.0)
        self.assertEqual(snapshot["dnn2"][key], 0.0)

    def test_get_captured_state_includes_disk_snapshot(self):
        node = _FakeNode("server_node0")
        nodes = SimpleNamespace(items=[node])
        helper = StateCaptureHelper(env=MagicMock(now=0.0), nodes=nodes)
        system_state = SimpleNamespace(replicas={})

        captured = helper.get_captured_state(system_state, total_rtt=0.0)

        self.assertIn("disk_snapshot_by_task_type", captured)
        self.assertEqual(
            captured["disk_snapshot_by_task_type"]["dnn1"]["server_node0:0"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
