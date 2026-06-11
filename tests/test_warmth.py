"""Unit tests for src.placement.warmth predicates."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock

from src.placement.warmth import (
    NODE_DISK_V2,
    PLATFORM_REUSE_V1,
    InvalidWarmthPhysicsError,
    image_pull_disk_hit,
    needs_image_pull,
    node_has_cached_image,
    resolve_warmth_physics,
    sandbox_is_warm,
)


def _task_type(name: str = "dnn1") -> Dict[str, Any]:
    return {"name": name, "imageSize": {"rpiCpu": 1.0}}


def _platform(previous_type: str | None = None, short_name: str = "rpiCpu") -> SimpleNamespace:
    if previous_type is None:
        previous_task = None
    else:
        previous_task = SimpleNamespace(type={"name": previous_type})
    return SimpleNamespace(
        previous_task=previous_task,
        type={"shortName": short_name},
    )


def _storage(*, remote: bool = False, cached: bool = False, platform: str = "rpiCpu", task_type: Dict[str, Any] | None = None):
    tt = task_type or _task_type()
    storage = MagicMock()
    storage.type = {"remote": remote}
    storage.has_function = MagicMock(return_value=cached)
    storage.functions_cache: List = [(platform, tt)] if cached else []
    storage.cache_eviction = MagicMock(side_effect=lambda: storage.functions_cache.pop(0) if storage.functions_cache else (_ for _ in ()).throw(IndexError()))
    return storage


def _node(storages: List[Any]) -> SimpleNamespace:
    return SimpleNamespace(storage=SimpleNamespace(items=storages))


class TestResolveWarmthPhysics(unittest.TestCase):
    def test_default_v1(self):
        self.assertEqual(resolve_warmth_physics(), PLATFORM_REUSE_V1)

    def test_infra_value(self):
        self.assertEqual(resolve_warmth_physics(infra_value=NODE_DISK_V2), NODE_DISK_V2)

    def test_invalid_raises(self):
        with self.assertRaises(InvalidWarmthPhysicsError):
            resolve_warmth_physics(infra_value="bogus")


class TestSandboxIsWarm(unittest.TestCase):
    def test_cold_when_no_previous(self):
        self.assertFalse(sandbox_is_warm(_platform(), _task_type()))

    def test_warm_on_match(self):
        self.assertTrue(sandbox_is_warm(_platform("dnn1"), _task_type("dnn1")))

    def test_cold_on_mismatch(self):
        self.assertFalse(sandbox_is_warm(_platform("dnn2"), _task_type("dnn1")))


class TestNeedsImagePullV1(unittest.TestCase):
    def test_v1_disk_hit_does_not_skip_pull(self):
        node = _node([_storage(cached=True)])
        plat = _platform(previous_type=None)
        self.assertTrue(needs_image_pull(PLATFORM_REUSE_V1, plat, node, _task_type()))

    def test_v1_previous_task_skips_pull(self):
        node = _node([_storage(cached=False)])
        plat = _platform(previous_type="dnn1")
        self.assertFalse(needs_image_pull(PLATFORM_REUSE_V1, plat, node, _task_type()))


class TestNeedsImagePullV2(unittest.TestCase):
    def test_v2_disk_hit_skips_pull(self):
        node = _node([_storage(cached=True)])
        plat = _platform(previous_type=None)
        self.assertFalse(needs_image_pull(NODE_DISK_V2, plat, node, _task_type()))

    def test_v2_previous_task_with_disk_miss_still_pulls(self):
        node = _node([_storage(cached=False)])
        plat = _platform(previous_type="dnn1")
        self.assertTrue(needs_image_pull(NODE_DISK_V2, plat, node, _task_type()))

    def test_eviction_forces_repull_despite_previous_task(self):
        tt = _task_type()
        storage = _storage(cached=True, task_type=tt)
        node = _node([storage])
        plat = _platform(previous_type="dnn1")
        self.assertFalse(needs_image_pull(NODE_DISK_V2, plat, node, tt))
        storage.functions_cache.clear()
        storage.has_function.return_value = False
        self.assertTrue(needs_image_pull(NODE_DISK_V2, plat, node, tt))
        self.assertTrue(sandbox_is_warm(plat, tt))


class TestPingPongABA(unittest.TestCase):
    def test_v2_third_invocation_pull_skipped_sandbox_cold(self):
        tt_a = _task_type("dnn1")
        tt_b = _task_type("dnn2")
        node = _node([_storage(cached=True, task_type=tt_a)])
        plat = _platform(previous_type="dnn2")
        self.assertFalse(needs_image_pull(NODE_DISK_V2, plat, node, tt_a))
        self.assertFalse(sandbox_is_warm(plat, tt_a))


class TestActiveStorageDiskHit(unittest.TestCase):
    def test_checked_out_storage_visible_with_active_storage(self):
        tt = _task_type()
        storage = _storage(cached=True, task_type=tt)
        node = _node([])  # flash checked out — not in FilterStore.items
        self.assertTrue(
            node_has_cached_image(node, "rpiCpu", tt, active_storage=storage)
        )


class TestImagePullDiskHit(unittest.TestCase):
    def test_v1_always_false(self):
        node = _node([_storage(cached=True)])
        plat = _platform()
        self.assertFalse(image_pull_disk_hit(PLATFORM_REUSE_V1, plat, node, _task_type()))

    def test_v2_true_when_cached(self):
        node = _node([_storage(cached=True)])
        plat = _platform()
        self.assertTrue(image_pull_disk_hit(NODE_DISK_V2, plat, node, _task_type()))


if __name__ == "__main__":
    unittest.main()
