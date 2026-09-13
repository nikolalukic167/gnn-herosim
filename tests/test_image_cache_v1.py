"""image_cache_v1 — the bounded node image cache lever.

Two env levers, applied at generation time in `load_simulation_inputs`:

    HEROSIM_IMAGE_SIZE_MIN_GB / _MAX_GB   spread image size across TASK TYPES, the type
                                          that yields least willingly getting the largest
    HEROSIM_DISK_CAPACITY_GB              bound every LOCAL storage tier

The claim these tests defend is not "the numbers changed" but the three structural
properties the mechanism rests on:

  1. **Anti-alignment.** The task type that loses most by giving up its favourite platform
     carries the image that is most expensive to keep cached. Route A measured that coupling
     ALIGNED with the pointwise optimum leaves the componentwise argmin intact, so the
     direction is the mechanism, not a detail.
  2. **Heterogeneity ACROSS CO-RESIDENTS.** The shipped sizes are near-uniform
     (3.057/2.990/2.987) and a uniform-weight knapsack is a COUNT — the exact shape one
     occupancy integer repaired in five previous mechanisms. The spread must therefore be
     across TASK TYPES: on this grid each hosting node carries platform instances of a
     single type, so co-located tasks always share a platform and a per-platform spread
     would leave every co-resident image the same size.
  3. **Fail loud, never silently inert.** A half-set lever, a capacity that cannot hold the
     largest image, or an eviction that cannot make room are all config errors. Each raises.

Teeth (each fails if the behaviour it names regresses):
  - test_type_that_yields_least_willingly_gets_the_largest_image -> anti-alignment
  - test_co_located_tasks_get_different_sized_images             -> knapsack, not count
  - test_capacity_below_largest_image_is_refused                 -> the startup guard
  - test_store_function_raises_when_no_eviction_can_fit          -> the infinite-loop fix
  - test_both_levers_unset_change_nothing                        -> inertness by default
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.executecosimulation as ec  # noqa: E402
from src.placement.model import CacheEvictionError  # noqa: E402


LEVERS = (
    "HEROSIM_IMAGE_SIZE_MIN_GB",
    "HEROSIM_IMAGE_SIZE_MAX_GB",
    "HEROSIM_DISK_CAPACITY_GB",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in LEVERS:
        monkeypatch.delenv(name, raising=False)


def task_types():
    """The REAL four-type table from data/nofs-ids/task-types.json, verbatim.

    Faithful values are load-bearing: the ranking is by yield cost (second-best minus best
    coldStartDuration + executionTime), and the assertions below name the real ordering
    cnn > dnn2 > dnn1 > rf. A made-up table could not catch a ranking regression, and
    test_yield_cost_matches_the_shipped_table pins these against the file on disk.
    """
    return {
        "dnn1": {
            "name": "dnn1",
            "platforms": ["rpiCpu", "xavierGpu", "xavierCpu", "pynqFpga"],
            "coldStartDuration": {
                "rpiCpu": 0.33, "xavierGpu": 6.17, "xavierCpu": 0.071, "pynqFpga": 0.98,
            },
            "executionTime": {
                "rpiCpu": 0.00290875, "xavierGpu": 0.020835,
                "xavierCpu": 0.0010475, "pynqFpga": 0.00056625,
            },
            "imageSize": {
                "rpiCpu": 3.057, "xavierGpu": 2.990,
                "xavierCpu": 2.987, "pynqFpga": 0.004,
            },
        },
        "dnn2": {
            "name": "dnn2",
            "platforms": ["rpiCpu", "xavierGpu", "xavierCpu"],
            "coldStartDuration": {"rpiCpu": 0.636, "xavierGpu": 6.14, "xavierCpu": 0.096},
            "executionTime": {
                "rpiCpu": 0.16842, "xavierGpu": 0.0362488, "xavierCpu": 0.0239175,
            },
            "imageSize": {"rpiCpu": 3.057, "xavierGpu": 2.990, "xavierCpu": 2.987},
        },
        "rf": {
            "name": "rf",
            "platforms": ["rpiCpu", "xavierGpu", "xavierCpu"],
            "coldStartDuration": {"rpiCpu": 0.067, "xavierGpu": 12.72, "xavierCpu": 0.05},
            "executionTime": {
                "rpiCpu": 0.00422375, "xavierGpu": 0.0004975, "xavierCpu": 0.00357625,
            },
            "imageSize": {"rpiCpu": 3.057, "xavierGpu": 2.990, "xavierCpu": 2.987},
        },
        "cnn": {
            "name": "cnn",
            "platforms": ["rpiCpu", "xavierCpu", "xavierGpu", "xavierDla"],
            "coldStartDuration": {
                "rpiCpu": 1.162, "xavierCpu": 0.135, "xavierGpu": 10.62, "xavierDla": 38.0,
            },
            "executionTime": {
                "rpiCpu": 3.0858438, "xavierCpu": 0.7055338,
                "xavierGpu": 0.1036875, "xavierDla": 0.5617938,
            },
            "imageSize": {
                "rpiCpu": 3.057, "xavierCpu": 2.987, "xavierGpu": 2.990, "xavierDla": 2.990,
            },
        },
    }


def storage_types():
    return {
        "eMMC": {"name": "eMMC", "remote": False, "capacity": 32},
        "flashCard": {"name": "Flash Card", "remote": False, "capacity": 64},
        "someRemote": {"name": "Some Remote Storage", "remote": True, "capacity": 10 ** 12},
    }


def sim_inputs():
    return {"task_types": task_types(), "storage_types": storage_types()}


# --------------------------------------------------------------------------------------
# 1. anti-alignment
# --------------------------------------------------------------------------------------

def test_type_that_yields_least_willingly_gets_the_largest_image(monkeypatch):
    """The task that loses most by moving off its favourite carries the priciest image.

    Yield cost = (second-best - best) of coldStart+exec, measured on the shipped table:
    cnn 3.408 > dnn2 0.684 > dnn1 0.261 > rf 0.018.
    """
    monkeypatch.setenv("HEROSIM_IMAGE_SIZE_MIN_GB", "1.0")
    monkeypatch.setenv("HEROSIM_IMAGE_SIZE_MAX_GB", "8.0")
    si = sim_inputs()
    ec.apply_image_size_override(si)

    size = {t: si["task_types"][t]["imageSize"]["rpiCpu"] for t in si["task_types"]}
    assert size["cnn"] == pytest.approx(8.0)
    assert size["rf"] == pytest.approx(1.0)
    assert size["cnn"] > size["dnn2"] > size["dnn1"] > size["rf"]


def test_yield_cost_matches_the_shipped_table():
    """Pins the ranking against data/nofs-ids/task-types.json, not against the fixture.

    If the shipped timings ever move, the anti-alignment argument has to be re-derived
    rather than silently inheriting a stale ordering.
    """
    import json

    shipped = json.load(open(Path(__file__).resolve().parent.parent
                             / "data/nofs-ids/task-types.json"))
    assert ec._image_task_type_order(shipped) == ["cnn", "dnn2", "dnn1", "rf"]
    assert ec._image_task_type_order(task_types()) == ["cnn", "dnn2", "dnn1", "rf"]


def test_second_best_not_worst_is_what_ranks(monkeypatch):
    """Teeth for the ranking CHOICE, not just its output.

    A max-minus-min gap would rank rf second (its xavierGpu is 12.7 s, an option no plan
    takes) and put the type that cares least about yielding near the top.
    """
    tt = task_types()
    worst_gap_order = sorted(
        tt,
        key=lambda n: -(max(tt[n]["coldStartDuration"][p] + tt[n]["executionTime"][p]
                            for p in tt[n]["platforms"])
                        - min(tt[n]["coldStartDuration"][p] + tt[n]["executionTime"][p]
                              for p in tt[n]["platforms"])),
    )
    assert worst_gap_order == ["cnn", "rf", "dnn1", "dnn2"]
    assert ec._image_task_type_order(tt) != worst_gap_order


def test_ranking_is_deterministic_under_a_tie():
    """Ties break by task type NAME, not by dict/set order.

    The repo's classic determinism leak is an unordered tie-break over objects, and
    PYTHONHASHSEED does not pin it (herosim-pythonhashseed-tiebreak-nondeterminism).
    """
    flat = {"cold": 1.0, "warm": 0.0}
    tt = {
        name: {
            "platforms": ["p1", "p2"],
            "coldStartDuration": {"p1": flat["warm"], "p2": flat["cold"]},
            "executionTime": {"p1": 0.0, "p2": 0.0},
            "imageSize": {"p1": 3.0, "p2": 3.0},
        }
        for name in ("bbb", "aaa", "ccc")
    }
    assert ec._image_task_type_order(tt) == ["aaa", "bbb", "ccc"]


def test_single_platform_type_cannot_be_ranked():
    tt = {
        "t": {
            "platforms": ["only"],
            "coldStartDuration": {"only": 1.0},
            "executionTime": {"only": 0.0},
            "imageSize": {"only": 3.0},
        }
    }
    with pytest.raises(RuntimeError, match="fewer than 2 platforms"):
        ec._image_task_type_order(tt)


# --------------------------------------------------------------------------------------
# 2. heterogeneity — the property that separates a knapsack from a count
# --------------------------------------------------------------------------------------

def test_spread_is_geometric_and_actually_spread(monkeypatch):
    monkeypatch.setenv("HEROSIM_IMAGE_SIZE_MIN_GB", "0.5")
    monkeypatch.setenv("HEROSIM_IMAGE_SIZE_MAX_GB", "8.0")
    si = sim_inputs()
    ec.apply_image_size_override(si)

    ordered = ec._image_task_type_order(si["task_types"])
    sizes = [si["task_types"][t]["imageSize"]["rpiCpu"] for t in ordered]
    ratios = [sizes[i] / sizes[i + 1] for i in range(len(sizes) - 1)]
    assert ratios == pytest.approx([ratios[0]] * len(ratios))  # constant ratio
    assert sizes[0] / sizes[-1] == pytest.approx(16.0)


def test_size_is_uniform_across_one_types_platforms(monkeypatch):
    """Per-type-uniform keeps the additive part of the lever a per-task constant, so it
    cannot move a task's platform preference — only co-residency pressure is new."""
    monkeypatch.setenv("HEROSIM_IMAGE_SIZE_MIN_GB", "1.0")
    monkeypatch.setenv("HEROSIM_IMAGE_SIZE_MAX_GB", "6.0")
    si = sim_inputs()
    ec.apply_image_size_override(si)
    for name, tt in si["task_types"].items():
        sizes = {tt["imageSize"][p] for p in tt["platforms"]}
        assert len(sizes) == 1, f"{name} image size varies by platform"


def test_co_located_tasks_get_different_sized_images(monkeypatch):
    """The property that makes the disk a knapsack instead of a count.

    On this grid a hosting node carries platform instances of ONE type (node0 = 4x rpiCpu),
    so co-residents always share a platform. Two tasks on that node must still be able to
    cost different amounts of disk, or one occupancy integer repairs the whole mechanism.
    """
    monkeypatch.setenv("HEROSIM_IMAGE_SIZE_MIN_GB", "0.5")
    monkeypatch.setenv("HEROSIM_IMAGE_SIZE_MAX_GB", "6.0")
    si = sim_inputs()
    ec.apply_image_size_override(si)

    on_one_node = {t: si["task_types"][t]["imageSize"]["rpiCpu"] for t in si["task_types"]}
    assert len(set(on_one_node.values())) == len(on_one_node)
    # Two pairs of the SAME size (2 tasks) needing materially different disk is the point.
    pair_a = on_one_node["cnn"] + on_one_node["rf"]
    pair_b = on_one_node["dnn1"] + on_one_node["dnn2"]
    assert abs(pair_a - pair_b) > 1.0, (
        "two-task sets cost nearly the same disk; a count would be a sufficient statistic"
    )


def test_shipped_sizes_are_too_uniform_to_be_a_knapsack():
    """Documents WHY the size lever is mandatory rather than optional.

    The shipped table gives every task type the same ~3 GB on any given platform, so a
    bounded disk alone would price a co-resident set by its COUNT. If someone ever makes the
    shipped table heterogeneous across task types, this fails and the "both levers are
    required" reasoning in the grid preset needs rewriting.
    """
    shipped = task_types()
    for platform in ("rpiCpu", "xavierCpu"):
        per_type = [tt["imageSize"][platform] for tt in shipped.values()
                    if platform in tt["imageSize"]]
        assert max(per_type) / min(per_type) < 1.05, (
            f"shipped image sizes on {platform} are no longer uniform across task types; "
            f"the argument that the size lever is load-bearing has to be re-derived"
        )


# --------------------------------------------------------------------------------------
# 3. fail loud
# --------------------------------------------------------------------------------------

def test_half_set_size_lever_is_refused(monkeypatch):
    monkeypatch.setenv("HEROSIM_IMAGE_SIZE_MAX_GB", "4.0")
    with pytest.raises(RuntimeError, match="must be set together"):
        ec.apply_image_size_override(sim_inputs())


def test_inverted_bounds_are_refused(monkeypatch):
    monkeypatch.setenv("HEROSIM_IMAGE_SIZE_MIN_GB", "4.0")
    monkeypatch.setenv("HEROSIM_IMAGE_SIZE_MAX_GB", "1.0")
    with pytest.raises(ValueError, match="must be >="):
        ec.apply_image_size_override(sim_inputs())


def test_platform_without_a_timing_entry_is_refused(monkeypatch):
    monkeypatch.setenv("HEROSIM_IMAGE_SIZE_MIN_GB", "1.0")
    monkeypatch.setenv("HEROSIM_IMAGE_SIZE_MAX_GB", "4.0")
    si = sim_inputs()
    si["task_types"]["dnn1"]["platforms"].append("ghostPlatform")
    with pytest.raises(RuntimeError, match="ghostPlatform"):
        ec.apply_image_size_override(si)


def test_capacity_below_largest_image_is_refused(monkeypatch):
    """The guard that keeps store_function's eviction loop off the unsatisfiable path."""
    monkeypatch.setenv("HEROSIM_DISK_CAPACITY_GB", "2.0")
    with pytest.raises(RuntimeError, match="smaller than the largest single image"):
        ec.apply_disk_capacity_override(sim_inputs())


def test_capacity_guard_sees_the_OVERRIDDEN_sizes_not_the_shipped_ones(monkeypatch):
    """Ordering teeth: the size lever must be applied BEFORE the capacity guard runs.

    Shipped max is 3.057, so a 5 GB disk looks fine against the file. Spread up to 8 GB and
    it is not. If load_simulation_inputs ever calls these two in the wrong order, a corpus
    would generate with a capacity no image can fit.
    """
    monkeypatch.setenv("HEROSIM_IMAGE_SIZE_MIN_GB", "1.0")
    monkeypatch.setenv("HEROSIM_IMAGE_SIZE_MAX_GB", "8.0")
    monkeypatch.setenv("HEROSIM_DISK_CAPACITY_GB", "5.0")
    si = sim_inputs()
    ec.apply_image_size_override(si)
    with pytest.raises(RuntimeError, match="smaller than the largest single image"):
        ec.apply_disk_capacity_override(si)


def test_load_simulation_inputs_applies_size_before_capacity():
    """Pins the call order structurally, not by re-running the whole loader."""
    import inspect

    source = inspect.getsource(ec.load_simulation_inputs)
    assert source.index("apply_image_size_override") < source.index(
        "apply_disk_capacity_override"
    ), "the capacity guard must see overridden image sizes; keep size first"


# --------------------------------------------------------------------------------------
# 4. what the levers touch, and what they must not
# --------------------------------------------------------------------------------------

def test_capacity_bounds_local_tiers_only(monkeypatch):
    """The image cache lives on local disk — warmth.node_has_cached_image skips remote."""
    monkeypatch.setenv("HEROSIM_DISK_CAPACITY_GB", "6.5")
    si = sim_inputs()
    applied = ec.apply_disk_capacity_override(si)

    assert applied == pytest.approx(6.5)
    assert si["storage_types"]["eMMC"]["capacity"] == pytest.approx(6.5)
    assert si["storage_types"]["flashCard"]["capacity"] == pytest.approx(6.5)
    assert si["storage_types"]["someRemote"]["capacity"] == 10 ** 12


def test_size_lever_touches_only_imageSize(monkeypatch):
    monkeypatch.setenv("HEROSIM_IMAGE_SIZE_MIN_GB", "1.0")
    monkeypatch.setenv("HEROSIM_IMAGE_SIZE_MAX_GB", "4.0")
    before = task_types()
    si = {"task_types": task_types(), "storage_types": storage_types()}
    ec.apply_image_size_override(si)

    for name, tt in si["task_types"].items():
        for key in ("platforms", "coldStartDuration", "executionTime"):
            assert tt[key] == before[name][key], f"{name}.{key} moved"
    assert si["storage_types"] == storage_types()


def test_both_levers_unset_change_nothing():
    si = sim_inputs()
    assert ec.apply_image_size_override(si) is None
    assert ec.apply_disk_capacity_override(si) is None
    assert si == sim_inputs()


# --------------------------------------------------------------------------------------
# 5. the storage-layer fail-loud fix
# --------------------------------------------------------------------------------------

class _FakeEnv:
    now = 0.0


class _FakeNode:
    def __repr__(self):
        return "node0"


def _storage(capacity_gb):
    """A Storage with the real store_function, built without SimPy's constructor path."""
    from src.placement.infrastructure import Storage

    s = object.__new__(Storage)
    s.env = _FakeEnv()
    s.node = _FakeNode()
    s.id = 0
    s.type = {"name": "tiny", "capacity": capacity_gb}
    s.functions_cache = []
    s.data_store = {}
    s.used = 0
    s.writes = 0
    s.erases = 0
    s.cache_usage = []
    s.data_usage = []
    s.total_usage = []
    s.eviction_policies = {"fifo": s.eviction_fifo}
    return s


def test_store_function_raises_when_no_eviction_can_fit():
    """Before the fix this SPUN FOREVER: cache_eviction swallows CacheEvictionError and
    returns False, so store_function's `except` was dead code and the while loop never
    terminated on an empty, still-too-small cache."""
    s = _storage(capacity_gb=2.0)
    s.node.policy = type("P", (), {"cache": "fifo"})()
    task_type = {"name": "dnn1", "imageSize": {"xavierCpu": 4.0}}

    with pytest.raises(CacheEvictionError, match="cannot cache"):
        s.store_function("xavierCpu", task_type)


def test_store_function_evicts_and_succeeds_when_room_can_be_made():
    """The intended path: a bounded disk evicts the older image and caches the new one.

    This is the coupling itself — the evicted task pays a full cold pull next time.
    """
    s = _storage(capacity_gb=6.5)
    s.node.policy = type("P", (), {"cache": "fifo"})()
    first = {"name": "dnn1", "imageSize": {"xavierCpu": 3.0}}
    second = {"name": "dnn2", "imageSize": {"xavierCpu": 3.0}}
    third = {"name": "rf", "imageSize": {"xavierCpu": 3.0}}

    assert s.store_function("xavierCpu", first) is True
    assert s.store_function("xavierCpu", second) is True
    assert len(s.functions_cache) == 2
    # The third does not fit alongside two 3 GB images in 6.5 GB -> FIFO drops the first.
    assert s.store_function("xavierCpu", third) is True
    assert [t["name"] for _, t in s.functions_cache] == ["dnn2", "rf"]


def test_unbounded_disk_never_evicts():
    """The paired control arm: same image sizes, capacity left alone, no coupling."""
    s = _storage(capacity_gb=64)
    s.node.policy = type("P", (), {"cache": "fifo"})()
    for name in ("a", "b", "c", "d"):
        assert s.store_function("xavierCpu", {"name": name, "imageSize": {"xavierCpu": 3.0}})
    assert len(s.functions_cache) == 4


# --------------------------------------------------------------------------------------
# 6. the grid preset
# --------------------------------------------------------------------------------------

def test_grid_preset_shape_and_fresh_seeds():
    from scripts_cosim.generate_gnn_datasets_fast import (
        GRID_PRESETS,
        IMAGE_CACHE_V1_GRID,
        ROUTE_B_PIVOT_H2_GRID,
    )

    assert GRID_PRESETS["image_cache_v1"] is IMAGE_CACHE_V1_GRID
    # H2's shape is borrowed on purpose: it is the only 4-task regime whose sweeps are big
    # enough for the additivity fit to mean anything (44-72 rows/param vs H0/H1's 1.14).
    assert IMAGE_CACHE_V1_GRID["replica_configs"] == ROUTE_B_PIVOT_H2_GRID["replica_configs"]
    assert IMAGE_CACHE_V1_GRID["replica_overlap"] is True
    # Fresh seeds: no overlap with ANY block used by route_b or the pivot ladder.
    burned = (
        set(range(901, 918)) | set(range(2001, 2043)) | set(range(3001, 3018))
        | set(range(3101, 3118)) | set(range(3201, 3218)) | set(range(3301, 3318))
        | set(range(3401, 3418)) | set(range(3501, 3518))
    )
    assert not burned & set(IMAGE_CACHE_V1_GRID["seeds"])
    assert IMAGE_CACHE_V1_GRID["default_output_subdir"] != ROUTE_B_PIVOT_H2_GRID[
        "default_output_subdir"
    ], "must not generate on top of the amended H2 corpus"


def test_skip_threshold_default_covers_this_grid():
    """Derived from THIS grid, not a neighbouring rung's — the defect that kept a test
    green while it asserted H0's bound for H3 (route-b-preflight, instance 7)."""
    from scripts_cosim.generate_gnn_datasets_fast import (
        IMAGE_CACHE_V1_GRID,
        MAX_PLACEMENT_COMBINATIONS_SKIP_DEFAULT,
    )

    per_server = max(cfg[1] for cfg in IMAGE_CACHE_V1_GRID["replica_configs"])
    server_nodes = max(IMAGE_CACHE_V1_GRID["server_node_counts"])
    n_tasks = len(IMAGE_CACHE_V1_GRID["dag_task_types"])
    assert IMAGE_CACHE_V1_GRID.get("dag_instances", 1) == 1, "4-task preset only"
    bound = (per_server * server_nodes) ** n_tasks
    assert bound <= MAX_PLACEMENT_COMBINATIONS_SKIP_DEFAULT, (
        f"pre-uniqueness bound {bound} exceeds the {MAX_PLACEMENT_COMBINATIONS_SKIP_DEFAULT} "
        f"default; generation must export MAX_PLACEMENT_COMBINATIONS_SKIP"
    )
