"""P0 on a real peer-corpus cache: the same ingredients, read under v2 and v3, must give
bit-identical base/linkrank columns and a fully recoverable rank on every edge."""
from pathlib import Path

import pytest

from scripts_cosim.partial_state_v3_p0 import compare_caches, compare_dataset, _load

SMOKE = Path("simulation_data/graphs_cache_peer_affinity_v1_r2_smoke")


@pytest.mark.skipif(not SMOKE.is_dir(), reason=f"cache not present at {SMOKE}")
def test_p0_on_the_r2_smoke_cache_is_clean_on_every_dataset():
    res = compare_caches(SMOKE, SMOKE)
    rows = res["per_dataset"]
    assert len(rows) == 34
    assert all(r["base_diff"] == 0.0 and r["link_diff"] == 0.0 for r in rows.values())
    assert all(r["rank_recovery"] == 1.0 for r in rows.values())
    assert res["n_prefix_covered"] == 34          # the teacher-forced prefix path is exercised
    assert res["verdict"] == "UNREADABLE"          # 34 < P0_MIN_DATASETS: the bar wants the corpus


@pytest.mark.skipif(not SMOKE.is_dir(), reason=f"cache not present at {SMOKE}")
def test_p0_catches_a_moved_ingredient():
    g = _load(SMOKE)
    gid = sorted(g)[0]
    import copy
    g3 = copy.deepcopy(g[gid])
    node = next(iter(g3.partial_state_ctx["node_caps"]))
    g3.partial_state_ctx["node_caps"][node] += 1.0
    with pytest.raises(ValueError, match="not a representation change"):
        compare_dataset(g[gid], g3)
