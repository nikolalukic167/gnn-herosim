import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import uncut_caphit as C  # noqa: E402


def _cell(tmp, topo, w, rung, arm, eb, ch, plans, served):
    sc = {"pg_exact_batches": eb, "pg_exact_fallbacks": ch, "pg_exact_plans": plans, "pgb_max_batch": 40}
    for b, n in zip(C.BUCKETS, served):
        sc[f"pgb_whole_{b}"], sc[f"pgb_split_{b}"] = n, 0
    (tmp / f"cc40s{topo}__{w}{rung}__{arm}_s0.summary.json").write_text(json.dumps({"schedulerCounters": sc}))


def test_pool_shares_and_failed_cells(tmp_path):
    _cell(tmp_path, 1, "g0", "heavy", "cd_exactS_ib0", 80, 20, 8000, (100, 10, 5, 1))
    _cell(tmp_path, 2, "g1", "heavy", "cd_exactS_ib0", 20, 0, 2000, (50, 0, 0, 0))
    _cell(tmp_path, 1, "g0", "heavy", "cd_exactS", 1, 1, 1, (1, 0, 0, 0))
    (tmp_path / "cc40s3__g0heavy__cd_exactS_ib0_s0.failed.json").write_text("{}")
    r = C.read_dir(tmp_path, ["cd_exactS_ib0", "cd_exactS"])
    p = r["cd_exactS_ib0"]["heavy"]["pooled"]
    assert p["cells"] == 2 and p["exact_batches"] == 100 and p["cap_hits"] == 20 and abs(p["cap_hit_share_of_scored_batches"] - 20 / 120) < 1e-12
    assert p["plans_per_exact_batch"] == 100 and p["served_batches"]["11_50"] == 10 and r["cd_exactS_ib0"]["heavy"]["failed_cells"] == 1
    assert r["cd_exactS_ib0"]["moderate"]["pooled"] == {"cells": 0}
    assert r["cd_exactS"]["heavy"]["pooled"]["cells"] == 1


def test_missing_counter_fails_loud():
    with pytest.raises(KeyError, match="FAIL LOUD"):
        C.cell_row({"schedulerCounters": {"pg_exact_batches": 1}})
