import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
import selfsearch_gap_batches as M  # noqa: E402


def world(monkeypatch):
    monkeypatch.setattr(M, "OPENING_TASKS", 6)
    sizes = [2, 4, 6, 10]  # peer groups of 2, 4, 6 and 10 tasks: ids 0-1, 2-5, 6-11, 12-21
    group, ts = [], []
    for g, n in enumerate(sizes):
        group += [g] * n
        ts += [float(g)] * n
    members = {}
    for i, g in enumerate(group):
        members.setdefault(g, []).append(i)
    return {"ts": ts, "group": group, "members": members}


def tasks(groups, lat, plat, cold=lambda i: 0):
    return {i: {"disp": groups["ts"][i], "lat": lat(i), "cold": cold(i), "plat": plat(i)} for i in range(len(groups["ts"]))}


def test_buckets_add_to_the_gap_and_cut_by_size_and_phase(monkeypatch, tmp_path):
    g = world(monkeypatch)
    a = tasks(g, lambda i: 3.0 if i < 2 else 1.0, lambda i: ("n", i))
    b = tasks(g, lambda i: 1.0, lambda i: ("n", i))
    cdr = tmp_path / "c.jsonl"
    cdr.write_text("")
    r = M.analyse(a, b, g, str(cdr))
    assert r["gap_s"] == pytest.approx(2.0 * 2 / 22)
    bk = r["buckets"]
    assert bk["quarter|opening|le4"]["gap_s"] == pytest.approx(r["gap_s"])  # only ids 0,1 differ, a size-2 group in the opening
    assert bk["quarter|rest|ge9"]["gap_s"] == 0.0
    parts = sum(v["gap_s"] for k, v in bk.items() if k.startswith("quarter|") and not k.endswith("|all"))
    assert parts == pytest.approx(r["gap_s"])
    assert sum(v["share"] for k, v in bk.items() if k.startswith("quarter|") and not k.endswith("|all")) == pytest.approx(1.0)


def test_stacking_and_cold_per_batch(monkeypatch):
    g = world(monkeypatch)
    t = tasks(g, lambda i: 1.0, lambda i: ("n", 0 if i in (2, 3) else i), cold=lambda i: 1 if i == 6 else 0)
    sc = M.stack_cold(t, g)
    assert sc[("all", "le4")][:2] == [2, 1]  # sizes 2 and 4: only the size-4 group stacks (ids 2 and 3 share a platform)
    assert sc[("all", "le4")][2] == 2 and sc[("all", "5to8")][3] == 1 and sc[("all", "all")][0] == 4


def test_disagreement_counts_differing_plans_and_s_gap(monkeypatch, tmp_path):
    g = world(monkeypatch)
    recs = [{"tasks": [0, 1], "seed": [[0, 0], [1, 0]], "refined": [[0, 0], [1, 0]], "s_seed": 5.0, "s_refined": 5.0, "exact_ran": 1, "exact_fallback": 0},
            {"tasks": [2, 3, 4, 5], "seed": [[0, 0]] * 4, "refined": [[0, 0], [1, 0], [0, 0], [1, 0]], "s_seed": 10.0, "s_refined": 8.0, "exact_ran": 1, "exact_fallback": 0}]
    p = tmp_path / "c.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs))
    d = M.disagreement(str(p), g)[("all", "all")]
    assert d["batches"] == 2 and d["differ_share"] == 0.5 and d["tasks_moved_share"] == pytest.approx(2 / 6)
    assert d["s_gap_mean"] == pytest.approx(1.0) and d["s_gap_rel_mean"] == pytest.approx(0.125)


def test_misaligned_task_ids_fail_loudly(monkeypatch):
    g = world(monkeypatch)
    t = tasks(g, lambda i: 1.0, lambda i: ("n", i))
    t[0]["disp"] += 10.0
    with pytest.raises(SystemExit, match="line up"):
        M.check_alignment(t, g)
