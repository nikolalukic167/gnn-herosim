"""uncut_groups_s0_v1 K1(b) tool: the ILS (pure, synthetic S), the reader and the bucket rule. The simulator-backed parts are the datalab smoke."""
import itertools
import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import uncut_ils as U  # noqa: E402


def _problem(n, c, seed):
    rng = random.Random(seed)
    cands = [[(a, 0) for a in range(c)] for _ in range(n)]
    uni = [[rng.random() * 5 for _ in range(c)] for _ in range(n)]
    pair = {(i, j): [[rng.random() * 3 for _ in range(c)] for _ in range(c)] for i in range(n) for j in range(i + 1, n) if rng.random() < 0.5}

    def S(plan):
        v = sum(uni[i][plan[i][0]] for i in range(n))
        for (i, j), m in pair.items():
            v += m[plan[i][0]][plan[j][0]]
        return v
    return cands, S


@pytest.mark.parametrize("seed", range(12))
def test_reaches_the_exact_optimum_on_small_states(seed):
    n = 4 + seed % 3
    cands, S = _problem(n, 3, seed)
    opt = min(S(p) for p in itertools.product(*cands))
    r = U.ils(cands, tuple(c[0] for c in cands), S, seed=seed, max_evals=3000)
    assert abs(r["best_S"] - opt) < 1e-9


def test_two_runs_are_byte_identical():
    cands, S = _problem(30, 4, 5)
    start = tuple(c[0] for c in cands)
    a = U.ils(cands, start, S, seed=3, max_evals=4000)
    b = U.ils(cands, start, S, seed=3, max_evals=4000)
    assert (a["best"], a["evals"], a["trajectory"]) == (b["best"], b["evals"], b["trajectory"])
    assert U.ils(cands, start, S, seed=4, max_evals=4000)["evals"] != a["evals"]


def test_budget_incumbents_and_start():
    cands, S = _problem(40, 4, 1)
    start = tuple(c[0] for c in cands)
    r = U.ils(cands, start, S, seed=0, max_evals=500)
    assert r["n_evals"] == 500 and r["stopped"] == "evals" and len(r["evals"]) == 500
    assert r["trajectory"][0]["eval"] == 1 and r["trajectory"][0]["plan"] == start
    assert [t["S"] for t in r["trajectory"]] == sorted((t["S"] for t in r["trajectory"]), reverse=True)
    assert r["best_S"] <= r["start_S"] and r["best_S"] == r["trajectory"][-1]["S"]
    assert len({e["plan"] for e in r["evals"]}) == 500          # distinct plans only


def test_pinned_task_never_moves_and_cap_fails_loud():
    cands, S = _problem(8, 3, 2)
    start = [c[0] for c in cands]
    start[3] = (9, 9)                                           # not among its candidates: pinned
    r = U.ils(cands, tuple(start), lambda p: S(tuple(k if k[0] < 3 else (0, 0) for k in p)), seed=0, max_evals=800)
    assert all(e["plan"][3] == (9, 9) for e in r["evals"])
    big = [[(0, 0), (1, 0)]] * (U.CAP + 1)
    with pytest.raises(ValueError, match="label cap"):
        U.ils(big, tuple(c[0] for c in big), lambda p: 0.0, seed=0, max_evals=10)


def test_default_budget_is_the_registered_one():
    assert U.default_budget([[(0, 0)] * 5] * 10) == 200 * 10 * 5


def test_bucket_rule():
    assert [U.bucket_of(n) for n in (11, 50, 51, 100, 101, 226)] == ["11_50", "11_50", "51_100", "51_100", "101_226", "101_226"]
    with pytest.raises(ValueError):
        U.bucket_of(10)


def _row(bucket, hand, ils, s_hand, s_ils):
    return {"bucket": bucket, "regret": (hand - ils) / ils, "s_regret": (s_hand - s_ils) / s_ils}


def test_reader_mean_share_p90_and_replay_rules():
    rows = [_row("11_50", 100, 100, 5, 5), _row("11_50", 110, 100, 6, 5), _row("51_100", 100, 100, 7, 5), _row("101_226", 90, 100, 6, 5)]
    r = U.read_rows(rows)
    assert r["pooled"]["n"] == 4 and abs(r["pooled"]["mean_regret_pct"] - 0.0) < 1e-9      # +10, 0, 0, -10
    assert r["pooled"]["share_within_1pct"] == 0.75 and r["pooled"]["share_ils_worse_than_hand"] == 0.25
    assert r["11_50"]["n"] == 2 and abs(r["11_50"]["mean_regret_pct"] - 5.0) < 1e-9 and r["11_50"]["kill_K1b_mean_ge_1pct"]
    assert r["101_226"]["share_s_and_replay_disagree"] == 1.0                 # S says ILS better, the replay says it is worse
    assert r["pooled"]["kill_K1b_mean_ge_1pct"] is False


def test_reader_refuses_rows_without_a_replayed_regret():
    with pytest.raises(ValueError, match="FAIL LOUD"):
        U.read_rows([{"bucket": "11_50", "n": 12}])


def _cells(tmp_path, recs):
    cells, inputs = tmp_path / "cells", tmp_path / "inputs"
    (inputs / "wf1_heavy" / "cfg").mkdir(parents=True)
    cells.mkdir()
    (inputs / "wf1_heavy" / "cfg" / "cc40s1.json").write_text("{}")
    (cells / "cc40s1__g0heavy__cd_expand_k100_s0.snapshots.jsonl").write_text(
        "".join(json.dumps({"trigger_task_id": t, "time": tm, "fidelity": {"batch": [0] * n}}) + "\n" for t, tm, n in recs))
    (cells / "cc40s1__g0heavy__cd_expand_k100_s0.summary.json").write_text("{}")
    return cells, inputs


def _st(sid, trig, bucket="11_50"):
    return {"state_id": sid, "topology": 1, "window": "g0", "rung": "heavy", "bucket": bucket, "group_size": 12, "trigger_task_id": trig}


def test_make_jobs_fails_loud_on_a_missing_file(tmp_path):
    cells, inputs = _cells(tmp_path, [(7, 1.0, 12)])
    with pytest.raises(FileNotFoundError, match="FAIL LOUD"):
        U.make_jobs([_st("a", 7)], cells, "no_such_token", inputs, "x")


def test_first_capture_is_the_state_and_extras_are_counted(tmp_path):
    cells, inputs = _cells(tmp_path, [(7, 1.0, 12), (9, 2.0, 11), (7, 1.01, 5), (7, 1.02, 3)])
    jobs, gone = U.make_jobs([_st("a", 7), _st("b", 9)], cells, "cd_expand_k100", inputs, "x")
    a, b = jobs
    assert gone == [] and a["time"] == 1.0 and a["first_capture_n"] == 12 and a["n_extra_captures"] == 2 and b["n_extra_captures"] == 0
    snap = tmp_path / "cells" / "cc40s1__g0heavy__cd_expand_k100_s0.snapshots.jsonl"
    assert U._load_snapshot({"snapfile": str(snap), "trigger_task_id": 7, "time": a["time"]})["time"] == 1.0


def test_ids_the_cell_lists_as_missing_are_skipped(tmp_path):
    cells, inputs = _cells(tmp_path, [(7, 1.0, 12)])
    (cells / "cc40s1__g0heavy__cd_expand_k100_s0.summary.json").write_text(json.dumps({"sealed_snapshots": {"wanted": 2, "written": 1, "missing": [99]}}))
    jobs, gone = U.make_jobs([_st("a", 7), _st("gone", 99, "101_226")], cells, "cd_expand_k100", inputs, "x")
    assert [j["state_id"] for j in jobs] == ["a"] and [(g["state_id"], g["bucket"]) for g in gone] == [("gone", "101_226")]
    with pytest.raises(FileNotFoundError, match="never captured"):
        U.make_jobs([_st("other", 98)], cells, "cd_expand_k100", inputs, "x")


def test_missing_ids_are_skipped_and_reported_but_unlisted_ones_fail_loud(tmp_path):
    cells, inputs = _cells(tmp_path, [(7, 1.0, 12)])
    sealed = [_st("a", 7), _st("gone", 99, "51_100"), _st("gone2", 98, "101_226")]
    with pytest.raises(FileNotFoundError, match="never captured"):
        U.make_jobs(sealed, cells, "cd_expand_k100", inputs, "x", missing=["gone"])
    jobs, gone = U.make_jobs(sealed, cells, "cd_expand_k100", inputs, "x", missing=["gone", {"topology": 1, "window": "g0", "rung": "heavy", "trigger_task_id": 98}])
    assert [j["state_id"] for j in jobs] == ["a"] and [(g["state_id"], g["bucket"]) for g in gone] == [("gone", "51_100"), ("gone2", "101_226")]
    r = U.read_rows([_row("11_50", 100, 100, 5, 5) | {"n_extra_captures": 2}], gone)
    assert r["missing"]["total"] == 2 and r["missing"]["51_100"] == ["gone"] and r["11_50"]["n_extra_captures_total"] == 2 and r["11_50"]["states_with_extra_captures"] == 1


def _snap(chosen):
    return {"fidelity": {"batch": [{"gid": 5}, {"gid": 3}]}, "chosen": chosen}


def test_chosen_check_is_bit_equal_and_fails_loud():
    assert U.check_chosen(_snap({"3": [1, 2], "5": [4, 5]}), [[1, 2], [4, 5]], "s")
    with pytest.raises(RuntimeError, match="differs from the served plan"):
        U.check_chosen(_snap({"3": [1, 2], "5": [4, 5]}), [[1, 2], [4, 6]], "s")
    with pytest.raises(ValueError, match="not a map over the batch"):
        U.check_chosen(_snap({"3": [1, 2]}), [[1, 2], [4, 5]], "s")
