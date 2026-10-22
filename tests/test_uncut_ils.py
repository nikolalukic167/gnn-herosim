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
    a, b = [j for j in jobs if j["role"] == "first"]
    assert len(jobs) == 4 and gone == [] and a["time"] == 1.0 and a["first_capture_n"] == 12 and a["n_extra_captures"] == 2 and b["n_extra_captures"] == 0
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
    r = U.read_rows([_row("11_50", 100, 100, 5, 5)], gone)
    assert r["missing"]["total"] == 2 and r["missing"]["51_100"] == ["gone"] and r["missing"]["101_226"] == ["gone2"]


def _snap():
    return {"fidelity": {"batch": [{"gid": 5}, {"gid": 3}, {"gid": 9}]}}


def _rec(chosen, deferred=()):
    return {"trigger_task_id": 3, "chosen": chosen, "deferred": list(deferred), "arm": "cd_expand", "plan_evals": 10, "cap_hit": False}


def test_chosen_check_is_bit_equal_and_fails_loud():
    hand = [[1, 2], [4, 5], [7, 8]]                                  # gids ascending: 3, 5, 9
    ok = U.check_chosen(_snap(), hand, _rec([[3, 1, 2], [5, 4, 5], [9, 7, 8]]), "s")
    assert ok["match"] and ok["compared"] == 3 and ok["live_cap_hit"] is False
    with pytest.raises(RuntimeError, match="differs from the served plan"):
        U.check_chosen(_snap(), hand, _rec([[3, 1, 2], [5, 4, 6], [9, 7, 8]]), "s")
    with pytest.raises(ValueError, match="do not partition"):
        U.check_chosen(_snap(), hand, _rec([[3, 1, 2], [5, 4, 5]]), "s")


def test_chosen_check_in_a_state_with_deferrals_flags_but_does_not_stop():
    hand = [[1, 2], [4, 5], [7, 8]]
    r = U.check_chosen(_snap(), hand, _rec([[3, 1, 2], [5, 4, 6]], deferred=[9]), "s")
    assert r["match"] is False and r["deferred"] == 1 and r["mismatched_tasks"] == 1 and r["compared"] == 2
    assert U.check_chosen(_snap(), hand, _rec([[3, 1, 2], [5, 4, 5]], deferred=[9]), "s")["match"] is True


def _r(sid, bucket, hand, ils, **kw):
    return {"state_id": sid, "bucket": bucket, "regret": (hand - ils) / ils, "s_regret": 0.0, "rtt_hand": hand, "rtt_ils": ils, **kw}


def test_mean_is_printed_twice_and_imputation_can_unsatisfy_k1b():
    rows = [_r("a", "11_50", 100, 100), _r("b", "11_50", 100, 100), _r("c", "11_50", 150, 100), _r("d", "51_100", 100, 100)]    # regrets 0, 0, 50 %, 0
    ok = U.read_rows(rows)
    assert ok["mean_captured_pct"] == ok["mean_imputed_pct"] == 12.5 and ok["NOT_SATISFIED"] is False
    rows = [_r(c, "11_50", 100, 100) for c in "abcd"] + [_r("e", "51_100", 100.5, 100)]      # captured mean 0.1 %
    gone = [{"state_id": "m1", "bucket": "11_50"}, {"state_id": "m2", "bucket": "51_100"}]
    r = U.read_rows(rows, gone)
    assert abs(r["mean_captured_pct"] - 0.1) < 1e-9
    assert abs(r["mean_imputed_pct"] - 100 * (0.005 + 0.0 + 0.005) / 7) < 1e-9          # m1 at 11_50's max 0, m2 at 51_100's max 0.5 %
    assert r["NOT_SATISFIED"] is False
    rows = [_r("a", "11_50", 100, 100), _r("b", "11_50", 101.5, 100), _r("c", "11_50", 100, 100)]                                  # captured mean 0.5 %
    r = U.read_rows(rows, [{"state_id": "m1", "bucket": "11_50"}, {"state_id": "m2", "bucket": "11_50"}])        # imputed at 1.5 %: (0+1.5+0+1.5+1.5)/5 = 0.9 %
    assert r["mean_imputed_pct"] < 1.0 and r["NOT_SATISFIED"] is False
    r = U.read_rows(rows, [{"state_id": f"m{i}", "bucket": "11_50"} for i in range(4)])                           # (1.5 + 4 x 1.5)/7 = 1.07 %
    assert r["mean_captured_pct"] < 1.0 <= r["mean_imputed_pct"] and r["NOT_SATISFIED"] is True


def test_sealed_bucket_name_gt100_lands_in_the_top_bucket_and_unknown_buckets_fail():
    rows = [_r("a", "gt100", 100, 100), _r("b", "11_50", 110, 100)]
    r = U.read_rows(rows, [{"state_id": "m", "bucket": "gt100"}])
    assert r["101_226"]["n"] == 1 and r["missing"]["101_226"] == ["m"] and r["accounting"]["by_bucket"]["101_226"]["missing"] == 1
    with pytest.raises(ValueError):
        U.read_rows([_r("z", "weird", 100, 100)])


def test_unreplayable_states_are_listed_apart_from_missing_and_imputed_like_missing():
    rows = [_r("a", "11_50", 100, 100), _r("b", "11_50", 101.5, 100), _r("c", "11_50", 100, 100)]            # captured mean 0.5 %
    bad = [{"state_id": f"u{i}", "bucket": "11_50"} for i in range(4)]
    r = U.read_rows(rows, (), bad)
    assert r["unreplayable"]["total"] == 4 and r["unreplayable"]["11_50"] == ["u0", "u1", "u2", "u3"] and r["missing"]["total"] == 0
    assert abs(r["unreplayable"]["share_of_sealed"] - 4 / 7) < 1e-9
    assert r["mean_captured_pct"] < 1.0 <= r["mean_imputed_pct"] and r["NOT_SATISFIED"] is True        # (1.5 + 4 x 1.5)/7 = 1.07 %
    assert r["accounting"]["sealed"] == 7 and r["accounting"]["by_bucket"]["11_50"] == {"replayable_main": 3, "unreplayable": 4, "missing": 0, "excluded_retry": 0}
    assert abs(r["11_50"]["max_pct"] - 1.5) < 1e-9 and abs(r["11_50"]["mean_imputed_pct"] - 100 * 0.075 / 7) < 1e-9


def test_retry_states_are_out_of_the_mean_listed_and_flagged_with_a_composite():
    rows = [_r(f"s{i}", "11_50", 100, 100) for i in range(8)]
    rows += [_r("x", "51_100", 200, 100, n_extra_captures=1), _r("x#r1", "51_100", 120, 100, role="retry", parent="x", n_extra_captures=0),
             _r("y", "11_50", 100, 100, n_extra_captures=1)]                    # y: retry capture absent -> no composite
    rows[0]["chosen_match"] = True
    r = U.read_rows(rows)
    assert r["chosen_check"] == {"match": 1, "mismatch_in_deferral_states": 0, "not_checked_no_sidecar": 9}
    assert r["pooled"]["n"] == 8 and r["pooled"]["mean_regret_pct"] == 0.0              # x and y are not in the mean, the retry row neither
    e = r["excluded_retry_states"]
    assert e["count"] == 2 and e["by_bucket"]["51_100"] == ["x"] and e["by_bucket"]["11_50"] == ["y"]
    assert abs(e["share_of_captured_states"] - 0.2) < 1e-12 and e["flag_gt_10pct"] is True
    assert [(c["state_id"], c["captures"]) for c in e["composite"]] == [("x", 2)] and abs(e["composite"][0]["composite_regret_pct"] - 60.0) < 1e-9
    assert U.read_rows([_r(f"s{i}", "11_50", 100, 100) for i in range(11)] + [_r("z", "11_50", 100, 100, n_extra_captures=1)])["excluded_retry_states"]["flag_gt_10pct"] is False


def test_make_jobs_emits_retry_captures_as_descriptive_jobs(tmp_path):
    cells, inputs = _cells(tmp_path, [(7, 1.0, 12), (7, 1.01, 5)])
    jobs, _ = U.make_jobs([_st("a", 7)], cells, "cd_expand_k100", inputs, "x")
    assert [(j["state_id"], j["role"], j["time"]) for j in jobs] == [("a", "first", 1.0), ("a#r1", "retry", 1.01)] and jobs[1]["parent"] == "a"


def _gate(jobs, commit=None, dry=False, sealed=()):
    U.label_gate(jobs, sealed, commit, dry)


_IDM = {"state_id": "idm_0", "topology": 16490, "window": "g0", "rung": "heavy", "trigger_task_id": 9743}
_CORPUS = {"state_id": "c_0", "topology": 20001, "window": "g0", "rung": "heavy", "trigger_task_id": 5}


def test_labels_refuse_without_a_k1_read_commit():
    with pytest.raises(RuntimeError, match="until K1 is read"):
        _gate([_IDM])                                              # not even identity (m) without --dry-run
    _gate([_IDM], dry=True)                                       # the dry run on identity (m) states is allowed
    with pytest.raises(RuntimeError, match="identity \\(m\\) state"):
        _gate([_IDM, _CORPUS], dry=True)                          # a corpus state is never allowed without the commit
    with pytest.raises(RuntimeError, match="until K1 is read"):
        _gate([_CORPUS])


def test_labels_need_a_real_commit_and_never_touch_a_sealed_state():
    head = __import__("subprocess").run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    _gate([_CORPUS], commit=head)
    with pytest.raises(ValueError, match="not a commit"):
        _gate([_CORPUS], commit="deadbeef" * 5)
    with pytest.raises(ValueError, match="not a commit hash"):
        _gate([_CORPUS], commit="zzz")
    sealed = [{"topology": 20001, "window": "g0", "rung": "heavy", "trigger_task_id": 5}]
    with pytest.raises(RuntimeError, match="sealed K1\\(b\\) state"):
        _gate([_CORPUS], commit=head, sealed=sealed)
    with pytest.raises(ValueError, match="lacks"):
        _gate([{"state_id": "idm_9"}], dry=True)


def test_label_file_has_the_header_and_replays_only_the_hand_and_new_incumbents(tmp_path):
    row = {"state_id": "idm_0", "bucket": "11_50", "n_batch": 3, "n_pinned": 0, "plan_space": 8, "budget": 100, "n_evals": 4, "stopped": "converged", "revisits": 2, "arm": "cd_expand",
           "n_incumbents": 2, "cands_mean": 2.0, "rtt_hand": 10.0, "rtt_incumbents": [7.0], "chosen_match": None,
           "trajectory": [{"eval": 1, "S": 5.0, "iter": 0}, {"eval": 3, "S": 4.0, "iter": 1}],
           "evals_full": [(1, 5.0, 0, [[0, 0], [0, 0], [0, 0]]), (2, 5.5, 1, [[1, 0], [0, 0], [0, 0]]), (3, 4.0, 1, [[1, 0], [1, 0], [0, 0]]), (4, 4.5, 2, [[1, 0], [0, 0], [1, 0]])]}
    h = U.label_header(row, {}, 7, None, True, "k1_sealed_states.json", "ab" * 32, "abc123")
    assert h["header"] and h["exhaustive"] is False and h["plan_space_size"] == 8 and h["eval_budget"] == 100 and h["evals_done"] == 4 and h["seed"] == 7 and h["cap"] == 256
    assert h["s_pin"]["batch_alpha"] == 0.0 and h["replayed_rows"] == 2 and h["top8_replayed"] is False and h["sealed_exclusion"]["sha256"] == "ab" * 32 and h["dry_run"] is True
    rows = U.label_rows(row)
    assert [(r["eval"], r["replayed"], r["rtt"]) for r in rows] == [(1, True, 10.0), (2, False, None), (3, True, 7.0), (4, False, None)]
    assert all(r["S"] is not None for r in rows) and rows[1]["placement_plan"] == {"0": [1, 0], "1": [0, 0], "2": [0, 0]}
    f = U.write_labels(tmp_path, h, rows)
    lines = [json.loads(x) for x in f.read_text().splitlines()]
    assert lines[0]["header"] is True and len(lines) == 5 and all("header" not in x for x in lines[1:])


def test_ils_counts_revisits_and_keeps_distinct_plans():
    cands, S = _problem(6, 3, 0)
    r = U.ils(cands, tuple(c[0] for c in cands), S, seed=0, max_evals=3000)
    assert r["revisits"] > 0 and len({e["plan"] for e in r["evals"]}) == len(r["evals"])


def test_chosen_mismatch_is_counted_not_raised_when_the_capture_is_a_learned_arm():
    hand = [[1, 2], [4, 5], [7, 8]]
    bad = _rec([[3, 1, 2], [5, 4, 6], [9, 7, 9]])
    with pytest.raises(RuntimeError, match="differs from the served plan on 2 of 3"):
        U.check_chosen(_snap(), hand, bad, "s")                                  # default: unchanged, fail loud
    r = U.check_chosen(_snap(), hand, bad, "s", fatal_mismatch=False)
    assert r["match"] is False and r["mismatched_tasks"] == 2 and r["compared"] == 3
    with pytest.raises(ValueError, match="do not partition"):
        U.check_chosen(_snap(), hand, _rec([[3, 1, 2], [5, 4, 5]]), "s", fatal_mismatch=False)       # any other check stays fatal


def test_run_state_passes_the_job_arm_check_flag_and_records_it(monkeypatch):
    seen = {}

    class _Stop(BaseException):
        pass

    def fake(job, arm, seed, max_evals, per, max_seconds, check_arm=True):
        seen.update(check_arm=check_arm, per=per)
        raise _Stop()
    monkeypatch.setattr(U, "decision_pass", fake)
    for flag, want in (({}, True), ({"check_arm": False}, False)):
        with pytest.raises(_Stop):
            U.run_state({"state_id": "s", **flag}, "cd_ib0", 0, None, 50, None, "none")
        assert seen == {"check_arm": want, "per": 50}


def test_read_rows_counts_learned_arm_mismatches_per_capturing_arm():
    def row(i, match, nbad):
        r = _r(f"s{i}", "11_50", 100, 100)
        r.update(check_arm=False, chosen_match=match, chosen_check={"match": match, "mismatched_tasks": nbad, "compared": 10, "live_arm": "gnn"})
        return r
    out = U.read_rows([row(0, False, 3), row(1, True, 0), row(2, False, 1)])
    assert out["chosen_check_check_arm_false"] == {"gnn": {"states": 3, "mismatch_states": 2, "mismatched_tasks": 4, "compared_tasks": 30}}
    assert "chosen_check_check_arm_false" not in U.read_rows([_r("s0", "11_50", 100, 100)])
