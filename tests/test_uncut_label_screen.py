import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import uncut_label_screen as L  # noqa: E402


def frame(counts, topo=16201):
    f, i = {}, 0
    for n, how_many in counts.items():
        for _ in range(how_many):
            f[f"gnn_datasets_wf1_train/ds_{i:05d}"] = {"n_tasks": n, "state_id": f"C1/cc40s{topo}__g1heavy__cd_ib0_s0#{i}", "plan_space_size": 100 * n, "label_source": "exact"}
            i += 1
    return f


def test_draw_takes_125_per_bucket_or_all_and_is_deterministic():
    f = frame({5: 300, 30: 300, 70: 300, 150: 40})
    a, b = L.draw(f), L.draw(f)
    assert a == b and a["state_list_sha256"] == b["state_list_sha256"]
    assert {k: v["drawn"] for k, v in a["counts"].items()} == {"le10": 125, "11_50": 125, "51_100": 125, "101_256": 40}
    assert a["n"] == 415 and len({s["gid"] for s in a["states"]}) == 415


def test_draw_changes_with_seed():
    f = frame({5: 300, 30: 300, 70: 300, 150: 300})
    assert L.draw(f)["state_list_sha256"] != L.draw(f, seed=1)["state_list_sha256"]


def test_forbidden_topology_fails_loud():
    for t in (19537, 19548, 19554, 19589):
        with pytest.raises(SystemExit, match="forbidden"):
            L.draw(frame({5: 10}, topo=t))
    L.draw(frame({5: 10}, topo=19549))      # not in either set


def test_accel_test_id_clash_fails_loud():
    f = frame({5: 10})
    with pytest.raises(SystemExit, match="accel"):
        L.draw(f, accel_test_ids=[next(iter(f))])


def test_bucket_edges_and_out_of_range():
    assert [L.bucket_of(n) for n in (1, 10, 11, 50, 51, 100, 101, 256)] == ["le10", "le10", "11_50", "11_50", "51_100", "51_100", "101_256", "101_256"]
    with pytest.raises(ValueError):
        L.bucket_of(257)


def test_small_flag_follows_plan_space_600():
    f = frame({5: 3})
    k = sorted(f)
    f[k[0]]["plan_space_size"] = 600
    f[k[1]]["plan_space_size"] = 601
    f[k[2]]["plan_space_size"] = None
    out = {s["gid"]: s["small"] for s in L.draw(f)["states"]}
    assert out == {k[0]: True, k[1]: False, k[2]: False}


# ---- plan lists, the decision and the reader -------------------------------------------------------------------------------------------------
def P(*pairs):
    return [list(x) for x in pairs]


def test_gid_order_to_dataset_order_follows_trace_ids():
    trace, gids = [2722, 2724, 2720, 2721, 2723], [2720, 2721, 2722, 2723, 2724]
    ev = P((1, 1), (2, 2), (3, 3), (4, 4), (5, 5))             # gid 2720 -> (1,1) ... gid 2724 -> (5,5)
    assert L.ds_order_from_gid_order(ev, trace, gids) == P((3, 3), (5, 5), (1, 1), (2, 2), (4, 4))


def test_top_k_distinct_ranks_by_s_then_eval_and_drops_repeats():
    rows = [{"S": 3.0, "eval": 1, "plan": P((1, 1))}, {"S": 1.0, "eval": 5, "plan": P((2, 2))}, {"S": 1.0, "eval": 2, "plan": P((3, 3))},
            {"S": 1.0, "eval": 9, "plan": P((2, 2))}, {"S": 2.0, "eval": 4, "plan": P((4, 4))}]
    got = L.top_k_distinct(rows, 3)
    assert [r["plan"] for r in got] == [P((3, 3)), P((2, 2)), P((4, 4))]


def test_plan_list_large_ils_is_label_plus_8_best_distinct():
    label = P((99, 99), (99, 99))            # not among the 8 best by S
    rows = [{"S": float(i), "eval": i, "plan": P((i, i), (i, i))} for i in range(20)]
    out = L.plan_list(label, False, ils_rows=rows)
    assert out[0]["tag"] == "label" and out[0]["plan"] == label
    assert len(out) == 9 and [o["tag"] for o in out[1:]] == ["ils_top"] * 8


def test_plan_list_label_among_the_best_is_not_duplicated():
    label = P((1, 1), (1, 1))
    rows = [{"S": float(i), "eval": i, "plan": P((i, i), (i, i))} for i in range(20)]
    assert len(L.plan_list(label, False, ils_rows=rows)) == 8        # the label is one of the 8 best


def test_plan_list_exact_large_uses_the_enumeration_by_s_then_order():
    label = P((9, 9))
    enum = [{"S": 1.0, "k": 3, "plan": P((3, 3))}, {"S": 1.0, "k": 1, "plan": P((1, 1))}, {"S": 0.5, "k": 7, "plan": P((7, 7))}] + \
           [{"S": 5.0 + i, "k": 10 + i, "plan": P((20 + i, 20 + i))} for i in range(10)]
    out = L.plan_list(label, False, enum=enum)
    assert [o["plan"] for o in out[:4]] == [label, P((7, 7)), P((1, 1)), P((3, 3))] and len(out) == 9


def test_plan_list_small_replays_every_plan_once():
    label = P((1, 1))
    enum = [{"S": float(i), "k": i, "plan": P((i, i))} for i in range(5)]
    out = L.plan_list(label, True, enum=enum)
    assert [o["plan"] for o in out] == [P((1, 1)), P((0, 0)), P((2, 2)), P((3, 3)), P((4, 4))]


def test_decide_tolerance_and_gain():
    assert L.decide([10.0, 10.0, 10.0 * (1 - 1e-12)]) == {"best_index": 0, "differs": False, "gain": 0.0}     # an equal-RTT alternative agrees
    d = L.decide([10.0, 9.0, 9.5])
    assert d["differs"] and d["best_index"] == 1 and d["gain"] == pytest.approx(0.1)
    assert L.decide([5.0, 6.0])["gain"] == 0.0


def _rows(bucket, gains):
    return [{"bucket": bucket, "differs": g > 0, "gain": g} for g in gains]


def test_read_rule_needs_share_mean_and_an_interval_above_zero():
    big = _rows("le10", [0.10] * 30 + [0.0] * 70)         # (i) 30 %, mean 3 %, interval above 0
    r = L.read_rows(big)
    assert r["le10"]["share_differs"] == pytest.approx(0.30) and r["le10"]["meets_rule"] and r["verdict"] == "PROCEED-TO-TRAIN"
    assert r["le10"]["share_gain_ge_5pct"] == pytest.approx(0.30)
    low_share = L.read_rows(_rows("11_50", [0.30] * 15 + [0.0] * 85))         # (i) 15 % < 20 %
    assert not low_share["11_50"]["meets_rule"] and low_share["verdict"] == "LABELS-AGREE"
    low_mean = L.read_rows(_rows("51_100", [0.05] * 30 + [0.0] * 70))         # mean 1.5 % < 2 %
    assert not low_mean["51_100"]["meets_rule"]
    none = L.read_rows(_rows("101_256", [0.0] * 50))
    assert not none["101_256"]["meets_rule"] and none["101_256"]["mean_gain_ci95"] == [0.0, 0.0]


def test_bootstrap_is_deterministic():
    v = [0.0, 0.1, 0.0, 0.3, 0.05, 0.0]
    assert L.bootstrap_mean_ci(v) == L.bootstrap_mean_ci(v)


# ---- the relabel sidecar ---------------------------------------------------------------------------------------------------------------------
def _r(gid="g/ds_1", differs=True):
    return {"gid": gid, "bucket": "le10", "n_tasks": 2, "label_source": "exact", "method": "exhaustive", "label_plan": P((1, 2), (3, 4)), "rtt_sim": 8.0,
            "rtt_label": 10.0, "differs": differs, "gain": 0.2 if differs else 0.0, "n_plans": 12, "current_plan": P((5, 6), (7, 8)), "best_tag": "enum"}


def test_sidecar_entry_carries_the_override_keys_and_both_rtts():
    e = L.sidecar_entry(_r(), "val")
    assert e["plans"] == [P((1, 2), (3, 4))] and e["cd_rtt"] == 8.0           # what NEAR_RTT_LABEL_OVERRIDE_JSON reads
    assert e["split"] == "val" and e["rtt_current"] == 10.0 and e["rtt_sim"] == 8.0 and e["current_label_source"] == "exact" and e["method"] == "exhaustive"


def test_merge_requires_exact_coverage(tmp_path, monkeypatch):
    states = [{"gid": "a/ds_1", "split": "train"}, {"gid": "a/ds_2", "split": "val"}]
    monkeypatch.setattr(L, "all_states", lambda split, corpus: states)
    shard = tmp_path / "s0.jsonl"
    shard.write_text(json.dumps({"gid": "a/ds_1", **L.sidecar_entry(_r("a/ds_1"), "train"), "wall_s": 1.0}) + "\n")
    import argparse
    args = argparse.Namespace(split="x", corpus="y", inp=[str(shard)], out=str(tmp_path / "side.json"))
    with pytest.raises(SystemExit, match="missing"):
        L.cmd_merge(args)
    shard.write_text(shard.read_text() + json.dumps({"gid": "a/ds_2", **L.sidecar_entry(_r("a/ds_2"), "val"), "wall_s": 1.0}) + "\n")
    L.cmd_merge(args)
    side = json.load(open(args.out))
    assert side["_meta"]["n"] == 2 and side["a/ds_2"]["split"] == "val" and "gid" not in side["a/ds_1"] and "wall_s" not in side["a/ds_1"]


def test_merge_rejects_a_state_in_two_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "all_states", lambda split, corpus: [{"gid": "a/ds_1", "split": "train"}])
    rows = json.dumps({"gid": "a/ds_1", **L.sidecar_entry(_r("a/ds_1"), "train"), "wall_s": 1.0}) + "\n"
    f1, f2 = tmp_path / "1.jsonl", tmp_path / "2.jsonl"
    f1.write_text(rows)
    f2.write_text(rows)
    import argparse
    with pytest.raises(SystemExit, match="two shards"):
        L.cmd_merge(argparse.Namespace(split="x", corpus="y", inp=[str(f1), str(f2)], out=str(tmp_path / "o.json")))
