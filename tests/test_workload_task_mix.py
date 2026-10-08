"""workload_fix_v1 W4: the task-type relabel changes labels only, and the default path is untouched."""
import collections
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_cosim"))
import workload_fix_v1_build as B  # noqa: E402

from src.placement import workload_task_mix as M  # noqa: E402

TYPES = ["dnn1", "dnn2", "rf", "cnn"]


def _window(i, n=400):
    events = []
    for k in range(n):
        t = "dnn1" if k % 3 else "dnn2"
        events.append({"timestamp": 10.0 * k + 0.001 * (k % 4), "peer_group": k // 4, "node_name": f"client_node{(k * 3) % 7}",
                       "qos": {"name": "medium", "maxDurationDeviation": 15},
                       "application": {"name": f"nofs-{t}", "dag": {t: []}, "demand_scale": {t: 1.0 + 0.01 * k}}})
    pairs = [[a, a + 1, 1e8 + a] for a in range(0, n - 1, 2)]
    return {"rps": 1, "duration": 1, "events": events, "peer_exchange": pairs, "grounded_workload_v1": {"seed": 7400 + i}}


@pytest.fixture()
def x1(tmp_path):
    d = tmp_path / "x1"
    d.mkdir()
    for i, name in enumerate(B.WINDOWS):
        (d / name).write_text(json.dumps(_window(i)))
    return d


def _load(d):
    return {n: json.loads((d / n).read_text()) for n in B.WINDOWS}


def test_defined_types_are_the_four_in_task_types_json():
    assert list(M.defined_application_types()) == TYPES


def test_relabel_changes_only_type_labels(tmp_path, x1):
    w2 = _load_after(tmp_path / "w2", x1, "wf1_v1", "none")
    w4 = _load_after(tmp_path / "w4", x1, "wf1_v1", "wf1_v1")
    for name in B.WINDOWS:
        a, b = w2[name], w4[name]
        assert a["peer_exchange"] == b["peer_exchange"]  # pairs and W2 payloads, byte-identical values
        assert json.dumps(a["peer_exchange"]) == json.dumps(b["peer_exchange"])
        for key in set(a) - {"events", "workload_fix_v1"}:
            assert a[key] == b[key]
        for ea, eb in zip(a["events"], b["events"]):
            assert {k: v for k, v in ea.items() if k != "application"} == {k: v for k, v in eb.items() if k != "application"}
            (ta,), (tb,) = ea["application"]["dag"], eb["application"]["dag"]
            assert eb["application"]["name"] == f"nofs-{tb}"
            assert list(eb["application"]["demand_scale"]) == [tb]
            assert eb["application"]["demand_scale"][tb] == ea["application"]["demand_scale"][ta]
        assert len(a["events"]) == len(b["events"])
        assert b["workload_fix_v1"]["task_mix"]["types"] == TYPES
        assert {k: v for k, v in b["workload_fix_v1"].items() if k != "task_mix"} == a["workload_fix_v1"]


def test_mix_is_equal_and_seeded(tmp_path, x1):
    w4 = _load_after(tmp_path / "w4", x1, "wf1_v1", "wf1_v1")
    again = _load_after(tmp_path / "w4b", x1, "wf1_v1", "wf1_v1")
    for name in B.WINDOWS:
        counts = collections.Counter(next(iter(e["application"]["dag"])) for e in w4[name]["events"])
        assert set(counts) == set(TYPES) and set(counts.values()) == {100}
        assert w4[name]["events"] == again[name]["events"]
    labels = [[next(iter(e["application"]["dag"])) for e in w4[n]["events"]] for n in B.WINDOWS]
    assert len({tuple(x) for x in labels}) == len(B.WINDOWS)  # per-window seeds give different draws


def test_uneven_count_stays_within_one():
    rng = M.task_mix_rng(1)
    counts = collections.Counter(M.balanced_labels(10, TYPES, rng))
    assert sorted(counts.values()) == [2, 2, 3, 3]


def test_relabel_survives_the_rest_of_the_builder(tmp_path, x1):
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "cc40s9601.json").write_text(json.dumps({"scheduler": {"batch_timeout": 16.0}}))
    plain, mixed = tmp_path / "plain", tmp_path / "mixed"
    for out, mix in ((plain, "none"), (mixed, "wf1_v1")):
        B.apply_sampler(x1, out / "_x1_windows", "wf1_v1", mix)
        B.build_rung(out / "_x1_windows", cfg, out, "a", 0.5, [9601])
    for name in B.WINDOWS:
        a = json.loads((plain / "wf1_a" / "wl" / name).read_text())
        b = json.loads((mixed / "wf1_a" / "wl" / name).read_text())
        assert [e["timestamp"] for e in a["events"]] == [e["timestamp"] for e in b["events"]]
        assert [(e["node_name"], e["peer_group"]) for e in a["events"]] == [(e["node_name"], e["peer_group"]) for e in b["events"]]
        assert a["peer_exchange"] == b["peer_exchange"]
        assert {next(iter(e["application"]["dag"])) for e in b["events"]} == set(TYPES)


def test_default_path_is_unchanged(tmp_path, x1):
    # legacy sampler, no mix: a byte copy of the source
    B.apply_sampler(x1, tmp_path / "legacy", "legacy")
    for name in B.WINDOWS:
        assert (tmp_path / "legacy" / name).read_bytes() == (x1 / name).read_bytes()
    # wf1_v1 sampler, no mix: the pre-W4 output (payload resample + payload meta only), byte for byte
    B.apply_sampler(x1, tmp_path / "w2", "wf1_v1")
    for name in B.WINDOWS:
        src = json.loads((x1 / name).read_text())
        seed = int(src["grounded_workload_v1"]["seed"])
        src["peer_exchange"] = B.resample_peer_exchange(src["peer_exchange"], seed)
        src["workload_fix_v1"] = {**B.payload_sampler_meta(), "seed": seed,
                                  "source_sha256": hashlib.sha256((x1 / name).read_bytes()).hexdigest()}
        assert (tmp_path / "w2" / name).read_text() == json.dumps(src)
        assert "task_mix" not in src["workload_fix_v1"]
        assert {next(iter(e["application"]["dag"])) for e in json.loads((tmp_path / "w2" / name).read_text())["events"]} == {"dnn1", "dnn2"}


def test_unknown_mix_and_non_single_node_events_fail_loud(tmp_path, x1):
    with pytest.raises(ValueError, match="task mix"):
        B.apply_sampler(x1, tmp_path / "o", "wf1_v1", "bogus")
    ev = _window(0, 4)["events"]
    ev[0]["application"]["dag"] = {"dnn1": ["dnn2"], "dnn2": []}
    with pytest.raises(ValueError, match="single-node"):
        M.relabel_events(ev, 1, M.defined_application_types())


def _load_after(dst, x1, sampler, mix):
    B.apply_sampler(x1, dst, sampler, mix)
    return _load(dst)


def test_w4_config_differs_from_w3_only_by_the_repair_block():
    import workload_fix_v1_stage_topologies as T

    cfg = json.loads((ROOT / "tests" / "fixtures" / "workload_fix_v1" / "cc40s9473.json").read_text())
    c3 = T.w3_config(cfg)
    c4 = T.w4_config(c3)
    assert c3["network"]["backbone"]["access_classes"]["mix"] == {"wired": 0.4, "wifi": 0.4, "cellular": 0.2}
    assert {k: v for k, v in c4["network"].items() if k != "reachability_repair"} == c3["network"]
    assert {k: v for k, v in c4.items() if k != "network"} == {k: v for k, v in c3.items() if k != "network"}
    assert c4["network"]["reachability_repair"] == {"task_types": "all"}
    with pytest.raises(SystemExit, match="already carries"):
        T.w3_config(c3)
    d = T.diff_infra(T.generate(c3, ROOT / "data" / "nofs-ids"), T.generate(c4, ROOT / "data" / "nofs-ids"))
    assert d["access_classes_identical"] and not d["replica_placements_differ"]
    assert d["added_edges"] == [] and d["keys_differing"] == []  # the repair has nothing to add on this config
