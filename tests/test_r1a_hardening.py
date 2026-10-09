"""r1_attribution_v1 corpus hardening: sweep completeness, spread order, stall watchdog, tied-plan order."""
import concurrent.futures
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.placement.sweep_status import sweep_complete, sweep_status  # noqa: E402


def _dataset(tmp_path, *, plans=3, rows=3, complete=True, best=True):
    d = tmp_path / "ds_00000"
    (d / "placements").mkdir(parents=True)
    (d / "placement_metadata.json").write_text(json.dumps({"num_placements": plans, "sweep_complete": complete}))
    (d / "placements" / "placements.jsonl").write_text("".join(json.dumps({"placement_plan": {}, "rtt": 1.0}) + "\n" for _ in range(rows)))
    (d / "best.json").write_text('{"rtt": 1.0}' if best else "")
    return d


def test_complete_sweep(tmp_path):
    assert sweep_status(_dataset(tmp_path)) == (True, "complete")


@pytest.mark.parametrize("kw", [{"rows": 2}, {"complete": False}, {"best": False}])
def test_truncated_sweep_is_not_complete(tmp_path, kw):
    assert not sweep_complete(_dataset(tmp_path, **kw))


def test_missing_metadata_is_not_complete(tmp_path):
    d = _dataset(tmp_path)
    (d / "placement_metadata.json").unlink()
    assert not sweep_complete(d)


def test_spread_order_is_a_permutation_with_spread_prefixes():
    from scripts_cosim.wf1_shard_snapshots import spread_order

    for n in (1, 2, 7, 300, 513):
        assert sorted(spread_order(n)) == list(range(n))
    head = spread_order(300)[:6]
    assert max(head) - min(head) > 200


def test_stall_watchdog_stops_when_nothing_finishes():
    from src.executecosimulation import _completed_or_stalled

    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        futs = {ex.submit(lambda: 1): 0, ex.submit(lambda: 2): 1, ex.submit(time.sleep, 3): 2, ex.submit(time.sleep, 3): 3}
        stall = {"pending": 0}
        got = list(_completed_or_stalled(futs, 0.5, stall))
    assert len(got) == 2 and stall["pending"] == 2


def test_no_stall_without_timeout():
    from src.executecosimulation import _completed_or_stalled

    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        futs = {ex.submit(lambda i=i: i): i for i in range(5)}
        stall = {"pending": 0}
        assert len(list(_completed_or_stalled(futs, None, stall))) == 5 and stall["pending"] == 0


def test_plan_combo_orders_by_task_index_then_node_platform():
    from src.executecosimulation import _plan_combo

    assert _plan_combo({"10": [1, 0], "2": [0, 3]}) == ((0, 3), (1, 0))
    assert _plan_combo({"0": [1, 0]}) > _plan_combo({"0": [0, 9]})


def test_planned_then_deferred_partner_is_unplaced_in_the_snapshot():
    from types import SimpleNamespace as NS

    from src.placement.snapshot_fidelity import peer_node_name

    node = NS(node_name="n3")
    assert peer_node_name(None) is None
    assert peer_node_name(NS(platform=NS(node=node), planned_node_name="n9")) == "n3"
    assert peer_node_name(NS(platform=None, planned_node_name="n9", postponed_count=0)) == "n9"
    assert peer_node_name(NS(platform=None, planned_node_name="n9")) == "n9"
    assert peer_node_name(NS(platform=None, planned_node_name="n9", postponed_count=2)) is None
    assert peer_node_name(NS(platform=None, planned_node_name=None)) is None


def test_exposure_flags_a_net_stage_ingress_ghost_on_a_candidate_node():
    from src.placement.fidelity_replay import exposure

    ghost = lambda q, stage, ls: {"q": q, "stage": stage, "link_stage": ls}
    snap = {"fidelity": {"ghosts": [ghost("n1:0", "ingress", "net"), ghost("n2:1", "ingress", "hold"), ghost("n3:0", "compute", None)]}}
    cands = [{"node_name": "n1"}, {"node_name": "n4"}]
    e = exposure(snap, cands)
    assert e["net_ingress_ghost_on_candidate_node"] and e["net_ingress_ghost_nodes_on_candidates"] == ["n1"]
    assert e["net_ingress_ghosts_total"] == 1 and e["candidate_nodes"] == 2
    assert not exposure(snap, [{"node_name": "n2"}, {"node_name": "n3"}])["net_ingress_ghost_on_candidate_node"]
    assert not exposure({"fidelity": {}}, cands)["net_ingress_ghost_on_candidate_node"]


def test_manifest_read_skips_and_counts_another_tasks_half_written_line(tmp_path):
    import json

    import pytest

    from scripts_cosim.make_warm_corpus import _own_manifest_entries

    p = tmp_path / "warm_manifest.jsonl"
    p.write_text(json.dumps({"source_tag": "a", "status": "success"}) + "\n" + json.dumps({"source_tag": "b"}) + "\n" + '{"source_tag": "b", "sta')
    entries, skipped = _own_manifest_entries(p, "a")
    assert [e["status"] for e in entries] == ["success"] and skipped == 1
    p.write_text(json.dumps({"source_tag": "a"}) + "\n" + '{"source_tag": "a", "sta')
    with pytest.raises(json.JSONDecodeError):
        _own_manifest_entries(p, "a")
    q = tmp_path / "warm_manifest.a.jsonl"  # per-source: one writer, a bad line is always ours
    q.write_text(json.dumps({"source_tag": "a"}) + "\n" + 'garbled')
    with pytest.raises(json.JSONDecodeError):
        _own_manifest_entries(q, "a")


def test_manifest_union_reads_every_per_source_file_and_fails_on_a_garbled_line(tmp_path):
    import json

    import pytest

    from scripts_cosim.wf1_manifest import read_manifests

    (tmp_path / "warm_manifest.t1_light_g0.jsonl").write_text(json.dumps({"source_tag": "t1_light_g0", "dataset_id": "ds_0"}) + "\n")
    (tmp_path / "warm_manifest.t2_heavy_g1.jsonl").write_text(json.dumps({"source_tag": "t2_heavy_g1", "dataset_id": "ds_1"}) + "\n")
    entries, skipped = read_manifests(tmp_path)
    assert sorted(e["dataset_id"] for e in entries) == ["ds_0", "ds_1"] and skipped == 0
    (tmp_path / "warm_manifest.t2_heavy_g1.jsonl").write_text('{"source_tag": "t2_hea')
    with pytest.raises(ValueError, match="garbled"):
        read_manifests(tmp_path)
    entries, skipped = read_manifests(tmp_path, strict=False)
    assert len(entries) == 1 and skipped == 1
