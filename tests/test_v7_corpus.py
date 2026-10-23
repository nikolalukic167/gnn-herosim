"""live_link_features_v1: the by-topology split and the combined view of the recaptured corpus (scripts_cosim/v7_corpus.py)."""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))

import v7_corpus as C  # noqa: E402


def manifest(tmp_path, arm="flowcap_a", topos=range(16201, 16241), per=5, name="build_0.jsonl"):
    rows = []
    k = 0
    for t in topos:
        for _ in range(per):
            rows.append({"status": "built", "dataset": f"ds_{k:05d}", "state_id": f"{arm}:{t}:g0:heavy:{k}", "topology": t})
            k += 1
    rows.append({"status": "dropped", "dataset": "ds_99999", "state_id": f"{arm}:16201:g0:heavy:9", "topology": 16201})
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_the_split_is_by_topology_deterministic_and_covers_every_built_dataset(tmp_path):
    p = manifest(tmp_path)
    rows = C.dataset_rows([str(p)])
    assert len(rows) == 200 and all(r["id"].startswith("gnn_datasets_wf1_flowcap_a/ds_") for r in rows)
    s = C.topology_split(rows, 4, 1, 42)
    s2 = C.topology_split(rows, 4, 1, 42)
    assert s == s2
    assert s["counts"] == {"train": 175, "val": 20, "test": 5} and s["n_parents"] == 200 and s["split_by"] == "topology" and s["test_placeholder"] is True
    t = s["topologies"]
    assert len(t["val"]) == 4 and len(t["test"]) == 1 and len(t["train"]) == 35
    assert not (set(t["train"]) & set(t["val"])) and not (set(t["train"]) & set(t["test"])) and not (set(t["val"]) & set(t["test"]))
    assert set(s["train"]) | set(s["val"]) | set(s["test"]) == {r["id"] for r in rows}
    by_topo = {r["id"]: r["topology"] for r in rows}
    for part in ("train", "val", "test"):
        assert {by_topo[i] for i in s[part]} == set(t[part])
    assert C.topology_split(rows, 4, 1, 7)["topologies"] != t  # the seed is the draw


def test_the_split_fails_loud(tmp_path):
    p = manifest(tmp_path)
    rows = C.dataset_rows([str(p)])
    with pytest.raises(SystemExit, match="cannot give"):
        C.topology_split(rows, 20, 20, 42)
    accel = [rows[0]["id"]]
    with pytest.raises(SystemExit, match="accel_replica_v1 test names"):
        C.topology_split(rows, 4, 1, 42, accel)
    dup = manifest(tmp_path, name="build_1.jsonl")
    with pytest.raises(SystemExit, match="built twice"):
        C.dataset_rows([str(p), str(dup)])
    with pytest.raises(SystemExit, match="no manifest"):
        C.dataset_rows([str(tmp_path / "none*.jsonl")])
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"status": "built", "dataset": "ds_0", "state_id": "a:1:g0:heavy:0", "topology": None}) + "\n")
    with pytest.raises(SystemExit, match="no topology"):
        C.dataset_rows([str(bad)])


def test_write_split_hash_is_the_file_hash_and_main_runs(tmp_path, capsys):
    import hashlib

    p = manifest(tmp_path)
    acc = tmp_path / "accel.json"
    acc.write_text(json.dumps({"test": ["gnn_datasets_wf1_train/ds_40000"]}))
    out = tmp_path / "split.json"
    assert C.main(["split", "--manifests", str(p), "--out", str(out), "--accel-split", str(acc)]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest() and info["n_parents"] == 200


def test_view_symlinks_each_arm_and_refuses_an_existing_dir(tmp_path):
    corpus = tmp_path / "corpus"
    (corpus / "flowcap_a" / "ds_00000").mkdir(parents=True)
    names = C.make_view(corpus, ["flowcap_a"], tmp_path / "view")
    assert names == ["gnn_datasets_wf1_flowcap_a"] and (tmp_path / "view" / names[0] / "ds_00000").is_dir()
    with pytest.raises(SystemExit, match="exists"):
        C.make_view(corpus, ["flowcap_a"], tmp_path / "view")
    with pytest.raises(SystemExit, match="not a directory"):
        C.make_view(corpus, ["nope"], tmp_path / "view2")


def test_the_live_vs_cache_check_compares_the_v7_context_tables():
    import peer_affinity_live_serve_check as P

    assert "link_counts" in P.PSC_FIELDS and "link_routes" in P.PSC_FIELDS
    a = {(0, 1): (("s0>core_0", 4.0), ("core_0>s1", 10.0))}
    b = {(0, 1): [["s0>core_0", 4.0], ["core_0>s1", 10.0]]}
    assert P._close(P._norm(a), P._norm(b))
    c = {(0, 1): (("s0>core_0", 4.0), ("core_0>s1", 12.0))}
    assert not P._close(P._norm(a), P._norm(c))
    assert P._close(P._norm({"s0>core_0": 2}), P._norm({"s0>core_0": 2})) and not P._close(P._norm({"s0>core_0": 2}), P._norm({"s0>core_0": 3}))
    assert P._close(P._norm(None), P._norm(None))


def test_the_fidelity_parity_loads_the_checkpoint_under_the_cells_own_config():
    src = open(os.path.join(os.path.dirname(__file__), "..", "scripts_cosim", "wf1_fidelity_parity.py")).read()
    assert 'load_gnn_model(Path(job["checkpoint"]), space_config=space)' in src and "space_config=None" not in src
