import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
import grounded_workload_v1_mint as M  # noqa: E402
import grounded_workload_v1_read as R  # noqa: E402

TOPOS = list(range(9001, 9013))


def _write(d, topo, window, kind, seed, elapsed, queue=1.0):
    name = f"cc40s{topo}__{window}__{kind}_s{seed}"
    with open(os.path.join(d, name + ".summary.json"), "w") as fh:
        json.dump({"arm": name, "topology": topo, "window": window, "checkpoint_seed": seed,
                   "averageElapsedTime": elapsed, "averageQueueTime": queue}, fh)


def _gate(d, learned_factor, inadmissible=()):
    for t in TOPOS:
        for w in R.WINDOWS:
            q = 9.0 if (t, w) in inadmissible else 1.0
            _write(d, t, w, "reactive", 0, 10.0, q)
            _write(d, t, w, "cd", 0, 10.0)
            for s in R.SEEDS:
                _write(d, t, w, R.PRIMARY, s, 10.0 * learned_factor)


def test_confirmed_when_learned_is_faster_everywhere(tmp_path):
    _gate(str(tmp_path), 0.9)
    c = R.contrast(R.load(str(tmp_path)), R.admissible(R.load(str(tmp_path)), TOPOS), TOPOS, R.PRIMARY, "cd")
    assert c["label"] == "CONFIRMED" and c["median_pct"] == pytest.approx(-10.0)


def test_inadmissible_cells_are_excluded(tmp_path):
    bad = {(t, "g0") for t in TOPOS}
    _gate(str(tmp_path), 1.0, bad)
    for t in TOPOS:
        for s in R.SEEDS:
            _write(str(tmp_path), t, "g0", R.PRIMARY, s, 1.0)  # a huge "win" on an inadmissible cell
    s = R.load(str(tmp_path))
    c = R.contrast(s, R.admissible(s, TOPOS), TOPOS, R.PRIMARY, "cd")
    assert c["label"] == "NOT-SEPARATED" and c["median_pct"] == 0.0


def test_missing_reactive_is_not_admissible(tmp_path):
    _gate(str(tmp_path), 0.9)
    for t in TOPOS[:5]:
        for w in R.WINDOWS:
            os.remove(os.path.join(str(tmp_path), f"cc40s{t}__{w}__reactive_s0.summary.json"))
    s = R.load(str(tmp_path))
    c = R.contrast(s, R.admissible(s, TOPOS), TOPOS, R.PRIMARY, "cd")
    assert c["label"] == "DESIGN-SHORT" and c["n_topologies"] == 7


def test_mint_keeps_rate_order_and_group_peers():
    lib = {"groups": [{"t0_ms": 1000 * i, "offsets_ms": [0, 1, 5][: 1 + i % 3]} for i in range(60)]}
    base = {"rps": 1, "duration": 1, "events": [
        {"timestamp": 2.0 * i, "application": {"name": "nofs-dnn1", "dag": {"dnn1": []}}, "qos": {"name": "m"},
         "node_name": "c0"} for i in range(100)]}
    doc, meta = M.mint(lib, base, seed=1, n_tasks=100)
    ts = [e["timestamp"] for e in doc["events"]]
    assert len(ts) == 100 and ts == sorted(ts)
    assert meta["rate_per_s"] == pytest.approx(100 / (ts[-1] - ts[0]), rel=1e-9)
    for i, j, _b in doc["peer_exchange"]:
        assert doc["events"][i]["peer_group"] == doc["events"][j]["peer_group"]


def test_group_size_makes_exact_groups_at_the_same_task_rate():
    lib = {"groups": [{"t0_ms": 1000 * i, "offsets_ms": [0, 1, 5][: 1 + i % 3]} for i in range(200)]}
    base = {"rps": 1, "duration": 1, "events": [
        {"timestamp": 2.0 * i, "application": {"name": "nofs-dnn1", "dag": {"dnn1": []}}, "qos": {"name": "m"},
         "node_name": "c0"} for i in range(300)]}
    ref, ref_meta = M.mint(lib, base, seed=1, n_tasks=300)
    for g in (8, 12, 16):
        doc, meta = M.mint(lib, base, seed=1, n_tasks=300, group_size=g)
        sizes = {}
        for e in doc["events"]:
            sizes[e["peer_group"]] = sizes.get(e["peer_group"], 0) + 1
        assert sorted(sizes.values(), reverse=True)[0] == g and sum(sizes.values()) == 300
        assert sorted(sizes.values()).count(g) == 300 // g
        assert meta["rate_per_s"] == pytest.approx(ref_meta["rate_per_s"], rel=0.02)
        assert [e["application"]["name"] for e in doc["events"]] == [e["application"]["name"] for e in ref["events"]]
        for i, j, _b in doc["peer_exchange"]:
            assert doc["events"][i]["peer_group"] == doc["events"][j]["peer_group"]
    with pytest.raises(SystemExit):
        M.mint(lib, base, seed=1, n_tasks=300, merge_k=2, group_size=8)


def _lib_base(n_groups=200, n_events=300):
    lib = {"source": {"md5": "abc"}, "groups": [{"t0_ms": 1000 * i, "offsets_ms": [0, 1, 5][: 1 + i % 3]} for i in range(n_groups)]}
    base = {"rps": 1, "duration": 1, "events": [
        {"timestamp": 2.0 * i, "application": {"name": "nofs-dnn1", "dag": {"dnn1": []}}, "qos": {"name": "m"},
         "node_name": "c0"} for i in range(n_events)]}
    return lib, base


def test_start_group_zero_is_todays_mint_byte_for_byte():
    """uncut_groups_s0_v1 identity (a): --start-group 0 adds nothing to the workload or its metadata."""
    lib, base = _lib_base()
    ref = M.mint(lib, base, seed=3, n_tasks=200)
    got = M.mint(lib, base, seed=3, n_tasks=200, start_group=0)
    assert json.dumps(got, sort_keys=True) == json.dumps(ref, sort_keys=True)
    assert "start_group" not in got[1]


def test_start_group_mints_the_later_requests():
    lib, base = _lib_base()
    doc, meta = M.mint(lib, base, seed=3, n_tasks=100, start_group=60)
    assert meta["start_group"] == 60
    want = sum(len(g["offsets_ms"]) for g in lib["groups"][60:60 + meta["n_groups"]])
    assert want >= 100 and len(doc["events"]) == 100
    for bad in (-1, 200):
        with pytest.raises(SystemExit):
            M.mint(lib, base, seed=3, n_tasks=100, start_group=bad)
    with pytest.raises(SystemExit):
        M.mint(lib, base, seed=3, n_tasks=100, start_group=5, group_size=8)


def test_corpus_mint_must_not_overlap_the_gate_windows(tmp_path):
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"n_groups": 50, "library_source": {"md5": "abc"}}))
    M.check_disjoint(50, 40, "abc", [str(gate)])  # groups 50..89 after the gate's 0..49
    with pytest.raises(SystemExit, match="overlap"):
        M.check_disjoint(49, 40, "abc", [str(gate)])
    with pytest.raises(SystemExit, match="source CSV"):
        M.check_disjoint(100, 40, "other", [str(gate)])
    later = tmp_path / "later.json"
    later.write_text(json.dumps({"grounded_workload_v1": {"n_groups": 10, "start_group": 95, "library_source": {"md5": "abc"}}}))
    with pytest.raises(SystemExit, match="overlap"):
        M.check_disjoint(50, 50, "abc", [str(later)])


def _golden(group_size=0, sampler="legacy"):
    """Digest of the mint at the registered rule, taken from origin/reference-physics (0f3d0de1) before the nested-partner knob existed."""
    import hashlib
    lib, base = _lib_base()
    doc, meta = M.mint(lib, base, seed=3, n_tasks=200, group_size=group_size, payload_sampler=sampler)
    return hashlib.sha256(json.dumps([doc, meta], sort_keys=True).encode()).hexdigest()


GOLDEN_DEFAULT = {("legacy", 0): "5304e8c2a6a76b61b634f91f055cc025f3fef953fbe0a154f6c3a24858ea3465",
                  ("legacy", 16): "a4822a1af5c5e0f1f4643346b490672e1663e2dbba7011f9af7a1cf9cb6f4809",
                  ("wf1_v1", 16): "325f10f2582999658eba6e22ff8bb5f049ab1b3305c3813a847dde50a5d8509c"}


@pytest.mark.parametrize("sampler,gs", sorted(GOLDEN_DEFAULT))
def test_default_mint_is_the_old_mint_byte_for_byte(sampler, gs):
    assert _golden(gs, sampler) == GOLDEN_DEFAULT[(sampler, gs)]
    lib, base = _lib_base()
    _doc, meta = M.mint(lib, base, seed=3, n_tasks=200, group_size=gs, payload_sampler=sampler, partners_nested=0)
    assert "partner_nested" not in meta and meta["peer_rule"]["partners"] == 2


def _levels(n_events=400, n_tasks=320):
    lib, base = _lib_base(200, n_events)
    return {k: M.mint(lib, base, seed=3, n_tasks=n_tasks, group_size=16, payload_sampler="wf1_v1", partners_nested=k) for k in (1, 2, 4, 8)}


def test_nested_levels_share_every_non_pair_draw_and_nest_their_pairs():
    L = _levels()
    shas = {m["partner_nested"]["nonpair_sha256"] for _d, m in L.values()}
    assert len(shas) == 1
    assert len({json.dumps({k: v for k, v in d.items() if k not in ("peer_exchange",)}, sort_keys=True) for d, _m in L.values()}) == 1
    plan = [(0, [0] * 16)] * 20
    new_idx = {i: i for i in range(320)}
    events = [{"peer_group": i // 16} for i in range(320)]
    pk = {k: set(M._nested_pairs(plan, new_idx, events, 3, k, False)["pairs"]) for k in (1, 2, 4, 8)}
    assert pk[1] <= pk[2] <= pk[4] <= pk[8] and len(pk[1]) < len(pk[2]) < len(pk[4]) < len(pk[8])
    assert all(L[k][1]["partner_nested"]["connector_pairs"] >= 0 for k in L)
    for k, (d, m) in L.items():
        assert m["peer_rule"]["partners"] == k and m["peer_rule"]["nested"] is True
        for i, j, _b in d["peer_exchange"]:
            assert d["events"][i]["peer_group"] == d["events"][j]["peer_group"]
        assert m["partner_nested"]["realised_mean_degree"] == pytest.approx(2.0 * len(d["peer_exchange"]) / 320)


def test_nested_bytes_are_held_per_group_at_the_level_two_total():
    L = _levels()
    def per_group(d):
        out = {}
        for i, j, b in d["peer_exchange"]:
            g = d["events"][i]["peer_group"]; out[g] = out.get(g, 0.0) + b
        return out
    ref = per_group(L[2][0])
    for k in (1, 4, 8):
        got = per_group(L[k][0])
        assert set(got) == set(ref)
        assert all(got[g] == pytest.approx(ref[g], rel=1e-9) for g in ref)
    assert L[4][1]["partner_nested"]["total_bytes"] == pytest.approx(L[2][1]["partner_nested"]["total_bytes"], rel=1e-9)


def test_nested_needs_wf1_v1_and_a_valid_level():
    lib, base = _lib_base()
    for kw in ({"partners_nested": 3, "payload_sampler": "legacy"}, {"partners_nested": 9, "payload_sampler": "wf1_v1"},
               {"partners_nested": 2, "payload_sampler": "wf1_v1", "merge_k": 2}):
        with pytest.raises(SystemExit):
            M.mint(lib, base, seed=3, n_tasks=200, group_size=16, **kw)
