"""Tests for the peer_only_v1 gate glue."""
from scripts_cosim.peer_only_v1_gate_read import (
    read_a0_from, read_b1, read_b3, read_phase, tables,
)
from scripts_cosim.peer_only_v1_read import (
    CHECKPOINT_SEEDS, V_A0_FAIL, V_A0_PASS, V_BEATS, V_CORPUS_HELPS, V_GIN_OVERREACTION,
    V_PEER_KEPT, V_POINTWISE, V_TIE, V_UNREADABLE,
)

CELLS = {"R0": ["cs6s9001", "cs6s9002", "cs6s9003", "cs6s9005"],
         "R3": ["cs80s9001", "cs80s9002", "cs80s9003", "cs80s9005"]}


def _doc(rung, cell, kind, seed, elapsed, queue, corpus=None, peer=10.0):
    d = {"arm": f"{cell}__{rung}__{(corpus + '_') if corpus else ''}{kind}_s{seed}", "cell": cell, "rung": rung,
         "arm_kind": kind, "checkpoint_seed": seed, "averageElapsedTime": elapsed, "averageQueueTime": queue,
         "num_tasks": 50000, "totalPeerExchangeTime": peer * 50000 * 0.7, "totalPeerRendezvousWait": peer * 50000 * 0.3}
    if corpus:
        d["corpus"] = corpus
    return d


def _p3():
    docs = []
    for rung, cells in CELLS.items():
        for i, c in enumerate(cells):
            react = 20.0 + i
            docs.append(_doc(rung, c, "reactive", 0, react, react * 0.6))
            for s in CHECKPOINT_SEEDS:
                docs.append(_doc(rung, c, "gnn", s, react * 1.6 + s * 0.01, react * 1.4))      # gnn: high queue
                docs.append(_doc(rung, c, "mpoff", s, react * 1.3 + s * 0.01, react * 1.0))
    return docs


def _po(po_scale=1.0, po_queue_scale=1.0, po_peer=10.0, a0_drift=0.0, corpus="516"):
    docs = []
    for rung, cells in CELLS.items():
        # A0 re-serves: v3 gnn/mpoff seed 1 on the first cell
        c0 = cells[0]; react = 20.0
        docs.append(_doc(rung, c0, "gnn", 1, react * 1.6 + 0.01 + a0_drift, react * 1.4, corpus="v3"))
        docs.append(_doc(rung, c0, "mpoff", 1, react * 1.3 + 0.01 + a0_drift, react * 1.0, corpus="v3"))
        for i, c in enumerate(cells):
            react = 20.0 + i
            for s in CHECKPOINT_SEEDS:
                docs.append(_doc(rung, c, "peeronly", s, (react * 1.3 + s * 0.01) * po_scale,
                                 react * 1.0 * po_queue_scale, corpus=corpus, peer=po_peer))
    return docs


def test_tables_key_arms_by_corpus_and_pair_by_cell_seed():
    tab = tables(_po(), _p3())
    assert set(tab) == {"R0", "R3"}
    e = tab["R0"]["elapsed"]
    assert set(e) == {"reactive", "516_gnn", "516_mpoff", "v3_gnn", "v3_mpoff", "516_peeronly"}
    assert len(e["516_peeronly"]) == 16 and len(e["516_gnn"]) == 16 and len(e["reactive"]) == 4
    assert ("cs6s9001", 1) in e["v3_gnn"]


def test_a0_passes_when_the_reserve_reproduces_and_fails_on_drift():
    assert read_a0_from(tables(_po(), _p3()))["verdict"] == V_A0_PASS
    assert read_a0_from(tables(_po(a0_drift=0.002), _p3()))["verdict"] == V_A0_FAIL


def test_phase_a_tie_mechanism_and_peer_term():
    res = read_phase(tables(_po(), _p3()), "516")
    for rung in ("R0", "R3"):
        r = res["rungs"][rung]
        assert r["a2"]["verdict"] == V_TIE                       # same elapsed as mpoff
        assert r["a3"]["verdict"] == V_GIN_OVERREACTION         # queue like mpoff, well below gnn
        assert r["a4"]["verdict"] == V_PEER_KEPT
        assert r["d_peeronly"] is not None and abs(r["d_peeronly"] - r["d_mpoff"]) < 1e-6
    assert res["headline"] == V_TIE


def test_phase_a_beats_and_pointwise():
    assert read_phase(tables(_po(po_scale=0.85), _p3()), "516")["headline"] == V_BEATS
    assert read_phase(tables(_po(po_scale=1.2), _p3()), "516")["headline"] == V_POINTWISE


def test_b1_reads_the_corpus_lever_per_arm():
    po = _po() + _po(po_scale=0.9, corpus="1670")
    # add 1670 gnn/mpoff arms that are 10 % faster than their 516 twins
    for d in _p3():
        if d["arm_kind"] in ("gnn", "mpoff"):
            po.append({**d, "arm": d["arm"].replace("__" + d["arm_kind"], "__1670_" + d["arm_kind"]),
                       "corpus": "1670", "averageElapsedTime": d["averageElapsedTime"] * 0.9})
    res = read_b1(tables(po, _p3()))
    assert res["gnn"]["R0"]["verdict"] == V_CORPUS_HELPS and res["peeronly"]["R3"]["verdict"] == V_CORPUS_HELPS


def _sum(cell, rung, corpus, kind, seed, elapsed):
    return {"arm": f"{cell}__{rung}__{corpus}_{kind}_s{seed}", "cell": cell, "rung": rung,
            "corpus": corpus, "arm_kind": kind, "checkpoint_seed": seed, "num_tasks": 50000,
            "averageElapsedTime": elapsed, "averageQueueTime": elapsed * 0.99,
            "averageWaitTime": 0.73, "totalPeerExchangeTime": 1.0, "totalPeerRendezvousWait": 1.0}


def test_b3_collapses_to_one_value_per_checkpoint_and_needs_all_16():
    cells = ("cs80s9001", "cs80s9002", "cs80s9003", "cs80s9005")
    po, full = [], []
    for s in range(1, 17):
        for c in cells:
            full.append(_sum(c, "R3", "1670", "peeronly", s, 80.0))
            full.append(_sum(c, "R3", "1670", "mpoff", s, 100.0))
            if s <= 4:
                po.append(_sum(c, "R3", "1670", "peeronly", s, 80.0))
                po.append(_sum(c, "R3", "1670", "mpoff", s, 100.0))
    # only 4 checkpoints served -> the bar refuses to read rather than reporting n=4 as a result
    assert read_b3(tables(po, []))["verdict"] == V_UNREADABLE
    r = read_b3(tables(full, []))
    assert r["n"] == 16 and r["median"] < -5.0 and r["rung"] == "R3"
    assert sorted(r["per_seed_pct"]) == list(range(1, 17))
