"""Tests for the peer_only_v1 gate glue."""
from scripts_cosim.peer_only_v1_gate_read import (
    read_a0_from, read_b1, read_b3, read_b5, read_phase, tables,
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


def _po(po_scale=1.0, po_queue_scale=1.0, po_peer=10.0, a0_drift=0.0, corpus="516", with_a0=True):
    """with_a0=False when a second _po() is concatenated: a real gate writes one summary FILE
    per arm name, so two A0 re-serves of the same arm cannot exist and tables() refuses them."""
    docs = []
    for rung, cells in CELLS.items():
        # A0 re-serves: v3 gnn/mpoff seed 1 on the first cell
        c0 = cells[0]; react = 20.0
        if with_a0:
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
    po = _po() + _po(po_scale=0.9, corpus="1670", with_a0=False)
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


def test_tables_refuses_two_summaries_for_one_arm():
    """The arm name is the summary file name; a duplicate means a naming collision upstream."""
    d = _sum("cs80s9001", "R3", "1670", "peeronly", 1, 80.0)
    try:
        tables([d, {**d, "averageElapsedTime": 999.0}], [])
    except ValueError as exc:
        assert "colliding" in str(exc)
    else:
        raise AssertionError("a duplicate arm must fail loud, not let the last one win")


def test_v3ext_extends_the_516_mpoff_arm():
    """B4 serves partial_state_v3's mpoff under the tag v3ext; it must land on the 516 label."""
    cells = ("cs80s9001", "cs80s9002", "cs80s9003", "cs80s9005")
    docs = [_sum(c, "R3", "v3ext", "mpoff", s, 100.0) for c in cells for s in (3, 6)]
    tab = tables(docs, [])
    assert set(tab["R3"]["elapsed"]) == {"516_mpoff"}
    assert sorted({s for _, s in tab["R3"]["elapsed"]["516_mpoff"]}) == [3, 6]


def _ladder(seeds, rungs=(("R0", 6), ("R1", 12), ("R2", 24), ("R3", 80)), drop=None):
    """Full B5 ladder; `drop` is a (rung, cell, seed) triple to omit, as a resource kill would."""
    cells = ("cs80s9001", "cs80s9002", "cs80s9003", "cs80s9005")
    docs = []
    for rung, _ in rungs:
        for c in cells:
            for s in seeds:
                if drop and (rung, c, s) == drop:
                    continue
                docs.append(_sum(c, rung, "1670", "peeronly", s, 80.0))
                docs.append(_sum(c, rung, "1670", "mpoff", s, 100.0))
    return docs


def test_b5_reads_the_whole_ladder_when_every_arm_landed():
    r = read_b5(tables(_ladder(range(1, 17)), []))
    assert r["verdict"] in ("MARGIN-GROWS-WITH-SCALE", "MARGIN-NOT-MONOTONE-IN-SCALE")
    assert "disclosed" not in r


def test_b5_stays_unreadable_on_a_lost_arm_but_discloses_the_rest():
    """The registered bar is never relaxed; the 15 complete checkpoints still get printed."""
    r = read_b5(tables(_ladder(range(1, 17), drop=("R1", "cs80s9001", 9)), []))
    assert r["verdict"] == V_UNREADABLE
    d = r["disclosed"]
    assert d["excluded_seeds"] == [9] and d["n_seeds"] == 15
    # the disclosed read must actually RESOLVE on 15 -- it is not the registered bar
    for rung in ("R0", "R1", "R2", "R3"):
        assert d["per_rung"][rung]["verdict"] != V_UNREADABLE, rung
        assert d["per_rung"][rung]["n"] == 15


def test_tables_keeps_the_psv3_arms_at_every_b5_rung():
    """A2_RUNGS is (R0, R3); B5's ladder also needs reactive at R1 and R2."""
    p3 = []
    for rung in ("R0", "R1", "R2", "R3"):
        d = _sum("cs80s9001", rung, "516", "reactive", 0, 500.0)
        d["arm_kind"] = "reactive"
        p3.append(d)
    tab = tables([], p3)
    for rung in ("R0", "R1", "R2", "R3"):
        assert "reactive" in tab[rung]["elapsed"], rung


# --- C1: the bipartite arm at power (AMENDMENT 6) -----------------------------------------

def _c1_docs(seeds, rungs=("R3", "R0"), gnn=100.0, peeronly=80.0, drop=None):
    """peeronly and gnn at the C1 rungs; `drop` omits a (rung, cell, seed) as a kill would."""
    cells = ("cs80s9001", "cs80s9002", "cs80s9003", "cs80s9005")
    docs = []
    for rung in rungs:
        for c in cells:
            for s in seeds:
                if drop and (rung, c, s) == drop:
                    continue
                docs.append(_sum(c, rung, "1670", "peeronly", s, peeronly + s * 0.1))
                docs.append(_sum(c, rung, "1670", "gnn", s, gnn + s * 0.1))
    return docs


def test_c1_reads_both_rungs_when_every_checkpoint_landed():
    from scripts_cosim.peer_only_v1_gate_read import read_c1
    from scripts_cosim.peer_only_v1_read import V_BIPARTITE_COSTS
    r = read_c1(tables(_c1_docs(range(1, 17)), []))
    assert r["verdict"] == V_BIPARTITE_COSTS
    assert set(r["per_rung"]) == {"R3", "R0"}
    assert all(v["n"] == 16 for v in r["per_rung"].values())


def test_c1_is_unreadable_on_four_checkpoints_which_is_the_whole_point():
    from scripts_cosim.peer_only_v1_gate_read import read_c1
    r = read_c1(tables(_c1_docs((1, 2, 4, 5)), []))
    assert r["verdict"] == V_UNREADABLE


def test_c1_names_a_lost_arm_rather_than_averaging_around_it():
    from scripts_cosim.peer_only_v1_gate_read import read_c1
    r = read_c1(tables(_c1_docs(range(1, 17), drop=("R3", "cs80s9001", 9)), []))
    assert r["verdict"] == V_UNREADABLE
    assert "missing cells" in r["per_rung"]["R3"]["reason"]


def test_c1_can_report_the_gin_helping_at_one_rung():
    """peeronly SLOWER than gnn at R0 must surface, not be smoothed into a ladder headline."""
    from scripts_cosim.peer_only_v1_gate_read import read_c1
    from scripts_cosim.peer_only_v1_read import V_BIPARTITE_HELPS, V_BIPARTITE_COSTS
    docs = (_c1_docs(range(1, 17), rungs=("R3",))
            + _c1_docs(range(1, 17), rungs=("R0",), gnn=80.0, peeronly=100.0))
    r = read_c1(tables(docs, []))
    assert r["per_rung"]["R3"]["verdict"] == V_BIPARTITE_COSTS
    assert r["per_rung"]["R0"]["verdict"] == V_BIPARTITE_HELPS
    assert r["verdict"] == V_BIPARTITE_HELPS
