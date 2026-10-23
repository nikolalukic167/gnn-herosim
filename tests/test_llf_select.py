import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
import llf_select as S  # noqa: E402

D_BLOCKS = {"peerExchangeFairshare": {"partners": "serial"}, "peerExchangeTcp": {"model": "slow_start_v1"}, "accessCellularOutMbps": None}


def write_variant(d, name, gnn_of, twin=5.0, skip=None, extra=None):
    """gnn_of(t, w, r) -> the gnn's latency; the twin is `twin` everywhere."""
    os.makedirs(d, exist_ok=True)
    g, tw = S.PAIRS[name]
    for kind, f in ((g, gnn_of), (tw, lambda t, w, r: twin)):
        for t in S.TOPOS:
            for w in S.WINDOWS:
                for r in S.RUNGS:
                    p = S.cell_path(d, kind, t, w, r)
                    if skip and os.path.basename(p) == skip:
                        continue
                    s = {"averageElapsedTime": f(t, w, r), "num_tasks": 50000, **D_BLOCKS, **(extra or {})}
                    with open(p, "w") as fh:
                        json.dump(s, fh)


def flags(**over):
    f = {v: {"parity": True, "collapse": True, "curves": True, "source": "test"} for v in S.VARIANTS}
    f.update(over)
    return f


def run(tmp_path, gnn, fl=None, **kw):
    dirs = {}
    for v in S.VARIANTS:
        dirs[v] = str(tmp_path / v)
        write_variant(dirs[v], v, gnn[v], **kw.get(v, {}))
    fp = tmp_path / "flags.json"
    fp.write_text(json.dumps(fl or flags()))
    out = tmp_path / "sel.json"
    rc = S.main(["--as-built", dirs["as_built"], "--static", dirs["static"], "--conv", dirs["conv"], "--flags", str(fp), "--out", str(out)])
    return rc, json.loads(out.read_text())


def const(x):
    return lambda t, w, r: x


def test_lowest_score_wins(tmp_path):
    rc, o = run(tmp_path, {"as_built": const(5.0), "static": const(4.5), "conv": const(4.8)})
    assert rc == 0 and o["choice"] == "static"
    assert o["variants"]["static"]["score"] == pytest.approx(-10.0) and o["variants"]["static"]["n"] == 12
    assert o["variants"]["as_built"]["score"] == pytest.approx(0.0)


def test_runaway_pair_is_dropped_only_from_the_without_score(tmp_path):
    def g(t, w, r):
        return 100.0 if (t, w, r) == (30179, "g0", "heavy") else 4.75
    rc, o = run(tmp_path, {"as_built": const(5.0), "static": g, "conv": const(5.0)})
    x = o["variants"]["static"]
    assert x["runaway_cells"] == ["gnn:30179:g0:heavy"]
    assert x["per_topology"]["heavy:30179"] == pytest.approx(-5.0)       # g1 alone
    assert x["per_topology_all"]["heavy:30179"] == pytest.approx((1900.0 - 5.0) / 2)
    assert x["score"] == pytest.approx(-5.0) and o["choice"] == "static"


def test_tie_goes_to_all_cells_then_simplicity(tmp_path):
    rc, o = run(tmp_path, {"as_built": const(4.99), "static": const(4.985), "conv": const(4.99)})
    assert o["choice"] == "as_built" and "simpler" in o["reason"]          # within 0.5 pp twice


def r(score, score_all, el=True):
    return {"eligible": el, "score": score, "score_all": score_all}


def test_choose_tie_breaks():
    assert S.choose({"as_built": r(-1.0, -1.0), "static": r(-3.0, -3.0), "conv": r(-2.0, -2.0)})[0] == "static"
    # static and conv tie without runaways (0.4 pp); conv is lower all-cells by more than 0.5 pp
    assert S.choose({"as_built": r(0.0, 0.0), "static": r(-3.0, -1.0), "conv": r(-2.6, -2.0)})[0] == "conv"
    # tie on both: the simpler wins
    assert S.choose({"as_built": r(-2.9, -2.0), "static": r(-3.0, -2.2), "conv": r(-2.6, -2.4)})[0] == "as_built"
    # the all-cells stage only sees the variants tied on the primary score
    assert S.choose({"as_built": r(-1.0, -9.0), "static": r(-3.0, -1.0), "conv": r(-2.6, -2.0)})[0] == "conv"
    assert S.choose({"as_built": r(-1.0, -1.0, el=False), "static": r(0.0, 0.0), "conv": r(5.0, 5.0)})[0] == "static"
    assert S.choose({"as_built": r(None, -1.0), "static": r(0.0, 0.0), "conv": r(5.0, 5.0)})[0] is None


def test_ineligible_variant_cannot_win_and_unknown_is_not_a_pass(tmp_path):
    fl = flags(static={"parity": True, "collapse": True, "curves": "yes"}, conv={"parity": True, "collapse": False, "curves": True})
    rc, o = run(tmp_path, {"as_built": const(5.0), "static": const(4.0), "conv": const(4.0)}, fl=fl)
    assert o["choice"] == "as_built"
    assert not o["variants"]["static"]["eligible"] and not o["variants"]["conv"]["eligible"]


def test_hold_when_none_eligible(tmp_path):
    fl = {"as_built": {}, "static": {}, "conv": {}}
    rc, o = run(tmp_path, {v: const(4.0) for v in S.VARIANTS}, fl=fl)
    assert rc == 2 and o["choice"] is None and o["reason"].startswith("HOLD")


def test_refuses_while_any_variant_is_incomplete(tmp_path):
    skip = os.path.basename(S.cell_path("", "ra_gnn_eng_combfl_linkconv", 30181, "g1", "heavy"))
    with pytest.raises(SystemExit, match="variant conv is not complete"):
        run(tmp_path, {v: const(4.0) for v in S.VARIANTS}, conv={"skip": skip})
    assert not (tmp_path / "sel.json").exists()


def test_refuses_non_option_d_cells(tmp_path):
    with pytest.raises(SystemExit, match="not an option D cell"):
        run(tmp_path, {v: const(4.0) for v in S.VARIANTS}, static={"extra": {"accessCellularOutMbps": 5.0}})


def test_read_once(tmp_path):
    run(tmp_path, {v: const(4.0) for v in S.VARIANTS})
    with pytest.raises(SystemExit, match="read once"):
        run(tmp_path, {v: const(4.0) for v in S.VARIANTS})


def test_pairs_are_the_registered_gate_kinds():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
    import fresh_topo_burst_v1_gate as G
    assert S.PAIRS["as_built"] == G.COMB_LINK[:2] and S.PAIRS["static"] == G.COMB_STATIC[:2]
    assert S.PAIRS["conv"] == G.COMB_LINKX[:2]
