"""slate_fix_v2 reader (docs/lineages/slate_fix_v2.md): identity against slate_fix_v1, steady-state primary with the transient
delta, the 4-seed MP family and the bar's own transient share."""
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts_cosim"))

COMMIT = {3: "f4f0453f0000", 4: "f4f0453f0000"}  # seeds 3-4 at the second pin, the rest at the first
IB0, SS, NS, TW = "cd_exactS_ib0", "ra_gnn_eng_selfsearch", "ra_gnn_eng_nosplit", "ra_twin_eng_nosplit"


def _cell(d, topo, win, rung, kind, seed, whole, steady):
    s = {"averageElapsedTime": whole, "wallclock_s": 1.0, "num_tasks": 50000, "code": {"commit": COMMIT.get(seed, "6a5c922cd1b2")},
         "steady_state": {"after_360s": {"T_s": 360.0, "n": 45000, "mean": steady},
                          "after_p5_dispatch": {"T_s": 200.0, "n": 47500, "mean": steady},
                          "decile_mean": [whole * 3] + [steady] * 9}}
    (d / f"cc40s{topo}__{win}{rung}__{kind}_s{seed}.summary.json").write_text(json.dumps(s))


def _dirs(tmp_path, gap_whole, gap_steady, mp_pct, mp_pct_34=None):
    gate, ref = tmp_path / "gate", tmp_path / "ref"
    gate.mkdir(), ref.mkdir()
    for i, t in enumerate(range(19420, 19432)):
        jit = 0.01 * (i + 1)
        for w in ("g0", "g1"):
            for r in ("moderate", "heavy"):
                hw, hs = 10.0, 6.0  # the bar: 40 % of its whole-run mean is the opening
                for d in (gate, ref):
                    _cell(d, t, w, r, IB0, 0, hw, hs)
                for kind in (SS, NS):
                    for s in (1, 2):
                        lw, ls = hw * (1 + (gap_whole + jit) / 100), hs * (1 + (gap_steady + jit) / 100)
                        for d in (gate, ref):
                            _cell(d, t, w, r, kind, s, lw, ls)
                for s in (1, 2, 3, 4):
                    pct = mp_pct if (s <= 2 or mp_pct_34 is None) else mp_pct_34
                    gw = hw * (1 + (gap_whole + jit) / 100)
                    gs = hs * (1 + (gap_steady + jit) / 100)
                    _cell(gate, t, w, r, NS, s, gw, gs)
                    _cell(gate, t, w, r, TW, s, gw / (1 + (pct - jit) / 100), gs / (1 + (pct - jit) / 100))
    return str(gate), str(ref)


def test_hides_when_the_steady_gap_is_larger(tmp_path):
    rd = importlib.import_module("slate_fix_v2_read")
    r = rd.read(*_dirs(tmp_path, 5.0, 9.0, -3.0))
    assert r["identity"]["checked"] == 12 * 2 * 2 * 5 and r["identity"]["mismatched"] == []
    c = r["primary_s360"]["self-search|heavy"]
    assert 9.0 < c["median_pct"] < 9.2 and c["label"] == "E-FASTER" and c["delta"]["label"] == "HIDES"
    assert 5.0 < c["whole"]["median_pct"] < 5.2 and 3.9 < c["delta"]["median_pp"] < 4.1
    assert r["mp"]["labels"] == {"whole": "CONFIRMED", "s360": "CONFIRMED"}
    h = r["hand_transient"]["heavy"]
    assert h["whole_s"] == 10.0 and h["s360_s"] == 6.0 and abs(h["transient_share"] - 0.4) < 1e-12


def test_inflates_and_mp_direction_only(tmp_path):
    rd = importlib.import_module("slate_fix_v2_read")
    r = rd.read(*_dirs(tmp_path, 5.0, 2.0, -1.0))
    assert r["primary_s360"]["no-split|moderate"]["delta"]["label"] == "INFLATES"
    assert r["mp"]["labels"]["whole"] == "DIRECTION-ONLY"
    assert set(k.split(" ")[1] for k in r["descriptive"] if k.startswith("mp ")) == {"seeds"}


def test_mismatched_whole_run_leaves_the_steady_families(tmp_path, capsys):
    rd = importlib.import_module("slate_fix_v2_read")
    gate, ref = _dirs(tmp_path, 5.0, 9.0, -3.0)
    _cell(Path(ref), 19420, "g0", "heavy", SS, 1, 99.0, 6.0)  # slate_fix_v1 recorded another whole-run number for this cell
    r = rd.read(gate, ref)
    assert [m["cell"] for m in r["identity"]["mismatched"]] == [[SS, 1, 19420, "g0", "heavy"]]
    rd.print_report(r)
    out = capsys.readouterr().out
    assert "1 mismatched" in out and "the bar's own transient" in out and "never a hand win" in out


def test_refuses_topologies_outside_the_registered_range(tmp_path):
    rd = importlib.import_module("slate_fix_v2_read")
    gate, ref = _dirs(tmp_path, 5.0, 9.0, -3.0)
    _cell(Path(gate), 19500, "g0", "heavy", IB0, 0, 1.0, 1.0)
    with pytest.raises(SystemExit):
        rd.read(gate, ref)


def test_both_pins_are_recorded_and_any_other_commit_or_window_is_refused(tmp_path):
    rd = importlib.import_module("slate_fix_v2_read")
    gate, ref = _dirs(tmp_path, 5.0, 9.0, -3.0)
    r = rd.read(gate, ref)
    assert r["cells_per_pin"] == {"6a5c922c": 12 * 2 * 2 * 7, "f4f0453f": 12 * 2 * 2 * 4}
    stray = Path(gate) / "cc40s19420__g0heavy__cd_exactS_ib0_s0.summary.json"
    s = json.loads(stray.read_text()); s["code"] = {"commit": "3b1ebd00aaaa"}; stray.write_text(json.dumps(s))
    with pytest.raises(SystemExit):
        rd.read(gate, ref)
    s["code"] = {"commit": "6a5c922cd1b2"}; stray.write_text(json.dumps(s))
    _cell(Path(gate), 19420, "g2", "heavy", IB0, 0, 1.0, 1.0)
    with pytest.raises(SystemExit):
        rd.read(gate, ref)
