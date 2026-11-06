"""Backlog ladder (scratch): the dispatch by groups waiting, the charge table of each path, the thresholds parser."""
import types

import pytest

from src.placement import decision_cost as dc
from src.policy.gnn import scheduler as gs
from src.policy.gnn.scheduler import GNNScheduler, ladder_thresholds
from src.policy.peer_greedy_network import scheduler as pg


def _sched(monkeypatch, q, thresholds=(3, 9999), g=5):
    s = pg.HybridGnnCdScheduler.__new__(pg.HybridGnnCdScheduler)
    s._ladder, s.hyb_g, s._bud_q, s.pg_ils_per = thresholds, g, q, 200
    s._ladder_big, s._ladder_nopass = None, False
    for name in pg.LAD_COUNTERS + pg.HYB_COUNTERS:
        setattr(s, name, 0)
    calls = []
    monkeypatch.setattr(pg.HybridGnnCdScheduler, "_pg_decide", lambda self, b, st: calls.append(("rule", self.pg_ils_per)) or "rule-plan")
    monkeypatch.setattr(GNNScheduler, "_prefix_inference", lambda self, b, st, qs, ts: calls.append(("gnn", self._ladder_nopass)) or "gnn-plan")
    return s, calls


def test_thresholds_parser(monkeypatch):
    monkeypatch.delenv(gs.LADDER_ENV, raising=False)
    assert ladder_thresholds() is None
    monkeypatch.setenv(gs.LADDER_ENV, "3,10")
    assert ladder_thresholds() == (3, 10)
    for bad in ("3", "a,b", "10,3", "-1,5"):
        monkeypatch.setenv(gs.LADDER_ENV, bad)
        with pytest.raises(ValueError, match="FAIL LOUD"):
            ladder_thresholds()


def test_low_serves_the_hybrid_as_it_is(monkeypatch):
    s, calls = _sched(monkeypatch, q=2)
    small = [object()] * 4      # <= G: the rule with its ILS
    assert s._prefix_inference(small, None, None, None) == "rule-plan" and calls == [("rule", 200)] and s._ladder_big is False
    big = [object()] * 8        # > G: the GNN with its refine passes
    assert s._prefix_inference(big, None, None, None) == "gnn-plan" and calls[-1] == ("gnn", False) and s._ladder_big is True
    assert s._ladder == (3, 9999) and s.lad_low_groups == 2 and s.lad_med_groups == s.lad_high_groups == 0


def test_medium_is_the_refined_gnn_alone_whatever_the_batch(monkeypatch):
    s, calls = _sched(monkeypatch, q=3, thresholds=(3, 9999))
    assert s._prefix_inference([object()] * 2, None, None, None) == "gnn-plan"
    assert calls == [("gnn", False)] and s._ladder_big is True and s.lad_med_groups == 1 and s.lad_med_tasks == 2 and s.lad_high_groups == 0


def test_high_is_the_gnn_without_refine_passes_when_switched_on(monkeypatch):
    s, calls = _sched(monkeypatch, q=10, thresholds=(3, 10))
    assert s._prefix_inference([object()] * 9, None, None, None) == "gnn-plan"
    assert calls == [("gnn", True)] and s._ladder_big is True and s._ladder_nopass is False and s.lad_high_groups == 1 and s.lad_q_sum == 10


def test_pinned_ladder_never_leaves_low(monkeypatch):
    s, calls = _sched(monkeypatch, q=500, thresholds=(9999, 9999))
    s._prefix_inference([object()] * 3, None, None, None)
    assert calls == [("rule", 200)] and s.lad_low_groups == 1 and s.lad_med_groups == s.lad_high_groups == 0


def test_the_charge_table_follows_the_path(monkeypatch):
    monkeypatch.setenv(dc.BUCKET_ENV, '{"le10": 1.0, "11_50": 2.0, "gt50": 3.0}')
    monkeypatch.setenv(dc.BIG_BUCKET_ENV, '{"le10": 10.0, "11_50": 20.0, "gt50": 30.0}')
    monkeypatch.setenv("HEROSIM_HYB_G", "5")
    assert dc.rate_ms(4) == 1.0 and dc.rate_ms(8) == 10.0 and dc.rate_ms(30) == 20.0   # size rule unchanged
    assert dc.rate_ms(4, True) == 10.0 and dc.rate_ms(30, False) == 2.0 and dc.rate_ms(8, False) == 1.0
    assert dc.rate_ms(8, None) == dc.rate_ms(8)


def test_the_ladder_counters_reach_the_summary():
    from src.placement import orchestrator

    src = open(orchestrator.__file__).read()
    for name in pg.LAD_COUNTERS:
        assert f'"{name}"' in src


def test_the_gate_knows_the_pinned_ladder_kind():
    import importlib
    G = importlib.import_module("scripts_cosim.fresh_topo_burst_v1_gate")
    k = "ra_gnn_eng_combfl_linkstatic_e40_nosplit_sr2_hybg5_lad9999_9999"
    assert (4, 9999) in G.LAD_ALLOWED and k in G.LAD_KINDS and k in G.R1A_ARMS and G.split_dc(k + "_k100")[0] == k


def test_the_refined_gnn_rung_is_served_without_best_of_k_and_the_env_is_restored(monkeypatch):
    seen = []
    s, calls = _sched(monkeypatch, q=5, thresholds=(4, 9999))
    monkeypatch.setattr(GNNScheduler, "_prefix_inference", lambda self, b, st, qs, ts: seen.append(__import__("os").environ.get("HEROSIM_BOK_K")) or "gnn-plan")
    monkeypatch.setenv("HEROSIM_BOK_K", "4")
    assert s._prefix_inference([object()] * 8, None, None, None) == "gnn-plan"
    assert seen == ["0"] and __import__("os").environ["HEROSIM_BOK_K"] == "4"
    s._bud_q = 1   # low rung: the hybrid as it is, best-of-K left on for its GNN half
    s._prefix_inference([object()] * 8, None, None, None)
    assert seen == ["0", "4"]


def test_the_high_rung_identity_arm_is_registered():
    import importlib
    G = importlib.import_module("scripts_cosim.fresh_topo_burst_v1_gate")
    k = "ra_gnn_eng_combfl_linkstatic_e40_nosplit_sr2_hybg5_lad0_9999"
    assert k in G.LAD_KINDS and k in G.R1A_ARMS and k in G.R1A_SCRATCH
