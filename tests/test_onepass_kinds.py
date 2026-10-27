"""One-pass strong hand rules (batched_sym, batched_fsc): the CD sym / symfsawarec score with no refine. Existing kinds' env and dispatch are unchanged; the
new kinds dispatch to the batch scheduler (no CD passes), carry the same env and the same decision-cost w alias, and the instrument checks fail loud."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts_cosim"))

import os  # noqa: E402
os.environ.setdefault("WF1_RUNGS", "moderate,heavy")
import fresh_topo_burst_v1_gate as G  # noqa: E402

PRINTED = ("HEROSIM_PG_BATCH_ALPHA", "HEROSIM_PG_INFLIGHT_FULL", "HEROSIM_PG_EXCHANGE_SYMMETRIC", "HEROSIM_PG_FAIRSHARE_AWARE", "HEROSIM_PG_FAIRSHARE_AWARE_DRAIN",
           "HEROSIM_PG_FAIRSHARE_AWARE_COMMIT", "HEROSIM_PG_CD_EXPANSION", "HEROSIM_PG_CD_EXACT", "HEROSIM_PG_SELFPREDICT_BATCH", "HEROSIM_PG_BATCH_ORDER", "GNN_BATCH_BY_PEER_GROUP", "GNN_DECODE_MODE")


class _Stop(Exception):
    pass


def served_env(kind, monkeypatch, tmp_path):
    seen = {}

    def fake(cmd, env, *a, **k):
        seen.update(env)
        raise _Stop

    monkeypatch.setattr(G, "run_logged", fake)
    window = next(iter(G.WF1_LADDER))
    inputs = tmp_path / "in"
    sub = G.WF1_LADDER[window][1]
    (inputs / sub / "cfg").mkdir(parents=True, exist_ok=True)
    (inputs / sub / "cfg" / "cc40s30572.json").write_text("{}")
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    with pytest.raises(_Stop):
        G.run_one({"kind": kind, "seed": 0, "topo": 30572, "window": window}, str(inputs), str(out), "8g", 100)
    return seen


def test_dispatch_and_aliases():
    assert G.ONEPASS_BASE == ("batched_sym", "batched_fsc")
    assert G.ONEPASS_KINDS == ("batched_sym", "batched_fsc", "batched_sym_sp", "batched_fsc_sp", "batched_sym_sp_order", "batched_fsc_sp_order")
    for k in G.ONEPASS_KINDS:
        assert G.RULE_POLICY[k] == "peer_greedy_network_batch" and G.check_alias(k) == "batched" and k in G.R1A_DIAG and k in G.R1A_ARMS
        assert G.ib0_alpha(k) == {"HEROSIM_PG_BATCH_ALPHA": "0"} and G.split_dc(k + "_k100") == (k, 1.0)
        assert G.dc_cost_ms(k, 1.0, "moderate") == G.dc_cost_ms("cd_ib0_inflightfull_sym", 1.0, "moderate")
        assert G.dc_cost_ms(k, 1.0, "heavy") == G.dc_cost_ms("cd_ib0_inflightfull_fsawarec", 1.0, "heavy")
    assert json.load(open(G.DC_W_FILE))["w_ms_per_plan_alias"]["batched_fsc"] == "cd_ib0_inflightfull"
    for k in ("cd_ib0_inflightfull_sym", "cd_ib0_inflightfull_fsawarec", "cd_ib0_inflightfull_fsaware2", "cd_ib0_inflightfull"):
        assert G.RULE_POLICY[k] == "peer_greedy_network_cd" and G.check_alias(k) == "cd"


def test_new_kinds_env_and_existing_unchanged(monkeypatch, tmp_path):
    want = {
        "batched_sym": dict(HEROSIM_PG_BATCH_ALPHA="0", HEROSIM_PG_INFLIGHT_FULL="1", HEROSIM_PG_EXCHANGE_SYMMETRIC="1"),
        "batched_fsc": dict(HEROSIM_PG_BATCH_ALPHA="0", HEROSIM_PG_INFLIGHT_FULL="1", HEROSIM_PG_EXCHANGE_SYMMETRIC="1", HEROSIM_PG_FAIRSHARE_AWARE="1", HEROSIM_PG_FAIRSHARE_AWARE_DRAIN="0"),
        "cd_ib0_inflightfull_sym": dict(HEROSIM_PG_BATCH_ALPHA="0", HEROSIM_PG_INFLIGHT_FULL="1", HEROSIM_PG_EXCHANGE_SYMMETRIC="1"),
        "cd_ib0_inflightfull_fsawarec": dict(HEROSIM_PG_BATCH_ALPHA="0", HEROSIM_PG_INFLIGHT_FULL="1", HEROSIM_PG_FAIRSHARE_AWARE="1", HEROSIM_PG_FAIRSHARE_AWARE_DRAIN="0"),
        "batched_sym_sp": dict(HEROSIM_PG_BATCH_ALPHA="0", HEROSIM_PG_INFLIGHT_FULL="1", HEROSIM_PG_EXCHANGE_SYMMETRIC="1", HEROSIM_PG_SELFPREDICT_BATCH="1"),
        "batched_fsc_sp": dict(HEROSIM_PG_BATCH_ALPHA="0", HEROSIM_PG_INFLIGHT_FULL="1", HEROSIM_PG_EXCHANGE_SYMMETRIC="1", HEROSIM_PG_FAIRSHARE_AWARE="1", HEROSIM_PG_FAIRSHARE_AWARE_DRAIN="0", HEROSIM_PG_SELFPREDICT_BATCH="1"),
        "batched_sym_sp_order": dict(HEROSIM_PG_BATCH_ALPHA="0", HEROSIM_PG_INFLIGHT_FULL="1", HEROSIM_PG_EXCHANGE_SYMMETRIC="1", HEROSIM_PG_SELFPREDICT_BATCH="1", HEROSIM_PG_BATCH_ORDER="bytes"),
        "batched_fsc_sp_order": dict(HEROSIM_PG_BATCH_ALPHA="0", HEROSIM_PG_INFLIGHT_FULL="1", HEROSIM_PG_EXCHANGE_SYMMETRIC="1", HEROSIM_PG_FAIRSHARE_AWARE="1", HEROSIM_PG_FAIRSHARE_AWARE_DRAIN="0", HEROSIM_PG_SELFPREDICT_BATCH="1", HEROSIM_PG_BATCH_ORDER="bytes"),
        "batched": dict(),
        "cd": dict(),
    }
    for kind, exp in want.items():
        env = served_env(kind, monkeypatch, tmp_path)
        got = {k: env[k] for k in PRINTED if k in env and k.startswith("HEROSIM_PG_")}
        assert got == exp, (kind, got)
        assert env.get("GNN_BATCH_BY_PEER_GROUP") == "1" and env.get("GNN_DECODE_MODE") == "masked_topo"
        assert "HEROSIM_PG_CD_EXPANSION" not in env and "HEROSIM_PG_CD_EXACT" not in env


def good_cell(env, **c):
    return {"env": env, "counters": {"pg_batches": 5, "pg_cd_passes": 0, "pg_inflight_full_charged": 3, "pg_exchange_sym_evals": 9, **c}}


def test_scheduler_has_no_refine_passes():
    src = (REPO_ROOT / "src/policy/peer_greedy_network/scheduler.py").read_text()
    batch = src.split("class PeerGreedyNetworkBatchScheduler")[1].split("class PeerGreedyNetworkCDScheduler")[0]
    assert "_pg_cd_passes(" not in batch


def test_instrument_checks_are_wired():
    src = (REPO_ROOT / "scripts_cosim" / "fresh_topo_burst_v1_gate.py").read_text()
    assert 'orig_kind in (SYM_KIND, *ONEPASS_KINDS) and (out["env"].get("HEROSIM_PG_EXCHANGE_SYMMETRIC") != "1"' in src
    assert 'if orig_kind in ONEPASS_KINDS:\n        if int(c.get("pg_cd_passes") or 0) != 0:' in src
    assert 'if orig_kind in (*FSAWARE_KINDS, *ONEPASS_FSC):' in src


def test_provisional_partner_node_is_read_only():
    from types import SimpleNamespace
    from src.policy.peer_greedy_network.scheduler import PeerGreedyNetworkBatchScheduler as B

    sch = B.__new__(B)
    sch._obs = None; sch.pg_batch_blind = False; sch._pg_batch_ids = frozenset({1, 2})
    sch.pg_partners_known = sch.pg_partners_unknown = sch.pg_sp_provisional_used = sch.pg_partners_blinded = 0
    orch = SimpleNamespace(peer_exchange={1: {2: 5.0}}, task_by_id={})
    planned = {}
    sch._pg_prov = {}
    assert sch._pg_peer_nodes(SimpleNamespace(id=1), orch, planned) == [] and sch.pg_partners_unknown == 1
    sch._pg_prov = {2: "nodeB"}
    assert sch._pg_peer_nodes(SimpleNamespace(id=1), orch, planned) == [("nodeB", 5.0)]
    assert sch.pg_sp_provisional_used == 1 and planned == {}
    planned[2] = "nodeC"  # a placed partner always wins over the provisional node
    assert sch._pg_peer_nodes(SimpleNamespace(id=1), orch, planned) == [("nodeC", 5.0)] and sch.pg_sp_provisional_used == 1


def test_sp_and_order_wiring_in_source():
    src = (REPO_ROOT / "src/policy/peer_greedy_network/scheduler.py").read_text()
    assert "finally:\n            self._pg_prov = {}" in src and "-sum(float(b)" in src
    assert "planned[tid] = node.node_name" in src and "self._pg_prov[" not in src.split("def _pg_provisional")[1].split("def _pg_remote_edges")[0]
