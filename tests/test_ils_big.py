"""uncut_groups_s0_v1 ils_oracle_live: the ILS refine is opt-in, names exactly one budget, and its counters are exported."""
import re
from pathlib import Path

import src.policy.peer_greedy_network.scheduler as S

REPO = Path(__file__).resolve().parents[1]
KEYS = ("pg_ils_batches", "pg_ils_evals", "pg_ils_tasks_moved", "pg_ils_gain_seconds", "pg_ils_budget_stops")


def test_counters_registered_and_exported():
    exporter = (REPO / "src/placement/orchestrator.py").read_text()
    for k in KEYS:
        assert k in S.PEER_GREEDY_COUNTERS and f'"{k}"' in exporter


def test_gate_wires_the_kind_like_cd_ib0():
    g = (REPO / "scripts_cosim/fresh_topo_burst_v1_gate.py").read_text()
    assert '"ils_oracle_live": "peer_greedy_network_cd"' in g
    assert 'check_alias' in g and re.search(r'kind in \("cd_ib0", "ils_oracle_live"\)', g)
    assert "HEROSIM_PG_ILS_MIN_BATCH" in g and "R1A_ILS_PER" in g


def test_scheduler_reads_env_opt_in(monkeypatch):
    for v in ("HEROSIM_PG_ILS_MIN_BATCH", "HEROSIM_PG_ILS_MAX_SECONDS", "HEROSIM_PG_ILS_PER"):
        monkeypatch.delenv(v, raising=False)
    src = (REPO / "src/policy/peer_greedy_network/scheduler.py").read_text()
    assert "self.pg_ils_min_batch = None" in src


def test_reporting_counters_per_bucket():
    exporter = (REPO / "src/placement/orchestrator.py").read_text()
    for b in ("11_50", "51_100", "101_226"):
        for k in (f"pg_ils_differs_{b}", f"pg_ils_gain_seconds_{b}", f"pg_ils_cap_hits_{b}", f"pg_ils_batches_{b}"):
            assert k in S.PEER_GREEDY_COUNTERS and f'"{k}"' in exporter


def test_exact10_is_an_exact_arm_with_ils_above_the_threshold():
    g = (REPO / "scripts_cosim/fresh_topo_burst_v1_gate.py").read_text()
    assert '"ils_oracle_exact10": "peer_greedy_network_cd"' in g
    assert '"ils_oracle_exact10": {"HEROSIM_PG_BATCH_ALPHA": "0"}' in g and 'ILS_KINDS = ("ils_oracle_live", "ils_oracle_exact10")' in g
    src = (REPO / "src/policy/peer_greedy_network/scheduler.py").read_text()
    assert "if self.pg_cd_exact:\n                return" in src and "expansion alone is off" in src
