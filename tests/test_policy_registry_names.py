"""Every policy the simulation registry can build must have a short name in model.scheduling_strategies.

executor.execute_sim looks the strategy up there before the simulation starts; local_first_network was registered in
simulation.py only, and all 228 client_local_v1 runs of it died with a KeyError at start-up (2026-10-05).
"""
import re
from pathlib import Path

from src.placement.model import scheduling_strategies

REPO = Path(__file__).resolve().parents[1]


def test_every_registered_policy_has_a_short_name():
    src = (REPO / "src/placement/simulation.py").read_text()
    block = src[src.index("    policies: Dict["):]
    block = block[: block.index("\n    }\n")]
    keys = re.findall(r'^\s+"([a-z0-9_]+)": \(', block, flags=re.M)
    assert len(keys) > 20, keys
    missing = [k for k in keys if k not in scheduling_strategies]
    assert not missing, f"registered in simulation.py but absent from model.scheduling_strategies: {missing}"
