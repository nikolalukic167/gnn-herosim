import json
import random

import pytest

from src.generate_infrastructure import build_core_backbone
from src.placement import fidelity_replay as F


def space(bw=None, core=None):
    bb = {"n_core": 4, "attach_degree": 1, "chord_count": 0, "rng_stream": "independent_v1"}
    if bw is not None:
        bb["bandwidth_mbps"] = bw
    if core is not None:
        bb["core_bandwidth_mbps"] = core
    return {"network": {"backbone": bb} if bw is not None else {}}


def test_explicit_backbone_beats_env_and_silent_config_takes_it(monkeypatch):
    monkeypatch.setenv("HEROSIM_LINK_BANDWIDTH_MBPS", "62.5")
    assert F.effective_backbone(space(1000.0)) == {"bandwidth_mbps": 1000.0, "core_bandwidth_mbps": 1000.0}
    assert F.effective_backbone(space()) == {"bandwidth_mbps": 62.5, "core_bandwidth_mbps": 62.5}
    monkeypatch.delenv("HEROSIM_LINK_BANDWIDTH_MBPS")
    assert F.effective_backbone(space()) is None


def test_core_links_differ_across_the_ladder_and_match_the_stamp():
    nodes = [{"node_name": n} for n in ("client_node0", "node0", "node1")]
    maps = {"client_node0": {"node0": 0.1, "node1": 0.1}, "node0": {}, "node1": {}}
    seen = {}
    for bw in (1000.0, 62.5):
        topo = build_core_backbone(maps, nodes, space(bw), random.Random(1), seed=7)
        core = {k: v["bandwidth_mbps"] for k, v in topo["links"].items() if k.startswith("core")}
        assert core and set(core.values()) == {bw}
        assert topo["params"]["core_bandwidth_mbps"] == F.effective_backbone(space(bw))["core_bandwidth_mbps"]
        seen[bw] = core
    assert seen[1000.0] != seen[62.5]


def test_replay_raises_on_a_backbone_that_differs_from_the_stamp(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps(space(62.5)))
    spec = {"cell_config": str(cfg), "backbone": {"bandwidth_mbps": 1000.0, "core_bandwidth_mbps": 1000.0}}
    with pytest.raises(ValueError, match="backbone at replay"):
        F.FidelityReplay(dict(spec, snapshot={"fidelity": {}}, sim_input=str(tmp_path)), [], {})
