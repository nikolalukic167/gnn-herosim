"""HEROSIM_TRANSFER_MODEL: store-and-forward (default) vs pipelined transmission over a multi-hop route.

Pins:
- store_forward charges n_hops x size / bottleneck (every run before 2026-10-07), pipelined charges size / bottleneck;
- the rules' estimate (Platform._payload_transfer_time, which peer_greedy_network calls) and the GNN's exchange
  feature (transmission_hops) move together, so no arm reads a different physics from the one charged;
- an unknown model fails loud.
"""
from __future__ import annotations

import pytest

from src.placement.infrastructure import Platform
from src.placement.network_fabric import transmission_hops

MB = 1024 * 1024


class FakeFabric:
    def __init__(self, hops):
        self._hops = hops

    def hops(self, src, dst):
        return self._hops


class FakeNode:
    def __init__(self, fabric):
        self.node_name = "node0"
        self.fabric = fabric
        self.network = {"bandwidth": 1000.0}


class FakePlatform:
    def __init__(self, node):
        self.node = node


def _charge(hops, payload):
    return Platform._payload_transfer_time(FakePlatform(FakeNode(FakeFabric(hops))), "node1", payload)


ROUTE = [("a|b", 1000.0), ("b|c", 500.0), ("c|d", 1000.0), ("d|e", 1000.0), ("e|f", 1000.0)]


def test_store_forward_is_the_default_and_multiplies_by_hops(monkeypatch):
    monkeypatch.delenv("HEROSIM_TRANSFER_MODEL", raising=False)
    assert _charge(ROUTE, 200 * MB) == pytest.approx(5 * 200 / 500)
    assert transmission_hops(5) == 5


def test_pipelined_charges_one_transmission_at_the_bottleneck(monkeypatch):
    monkeypatch.setenv("HEROSIM_TRANSFER_MODEL", "pipelined")
    assert _charge(ROUTE, 200 * MB) == pytest.approx(200 / 500)
    assert transmission_hops(5) == 1.0 and transmission_hops(0) == 0.0


def test_feature_and_physics_agree_under_both_models(monkeypatch):
    for model in ("store_forward", "pipelined"):
        monkeypatch.setenv("HEROSIM_TRANSFER_MODEL", model)
        feature_seconds_per_byte = transmission_hops(len(ROUTE)) / (500.0 * MB)
        assert _charge(ROUTE, 123 * MB) == pytest.approx(feature_seconds_per_byte * 123 * MB)


def test_unknown_model_fails_loud(monkeypatch):
    monkeypatch.setenv("HEROSIM_TRANSFER_MODEL", "cut_through")
    with pytest.raises(ValueError, match="HEROSIM_TRANSFER_MODEL"):
        _charge(ROUTE, MB)
