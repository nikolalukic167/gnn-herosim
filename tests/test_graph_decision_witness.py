import itertools
import json

import numpy as np
import pytest

from scripts_cosim import graph_decision_witness as witness


@pytest.fixture
def cfg():
    return json.loads((witness.ROOT / 'experiments/graph_decision_witness_v1.json').read_text())


@pytest.mark.parametrize('family', ['chains', 'bridge', 'loop'])
def test_local_ambiguity_but_distant_messages_distinguish(cfg, family):
    edges = [witness.edges_for(cfg, family, v) for v in (0, 1)]
    assert witness.local_signature(cfg, edges[0]) == witness.local_signature(cfg, edges[1])
    domains = witness.domains_for(cfg)
    plans = list(itertools.product(*domains))
    for variant, graph in enumerate(edges):
        q = [min(witness.energy(cfg, graph, p) for p in plans if p[0] == host) for host in (0, 1)]
        assert q[variant] < q[1 - variant]
        columns = witness.beliefs(cfg, graph, domains, 3)[0]
        assert np.argmin(columns) == variant


@pytest.mark.parametrize('family', ['chains', 'bridge', 'loop'])
@pytest.mark.parametrize('variant', [0, 1])
def test_cut_matches_exhaustive_feasible_optimum(cfg, family, variant):
    edges = witness.edges_for(cfg, family, variant)
    domains = witness.domains_for(cfg)
    plan = witness.cut_plan(cfg, edges)
    assert all(p in d for p, d in zip(plan, domains))
    optimum = min(witness.energy(cfg, edges, p) for p in itertools.product(*domains))
    assert witness.energy(cfg, edges, plan) == pytest.approx(optimum)


def test_anchors_are_physical_compatibility_not_future_placements(cfg):
    infra, inputs = witness.substrate(cfg)
    assert 'forced_placements' not in infra
    for tid, choices in enumerate(witness.domains_for(cfg)):
        hardware = inputs['task_types'][f'w{tid}']['platforms'][0]
        actual = [host for host in (0, 1) if hardware in infra['nodes'][host + 1]['platforms']]
        assert actual == choices


def test_unknown_control_fails(cfg):
    with pytest.raises(ValueError, match='unknown control'):
        witness.control_plan(cfg, witness.edges_for(cfg, 'chains', 0), 'typo')
