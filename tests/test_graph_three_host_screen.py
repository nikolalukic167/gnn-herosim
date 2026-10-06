import json
import io

import numpy as np
import pytest

from scripts_cosim import graph_three_host_screen as screen
from scripts_cosim import graph_decision_witness as witness


@pytest.fixture
def cfg():
    return json.loads((witness.ROOT / 'experiments/graph_three_host_screen_v1.json').read_text())


@pytest.mark.parametrize('seed', [46200, 46201, 46215])
def test_vectorized_search_matches_scalar_physics(cfg, seed):
    edges = screen.make_edges(cfg, seed)
    plans, values = screen.exhaustive(cfg, edges)
    assert len(plans) == 729
    assert len({tuple(p) for p in plans}) == 729
    assert np.allclose(values, [witness.energy(cfg, edges, p) for p in plans], rtol=0, atol=1e-10)
    assert np.all(plans[:, 6:] == [0, 1, 2])


def test_multistart_preserves_pins_and_never_worsens_best_start(cfg):
    edges = screen.make_edges(cfg, 46201)
    starts = [witness.control_plan(cfg, edges, name) for name in ['immediate_paper', 'peer_mass', 'min_sum_3']]
    result = screen.coordinate_descent(cfg, edges, starts)
    assert result[6:] == [0, 1, 2]
    assert witness.energy(cfg, edges, result) <= min(witness.energy(cfg, edges, p) for p in starts) + 1e-10


def test_three_host_substrate_domains_and_links(cfg):
    infra, inputs = witness.substrate(cfg)
    hosts = infra['nodes'][1:]
    assert len(hosts) == 3
    for i, domain in enumerate(witness.domains_for(cfg)):
        hardware = inputs['task_types'][f'w{i}']['platforms'][0]
        assert [h for h, node in enumerate(hosts) if hardware in node['platforms']] == domain
    for h, node in enumerate(hosts):
        assert node['network_map'] == {'client_node0': 0., **{f'node{k}': cfg['peer_latency_s'] for k in range(3) if k != h}}


def test_live_nine_task_group_counts_every_task(cfg, monkeypatch):
    monkeypatch.setenv('HEROSIM_PEER_EXCHANGE', '1')
    monkeypatch.setenv('HEROSIM_PG_DRAIN_CONTRACT', 'availability_v2')
    monkeypatch.setenv('SIM_FORCE_FULL_STATS', '1')
    infra, inputs = witness.substrate(cfg)
    edges = screen.make_edges(cfg, 46201)
    plans, values = screen.exhaustive(cfg, edges)
    for plan in (plans[0], plans[int(np.argmin(values))]):
        stats = witness.live(cfg, infra, inputs, edges, cfg['arrival_gap_s'], plan, io.StringIO())
        assert len(stats['tasks']) == 9
        total = sum(t['doneTime'] - t['dispatchedTime'] for t in stats['tasks'])
        assert stats['total_rtt'] == pytest.approx(total)
        assert stats['group_rtt'] == pytest.approx(total)
        assert stats['group_rtt'] > sum(t['elapsedTime'] for t in stats['tasks'] if t['taskId'] < 8)
        assert stats['exchange'] == pytest.approx(witness.energy(cfg, edges, plan))
