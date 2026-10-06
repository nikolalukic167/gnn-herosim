#!/usr/bin/env python3
"""Matched downstream-structure screen; paper costs, not a live simulator gate.

Reuses peer_affinity_probe's source costs/candidates/count-shaped sharing. Exchange
is charged at BOTH endpoints, as the runtime does. Exhaustive completion measures
the cost of choosing task 0's candidate, not a clairvoyant deployable policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts_cosim.peer_affinity_probe import Paper, Source


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cycle_edges(cycle):
    if sorted(cycle) != list(range(len(cycle))):
        raise ValueError("cycle must visit each task exactly once")
    return sorted(tuple(sorted((a, b))) for a, b in zip(cycle, cycle[1:] + cycle[:1]))


def factors(paper, edges, payload):
    """Undirected factors count both tasks' transfer and one sharing pair."""
    peer, full = {}, {}
    edge_set = set(edges)
    for i in range(paper.k):
        for j in range(i + 1, paper.k):
            ni, nj = paper.node[i][:, None], paper.node[j][None, :]
            x = np.zeros((paper.n_cand, paper.n_cand))
            if (i, j) in edge_set and payload > 0:
                x = 2 * (payload * paper.src.per_byte[ni, nj] + paper.src.latency[ni, nj])
                peer[i, j] = x
            sharing = (paper.plat[i][:, None] == paper.plat[j][None, :]) * paper.q[paper.plat[i]][:, None]
            if np.any(x + sharing):
                full[i, j] = x + sharing
    return peer, full


def plan_cost(cost, pair, plan):
    value = sum(cost[i, c] for i, c in enumerate(plan))
    return float(value + sum(mat[plan[i], plan[j]] for (i, j), mat in pair.items()))


def sweep(cost, pair, plans):
    out = cost[np.arange(len(cost)), plans].sum(1)
    for (i, j), mat in pair.items():
        out += mat[plans[:, i], plans[:, j]]
    return out


def messages(cost, pair, rounds):
    """Hand-computed min-sum candidate columns; no labels or learned parameters.

    Synchronous, non-backtracking updates. On cycles these are a heuristic, not an
    exact solver. Cost is O(rounds * directed_edges * candidates**2).
    """
    directed = {}
    neighbors = [[] for _ in cost]
    for (i, j), mat in pair.items():
        directed[i, j], directed[j, i] = mat, mat.T
        neighbors[i].append(j)
        neighbors[j].append(i)
    msg = {key: np.zeros(cost.shape[1]) for key in directed}
    for _ in range(rounds):
        new = {}
        for (i, j), mat in directed.items():
            cavity = cost[i].copy()
            for h in neighbors[i]:
                if h != j:
                    cavity += msg[h, i]
            value = (cavity[:, None] + mat).min(0)
            new[i, j] = value - value.min()
        msg = new
    belief = cost.copy()
    for (i, j), value in msg.items():
        belief[j] += value
    return belief


def greedy(cost, pair, lookahead=None):
    plan = []
    for i in range(len(cost)):
        score = cost[i].copy()
        for j, c in enumerate(plan):
            if (j, i) in pair:
                score += pair[j, i][c]
        for j in range(i + 1, len(cost)):
            if lookahead is not None and (i, j) in lookahead:
                score += lookahead[i, j].mean(1)
        plan.append(int(score.argmin()))
    return plan


def coordinate_descent(cost, pair, initial, passes):
    plan = list(initial)
    for _ in range(passes):
        for i in range(len(cost)):
            current = plan_cost(cost, pair, plan)
            best, value = plan[i], current
            for c in range(cost.shape[1]):
                trial = list(plan)
                trial[i] = c
                v = plan_cost(cost, pair, trial)
                if v < value - 1e-10:
                    best, value = c, v
            plan[i] = best
    return plan


def read_variant(paper, edges, config, payload=None):
    payload = config['payload_bytes'] if payload is None else payload
    peer, pair = factors(paper, edges, payload)
    values = sweep(paper.cost, pair, paper.P)
    q = np.array([values[paper.P[:, 0] == c].min() for c in range(paper.n_cand)])
    optimum = float(q.min())
    if not np.isfinite(values).all() or optimum <= 0:
        raise ValueError("invalid objective")
    mass = paper.cost.copy()
    for (i, j), mat in peer.items():
        mass[i] += mat.mean(1)
        mass[j] += mat.mean(0)
    beliefs = {f'peer_min_sum_{r}': messages(paper.cost, peer, r) for r in config['message_rounds']}
    beliefs['full_min_sum_8'] = messages(paper.cost, pair, 8)
    plans = {'immediate': greedy(paper.cost, pair),
             'peer_mass': greedy(paper.cost, pair, lookahead=peer)}
    plans.update({name: v.argmin(1).tolist() for name, v in beliefs.items()})
    plans['cd3'] = coordinate_descent(paper.cost, pair, plans['immediate'], config['cd_passes'])
    # Every start is available before seeing the exhaustive labels. Selecting by
    # the same explicit paper objective is search, not validation-set selection.
    starts = list(plans.values())
    plans['cd3_multistart'] = min(
        (coordinate_descent(paper.cost, pair, p, config['cd_passes']) for p in starts),
        key=lambda p: (plan_cost(paper.cost, pair, p), tuple(p)),
    )
    controls = {}
    for name, plan in plans.items():
        controls[name] = {
            'root_candidate': int(plan[0]),
            'root_regret_pct': float(100 * (q[plan[0]] / optimum - 1)),
            'full_plan_regret_pct': float(100 * (plan_cost(paper.cost, pair, plan) / optimum - 1)),
            'plan': list(map(int, plan)),
        }
    optimal = np.flatnonzero(values <= optimum + 1e-9)
    exchange = np.zeros(len(values))
    for (i, j), mat in peer.items():
        exchange += mat[paper.P[:, i], paper.P[:, j]]
    best = int(optimal[0])
    return {
        'edges': edges, 'optimum': optimum, 'root_completion_costs': q.tolist(),
        'optimal_root_candidates': np.flatnonzero(q <= optimum + 1e-9).tolist(),
        'root_peer_mass': mass[0].tolist(),
        'root_hand_columns': {k: v[0].tolist() for k, v in beliefs.items()},
        'controls': controls, 'enumerated_plans': len(values),
        'optimal_plan': paper.P[best].tolist(),
        'optimal_exchange_min': float(exchange[optimal].min()),
        'optimal_exchange_max': float(exchange[optimal].max()),
        'zero_exchange_optimum_exists': bool(exchange[optimal].min() <= 1e-9),
        'selected_optimum_decomposition': {'base': float(paper.BASE[best]),
                                          'sharing': float(paper.SHARE[best]),
                                          'exchange': float(exchange[best])},
        '_sweep': values,
    }


def audit_live_block():
    """Exercise the actual prefix attachment on a lone arrived task."""
    import torch
    from src.policy.gnn.prefix_serving import PrefixServingOptions, attach_live_prefix_block

    topology = {'routes': {'node0': {'node1': ['node0', 'node1']},
                           'node1': {'node0': ['node1', 'node0']}},
                'links': {'node0|node1': {'bandwidth_mbps': 1000.0}}}
    fabric = SimpleNamespace(link_topology=topology)
    nodes = [SimpleNamespace(node_name=f'node{i}', fabric=fabric,
                             network_map={f'node{1-i}': 0.01}) for i in range(2)]
    opts = PrefixServingOptions(alpha_key='inf', allow_replica_reuse=True, relax_on_stuck=True,
                                task_type_vocab=('dnn1', 'dnn2', 'rf', 'cnn'),
                                peer_mass=True, mp_peer_edges=True, partial_state_contract='partial_state_v3')
    task = SimpleNamespace(id=0, node_name='node0', application=None,
                           type={'name': 'dnn1', 'memoryRequirements': {'cpu': 1.0}})
    graph = SimpleNamespace(n_tasks=1, task_logit_to_placement={0: [(0, 10), (1, 11)]},
                            queue_key_to_platform_meta={i: {'platform_id': 10+i, 'platform_type': 'cpu',
                                                           'node_id': i, 'node_name': f'node{i}'} for i in range(2)})
    prior = {key: os.environ.get(key) for key in ('PARTIAL_STATE_CONTRACT', 'PARTIAL_STATE_PEER_MASS')}
    try:
        os.environ.update(PARTIAL_STATE_CONTRACT='partial_state_v3', PARTIAL_STATE_PEER_MASS='1')
        diagnostics = attach_live_prefix_block(graph, [task], nodes=nodes,
                                               peer_table={0: {1: 200e6}, 1: {0: 200e6, 2: 200e6}, 2: {1: 200e6}},
                                               options=opts)
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return {'diagnostics': diagnostics, 'peer_edge_count': int(graph.peer_edge_index.shape[1]),
            'prefix_peer_pair_count': len(graph.partial_state_ctx['peer_pairs']),
            'dtype': str(torch.get_default_dtype()),
            'future_peer_graph_visible_to_current_decoder': bool(graph.peer_edge_index.numel())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=ROOT / 'experiments/peer_lookahead_screen_v1.json')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    started = time.monotonic()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    observability = audit_live_block()
    config = json.loads(args.config.read_text())
    corpus = ROOT / config['source_corpus']
    selected = sorted(corpus.glob('ds_*'))[::config['source_stride']][:config['source_count']]
    if len(selected) != config['source_count']:
        raise ValueError('source selection incomplete')
    records, enumerations = [], {}
    for idx, ds in enumerate(selected):
        src = Source(ds)
        if src.max_asymmetry > 1e-6:
            raise ValueError(f'{ds}: asymmetric routes invalidate symmetric paper factors')
        paper = Paper(src, k=config['tasks'], n_cand=config['candidates'], partners=2,
                      seed=config['paper_seed_start'] + idx)
        paper.enumerate()
        variants = [read_variant(paper, cycle_edges(c), config) for c in config['cycles']]
        enumerations['plans'] = paper.P
        for v, result in enumerate(variants):
            enumerations[f'source_{idx}_variant_{v}'] = result.pop('_sweep')
        a, b = variants
        for key in ('root_peer_mass',):
            if not np.allclose(a[key], b[key], atol=1e-10, rtol=0):
                raise AssertionError(f'{ds}: matched feature changed: {key}')
        for name in ('peer_min_sum_1', 'peer_min_sum_2'):
            if not np.allclose(a['root_hand_columns'][name], b['root_hand_columns'][name], atol=1e-10, rtol=0):
                raise AssertionError(f'{ds}: two-hop match failed')
        zeros = [read_variant(paper, cycle_edges(c), config, payload=0) for c in config['cycles']]
        if not np.allclose(zeros[0]['_sweep'], paper.BASE + paper.SHARE, atol=1e-9, rtol=0):
            raise AssertionError('pair factors do not reproduce the source sharing objective')
        for v, result in enumerate(zeros):
            enumerations[f'source_{idx}_exchange_off_{v}'] = result.pop('_sweep')
        if zeros[0]['root_completion_costs'] != zeros[1]['root_completion_costs']:
            raise AssertionError('exchange-off control differs')
        regrets = [100 * (np.array(v['root_completion_costs']) / v['optimum'] - 1) for v in variants]
        records.append({
            'source': str(ds.relative_to(ROOT)), 'paper_seed': paper.seed,
            'source_hashes': {f: sha256(ds / f) for f in ('infrastructure.json', 'workload.json', 'placements/placements.jsonl')},
            'costs': paper.cost.tolist(), 'candidates': paper.cands,
            'node_indices': paper.node.tolist(), 'platform_indices': paper.plat.tolist(),
            'task_types': paper.types,
            'sharing_coefficients': paper.q.tolist(),
            'variants': variants,
            'disjoint_optimal_root_sets': not bool(set(a['optimal_root_candidates']) & set(b['optimal_root_candidates'])),
            'matched_blind_best_mean_regret_pct': float(np.mean(regrets, axis=0).min()),
            'exchange_off': zeros[0],
        })
    names = records[0]['variants'][0]['controls']
    rows = [v for rec in records for v in rec['variants']]
    summary = {'source_pairs': len(records), 'variants': len(rows),
               'disjoint_optimal_root_pairs': sum(r['disjoint_optimal_root_sets'] for r in records),
               'variants_with_zero_exchange_optimum': sum(v['zero_exchange_optimum_exists'] for v in rows),
               'pairs_with_changed_optimal_cost': sum(abs(r['variants'][0]['optimum'] - r['variants'][1]['optimum']) > 1e-9 for r in records),
               'matched_blind_best_mean_regret_median_pct': float(np.median([r['matched_blind_best_mean_regret_pct'] for r in records])),
               'controls': {}}
    for name in names:
        root = [v['controls'][name]['root_regret_pct'] for v in rows]
        full = [v['controls'][name]['full_plan_regret_pct'] for v in rows]
        summary['controls'][name] = {
            'root_regret_median_pct': float(np.median(root)),
            'root_regret_mean_pct': float(np.mean(root)),
            'root_regret_max_pct': float(max(root)),
            'root_regret_above_bar_count': sum(x >= config['material_regret_pct'] for x in root),
            'full_plan_regret_median_pct': float(np.median(full)),
            'full_plan_regret_above_bar_count': sum(x >= config['material_regret_pct'] for x in full),
        }
    sweep_path = args.out.with_suffix('.sweeps.npz')
    np.savez_compressed(sweep_path, **enumerations)
    dependencies = ('scripts_cosim/peer_affinity_probe.py', 'scripts_cosim/score_route_b_contention.py',
                    'src/policy/gnn/prefix_serving.py', 'src/policy/tabular/reduced_features.py',
                    'src/placement/infrastructure.py', 'data/nofs-ids/task-types.json')
    output = {'config': config, 'config_sha256': sha256(args.config), 'script_sha256': sha256(__file__),
              'dependency_hashes': {p: sha256(ROOT / p) for p in dependencies},
              'exhaustive_paper_sweeps': {'path': str(sweep_path), 'sha256': sha256(sweep_path)},
              'wall_seconds': time.monotonic() - started,
              'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'working_tree_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)),
              'observability': observability, 'summary': summary, 'sources': records}
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'observability': output['observability'], 'summary': summary}, indent=2))


if __name__ == '__main__':
    main()
