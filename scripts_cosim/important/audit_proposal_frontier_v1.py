"""Check the frozen inputs and complete live traces for one proposal frontier seed."""

import argparse
import hashlib
import json
import math
from pathlib import Path


ARMS = ('gnn', 'mpoff', 'handmulti')
KEYS = {
    'gnn': {'new', 'mpoff', 'hand_s4'},
    'mpoff': {'new', 'mpoff', 'hand_s4'},
    'handmulti': {'mpoff', 'hand_s0.5', 'hand_s1', 'hand_s2', 'hand_s4', 'hand_s8', 'hand_s16'},
}
MODELS = {
    'gnn': Path('/tmp/g32train/models/g32-sampled-slate-v1-gnn-seed1-final.pt'),
    'mpoff': Path('/tmp/g32train/models/g32-sampled-slate-v1-mpoff-seed1-final.pt'),
    'handmulti': Path('/tmp/jb2-mpoff-seed1.pt'),
}
OLD = Path('/tmp/jb2-mpoff-seed1.pt')
WORKLOAD = Path('/tmp/fixed-g32-weighted_bridge-5120.json')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def audit_arm(seed, arm, root):
    prefix = root / f'seed-{seed}-{arm}'
    result = json.loads(prefix.with_suffix('.json').read_text())
    trace = [json.loads(line) for line in Path(f'{prefix}-plans.jsonl').open()]
    manifest = json.loads(Path(f'{prefix}-manifest.json').read_text())
    check(result['status'] == 'success', f'{arm}: failed result')
    check(result['policy'] == 'gnn_seeded_cd', f'{arm}: wrong policy')
    check(int(result['placement_seed']) == seed, f'{arm}: wrong placement seed')
    check(result['num_tasks'] == 5120, f'{arm}: wrong task count')
    check(len(trace) == 160, f'{arm}: expected 160 trace groups')
    check(Path(result['config_file']).resolve() == Path(f'/tmp/fixed-g32-config-{seed}.json').resolve(), f'{arm}: wrong config')
    check(Path(result['workload_file']).resolve() == WORKLOAD.resolve(), f'{arm}: wrong workload')
    check(result['stats']['scaleEvents'] == [], f'{arm}: scaling occurred')
    peer_time = float(result['stats']['totalPeerExchangeTime'])
    check(math.isfinite(peer_time) and peer_time >= 0, f'{arm}: invalid peer exchange accounting')
    provenance = result['run_provenance']
    env = provenance['env']
    expected = {
        'GNN_MODEL_PATH': str(MODELS[arm]),
        'GNN_DISABLE_MESSAGE_PASSING': '0' if arm == 'gnn' else '1',
        'GNN_MP_PLATFORM_EDGES_OFF': '0',
        'GNN_BATCH_SIZE': '32',
        'GNN_BATCH_BY_PEER_GROUP': '1',
        'GNN_PREFIX_ALPHA_KEY': 'inf',
        'HEROSIM_PEER_EXCHANGE': '1',
        'HEROSIM_SERVER_ONLY_REPLICAS': '1',
        'PARTIAL_STATE_CONTRACT': 'partial_state_v3',
        'PARTIAL_STATE_PEER_MASS': '1',
        'INFERENCE_FEATURE_LAYOUT': 'dim22',
        'TOPOLOGY_FEATURE_CONTRACT': 'src_index_v0',
        'QUEUE_FEATURE_CONTRACT': 'legacy_v0',
        'HEROSIM_GNN_DEVICE': 'cpu',
    }
    for key, value in expected.items():
        check(env.get(key) == value, f'{arm}: {key} mismatch')
    check(provenance['warmth_physics'] == 'node_disk_v2', f'{arm}: wrong warmth physics')
    check(provenance['python_env']['thread_env'] == {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'PYTHONHASHSEED': '0'}, f'{arm}: wrong thread env')
    check(provenance['python_env']['gnn_serving_device'] == 'cpu', f'{arm}: wrong serving device')
    check(result['rtt_overview']['num_tasks'] == 5120, f'{arm}: incomplete RTT overview')
    check(result['rtt_overview']['tasks_with_inference'] == 5120, f'{arm}: incomplete inference accounting')
    all_ids = set()
    wins = {key: 0 for key in KEYS[arm]}
    unique = 0
    for group, row in enumerate(trace):
        ids = row['ids']
        check(len(ids) == 32 and len(set(ids)) == 32, f'{arm}: group {group} has wrong IDs')
        check(not (set(ids) & all_ids), f'{arm}: duplicate task IDs at group {group}')
        all_ids.update(ids)
        check(set(row['scores']) == KEYS[arm] == set(row['proposals']), f'{arm}: group {group} wrong candidate keys')
        check(row['winner'] == min(row['scores'], key=lambda key: (row['scores'][key], key)), f'{arm}: group {group} wrong winner')
        wins[row['winner']] += 1
        plans = []
        for key, plan in row['proposals'].items():
            check(set(plan) == {str(i) for i in range(32)}, f'{arm}: group {group} incomplete {key} plan')
            check(math.isfinite(float(row['scores'][key])), f'{arm}: group {group} nonfinite score')
            plans.append(tuple(tuple(plan[str(i)]) for i in range(32)))
        unique += len(set(plans))
    check(len(all_ids) == 5120, f'{arm}: incomplete task identity coverage')
    files = [Path(f'/tmp/fixed-g32-config-{seed}.json'), WORKLOAD, MODELS[arm], MODELS[arm].with_suffix('.contract.json'), OLD, OLD.with_suffix('.contract.json')]
    files += [Path('scripts_cosim/important/proposal_frontier_v1.py'), Path('scripts_cosim/important/run_proposal_frontier_v1.sh')]
    recorded = {}
    for line in Path(f'{prefix}-inputs.sha256').read_text().splitlines():
        sha, path = line.split(maxsplit=1)
        recorded[str(Path(path))] = sha
    check(set(recorded) == {str(path) for path in files}, f'{arm}: input manifest paths differ')
    for path in files:
        check(recorded[str(path)] == digest(path), f'{arm}: input changed: {path}')
    return {
        'arm': arm,
        'total_rtt_s': result['total_rtt'],
        'inference_s': result['rtt_overview']['total_inference_time_s'],
        'mean_inference_ms': 1000 * result['rtt_overview']['average_inference_time_s'],
        'unique_proposals': unique,
        'winner_counts': wins,
        'peer_exchange_s': peer_time,
        'code': {key: provenance['code'][key] for key in ('commit', 'diff_sha256')},
        'manifest': manifest,
        'source_sha256': recorded,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type=int)
    parser.add_argument('--root', type=Path, default=Path('/tmp/proposal-frontier-v1'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    arms = {arm: audit_arm(args.seed, arm, args.root) for arm in ARMS}
    check(len({json.dumps(row['manifest'], sort_keys=True) for row in arms.values()}) == 1, 'fixed replica manifests differ')
    check(len({json.dumps(row['code'], sort_keys=True) for row in arms.values()}) == 1, 'code fingerprints differ')
    report = {'contract': 'proposal_frontier_v1_audit', 'seed': args.seed, 'arms': arms, 'status': 'pass'}
    for row in arms.values():
        row.pop('manifest')
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + '\n')
    print(rendered)


if __name__ == '__main__':
    main()
