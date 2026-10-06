import hashlib
import json
import os
from pathlib import Path
from src.policy.gnn.orchestrator import GNNOrchestrator
from src.policy.gnn.autoscaler import KnativeAutoscaler
from src.executesimulation import main

original_initialize = GNNOrchestrator.initialize_state

def fixed_initialize(self):
    state = original_initialize(self)
    if any(state.replicas.values()):
        raise RuntimeError('fixed-replica control expected an empty initial replica set')
    servers = sorted(
        (node for node in self.nodes.items if not node.node_name.startswith('client_node')),
        key=lambda node: node.id,
    )
    claimed = set()
    for task_name, task_type in sorted(self.data.task_types.items()):
        for node in servers:
            compatible = [platform for platform in node.platforms.items
                          if platform.type['shortName'] in task_type['platforms']
                          and platform.type['shortName'] in task_type['executionTime']]
            if not compatible:
                continue
            platform = min(
                compatible,
                key=lambda item: hashlib.sha256(
                    f'9106:{task_name}:{node.node_name}:{item.id}'.encode()
                ).digest(),
            )
            pair = (node, platform)
            state.replicas[task_name].add(pair)
            state.scheduler_state.average_contention[task_name][(node.id, platform.id)] = 0.0
            if pair not in claimed:
                if platform not in state.available_resources[node]:
                    raise RuntimeError(f'platform already claimed: {node.node_name}:{platform.id}')
                state.available_resources[node].remove(platform)
                node.available_platforms -= 1
                claimed.add(pair)
            if not platform.initialized.triggered:
                platform.initialized.succeed()
        if not state.replicas[task_name]:
            raise RuntimeError(f'no fixed replica for {task_name}')
    for node, platform in claimed:
        kind = platform.type['shortName']
        memory = max(
            self.data.task_types[name]['memoryRequirements'][kind]
            for name, replicas in state.replicas.items()
            if (node, platform) in replicas
        )
        node.available_memory -= memory
        if node.available_memory < 0:
            raise RuntimeError(f'fixed replicas exceed memory on {node.node_name}')
    manifest = {name: sorted([f'{node.node_name}:{platform.id}' for node, platform in replicas])
                for name, replicas in state.replicas.items()}
    uncovered = []
    for source in self.nodes.items:
        if not source.node_name.startswith('client_node'):
            continue
        for task_name, replicas in state.replicas.items():
            if not any(node.node_name in source.network_map for node, _ in replicas):
                uncovered.append((source.node_name, task_name))
    if uncovered:
        raise RuntimeError(f'fixed-replica pool does not cover every source: {uncovered}')
    path = os.environ['FIXED_REPLICA_MANIFEST_PATH']
    with open(path, 'w') as out:
        json.dump(manifest, out, indent=2, sort_keys=True)
    print('[FIXED REPLICAS]', {name: len(items) for name, items in manifest.items()}, flush=True)
    if os.environ.get('FIXED_REPLICA_PREFLIGHT') == '1':
        raise SystemExit(0)
    return state

def no_scaling(self, system_state, task_type):
    if False:
        yield
    return {kind['shortName']: 0.0 for kind in self.data.platform_types.values()}

GNNOrchestrator.initialize_state = fixed_initialize
KnativeAutoscaler.scaling_level = no_scaling
from src.policy.gnn import scheduler as gnn_scheduler
gnn_scheduler.MAX_BATCH_SIZE_FOR_PREFIX_GNN = 32
from src.policy.gnn.scheduler import GNNScheduler
from src.policy.peer_greedy_network.gnn_seeded_cd import GNNSeededCDScheduler
from src.policy.peer_greedy_network.scheduler import PeerGreedyNetworkCDScheduler
from src.placement.live_audit import _task_payload
from src.executesimulation import load_gnn_model

_original_set_models = GNNSeededCDScheduler.set_models
_original_learned = GNNSeededCDScheduler._prefix_inference
_original_hand = PeerGreedyNetworkCDScheduler._prefix_inference

def _set_models_with_mpoff(self, models):
    _original_set_models(self, models)
    old_path = Path(os.environ.get('FRONTIER_OLD_MPOFF_PATH', '/tmp/jb2-mpoff-seed1.pt'))
    if not old_path.is_file() or not old_path.with_suffix('.contract.json').is_file():
        raise RuntimeError(f'missing old MP-OFF checkpoint or contract: {old_path}')
    before_mp = os.environ.get("GNN_DISABLE_MESSAGE_PASSING")
    before_platform = os.environ.get("GNN_MP_PLATFORM_EDGES_OFF")
    os.environ["GNN_DISABLE_MESSAGE_PASSING"] = "1"
    os.environ["GNN_MP_PLATFORM_EDGES_OFF"] = "0"
    try:
        mp_model, mp_device = load_gnn_model(old_path)
    finally:
        if before_mp is None:
            os.environ.pop("GNN_DISABLE_MESSAGE_PASSING", None)
        else:
            os.environ["GNN_DISABLE_MESSAGE_PASSING"] = before_mp
        if before_platform is None:
            os.environ.pop("GNN_MP_PLATFORM_EDGES_OFF", None)
        else:
            os.environ["GNN_MP_PLATFORM_EDGES_OFF"] = before_platform
    if str(mp_device) != str(models['device']):
        raise RuntimeError('matched CPU device required')
    self._selector_mp_model = mp_model
    self._selector_mp_options = getattr(mp_model, 'prefix_serving_options')

def _selector_prefix(self, batch_tasks, system_state, queue_snapshot, temporal_state):
    arm = os.environ['FRONTIER_ARM']
    if arm not in {'gnn', 'mpoff', 'handmulti'}:
        raise RuntimeError(f'unknown frontier arm: {arm}')
    original_model, original_options = self.gnn_model, self._prefix_options
    proposals = {}
    if arm != 'handmulti':
        proposals['new'] = _original_learned(self, batch_tasks, system_state, queue_snapshot, temporal_state)
    self.gnn_model, self._prefix_options = self._selector_mp_model, self._selector_mp_options
    try:
        proposals['mpoff'] = _original_learned(self, batch_tasks, system_state, queue_snapshot, temporal_state)
    finally:
        self.gnn_model, self._prefix_options = original_model, original_options
    scales = (0.5, 1, 2, 4, 8, 16) if arm == 'handmulti' else (4,)
    old_scale = self.pg_exchange_scale
    try:
        for scale in scales:
            self.pg_exchange_scale = scale
            proposals[f'hand_s{scale:g}'] = _original_hand(self, batch_tasks, system_state, queue_snapshot, temporal_state)
    finally:
        self.pg_exchange_scale = old_scale
    self._drain_memo = {}
    try:
        rows = [_task_payload(self, system_state, task) for task in batch_tasks]
    finally:
        self._drain_memo = None
    candidate_rows = [{(int(c['node_id']), int(c['platform_id'])): c for c in row['candidates']} for row in rows]
    replicas = {}
    for task in batch_tasks:
        for node, platform in system_state.replicas.get(task.type['name'], set()):
            replicas[(int(node.id), int(platform.id))] = (node, platform)
    idx = {int(t.id):i for i,t in enumerate(batch_tasks)}
    orch = self._pg_orchestrator()
    scores = {}
    for arm, plan in proposals.items():
        if set(plan) != set(range(len(batch_tasks))):
            raise RuntimeError(f'selector: incomplete {arm} plan')
        peer = [0.0]*len(batch_tasks)
        for task in batch_tasks:
            i = idx[int(task.id)]
            for peer_id, payload in (orch.peer_exchange.get(int(task.id)) or {}).items():
                j = idx.get(int(peer_id))
                if j is None or i >= j:
                    continue
                ni, pi = replicas[tuple(plan[i])]
                nj, pj = replicas[tuple(plan[j])]
                if ni.node_name == nj.node_name:
                    continue
                e_i = ni.network_map[nj.node_name]
                e_j = nj.network_map[ni.node_name]
                lat_i = float(e_i.get('latency', 0.0)) if isinstance(e_i, dict) else float(e_i)
                lat_j = float(e_j.get('latency', 0.0)) if isinstance(e_j, dict) else float(e_j)
                peer[i] += pi._payload_transfer_time(nj.node_name, float(payload)) + lat_i
                peer[j] += pj._payload_transfer_time(ni.node_name, float(payload)) + lat_j
        base = 0.0
        services = {}
        for i, placement in sorted(plan.items()):
            p = tuple(placement)
            c = candidate_rows[i][p]
            service = c['execution_time']+c['communications_time']+peer[i]
            base += c['queue_drain_seconds']+c['cold_start_time']+c['network_latency']+service
            services.setdefault(p,[]).append(service)
        queue = sum(sum((len(v)-r-1)*s for r,s in enumerate(v)) for v in services.values())
        scores[arm] = base+queue
    winner = min(scores, key=lambda name:(scores[name],name))
    trace_path = os.environ.get('SELECTOR_TRACE_PATH')
    if trace_path:
        with open(trace_path,'a') as fh:
            fh.write(json.dumps({'ids':[int(t.id) for t in batch_tasks], 'winner':winner, 'scores':scores, 'proposals':proposals})+'\n')
    return proposals[winner]

GNNSeededCDScheduler.set_models = _set_models_with_mpoff
GNNSeededCDScheduler._prefix_inference = _selector_prefix
main()
