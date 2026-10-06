import hashlib
import json
import os
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
main()
