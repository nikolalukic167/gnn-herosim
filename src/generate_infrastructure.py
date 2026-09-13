"""
Generate deterministic infrastructure configuration for a dataset.

This script pre-generates:
1. Network topology (connections and latencies)
2. Replica placements (which platforms get replicas)
3. Queue distributions (how many warmup tasks per platform)

All randomness is seeded and saved to infrastructure.json
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.placement.network_fabric import CORE_PREFIX, link_key
from src.utils.distributions import sample_bounded_int, sample_replica_count

SKEW_TOPOLOGY_TYPES = frozenset({"degree_skewed_core"})


def apply_degree_skew_core_server_device_types(nodes: List[Dict], config: Dict[str, Any]) -> None:
    """Force first k_core server nodes to xavier for faster exec on hub platforms."""
    topology_config = config.get("network", {}).get("topology", {})
    if topology_config.get("type") not in SKEW_TOPOLOGY_TYPES:
        return

    k_core = int(topology_config.get("k_core", 4))
    if "xavier" not in config.get("pci", {}):
        raise ValueError("degree_skewed_core requires xavier in pci config")

    xavier_specs = config["pci"]["xavier"]["specs"]
    servers = [n for n in nodes if not n["node_name"].startswith("client_node")]
    for server in servers[:k_core]:
        node_name = server["node_name"]
        server.clear()
        server.update(xavier_specs.copy())
        server["node_name"] = node_name
        server["type"] = "xavier"


def _minimal_skew_connectivity_repair(
    network_maps: Dict[str, Dict[str, float]],
    clients: List[Dict],
    servers: List[Dict],
    latency_s: float,
    *,
    k_core: int,
) -> None:
    """Connect isolated clients to one core server only — preserves degree skew."""
    core_servers = {s["node_name"] for s in servers[:k_core]}
    if not core_servers:
        return

    for client in clients:
        client_name = client["node_name"]
        if network_maps[client_name]:
            continue
        core_name = next(iter(core_servers))
        network_maps[client_name][core_name] = latency_s
        network_maps[core_name][client_name] = latency_s
        print(f"[infra-gen] Skew minimal repair: {client_name} -> {core_name}")


def generate_network_topology_deterministic(
    nodes: List[Dict],
    config: Dict[str, Any],
    rng: random.Random,
    task_types_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Generate network topology deterministically using seeded RNG.
    
    Args:
        nodes: List of node configurations
        config: Configuration containing network latency and topology settings
        rng: Seeded random number generator
    
    Returns:
        Dictionary mapping node names to their network maps
    """
    network_config = config.get('network', {})
    latency_config = network_config.get('latency', {})
    topology_config = network_config.get('topology', {})
    
    device_latencies = latency_config.get('device_latencies', {})
    base_latency = latency_config.get('base_latency', 0.1)
    topology_type = topology_config.get('type', 'sparse')
    connection_probability = topology_config.get('connection_probability', 0.85)
    custom_edges = topology_config.get('edges', [])
    
    # Separate clients and servers
    clients = [node for node in nodes if node['node_name'].startswith('client_node')]
    servers = [node for node in nodes if not node['node_name'].startswith('client_node')]
    
    # Initialize network maps
    network_maps = {node['node_name']: {} for node in nodes}
    
    def generate_latency(device_type1: str, device_type2: str) -> float:
        """Generate latency between two device types."""
        if device_type1 in device_latencies and device_type2 in device_latencies[device_type1]:
            latency_config = device_latencies[device_type1][device_type2]
            min_latency = latency_config.get('min', base_latency)
            max_latency = latency_config.get('max', base_latency)
            return rng.uniform(min_latency, max_latency)
        else:
            return base_latency
    
    if topology_type == 'degree_skewed_core':
        k_core = int(topology_config.get('k_core', 4))
        hub_frac = float(topology_config.get('hub_seeker_fraction', 0.40))
        p_core = float(topology_config.get('p_core', 0.95))
        p_periphery = float(topology_config.get('p_periphery', 0.15))
        lat_core = float(topology_config.get('latency_core_ms', 5.0)) / 1000.0
        lat_periphery = float(
            topology_config.get('latency_periphery_ms', topology_config.get('latency_core_ms', 5.0))
        ) / 1000.0

        core_servers = {s['node_name'] for s in servers[:k_core]}

        for client in clients:
            c_name = client['node_name']
            is_hub_seeker = rng.random() < hub_frac
            for server in servers:
                s_name = server['node_name']
                in_core = s_name in core_servers
                if is_hub_seeker and in_core:
                    p_conn, latency = p_core, lat_core
                elif not is_hub_seeker and not in_core:
                    p_conn, latency = p_periphery, lat_periphery
                elif is_hub_seeker and not in_core:
                    p_conn, latency = p_periphery * 0.5, lat_periphery
                else:
                    p_conn, latency = p_core * 0.3, lat_core
                if rng.random() < p_conn:
                    network_maps[c_name][s_name] = latency
                    network_maps[s_name][c_name] = latency

        _minimal_skew_connectivity_repair(network_maps, clients, servers, lat_core, k_core=k_core)

    elif topology_type == 'custom' and custom_edges:
        # Use custom topology edges
        for edge in custom_edges:
            if len(edge) == 2:
                client_name, server_name = edge
                
                client_node = next((n for n in clients if n['node_name'] == client_name), None)
                server_node = next((n for n in servers if n['node_name'] == server_name), None)
                
                if client_node and server_node:
                    latency = generate_latency(client_node['type'], server_node['type'])
                    network_maps[client_name][server_name] = latency
                    network_maps[server_name][client_name] = latency
    else:
        # Generate connections based on connection probability
        for client in clients:
            client_name = client['node_name']
            client_type = client['type']
            
            for server in servers:
                server_name = server['node_name']
                server_type = server['type']
                
                if rng.random() < connection_probability:
                    latency = generate_latency(client_type, server_type)
                    network_maps[client_name][server_name] = latency
                    network_maps[server_name][client_name] = latency
    
    if topology_type not in SKEW_TOPOLOGY_TYPES:
        # Ensure minimum connectivity
        for node_name, connections in network_maps.items():
            if len(connections) == 0:
                if node_name.startswith('client_node'):
                    available_servers = [s for s in servers if s['node_name'] not in connections]
                    if available_servers:
                        server = rng.choice(available_servers)
                        server_name = server['node_name']
                        server_type = server['type']
                        client_type = next(n['type'] for n in clients if n['node_name'] == node_name)
                        latency = generate_latency(client_type, server_type)
                        network_maps[node_name][server_name] = latency
                        network_maps[server_name][node_name] = latency
                else:
                    available_clients = [c for c in clients if c['node_name'] not in connections]
                    if available_clients:
                        client = rng.choice(available_clients)
                        client_name = client['node_name']
                        client_type = client['type']
                        server_type = next(n['type'] for n in servers if n['node_name'] == node_name)
                        latency = generate_latency(client_type, server_type)
                        network_maps[node_name][client_name] = latency
                        network_maps[client_name][node_name] = latency
    
    # Ensure platform-compatibility-aware connectivity for task types (dnn1 and dnn2)
    # Check each client node to ensure it can execute tasks (either locally or remotely)
    if task_types_data and topology_type not in SKEW_TOPOLOGY_TYPES:
        # Check both dnn1 and dnn2
        for task_type_name in ['dnn1', 'dnn2']:
            if task_type_name not in task_types_data:
                continue
                
            task_type = task_types_data[task_type_name]
            compatible_platforms = set(task_type.get('platforms', []))
            
            for client in clients:
                client_name = client['node_name']
                client_platforms = set(client.get('platforms', []))
                
                # Check if client has compatible platforms locally
                has_local_support = bool(client_platforms & compatible_platforms)
                
                # Check if client is already connected to a server with compatible platforms
                has_remote_support = False
                for server_name in network_maps[client_name].keys():
                    server = next((s for s in servers if s['node_name'] == server_name), None)
                    if server:
                        server_platforms = set(server.get('platforms', []))
                        if bool(server_platforms & compatible_platforms):
                            has_remote_support = True
                            break
                
                # If client lacks both local and remote support, add connection to a server with support
                if not has_local_support and not has_remote_support:
                    # Find servers with compatible platforms
                    compatible_servers = [
                        s for s in servers
                        if bool(set(s.get('platforms', [])) & compatible_platforms)
                        and s['node_name'] not in network_maps[client_name]
                    ]
                    
                    if compatible_servers:
                        # Connect to a random server with support
                        server = rng.choice(compatible_servers)
                        server_name = server['node_name']
                        server_type = server['type']
                        client_type = client['type']
                        latency = generate_latency(client_type, server_type)
                        network_maps[client_name][server_name] = latency
                        network_maps[server_name][client_name] = latency
                        print(
                            f"[infra-gen] Added {task_type_name}-compatibility connection: {client_name} -> {server_name} "
                            f"(client platforms: {client_platforms}, server platforms: {set(server.get('platforms', []))})"
                        )

    build_server_mesh(network_maps, nodes, config, rng, latency_fn=generate_latency)

    return network_maps


def _config_latency_fn(config: Dict[str, Any], rng: random.Random):
    """Same latency lookup `generate_network_topology_deterministic` uses internally.

    That one is a closure over the config it already parsed; this rebuilds it for callers
    that only have the config, so the two cannot drift into different distributions.
    """
    latency_config = (config.get('network', {}) or {}).get('latency', {}) or {}
    device_latencies = latency_config.get('device_latencies', {})
    base_latency = latency_config.get('base_latency', 0.1)

    def latency(device_type1: str, device_type2: str) -> float:
        if device_type1 in device_latencies and device_type2 in device_latencies[device_type1]:
            entry = device_latencies[device_type1][device_type2]
            return rng.uniform(entry.get('min', base_latency), entry.get('max', base_latency))
        return base_latency

    return latency


def build_server_mesh(
    network_maps: Dict[str, Dict[str, float]],
    nodes: List[Dict],
    config: Dict[str, Any],
    rng: random.Random,
    latency_fn=None,
) -> int:
    """route_a: add server<->server latencies so parent->child transfers have a distance.

    Every edge this generator produces is client<->server: `network_maps[server]` contains
    only clients, and the backbone rewrite iterates clients too. That is fine while an
    application is a single task, because the only distance anything prices is
    (source client -> execution node). A DAG needs the distance between two *execution*
    nodes, and for a parent and child that both landed on servers there is currently no
    entry and no route at all.

    Opt-in via `config['network']['server_mesh']`, absent from every existing config, so
    this is a no-op for every corpus generated so far. It runs BEFORE build_core_backbone
    so the backbone can route these edges like any other.

    Returns the number of edges added.
    """
    network_config = config.get('network', {}) or {}
    if not network_config.get('server_mesh'):
        return 0

    # Complete, deliberately. Any server can host any task, so a parent and child can land
    # on any pair; a sparse mesh would leave pairs with no entry, and the transfer term
    # fails loud on a missing one rather than charging 0.0. Distance heterogeneity — which
    # is the signal route A needs — comes from `generate_latency` varying with node type,
    # and from build_core_backbone rewriting these edges as path sums when a backbone is
    # configured, not from dropping edges.
    if latency_fn is None:
        latency_fn = _config_latency_fn(config, rng)

    servers = [n for n in nodes if not n['node_name'].startswith('client_node')]
    added = 0
    for i, left in enumerate(servers):
        for right in servers[i + 1:]:
            left_name, right_name = left['node_name'], right['node_name']
            if right_name in network_maps[left_name]:
                continue
            latency = latency_fn(left['type'], right['type'])
            network_maps[left_name][right_name] = latency
            network_maps[right_name][left_name] = latency
            added += 1

    return added


def _dijkstra_paths(
    adjacency: Dict[str, Dict[str, float]],
    source: str,
) -> Dict[str, List[str]]:
    """Shortest paths by latency from ``source``. The graph is ~46 nodes, so a heap-free
    scan is plenty and keeps this dependency-free (no networkx in the Pipfile)."""
    import heapq

    dist = {source: 0.0}
    prev: Dict[str, str] = {}
    visited = set()
    heap = [(0.0, source)]
    while heap:
        d, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        for neighbour, weight in adjacency.get(node, {}).items():
            nd = d + weight
            if nd < dist.get(neighbour, float('inf')):
                dist[neighbour] = nd
                prev[neighbour] = node
                heapq.heappush(heap, (nd, neighbour))

    paths: Dict[str, List[str]] = {source: [source]}
    for node in dist:
        if node == source:
            continue
        path = [node]
        cursor = node
        while cursor != source:
            cursor = prev[cursor]
            path.append(cursor)
        paths[node] = list(reversed(path))
    return paths


def build_core_backbone(
    network_maps: Dict[str, Dict[str, float]],
    nodes: List[Dict],
    config: Dict[str, Any],
    rng: random.Random,
    seed: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """link_contention_v1: overlay a core router tier and route every logical edge over it.

    Applied as a post-processing overlay *after* every connectivity repair has run, so it
    composes with all three topology types and does not disturb the logical client<->server
    adjacency that candidate filtering (``node_name in source_node.network_map``) depends
    on. What it changes is the *meaning* of the latency: instead of a one-hop constant it
    becomes the sum along a real path, and that path is recorded so the simulator can
    charge contention on each hop.

    Deliberately a ring-plus-chords core rather than a hub. A single bottleneck link would
    reproduce the node-ingress degeneracy one level up -- "load on the busiest link" would
    then be one scalar that repairs everything, which is precisely what
    ``--gate-link-repair`` exists to catch. Multiple comparably-loaded segments, traversed
    by different subsets of client/server pairs, are the whole point.

    Returns ``None`` when no ``network.backbone`` block is configured, in which case
    latencies are left exactly as generated and no fabric is built downstream.
    """
    backbone_config = config.get('network', {}).get('backbone')
    if not backbone_config:
        return None

    # rng_stream=legacy_v0 draws jitter from the caller's shared stream, whose position
    # differs between the corpus path (replica-reachability repair has consumed draws)
    # and the live path (no repair) — the parity divergence recorded in LINEAGES.md
    # (link_contention_v1, 2026-08-21). independent_v1 draws from a dedicated stream
    # derived from the topology seed alone, so both venues produce identical backbone
    # latencies regardless of what ran before. legacy_v0 stays the default so every
    # already-minted backbone cell regenerates byte-identically from its config.
    rng_stream = str(backbone_config.get('rng_stream', 'legacy_v0'))
    if rng_stream == 'independent_v1':
        if seed is None:
            raise ValueError(
                "network.backbone.rng_stream=independent_v1 requires the topology seed "
                "to derive the dedicated backbone rng, but no seed was passed"
            )
        draw_rng = random.Random(f"{seed}:backbone_v1")
    elif rng_stream == 'legacy_v0':
        draw_rng = rng
    else:
        raise ValueError(
            f"network.backbone.rng_stream must be 'legacy_v0' or 'independent_v1', "
            f"got {rng_stream!r}"
        )

    n_core = int(backbone_config.get('n_core', 6))
    if n_core < 2:
        raise ValueError(
            f"network.backbone.n_core must be >= 2 to form a backbone, got {n_core}"
        )
    attach_degree = int(backbone_config.get('attach_degree', 2))
    if not 1 <= attach_degree <= n_core:
        raise ValueError(
            f"network.backbone.attach_degree must be in [1, n_core={n_core}], "
            f"got {attach_degree}"
        )
    chord_count = int(backbone_config.get('chord_count', n_core // 2))
    core_latency = float(backbone_config.get('core_link_latency_ms', 4.0)) / 1000.0
    access_latency = float(backbone_config.get('access_link_latency_ms', 20.0)) / 1000.0
    latency_jitter = float(backbone_config.get('access_latency_jitter', 0.3))

    bandwidth_mbps = backbone_config.get('bandwidth_mbps')
    if bandwidth_mbps is None or float(bandwidth_mbps) <= 0:
        raise ValueError(
            f"network.backbone.bandwidth_mbps must be > 0 when a backbone is configured, "
            f"got {bandwidth_mbps}"
        )
    bandwidth_mbps = float(bandwidth_mbps)
    core_bandwidth_mbps = float(
        backbone_config.get('core_bandwidth_mbps') or bandwidth_mbps
    )
    if core_bandwidth_mbps <= 0:
        raise ValueError(
            f"network.backbone.core_bandwidth_mbps must be > 0, got {core_bandwidth_mbps}"
        )

    core_names = [f"{CORE_PREFIX}{i}" for i in range(n_core)]
    links: Dict[str, Dict[str, float]] = {}
    adjacency: Dict[str, Dict[str, float]] = {name: {} for name in core_names}

    def _add_link(a: str, b: str, latency: float, bandwidth: float) -> None:
        links[link_key(a, b)] = {
            "latency": latency,
            "bandwidth_mbps": bandwidth,
        }
        adjacency.setdefault(a, {})[b] = latency
        adjacency.setdefault(b, {})[a] = latency

    # Core ring, then chords across it. Both are deterministic in n_core -- the skew we
    # want comes from where clients and servers attach, not from a random core.
    for i in range(n_core):
        _add_link(core_names[i], core_names[(i + 1) % n_core], core_latency, core_bandwidth_mbps)
    for i in range(min(chord_count, n_core)):
        partner = core_names[(i + n_core // 2) % n_core]
        if partner != core_names[i]:
            _add_link(core_names[i], partner, core_latency, core_bandwidth_mbps)

    # Attach every node to `attach_degree` cores. Access links carry one node's traffic
    # only, so they stay additive; they exist to give paths somewhere to diverge.
    attachments: Dict[str, List[str]] = {}
    for node in nodes:
        node_name = node['node_name']
        chosen = draw_rng.sample(core_names, attach_degree)
        attachments[node_name] = chosen
        for core_name in chosen:
            jitter = 1.0 + draw_rng.uniform(-latency_jitter, latency_jitter)
            _add_link(node_name, core_name, access_latency * jitter, bandwidth_mbps)

    # Route every logical edge and rewrite its latency as the path sum.
    routes: Dict[str, Dict[str, List[str]]] = {}
    used_links = set()
    clients = [n['node_name'] for n in nodes if n['node_name'].startswith('client_node')]
    # Sources are clients plus — when a server mesh exists (route_a) — servers, so that
    # server<->server edges are path sums over the same core tier rather than the one-hop
    # constants generate_latency produced. Without this a DAG's parent->child distance
    # would ignore the backbone that every other distance in the run respects.
    servers_with_mesh = [
        n['node_name'] for n in nodes
        if not n['node_name'].startswith('client_node')
        and any(
            not peer.startswith('client_node')
            for peer in network_maps.get(n['node_name'], {})
        )
    ]
    for source_name in clients + servers_with_mesh:
        paths = _dijkstra_paths(adjacency, source_name)
        for peer_name in list(network_maps.get(source_name, {})):
            if source_name in routes and peer_name in routes[source_name]:
                continue
            path = paths.get(peer_name)
            if path is None:
                raise RuntimeError(
                    f"link_contention_v1: {source_name} -> {peer_name} is a logical "
                    f"network_map edge with no path over the backbone. The core tier must "
                    f"span every node or the simulator would silently charge nothing."
                )
            routes.setdefault(source_name, {})[peer_name] = path
            total = 0.0
            for i in range(len(path) - 1):
                key = link_key(path[i], path[i + 1])
                used_links.add(key)
                total += links[key]["latency"]
            network_maps[source_name][peer_name] = total
            network_maps[peer_name][source_name] = total

    # Keep only links some route actually traverses: an untraversed pipe never contends,
    # and pruning keeps `link_keys` an honest denominator for the overlap pre-check.
    links = {key: attrs for key, attrs in links.items() if key in used_links}

    return {
        "links": links,
        "routes": routes,
        "params": {
            "n_core": n_core,
            "attach_degree": attach_degree,
            "chord_count": chord_count,
            "core_link_latency_ms": core_latency * 1000.0,
            "access_link_latency_ms": access_latency * 1000.0,
            "bandwidth_mbps": bandwidth_mbps,
            "core_bandwidth_mbps": core_bandwidth_mbps,
            "rng_stream": rng_stream,
        },
    }


class ReplicaStarvationError(RuntimeError):
    """A task type asked for replicas and the FCFS allocator gave it none.

    Raised by `generate_replica_placements_deterministic` so the cause is named where it
    happens instead of surfacing as an opaque warmup-capture failure one stage later.
    """


def generate_replica_placements_deterministic(
    nodes: List[Dict],
    config: Dict[str, Any],
    sim_inputs: Dict[str, Any],
    rng: random.Random
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate replica placements deterministically.
    
    Returns:
        {
            task_type: [
                {"node_name": str, "platform_id": int, "platform_type": str}
            ]
        }
    """
    preinit_config = config.get('preinit', {})
    replicas_config = config.get('replicas', {})
    
    # Get preinit nodes
    all_client_nodes = [node for node in nodes if node.get('node_name', '').startswith('client_node')]
    all_server_nodes = [node for node in nodes if not node.get('node_name', '').startswith('client_node')]
    
    # Handle percentage-based configuration
    # NOTE: For replica placement (not preinit), we use higher percentages to ensure
    # replicas are spread across enough nodes for network reachability
    preinit_clients = preinit_config.get('clients', [])
    preinit_servers = preinit_config.get('servers', [])
    
    if not preinit_clients and 'client_percentage' in preinit_config:
        client_pct = float(preinit_config.get('client_percentage', 0))
        # For replica placement, use at least 50% of clients even for cold start
        replica_client_pct = max(client_pct, 0.5)
        k = max(1, int(len(all_client_nodes) * replica_client_pct))
        preinit_clients = [n['node_name'] for n in all_client_nodes[:k]]
    
    if not preinit_servers and 'server_percentage' in preinit_config:
        server_pct = float(preinit_config.get('server_percentage', 0))
        # For replica placement, use at least 60% of servers even for cold start
        # This ensures replicas are spread across enough nodes for network reachability
        #
        # network_contention_v1: that 0.6 floor is also what keeps candidate sets DISPERSED.
        # Measured on the pilots, tasks' mean pairwise candidate-node overlap was 0.93 (dense
        # grid), 0.36 (scarce) and 0.14 (funnelled) out of 4 tasks — so every task had its own
        # favourite node and spreading was free, which is why M1 marginal-greedy regret sat at
        # exactly 0%. Concentrating replicas onto few hosts is what forces tasks to compete for
        # the same nodes. Explicit replica_server_percentage overrides the floor; absent, the
        # floor applies and every existing grid is unchanged.
        explicit = preinit_config.get('replica_server_percentage')
        if explicit is not None:
            replica_server_pct = float(explicit)
            if not 0.0 < replica_server_pct <= 1.0:
                raise ValueError(
                    f"preinit.replica_server_percentage must be in (0, 1], got {explicit}"
                )
        else:
            replica_server_pct = max(server_pct, 0.6)
        k = max(1, int(len(all_server_nodes) * replica_server_pct))
        preinit_servers = [n['node_name'] for n in all_server_nodes[:k]]
    
    if preinit_clients == "all":
        preinit_clients = [n['node_name'] for n in all_client_nodes]
    if preinit_servers == "all":
        preinit_servers = [n['node_name'] for n in all_server_nodes]
    
    # Create node_id and platform_id mappings (same as simulation.py)
    node_id_map = {node['node_name']: i for i, node in enumerate(nodes)}
    platform_id = 0
    node_platforms = {}  # node_name -> list of (platform_id, platform_type)
    
    for node in nodes:
        node_name = node['node_name']
        node_platforms[node_name] = []
        for platform_type_name in node.get('platforms', []):
            node_platforms[node_name].append({
                'platform_id': platform_id,
                'platform_type': platform_type_name
            })
            platform_id += 1
    
    # Generate replica placements
    replica_placements = {}
    task_types = sim_inputs.get('task_types', {})

    # route_b env pivot (2026-08-27), W3: relax the disjoint-assigned_platforms
    # invariant so task types may SHARE replica hosts/platforms — the organic overlap
    # recipe (CONTEXT: with the masked decoder's no-replica-reuse mask, a shared
    # (node, platform) slot becomes an indivisible resource contested ACROSS task
    # types, not just within one). Default False -> assigned_platforms is still
    # checked and populated exactly as before, so every existing grid (no
    # preinit.replica_overlap key) reproduces byte-identically.
    replica_overlap = bool(preinit_config.get('replica_overlap', False))

    assigned_platforms = set()  # Set of (node_name, platform_id) tuples

    for task_type_name, replica_config in replicas_config.items():
        if task_type_name not in task_types:
            continue

        task_type = task_types[task_type_name]
        supported_platforms = task_type.get('platforms', [])

        placements = []

        # Create server replicas
        per_server = replica_config.get('per_server', 0)
        if per_server > 0:
            for node in all_server_nodes:
                node_name = node['node_name']
                if node_name in preinit_servers:
                    suitable_platforms = [
                        p for p in node_platforms[node_name]
                        if p['platform_type'] in supported_platforms
                        and (replica_overlap
                             or (node_name, p['platform_id']) not in assigned_platforms)
                    ]

                    replicas_created = 0
                    for platform_info in suitable_platforms:
                        if replicas_created >= per_server:
                            break

                        platform_key = (node_name, platform_info['platform_id'])
                        placements.append({
                            'node_name': node_name,
                            'platform_id': platform_info['platform_id'],
                            'platform_type': platform_info['platform_type']
                        })
                        assigned_platforms.add(platform_key)  # Mark as assigned
                        replicas_created += 1

        # Create client replicas
        per_client = replica_config.get('per_client', 0)
        if per_client > 0:
            for node in all_client_nodes:
                node_name = node['node_name']
                if node_name in preinit_clients:
                    suitable_platforms = [
                        p for p in node_platforms[node_name]
                        if p['platform_type'] in supported_platforms
                        and (replica_overlap
                             or (node_name, p['platform_id']) not in assigned_platforms)
                    ]

                    replicas_created = 0
                    for platform_info in suitable_platforms:
                        if replicas_created >= per_client:
                            break

                        platform_key = (node_name, platform_info['platform_id'])
                        placements.append({
                            'node_name': node_name,
                            'platform_id': platform_info['platform_id'],
                            'platform_type': platform_info['platform_type']
                        })
                        assigned_platforms.add(platform_key)  # Mark as assigned
                        replicas_created += 1

        replica_placements[task_type_name] = placements

    print(f"\n[infra-gen] Replica placement summary:")
    for task_type, placements in replica_placements.items():
        print(f"  {task_type}: {len(placements)} replicas")
    print(f"  Total unique platforms assigned: {len(assigned_platforms)}")
    if replica_overlap:
        print(f"  replica_overlap=True: task types MAY share (node, platform) slots")

    # Fail at the point of cause, with the counts. The FCFS walk above lets early task
    # types consume the pool, and a type that asked for replicas and got ZERO used to die
    # much later as an unlabelled `System state capture FAILED` at warmup (co-sim) with
    # nothing on disk naming the cause — measured 12/24 datasets on the route_b
    # `per_server=5` no-overlap probe (docs/gates/gate-tools.md, 2026-08-28). A type that
    # requested no replicas at all is not starved and is left alone.
    starved = {
        name: {
            "requested_per_server": int(replicas_config[name].get('per_server', 0)),
            "requested_per_client": int(replicas_config[name].get('per_client', 0)),
        }
        for name, placements in replica_placements.items()
        if not placements
        and (int(replicas_config[name].get('per_server', 0)) > 0
             or int(replicas_config[name].get('per_client', 0)) > 0)
    }
    if starved:
        counts = {name: len(p) for name, p in replica_placements.items()}
        raise ReplicaStarvationError(
            f"CRITICAL: task type(s) {sorted(starved)} requested replicas and were "
            f"allocated ZERO. Per-type counts (FCFS order): {counts}; requested: {starved}; "
            f"replica_overlap={replica_overlap}; hosting servers={len(preinit_servers)}, "
            f"clients={len(preinit_clients)}. Earlier types consumed the pool — raise "
            f"per_server for the LATER types, enable preinit.replica_overlap, or widen the "
            f"hosting set. Raising replica_server_percentage does not help (measured)."
        )

    # verify no duplicates WITHIN a task type (a task type still can't double-book its
    # own platform_id) — cross-type sharing is the point of replica_overlap and is
    # exactly what this check must NOT flag when it's on.
    for task_type_name, placements in replica_placements.items():
        keys = [(p['node_name'], p['platform_id']) for p in placements]
        if len(keys) != len(set(keys)):
            dups = [k for k in keys if keys.count(k) > 1]
            raise RuntimeError(
                f"CRITICAL: task type {task_type_name!r} has duplicate platform "
                f"assignments within itself: {dups}. This should not happen even "
                f"under replica_overlap."
            )
    if not replica_overlap:
        all_platform_keys = []
        for placements in replica_placements.values():
            for p in placements:
                platform_key = (p['node_name'], p['platform_id'])
                all_platform_keys.append(platform_key)

        if len(all_platform_keys) != len(set(all_platform_keys)):
            duplicates = [k for k in all_platform_keys if all_platform_keys.count(k) > 1]
            raise RuntimeError(
                f"CRITICAL: Found duplicate platform assignments: {duplicates}. "
                f"This should not happen - each platform can only be assigned to one "
                f"task type (preinit.replica_overlap is off)."
            )

    return replica_placements


def generate_queue_distributions_deterministic(
    replica_placements: Dict[str, List[Dict[str, Any]]],
    config: Dict[str, Any],
    rng: random.Random
) -> Dict[str, Dict[str, int]]:
    """
    Generate queue distributions deterministically.
    
    Returns:
        {
            task_type: {
                "node_name:platform_id": queue_length
            }
        }
    """
    queue_distributions = {}
    prewarm_config = config.get('prewarm', {})
    
    for task_type_name, placements in replica_placements.items():
        task_prewarm = prewarm_config.get(task_type_name, {})
        queue_dist = {}
        
        for placement in placements:
            node_name = placement['node_name']
            platform_id = placement['platform_id']
            platform_key = f"{node_name}:{platform_id}"
            
            # Get queue distribution parameters
            initial_queue = task_prewarm.get('initial_queue', 0)
            
            if task_prewarm.get('queue_distribution') == 'statistical':
                q_params = task_prewarm.get('queue_distribution_params') or {}
                if 'min' not in q_params:
                    q_params['min'] = 0
                sampled_q = sample_bounded_int(q_params, rng)
                queue_length = max(0, int(sampled_q))
            else:
                queue_length = initial_queue
            
            queue_dist[platform_key] = queue_length
        
        queue_distributions[task_type_name] = queue_dist
    
    return queue_distributions


def generate_deterministic_infrastructure(
    config_file: str,
    sim_input_path: Path,
    output_file: str,
    seed: int
) -> Dict[str, Any]:
    """
    Generate deterministic infrastructure.
    
    Args:
        config_file: Path to infrastructure configuration file
        sim_input_path: Path to simulation input files directory
        output_file: Path to output infrastructure.json file
        seed: Random seed for deterministic generation
    
    Returns:
        Infrastructure dictionary
    """
    # Load config
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Load simulation inputs
    from src.executecosimulation import load_simulation_inputs
    sim_inputs = load_simulation_inputs(sim_input_path)
    
    # Create seeded RNG
    rng = random.Random(seed)
    
    # Generate nodes (same logic as prepare_simulation_config)
    client_nodes_count = config['nodes']['client_nodes']['count']
    server_nodes_count = config['nodes']['server_nodes']['count']
    
    nodes = []
    device_types = list(config['pci'].keys())
    
    # Generate client nodes
    for i in range(client_nodes_count):
        device_type = device_types[i % len(device_types)]
        device_specs = config['pci'][device_type]['specs']
        node_config = device_specs.copy()
        node_config['node_name'] = f"client_node{i}"
        node_config['type'] = device_type
        nodes.append(node_config)
    
    # Generate server nodes
    for i in range(server_nodes_count):
        device_type = device_types[i % len(device_types)]
        device_specs = config['pci'][device_type]['specs']
        node_config = device_specs.copy()
        node_config['node_name'] = f"node{i}"
        node_config['type'] = device_type
        nodes.append(node_config)

    apply_degree_skew_core_server_device_types(nodes, config)
    
    # Load task-types.json for platform compatibility checks
    task_types_path = sim_input_path / "task-types.json"
    task_types_data = None
    if task_types_path.exists():
        with open(task_types_path, 'r') as f:
            task_types_data = json.load(f)
    
    # 1. Generate network topology
    print("[infra-gen] Generating network topology...")
    network_maps = generate_network_topology_deterministic(nodes, config, rng, task_types_data=task_types_data)
    
    # 2. Generate replica placements
    print("[infra-gen] Generating replica placements...")
    replica_placements = generate_replica_placements_deterministic(
        nodes, config, sim_inputs, rng
    )
    
    topology_type = config.get('network', {}).get('topology', {}).get('type', 'sparse')

    # 2b. Ensure network connectivity to replica servers
    # After placing replicas, ensure every client can reach MULTIPLE servers
    # that have replicas for each task type (for uniqueness constraint)
    clients = [n for n in nodes if n['node_name'].startswith('client_node')]
    servers = [n for n in nodes if not n['node_name'].startswith('client_node')]

    # Minimum number of replica servers each client should reach per task type
    MIN_REPLICA_SERVERS = 2

    skew_topo = config.get('network', {}).get('topology', {})
    skew_core_servers: set = set()
    skew_lat_core = 0.1
    skew_lat_periphery = 0.1
    if topology_type in SKEW_TOPOLOGY_TYPES:
        k_core = int(skew_topo.get('k_core', 4))
        skew_core_servers = {s['node_name'] for s in servers[:k_core]}
        skew_lat_core = float(skew_topo.get('latency_core_ms', 5.0)) / 1000.0
        skew_lat_periphery = float(
            skew_topo.get('latency_periphery_ms', skew_topo.get('latency_core_ms', 5.0))
        ) / 1000.0
        print("[infra-gen] Ensuring replica reachability (degree_skewed_core)...")
    else:
        print("[infra-gen] Ensuring replica reachability...")

    def _repair_latency(server_name: str) -> float:
        if topology_type in SKEW_TOPOLOGY_TYPES:
            return skew_lat_core if server_name in skew_core_servers else skew_lat_periphery
        return config.get('network', {}).get('latency', {}).get('base_latency', 0.1)

    for task_type_name, placements in replica_placements.items():
        # Get servers that have replicas for this task type
        replica_servers = set(
            p['node_name'] for p in placements if not p['node_name'].startswith('client_')
        )

        if not replica_servers:
            continue

        for client in clients:
            client_name = client['node_name']

            # Count how many replica servers this client can reach
            reachable_replica_servers = [
                server_name for server_name in replica_servers
                if server_name in network_maps[client_name]
            ]

            # Add connections until we reach the minimum
            needed = MIN_REPLICA_SERVERS - len(reachable_replica_servers)
            if needed > 0:
                # Find servers with replicas that we're not connected to
                available_servers = [
                    s for s in servers
                    if s['node_name'] in replica_servers
                    and s['node_name'] not in network_maps[client_name]
                ]
                rng.shuffle(available_servers)

                for server in available_servers[:needed]:
                    server_name = server['node_name']
                    latency = _repair_latency(server_name)
                    network_maps[client_name][server_name] = latency
                    network_maps[server_name][client_name] = latency
                    print(
                        f"[infra-gen] Added {task_type_name} reachability: "
                        f"{client_name} -> {server_name}"
                    )
    
    # 2c. link_contention_v1: overlay the core backbone. Runs after every connectivity
    # repair so each logical edge that survives gets a route; absent a
    # network.backbone block this is a no-op and latencies stay one-hop constants.
    link_topology = build_core_backbone(network_maps, nodes, config, rng, seed=seed)
    if link_topology is not None:
        print(
            f"[infra-gen] Core backbone: {len(link_topology['links'])} links, "
            f"{sum(len(v) for v in link_topology['routes'].values())} routes"
        )

    # 3. Generate queue distributions
    print("[infra-gen] Generating queue distributions...")
    queue_distributions = generate_queue_distributions_deterministic(
        replica_placements, config, rng
    )
    
    # node_contention_v3: shared execution slots per node. Absent from the config this
    # stays None and platforms run independently (node_disk_v2), so existing corpora
    # regenerate unchanged.
    compute_slots_per_node = config.get("nodes", {}).get("compute_slots_per_node")
    if compute_slots_per_node is not None and int(compute_slots_per_node) < 1:
        raise ValueError(
            f"nodes.compute_slots_per_node must be >= 1 when set, "
            f"got {compute_slots_per_node}"
        )

    # network_contention_v1: shared inbound bandwidth (MB/s) per node. Absent from the
    # config this stays None, no ingress pipe is built and no transmission time is
    # charged, so existing corpora regenerate unchanged.
    ingress_bandwidth_mbps = config.get("nodes", {}).get("ingress_bandwidth_mbps")
    if ingress_bandwidth_mbps is not None and float(ingress_bandwidth_mbps) <= 0:
        raise ValueError(
            f"nodes.ingress_bandwidth_mbps must be > 0 when set, "
            f"got {ingress_bandwidth_mbps}"
        )

    infrastructure = {
        "network_maps": network_maps,
        "replica_placements": replica_placements,
        "queue_distributions": queue_distributions,
        "compute_slots_per_node": (
            int(compute_slots_per_node) if compute_slots_per_node is not None else None
        ),
        "ingress_bandwidth_mbps": (
            float(ingress_bandwidth_mbps)
            if ingress_bandwidth_mbps is not None
            else None
        ),
        # link_contention_v1: None keeps today's physics exactly -- no fabric is built and
        # no per-hop transmission is charged.
        "link_topology": link_topology,
        "metadata": {
            "seed": seed,
            "config_file": config_file,
            "generation_time": datetime.now().isoformat(),
            # Which warmth physics this dataset was generated for. Metadata extraction
            # used to report a .get() default here and call it measured.
            "warmth_physics": config.get("warmth_physics"),
        }
    }
    
    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(infrastructure, f, indent=2)
    
    print(f"[infra-gen] Generated deterministic infrastructure: {output_file}")
    print(f"  Network maps: {len(network_maps)} nodes")
    print(f"  Replica placements: {sum(len(v) for v in replica_placements.values())} total replicas")
    print(f"  Queue distributions: {sum(len(v) for v in queue_distributions.values())} platforms")
    
    return infrastructure


def main():
    """Main entry point for infrastructure generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate deterministic infrastructure configuration"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to infrastructure configuration file'
    )
    parser.add_argument(
        '--sim-input',
        type=str,
        required=True,
        help='Path to simulation input files directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output infrastructure.json file'
    )
    parser.add_argument(
        '--seed',
        type=int,
        required=True,
        help='Random seed for deterministic generation'
    )
    
    args = parser.parse_args()
    
    generate_deterministic_infrastructure(
        args.config,
        Path(args.sim_input),
        args.output,
        args.seed
    )
    print("[infra-gen] Done.")


if __name__ == "__main__":
    main()

