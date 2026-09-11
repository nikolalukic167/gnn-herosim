"""peer_affinity_v1 stage 3: live serving of prefix-conditioned GNN checkpoints.

Until 2026-09-11 a checkpoint whose sidecar declares ``partial_state_edge_features`` could
be scored offline only (``scripts_cosim/eval_route_b_stage2_arm.py``); ``load_gnn_model``
refused it outright, because the live path had no way to build the 38 partial-state
columns, the 4-way task-type one-hot, or the task<->task peer edges the model was fitted
on. This module is that missing piece, and it is deliberately ONE piece: the offline
evaluator and the live scheduler both import it, so the graph the model is served on and
the graph it was read on are produced by the same code — the lesson every train/serve
mismatch in this repo has taught (``docs/lessons.md``).

Three functions:

* :func:`load_prefix_conditioned_gnn` — sidecar-validated construction of the model
  (dims read off the weights, weight-invisible options off the sidecar, contracts adopted
  into the environment or verified against it), plus the decode options the checkpoint
  was registered with.
* :func:`attach_live_prefix_block` — the live twin of
  ``prepare_graphs_cache.attach_dag_partial_state_block``: given the graph
  ``build_pyg_inference_graph`` just built for a batch, attach ``task_type_onehot4``,
  ``peer_edge_index``/``peer_edge_attr`` and a ``partial_state_ctx`` with exactly the keys
  the cache writes, computed from live objects (tasks, nodes, the fabric, the
  orchestrator's peer table) instead of dataset files. Every formula is imported, not
  re-typed: demands from ``memoryRequirements`` x the event's ``demand_scale``, caps by the
  ``alpha_max`` rule, node ranks from ``krank_node_order``, route metrics from
  ``route_hops_and_bottleneck``.
* :func:`decode_prefix_conditioned` — the registered decoder call (``masked_topo`` with
  per-step re-scoring, replica reuse and counted relaxation as the sidecar declares).

Parity is not assumed: ``scripts_cosim/peer_affinity_live_serve_check.py`` runs the live
scheduler on the held-out co-sim datasets and asserts the live graph equals the cache
graph attribute by attribute, the decoded plan equals the offline report's, and the engine
RTT equals the sweep row.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from src.placement.dag_workload import route_hops_and_bottleneck
from src.placement.network_fabric import is_core_link, route_links
from src.policy.gnn.gnn_model import TaskPlacementGNN
from src.policy.gnn.partial_state_edges import make_partial_state_score_fn
from src.policy.gnn.seq_decode import GnnDecodeRunStats, decode_masked_topo_placement
from src.policy.tabular.reduced_features import (
    PARTIAL_STATE_CONTRACT_ENV,
    PARTIAL_STATE_FEATURE_DIM,
    PARTIAL_STATE_PEER_MASS_ENV,
    build_partial_state_context_from_graph,
    krank_node_order,
    peer_mass_enabled,
    require_matching_partial_state_contract,
    resolve_partial_state_contract,
)

# The one-hot column order the cache builder and the sidecar record. Imported lazily from
# its single home (prepare_graphs_cache pulls pandas and the whole cache toolchain in at
# import, which a live scheduler should not pay for until it actually serves one of these
# checkpoints).
def _task_type_vocab() -> Tuple[str, ...]:
    from src.notebooks.prepare_graphs_cache import DAG_TASK_TYPE_VOCAB
    return tuple(DAG_TASK_TYPE_VOCAB)


class PrefixServingError(RuntimeError):
    """A prefix-conditioned checkpoint cannot be served as configured. Never absorbed."""


@dataclass(frozen=True)
class PrefixServingOptions:
    """What the checkpoint's sidecar says about how it must be decoded."""

    alpha_key: str
    allow_replica_reuse: bool
    relax_on_stuck: bool
    task_type_vocab: Tuple[str, ...]
    peer_mass: bool
    mp_peer_edges: bool
    partial_state_contract: str
    # peer_affinity_v1 stage 3 (2026-09-11), both SERVING knobs, both default OFF so the
    # registered gate stays byte-identical. Measured motivation in the node: the decoder's
    # per-node load resets every batch, so the cap is intra-batch and blind to the queue
    # already standing on a node; across 45,375 batches the same preferred node keeps
    # clearing it and concurrency collapses (15.3 -> 5.4 tasks in service).
    #   load_seed_scale       >0 seeds the decode's per-node load with the node's queue
    #                         IMBALANCE (depth above the least-loaded candidate node),
    #                         converted into demand units, x this scale. The least-loaded
    #                         node always keeps its full cap, so the mask can never empty.
    #   concurrency_penalty   >0 subtracts penalty x normalised candidate queue depth from
    #                         each candidate's score. A soft mask: it reorders, never
    #                         forbids, so no decode can fail because of it.
    load_seed_scale: float = 0.0
    concurrency_penalty: float = 0.0


def _serving_knob(name: str) -> float:
    """A live-only serving knob: absent or empty is 0.0 (off). Anything unparseable is
    fatal rather than silently off — a mistyped gate env must not look like the baseline."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return 0.0
    try:
        value = float(raw)
    except ValueError as exc:
        raise PrefixServingError(f"{name}={raw!r} is not a number") from exc
    if value < 0.0:
        raise PrefixServingError(f"{name}={value} must be >= 0")
    return value


def _adopt_or_verify_env(
    name: str, trained: str, label: str, *, adopt: bool, default: Optional[str] = None
) -> None:
    """Contracts the sidecar records must match the environment when it speaks. When it
    is silent the LIVE loader adopts them (the rule the queue/topology contracts follow);
    the offline evaluator (adopt=False) leaves the environment alone — its registered
    protocol exports every flag explicitly — and a silent environment is compared as the
    value it resolves to (`default`), which is how a trained-MP-OFF checkpoint scored with
    the flag unset is a mismatch, not a pass (route_b 2026-09-03)."""
    declared = os.environ.get(name, "").strip()
    if not declared:
        if adopt:
            os.environ[name] = trained
            print(f"[PREFIX SERVING] {label}: adopting {name}={trained}", flush=True)
            return
        if default is None:
            return
        declared = default
    if declared != trained:
        raise PrefixServingError(
            f"{label}: trained with {name}={trained!r} but this run resolves to {declared!r} "
            "— a train/serve mismatch, not an ablation; export the flag to match the sidecar"
        )


def load_prefix_conditioned_gnn(
    checkpoint_path: Path, *, device: Optional[torch.device] = None, adopt_env: bool = True
) -> Tuple[TaskPlacementGNN, PrefixServingOptions, dict]:
    """Construct a prefix-conditioned ``TaskPlacementGNN`` from ``<ckpt>.pt`` and its
    mandatory ``.contract.json`` sidecar. Returns (model in eval mode, decode options,
    sidecar payload). Sidecar keys a route_b-era T2 checkpoint may lack (``peer_mass``,
    ``disable_message_passing``, ``dag_alpha_key``) are checked only when present, as
    the offline evaluator always did; the live decode needs a cap rung, so a missing
    ``dag_alpha_key`` must be supplied as GNN_PREFIX_ALPHA_KEY there."""
    checkpoint_path = Path(checkpoint_path)
    label = checkpoint_path.name
    sidecar_path = checkpoint_path.with_suffix(".contract.json")
    if not sidecar_path.is_file():
        raise PrefixServingError(
            f"{label}: no .contract.json sidecar — a prefix-conditioned checkpoint's "
            "architecture (peer edges, one-hot width, partial-state contract) is not "
            "recoverable from its weights"
        )
    sidecar = json.loads(sidecar_path.read_text())
    if not sidecar.get("partial_state_edge_features"):
        raise PrefixServingError(
            f"{label}: sidecar does not declare partial_state_edge_features=true — this is "
            "not a stage-2 T2 (A1) checkpoint (prefix conditioning is what this loader "
            "and the masked_topo decode assume)"
        )

    # Message passing first: weight-invisible, and a mismatch corrupts inference in both
    # directions (route_b 2026-09-03: 5.7x train-regret error). Sidecars before that date
    # carry no key and are not blocked. Unset at serve time means MP ON.
    declared_mp_off = sidecar.get("disable_message_passing")
    if declared_mp_off is not None:
        _adopt_or_verify_env(
            "GNN_DISABLE_MESSAGE_PASSING", "1" if declared_mp_off else "0", label,
            adopt=adopt_env, default="0",
        )

    # Contracts: partial-state column meaning, peer-mass column.
    trained_contract = sidecar.get("partial_state_contract")
    if not trained_contract:
        raise PrefixServingError(f"{label}: sidecar has no partial_state_contract")
    _adopt_or_verify_env(PARTIAL_STATE_CONTRACT_ENV, str(trained_contract), label, adopt=adopt_env)
    require_matching_partial_state_contract(
        trained_contract, resolve_partial_state_contract(), model_label=label
    )
    trained_peer_mass = sidecar.get("peer_mass")
    if trained_peer_mass is not None:
        _adopt_or_verify_env(
            PARTIAL_STATE_PEER_MASS_ENV, "1" if trained_peer_mass else "0", label,
            adopt=adopt_env, default="1",
        )
        if bool(trained_peer_mass) != peer_mass_enabled():
            raise PrefixServingError(
                f"{label}: sidecar peer_mass={bool(trained_peer_mass)} but "
                f"{PARTIAL_STATE_PEER_MASS_ENV} resolves to {peer_mass_enabled()}; export it to match"
            )

    vocab = _task_type_vocab()
    onehot_dim = int(sidecar.get("task_type_onehot_dim") or 0)
    if onehot_dim != len(vocab):
        raise PrefixServingError(
            f"{label}: task_type_onehot_dim={onehot_dim} != live vocab width {len(vocab)}"
        )
    if list(sidecar.get("dag_task_type_vocab") or []) != list(vocab):
        raise PrefixServingError(
            f"{label}: sidecar dag_task_type_vocab {sidecar.get('dag_task_type_vocab')!r} "
            f"!= live vocab {list(vocab)!r} — a reorder would silently permute task types"
        )
    partial_dim = int(sidecar.get("partial_state_feature_dim") or 0)
    if partial_dim != PARTIAL_STATE_FEATURE_DIM:
        raise PrefixServingError(
            f"{label}: partial_state_feature_dim={partial_dim} != live "
            f"PARTIAL_STATE_FEATURE_DIM={PARTIAL_STATE_FEATURE_DIM}"
        )
    alpha_key = str(sidecar.get("dag_alpha_key") or "").strip()
    env_alpha = os.environ.get("GNN_PREFIX_ALPHA_KEY", "").strip()
    if alpha_key and env_alpha and env_alpha != alpha_key:
        raise PrefixServingError(
            f"{label}: trained/selected at dag_alpha_key={alpha_key} but "
            f"GNN_PREFIX_ALPHA_KEY={env_alpha}; the cap rung is part of the checkpoint"
        )
    alpha_key = alpha_key or env_alpha  # "" -> the live decode refuses to run

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    task_w = state_dict["task_encoder.net.0.weight"]
    plat_w = state_dict["platform_encoder.net.0.weight"]
    task_feature_dim = int(task_w.shape[1]) - onehot_dim
    platform_feature_dim = int(plat_w.shape[1])
    hidden_dim = int(task_w.shape[0])
    embedding_dim = int(state_dict["task_encoder.net.4.weight"].shape[0])
    num_layers = sum(
        1 for k in state_dict if k.startswith("gin.convs.") and k.endswith(".nn.lins.0.weight")
    )
    if num_layers <= 0:
        raise PrefixServingError(f"{label}: could not infer num_layers from gin.convs.* keys")

    model = TaskPlacementGNN(
        task_feature_dim=task_feature_dim,
        platform_feature_dim=platform_feature_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        mp_residual=bool(sidecar.get("mp_residual", False)),
        mp_node_edges=bool(sidecar.get("mp_node_edges", False)),
        mp_node_edges_candidates_only=bool(sidecar.get("mp_node_edges_candidates_only", True)),
        mp_network_entities=bool(sidecar.get("mp_network_entities", False)),
        mp_dag_edges=bool(sidecar.get("mp_dag_edges", False)),
        mp_peer_edges=bool(sidecar.get("mp_peer_edges", False)),
        task_type_onehot_dim=onehot_dim,
        partial_state_edge_dim=partial_dim,
        normalize_platform_inputs=sidecar.get("feature_dim") == 21,
    )
    model.load_state_dict(state_dict)
    if device is not None:
        model = model.to(device)
    model.eval()

    options = PrefixServingOptions(
        alpha_key=alpha_key,
        allow_replica_reuse=bool(sidecar.get("decode_replica_reuse", False)),
        relax_on_stuck=bool(sidecar.get("decode_relax_on_stuck", False)),
        task_type_vocab=vocab,
        peer_mass=bool(trained_peer_mass),
        mp_peer_edges=bool(sidecar.get("mp_peer_edges", False)),
        partial_state_contract=str(trained_contract),
        load_seed_scale=_serving_knob("GNN_PREFIX_LOAD_SEED") if adopt_env else 0.0,
        concurrency_penalty=_serving_knob("GNN_PREFIX_CONCURRENCY_PENALTY") if adopt_env else 0.0,
    )
    print(
        f"[PREFIX SERVING] {label}: task_dim={task_feature_dim}+{onehot_dim} platform_dim="
        f"{platform_feature_dim} hidden={hidden_dim} emb={embedding_dim} layers={num_layers} "
        f"mp_peer_edges={options.mp_peer_edges} mp_off={declared_mp_off} "
        f"alpha={alpha_key or '(none)'} reuse={options.allow_replica_reuse} relax={options.relax_on_stuck} "
        f"load_seed={options.load_seed_scale} conc_penalty={options.concurrency_penalty}",
        flush=True,
    )
    return model, options, sidecar


def _fabric_topology(nodes: Sequence[Any]) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    fabrics = {id(getattr(n, "fabric", None)): getattr(n, "fabric", None) for n in nodes}
    fabrics.pop(id(None), None)
    if len(fabrics) != 1:
        raise PrefixServingError(
            f"prefix serving needs exactly one link fabric shared by every node; found "
            f"{len(fabrics)} — the partial-state columns are defined on the fabric's routes"
        )
    lt = next(iter(fabrics.values())).link_topology
    routes, links = lt.get("routes") or {}, lt.get("links") or {}
    if not routes or not links:
        raise PrefixServingError("link_topology has no routes/links")
    return routes, links


def _latency(entry: Any) -> float:
    return float(entry.get("latency", 0.0)) if isinstance(entry, dict) else float(entry)


def _demand_scale(task: Any) -> float:
    """The event's per-type demand multiplier (``application.demand_scale``), 1.0 when the
    trace carries none — the rule ``score_route_b_contention.load_demand_scales`` and
    ``dag_workload.load_workload_dag`` apply."""
    app_type = getattr(getattr(task, "application", None), "type", None) or {}
    per_type = app_type.get("demand_scale") or {}
    return float(per_type.get(task.type["name"], 1.0))


def attach_live_prefix_block(
    graph: Any,
    batch_tasks: Sequence[Any],
    *,
    nodes: Sequence[Any],
    peer_table: Mapping[int, Mapping[int, float]],
    options: PrefixServingOptions,
) -> Dict[str, Any]:
    """Attach the peer_affinity_v1 block to a live inference graph, in place.

    Returns a small diagnostics dict (in-batch pair count, peers the batch cannot see).
    Raises ``PrefixServingError`` on anything the cache builder would refuse.
    """
    n_tasks = len(batch_tasks)
    if int(graph.n_tasks) != n_tasks:
        raise PrefixServingError(f"graph has {int(graph.n_tasks)} tasks, batch has {n_tasks}")
    if not options.alpha_key:
        raise PrefixServingError(
            "no cap rung: the sidecar has no dag_alpha_key and GNN_PREFIX_ALPHA_KEY is unset"
        )
    vocab = options.task_type_vocab
    type_index = {name: k for k, name in enumerate(vocab)}
    tl = graph.task_logit_to_placement
    meta_by_key = graph.queue_key_to_platform_meta

    onehot4 = torch.zeros((n_tasks, len(vocab)), dtype=torch.float32)
    task_type_idx: Dict[int, int] = {}
    for t, task in enumerate(batch_tasks):
        name = str(task.type["name"])
        if name not in type_index:
            raise PrefixServingError(f"task type {name!r} not in vocab {list(vocab)}")
        onehot4[t, type_index[name]] = 1.0
        task_type_idx[t] = type_index[name]

    ptype_by_pid: Dict[int, str] = {}
    name_by_node_id: Dict[int, str] = {}
    for meta in meta_by_key.values():
        ptype_by_pid[int(meta["platform_id"])] = str(meta["platform_type"])
        nid, nname = int(meta["node_id"]), str(meta["node_name"])
        if name_by_node_id.get(nid, nname) != nname:
            raise PrefixServingError(f"node_id {nid} maps to two names")
        name_by_node_id[nid] = nname
    network_map_by_name = {str(n.node_name): getattr(n, "network_map", {}) or {} for n in nodes}

    # demands + caps: the sweep-side rule (cap = alpha x the max single candidate demand
    # on the node), over this batch's candidate set
    demand: Dict[Tuple[int, Tuple[int, int]], float] = {}
    for t, task in enumerate(batch_tasks):
        ttype = str(task.type["name"])
        mem = task.type.get("memoryRequirements") or {}
        scale = _demand_scale(task)
        for cand in tl.get(t, []):
            placement = (int(cand[0]), int(cand[1]))
            pid = placement[1]
            if pid not in ptype_by_pid:
                raise PrefixServingError(f"candidate platform_id {pid} absent from platform meta")
            ptype = ptype_by_pid[pid]
            if ptype not in mem:
                raise PrefixServingError(
                    f"no memoryRequirements[{ttype}][{ptype}] — refusing to invent a demand"
                )
            demand[(t, placement)] = float(mem[ptype]) * scale
    if not demand:
        raise PrefixServingError("batch has no candidate placements at all")
    peak: Dict[int, float] = {}
    for (_t, placement), d in demand.items():
        nid = int(placement[0])
        if nid not in peak or d > peak[nid]:
            peak[nid] = d
    alpha = float(options.alpha_key)
    caps: Dict[int, float] = {nid: alpha * m for nid, m in peak.items() if m > 0}
    cand_node_ids = sorted(peak)

    routes, links = _fabric_topology(nodes)
    route_hb: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for a in cand_node_ids:
        for b in cand_node_ids:
            h, bneck = (0, math.inf) if a == b else route_hops_and_bottleneck(
                routes, links, name_by_node_id[a], name_by_node_id[b])
            route_hb[(a, b)] = (float(h), float(bneck))
    mean_hop = {
        a: (sum(route_hb[(b, a)][0] for b in cand_node_ids if b != a) / max(1, len(cand_node_ids) - 1))
        for a in cand_node_ids
    }
    node_rank = krank_node_order({nid: caps.get(nid, 0.0) for nid in cand_node_ids}, mean_hop)

    ingress: Dict[Tuple[int, int], Tuple[str, ...]] = {}
    for t, task in enumerate(batch_tasks):
        src = str(task.node_name)
        own_nodes = {int(c[0]) for c in tl.get(t, [])}
        for nid in cand_node_ids:
            dst = name_by_node_id[nid]
            if src == dst:
                ingress[(t, nid)] = ()
                continue
            try:
                ingress[(t, nid)] = tuple(route_links(routes, src, dst))
            except KeyError:
                if nid in own_nodes:
                    raise
    core = frozenset(lk for lk in links if is_core_link(lk))

    # peers: the orchestrator's symmetric table restricted to this batch (batch-local
    # indices). A peer outside the batch is invisible to the decoder by contract — the
    # cache never saw one — and is reported, never silently dropped from the count.
    idx_by_id = {int(task.id): t for t, task in enumerate(batch_tasks)}
    peer_pairs: Dict[Tuple[int, int], float] = {}
    outside = 0
    for t, task in enumerate(batch_tasks):
        for j_id, b in (peer_table.get(int(task.id)) or {}).items():
            j = idx_by_id.get(int(j_id))
            if j is None:
                outside += 1
                continue
            peer_pairs[(t, j)] = float(b)
            peer_pairs[(j, t)] = float(b)
    if peer_pairs:
        keys = sorted(peer_pairs)
        peer_edge_index = torch.tensor([[i for i, _j in keys], [j for _i, j in keys]], dtype=torch.long)
        peer_edge_attr = torch.tensor([[math.log1p(peer_pairs[k] / 1e6)] for k in keys], dtype=torch.float32)
    else:
        peer_edge_index = torch.empty((2, 0), dtype=torch.long)
        peer_edge_attr = torch.empty((0, 1), dtype=torch.float32)
    node_exchange: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for a in cand_node_ids:
        for b in cand_node_ids:
            if a == b:
                node_exchange[(a, b)] = (0.0, 0.0)
                continue
            h, bneck = route_hb[(a, b)]
            entry = (network_map_by_name.get(name_by_node_id[a]) or {}).get(name_by_node_id[b])
            if entry is None:
                raise PrefixServingError(
                    f"no network_map[{name_by_node_id[a]}][{name_by_node_id[b]}] — the exchange "
                    "latency is undefined"
                )
            node_exchange[(a, b)] = (float(h) / (float(bneck) * 1024 * 1024), _latency(entry))
    if peer_pairs:
        peer_norm = (max(peer_pairs.values()) * max(pb for pb, _l in node_exchange.values())
                     + max(l for _pb, l in node_exchange.values()))
    else:
        peer_norm = 0.0
    if peer_norm <= 0.0:
        # No in-batch pair, or every candidate sits on ONE node (early in a live episode,
        # before the autoscaler has spread replicas): columns 7-9 are identically zero
        # whatever the norm, and 1.0 keeps the v2 context constructible. A cache dataset
        # without peers records 0.0 and is refused, which is the right outcome THERE —
        # offline, a peer corpus without peers is a broken dataset.
        peer_norm = 1.0
    cand_nodes = {t: [int(c[0]) for c in tl.get(t, [])] for t in range(n_tasks)}

    # Standing load per candidate node, in the cap's own units. `graph.queue_snapshot` is
    # the same queue depth the platform features are built from (feature_builder), keyed by
    # queue_key, and `queue_key_to_platform_meta` maps that to a node. We charge the
    # IMBALANCE (depth above the least-loaded candidate node) so the least-loaded node keeps
    # its whole cap and the mask can never empty, and convert a task count into demand units
    # with that node's mean candidate demand.
    base_load: Dict[int, float] = {}
    queue_depth_by_node: Dict[int, int] = {}
    if options.load_seed_scale > 0.0 or options.concurrency_penalty > 0.0:
        snapshot = getattr(graph, "queue_snapshot", None) or {}
        for qk, meta in meta_by_key.items():
            nid = int(meta["node_id"])
            if nid not in peak:
                continue
            queue_depth_by_node[nid] = queue_depth_by_node.get(nid, 0) + int(snapshot.get(qk, 0) or 0)
    if options.load_seed_scale > 0.0 and queue_depth_by_node:
        mean_demand: Dict[int, float] = {}
        for (_t, placement), d in demand.items():
            mean_demand.setdefault(int(placement[0]), []).append(float(d))
        mean_demand = {nid: (sum(v) / len(v)) for nid, v in mean_demand.items()}
        floor = min(queue_depth_by_node.get(nid, 0) for nid in cand_node_ids)
        for nid in cand_node_ids:
            excess = queue_depth_by_node.get(nid, 0) - floor
            if excess > 0 and mean_demand.get(nid, 0.0) > 0.0:
                base_load[nid] = options.load_seed_scale * excess * mean_demand[nid]

    graph.peer_edge_index = peer_edge_index
    graph.peer_edge_attr = peer_edge_attr
    graph.dag_edge_index = torch.empty((2, 0), dtype=torch.long)
    graph.dag_parents = {t: [] for t in range(n_tasks)}
    graph.task_type_onehot4 = onehot4
    graph.dag_task_type_vocab = list(vocab)
    graph.node_caps_by_alpha = {options.alpha_key: dict(caps)}
    graph.dag_primary_alpha_key = options.alpha_key
    graph.partial_state_ctx = {
        "node_caps": dict(caps),
        "node_caps_by_alpha": graph.node_caps_by_alpha,
        "demand": demand,
        "task_type_index": task_type_idx,
        "parents": graph.dag_parents,
        "route_hops_bneck": route_hb,
        "payload_bytes": 0.0,
        "transfer_norm": 0.0,
        "node_rank": dict(node_rank),
        "ingress_links": ingress,
        "core_links": sorted(core),
        "peer_pairs": peer_pairs,
        "node_exchange": node_exchange,
        "peer_norm": float(peer_norm),
        "cand_nodes": cand_nodes,
        "base_load": base_load,
        "queue_depth_by_node": queue_depth_by_node,
    }
    return {
        "n_pairs_in_batch": len(peer_pairs) // 2,
        "peers_outside_batch": outside,
        "seeded_nodes": len(base_load),
        "seeded_load_total": float(sum(base_load.values())),
    }


def _with_concurrency_penalty(score_fn: Any, graph: Any, penalty: float) -> Any:
    """Wrap a per-step score function so a candidate on a node already holding a deep
    queue scores lower, in proportion to its share of the batch's worst candidate queue.

    Soft on purpose: it reorders candidates and never removes one, so no decode can fail
    because of it and the relaxation counters keep meaning what they meant. The scale is
    the spread of the model's own scores at that step, so one penalty value behaves the
    same across checkpoints whose logits live on different scales."""
    depth = (getattr(graph, "partial_state_ctx", {}) or {}).get("queue_depth_by_node") or {}
    tl = graph.task_logit_to_placement
    worst = max(depth.values()) if depth else 0

    def wrapped(task_idx: int, committed: Mapping[int, Tuple[int, int]]) -> Sequence[float]:
        scores = [float(v) for v in score_fn(task_idx, committed)]
        if worst <= 0 or not scores:
            return scores
        spread = max(scores) - min(scores)
        if spread <= 0.0:
            spread = 1.0
        out = []
        for i, cand in enumerate(tl[task_idx]):
            share = depth.get(int(cand[0]), 0) / worst
            out.append(scores[i] - penalty * spread * share)
        return out

    return wrapped


def decode_prefix_conditioned(
    model: TaskPlacementGNN,
    graph: Any,
    options: PrefixServingOptions,
    *,
    stats: Optional[GnnDecodeRunStats] = None,
) -> Tuple[Tuple[int, int], ...]:
    """The registered decoder on a graph carrying the prefix block. A failed decode is an
    error here, never a fallback: a live gate that silently served shortest-queue for a
    batch would report a number that is not the model's."""
    if not options.alpha_key:
        raise PrefixServingError(
            "no cap rung: the sidecar has no dag_alpha_key and GNN_PREFIX_ALPHA_KEY is unset"
        )
    ctx = build_partial_state_context_from_graph(graph)
    caps_by_alpha = graph.partial_state_ctx["node_caps_by_alpha"]
    if options.alpha_key not in caps_by_alpha:
        raise PrefixServingError(f"alpha_key {options.alpha_key!r} not in node_caps_by_alpha")
    ctx.node_caps = caps_by_alpha[options.alpha_key]
    n_tasks = int(graph.n_tasks)
    tl = graph.task_logit_to_placement
    demands = {
        t: [float(ctx.demand[(t, (int(c[0]), int(c[1])))]) for c in tl[t]]
        for t in range(n_tasks)
    }
    score_fn = make_partial_state_score_fn(model, graph, ctx)
    if options.concurrency_penalty > 0.0:
        score_fn = _with_concurrency_penalty(score_fn, graph, options.concurrency_penalty)
    combo = decode_masked_topo_placement(
        [None] * n_tasks,
        tl,
        n_tasks,
        dag_parents=graph.dag_parents,
        node_caps=ctx.node_caps,
        demands=demands,
        score_fn=score_fn,
        stats=stats,
        allow_replica_reuse=options.allow_replica_reuse,
        relax_on_stuck=options.relax_on_stuck,
        initial_load=ctx.base_load or None,
    )
    if combo is None:
        raise PrefixServingError(
            "masked_topo decode found no cap-feasible plan and relaxation is "
            f"{'on' if options.relax_on_stuck else 'off'} — refusing to fall back"
        )
    return tuple((int(a), int(b)) for a, b in combo)
