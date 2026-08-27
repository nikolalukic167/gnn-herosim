"""
CE-reduced feature ablation for tabular MLP (task=3, platform=6, edge=2).

Slices full sequential graph caches in-process — no cache regen.
Matches archive/warmth_sparse/src/notebooks/train_near_rtt_ce_reduced_features.py / feature_builder ce_reduced layout.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from src.placement.topology_features import build_source_feature_context
from src.policy.tabular.graph_extraction import (
    TabularEdgeRow,
    _as_numpy,
    _directed_edge_attr_slice,
    _directed_edge_offset,
    _task_placement_map,
    _task_queue_key_map,
    generate_row_id,
    resolve_platform_pos,
    should_emit_graph,
)

REDUCED_TASK_FEATURE_DIM = 3
REDUCED_PLATFORM_FEATURE_DIM = 6
REDUCED_EDGE_FEATURE_DIM = 2
REDUCED_FEATURE_DIM = REDUCED_TASK_FEATURE_DIM + REDUCED_PLATFORM_FEATURE_DIM + REDUCED_EDGE_FEATURE_DIM

FULL_PLATFORM_QUEUE_DIM = 7
REDUCED_PLATFORM_QUEUE_DIM = 5
_PLATFORM_FEATURE_INDICES = [0, 1, 2, 3, 4, FULL_PLATFORM_QUEUE_DIM]
_EDGE_FEATURE_INDICES = [0, 1]

REDUCED_FEATURE_COLUMN_NAMES = [f"x_{i}" for i in range(REDUCED_FEATURE_DIM)]


def parent_dataset_id(dataset_id: Any) -> str:
    """Strip @os / @seq suffixes to the canonical co-sim parent id."""
    s = str(dataset_id or "")
    for sep in ("@os", "@seq"):
        idx = s.find(sep)
        if idx >= 0:
            s = s[:idx]
    return s


@lru_cache(maxsize=512)
def _load_parent_task_src_norms(parent_id: str, repo_root: str) -> Tuple[tuple, int]:
    ds_path = Path(repo_root) / "simulation_data" / parent_id
    optimal_path = ds_path / "optimal_result.json"
    if not optimal_path.exists():
        raise FileNotFoundError(f"Missing optimal_result.json for parent {parent_id!r}: {optimal_path}")

    with open(optimal_path, "r", encoding="utf-8") as f:
        optimal = json.load(f)

    task_results = optimal.get("taskResults")
    if not task_results:
        task_results = (optimal.get("stats") or {}).get("taskResults")
    if not task_results:
        raise ValueError(f"No taskResults in {optimal_path}")

    config = optimal.get("config") or {}
    infra = config.get("infrastructure") or {}
    nodes = infra.get("nodes") if isinstance(infra, dict) else None

    if not nodes:
        config_path = ds_path / "space_with_network.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                space_cfg = json.load(f)
            if isinstance(space_cfg.get("infrastructure"), dict):
                nodes = space_cfg["infrastructure"].get("nodes")
            if not nodes and isinstance(space_cfg.get("nodes"), list):
                nodes = space_cfg["nodes"]
    if not nodes:
        raise ValueError(f"Cannot resolve node list for parent {parent_id!r}")

    first_idx: Dict[str, int] = {}
    node_names: List[str] = []
    network_maps: Dict[str, Any] = {}
    for i, node in enumerate(nodes):
        name = node.get("node_name") or node.get("nodeName") or node.get("name")
        if name is not None:
            first_idx[str(name)] = i
            node_names.append(str(name))
            network_maps[str(name)] = (
                node.get("network_map") or node.get("networkMap") or {}
            )
    n_nodes = len(nodes)
    source_ctx = build_source_feature_context(
        node_names, network_maps, first_idx_by_name=first_idx, node_count=n_nodes
    )

    def _task_sort_key(tr: Mapping[str, Any]) -> int:
        for key in ("taskId", "task_id", "id"):
            if key in tr:
                return int(tr[key])
        raise KeyError(f"task result missing id keys in {parent_id}: {list(tr.keys())[:8]}")

    src_norms: List[float] = []
    for tr in sorted(task_results, key=_task_sort_key):
        src = tr.get("sourceNode") or tr.get("source_node") or ""
        src_norms.append(source_ctx.feature(str(src)))
    return tuple(src_norms), n_nodes


def enrich_task_features_with_src_norm(graph: Any, repo_root: Path) -> None:
    """Promote 2-d seq-cache task onehot to 3-d (onehot + src_norm) when needed."""
    task_features = graph.task_features
    if not isinstance(task_features, torch.Tensor):
        task_features = torch.as_tensor(task_features, dtype=torch.float32)
    if int(task_features.size(-1)) >= REDUCED_TASK_FEATURE_DIM:
        graph.task_features = task_features[:, :REDUCED_TASK_FEATURE_DIM].clone()
        return

    parent_id = str(getattr(graph, "parent_dataset_id", "") or parent_dataset_id(getattr(graph, "dataset_id", "")))
    if not parent_id:
        raise ValueError("Graph missing parent_dataset_id for src_norm enrichment")

    src_norms, _ = _load_parent_task_src_norms(parent_id, str(repo_root.resolve()))
    n_tasks = int(task_features.shape[0])
    if len(src_norms) < n_tasks:
        raise ValueError(
            f"Parent {parent_id!r} has {len(src_norms)} src_norm values but graph has {n_tasks} tasks"
        )
    src_col = torch.tensor(src_norms[:n_tasks], dtype=torch.float32).reshape(-1, 1)
    graph.task_features = torch.cat([task_features[:, :2], src_col], dim=1)


def apply_reduced_features_to_graph(graph: Any, repo_root: Path) -> Any:
    enrich_task_features_with_src_norm(graph, repo_root)
    graph.task_features = graph.task_features[:, :REDUCED_TASK_FEATURE_DIM].clone()
    graph.platform_features = graph.platform_features[:, _PLATFORM_FEATURE_INDICES].clone()
    if hasattr(graph, "edge_attr") and graph.edge_attr is not None and graph.edge_attr.numel() > 0:
        graph.edge_attr = graph.edge_attr[:, _EDGE_FEATURE_INDICES].clone()
    return graph


DIM22_FEATURE_DIM = 22  # 3-task + 14-platform + 5-edge
DIM22_FEATURE_COLUMN_NAMES = [f"x_{i}" for i in range(DIM22_FEATURE_DIM)]
DIM24_FEATURE_DIM = 24  # 3-task + 16-platform (pull obs) + 5-edge
DIM24_PLATFORM_FEATURE_DIM = 16
DIM24_FEATURE_COLUMN_NAMES = [f"x_{i}" for i in range(DIM24_FEATURE_DIM)]

# P5b (program_verdict_v1): dim22 + 3 candidate-relative queue columns.
# 25 is unambiguous against every other layout width (11 ce_reduced / 21 atomic21 /
# 22 dim22 / 24 dim24), so the input_dim -> layout fallback stays a lookup, never a guess.
CANDIDATE_RELATIVE_FEATURE_DIM = 3
DIM25CR_FEATURE_DIM = DIM22_FEATURE_DIM + CANDIDATE_RELATIVE_FEATURE_DIM
DIM25CR_FEATURE_COLUMN_NAMES = [f"x_{i}" for i in range(DIM25CR_FEATURE_DIM)]
CANDIDATE_RELATIVE_COLUMN_SPEC = (
    "x_22=q-min(q_cand), x_23=avg_rank(q)/max(1,n-1), x_24=(q-mean)/std (std==0 -> 0)"
)


def candidate_relative_queue_columns(queue_vals) -> np.ndarray:
    """Set-relative view of one task's candidate queues → [n, 3] float32.

    THE single definition of the P5b feature. Both the training extractor
    (``_extract_dim22_rows_for_task``) and the live scheduler
    (``MLPBatchScheduler._mlp_logits_from_bundle``) import this and nothing else may
    recompute it: a second copy of the formula is precisely how a served model comes to
    see features its weights were never fitted on (see the train/serve MP mismatch,
    scripts_cosim/test_train_serve_mp_parity.py).

    ``queue_vals`` is the *normalized* platform queue column (platform feature index
    ``FULL_PLATFORM_QUEUE_DIM``) for the candidates of ONE task, in logit order.

    Columns:
      0  q - min(q)            absolute headroom over the best candidate
      1  avg_rank(q)/(n-1)     within-set rank fraction; ties share their average rank
      2  (q - mean)/std        within-set z-score; std == 0 -> 0.0

    All three are invariant to adding a constant to every candidate (cols 1-2 are also
    scale-invariant), and a single-candidate set yields zeros — there is no choice to
    inform.
    """
    q = np.asarray(queue_vals, dtype=np.float64).reshape(-1)
    n = int(q.shape[0])
    if n == 0:
        return np.zeros((0, CANDIDATE_RELATIVE_FEATURE_DIM), dtype=np.float32)
    if not np.isfinite(q).all():
        raise ValueError(f"Non-finite candidate queue values: {q!r}")

    out = np.zeros((n, CANDIDATE_RELATIVE_FEATURE_DIM), dtype=np.float64)
    if n == 1:
        return out.astype(np.float32)

    out[:, 0] = q - q.min()

    # Average ranks for ties, so the columns are a function of the SET and not of the
    # order candidates happen to be enumerated in.
    order = np.argsort(q, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and q[order[j + 1]] == q[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j)
        i = j + 1
    out[:, 1] = ranks / float(n - 1)

    std = float(q.std())
    if std > 0.0:
        out[:, 2] = (q - q.mean()) / std

    return out.astype(np.float32)


# --- dim63crk: route_b stage 2 T1 layout (docs/lineages/route_b_v1/stage2-preregistration.md §2) ---
#
# dim63crk = dim25cr (25) + 38 partial-state columns computed per (task, candidate)
# edge GIVEN the partial assignment of the shared masked decoder (§4, masked_topo):
#
#   10 base partial-state columns (§2 cols 25-34):
#     0-3  per-type occupancy count on the candidate's node (4 types, sorted order)
#     4    committed memory demand on the candidate's node / cap_node(alpha)
#     5    remaining capacity AFTER hypothetically placing this task, / cap_node(alpha)
#     6    would-violate indicator (redundant with the mask; lets the score anticipate it)
#     7-8  min / max hop distance from committed parents' nodes to the candidate's
#          node (0 for a task with no parents)
#     9    sum over committed parents of n_hops * payload / bottleneck_bandwidth on the
#          parent->candidate route, / the per-dataset max of that quantity
#   24 krank one-hot columns (§2 cols 35-58): indicator of (candidate node's canonical
#     identity-free rank r, THIS task's type k); rank ascending (capacity at alpha,
#     mean hop, node name), padded to width KRANK_WIDTH = 6; rank-major, type-minor —
#     summing the block over a plan's edges reproduces the §9c/§9d plan-level
#     krank_cols construction exactly, which is what puts the pooled-krank closure
#     inside MLP(T1)'s hypothesis space by construction.
#   4 linkrank edge columns (§2 cols 59-62), registered no-op expectation (route_c
#     screen FAILED BY EXHAUSTION), included for feature parity:
#     34   max per-link co-use over the candidate's own ingress route counting this
#          task (0 when client and candidate are co-located)
#     35   number of candidate-route links already used by >= 1 committed task
#     36-37 the same two restricted to core links
#
# The retracted layout's "fraction of parents committed" column is NOT here: under
# §4's topological decode order it is identically 1 (a constant discriminator, the
# §9c defect class), and partial_state_columns enforces the all-parents-committed
# invariant loudly instead.

PARTIAL_STATE_BASE_DIM = 10
KRANK_WIDTH = 6  # registered pad width R = the route_b grid's max node count
KRANK_TYPES = 4  # task types on the route_b grids, in sorted-name order
KRANK_FEATURE_DIM = KRANK_WIDTH * KRANK_TYPES
LINKRANK_FEATURE_DIM = 4
PARTIAL_STATE_FEATURE_DIM = (
    PARTIAL_STATE_BASE_DIM + KRANK_FEATURE_DIM + LINKRANK_FEATURE_DIM
)
DIM63CRK_FEATURE_DIM = DIM25CR_FEATURE_DIM + PARTIAL_STATE_FEATURE_DIM
DIM63CRK_FEATURE_COLUMN_NAMES = [f"x_{i}" for i in range(DIM63CRK_FEATURE_DIM)]
_PARTIAL_STATE_EPS = 1e-12  # the scorer's feasibility EPS, kept in agreement

# Contract versioning, the queue_features.py pattern: a checkpoint's sidecar declares
# the partial-state contract it was trained under, and serving fails loudly on a
# mismatch instead of silently changing what the 38 columns mean. Only v1 exists;
# the machinery exists so a future change is a new contract, never an in-place edit.
PARTIAL_STATE_CONTRACT_V1 = "partial_state_v1"
VALID_PARTIAL_STATE_CONTRACTS = frozenset({PARTIAL_STATE_CONTRACT_V1})
DEFAULT_PARTIAL_STATE_CONTRACT = PARTIAL_STATE_CONTRACT_V1
PARTIAL_STATE_CONTRACT_ENV = "PARTIAL_STATE_CONTRACT"


class InvalidPartialStateContractError(ValueError):
    """Raised when a partial-state contract name is not recognized."""


class PartialStateContractMismatchError(ValueError):
    """Raised when a checkpoint's training contract differs from the serving one."""


def validate_partial_state_contract(contract: str) -> str:
    normalized = str(contract).strip().lower()
    if normalized not in VALID_PARTIAL_STATE_CONTRACTS:
        raise InvalidPartialStateContractError(
            f"Unknown partial-state contract {contract!r}; expected one of "
            f"{sorted(VALID_PARTIAL_STATE_CONTRACTS)}"
        )
    return normalized


def resolve_partial_state_contract(explicit: Optional[str] = None) -> str:
    """Explicit argument wins, then $PARTIAL_STATE_CONTRACT, then the default."""
    import os as _os

    if explicit is not None and str(explicit).strip():
        return validate_partial_state_contract(explicit)
    from_env = _os.environ.get(PARTIAL_STATE_CONTRACT_ENV, "").strip()
    if from_env:
        return validate_partial_state_contract(from_env)
    return DEFAULT_PARTIAL_STATE_CONTRACT


def require_matching_partial_state_contract(
    trained_contract: Optional[str], serving_contract: str, *, model_label: str
) -> None:
    """Fail loudly rather than serve a dim63crk checkpoint features it was never
    trained on. A sidecar-less trained_contract (None) is NOT accepted for the
    dim63crk layout — a checkpoint without a contract is not evidence."""
    serving = validate_partial_state_contract(serving_contract)
    if trained_contract is None:
        raise PartialStateContractMismatchError(
            f"{model_label} declares no partial-state contract in its sidecar; a "
            "dim63crk checkpoint without one cannot be served (the sidecar rule)."
        )
    trained = validate_partial_state_contract(trained_contract)
    if trained != serving:
        raise PartialStateContractMismatchError(
            f"{model_label} was trained under partial-state contract {trained!r} "
            f"but the run resolves to {serving!r}. Set "
            f"{PARTIAL_STATE_CONTRACT_ENV}={trained} or load a matching checkpoint."
        )


def krank_node_order(
    caps_at_alpha: Mapping[Any, float], mean_hop: Mapping[Any, float]
) -> Dict[Any, int]:
    """THE canonical identity-free node ranking (§2): ascending (capacity at alpha,
    mean hop to the other candidate-hosting nodes, node name). Single source — the
    cache builder and the offline-eval/serving path must both import this; the
    scripts_cosim analysis copy (route_b_coefficient_transfer.krank_cols) is pinned
    to the same ordering and independently verified by PP0'."""
    nodes = sorted(mean_hop)
    if sorted(caps_at_alpha) != nodes:
        missing = set(nodes) ^ set(caps_at_alpha)
        raise ValueError(f"krank_node_order: cap/hop node sets differ on {missing}")
    order = sorted(nodes, key=lambda n: (caps_at_alpha[n], mean_hop[n], n))
    if len(order) > KRANK_WIDTH:
        raise ValueError(
            f"krank_node_order: {len(order)} nodes exceed the registered pad "
            f"width {KRANK_WIDTH}"
        )
    return {n: i for i, n in enumerate(order)}


class PartialStateContext:
    """Static per-dataset context for partial_state_columns.

    Everything here is a function of the dataset/infrastructure alone — no decode
    state. The dynamic argument is `committed` (the partial assignment). Node keys
    and candidate keys are opaque to this module; the caller uses them consistently.

      node_caps        node -> cap_node(alpha); a node absent is uncapped
      demand           (task_id, candidate) -> memory demand of that task there
      node_of          candidate -> node
      task_type_index  task_id -> type index k in the sorted-type order (0..3)
      parents          task_id -> parent task ids (the batch DAG)
      route_hops_bneck (parent_node, cand_node) -> (n_hops, bottleneck_mbps);
                       a same-node pair MUST be present as (0, inf)
      payload_bytes    the uniform parent->child payload
      transfer_norm    per-dataset max of n_hops*payload/bottleneck over node pairs
                       (0 or negative disables col 9's normalization -> zeros)
      node_rank        node -> canonical rank (krank_node_order output)
      ingress_links    (task_id, node) -> link keys on that task's client->node
                       ingress route (empty when co-located / no fabric)
      core_links       the subset of link keys that are core links
    """

    def __init__(
        self,
        *,
        node_caps: Mapping[Any, float],
        demand: Mapping[Tuple[int, Any], float],
        node_of: Mapping[Any, Any],
        task_type_index: Mapping[int, int],
        parents: Mapping[int, Sequence[int]],
        route_hops_bneck: Mapping[Tuple[Any, Any], Tuple[float, float]],
        payload_bytes: float,
        transfer_norm: float,
        node_rank: Mapping[Any, int],
        ingress_links: Mapping[Tuple[int, Any], Sequence[str]],
        core_links: frozenset,
    ) -> None:
        self.node_caps = node_caps
        self.demand = demand
        self.node_of = node_of
        self.task_type_index = task_type_index
        self.parents = parents
        self.route_hops_bneck = route_hops_bneck
        self.payload_bytes = float(payload_bytes)
        self.transfer_norm = float(transfer_norm)
        self.node_rank = node_rank
        self.ingress_links = ingress_links
        self.core_links = core_links


def partial_state_columns(
    ctx: PartialStateContext,
    task_id: int,
    candidates: Sequence[Any],
    committed: Mapping[int, Any],
) -> np.ndarray:
    """The 38 dim63crk partial-state columns for ONE task's candidate set, given the
    partial assignment → [n_candidates, PARTIAL_STATE_FEATURE_DIM] float32.

    THE single definition (§2's one-definition rule, same as
    candidate_relative_queue_columns): the training extractor and the
    MLPBatchScheduler/offline-eval path import this and nothing else may recompute
    it. Layout: [occ x4, load/cap, remaining/cap, would_violate, min_hop, max_hop,
    transfer_norm, krank one-hot x24, linkrank x4].

    Invariant enforced loudly: every parent of `task_id` must already be committed —
    that is §4's topological-order guarantee, and a violation means the caller is
    not decoding in the registered order.
    """
    import math as _math

    n = len(candidates)
    out = np.zeros((n, PARTIAL_STATE_FEATURE_DIM), dtype=np.float64)

    occ: Dict[Any, List[float]] = {}
    load: Dict[Any, float] = {}
    for t, cand in committed.items():
        node = ctx.node_of[cand]
        k = int(ctx.task_type_index[t])
        occ.setdefault(node, [0.0] * KRANK_TYPES)[k] += 1.0
        load[node] = load.get(node, 0.0) + float(ctx.demand[(t, cand)])

    parent_ids = list(ctx.parents.get(task_id) or ())
    missing = [p for p in parent_ids if p not in committed]
    if missing:
        raise ValueError(
            f"partial_state_columns: task {task_id} scored with uncommitted "
            f"parents {missing} — the §4 topological decode order guarantees all "
            "parents are committed; the caller is not honoring it"
        )

    couse: Dict[str, int] = {}
    for t, cand in committed.items():
        for lk in ctx.ingress_links.get((t, ctx.node_of[cand]), ()):
            couse[lk] = couse.get(lk, 0) + 1

    k_self = int(ctx.task_type_index[task_id])
    if not (0 <= k_self < KRANK_TYPES):
        raise ValueError(f"partial_state_columns: type index {k_self} out of range")

    for i, cand in enumerate(candidates):
        node = ctx.node_of[cand]
        cap = float(ctx.node_caps.get(node, _math.inf))
        node_occ = occ.get(node)
        if node_occ is not None:
            out[i, 0:KRANK_TYPES] = node_occ
        d = float(ctx.demand[(task_id, cand)])
        ld = load.get(node, 0.0)
        if _math.isfinite(cap) and cap > 0.0:
            out[i, 4] = ld / cap
            out[i, 5] = (cap - ld - d) / cap
            out[i, 6] = 1.0 if ld + d > cap + _PARTIAL_STATE_EPS else 0.0
        else:
            # an uncapped node has no contention pressure, full headroom, and can
            # never violate — stated, not implied
            out[i, 4] = 0.0
            out[i, 5] = 1.0
            out[i, 6] = 0.0

        if parent_ids:
            hops: List[float] = []
            transfer = 0.0
            for p in parent_ids:
                pnode = ctx.node_of[committed[p]]
                pair = (pnode, node)
                if pair not in ctx.route_hops_bneck:
                    raise ValueError(
                        f"partial_state_columns: no route metrics for {pair!r}"
                    )
                h, bneck = ctx.route_hops_bneck[pair]
                hops.append(float(h))
                if h > 0:
                    transfer += float(h) * ctx.payload_bytes / float(bneck)
            out[i, 7] = min(hops)
            out[i, 8] = max(hops)
            out[i, 9] = (
                transfer / ctx.transfer_norm if ctx.transfer_norm > 0.0 else 0.0
            )

        r = int(ctx.node_rank[node])
        if not (0 <= r < KRANK_WIDTH):
            raise ValueError(
                f"partial_state_columns: node rank {r} outside pad width "
                f"{KRANK_WIDTH}"
            )
        out[i, PARTIAL_STATE_BASE_DIM + r * KRANK_TYPES + k_self] = 1.0

        links = ctx.ingress_links.get((task_id, node), ())
        if links:
            base = PARTIAL_STATE_BASE_DIM + KRANK_FEATURE_DIM
            counts = [couse.get(lk, 0) for lk in links]
            out[i, base + 0] = float(max(c + 1 for c in counts))
            out[i, base + 1] = float(sum(1 for c in counts if c >= 1))
            core = [c for lk, c in zip(links, counts) if lk in ctx.core_links]
            if core:
                out[i, base + 2] = float(max(c + 1 for c in core))
                out[i, base + 3] = float(sum(1 for c in core if c >= 1))

    return out.astype(np.float32)


def _batch_edge_feature_dims(
    platform_feature_dim: int, *, candidate_relative: bool = False,
    partial_state: bool = False
) -> Tuple[int, List[str], str]:
    """Map platform width → (total feature dim, column names, inference layout)."""
    if partial_state and platform_feature_dim != 14:
        raise ValueError(
            "the dim63crk partial-state layout is defined on the dim22/dim25cr "
            f"platform width (14) only; got platform_feature_dim={platform_feature_dim}"
        )
    if platform_feature_dim == 14:
        if partial_state:
            if not candidate_relative:
                raise ValueError(
                    "dim63crk is dim25cr + partial state; partial_state=True "
                    "requires candidate_relative=True"
                )
            return DIM63CRK_FEATURE_DIM, DIM63CRK_FEATURE_COLUMN_NAMES, "dim63crk"
        if candidate_relative:
            return DIM25CR_FEATURE_DIM, DIM25CR_FEATURE_COLUMN_NAMES, "dim25cr"
        return DIM22_FEATURE_DIM, DIM22_FEATURE_COLUMN_NAMES, "dim22"
    if platform_feature_dim == DIM24_PLATFORM_FEATURE_DIM:
        if candidate_relative:
            raise ValueError(
                "candidate-relative columns are defined on the dim22 platform layout only; "
                f"got platform_feature_dim={platform_feature_dim} (dim24)"
            )
        return DIM24_FEATURE_DIM, DIM24_FEATURE_COLUMN_NAMES, "dim24"
    raise ValueError(
        f"Unsupported platform_feature_dim={platform_feature_dim}; expected 14 (dim22) or 16 (dim24)"
    )


def _extract_dim22_rows_for_task(
    graph: Any,
    graph_id: str,
    parent_id: str,
    task_idx: int,
    seq_n_tasks: int,
    *,
    prefix_augment: bool = False,
    candidate_relative: bool = False,
    partial_state_block: Optional[np.ndarray] = None,
    target_override: Optional[int] = None,
) -> Tuple[List[TabularEdgeRow], Optional[str]]:
    """Shared dim22/dim24/dim25cr/dim63crk row builder for one task decision.

    ``partial_state_block`` ([n_candidates, PARTIAL_STATE_FEATURE_DIM], from the
    single-source partial_state_columns) appends the dim63crk partial-state
    columns; ``target_override`` replaces graph.y's label — the dim63crk path
    teacher-forces along a tied-optimal plan, whose per-task placement is the
    label for that decision, not the unconstrained sweep minimum in y."""
    task_placement_map = _task_placement_map(graph)
    task_queue_map = _task_queue_key_map(graph)

    if task_idx not in task_placement_map:
        return [], f"task_idx {task_idx} missing from task_logit_to_placement"

    if target_override is not None:
        target_class_idx = int(target_override)
    else:
        y_raw = getattr(graph, "y", None)
        if y_raw is None:
            return [], "missing y labels"
        target_class_idx = (
            int(y_raw[task_idx].item())
            if isinstance(y_raw, torch.Tensor) else int(y_raw[task_idx])
        )
    if target_class_idx < 0:
        return [], f"invalid label y[{task_idx}]={target_class_idx}"

    task_features = _as_numpy(graph.task_features)
    platform_features = _as_numpy(graph.platform_features)
    edge_attr_all = _as_numpy(graph.edge_attr)
    edge_attr_directed = _directed_edge_attr_slice(edge_attr_all)
    edge_offset = _directed_edge_offset(task_placement_map, task_idx)

    if task_features.shape[1] < 3:
        raise ValueError(
            f"graph {graph_id}: task_features has {task_features.shape[1]} cols, expected >= 3"
        )
    plat_dim = int(platform_features.shape[1])
    if plat_dim < 14:
        raise ValueError(
            f"graph {graph_id}: platform_features has {plat_dim} cols, expected >= 14"
        )
    feature_dim, _colnames, _layout = _batch_edge_feature_dims(
        plat_dim,
        candidate_relative=candidate_relative,
        partial_state=partial_state_block is not None,
    )
    if edge_attr_directed.shape[1] < 5:
        raise ValueError(
            f"graph {graph_id}: edge_attr_directed has {edge_attr_directed.shape[1]} cols, expected >= 5"
        )

    candidates = task_placement_map[task_idx]
    queue_keys = task_queue_map[task_idx]
    if len(candidates) != len(queue_keys):
        raise ValueError(
            f"task {task_idx}: placement count {len(candidates)} != queue key count {len(queue_keys)}"
        )

    decision_graph_id = f"{graph_id}@task{task_idx}"

    # Platform positions are resolved up front because the candidate-relative columns are a
    # function of the whole candidate SET, not of one edge — the same reason the serving
    # side computes them per task_boundaries group rather than per row.
    plat_pos_by_logit = [
        resolve_platform_pos(graph, int(node_id), int(plat_id), str(queue_keys[logit_idx]))
        for logit_idx, (node_id, plat_id) in enumerate(candidates)
    ]
    cand_rel = None
    if candidate_relative:
        cand_rel = candidate_relative_queue_columns(
            platform_features[plat_pos_by_logit, FULL_PLATFORM_QUEUE_DIM]
        )

    rows: List[TabularEdgeRow] = []
    for logit_idx, (node_id, plat_id) in enumerate(candidates):
        queue_key = str(queue_keys[logit_idx])
        plat_pos = plat_pos_by_logit[logit_idx]
        global_edge_idx = edge_offset + logit_idx
        if global_edge_idx >= edge_attr_directed.shape[0]:
            raise IndexError(
                f"global_edge_idx={global_edge_idx} out of range for directed edge_attr "
                f"(size={edge_attr_directed.shape[0]}, task_idx={task_idx}, logit_idx={logit_idx})"
            )

        x_task = task_features[task_idx, :3]
        x_plat = platform_features[plat_pos, :plat_dim]
        x_edge = edge_attr_directed[global_edge_idx, :5]
        parts = [x_task, x_plat, x_edge]
        if cand_rel is not None:
            parts.append(cand_rel[logit_idx])
        if partial_state_block is not None:
            if partial_state_block.shape[0] != len(candidates):
                raise ValueError(
                    f"partial_state_block rows {partial_state_block.shape[0]} != "
                    f"{len(candidates)} candidates for task {task_idx}"
                )
            parts.append(partial_state_block[logit_idx])
        features = np.concatenate(parts).astype(np.float64)
        if features.shape[0] != feature_dim:
            raise ValueError(
                f"Expected {feature_dim} features (plat_dim={plat_dim}), got {features.shape[0]}"
            )
        if not np.isfinite(features).all():
            raise ValueError(
                f"Non-finite features for graph={decision_graph_id} task={task_idx} logit={logit_idx}"
            )

        y_class = 1 if logit_idx == target_class_idx else 0
        rows.append(
            TabularEdgeRow(
                row_id=generate_row_id(parent_id, decision_graph_id, task_idx, logit_idx),
                parent_dataset_id=parent_id,
                graph_id=decision_graph_id,
                seq_step=task_idx,
                seq_n_tasks=seq_n_tasks,
                task_idx=task_idx,
                logit_idx=logit_idx,
                node_id=int(node_id),
                platform_id=int(plat_id),
                queue_key=queue_key,
                prefix_augment=prefix_augment,
                y_class=y_class,
                y_logit=target_class_idx,
                features=features,
            )
        )

    return rows, None


def extract_rows_dim22_from_batch_graph(
    graph: Any, graph_id: str, *, candidate_relative: bool = False
) -> Tuple[List[TabularEdgeRow], Optional[str]]:
    """Extract dim22/dim24 rows for every task in a batch PyG graph (prepare_graphs_cache.py).

    Batch cache platform features match inference: normalized queue (dim 7),
    shared_fate (dim 8), usage_ratio (dim 13); CACHE 5.6 also has node_cold_count /
    estimated_pull_remaining_sec (dims 14–15) → dim24.

    ``candidate_relative=True`` appends the three P5b set-relative queue columns → dim25cr.
    They are derived in-process from columns the cache already holds, so this needs no
    cache regeneration.
    """
    parent_id = str(
        getattr(graph, "parent_dataset_id", None) or parent_dataset_id(graph_id)
    )
    n_tasks = int(getattr(graph, "n_tasks"))
    if n_tasks <= 0:
        return [], "n_tasks <= 0"

    all_rows: List[TabularEdgeRow] = []
    for task_idx in range(n_tasks):
        rows, skip_reason = _extract_dim22_rows_for_task(
            graph,
            graph_id,
            parent_id,
            task_idx,
            n_tasks,
            candidate_relative=candidate_relative,
        )
        if skip_reason:
            return [], skip_reason
        all_rows.extend(rows)
    return all_rows, None


def build_partial_state_context_from_graph(graph: Any) -> "PartialStateContext":
    """PartialStateContext from a DAG cache graph's partial_state_ctx block
    (prepare_graphs_cache.attach_dag_partial_state_block). Node keys are node_ids;
    candidate keys are (node_id, platform_id) tuples."""
    psc = getattr(graph, "partial_state_ctx", None)
    if not psc:
        raise ValueError(
            "graph carries no partial_state_ctx — not a --dag-partial-state cache"
        )
    node_of = {}
    tl = graph.task_logit_to_placement
    for t in range(int(graph.n_tasks)):
        for cand in tl[t]:
            node_of[tuple(cand)] = int(cand[0])
    return PartialStateContext(
        node_caps=psc["node_caps"],
        demand=psc["demand"],
        node_of=node_of,
        task_type_index=psc["task_type_index"],
        parents=psc["parents"],
        route_hops_bneck=psc["route_hops_bneck"],
        payload_bytes=psc["payload_bytes"],
        transfer_norm=psc["transfer_norm"],
        node_rank=psc["node_rank"],
        ingress_links=psc["ingress_links"],
        core_links=frozenset(psc["core_links"]),
    )


def _extract_tied_plan_rows_from_batch_graph(
    graph: Any,
    graph_id: str,
    *,
    include_partial_state: bool,
    alpha_key: Optional[str] = None,
) -> Tuple[List[TabularEdgeRow], Optional[str]]:
    """Shared tied-optimal-plan row walk for the dim63crk (T1) and tied-dim25cr (T0)
    extractors (route_b stage 2, W2: one function, no second copy of the plan walk).

    Teacher forcing along the tied-optimal label SET (§5's any-of-K labels): for
    every tied plan k at `alpha_key` (default: the cache's primary alpha, 2.0),
    tasks are walked in the SAME topological order the §4 masked decoder uses
    (seq_decode.topological_task_order — imported, not re-typed, so train-time
    prefixes and decode-time prefixes agree by construction), the partial
    assignment is plan k's committed prefix, and the decision label is plan k's
    own placement. Rows carry the plan in their graph_id (``...#planK@taskT``),
    so a loss can group per plan; per-row CE over these rows is the
    teacher-forced factorization — the exact any-of-K marginalized CE
    (-log sum_k prod_t p) is assembled by the stage-2 trainer configs (B6) from
    the same rows.

    ``include_partial_state=True`` appends the 38 partial-state/krank/linkrank
    columns computed from the SAME committed prefix (dim63crk, T1 — arm A2).
    ``include_partial_state=False`` omits that block entirely (candidate-relative
    dim25cr rows, T0 — arm A3's tied-label mode): the walk still commits plan k's
    placements in topological order so `target_override` and the row's graph_id
    match the T1 extraction exactly (byte-identical targets/graph_ids on the same
    graph is the parity property W2's tests pin), but no partial-state block is
    computed or appended, so a T0 model never sees decoder-state features.

    Note the §2 capacity columns (when included) use cap_node(alpha_key):
    extracting a sensitivity rung (e.g. "3.0") changes cols 29-31 AND the label
    set together.
    """
    from src.policy.gnn.seq_decode import topological_task_order

    parent_id = str(
        getattr(graph, "parent_dataset_id", None) or parent_dataset_id(graph_id)
    )
    n_tasks = int(getattr(graph, "n_tasks"))
    if n_tasks <= 0:
        return [], "n_tasks <= 0"
    tied = getattr(graph, "tied_optimal_logit_plans", None)
    if not tied:
        return [], "graph carries no tied_optimal_logit_plans (not a DAG cache)"
    key = alpha_key or str(getattr(graph, "dag_primary_alpha_key", "2.0"))
    if key not in tied:
        return [], f"alpha_key {key!r} not in tied_optimal_logit_plans {sorted(tied)}"
    plans = tied[key]
    if not plans:
        return [], f"empty tied-optimal set at alpha={key}"

    ctx = None
    if include_partial_state:
        ctx = build_partial_state_context_from_graph(graph)
        caps_by_alpha = graph.partial_state_ctx["node_caps_by_alpha"]
        if key not in caps_by_alpha:
            return [], f"alpha_key {key!r} not in node_caps_by_alpha"
        ctx.node_caps = caps_by_alpha[key]

    order = topological_task_order(n_tasks, graph.dag_parents)
    tl = graph.task_logit_to_placement

    all_rows: List[TabularEdgeRow] = []
    for k, plan in enumerate(plans):
        if len(plan) != n_tasks:
            raise ValueError(
                f"{graph_id}: tied plan {k} has {len(plan)} entries, expected "
                f"{n_tasks}"
            )
        committed: Dict[int, Tuple[int, int]] = {}
        plan_rows: List[TabularEdgeRow] = []
        for task_idx in order:
            candidates = [tuple(c) for c in tl[task_idx]]
            block = (
                partial_state_columns(ctx, task_idx, candidates, committed)
                if include_partial_state
                else None
            )
            rows, skip_reason = _extract_dim22_rows_for_task(
                graph,
                f"{graph_id}#plan{k}",
                parent_id,
                task_idx,
                n_tasks,
                candidate_relative=True,
                partial_state_block=block,
                target_override=int(plan[task_idx]),
            )
            if skip_reason:
                return [], f"plan {k}: {skip_reason}"
            plan_rows.extend(rows)
            committed[task_idx] = candidates[int(plan[task_idx])]
        all_rows.extend(plan_rows)
    return all_rows, None


def extract_rows_dim63crk_from_batch_graph(
    graph: Any, graph_id: str, *, alpha_key: Optional[str] = None
) -> Tuple[List[TabularEdgeRow], Optional[str]]:
    """dim63crk (T1) training rows from a --dag-partial-state cache graph (B3).

    Thin wrapper around ``_extract_tied_plan_rows_from_batch_graph`` with the
    partial-state block included. See that function for the full contract.
    """
    return _extract_tied_plan_rows_from_batch_graph(
        graph, graph_id, include_partial_state=True, alpha_key=alpha_key
    )


def extract_rows_dim25cr_tied_from_batch_graph(
    graph: Any, graph_id: str, *, alpha_key: Optional[str] = None
) -> Tuple[List[TabularEdgeRow], Optional[str]]:
    """dim25cr (T0) rows teacher-forced along the SAME tied-optimal plan set A1/A2
    use (route_b stage 2, W2 / A3's label-parity repair).

    §3 requires "same labels, same alpha" across arms: A3's plain dim25cr path
    (``extract_rows_dim22_from_batch_graph`` with ``candidate_relative=True``)
    labels from ``graph.y`` — the unconstrained sweep minimum — which is NOT the
    alpha=2.0 constrained tied-optimal label set A1/A2 teacher-force along. This
    function walks the identical tied-plan loop as
    ``extract_rows_dim63crk_from_batch_graph`` (same topological order, same
    per-plan target_override, same ``#planK@taskT`` graph_ids) but with
    ``include_partial_state=False``, so a T0 model trains on the same labels
    without ever seeing a partial-state/krank/linkrank column. See that function
    for the full contract; this is a thin wrapper.
    """
    return _extract_tied_plan_rows_from_batch_graph(
        graph, graph_id, include_partial_state=False, alpha_key=alpha_key
    )


def extract_rows_dim22_from_graph(graph: Any, graph_id: str) -> Tuple[List[TabularEdgeRow], Optional[str]]:
    """Extract full 22-dim (3-task + 14-plat + 5-edge) rows from a seq cache graph.

    Unlike extract_rows_from_reduced_graph, this keeps all platform and edge features.
    The resulting model is trained with INFERENCE_FEATURE_LAYOUT=dim22, which matches
    build_inference_feature_bundle(use_dim22=True): 3-task + 14-plat + 5-edge = 22-d.

    Key fix vs ce_reduced: retains is_warm (edge dim 2), is_cold/shared_fate (plat dim 8),
    has_dnn1/dnn2 (plat dims 5-6), and temporal dims. Queue is still raw here (plat dim 7)
    vs normalized at inference, but this mismatch is diluted across 21 other features — same
    situation as the working batch_edge_mlp.pt trained on the v6.5 seq cache.
    """
    parent_id = str(getattr(graph, "parent_dataset_id", graph_id))
    seq_step = int(getattr(graph, "seq_step"))
    seq_n_tasks = int(getattr(graph, "seq_n_tasks"))
    prefix_augment = bool(getattr(graph, "prefix_augment", False))
    task_idx = seq_step
    return _extract_dim22_rows_for_task(
        graph,
        graph_id,
        parent_id,
        task_idx,
        seq_n_tasks,
        prefix_augment=prefix_augment,
    )


def dim22_rows_to_dataframe(rows: Sequence[TabularEdgeRow]):
    import pandas as pd

    if not rows:
        raise ValueError("Cannot build dataframe from empty dim22/dim24 row list")
    feat_dim = int(rows[0].features.shape[0])
    colnames = [f"x_{i}" for i in range(feat_dim)]
    records: List[Dict[str, Any]] = []
    for row in rows:
        if int(row.features.shape[0]) != feat_dim:
            raise ValueError(
                f"Mixed feature dims in rows: expected {feat_dim}, got {row.features.shape[0]}"
            )
        rec: Dict[str, Any] = {
            "row_id": row.row_id,
            "parent_dataset_id": row.parent_dataset_id,
            "graph_id": row.graph_id,
            "seq_step": row.seq_step,
            "seq_n_tasks": row.seq_n_tasks,
            "task_idx": row.task_idx,
            "logit_idx": row.logit_idx,
            "node_id": row.node_id,
            "platform_id": row.platform_id,
            "queue_key": row.queue_key,
            "prefix_augment": int(row.prefix_augment),
            "y_class": row.y_class,
            "y_logit": row.y_logit,
        }
        for col, val in zip(colnames, row.features):
            rec[col] = float(val)
        records.append(rec)
    return pd.DataFrame.from_records(records)


def validate_dim22_frame(df) -> Dict[str, Any]:
    if len(df) == 0:
        raise ValueError("Extracted dim22/dim24 dataframe is empty")
    if not (df["task_idx"] == df["seq_step"]).all():
        raise ValueError("Invariant violated: task_idx != seq_step")
    feature_cols = [c for c in df.columns if str(c).startswith("x_")]
    if not feature_cols:
        raise ValueError("No feature columns (x_*) found in extracted dataframe")
    feature_cols = sorted(feature_cols, key=lambda c: int(str(c).split("_", 1)[1]))
    n_feat = len(feature_cols)
    _LAYOUT_BY_WIDTH = {
        DIM22_FEATURE_DIM: "dim22",
        DIM24_FEATURE_DIM: "dim24",
        DIM25CR_FEATURE_DIM: "dim25cr",
        DIM63CRK_FEATURE_DIM: "dim63crk",
    }
    if n_feat not in _LAYOUT_BY_WIDTH:
        raise ValueError(
            f"Unexpected feature width {n_feat}; expected one of {sorted(_LAYOUT_BY_WIDTH)}"
        )
    feature_values = df[feature_cols].to_numpy()
    if not np.isfinite(feature_values).all():
        raise ValueError("Non-finite feature values in dim22/dim24 extracted dataframe")
    pos_per_graph = df.groupby("graph_id")["y_class"].sum()
    if not (pos_per_graph == 1).all():
        bad = pos_per_graph[pos_per_graph != 1]
        raise ValueError(f"Expected exactly one positive edge per graph; bad graphs: {bad.to_dict()}")
    return {
        "num_rows": int(len(df)),
        "num_graphs": int(df["graph_id"].nunique()),
        "num_parents": int(df["parent_dataset_id"].nunique()),
        "positives": int(df["y_class"].sum()),
        "feature_dim": int(n_feat),
        "inference_feature_layout": _LAYOUT_BY_WIDTH[n_feat],
    }


def extract_rows_from_reduced_graph(graph: Any, graph_id: str) -> Tuple[List[TabularEdgeRow], Optional[str]]:
    """Extract reduced-dim tabular rows for the active seq step on this graph."""
    parent_id = str(getattr(graph, "parent_dataset_id", graph_id))
    seq_step = int(getattr(graph, "seq_step"))
    seq_n_tasks = int(getattr(graph, "seq_n_tasks"))
    prefix_augment = bool(getattr(graph, "prefix_augment", False))
    task_idx = seq_step

    task_placement_map = _task_placement_map(graph)
    task_queue_map = _task_queue_key_map(graph)

    if task_idx not in task_placement_map:
        return [], f"task_idx {task_idx} missing from task_logit_to_placement"

    y_raw = getattr(graph, "y", None)
    if y_raw is None:
        return [], "missing y labels"

    target_class_idx = int(y_raw[task_idx].item()) if isinstance(y_raw, torch.Tensor) else int(y_raw[task_idx])
    if target_class_idx < 0:
        return [], f"invalid label y[{task_idx}]={target_class_idx}"

    task_features = _as_numpy(graph.task_features)
    platform_features = _as_numpy(graph.platform_features)
    edge_attr_all = _as_numpy(graph.edge_attr)
    edge_attr_directed = _directed_edge_attr_slice(edge_attr_all)
    edge_offset = _directed_edge_offset(task_placement_map, task_idx)

    candidates = task_placement_map[task_idx]
    queue_keys = task_queue_map[task_idx]
    if len(candidates) != len(queue_keys):
        raise ValueError(
            f"task {task_idx}: placement count {len(candidates)} != queue key count {len(queue_keys)}"
        )

    rows: List[TabularEdgeRow] = []
    for logit_idx, (node_id, plat_id) in enumerate(candidates):
        queue_key = str(queue_keys[logit_idx])
        plat_pos = resolve_platform_pos(graph, int(node_id), int(plat_id), queue_key)
        global_edge_idx = edge_offset + logit_idx
        if global_edge_idx >= edge_attr_directed.shape[0]:
            raise IndexError(
                f"global_edge_idx={global_edge_idx} out of range for directed edge_attr "
                f"(size={edge_attr_directed.shape[0]}, task_idx={task_idx}, logit_idx={logit_idx})"
            )

        x_task = task_features[task_idx]
        x_plat = platform_features[plat_pos]
        x_edge = edge_attr_directed[global_edge_idx]
        features = np.concatenate([x_task, x_plat, x_edge]).astype(np.float64)
        if features.shape[0] != REDUCED_FEATURE_DIM:
            raise ValueError(f"Expected {REDUCED_FEATURE_DIM} features, got {features.shape[0]}")
        if not np.isfinite(features).all():
            raise ValueError(f"Non-finite features for graph={graph_id} task={task_idx} logit={logit_idx}")

        y_class = 1 if logit_idx == target_class_idx else 0
        rows.append(
            TabularEdgeRow(
                row_id=generate_row_id(parent_id, graph_id, task_idx, logit_idx),
                parent_dataset_id=parent_id,
                graph_id=graph_id,
                seq_step=seq_step,
                seq_n_tasks=seq_n_tasks,
                task_idx=task_idx,
                logit_idx=logit_idx,
                node_id=int(node_id),
                platform_id=int(plat_id),
                queue_key=queue_key,
                prefix_augment=prefix_augment,
                y_class=y_class,
                y_logit=target_class_idx,
                features=features,
            )
        )

    return rows, None


def reduced_rows_to_dataframe(rows: Sequence[TabularEdgeRow]):
    import pandas as pd

    records: List[Dict[str, Any]] = []
    for row in rows:
        rec: Dict[str, Any] = {
            "row_id": row.row_id,
            "parent_dataset_id": row.parent_dataset_id,
            "graph_id": row.graph_id,
            "seq_step": row.seq_step,
            "seq_n_tasks": row.seq_n_tasks,
            "task_idx": row.task_idx,
            "logit_idx": row.logit_idx,
            "node_id": row.node_id,
            "platform_id": row.platform_id,
            "queue_key": row.queue_key,
            "prefix_augment": int(row.prefix_augment),
            "y_class": row.y_class,
            "y_logit": row.y_logit,
        }
        for col, val in zip(REDUCED_FEATURE_COLUMN_NAMES, row.features):
            rec[col] = float(val)
        records.append(rec)
    return pd.DataFrame.from_records(records)


def validate_reduced_frame(df) -> Dict[str, Any]:
    if len(df) == 0:
        raise ValueError("Extracted reduced dataframe is empty")

    if not (df["task_idx"] == df["seq_step"]).all():
        raise ValueError("Invariant violated: task_idx != seq_step")

    feature_values = df[REDUCED_FEATURE_COLUMN_NAMES].to_numpy()
    if not np.isfinite(feature_values).all():
        raise ValueError("Non-finite feature values in reduced extracted dataframe")

    pos_per_graph = df.groupby("graph_id")["y_class"].sum()
    if not (pos_per_graph == 1).all():
        bad = pos_per_graph[pos_per_graph != 1]
        raise ValueError(f"Expected exactly one positive edge per graph; bad graphs: {bad.to_dict()}")

    return {
        "num_rows": int(len(df)),
        "num_graphs": int(df["graph_id"].nunique()),
        "num_parents": int(df["parent_dataset_id"].nunique()),
        "positives": int(df["y_class"].sum()),
    }
