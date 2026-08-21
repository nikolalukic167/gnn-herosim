"""Training-contract helpers: sweep-min labels, canonical parents, fail-loud RTT.

placements.jsonl is the only label/RTT ground truth. Graph instance IDs may carry
@os / @seq suffixes; RTT hash lookups and parent splits must use the canonical parent.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PlacementPlan = Dict[str, List[int]]
PlacementCombo = Tuple[Tuple[int, int], ...]


def canonical_parent_id(dataset_id: Any) -> str:
    """Strip @osN / @seqN augmentation suffixes from a graph instance id."""
    s = str(dataset_id or "")
    for sep in ("@os", "@seq"):
        idx = s.find(sep)
        if idx >= 0:
            s = s[:idx]
    return s


def combo_from_plan(plan: Mapping[str, Any]) -> PlacementCombo:
    keys = sorted(plan.keys(), key=lambda k: int(k))
    combo: List[Tuple[int, int]] = []
    for k in keys:
        pair = plan[k]
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            raise ValueError(f"Invalid placement entry for task {k}: {pair!r}")
        combo.append((int(pair[0]), int(pair[1])))
    return tuple(combo)


def plan_from_combo(combo: PlacementCombo) -> PlacementPlan:
    return {
        str(task_idx): [int(node_id), int(plat_id)]
        for task_idx, (node_id, plat_id) in enumerate(combo)
    }


def load_sweep_minimum(
    jsonl_path: Path,
    *,
    rtt_eps: float = 1e-9,
) -> Tuple[PlacementPlan, float, PlacementCombo]:
    """
    Derive (placement_plan, min_rtt, combo) from placements.jsonl.

    Tie policy: among rows within rtt_eps of the minimum RTT, pick the
    lexicographically smallest combo. Fail loud on missing/empty/malformed input.
    """
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Missing placements.jsonl: {jsonl_path}")

    min_rtt: Optional[float] = None
    best_combo: Optional[PlacementCombo] = None
    n_rows = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{jsonl_path}:{line_number}: invalid JSON") from exc
            if not isinstance(rec, dict):
                raise RuntimeError(f"{jsonl_path}:{line_number}: expected object")
            plan = rec.get("placement_plan")
            rtt = rec.get("rtt")
            if plan is None or rtt is None:
                raise RuntimeError(
                    f"{jsonl_path}:{line_number}: missing placement_plan or rtt"
                )
            try:
                rtt_f = float(rtt)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"{jsonl_path}:{line_number}: bad rtt={rtt!r}") from exc
            if not math.isfinite(rtt_f):
                raise RuntimeError(f"{jsonl_path}:{line_number}: non-finite rtt={rtt_f}")
            combo = combo_from_plan(plan)
            n_rows += 1
            if min_rtt is None or rtt_f < min_rtt - rtt_eps:
                min_rtt = rtt_f
                best_combo = combo
            elif abs(rtt_f - float(min_rtt)) <= rtt_eps:
                if best_combo is None or combo < best_combo:
                    best_combo = combo

    if n_rows == 0 or min_rtt is None or best_combo is None:
        raise RuntimeError(f"Empty placements.jsonl: {jsonl_path}")

    return plan_from_combo(best_combo), float(min_rtt), best_combo


def assert_zero_parent_overlap(
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
) -> None:
    train_p = {canonical_parent_id(x) for x in train_ids}
    val_p = {canonical_parent_id(x) for x in val_ids}
    test_p = {canonical_parent_id(x) for x in test_ids}
    leaks = (train_p & val_p) | (train_p & test_p) | (val_p & test_p)
    if leaks:
        raise RuntimeError(
            f"Canonical-parent leak across splits ({len(leaks)} parents); "
            f"examples={sorted(leaks)[:5]}"
        )


def split_ids_by_canonical_parent(
    graphs: Sequence[Any],
    dataset_ids: Sequence[str],
    *,
    test_size: float = 0.3,
    val_fraction_of_holdout: float = 0.5,
    random_state: int = 42,
    parent_ids: Optional[Sequence[str]] = None,
) -> Tuple[List[Any], List[str], List[Any], List[str], List[Any], List[str]]:
    """Split graphs by canonical parent (default 70/15/15 when test_size=0.3)."""
    from sklearn.model_selection import train_test_split

    if len(graphs) != len(dataset_ids):
        raise RuntimeError(f"graphs ({len(graphs)}) != dataset_ids ({len(dataset_ids)})")
    if parent_ids is not None and len(parent_ids) != len(dataset_ids):
        raise RuntimeError(
            f"parent_ids ({len(parent_ids)}) != dataset_ids ({len(dataset_ids)})"
        )

    by_parent: Dict[str, List[Tuple[Any, str]]] = {}
    for idx, (graph, graph_id) in enumerate(zip(graphs, dataset_ids)):
        if parent_ids is not None:
            parent = canonical_parent_id(parent_ids[idx])
        else:
            parent = canonical_parent_id(
                getattr(graph, "parent_dataset_id", None) or graph_id
            )
        by_parent.setdefault(parent, []).append((graph, graph_id))

    parent_keys = list(by_parent.keys())
    if len(parent_keys) < 3:
        raise RuntimeError(
            f"Need >=3 canonical parents for train/val/test; got {len(parent_keys)}"
        )

    train_parents, temp_parents = train_test_split(
        parent_keys, test_size=test_size, random_state=random_state
    )
    val_parents, test_parents = train_test_split(
        temp_parents, test_size=val_fraction_of_holdout, random_state=random_state
    )

    def _flatten(parents: Iterable[str]) -> Tuple[List[Any], List[str]]:
        out_g: List[Any] = []
        out_ids: List[str] = []
        for parent in parents:
            for graph, graph_id in by_parent[parent]:
                out_g.append(graph)
                out_ids.append(graph_id)
        return out_g, out_ids

    train_graphs, train_ids = _flatten(train_parents)
    val_graphs, val_ids = _flatten(val_parents)
    test_graphs, test_ids = _flatten(test_parents)
    assert_zero_parent_overlap(train_ids, val_ids, test_ids)
    return train_graphs, train_ids, val_graphs, val_ids, test_graphs, test_ids


def topology_size_of_dataset(dataset_id: Any, corpus_root: Path) -> int:
    """Server-node count for one dataset, read from its `infrastructure.json`.

    Not stored as a graph/cache attribute -- `generate_infrastructure.py` never wrote
    `server_node_count` into a dataset's metadata, so this reads it back from the same
    place `topology_transfer_v1`'s size axis actually lives: the node names in
    `network_maps`. A server is any node that is not a client (see `CLIENT_NODE_PREFIX`
    in `src.placement.topology_features`, the single source of truth for that split).
    """
    from src.placement.topology_features import CLIENT_NODE_PREFIX

    infra_path = Path(corpus_root) / str(canonical_parent_id(dataset_id)) / "infrastructure.json"
    with open(infra_path) as f:
        infra = json.load(f)
    node_names = infra["network_maps"].keys()
    return sum(1 for n in node_names if not str(n).startswith(CLIENT_NODE_PREFIX))


def topology_sizes_by_parent(
    dataset_ids: Sequence[str], corpus_root: Path
) -> Dict[str, int]:
    """`{canonical_parent_id: server_node_count}` for every unique parent in `dataset_ids`."""
    sizes: Dict[str, int] = {}
    for dsid in dataset_ids:
        parent = canonical_parent_id(dsid)
        if parent not in sizes:
            sizes[parent] = topology_size_of_dataset(parent, corpus_root)
    return sizes


def split_ids_by_topology_size(
    graphs: Sequence[Any],
    dataset_ids: Sequence[str],
    sizes_by_parent: Mapping[str, int],
    *,
    train_sizes: Sequence[int] = (20, 28, 40),
    held_out_sizes: Sequence[int] = (60, 80),
    val_fraction_of_train: float = 0.15,
    random_state: int = 42,
) -> Tuple[List[Any], List[str], List[Any], List[str], List[Any], List[str]]:
    """Split graphs by topology size: train on `train_sizes`, hold out `held_out_sizes`.

    This is topology_transfer_v1's inductive-generalization split, not a random one:
    "does the model transfer to LARGER topologies it never trained on" is unanswerable
    if train/test mix sizes. `val` is drawn from a held-out slice of `train_sizes`
    parents only -- never from `held_out_sizes` -- so model selection cannot peek at
    the transfer question the test split exists to answer.

    Uses a plain stdlib shuffle rather than sklearn's `train_test_split`, deliberately:
    this keeps the split usable on environments with a broken/mismatched scipy install
    (hit on datalab 2026-08-20 -- sklearn imports scipy transitively, `canonical_parent`
    split does not get this treatment since existing frozen reports depend on its exact
    sklearn shuffling for reproducibility, but nothing has been reported under
    `topology_size` yet).
    """
    import random as _random

    if len(graphs) != len(dataset_ids):
        raise RuntimeError(f"graphs ({len(graphs)}) != dataset_ids ({len(dataset_ids)})")

    train_size_set = set(train_sizes)
    held_out_size_set = set(held_out_sizes)
    registered = train_size_set | held_out_size_set

    by_parent: Dict[str, List[Tuple[Any, str]]] = {}
    for graph, graph_id in zip(graphs, dataset_ids):
        parent = canonical_parent_id(graph_id)
        by_parent.setdefault(parent, []).append((graph, graph_id))

    unknown = sorted(set(by_parent) - set(sizes_by_parent))
    if unknown:
        raise KeyError(f"no topology size recorded for {len(unknown)} parents, e.g. {unknown[:3]}")
    off_ladder = {
        parent: sizes_by_parent[parent] for parent in by_parent if sizes_by_parent[parent] not in registered
    }
    if off_ladder:
        raise RuntimeError(
            f"{len(off_ladder)} parents have a topology size outside the registered ladder "
            f"train={sorted(train_size_set)} held_out={sorted(held_out_size_set)}: "
            f"{dict(list(off_ladder.items())[:3])}"
        )

    train_pool_parents = [p for p in by_parent if sizes_by_parent[p] in train_size_set]
    held_out_parents = [p for p in by_parent if sizes_by_parent[p] in held_out_size_set]
    if not train_pool_parents:
        raise RuntimeError(f"no parents at the train sizes {sorted(train_size_set)}")
    if not held_out_parents:
        raise RuntimeError(f"no parents at the held-out sizes {sorted(held_out_size_set)}")

    shuffled = sorted(train_pool_parents)  # sort first: dict/set order is not guaranteed
    _random.Random(random_state).shuffle(shuffled)
    n_val = math.ceil(len(shuffled) * val_fraction_of_train)
    val_parents, train_parents = shuffled[:n_val], shuffled[n_val:]

    def _flatten(parents: Iterable[str]) -> Tuple[List[Any], List[str]]:
        out_g: List[Any] = []
        out_ids: List[str] = []
        for parent in parents:
            for graph, graph_id in by_parent[parent]:
                out_g.append(graph)
                out_ids.append(graph_id)
        return out_g, out_ids

    train_graphs, train_ids = _flatten(train_parents)
    val_graphs, val_ids = _flatten(val_parents)
    test_graphs, test_ids = _flatten(held_out_parents)
    assert_zero_parent_overlap(train_ids, val_ids, test_ids)
    return train_graphs, train_ids, val_graphs, val_ids, test_graphs, test_ids


def lookup_opt_rtt(
    dataset_id: Any,
    optimal_rtt_map: Mapping[str, float],
    *,
    parent_dataset_id: Any = None,
) -> float:
    key = str(dataset_id or "")
    if key in optimal_rtt_map:
        return float(optimal_rtt_map[key])
    parent = canonical_parent_id(parent_dataset_id or key)
    if parent in optimal_rtt_map:
        return float(optimal_rtt_map[parent])
    raise KeyError(f"opt_rtt missing for dataset_id={key!r} parent={parent!r}")


def lookup_rtt_hash(
    placement_rtt_hash_table: Mapping[Tuple[str, PlacementCombo], float],
    dataset_id: Any,
    combo: PlacementCombo,
    *,
    parent_dataset_id: Any = None,
) -> float:
    parent = canonical_parent_id(parent_dataset_id or dataset_id)
    key = (parent, combo)
    if key not in placement_rtt_hash_table:
        raw = str(dataset_id or "")
        alt = (raw, combo)
        if alt in placement_rtt_hash_table:
            return float(placement_rtt_hash_table[alt])
        raise KeyError(
            f"RTT hash miss for parent={parent!r} combo={combo!r} (graph_id={raw!r})"
        )
    return float(placement_rtt_hash_table[key])


def require_rtt_parent_coverage(
    dataset_ids: Sequence[str],
    optimal_rtt_map: Mapping[str, float],
    *,
    placement_rtt_parent_ids: Optional[Iterable[str]] = None,
    context: str = "RTT identity",
) -> None:
    missing_opt: List[str] = []
    missing_hash: List[str] = []
    hash_parents = set(placement_rtt_parent_ids) if placement_rtt_parent_ids is not None else None
    for graph_id in dataset_ids:
        parent = canonical_parent_id(graph_id)
        if parent not in optimal_rtt_map and graph_id not in optimal_rtt_map:
            missing_opt.append(graph_id)
        if hash_parents is not None and parent not in hash_parents:
            missing_hash.append(graph_id)
    if missing_opt:
        raise RuntimeError(
            f"{context}: opt_rtt missing for {len(missing_opt)} IDs (examples={missing_opt[:5]})"
        )
    if missing_hash:
        raise RuntimeError(
            f"{context}: RTT hash missing for {len(missing_hash)} IDs (examples={missing_hash[:5]})"
        )
