"""partial_state_v5: the legacy task block and platform replica flags cover all four task types.

The legacy blocks were written for the two-type workloads: the task block is [dnn1, dnn2, source] and the platform
block carries has_dnn1 / has_dnn2, so an rf or cnn task is an all-zero type row and a platform's rf / cnn replicas are
not a feature at all. Under PARTIAL_STATE_CONTRACT_V5 (reduced_features.FOUR_TYPE_CONTRACTS) the blocks gain APPENDED
columns, so no existing column index moves:

    task_features      [dnn1, dnn2, source, rf, cnn]                    3 -> 5
    platform_features  [<14 or 16 legacy columns>, has_rf, has_cnn]     +2

The cache builder (prepare_graphs_cache.build_graph) and the live builder (feature_builder) both call this module, so
the two paths cannot define the extra columns differently. Task-type vocabularies used elsewhere in the legacy blocks
(target concurrency, temporal remainders) stay two-type on purpose: widening them would move existing columns and break
cache <-> live parity (see feature_builder's note on target_concurrency).
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

EXTRA_TASK_TYPES = ("rf", "cnn")
TASK_EXTRA_DIM = len(EXTRA_TASK_TYPES)
PLATFORM_EXTRA_DIM = len(EXTRA_TASK_TYPES)


def four_type_enabled(contract: str | None = None) -> bool:
    from src.policy.tabular.reduced_features import FOUR_TYPE_CONTRACTS, resolve_partial_state_contract

    return resolve_partial_state_contract(contract) in FOUR_TYPE_CONTRACTS


def task_extra_columns(task_type_names: Iterable[str]) -> np.ndarray:
    """[n_tasks, 2] one-hot over (rf, cnn); a dnn1 / dnn2 row is zero here (it has its legacy columns)."""
    names = [str(n) for n in task_type_names]
    return np.asarray([[1.0 if n == t else 0.0 for t in EXTRA_TASK_TYPES] for n in names],
                      dtype=np.float64).reshape(len(names), TASK_EXTRA_DIM)


def platform_extra_columns(replica_types_by_platform: Sequence[Iterable[str]]) -> np.ndarray:
    """[n_platforms, 2] replica flags (has_rf, has_cnn) from each platform's set of task types it holds a replica of."""
    sets = [frozenset(str(t) for t in types) for types in replica_types_by_platform]
    return np.asarray([[1.0 if t in s else 0.0 for t in EXTRA_TASK_TYPES] for s in sets],
                      dtype=np.float64).reshape(len(sets), PLATFORM_EXTRA_DIM)


def replica_types_from_system_state(system_state, platform_keys: Sequence[tuple]) -> list:
    """Live side: for each (node_id, platform_id) the task types it holds a replica of, from SystemState.replicas."""
    held: Mapping[tuple, set] = {}
    for task_type, replicas in system_state.replicas.items():
        for node, plat in replicas:
            held.setdefault((int(node.id), int(plat.id)), set()).add(str(task_type))
    return [held.get(k, set()) for k in platform_keys]


def widen_graph(graph) -> None:
    """Cache side: append the held-apart extra columns to task_features / platform_features, in place, and drop them."""
    import torch

    te = getattr(graph, "four_type_task_extra", None)
    pe = getattr(graph, "four_type_platform_extra", None)
    if te is None or pe is None:
        raise ValueError("partial_state_v5: graph carries no four_type extras (built under another contract)")
    graph.task_features = torch.cat([graph.task_features, te.to(graph.task_features.dtype)], dim=1)
    graph.platform_features = torch.cat([graph.platform_features, pe.to(graph.platform_features.dtype)], dim=1)
    del graph.four_type_task_extra
    del graph.four_type_platform_extra
