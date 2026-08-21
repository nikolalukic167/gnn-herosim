from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _normalize_mapping(raw_mapping: Dict[Any, Any]) -> Dict[int, str]:
    return {int(k): str(v) for k, v in raw_mapping.items()}


def _ordered_keys_from_scenario(scenario: Dict[str, Any]) -> List[str]:
    keys = list(scenario.keys())
    ordered: List[str] = []

    for fixed_key in ("network_bandwidth", "cluster_size"):
        if fixed_key in scenario:
            ordered.append(fixed_key)

    device_keys = sorted(k for k in keys if k.startswith("device_prop_"))
    workload_keys = sorted(k for k in keys if k.startswith("workload_"))
    ordered.extend(device_keys)
    ordered.extend(workload_keys)

    remaining = sorted(k for k in keys if k not in set(ordered))
    ordered.extend(remaining)
    return ordered


def _load_from_json(sample_json_path: Path) -> Tuple[np.ndarray, Dict[int, str]]:
    with open(sample_json_path, "r") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {sample_json_path}")

    # Optional richer format:
    # {
    #   "sample": [..],
    #   "mapping": {"0": "network_bandwidth", ...}
    # }
    if "sample" in payload and "mapping" in payload:
        sample_values = payload["sample"]
        raw_mapping = payload["mapping"]
        if not isinstance(sample_values, list) or not isinstance(raw_mapping, dict):
            raise ValueError(
                f"Invalid sample/mapping structure in {sample_json_path}"
            )
        mapping = _normalize_mapping(raw_mapping)
        sample = np.array(sample_values, dtype=np.float64)
        return sample, mapping

    # Simple flat scenario format:
    # {
    #   "network_bandwidth": 100.0,
    #   "cluster_size": 50.0,
    #   ...
    # }
    scenario = {str(k): float(v) for k, v in payload.items()}
    ordered_keys = _ordered_keys_from_scenario(scenario)
    mapping = {idx: key for idx, key in enumerate(ordered_keys)}
    sample = np.array([scenario[key] for key in ordered_keys], dtype=np.float64)
    return sample, mapping


def load_primary_sample_and_mapping(
    sample_json_path: Path,
    samples_npy_path: Path,
    mapping_pkl_path: Path,
    sample_index: int = 0,
) -> Tuple[np.ndarray, Dict[int, str], str]:
    """
    Load one scenario sample and mapping with JSON-first fallback behavior.

    Priority:
    1) sample_json_path (if present)
    2) samples_npy_path + mapping_pkl_path
    """
    if sample_json_path.exists():
        sample, mapping = _load_from_json(sample_json_path)
        return sample, mapping, f"json:{sample_json_path}"

    if not samples_npy_path.exists():
        raise FileNotFoundError(
            f"Missing sample sources. Expected JSON at {sample_json_path} or NPY at {samples_npy_path}."
        )
    if not mapping_pkl_path.exists():
        raise FileNotFoundError(
            f"Missing mapping source. Expected JSON at {sample_json_path} or PKL mapping at {mapping_pkl_path}."
        )

    samples = np.load(samples_npy_path)
    if samples.ndim == 1:
        sample = np.array(samples, dtype=np.float64)
    else:
        if samples.shape[0] == 0:
            raise ValueError(f"No rows in sample array: {samples_npy_path}")
        sample = np.array(samples[sample_index], dtype=np.float64)

    with open(mapping_pkl_path, "rb") as f:
        raw_mapping = pickle.load(f)
    mapping = _normalize_mapping(raw_mapping)
    return sample, mapping, f"npy+pkl:{samples_npy_path},{mapping_pkl_path}"


def ensure_workload_params(
    sample: np.ndarray,
    mapping: Dict[int, str],
    apps: List[str],
) -> Tuple[np.ndarray, Dict[int, str]]:
    """Extend a sample/mapping in memory so every app has a `workload_<app>` parameter.

    A grid preset may name task types the shared sampled space never had — the generator
    synthesizes `wsc`/`prewarm`/`replicas` entries for those, but `prepare_workloads` then
    fails loud on the missing workload factor. Rather than rewrite the shared
    `sample_simple.json` / `lhs_samples_simple_mapping.pkl` (which every other grid reads,
    at fixed indices), grow the copy this run uses.

    New task types take over the *positions* of the existing ones, so a substituted pair
    like ("cnn", "rf") inherits the workload factors ("dnn1", "dnn2") would have had and
    the grid's workload semantics are preserved.
    """
    present = {name for name in mapping.values()}
    missing = [a for a in apps if f"workload_{a}" not in present]
    if not missing:
        return sample, mapping

    donor_indices = sorted(
        idx for idx, name in mapping.items() if name.startswith("workload_")
    )
    if not donor_indices:
        raise RuntimeError(
            f"Cannot synthesize workload parameters for {missing}: the sample mapping has "
            f"no workload_* entry to inherit a factor from. Mapping: {sorted(mapping.values())}"
        )

    extended = dict(mapping)
    values = list(np.asarray(sample, dtype=np.float64))
    next_index = max(extended) + 1
    for position, app_name in enumerate(missing):
        donor = donor_indices[position % len(donor_indices)]
        extended[next_index] = f"workload_{app_name}"
        values.append(float(sample[donor]))
        next_index += 1

    return np.array(values, dtype=np.float64), extended
