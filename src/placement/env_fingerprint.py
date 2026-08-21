"""Describe (and, later, enforce) the Python environment a run actually executed in.

Why this exists
---------------
`total_rtt` verdicts are cross-policy comparisons. A GNN arm whose numerics shift with the
machine while Knative's do not turns a gate into a measurement of the venue. That happened:
`run_full_corpus_siv1_live_gate.sh` calls ``pipenv run python3``, and ``pipenv run`` shells
straight past ``micromamba activate gnn`` — so the cluster gate ran under a third, unmanaged
venv (torch 2.12.0+cu130) while CLAUDE.md, the sbatch header and LINEAGES.md all asserted the
``gnn`` env (torch 2.5.1+cu121). Nothing in the result JSON recorded which interpreter served,
so the discrepancy survived three sessions as a phantom 11-26% GNN gap.

The single field that would have exposed it is ``executable`` — the interpreter path. It is
recorded here for exactly that reason, and deliberately *excluded* from the fingerprint hash:
two venues legitimately install the same versions at different paths, and a path-sensitive
hash would never match, which is the same as no check at all.

What counts as "the environment"
--------------------------------
Only libraries that can change floating-point results or scheduling behaviour. `torch` and
`torch_geometric` select different kernels and reduction orders across versions (the GIN
scatter aggregation in `src/policy/gnn/gnn_model.py` is the sensitive one); `numpy` changed
scalar promotion in 2.0 (NEP 50); `scipy`/`simpy` pin the simulation side. Jupyter, matplotlib
and friends cannot move a number and are not tracked — a fingerprint that trips on a plotting
upgrade gets disabled, and a disabled check protects nothing.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import sys
from typing import Any, Dict, List, Optional, Tuple

# Import name -> distribution name. Only packages that can change a result.
NUMERIC_PACKAGES: Tuple[Tuple[str, str], ...] = (
    ("torch", "torch"),
    ("numpy", "numpy"),
    ("torch_geometric", "torch-geometric"),
    ("scipy", "scipy"),
    ("simpy", "simpy"),
)

# Keys that must match for two runs to be numerically comparable. Interpreter path,
# hostname and thread counts are recorded but not hashed: the first two differ between
# venues by construction, and thread counts are a per-job knob rather than an env property.
STRICT_KEYS: Tuple[str, ...] = (
    "python_version",
    "torch",
    "torch_cuda",
    "numpy",
    "torch_geometric",
    "scipy",
    "simpy",
)

ALLOW_DRIFT_ENV = "HEROSIM_ALLOW_ENV_DRIFT"


class EnvironmentDriftError(RuntimeError):
    """Raised when the running environment does not match the declared lock."""


def _package_version(import_name: str) -> Optional[str]:
    """Version of an installed package, or None if it is not importable.

    Imports rather than reading distribution metadata, because the whole point is to
    report what *this* interpreter would actually load — a stale ``.dist-info`` left by a
    half-finished upgrade would otherwise be reported as the truth.
    """
    try:
        module = importlib.import_module(import_name)
    except Exception:
        return None
    version = getattr(module, "__version__", None)
    return str(version) if version is not None else "unknown"


def _torch_cuda_build() -> Optional[str]:
    """The CUDA toolkit a torch wheel was built against, e.g. '12.1'.

    This is the axis that separated the two cluster venvs (cu121 vs cu130) and it is not
    visible in ``torch.__version__`` alone once a local build strips the ``+cuXXX`` suffix.
    """
    try:
        import torch

        return getattr(torch.version, "cuda", None)
    except Exception:
        return None


def describe_python_env() -> Dict[str, Any]:
    """Everything about this interpreter that can move a number, plus how to find it."""
    description: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "executable": sys.executable,
        "hostname": platform.node(),
        "platform": platform.platform(),
    }
    for import_name, _dist_name in NUMERIC_PACKAGES:
        description[import_name] = _package_version(import_name)
    description["torch_cuda"] = _torch_cuda_build()
    description["thread_env"] = {
        name: os.environ.get(name)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONHASHSEED")
    }
    try:
        import torch

        description["torch_num_threads"] = int(torch.get_num_threads())
        description["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        description["torch_num_threads"] = None
        description["cuda_available"] = None
    return description


def strict_env_view(description: Dict[str, Any]) -> Dict[str, Any]:
    """The subset of `describe_python_env` that two comparable runs must agree on."""
    return {key: description.get(key) for key in STRICT_KEYS}


def env_fingerprint(description: Optional[Dict[str, Any]] = None) -> str:
    """Stable sha256 over the strict keys only."""
    if description is None:
        description = describe_python_env()
    payload = json.dumps(strict_env_view(description), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def format_env_banner(description: Optional[Dict[str, Any]] = None) -> str:
    """One-line human summary for run logs."""
    if description is None:
        description = describe_python_env()
    return (
        "[ENV] python={python_version} torch={torch} (cuda={torch_cuda}) "
        "numpy={numpy} pyg={torch_geometric} exe={executable}".format(**description)
    )


def diff_env(
    actual: Dict[str, Any], expected: Dict[str, Any]
) -> List[Tuple[str, Any, Any]]:
    """Strict-key differences as (key, actual, expected), in STRICT_KEYS order."""
    return [
        (key, actual.get(key), expected.get(key))
        for key in STRICT_KEYS
        if actual.get(key) != expected.get(key)
    ]
