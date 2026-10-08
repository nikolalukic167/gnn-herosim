"""workload_fix_v1 W2 — per-pair peer-exchange payload sampler.

Opt-in. The legacy payload (``200 MB * 10**U(-1, 1)``, 20 MB-2 GB) stays inline in the minters; this module is
used only when a minter is asked for ``wf1_v1``. Units are bytes with 1 MB = 1e6 bytes, the unit the legacy
``X_SCALE_BYTES = 200e6`` already uses.

Distribution (docs/lineages/workload_fix_v1.md, W2): log-normal with median 4 MB and sigma 1.2 in natural-log
space, plus a heavy tier — 10 % of pairs drawn log-uniform on 50-500 MB. The tier is assumed and swept; the body
approximates Eismann et al. (IEEE Software 2021).
"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Sequence

PAYLOAD_SAMPLERS = ("legacy", "wf1_v1")

MB = 1e6
MEDIAN_BYTES = 4 * MB
SIGMA = 1.2
HEAVY_SHARE = 0.10
HEAVY_LOW_BYTES = 50 * MB
HEAVY_HIGH_BYTES = 500 * MB


def payload_rng(seed: int) -> random.Random:
    """Dedicated stream, so switching the sampler never shifts a minter's pair-structure draws."""
    return random.Random(f"{seed}:payload_wf1_v1")


def sample_payload_bytes(rng: random.Random) -> float:
    """One pair's payload in bytes. Always two draws (tier, then value) so the stream position per pair is fixed."""
    heavy = rng.random() < HEAVY_SHARE
    u = rng.random()
    if heavy:
        return math.exp(math.log(HEAVY_LOW_BYTES) + u * (math.log(HEAVY_HIGH_BYTES) - math.log(HEAVY_LOW_BYTES)))
    # inverse-CDF on u keeps the draw count fixed (random.lognormvariate consumes a variable number)
    z = _norm_ppf(u)
    return MEDIAN_BYTES * math.exp(SIGMA * z)


def _norm_ppf(p: float) -> float:
    from statistics import NormalDist

    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return NormalDist().inv_cdf(p)


def resample_peer_exchange(pairs: Iterable[Sequence[float]], seed: int) -> List[List[float]]:
    """Replace the payload of each ``[i, j, bytes]`` triple, keeping the pair structure and order."""
    rng = payload_rng(seed)
    return [[int(p[0]), int(p[1]), sample_payload_bytes(rng)] for p in pairs]


def payload_sampler_meta() -> Dict[str, float]:
    return {
        "sampler": "wf1_v1",
        "median_bytes": MEDIAN_BYTES,
        "sigma": SIGMA,
        "heavy_share": HEAVY_SHARE,
        "heavy_low_bytes": HEAVY_LOW_BYTES,
        "heavy_high_bytes": HEAVY_HIGH_BYTES,
    }


def require_sampler(name: str) -> str:
    if name not in PAYLOAD_SAMPLERS:
        raise ValueError(f"payload sampler {name!r}; expected one of {PAYLOAD_SAMPLERS}")
    return name
