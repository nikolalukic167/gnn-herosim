"""workload_fix_v1 W2 — per-pair peer-exchange payload sampler.

Opt-in. The legacy payload (``200 MB * 10**U(-1, 1)``, 20 MB-2 GB) stays inline in the minters; this module is
used only when a minter is asked for ``wf1_v1``. Units are bytes with 1 MB = 1e6 bytes, the unit the legacy
``X_SCALE_BYTES = 200e6`` already uses.

Distribution (docs/lineages/workload_fix_v1.md, W2): log-normal with median 4 MB and sigma 1.2 in natural-log
space, plus a heavy tier — 10 % of pairs drawn log-uniform on 50-500 MB. The tier is assumed and swept; the body
approximates Eismann et al. (IEEE Software 2021).
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import random
from typing import Dict, Iterable, List, Optional, Sequence

PAYLOAD_SAMPLERS = ("legacy", "wf1_v1", "azblob")

MB = 1e6
MEDIAN_BYTES = 4 * MB
SIGMA = 1.2
HEAVY_SHARE = 0.10
HEAVY_LOW_BYTES = 50 * MB
HEAVY_HIGH_BYTES = 500 * MB


def payload_rng(seed: int) -> random.Random:
    """Dedicated stream, so switching the sampler never shifts a minter's pair-structure draws."""
    return random.Random(f"{seed}:payload_wf1_v1")


def sample_payload_bytes(rng: random.Random, heavy_share: float = HEAVY_SHARE, scale: float = 1.0) -> float:
    """One pair's payload in bytes. Always two draws (tier, then value) so the stream position per pair is fixed.

    ``heavy_share`` is the probability of the heavy tier (default the registered 0.10; 0 gives a formerly heavy pair the body value of
    the same u). ``scale`` multiplies the drawn bytes (default 1.0, which leaves them bit-identical; 0 gives 0-byte payloads)."""
    heavy = rng.random() < heavy_share
    u = rng.random()
    if heavy:
        v = math.exp(math.log(HEAVY_LOW_BYTES) + u * (math.log(HEAVY_HIGH_BYTES) - math.log(HEAVY_LOW_BYTES)))
    else:
        # inverse-CDF on u keeps the draw count fixed (random.lognormvariate consumes a variable number)
        z = _norm_ppf(u)
        v = MEDIAN_BYTES * math.exp(SIGMA * z)
    return v if scale == 1.0 else v * scale


def load_azblob_table(path: str) -> Dict[str, object]:
    """The empirical per-access blob-size quantiles (Azure Functions Blob Dataset 2020, ``BlobBytes``), derived once by
    scripts_cosim/azblob_bytes_quantiles.py: {"quantiles": [q_0 .. q_N] at probabilities i / N, "source_sha256": ..., ...}."""
    with open(path, "rb") as fh:
        raw = fh.read()
    table = json.loads(raw)
    q = table["quantiles"]
    if len(q) < 3 or any(b < a for a, b in zip(q, q[1:])) or q[0] < 0:
        raise ValueError(f"azblob table {path}: quantiles must be non-decreasing and >= 0")
    table["table_sha256"] = hashlib.sha256(raw).hexdigest()
    return table


def sample_azblob_bytes(rng: random.Random, table: Dict[str, object], scale: float = 1.0) -> float:
    """One pair's payload drawn from the empirical BlobBytes distribution. Same stream rule as the wf1_v1 sampler: two draws per pair
    (the first, the tier draw, is consumed and unused), then the value by inverse-CDF with linear interpolation between quantiles."""
    rng.random()
    u = rng.random()
    q = table["quantiles"]
    pos = u * (len(q) - 1)
    i = min(int(pos), len(q) - 2)
    v = q[i] + (pos - i) * (q[i + 1] - q[i])
    return v if scale == 1.0 else v * scale


def _norm_ppf(p: float) -> float:
    from statistics import NormalDist

    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return NormalDist().inv_cdf(p)


def resample_peer_exchange(pairs: Iterable[Sequence[float]], seed: int, sampler: str = "wf1_v1", heavy_share: float = HEAVY_SHARE,
                           scale: float = 1.0, azblob_table: Optional[Dict[str, object]] = None) -> List[List[float]]:
    """Replace the payload of each ``[i, j, bytes]`` triple, keeping the pair structure and order. Defaults give today's wf1_v1 bytes."""
    require_sampler(sampler)
    rng = payload_rng(seed)
    if sampler == "azblob":
        if azblob_table is None:
            raise ValueError("payload sampler azblob needs its quantile table")
        return [[int(p[0]), int(p[1]), sample_azblob_bytes(rng, azblob_table, scale)] for p in pairs]
    if sampler != "wf1_v1":
        raise ValueError(f"resample_peer_exchange: sampler {sampler!r} is not a resampling sampler")
    return [[int(p[0]), int(p[1]), sample_payload_bytes(rng, heavy_share, scale)] for p in pairs]


def payload_sampler_meta(sampler: str = "wf1_v1", heavy_share: float = HEAVY_SHARE, scale: float = 1.0,
                         azblob_table: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """The workload's recorded sampler meta. With the defaults this is exactly the dict of the original wf1_v1 sampler (the workload JSON bytes depend on it);
    a non-default ``scale`` / ``heavy_share`` and the azblob sampler add their own keys."""
    if sampler == "azblob":
        if azblob_table is None:
            raise ValueError("payload sampler azblob needs its quantile table")
        return {"sampler": "azblob", "scale": scale, "table_sha256": azblob_table["table_sha256"], "source_sha256": azblob_table.get("source_sha256"),
                "note": "per-access blob size as a proxy, not function-to-function exchange"}
    meta: Dict[str, object] = {
        "sampler": "wf1_v1",
        "median_bytes": MEDIAN_BYTES,
        "sigma": SIGMA,
        "heavy_share": heavy_share,
        "heavy_low_bytes": HEAVY_LOW_BYTES,
        "heavy_high_bytes": HEAVY_HIGH_BYTES,
    }
    if scale != 1.0:
        meta["scale"] = scale
    return meta


def require_sampler(name: str) -> str:
    if name not in PAYLOAD_SAMPLERS:
        raise ValueError(f"payload sampler {name!r}; expected one of {PAYLOAD_SAMPLERS}")
    return name
