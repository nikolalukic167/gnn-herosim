"""The physics environment a cache was built under, recorded in its metadata and checked by trainer and server.

Three environment switches change numbers that go straight into cached features, and the cache did not record them:
`HEROSIM_TRANSFER_MODEL` (transmission_hops is 1 under `pipelined`, n_hops under `store_forward`, so every
`node_exchange` second moves 3-4x), `HEROSIM_REPLICA_RELEASE` and `HEROSIM_SCALEOUT` (which replicas exist, which of them
the label's plans may use). A cache built in one shell and trained or served in another is a silent train/serve mismatch;
the v5 parity run caught it only because that script compares node_exchange. Absent key = a cache from before this field.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional

KEYS = ("transfer_model", "replica_release", "scaleout")


def current_physics_env() -> Dict[str, str]:
    from src.placement.network_fabric import transfer_model
    from src.placement.scaleout import scaleout_mode

    release = os.environ.get("HEROSIM_REPLICA_RELEASE", "0")
    if release not in ("0", "1"):
        raise ValueError(f"HEROSIM_REPLICA_RELEASE={release!r}; expected 0 or 1")
    return {"transfer_model": transfer_model(), "replica_release": release, "scaleout": scaleout_mode()}


def require_matching_physics_env(recorded: Optional[Mapping[str, Any]], *, what: str, require: bool = False) -> None:
    """Raise unless `recorded` (a cache's or checkpoint's physics_env) equals this process's. `recorded` None is a
    cache/checkpoint from before the field: tolerated unless `require` (the v5 contract demands it)."""
    now = current_physics_env()
    if recorded is None:
        if require:
            raise ValueError(f"{what} records no physics_env; rebuild it under the R1 environment")
        return
    bad = {k: (recorded.get(k), now[k]) for k in KEYS if recorded.get(k) != now[k]}
    if bad:
        detail = ", ".join(f"{k}: built/trained under {a!r}, this run has {b!r}" for k, (a, b) in bad.items())
        raise ValueError(f"{what}: physics environment mismatch ({detail}); export the environment it was built under")
