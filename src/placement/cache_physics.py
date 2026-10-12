"""The physics environment a cache was built under, recorded in its metadata and checked by trainer and server.

Three environment switches change numbers that go straight into cached features, and the cache did not record them:
`HEROSIM_TRANSFER_MODEL` (transmission_hops is 1 under `pipelined`, n_hops under `store_forward`, so every
`node_exchange` second moves 3-4x), `HEROSIM_REPLICA_RELEASE` and `HEROSIM_SCALEOUT` (which replicas exist, which of them
the label's plans may use). A cache built in one shell and trained or served in another is a silent train/serve mismatch;
the v5 parity run caught it only because that script compares node_exchange. Absent key = a cache from before this field.

`inflight_capture` (HEROSIM_INFLIGHT_CAPTURE: legacy | service_end_v1) is the fourth switch: it decides whether a snapshot's backlog carries the
in-flight task's remaining service time, so capture and serving must agree. A cache records the mode its SNAPSHOTS were captured under (read off the
datasets: a service_end_v1 capture writes `current_task_remaining` on every replica spec, a legacy one never does), not the shell that built it.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional

KEYS = ("transfer_model", "replica_release", "scaleout", "inflight_capture", "replica_placement_rule")
# A record from before the field priced in-flight tasks the only way there was (`legacy`) and placed replicas by the first compatible platform.
DEFAULT_WHEN_ABSENT = {"inflight_capture": "legacy", "replica_placement_rule": "first_compatible"}


def current_physics_env(space_config: Optional[Mapping[str, Any]] = None) -> Dict[str, str]:
    from src.placement.network_fabric import transfer_model
    from src.placement.scaleout import scaleout_mode

    release = os.environ.get("HEROSIM_REPLICA_RELEASE", "0")
    if release not in ("0", "1"):
        raise ValueError(f"HEROSIM_REPLICA_RELEASE={release!r}; expected 0 or 1")
    from src.placement.live_audit import inflight_capture_mode
    from src.placement import replica_rule

    # the rule of THIS run: the cell config when one is at hand (checkpoint loading precedes the simulator), else what the simulator was set to
    rule = replica_rule.validate((space_config.get("preinit") or {}).get("replica_placement_rule", replica_rule.FIRST)) \
        if space_config is not None else replica_rule.current_rule()
    return {"transfer_model": transfer_model(), "replica_release": release, "scaleout": scaleout_mode(),
            "inflight_capture": inflight_capture_mode(), "replica_placement_rule": rule}


def require_matching_physics_env(recorded: Optional[Mapping[str, Any]], *, what: str, require: bool = False,
                                 space_config: Optional[Mapping[str, Any]] = None, skip: tuple = ()) -> None:
    """Raise unless `recorded` (a cache's or checkpoint's physics_env) equals this process's. `recorded` None is a
    cache/checkpoint from before the field: tolerated unless `require` (the v5 contract demands it)."""
    now = current_physics_env(space_config)
    if recorded is None:
        if require:
            raise ValueError(f"{what} records no physics_env; rebuild it under the R1 environment")
        return
    bad = {k: (recorded.get(k, DEFAULT_WHEN_ABSENT.get(k)), now[k]) for k in KEYS if k not in skip
           if recorded.get(k, DEFAULT_WHEN_ABSENT.get(k)) != now[k]}
    if bad:
        detail = ", ".join(f"{k}: built/trained under {a!r}, this run has {b!r}" for k, (a, b) in bad.items())
        raise ValueError(f"{what}: physics environment mismatch ({detail}); export the environment it was built under")
