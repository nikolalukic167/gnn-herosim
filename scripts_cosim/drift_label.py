"""The one home for the `drainable_objective_v1` shaped label.

    L_V(plan) = rtt(plan) + V * sum_p (lambda_p / 2) * [(B_p + A_p(plan))^2 - B_p^2]

where, for platform p in a co-sim dataset's captured state:

  B_p        backlog already on p, in SECONDS (queued count x per-item drain)
  A_p(plan)  the work this plan adds to p, in the same seconds
  lambda_p   the arrival rate p will face, tasks/s

Why this shape. A `Platform` serves one task at a time (infrastructure.py:1240-1251), so
work put on p lengthens its busy period and every task that arrives during the extension
waits behind it. For a fluid arrival rate the extra waiting inflicted on FUTURE arrivals
is the integral of the added backlog over the time it persists, i.e.
(lambda/2)[(B+A)^2 - B^2]. The one-step sweep charges a batch its own queueing and
nothing beyond it; the reactive shortest-queue rule avoids the term without computing it.

What this is NOT: a rollout. Nothing here runs the simulator forward, which is the
mechanism that killed the h = 10 s horizon-return label (`objective_pivot_v1` Phase 2 --
"the decision plus everything the simulator did afterwards", lessons.md L32). It is a
deterministic function of (captured state, plan). The rank-stability control that stop
bequeathed is still owed and is bar A2-rank, over V rather than over a horizon.

Conceded in the registration: every term here is a function of per-platform counts and
their squares, i.e. registered count competitor v2 (peer_affinity_v1 A6, R^2 1.0000), so
a pointwise scorer handed those columns expresses it exactly. This is a LABEL lever, not
a GNN lever.

The per-item drain `c(task_type, platform_type)`:

  * DEFAULT (the corpus's own clock) -- `execution + comm`, exactly what
    `Platform.seed_virtual_warmup` (infrastructure.py:692-757) charges a seeded backlog.
  * MEASURED (the live clock) -- a table emitted by the A1 read from live captures, where
    a queued task also pays its peer transfers and source->platform latency. Under
    HEROSIM_PEER_EXCHANGE=1 the default understates a deep queue's drain ~100x
    (live_snapshot_seed.py:195-200), which is the defect this lineage's V = 0 arm tests.

Run the tests:

    PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH=. pipenv run python3 \
        -m pytest tests/test_drift_label.py -q
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

Plan = Dict[int, Tuple[int, int]]

# The environment variable the simulator and the cache builder both read for a measured
# drain table. One spelling, so a corpus and its label cannot disagree about the clock.
DRAIN_TABLE_ENV = "HEROSIM_BACKLOG_DRAIN_TABLE"

# Storage constants, copied from live_snapshot_seed._approx_comm so this module is
# importable without the simulator package (the cache builder and the read tools both
# import it in environments where src/ is on the path, but the sbatch read jobs run it
# standalone). A drift between the two would silently reprice every backlog, so the test
# suite asserts they agree against the real helper.
_STORAGE_THROUGHPUT = 100.0 * 1024.0 * 1024.0
_STORAGE_LATENCY = 0.001


def approx_comm(task_type: Mapping[str, object]) -> float:
    """Storage I/O seconds for one task of this type, the `seed_virtual_warmup` model."""
    state_size_map = task_type.get("stateSize", {})
    if not isinstance(state_size_map, dict) or not state_size_map:
        return 0.0
    app_state = next(iter(state_size_map.values()))
    if not isinstance(app_state, dict):
        return 0.0
    input_size = float(app_state.get("input", 0) or 0)
    output_size = float(app_state.get("output", 0) or 0)
    return (input_size / _STORAGE_THROUGHPUT + _STORAGE_LATENCY) + (
        output_size / _STORAGE_THROUGHPUT + _STORAGE_LATENCY
    )


def drain_table_key(task_type_name: str, platform_type: str) -> str:
    return f"{task_type_name}|{platform_type}"


def load_drain_table(path: Optional[Path] = None) -> Optional[Dict[str, float]]:
    """A measured per-item drain table, or None when the default clock is in force.

    `path` wins; otherwise HEROSIM_BACKLOG_DRAIN_TABLE. A path that is set but unreadable
    is fatal -- silently falling back to the default clock would relabel a whole corpus
    without saying so, which is the exact failure mode this lineage exists to fix.
    """
    raw = str(path) if path is not None else os.environ.get(DRAIN_TABLE_ENV, "")
    if not raw:
        return None
    p = Path(raw)
    if not p.exists():
        raise DriftLabelError(
            f"{DRAIN_TABLE_ENV}={raw} but that file does not exist -- refusing to fall "
            "back to the exec+comm clock silently"
        )
    payload = json.loads(p.read_text())
    table = payload.get("drain_seconds_per_item", payload)
    if not isinstance(table, dict) or not table:
        raise DriftLabelError(f"{p}: no drain_seconds_per_item map")
    out: Dict[str, float] = {}
    for key, value in table.items():
        seconds = float(value)
        if seconds <= 0.0:
            raise DriftLabelError(f"{p}: non-positive drain {seconds} for {key!r}")
        out[str(key)] = seconds
    return out


class DriftLabelError(RuntimeError):
    """Fail loud. Never substitute a default for a quantity that was asked for."""


@dataclass(frozen=True)
class StateContext:
    """Everything the shaped term needs about one dataset's captured state.

    platform_type_of   platform_id -> platform type shortName
    node_of            platform_id -> node name
    queue_key_of       platform_id -> "<node>:<platform_id>", the queue_distributions key
    backlog_counts     (task_type, platform_id) -> queued tasks of that type on p
    task_type_names    task id -> task type name, in task-id order
    drain              (task_type, platform_type) -> seconds per queued item
    lam                platform_id -> arrivals/s this platform will face
    """

    platform_type_of: Dict[int, str]
    node_of: Dict[int, str]
    queue_key_of: Dict[int, str]
    backlog_counts: Dict[Tuple[str, int], int]
    task_type_names: List[str]
    drain: Dict[Tuple[str, str], float]
    lam: Dict[int, float]
    seeded_backlog: Optional[Dict[int, float]] = None

    def drain_of(self, task_type: str, platform_id: int) -> float:
        ptype = self.platform_type_of.get(platform_id)
        if ptype is None:
            raise DriftLabelError(
                f"platform_id {platform_id} is not in this dataset's replica_placements"
            )
        value = self.drain.get((task_type, ptype))
        if value is None:
            raise DriftLabelError(
                f"no per-item drain for ({task_type}, {ptype}) -- a measured table that "
                "does not cover the corpus is a relabelling bug, not a missing default"
            )
        return value

    def backlog_seconds(self) -> Dict[int, float]:
        """B_p for every platform that carries any backlog."""
        if self.seeded_backlog is not None:
            return dict(self.seeded_backlog)
        out: Dict[int, float] = {}
        for (task_type, pid), count in self.backlog_counts.items():
            if count <= 0:
                continue
            out[pid] = out.get(pid, 0.0) + count * self.drain_of(task_type, pid)
        return out


def added_seconds(plan: Plan, ctx: StateContext) -> Dict[int, float]:
    """A_p(plan): the work this plan puts on each platform, in seconds."""
    out: Dict[int, float] = {}
    for task_id, placement in plan.items():
        pid = int(placement[1])
        if task_id >= len(ctx.task_type_names):
            raise DriftLabelError(
                f"plan references task {task_id} but the dataset declares "
                f"{len(ctx.task_type_names)} tasks"
            )
        ttype = ctx.task_type_names[task_id]
        out[pid] = out.get(pid, 0.0) + ctx.drain_of(ttype, pid)
    return out


def externality_seconds(plan: Plan, ctx: StateContext) -> float:
    """sum_p (lambda_p / 2) * [(B_p + A_p)^2 - B_p^2], in seconds of inflicted wait."""
    backlog = ctx.backlog_seconds()
    added = added_seconds(plan, ctx)
    total = 0.0
    for pid, a in added.items():
        if a <= 0.0:
            continue
        b = backlog.get(pid, 0.0)
        lam = ctx.lam.get(pid, 0.0)
        if lam <= 0.0:
            continue
        total += 0.5 * lam * ((b + a) ** 2 - b**2)
    return total


def shaped_value(rtt: float, plan: Plan, ctx: StateContext, v: float) -> float:
    """L_V for one sweep row. V = 0 returns `rtt` unchanged, bit for bit."""
    if v == 0.0:
        return float(rtt)
    return float(rtt) + v * externality_seconds(plan, ctx)


def shaped_rows(
    rows: Sequence[Tuple[Plan, float]], ctx: StateContext, v: float
) -> List[Tuple[Plan, float]]:
    """Relabel a whole enumerated sweep. Order is preserved."""
    if v == 0.0:
        return [(plan, float(value)) for plan, value in rows]
    return [(plan, shaped_value(value, plan, ctx, v)) for plan, value in rows]


# ---------------------------------------------------------------------------
# Building a StateContext from a co-sim dataset on disk
# ---------------------------------------------------------------------------


def _arrival_rate_from_workload(workload: Mapping[str, object]) -> float:
    """Tasks/s the trace delivers, from its own timestamps.

    A co-sim dataset's workload is one batch at t = 0, which carries no rate at all --
    the rate a state will face comes from the GATE trace, not from the batch. So a
    caller must pass `arrival_rate` explicitly for those; this helper exists for traces
    that do span time and it refuses to invent a rate from a zero span.
    """
    events = workload.get("events") or []
    if len(events) < 2:
        raise DriftLabelError(
            "workload has fewer than 2 events -- pass arrival_rate explicitly"
        )
    stamps = [float(e.get("timestamp", 0.0) or 0.0) for e in events]  # type: ignore[union-attr]
    span = max(stamps) - min(stamps)
    if span <= 0.0:
        raise DriftLabelError(
            "every event in this workload arrives at the same instant, so it carries no "
            "arrival rate -- pass arrival_rate explicitly (for a co-sim batch this is "
            "the GATE trace's rate, never the batch's)"
        )
    return len(events) / span


def build_state_context(
    ds_dir: Path,
    task_types_db: Mapping[str, Mapping[str, object]],
    *,
    arrival_rate: float,
    drain_table: Optional[Mapping[str, float]] = None,
    task_type_names: Optional[Sequence[str]] = None,
    backlog_clock: str = "counts",
) -> StateContext:
    """Read one co-sim dataset's captured state into a StateContext.

    `arrival_rate` is the tasks/s the SERVED cluster faces (the gate trace's rate), split
    across task types by the batch's own type mix and then across each type's replicas.
    It is a property of the deployment the label is meant for, not of the 10-task batch,
    so it is always passed in and never guessed.
    """
    infra_path = ds_dir / "infrastructure.json"
    if not infra_path.exists():
        raise DriftLabelError(f"{ds_dir}: infrastructure.json missing")
    infra = json.loads(infra_path.read_text())

    platform_type_of: Dict[int, str] = {}
    node_of: Dict[int, str] = {}
    queue_key_of: Dict[int, str] = {}
    replicas_of_type: Dict[str, set] = {}
    for ttype, replicas in (infra.get("replica_placements") or {}).items():
        for rep in replicas:
            pid = int(rep["platform_id"])
            ptype = str(rep["platform_type"])
            node = str(rep["node_name"])
            if pid in platform_type_of and platform_type_of[pid] != ptype:
                raise DriftLabelError(
                    f"{ds_dir}: platform_id {pid} is both {platform_type_of[pid]} and "
                    f"{ptype}"
                )
            platform_type_of[pid] = ptype
            node_of[pid] = node
            queue_key_of[pid] = f"{node}:{pid}"
            replicas_of_type.setdefault(ttype, set()).add(pid)
    if not platform_type_of:
        raise DriftLabelError(f"{ds_dir}: empty replica_placements")

    if task_type_names is None:
        from scripts_cosim.score_route_b_contention import load_task_type_names

        names = list(load_task_type_names(ds_dir))
    else:
        names = list(task_type_names)

    # Per-item drain, on whichever clock is in force.
    drain: Dict[Tuple[str, str], float] = {}
    for ttype in set(names) | set(replicas_of_type):
        type_row = task_types_db.get(ttype)
        if type_row is None:
            raise DriftLabelError(f"{ds_dir}: no task type {ttype!r} in the type table")
        comm = approx_comm(type_row)
        exec_by_ptype = type_row.get("executionTime") or {}
        for ptype in set(platform_type_of.values()):
            if drain_table is not None:
                measured = drain_table.get(drain_table_key(ttype, ptype))
                if measured is not None:
                    drain[(ttype, ptype)] = float(measured)
                    continue
                raise DriftLabelError(
                    f"measured drain table has no entry for ({ttype}, {ptype}) -- "
                    "refusing to mix a measured clock with the default one inside a "
                    "single dataset"
                )
            execution = float(exec_by_ptype.get(ptype, 0.0) or 0.0)  # type: ignore[union-attr]
            drain[(ttype, ptype)] = execution + comm

    backlog_counts: Dict[Tuple[str, int], int] = {}
    key_to_pid = {key: pid for pid, key in queue_key_of.items()}
    for ttype, per_key in (infra.get("queue_distributions") or {}).items():
        for key, count in (per_key or {}).items():
            pid = key_to_pid.get(str(key))
            if pid is None:
                # A queue key with no replica row is a state the sweep cannot place on;
                # it still loads the platform, but no plan can add to it, so it can only
                # contribute to B_p of a platform nothing is placed on -- which the
                # externality skips anyway. Recorded as a hard error instead of a silent
                # drop, because it would mean infrastructure.json is internally
                # inconsistent.
                raise DriftLabelError(
                    f"{ds_dir}: queue_distributions key {key!r} has no replica_placements "
                    "row"
                )
            if int(count) > 0:
                backlog_counts[(ttype, pid)] = backlog_counts.get((ttype, pid), 0) + int(count)

    # lambda_p: the trace's rate, split by the batch's type mix, then across each type's
    # replicas, then summed over the types a platform serves.
    if arrival_rate <= 0.0:
        raise DriftLabelError(f"arrival_rate must be positive, got {arrival_rate}")
    type_share: Dict[str, float] = {}
    for name in names:
        type_share[name] = type_share.get(name, 0.0) + 1.0 / len(names)
    lam: Dict[int, float] = {}
    for ttype, share in type_share.items():
        pids = replicas_of_type.get(ttype) or set()
        if not pids:
            continue
        per_replica = arrival_rate * share / len(pids)
        for pid in pids:
            lam[pid] = lam.get(pid, 0.0) + per_replica

    seeded = _seeded_backlog_by_pid(ds_dir, infra, platform_type_of, task_types_db, backlog_clock)

    return StateContext(
        platform_type_of=platform_type_of,
        node_of=node_of,
        queue_key_of=queue_key_of,
        backlog_counts=backlog_counts,
        task_type_names=names,
        drain=drain,
        lam=lam,
        seeded_backlog=seeded,
    )


def _seeded_backlog_by_pid(
    ds_dir: Path,
    infra: Mapping[str, object],
    platform_type_of: Mapping[int, str],
    task_types_db: Mapping[str, Mapping[str, object]],
    backlog_clock: str,
) -> Optional[Dict[int, float]]:
    """B_p on the clock the co-sim replays (live_snapshot_seed.seeded_backlog_seconds).

    backlog_corpus_v1: a synthetic backlog lives only in that clock (fake work in seconds);
    the count x drain-table clock cannot see its seconds, so a synthetic corpus labelled on
    `counts` would price a queue the simulator never ran. That combination is refused.
    """
    if backlog_clock not in BACKLOG_CLOCKS:
        raise DriftLabelError(f"unknown backlog clock {backlog_clock!r}; one of {BACKLOG_CLOCKS}")
    seed = infra.get("live_snapshot_seed") or {}
    platforms = seed.get("platforms") or []  # type: ignore[union-attr]
    synthetic = any(float(p.get("synthetic_backlog_seconds", 0) or 0) > 0 for p in platforms)
    if backlog_clock == "counts":
        if synthetic:
            raise DriftLabelError(
                f"{ds_dir}: carries a synthetic backlog but the label's backlog clock is "
                f"'counts' -- set {BACKLOG_CLOCK_ENV}=seeded"
            )
        return None
    if not platforms:
        raise DriftLabelError(f"{ds_dir}: backlog clock 'seeded' needs live_snapshot_seed.platforms")
    from src.placement.live_snapshot_seed import seeded_backlog_seconds

    spec_by_pid = {int(p["platform_id"]): p for p in platforms}  # type: ignore[index]
    out: Dict[int, float] = {}
    for pid, ptype in platform_type_of.items():
        spec = spec_by_pid.get(pid)
        if spec is None:
            raise DriftLabelError(f"{ds_dir}: replica platform {pid} is not in the snapshot seed")
        hint = task_types_db.get(str(spec.get("task_type_hint", "dnn1")))
        seconds = seeded_backlog_seconds(spec, hint, ptype)
        if seconds:
            out[pid] = float(seconds)
    return out


def parse_label_objective(spec: str) -> Tuple[str, float]:
    """'rtt' -> ('rtt', 0.0); 'rtt_drift:1.5' -> ('rtt_drift', 1.5).

    'rtt_drift' with no coefficient is rejected: an unstated V is the difference between
    a control arm and a treatment arm.
    """
    spec = (spec or "rtt").strip()
    if spec == "rtt":
        return "rtt", 0.0
    if spec.startswith("rtt_drift"):
        _, _, tail = spec.partition(":")
        if not tail:
            raise DriftLabelError(
                "rtt_drift needs an explicit coefficient, e.g. rtt_drift:1.0"
            )
        v = float(tail)
        if v < 0.0:
            raise DriftLabelError(f"V must be >= 0, got {v}")
        return "rtt_drift", v
    raise DriftLabelError(f"unknown label objective {spec!r}")


# ---------------------------------------------------------------------------
# The label configuration, carried in the environment
#
# The cache build fans out over a ProcessPoolExecutor, and a `run_experiment.py` config
# can only set env. Carrying the label config in the environment therefore makes it
# survive both, and makes a checkpoint's provenance able to state which label it was
# fitted to without any new plumbing.
# ---------------------------------------------------------------------------

LABEL_OBJECTIVE_ENV = "NEAR_RTT_LABEL_OBJECTIVE"
LABEL_ARRIVAL_RATE_ENV = "NEAR_RTT_LABEL_ARRIVAL_RATE"
BACKLOG_CLOCK_ENV = "NEAR_RTT_LABEL_BACKLOG_CLOCK"
BACKLOG_CLOCKS = ("counts", "seeded")


@dataclass(frozen=True)
class LabelConfig:
    objective: str
    v: float
    arrival_rate: float
    drain_table: Optional[Dict[str, float]]
    backlog_clock: str = "counts"

    @property
    def is_identity(self) -> bool:
        return self.objective == "rtt" or self.v == 0.0

    def describe(self) -> str:
        """The one string that goes into metadata.json and the checkpoint sidecar."""
        if self.is_identity:
            return "rtt"
        clock = "measured" if self.drain_table else "exec+comm"
        tail = ",backlog=seeded" if self.backlog_clock == "seeded" else ""
        return f"rtt_drift:{self.v:g}@lambda={self.arrival_rate:g},clock={clock}{tail}"


def label_config_from_env() -> LabelConfig:
    objective, v = parse_label_objective(os.environ.get(LABEL_OBJECTIVE_ENV, "rtt"))
    raw_rate = os.environ.get(LABEL_ARRIVAL_RATE_ENV, "")
    if objective == "rtt_drift" and not raw_rate:
        raise DriftLabelError(
            f"{LABEL_OBJECTIVE_ENV}={objective} needs {LABEL_ARRIVAL_RATE_ENV} -- the "
            "arrival rate the label assumes is a property of the deployment, never a "
            "default"
        )
    rate = float(raw_rate) if raw_rate else 0.0
    backlog_clock = os.environ.get(BACKLOG_CLOCK_ENV, "counts") or "counts"
    if backlog_clock not in BACKLOG_CLOCKS:
        raise DriftLabelError(f"{BACKLOG_CLOCK_ENV}={backlog_clock!r}; one of {BACKLOG_CLOCKS}")
    return LabelConfig(
        objective=objective, v=v, arrival_rate=rate, drain_table=load_drain_table(),
        backlog_clock=backlog_clock,
    )


class DatasetLabeler:
    """Applies a LabelConfig to sweep rows, caching one StateContext per dataset dir."""

    def __init__(self, config: LabelConfig, task_types_db: Optional[Mapping] = None):
        self.config = config
        self._task_types_db = task_types_db
        self._ctx: Dict[Path, StateContext] = {}

    def _db(self) -> Mapping:
        if self._task_types_db is None:
            path = Path(os.environ.get("HEROSIM_TASK_TYPES", "data/nofs-ids/task-types.json"))
            if not path.exists():
                raise DriftLabelError(
                    f"task type table not found at {path}; set HEROSIM_TASK_TYPES"
                )
            self._task_types_db = json.loads(path.read_text())
        return self._task_types_db

    def context(self, ds_dir: Path) -> StateContext:
        ds_dir = Path(ds_dir)
        if ds_dir not in self._ctx:
            self._ctx[ds_dir] = build_state_context(
                ds_dir,
                self._db(),
                arrival_rate=self.config.arrival_rate,
                drain_table=self.config.drain_table,
                backlog_clock=self.config.backlog_clock,
            )
        return self._ctx[ds_dir]

    def value(self, ds_dir: Path, plan: Plan, rtt: float) -> float:
        if self.config.is_identity:
            return float(rtt)
        return shaped_value(rtt, plan, self.context(ds_dir), self.config.v)

    def rows(self, ds_dir: Path, rows: Sequence[Tuple[Plan, float]]) -> List[Tuple[Plan, float]]:
        if self.config.is_identity:
            return [(p, float(v)) for p, v in rows]
        return shaped_rows(rows, self.context(ds_dir), self.config.v)


def plan_from_placement_plan(placement_plan: Mapping) -> Plan:
    """{'0': [node, platform], ...} (the placements.jsonl spelling) -> Plan."""
    return {
        int(k): (int(v[0]), int(v[1]))
        for k, v in placement_plan.items()
        if isinstance(v, (list, tuple)) and len(v) >= 2
    }
