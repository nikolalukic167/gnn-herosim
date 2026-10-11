"""plan_rerank_v1 Phase 0: the real single-batch RTT of each candidate plan, for the live oracle arm (HEROSIM_PG_ORACLE=1).

The label is the corpus' (`fidelity_replay.LABEL`): the sum over the batch's tasks of (doneTime - scheduledTime) in a fidelity replay of
the live snapshot with the plan forced. The replays run in a persistent pool of spawned worker processes, so the live SimPy run is never
touched: the snapshot is a read of the live state (`live_audit.capture_batch_snapshot`), the replays are separate simulations.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

CELL_CONFIG_ENV = "HEROSIM_ORACLE_CELL_CONFIG"
SIM_INPUT_ENV = "HEROSIM_ORACLE_SIM_INPUT"
WORKERS_ENV = "HEROSIM_ORACLE_WORKERS"
DIAG_DIR_ENV = "ORACLE_DIAG_DIR"  # not a HEROSIM_ name on purpose (the datalab sbatch unsets those); changes no decision
MIN_PLANS_PER_CHUNK = 4

# Variables of the live run that must not reach a replay: the watchdog's progress file, the event cap, forced placements and the capture path.
_LIVE_ONLY = ("HEROSIM_PROGRESS_FILE", "HEROSIM_PROGRESS_INTERVAL_S", "HEROSIM_MAX_EVENTS", "HEROSIM_FORCED_PLACEMENTS", "LIVE_AUDIT_SNAPSHOT_PATH")


def _memoize_static_inputs() -> None:
    """The replay's cell infrastructure and simulation inputs depend on the cell config alone, not on the batch: build them once per worker.
    `FidelityReplay` only reads them (every run deep-copies the infrastructure), so results are unchanged."""
    import src.executesimulation as ES

    prepare, load = ES.prepare_infrastructure_for_real_simulation, ES.load_simulation_inputs
    prepared: Dict[Any, Any] = {}
    loaded: Dict[str, Any] = {}

    def prepare_memo(space_config, seed=None, sim_input_path=None):
        key = (json.dumps(space_config, sort_keys=True), seed, str(sim_input_path))
        if key not in prepared:
            prepared[key] = prepare(space_config, seed=seed, sim_input_path=sim_input_path)
        return prepared[key]

    def load_memo(sim_input_path):
        key = str(sim_input_path)
        if key not in loaded:
            loaded[key] = load(sim_input_path)
        return loaded[key]

    ES.prepare_infrastructure_for_real_simulation = prepare_memo
    ES.load_simulation_inputs = load_memo


def _init_worker() -> None:
    for name in _LIVE_ONLY:
        os.environ.pop(name, None)
    if os.environ.get("ORACLE_NO_MEMO") != "1":
        _memoize_static_inputs()


def _label_chunk(spec: Mapping[str, Any], types: Sequence[str], plans: Sequence[Sequence[Tuple[int, int]]]) -> Tuple[List[float], float, List[float]]:
    """(labels, replay-setup seconds, seconds per plan) for the plans (task i of the batch, in gid order, on plans[k][i] = (node id, platform id))."""
    from timeit import default_timer

    from src.placement.fidelity_replay import FidelityReplay

    out: List[float] = []
    runs: List[float] = []
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        t0 = default_timer()
        replay = FidelityReplay(spec, types, {})
        setup = default_timer() - t0
        for plan in plans:
            t1 = default_timer()
            result = replay.run({i: (int(p[0]), int(p[1])) for i, p in enumerate(plan)})
            trs = {tr["taskId"]: tr for tr in result["stats"]["taskResults"] if tr.get("taskId", -1) >= 0}
            missing = [b for b in replay.batch_local if b not in trs]
            if missing:
                raise RuntimeError(f"FAIL LOUD: plan oracle: the replay finished without results for batch tasks {missing}")
            out.append(sum(float(trs[b]["doneTime"]) - float(trs[b]["scheduledTime"]) for b in replay.batch_local))
            runs.append(default_timer() - t1)
    return out, setup, runs


class PlanOracle:
    def __init__(self) -> None:
        self.cell_config = os.environ.get(CELL_CONFIG_ENV, "").strip()
        self.sim_input = os.environ.get(SIM_INPUT_ENV, "").strip()
        if not (self.cell_config and self.sim_input):
            raise RuntimeError(f"FAIL LOUD: the plan oracle needs {CELL_CONFIG_ENV} and {SIM_INPUT_ENV}")
        from src.placement import snapshot_fidelity

        if not snapshot_fidelity.enabled():
            raise RuntimeError(f"FAIL LOUD: the plan oracle needs {snapshot_fidelity.ENV}=1")
        self.workers = int(os.environ.get(WORKERS_ENV, "8"))
        if self.workers < 1:
            raise ValueError(f"{WORKERS_ENV} must be >= 1")
        self._pool: ProcessPoolExecutor | None = None

    def _executor(self) -> ProcessPoolExecutor:
        if self._pool is None:
            import multiprocessing as mp

            self._pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=mp.get_context("spawn"), initializer=_init_worker)
        return self._pool

    def spec(self, snapshot: Mapping[str, Any]) -> Dict[str, Any]:
        from src.placement.fidelity_replay import build_spec

        return build_spec(snapshot, Path(self.cell_config), Path(self.sim_input))

    def labels(self, snapshot: Mapping[str, Any], types: Sequence[str], plans: Sequence[Sequence[Tuple[int, int]]]) -> List[float]:
        """Labels of `plans`, aligned with the snapshot's batch tasks in gid order; `types[i]` is task i's type name."""
        if not plans:
            return []
        from timeit import default_timer

        t0 = default_timer()
        spec = self.spec(snapshot)
        chunks = max(1, min(self.workers, -(-len(plans) // MIN_PLANS_PER_CHUNK)))
        parts = [list(range(c, len(plans), chunks)) for c in range(chunks)]
        pool = self._executor()
        futures = [pool.submit(_label_chunk, spec, list(types), [plans[k] for k in part]) for part in parts]
        t1 = default_timer()
        out: List[float] = [0.0] * len(plans)
        setups: List[float] = []
        runs: List[float] = []
        for part, fut in zip(parts, futures):
            labels, setup, per_plan = fut.result()
            setups.append(setup)
            runs.extend(per_plan)
            for k, label in zip(part, labels):
                out[k] = label
        self.last = {"chunks": chunks, "t_spec": t1 - t0, "t_wait": default_timer() - t1, "setup_mean": sum(setups) / len(setups),
                     "setup_max": max(setups), "run_mean": sum(runs) / len(runs), "run_max": max(runs)}
        return out

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None


def diag_write(rec: Dict[str, Any]) -> None:
    """One JSON line per oracle batch to <ORACLE_DIAG_DIR>/oracle_diag_<pid>.jsonl, when the variable is set."""
    root = os.environ.get(DIAG_DIR_ENV, "").strip()
    if not root:
        return
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, f"oracle_diag_{os.getpid()}.jsonl"), "a") as fh:
        fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
