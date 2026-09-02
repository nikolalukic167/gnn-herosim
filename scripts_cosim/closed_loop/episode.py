#!/usr/bin/env python3
"""One live episode, run exactly the way the gates run one.

objective_pivot_v1 Phase 3 (P1 closed-loop). Every arm of the closed loop — sampled or
greedy, GNN or MLP — gets its return from here, and here shells out to
`src/executesimulation.py` rather than driving SimPy in-process. That is deliberate:

  * the episode return has to be the *live metric*, produced by the same serving path
    the live gates measure, or the loop optimises something adjacent to the claim;
  * a fresh process per episode means no SimPy state, no `INFERENCE_FEATURE_LAYOUT`
    residue and no module-level seed survives from one episode into the next, which is
    the failure class that made every pre-2026-08-24 MLP checkpoint an unreproducible
    draw.

The environment block below must stay identical to the one in
`scripts_cosim/closed_loop/run_sampling_probe.py`. That script is the registered
Increment-1 instrument and is deliberately left frozen at the code that produced job
733169's GO verdict, so the two cannot simply share a function — instead
`scripts_cosim/test_closed_loop_episode.py` asserts they still agree, and fails if
either drifts.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

PYTHON = os.environ.get("HEROSIM_PY", "python3")

# Policy registry names (src/placement/simulation.py), not run_simulation.py strategy
# strings. A wrong guess here costs a 5 s startup round-trip per episode.
ARM_POLICY = {"gnn": "gnn", "mlp": "mlp_batch"}


def episode_env(
    *,
    model: Path,
    arm: str,
    temperature: Optional[float],
    seed: Optional[int],
    reservoir_k: int = 0,
    replay_out: Optional[Path] = None,
) -> Dict[str, str]:
    """The exact environment an episode runs under.

    `temperature=None` selects the deterministic argmax policy — the self-critical
    baseline, and the same configuration the live gates serve.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONHASHSEED"] = "0"
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    env["HEROSIM_WARMTH_PHYSICS"] = "node_disk_v2"
    env["GNN_BATCH_SIZE"] = "4"
    env["GNN_BATCH_TIMEOUT"] = "0.002"
    env["GNN_MODEL_PATH"] = str(model)
    # INFERENCE_FEATURE_LAYOUT / NETWORK_GRAPH_CONTRACT / QUEUE_FEATURE_CONTRACT are
    # deliberately NOT exported: load_gnn_model adopts them from the checkpoint's
    # .contract.json and raises on a conflicting export.
    if arm == "mlp":
        env["MLP_MODEL_PATH"] = str(model)
    if temperature is None:
        env["GNN_DECODE_MODE"] = "argmax"
        for key in ("GNN_SAMPLE_TEMPERATURE", "GNN_SAMPLE_SEED",
                    "GNN_SAMPLE_RESERVOIR_K", "HEROSIM_EPISODE_REPLAY_OUT"):
            env.pop(key, None)
    else:
        env["GNN_DECODE_MODE"] = "sample"
        env["GNN_SAMPLE_TEMPERATURE"] = str(temperature)
        env["GNN_SAMPLE_SEED"] = str(seed)
        env["GNN_SAMPLE_RESERVOIR_K"] = str(int(reservoir_k))
        if replay_out is not None:
            env["HEROSIM_EPISODE_REPLAY_OUT"] = str(replay_out)
        else:
            env.pop("HEROSIM_EPISODE_REPLAY_OUT", None)
    return env


@dataclass
class EpisodeResult:
    """What one episode contributes to the loop."""

    cell: str
    arm: str
    temperature: Optional[float]
    seed: Optional[int]
    total_rtt: float
    num_tasks: int
    wall_s: float
    trajectory: Optional[Dict[str, Any]] = None
    replay_path: Optional[Path] = None

    @property
    def reward(self) -> float:
        """Reward is negated RTT: the loop maximises, the simulator reports a cost."""
        return -self.total_rtt


def run_episode(
    *,
    config: Path,
    workload: Path,
    model: Path,
    out_json: Path,
    cell: str,
    arm: str = "gnn",
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    reservoir_k: int = 0,
    replay_out: Optional[Path] = None,
    timeout_s: int = 7200,
) -> EpisodeResult:
    """Run one full live episode and return its result. Raises on any failure."""
    if arm not in ARM_POLICY:
        raise ValueError(f"FAIL LOUD: unknown arm {arm!r}, expected one of {sorted(ARM_POLICY)}")
    if temperature is not None and seed is None:
        raise ValueError("FAIL LOUD: a sampled episode needs a seed, or it is unpairable")
    if temperature is None and (reservoir_k or replay_out):
        raise ValueError(
            "FAIL LOUD: the greedy baseline episode cannot produce a replay reservoir — "
            "it takes no stochastic actions, so there is no log-prob to differentiate."
        )

    env = episode_env(
        model=model, arm=arm, temperature=temperature, seed=seed,
        reservoir_k=reservoir_k, replay_out=replay_out,
    )
    cmd = [
        *PYTHON.split(),
        "src/executesimulation.py",
        "--policy", ARM_POLICY[arm],
        "--config", str(config),
        "--workload", str(workload),
        "--output", str(out_json),
    ]
    if arm == "mlp":
        # mlp_batch reads its checkpoint from --mlp-model; MLP_MODEL_PATH only supplies
        # the default and is recorded in run_provenance. Passing both keeps the
        # provenance block honest about which file actually served.
        cmd += ["--mlp-model", str(model)]
    # --seed is deliberately never passed. It overrides the topology, so a "multi-seed"
    # sweep is silently a multi-topology sweep; episode stochasticity here comes from
    # GNN_SAMPLE_SEED alone, and the cell config is what varies the environment.
    out_json.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=timeout_s
    )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2500:]
        raise RuntimeError(
            f"FAIL LOUD: episode failed (rc={proc.returncode})\ncmd: {' '.join(cmd)}\n{tail}"
        )
    if not out_json.exists():
        raise RuntimeError(f"FAIL LOUD: episode wrote no result at {out_json}")
    # Result JSONs are large; total_rtt/num_tasks sit near the top, but parse the whole
    # document -- a truncated file whose PREFIX parses is a known failure mode here.
    with open(out_json) as f:
        res = json.load(f)
    rtt = res.get("total_rtt")
    n = res.get("num_tasks")
    if rtt is None or not n:
        raise RuntimeError(f"FAIL LOUD: {out_json} has no total_rtt/num_tasks")

    traj = res.get("episode_trajectory")
    if temperature is not None and not traj:
        raise RuntimeError(
            f"FAIL LOUD: {out_json} was run with GNN_DECODE_MODE=sample but carries no "
            "episode_trajectory. The episode took greedy actions and its return would "
            "enter the gradient as if it had explored."
        )
    if replay_out is not None and not replay_out.exists():
        raise RuntimeError(f"FAIL LOUD: no replay reservoir written at {replay_out}")

    return EpisodeResult(
        cell=cell, arm=arm, temperature=temperature, seed=seed,
        total_rtt=float(rtt), num_tasks=int(n), wall_s=round(elapsed, 1),
        trajectory=traj, replay_path=replay_out,
    )


def load_replay(path: Path) -> Tuple[float, int, List[Dict[str, Any]]]:
    """Read a reservoir file back. Returns (temperature, n_batches, batches)."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    batches = payload["batches"]
    n_batches = int(payload["n_batches"])
    if not batches:
        raise RuntimeError(f"FAIL LOUD: {path} holds no replay batches")
    if n_batches < len(batches):
        raise RuntimeError(
            f"FAIL LOUD: {path} claims {n_batches} batches but stored {len(batches)}; "
            "the N/k rescale would understate the gradient."
        )
    return float(payload["temperature"]), n_batches, batches
