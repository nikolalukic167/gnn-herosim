#!/usr/bin/env python3
"""objective_pivot_v1 Phase 3 — closed-loop policy-gradient training on the live metric.

Registered in `docs/lineages/objective_pivot_v1.md` (Phase 3 registration, 2026-09-01,
with Amendments A-C). This is the trainer for the CL-GNN and CL-MLP arms; Frozen-GNN,
Frozen-MLP and Knative are served, not trained, and are produced by the existing gates.

WHAT MAKES THIS DIFFERENT FROM EVERY OTHER TRAINER IN THE TREE
--------------------------------------------------------------
The supervised trainers fit a co-sim label. `program_verdict_v1` closed that path: the
co-sim target is pointwise-separable, so the MLP is the correctly specified model class
and no amount of data changes it. This trainer has no labels. Its objective is the live
episode's own total RTT, which is the metric the claim is about.

THE ESTIMATOR
-------------
REINFORCE with a self-critical baseline:

    grad = E[ A * grad log pi(actions | states) ],   A = (RTT_greedy - RTT_sampled) / RTT_greedy

`RTT_greedy` is the *current* policy's own argmax episode on the same cell and trace, so
A > 0 means "sampling beat my own greedy policy here, do more of that". The baseline is
recomputed once per cell per step because argmax on a fixed cell and trace is
deterministic — it does not depend on the sampling seed, so S sampled episodes share one
baseline episode rather than needing S of them.

Two passes, because an episode is ~30k decisions across ~7.5k forward passes and the
autograd graph for all of them does not fit:

  pass 1  the episode runs as a subprocess under `no_grad`, sampling actions and
          reservoir-sampling k of its decode batches (inputs + chosen indices) to disk;
  pass 2  this process replays those k batches WITH grad and forms
          (N/k) * sum_reservoir sum_t log pi(a_t | s_t).

Uniform subsampling of a sum, rescaled by N/k, is unbiased: E[(N/k) * sum over the
reservoir] = sum over the episode, because Algorithm R gives every batch inclusion
probability exactly k/N. The variance this adds is real and is why `--reservoir-k` is a
tuning knob rather than a constant.

WHAT THIS TRAINER DOES NOT DO
-----------------------------
It does not decide the pilot's n. The registered sizing rule
`n >= (2*sd*2.8/MDE)^2` returned n >= 1 on the Increment-1 measurement, which is
arithmetically correct and operationally useless; that is a registration defect named in
the node and it needs a signed amendment, not a number chosen here after seeing data.
`--steps` and `--episodes-per-cell` are therefore explicit arguments with no defaults
that could be mistaken for a registered budget.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from scripts_cosim.closed_loop.adapters import (
    batch_logprob,
    copy_checkpoint,
    load_policy,
    require_contract,
    save_policy,
)
from scripts_cosim.closed_loop.episode import load_replay, run_episode


def seed_everything(seed: int) -> None:
    """Seed every source of draw, and pin the nondeterministic autograd kernels.

    The MLP trainer seeded its split and batch order but never `torch.manual_seed`, so
    every MLP checkpoint before 2026-08-24 was an unreproducible draw and three
    published-track claims retired with them. Seeding alone is also not enough: at a
    fixed seed the GIN autograd path was measured diverging run to run even on CPU, which
    `torch.use_deterministic_algorithms` is what actually pins.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)


def _run_all(jobs: List[Dict[str, Any]], workers: int):
    """Run episodes, optionally concurrently, and return results IN JOB ORDER.

    Order matters even though the results are keyed by cell and seed: a step whose
    episode list depends on which subprocess finished first would give a different
    gradient on a differently loaded machine, at the same seeds. The concurrency is
    wall-clock only.
    """
    from concurrent.futures import ThreadPoolExecutor

    if workers <= 1:
        return [run_episode(**job) for job in jobs]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda job: run_episode(**job), jobs))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["gnn", "mlp"], required=True,
                    help="CL-GNN or CL-MLP. Both run the identical loop, by registration.")
    ap.add_argument("--init", type=Path, required=True,
                    help="Warm start. Amendment A fixes this to models/gnn-linkmp-lgon-s8.pt "
                         "for the GNN arm.")
    ap.add_argument("--sweep-dir", type=Path, required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--workload", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)

    ap.add_argument("--steps", type=int, required=True,
                    help="Gradient steps. No default: the pilot budget is a registration "
                         "question, not a trainer default.")
    ap.add_argument("--episodes-per-cell", type=int, required=True,
                    help="Sampled episodes per cell per step; they share one baseline episode.")
    ap.add_argument("--temperature", type=float, required=True,
                    help="Registered candidates 0.1 / 0.3 / 1.0. Increment 1 measured "
                         "T=0.1 at 17.3%% exploration for +0.05%% RTT.")
    ap.add_argument("--reservoir-k", type=int, default=64,
                    help="Decode batches replayed with grad per episode (out of ~7.5k).")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0,
                    help="Trainer seed. Episode seeds are derived from it and the step, so "
                         "two arms at the same --seed see the same sampling seeds (CRN).")
    ap.add_argument("--advantage-clip", type=float, default=0.10,
                    help="Clip |A| before the gradient, in relative-RTT units. A single "
                         "pathological episode can otherwise move the policy further than "
                         "a whole step of good ones. Under --standardize this is applied "
                         "after the division, so raise it accordingly.")
    ap.add_argument("--standardize", action="store_true",
                    help="Divide advantages by their within-step std. OFF by default: the "
                         "advantage is already a dimensionless relative improvement with a "
                         "meaningful zero, and Adam normalises per-parameter scale anyway.")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--logprob-tol", type=float, default=1e-4,
                    help="Max |replayed - recorded| mean log-prob before the run aborts.")
    ap.add_argument("--episode-workers", type=int, default=1,
                    help="Sampled episodes to run concurrently. They are independent "
                         "subprocesses at fixed seeds, so this changes wall-clock and "
                         "nothing else; size it to --cpus-per-task with OMP_NUM_THREADS=1.")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--wandb-project", default="herosim-closed-loop")
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--keep-episode-json", action="store_true",
                    help="Retain the per-episode result JSONs (~80MB each). Off by default.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.reservoir_k < 1:
        raise SystemExit("FAIL LOUD: --reservoir-k must be >= 1 or pass 2 has nothing to replay")
    seed_everything(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.out_dir / "checkpoints"
    work_dir = args.out_dir / "work"
    ckpt_dir.mkdir(exist_ok=True)
    work_dir.mkdir(exist_ok=True)

    # Check the warm start declares its own contract before anything is copied, so the
    # error names the file the user passed rather than the working copy.
    init_contract = require_contract(args.arm, args.init)
    print(f"[init] {args.init} contract: "
          f"{ {k: init_contract.get(k) for k in ('inference_feature_layout', 'queue_feature_contract', 'network_graph_contract') if k in init_contract} }",
          flush=True)

    # The live checkpoint the episode subprocesses serve. It starts as the warm start and
    # is overwritten after every step; the per-step copies are kept for the gate.
    current = ckpt_dir / f"cl_{args.arm}_current.pt"
    copy_checkpoint(args.init, current)

    configs = {}
    for cell in args.cells:
        path = args.sweep_dir / "configs" / f"{cell}.json"
        if not path.exists():
            raise SystemExit(f"FAIL LOUD: missing cell config {path}")
        configs[cell] = path
    if not args.workload.exists():
        raise SystemExit(f"FAIL LOUD: missing workload {args.workload}")

    import wandb  # rule 5: every training run logs to W&B, no exceptions

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"cl-{args.arm}-T{args.temperature}-s{args.seed}",
        config={
            "lineage": "objective_pivot_v1",
            "phase": "3 / P1 closed-loop",
            "arm": f"CL-{args.arm.upper()}",
            "init": str(args.init),
            "cells": args.cells,
            "workload": str(args.workload),
            "steps": args.steps,
            "episodes_per_cell": args.episodes_per_cell,
            "temperature": args.temperature,
            "reservoir_k": args.reservoir_k,
            "lr": args.lr,
            "seed": args.seed,
            "estimator": "REINFORCE, self-critical baseline, two-pass N/k replay",
        },
        tags=["objective_pivot_v1", "phase3", "closed-loop", f"cl-{args.arm}"],
    )

    history: List[Dict[str, Any]] = []
    t_start = time.perf_counter()

    for step in range(1, args.steps + 1):
        # ---- load the policy as the episodes are about to serve it -------------------
        policy = load_policy(args.arm, current)
        optimizer = torch.optim.Adam(policy.model.parameters(), lr=args.lr)
        optimizer.zero_grad(set_to_none=True)

        samples: List[Dict[str, Any]] = []
        baselines: Dict[str, float] = {}

        # ---- pass 1a: the self-critical baselines ------------------------------------
        # Argmax on a fixed cell and trace is deterministic, so it does not depend on the
        # sampling seed: one baseline episode per cell serves all `episodes_per_cell`
        # sampled ones, which is what keeps the loop at 1+S episodes per cell instead of 2S.
        jobs = [
            dict(config=configs[cell], workload=args.workload, model=current,
                 out_json=work_dir / f"s{step:03d}_{cell}_greedy.json",
                 cell=cell, arm=args.arm, temperature=None, timeout_s=args.timeout)
            for cell in args.cells
        ]
        for base in _run_all(jobs, args.episode_workers):
            baselines[base.cell] = base.total_rtt

        # ---- pass 1b: the sampled episodes -------------------------------------------
        jobs = []
        for cell in args.cells:
            for j in range(args.episodes_per_cell):
                # Derived from the trainer seed and the step, never from the clock: two
                # arms launched at the same --seed draw the same episode seeds, which is
                # the common-random-numbers pairing the registration's primary statistic
                # is computed over.
                ep_seed = (args.seed * 1_000_003 + step * 1009 + j) % (2**31 - 1)
                jobs.append(dict(
                    config=configs[cell], workload=args.workload, model=current,
                    out_json=work_dir / f"s{step:03d}_{cell}_j{j}.json",
                    cell=cell, arm=args.arm, temperature=args.temperature, seed=ep_seed,
                    reservoir_k=args.reservoir_k,
                    replay_out=work_dir / f"s{step:03d}_{cell}_j{j}.replay.pt",
                    timeout_s=args.timeout,
                ))
        for ep in _run_all(jobs, args.episode_workers):
            base_rtt = baselines[ep.cell]
            adv = (base_rtt - ep.total_rtt) / base_rtt
            samples.append({
                "cell": ep.cell, "seed": ep.seed, "rtt": ep.total_rtt,
                "baseline_rtt": base_rtt, "advantage_raw": adv,
                "replay": ep.replay_path,
                "explore_rate": (ep.trajectory or {}).get("explore_rate"),
                "n_batches": (ep.trajectory or {}).get("n_batches"),
                "wall_s": ep.wall_s,
            })
            print(
                f"[step {step}] {ep.cell} seed={ep.seed} sampled={ep.total_rtt:,.1f} "
                f"greedy={base_rtt:,.1f} A={adv:+.5f} "
                f"explore={samples[-1]['explore_rate']} ({ep.wall_s}s)",
                flush=True,
            )

        if not args.keep_episode_json:
            # Live result JSONs are ~80 MB each; a 40-step pilot would otherwise leave
            # tens of GB behind. The numbers that matter are in history.json and W&B.
            for cell in args.cells:
                (work_dir / f"s{step:03d}_{cell}_greedy.json").unlink(missing_ok=True)
                for j in range(args.episodes_per_cell):
                    (work_dir / f"s{step:03d}_{cell}_j{j}.json").unlink(missing_ok=True)

        # ---- advantage shaping -------------------------------------------------------
        # The advantages are NOT mean-centred, and that is deliberate. The usual
        # within-batch centring assumes the baseline is arbitrary; here it is the policy's
        # own greedy episode, so A = 0 already means "sampling matched greedy". Subtracting
        # the step's mean would erase exactly the information the self-critical baseline
        # exists to provide: a step in which every episode beat greedy would come out with
        # zero mean advantage and teach nothing.
        raw = [s["advantage_raw"] for s in samples]
        scale = 1.0
        if args.standardize and len(raw) > 1:
            sd = st.stdev(raw)
            # A step where every episode landed at the same return carries no signal;
            # dividing by ~0 would turn float noise into a full-size update.
            scale = sd if sd > 1e-9 else 1.0
        n_clipped = sum(1 for s in samples if abs(s["advantage_raw"] / scale) > args.advantage_clip)
        if len(samples) >= 2 and n_clipped == len(samples):
            # Every episode pinned to the clip boundary means the step carries only the
            # sign of the advantage, not its size — the episodes stop being distinguishable
            # from each other and the gradient is a constant times a sum of log-probs.
            # It is the signature of a clip set in the wrong units (z-scores vs raw RTT).
            raise RuntimeError(
                f"FAIL LOUD: all {len(samples)} advantages hit --advantage-clip="
                f"{args.advantage_clip} (scale={scale:.3e}, raw={['%.4f' % a for a in raw]}). "
                "The step would carry no relative information between episodes. The clip "
                "is in relative-RTT units (0.10 = a 10% episode swing); raise it, or drop "
                "--standardize, which rescales the advantages out from under it."
            )
        for s in samples:
            a = s["advantage_raw"] / scale
            s["advantage"] = float(max(-args.advantage_clip, min(args.advantage_clip, a)))

        # ---- pass 2: replay with grad ------------------------------------------------
        total_loss = torch.zeros((), dtype=torch.float64)
        n_replayed = 0
        lp_errors: List[float] = []
        for s in samples:
            temperature, n_batches, batches = load_replay(s["replay"])
            if abs(temperature - args.temperature) > 1e-12:
                raise RuntimeError(
                    f"FAIL LOUD: replay {s['replay']} was sampled at T={temperature} but "
                    f"this step trains at T={args.temperature}."
                )
            k = len(batches)
            rescale = n_batches / k
            ep_logprob = torch.zeros((), dtype=torch.float64)
            for b in batches:
                replayed = batch_logprob(policy, b["payload"], b["chosen"], temperature)
                ep_logprob = ep_logprob + replayed
                # Per decision, so the tolerance means the same thing for a 1-task batch
                # and a 4-task one.
                recorded = float(sum(b["logprobs"]))
                lp_errors.append(
                    abs(float(replayed.detach()) - recorded) / max(1, len(b["chosen"]))
                )
            # Ascent on A * logpi => descent on -A * logpi.
            total_loss = total_loss - s["advantage"] * rescale * ep_logprob
            n_replayed += k

        loss = total_loss / max(1, len(samples))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.model.parameters(), args.grad_clip)
        optimizer.step()

        # ---- the correspondence check ------------------------------------------------
        # Pass 2 must reproduce pass 1's log-probs on the SAME weights. It ran before the
        # optimizer step, so any disagreement means the stored payload no longer
        # reconstructs the decode it came from — the gradient would then be pointing at a
        # distribution the simulator never sampled from, and every number after it is
        # meaningless. This is checked every step, not once.
        max_lp_err = max(lp_errors) if lp_errors else 0.0
        if max_lp_err > args.logprob_tol:
            raise RuntimeError(
                f"FAIL LOUD: replayed log-probs differ from the recorded ones by "
                f"{max_lp_err:.3e} per decision (tolerance {args.logprob_tol:.1e}). "
                "Pass 2 is differentiating a different distribution than pass 1 sampled."
            )

        # ---- persist and report ------------------------------------------------------
        provenance = {
            "lineage": "objective_pivot_v1",
            "phase": "3 / P1 closed-loop",
            "arm": f"CL-{args.arm.upper()}",
            "init": str(args.init),
            "step": step,
            "temperature": args.temperature,
            "reservoir_k": args.reservoir_k,
            "lr": args.lr,
            "train_seed": args.seed,
            "torch_seeded": True,
            "estimator": "REINFORCE self-critical, two-pass N/k replay",
        }
        step_ckpt = ckpt_dir / f"cl_{args.arm}_step{step:03d}.pt"
        save_policy(policy, step_ckpt, init_path=args.init, provenance=provenance)
        copy_checkpoint(step_ckpt, current)

        row = {
            "step": step,
            "mean_advantage_raw": st.mean(raw),
            "median_advantage_raw": st.median(raw),
            "frac_sampled_beat_greedy": sum(1 for a in raw if a > 0) / len(raw),
            "mean_baseline_rtt": st.mean(list(baselines.values())),
            "mean_sampled_rtt": st.mean([s["rtt"] for s in samples]),
            "advantage_scale": scale,
            "n_advantages_clipped": n_clipped,
            "loss": float(loss.detach()),
            # Pre-clip: clip_grad_norm_ reports the norm it saw, not the one it applied.
            # It runs large because the N/k rescale multiplies a sum of log-probs by
            # ~N/k, and is expected to sit well above --grad-clip.
            "grad_norm_preclip": float(grad_norm),
            "batches_replayed": n_replayed,
            "max_logprob_replay_err": max_lp_err,
            "wall_s": round(time.perf_counter() - t_start, 1),
        }
        for cell, rtt in baselines.items():
            row[f"greedy_rtt/{cell}"] = rtt
        history.append(row)
        wandb.log(row, step=step)
        print(
            f"[step {step}] mean_A={row['mean_advantage_raw']:+.5f} "
            f"beat_greedy={row['frac_sampled_beat_greedy']:.2f} "
            f"greedy_rtt={row['mean_baseline_rtt']:,.1f} "
            f"grad_norm={row['grad_norm_preclip']:.1f} clipped={n_clipped}/{len(samples)}",
            flush=True,
        )
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=1))

        for s in samples:
            Path(s["replay"]).unlink(missing_ok=True)

    # ---- final: the paired statistic the kill criterion reads ------------------------
    # The registered primary is CL-GNN minus Frozen-GNN under CRN, measured by the gate,
    # not here. What this run can say is whether the greedy policy it produced improved
    # on the greedy policy it started from, on the training cells.
    first, last = history[0], history[-1]
    delta = (first["mean_baseline_rtt"] - last["mean_baseline_rtt"]) / first["mean_baseline_rtt"]
    summary = {
        "arm": f"CL-{args.arm.upper()}",
        "init": str(args.init),
        "steps": args.steps,
        "greedy_rtt_first_step": first["mean_baseline_rtt"],
        "greedy_rtt_last_step": last["mean_baseline_rtt"],
        "train_cell_improvement_rel": delta,
        "final_checkpoint": str(ckpt_dir / f"cl_{args.arm}_step{args.steps:03d}.pt"),
        "note": (
            "Training-cell improvement is NOT the registered result. The kill criterion "
            "reads the paired-seed CL vs Frozen difference from a held-out live gate; "
            "this number is a training curve and can be positive while the gate is not."
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    wandb.summary.update(summary)
    run.finish()
    print("\n=== CLOSED-LOOP TRAINING COMPLETE ===")
    print(f"  greedy RTT {first['mean_baseline_rtt']:,.1f} -> {last['mean_baseline_rtt']:,.1f} "
          f"({delta:+.2%} on the training cells)")
    print(f"  final checkpoint: {summary['final_checkpoint']}")
    print("  next: live-gate it against Frozen-GNN / Frozen-MLP / Knative under CRN.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
