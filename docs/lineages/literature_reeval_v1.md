# literature_reeval_v1 — the missing control, run on the literature's own code

**Status: `ACTIVE` (registered 2026-09-09, before any full run). Target 1: L2D (Zhang et al.,
NeurIPS 2020) — closed out 2026-09-10, all three registered sizes (6×6/10×10/15×15) at the
full 8 seeds/arm; see the Record's last four entries for the final reading. Target 2: Decima —
assessed, not started. Placeto — reference-only, see §6.**

## Why this lineage exists

This program answered its own question on HeROsim: every measured GNN-vs-pointwise edge fell
to a matched control (`throughline.md`, 2026-09-09 section). The scheduling-GNN literature
(Decima, Placeto, L2D) reports large wins over hand-written heuristics but almost never runs
the control this program ran — the same architecture with **message passing disabled**, or a
**pointwise MLP given the aggregates the GNN computes**. This lineage runs exactly that control
on the authors' own environment, reward, training loop and budget, so the question "does the
learned scheduler need message passing?" gets a measurement instead of a citation.

Nothing here touches HeROsim. Code lives in `scripts_reeval/l2d/`; upstream repos are cloned
outside the tree (no licence files ⇒ not vendored) and imported unmodified.

## Target 1 — L2D (job-shop scheduling, GIN + PPO)

**Upstream:** `zcaicaros/L2D` @ `7b2efbb` (2022-08-31), run under this repo's pipenv
(torch 2.5.1, a 2-name `gym` shim). Their saved 6×6 network evaluated in this environment
gives 572.08 on their test set vs 574.09 in the paper (Table 1) — the substrate is faithful.

**What the GNN can see (the T1 contract).** The env gives each operation at most three
in-neighbours — itself, its job predecessor, its machine predecessor
(`JSSP_Env.step`). With K = 2 GIN layers the receptive field is a **fixed 7-slot typed tuple**,
so "everything message passing aggregates" is a 20-dim hand-built vector
(`arm_features.typed_receptive_field`). This is the same move as the count theorem on HeROsim:
bounded typed neighbourhoods make graph aggregation a finite feature map.

### Arms — identical architecture, PPO, seeds, budget; only the policy's inputs differ

| Arm | Adjacency | Node input | What it tests |
|---|---|---|---|
| `gnn` | real disjunctive graph | raw 2 (LB/1000, finished) | upstream exactly |
| `mpoff` | identity | raw 2 | no neighbourhood MP, same global mean readout (DeepSets) — this program's MP-OFF analogue |
| `mlp_t1` | identity | 20 = typed 2-hop tuple + presence flags | same information as `gnn`, no learned MP |
| `mlp_t1x` | identity | 27 = t1 + 7 hand scalars (dur, work remaining, ops remaining, machine remaining work, machine ready, earliest start, global LB) | the hetdem/krank analogue |

The global mean readout stays in every arm on purpose: the ablated quantity is *neighbourhood*
message passing, not permutation-invariant global context. Model code is untouched
(`ActorCritic` with `input_dim` set per arm); parameter count differs only in the first layer
(2/20/27 → 64).

### Protocol (fixed before the first full run)

- Sizes: 6×6 and 10×10 with **8 seeds/arm**; 15×15 with **5 seeds/arm** (cost). Study seed
  `s` → `torch_seed = 600+s`, `np_seed_train = 1000+s`. Upstream's own defaults (200/600) are
  not used: upstream draws its first 100 training instances from the same RNG state that
  generated its validation set (`np.random.seed(200)` in both), a small leak we avoid.
- Budget: upstream's 10 000 updates × 4 envs, lr 2e-5, all upstream hyper-parameters.
- Checkpoint protocol: **primary = upstream's** (best validation makespan on their Seed200
  set, evaluated every 100 updates); **secondary = final checkpoint** — route_b Phase 2 showed a
  selector can manufacture a gap, so both are read and both are reported.
- Evaluation split: **100 fresh instances, np seed 300**, frozen at
  `simulation_data/literature_reeval_v1/l2d/test_{size}_seed300.npy`. Upstream's Seed200 set
  is their validation *and* their Table-1 test set; we report it but never as primary.
  Taillard `tai15x15` as a third split at 15×15. Greedy decoding, as upstream tests.
- Statistic: per-seed mean test makespan per arm; Δ% = (arm − gnn)/gnn, positive = arm worse.
  **Primary test: two-sided exact Mann–Whitney U on per-seed means, `gnn` vs each arm, α = 0.05
  per size.** Secondary: per-instance paired Wilcoxon on seed-averaged makespans.
- Readings (per size, per contrast, primary checkpoint protocol):
  `GNN-NEEDED` if p < 0.05 and median Δ ≥ +1.0 %; `POINTWISE-BETTER` if p < 0.05 and
  median Δ ≤ −1.0 %; `TIE` if p ≥ 0.05 and |median Δ| < 1.0 %; otherwise `INDETERMINATE`.
  The primary contrast is **`gnn` vs `mlp_t1`** (matched information, no MP); `mpoff` and
  `mlp_t1x` are secondary. Registered prediction from this program's record: `mlp_t1` TIE,
  `mpoff` GNN-NEEDED (it lacks the lookahead the tuple carries), `mlp_t1x` TIE or better.
- Validity gates, reported not silently applied: (G1) our `gnn` arm's seed-mean lands within
  3 % of upstream's SavedNetwork on the same split (else our training loop is not faithful);
  (G2) no NaN/inf loss; (G3) every arm's curve improves on its first validation read.
- Every run logs to W&B project `literature-reeval-l2d` (repo rule 5); each checkpoint carries
  a `.contract.json` sidecar (arm, dims, seeds, upstream commit).

Commands: `scripts_reeval/l2d/run_study.sh <n_j> <n_m> <n_seeds>` →
`scripts_reeval/l2d/evaluate.py` → `scripts_reeval/l2d/analyze.py`.

### Found before the first full run — the paper's baseline numbers do not reproduce (2026-09-09)

Running the four PDRs the paper compares against (supplement §2 definitions) **through the
paper's own environment** (semi-active schedule with permissible left shift), on the paper's own
Seed200 test instances:

| 6×6, mean makespan | SPT | MWKR | FDD/MWKR | MOPNR | learned (paper / their checkpoint here) |
|---|---|---|---|---|---|
| paper Table 1 | 691.95 | 656.95 | 604.64 | 630.19 | 574.09 |
| their env, their definitions | 690.17 | **572.07** | **547.50** | **567.56** | 572.08 |
| classical non-delay generator | 567.31 | 543.54 | 539.51 | 551.51 | — |

| 10×10 | SPT | MWKR | FDD/MWKR | MOPNR | learned |
|---|---|---|---|---|---|
| paper Table 1 | 1210.98 | 1151.41 | 1102.95 | 1101.08 | 988.58 |
| their env, their definitions | 1207.87 | 976.52 | 952.82 | 980.39 | — |
| classical non-delay generator | 980.71 | 939.18 | 920.31 | 940.64 | — |

SPT reproduces (so the harness is right); MWKR/MOPNR/FDD-MWKR are 10–15 % better than the paper
reports under every variant tried (remaining-work with/without the current op, deterministic or
random ties — `pdr_variants.paper_variant_probe`), and none of the variants lands on the paper's
numbers. Every schedule was re-derived from machine sequences by an independent validator
(`validate_sequences`: precedence, non-overlap, completeness) and matched the reported makespan.
**Read:** under the paper's own generator, FDD/MWKR beats the learned policy at 6×6 (547.5 vs
572–574) and at 10×10 (952.8 vs 988.6); under the standard non-delay generator all four rules do.
The paper's headline "outperforms all baseline PDRs by a large margin" does not survive its own
environment. This is independent of the message-passing question and is recorded here because
it changes what the arms are being compared against: the reference row in every analysis is the
PDRs *as run here*, not Table 1.

## Record

- **2026-09-09 — registered.** Arms, seeds, statistic and readings above fixed; smoke runs of
  all four arms (20 updates) pass; evaluator reproduces the upstream checkpoint; PDR discrepancy
  found and validated. Full 6×6 and 10×10 grids launched locally (CPU, 2 threads/run).

- **2026-09-09 — 6×6 read (32 runs, 0 failures) and Amendment A1.** Fresh test split, upstream
  checkpoint protocol (best validation): `gnn` 602.2 (per-seed 577–593 plus one collapsed draw at
  716), `mpoff` 607.3, `mlp_t1` 622.8, `mlp_t1x` 639.2. Contrasts: `mpoff` vs `gnn` −0.28 %,
  p = 0.083 → **TIE**; `mlp_t1` vs `gnn` +2.30 %, p = 0.083 → **INDETERMINATE**; `mlp_t1x` vs `gnn`
  +5.81 %, p = 0.028 → **GNN-NEEDED**. Final-checkpoint protocol: +1.07 % (p = 0.065), +7.38 %
  (p = 0.010), +12.88 % (p = 0.007). On upstream's own Seed200 set every contrast reads GNN-NEEDED.
  Reproduction gate G1 passes (our `gnn` mean is +2.5 % from upstream's saved network on the test
  split; median seed 588.7 vs 587.5). **Every learned arm loses to FDD/MWKR run through the same
  env (560.9) and to MOPNR (573.5) on the test split**, including the best single `gnn` seed (577.1).
  The registered prediction was wrong in an informative way: the raw-feature no-MP arm, which holds
  strictly less information than the tuple arm, does better than it, and three of eight `mlp_t1x`
  seeds never improved after their first 100 updates (best validation at update 100). That is an
  optimisation failure of the richer-input arms under the authors' learning rate (2e-5, tuned for
  their GNN), not a message-passing effect. The typed receptive-field builder was checked against
  an independent derivation from the env's machine sequences on 720 states (all slots, all flags).
  **Amendment A1 (registered before any sweep run):** each arm additionally trains at
  lr ∈ {1e-4, 5e-4}, 8 seeds, 6×6 first; per arm, the lr with the lowest mean *validation* makespan
  across seeds is its operating point; the registered contrasts are then re-read on the test split
  with each arm at its own operating point ("each arm at its own best lr"), alongside the fixed-lr
  reads. The 2e-5 reads above stand as the "authors' recipe" row and are not replaced.

  **What this means for the paper's headline claim.** Table 1's central claim — the learned policy
  beats four traditional dispatching rules "by a large margin" at every size — does not hold at 6×6
  once the rules are computed correctly. This is not limited to our retrained copies: on upstream's
  own Seed200 validation instances, upstream's own *published, pre-trained* checkpoint (587.5) also
  loses to FDD/MWKR run correctly through upstream's own environment (547.5). The message-passing
  question above is therefore secondary to this: even the full model with message passing, as the
  authors trained and shipped it, is beaten by a one-line rule once that rule is not under-computed.
  This is a reproducibility failure of the paper's primary empirical support, found independently of
  and prior to the message-passing ablation. Scope: confirmed at 6×6 only so far (the paper's
  smallest instance size); 10×10 and 15×15 reads (running) will show whether it persists as
  instances grow, which the paper's own reported gap pattern would predict it should.


- **2026-09-09 — Amendment A1 read (64 runs, 0 failures), and Amendment A2 (registered before
  re-running).** Each arm, at 6×6, picks the SAME operating point on validation:
  `lr=5e-4` beats `1e-4` and the authors' `2e-5` for all four arms (validation mean 559–587 vs
  572–625 vs 580–625). At `lr=5e-4` the earlier tie inverts: `mpoff` vs `gnn` +1.05 % (test,
  p = 0.007) / +2.22 % (val, p = 0.0006), both **GNN-NEEDED**; `mlp_t1` GNN-NEEDED on both splits;
  `mlp_t1x` INDETERMINATE (test) / GNN-NEEDED (val). At the intermediate `lr=1e-4` the reads sit
  between the tie and the reversal (`mpoff` INDETERMINATE, `mlp_t1`/`mlp_t1x` GNN-NEEDED). The GNN's
  across-seed sd also drops sharply as lr rises (46.2 → 34.3 → 4.7 on the test split), while the
  pointwise arms' sd does not fall the same way.

  **This is not read as a verdict — the fixed 10,000-update budget does not reach convergence for
  any arm at `lr=5e-4` or `1e-4`.** The validation-curve slope over the last 20 % of training is
  still negative (still improving) for every arm at every lr except the authors' own `2e-5` (where
  `gnn` is flat); at `lr=5e-4` the pointwise arms' tail slope (−0.52 to −0.63 makespan/100 updates)
  is *steeper* than the GNN's (−0.34), i.e. the pointwise arms are closing, not opening, the gap in
  the last fifth of training. Raising the learning rate uniformly changed a tie into a GNN advantage,
  but it did so inside a budget where the trailing arms have not finished catching up — the same
  shape of error the checkpoint-selection artifact above was. **Amendment A2 (registered before any
  run):** extend `lr=5e-4` training to `max_updates=30000` for all four arms × 8 seeds, same
  seeds/split, until each arm's validation tail slope is within its own noise of flat (post-hoc
  check, not a stopping rule that can be gamed by looking early); read the registered contrasts only
  at that point. Do not report an lr=5e-4 verdict before A2 completes.


- **2026-09-09 — 15×15 read (20 runs, 5 seeds/arm, 0 failures).** Upstream checkpoint protocol
  (best validation), fresh test split: `gnn` 1614.1, `mpoff` 1586.8, `mlp_t1` 1614.8,
  `mlp_t1x` 1815.8; `mpoff` vs `gnn` −2.51 % (p = 0.84, INDETERMINATE — n = 5 is under-powered for
  this spread, not evidence of a tie); `mlp_t1` vs `gnn` −2.40 % (p = 1.00, INDETERMINATE);
  `mlp_t1x` vs `gnn` +12.84 % (p = 0.10, INDETERMINATE). Final-checkpoint protocol: `mpoff` +0.05 %
  (p = 0.55, **TIE**), `mlp_t1` +5.50 % (p = 0.31, INDETERMINATE), `mlp_t1x` +15.61 % (p = 0.032,
  **GNN-NEEDED**, the only significant read at this size). Reads on `tai15x15` agree in shape
  (`mpoff` TIE at final, everything else INDETERMINATE). No message-passing direction reaches
  significance at n = 5 except the `mlp_t1x` outlier, which is consistently the worst arm at every
  size and checkpoint protocol so far — read as an optimisation/capacity issue of that feature set,
  not evidence for message passing.

  **The PDR finding replicates at the paper's own reported comparison size.** `FDD/MWKR` run
  through the paper's own environment scores 1463.3 (test) / 1464.8 (val) / 1534.6 (Taillard),
  beating every learned arm at every checkpoint protocol and every split, including upstream's own
  published checkpoint (1498.5 / 1507.3 / 1530.5) and our best-trained `gnn` (1614.1 / — / 1655.6).
  This is now confirmed at 6×6, 10×10 and 15×15: the paper's headline comparison does not hold at
  any size checked, using upstream's own formulas, environment and (at 6×6/15×15) upstream's own
  trained weights.


- **2026-09-09 — Amendment A3 (registered before any run): apply the same discipline to 10×10
  and 15×15.** Two gaps in the record so far: (1) 10×10 has only been trained at the authors'
  `lr=2e-5`, so it is unknown whether the 6×6 reversal (tie at the authors' rate, apparent
  GNN-NEEDED at a faster rate, unresolved by convergence) shows up at a second size; (2) 15×15
  ran only 5 seeds/arm (a cost cut from the registered 8), which is why its reads came back
  INDETERMINATE rather than a clean tie or a clean win — an under-powered null is not evidence of
  a tie. Actions: (a) 10×10 lr sweep, `lr ∈ {1e-4, 5e-4}`, 8 seeds × 4 arms = 64 runs, same
  validation-only selection rule as A1, then the same convergence check as A2 (train to a flat
  validation tail, not a fixed step count) before reading any contrast at the faster rate;
  (b) 15×15 topped up from 5 to the registered 8 seeds/arm at the authors' `lr=2e-5` (3 more seeds
  × 4 arms = 12 runs) so the existing reads can be re-read at full power. The authors'-recipe rows
  already recorded are not replaced; these add rows, they do not revise the existing ones.


- **2026-09-09 — caught and fixed a false-completion bug in the monitoring, before it reported
  anything.** The original 6×6→10×10 chain watcher counted completion with
  `ls l2d_{size}_*/final.pth | wc -l ≥ 32`; once Amendment A1/A3's lr-tagged runs
  (`l2d_10x10_gnn_lr0.0005_s0`, etc.) landed in the same `runs/` directory, the same glob matched
  them too, so the count reached 32 while 18 of the 32 authors'-recipe 10×10 runs were still
  training (confirmed: only 14/32 had `final.pth`, three `gnn` seeds mid-run at 63–75 % of budget).
  The watcher was stopped before it read or reported anything from the contaminated evaluation.
  Fixed by enumerating the exact 32 `{arm}_s{seed}` names rather than a glob. No numbers from the
  false trigger were recorded or reported.


- **2026-09-09 — 10×10 lr-sweep read (64 runs, 0 failures) — the 6×6 reversal replicates at a
  second size, and this time the training curves are much closer to flat.** Validation-only
  selection: `gnn`/`mpoff`/`mlp_t1x` pick `lr=5e-4`, `mlp_t1` picks `lr=1e-4` (all lower than the
  authors' `2e-5`). Validation-tail slope at the picked rate is near zero for every arm (|slope| ≤
  0.6 makespan per 100 updates, versus 3–6 makespan/100 updates at 6×6's `lr=5e-4`) — this read is
  materially closer to convergence than the 6×6 fast-rate read was, though not yet re-verified with
  an explicit A2-style extended run at this size. Each arm at its own picked rate, fresh test split:
  `gnn` 1000.0, `mpoff` 1026.3 (+2.28 %, p = 0.021), `mlp_t1` 1021.4 (+0.93 %, p = 0.235), `mlp_t1x`
  1086.4 (+9.32 %, p = 0.0002). Validation split agrees (+2.08 %/p=0.021, +1.44 %/p=0.51,
  +9.64 %/p=0.0002). **Direction and rough magnitude match the 6×6 fast-rate read** (there:
  `mpoff` +1.05 % p=0.007 test / +2.22 % p=0.0006 val): no message passing costs about 2 points
  once training is fast enough to actually converge, at both sizes checked so far. The authors'
  own `lr=2e-5` row for 10×10 is not yet complete (still training; see the false-completion bug
  above) and will be added when it finishes, alongside a convergence check at this size to match
  A2's discipline before this is read as more than a strong preliminary signal.

- **2026-09-09 — 15×15 re-read at the registered 8 seeds/arm (32 runs, 0 failures) — supersedes
  the n = 5 read above, which was underpowered rather than a tie.** Same authors' recipe
  (`lr=2e-5`, 10000 updates) as before, topped up from 5 to 8 seeds/arm. Best-validation checkpoint
  protocol, fresh test split: `gnn` 1614.6, `mpoff` 1591.3 (−2.57 %, p = 0.57, **INDETERMINATE**),
  `mlp_t1` 1633.1 (−3.04 %, p = 0.72, **INDETERMINATE**), `mlp_t1x` 1786.3 (+11.12 %, p = 0.028,
  GNN-NEEDED). Final-checkpoint protocol: `mpoff` 1667.3 (−0.06 %, p = 0.23, **TIE**), `mlp_t1`
  1695.7 (+1.91 %, p = 0.28, INDETERMINATE), `mlp_t1x` 1813.4 (+12.68 %, p = 0.028, GNN-NEEDED).
  `val` and `tai15x15` splits agree in shape at both checkpoint protocols (`mpoff` TIE or
  INDETERMINATE, never GNN-NEEDED; `mlp_t1` never GNN-NEEDED; `mlp_t1x` GNN-NEEDED or
  INDETERMINATE, never a tie). **No message-passing direction reaches GNN-NEEDED at this size
  except the `mlp_t1x` outlier** — consistent with 6×6 and 10×10 at the authors' own learning
  rate: at `lr=2e-5`, message passing is a tie, not a win, and the one arm that adds hand-built
  scalar features on top of the typed tuple (`mlp_t1x`) is consistently the worst-optimising arm
  at every size and protocol measured in this lineage, which reads as an optimisation/capacity
  cost of that particular feature set, not evidence for message passing.

- **2026-09-10 — moved the remaining local runs to datalab (Amendment A2 6×6, Amendment A3
  10×10-authors, and the 15×15 re-evaluation) to free the local host, which had two 14-way
  training arrays plus an evaluation process sharing one 32-core box (load average ~58, and an
  unrelated stale background task was killed by a low-memory guard as a result).** Completed
  local checkpoints were rsynced to datalab first so no seed was retrained; three new sbatch
  scripts (`literature_reeval_l2d_10x10_authors.sbatch`, `literature_reeval_l2d_6x6_a2.sbatch`,
  `literature_reeval_l2d_15x15_eval.sbatch`) carry a skip-if-`final.pth`-exists guard mirroring
  `run_study.sh`'s local pattern. All 32+32 training tasks and the evaluation completed with
  exit code 0 on datalab; two further sbatch jobs (`literature_reeval_l2d_10x10_eval.sbatch`,
  `literature_reeval_l2d_6x6_eval.sbatch`) ran the corresponding evaluations. No local box time
  was lost — completed local work was preserved, not repeated.

- **2026-09-10 — Amendment A2 read (6×6, `lr=5e-4`, 30000 updates, 32 runs, 0 failures) —
  convergence confirmed before reading.** Validation-tail slope over the last 20 % of training
  is ≤ 0.0005 makespan/update for every arm (essentially flat; contrast the −0.34 to −0.63
  makespan/100-updates seen in the unconverged 10000-update read above), so this is the first
  6×6 fast-rate read taken after every arm actually finished learning. Median validation
  makespan: `gnn` 569.1, `mpoff` 613.2, `mlp_t1` 608.6, `mlp_t1x` 657.8.

  **Held-out test split (job 749757 read): the first converged, adequately-powered GNN-NEEDED
  result in this lineage.** Best-validation checkpoint: `gnn` 577.8, `mpoff` 593.5 (+1.51 %,
  p = 0.0047, **GNN-NEEDED**), `mlp_t1` 592.7 (+3.30 %, p = 0.0148, **GNN-NEEDED**), `mlp_t1x`
  600.9 (+4.38 %, p = 0.0499, **GNN-NEEDED**). `val` split agrees and is cleaner: `mpoff` +2.65 %
  (p = 0.0002), `mlp_t1` +4.53 % (p = 0.0047), `mlp_t1x` +4.72 % (p = 0.0070), all GNN-NEEDED.
  Final-checkpoint protocol agrees in direction (`mpoff` +3.51 % test / INDETERMINATE at
  p = 0.065, +5.47 % val GNN-NEEDED; `mlp_t1` and `mlp_t1x` GNN-NEEDED on both splits). This is
  the only read anywhere in this lineage that is simultaneously (a) run at a learning rate the
  arms actually converge at (tail slope ≤ 0.0005 makespan/update, vs. −0.34 to −0.63 makespan/
  100-updates in the unconverged 10000-update budget) and (b) powered at the full registered
  8 seeds/arm. **Read plainly: at the authors' own learning rate (2e-5) every arm is
  under-trained and message passing is a tie; once training is pushed to convergence at a
  faster rate, message passing wins by roughly 1.5–5 points on this task-shop instance size.**
  This does not contradict the authors' published numbers (which never converge past 10000
  updates at their own rate either) — it identifies *why* the tie was measured: the authors'
  recipe stops before the pointwise arms' catch-up finishes, not because message passing has
  nothing to add once given the chance to matter. Whether this generalizes past 6×6 is the
  open question the 10×10 lr-sweep read above (measured GNN-NEEDED in the same direction, at a
  less complete convergence check) already points toward, and the 10×10 authors'-recipe read
  below settles the paper's own operating point at a second size.

- **2026-09-10 — 10×10 authors'-recipe read (32 runs, `lr=2e-5`, 10000 updates, 0 failures) —
  closes the gap left by the false-completion bug, and matches the 6×6/15×15 shape at the
  authors' own operating point.** Best-validation checkpoint, fresh test split: `gnn` 1036.0,
  `mpoff` 1039.5 (−0.48 %, p = 0.065, **TIE**), `mlp_t1` 1072.6 (+0.51 %, p = 0.130, **TIE**),
  `mlp_t1x` 1143.5 (+10.40 %, p = 0.010, GNN-NEEDED). `val` split agrees (`mpoff` INDETERMINATE,
  `mlp_t1` TIE, `mlp_t1x` GNN-NEEDED); final-checkpoint protocol agrees in direction on both
  splits (`mpoff`/`mlp_t1` TIE or INDETERMINATE, never GNN-NEEDED; `mlp_t1x` GNN-NEEDED
  throughout). `FDD/MWKR` again beats every learned arm and the upstream checkpoint at this size
  (955.4 test / 952.8 val vs. `gnn` 1036.0 / 1023.5 and upstream 988.9 / 981.8), confirming the
  PDR-baseline finding at the authors'-recipe operating point specifically (the earlier 10×10 PDR
  read used the lr-sweep arms).

  **Standing picture across all three sizes now measured at the authors' own learning rate
  (6×6, 10×10, 15×15): message passing is a tie or an indeterminate null — never GNN-NEEDED —
  for `mpoff` and `mlp_t1` at every size and checkpoint protocol. It only becomes GNN-NEEDED
  when training is pushed past the authors' 10000-update budget to actual convergence at a
  faster learning rate (6×6 Amendment A2, 10×10 lr-sweep) — both of which measure a small
  (roughly 1.5–5 pp) but statistically significant edge for message passing. The two findings
  are not in tension: the authors' own recipe under-trains, and an under-trained comparison
  cannot distinguish "no benefit from message passing" from "the pointwise arms haven't
  finished catching up yet." Once that confound is removed, message passing does help on this
  benchmark — a different and more interesting result than either "ties everywhere" or "wins
  everywhere" would have been, and the opposite of what this lineage's registered prediction
  expected going in (Amendment A1 registered mpoff catching up as `lr` rose, not the GNN pulling
  further ahead).** The `mlp_t1x` arm (typed tuple plus 7 hand-built scalars) is the worst
  arm at every size, every learning rate, and every checkpoint protocol measured in this
  lineage without exception — read as an optimisation cost of that specific feature
  concatenation, not as evidence against pointwise scoring in general (`mlp_t1`, the plainer
  typed-tuple arm, never shows this pathology).

## Targets assessed, not started

**Decima (`hongzimao/decima-sim` @ `c010dd7`).** Viable: TPC-H DAG data is checked in (28 MB),
seeding is clean, message passing is a one-line ablation (`gcn.py` — return the per-node
`prep` embedding before the depth loop), baselines FIFO and dynamic partition exist. Costs:
TF 1.15 / Python 3.7 environment (tf.contrib), **no pre-trained checkpoint ships** despite the
README, per-decision recomputation of descendant aggregates for the pointwise arm, and
~150–650 core-hours per run (16 simulator workers) ⇒ 3 arms × 3 seeds ≈ 1 500–6 000 core-hours.
Decision deferred until the L2D result is in.

**Placeto (`aravic/generalizable-device-placement` @ `d9a81a9`).** Not runnable as published: eight
imported modules and one class are missing from the repo, datasets are an external link. Its
`--no-msg-passing` flag shows the authors ran this ablation themselves; citable, not reproducible.
