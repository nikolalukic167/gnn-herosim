# Positioning: an exhaustive hand bar for learned schedulers (DRAFT, 2026-10-12)

**DRAFT for the user's morning read; not a submission, not the paper.** It frames what the program has
established as a contribution. Every number links to the node that owns it; the caveats on those nodes'
heads are carried here. Paper texts are cited as `file:line` in `/root/papers_text/`.

## The contribution in one paragraph

Learned (GNN) schedulers are reported as wins over hand heuristics. We built the hand baseline those
papers do not run: an exhaustive search over the placement plans of each batch, scored with the same
information the model receives. Against that bar, no GNN arm we trained wins, and a per-batch oracle is
under 3 % ahead of it live. Most of the oracle's remaining room sits in the cluster's cold start, not in
steady-state placement. The GNN's own wins are real, but they are wins over plain coordinate descent (CD)
and over its own message-passing-off twin. Those are the same kinds of baseline the literature reports
against. The result is a measurement: in this setting the problem is enumerable, and a learned
scheduler must be read against an enumerating hand bar.

## 1. The bar

`cd_exactS_ib0` (`ib0_confirm_v1.md:7`, `:14`; arm table `CONTEXT.md:233`) is the hand bar:

- CD over the batch's peer group;
- exact enumeration of the surrogate S over the top-5 candidate slate per task (100k-plan cap,
  expansion fallback above it);
- the in-batch stacking term zeroed (`HEROSIM_PG_BATCH_ALPHA=0`), with the platform backlog kept.

S uses the information the GNN is given: queues, warmth, execution time per platform type, and partner
placements. `sparse_obs_s0_v1` asks whether that information is more than a real scheduler has (§2).

**What the published learned schedulers compared against.** "Model's information" asks whether the
strongest baseline scores candidates with the same inputs the learned policy sees.

| paper | learned policy | strongest baseline | model's information? | exhaustive over the decision? |
|---|---|---|---|---|
| Decima (Mao+ SIGCOMM'19) | GNN + RL, Spark DAG jobs | tuned weighted fair, α grid-swept (`01_learned_placement_and_NCO/Decima_SIGCOMM19.txt:612–618`, 21 % gain `:703–704`); Graphene\* with grid search (`:622–626`, 32–43 % `:649`) | partly: the heuristics see job size and DAG structure, but not Decima's learned parallelism control | **closest case:** App. H searches all n! job orderings on a 10-job batch (`:1393–1400`), and Decima matches it or beats it by 9 % (`:1450–1451`). That search covers job order only, not executor placement or parallelism. |
| L2D (Zhang+ NeurIPS'20) | GNN + PPO, job-shop dispatching | priority rules SPT, MWKR, FDD/MWKR, MOPNR (`L2D_NeurIPS20.txt:379–383`); OR-Tools only as a gap reference (`:385–386`) | yes for the rules (static instance, exact times) | no exhaustive arm. **Our re-run** in their own environment: FDD/MWKR 547.5 beats learned 572.1 at 6×6 (`literature_reeval_v1.md:89`, `:103`). |
| Placeto (Addanki+ NeurIPS'19) | GNN + RL, device placement | Single GPU, Scotch, human expert, the RNN placer (`Placeto_NeurIPS19.txt:302–306`; 16.5 % vs RNN `:92`, `:356`) | Scotch sees the graph but uses its own cost model | no |
| Mirhoseini'17 (ICML) | seq2seq RL, device placement (not a GNN) | Scotch, MinCut, human expert (`Mirhoseini_DevicePlacement_ICML17.txt:76`, `:133–134`) | no (Scotch balances partitions and does not see measured runtime) | no |
| Sparrow (Ousterhout+ SOSP'13) | none (hand) | random, per-task sampling; "within 12 % of an ideal scheduler" (`p69-ousterhout.txt:24–25`) | — | no; batch sampling = least loaded of d·m probes (`:95–97`) |
| Firmament (Gog+ OSDI'16) | none (hand) | Quincy, distributed schedulers (`osdi16-gog.txt:45`, `:99–117`) | — | **yes, for its cost model:** min-cost max-flow over all tasks of a batch (`:39`, `:98`). Exact batch optimisation is a known hand technique; no learned scheduler we read compares against it. |
| HEFT/CPOP (Topcuoglu+ TPDS'02) | none (hand) | DLS, MH, LMT list heuristics (`HEFT_CPOP_Topcuoglu_TPDS02.txt:333–336`) | — | no |
| SONIC (Mahgoub+ ATC'21) | none (hand) | AWS Lambda, SAND, OpenLambda ± Pocket (`04_serverless_data_passing_coldstart_queueing/SONIC_ATC21_full.txt:55–57`) | — | **yes, for its own decision:** Viterbi over the DAG's data-passing choices (`:368`, `:377`) |

**Read.**

- Every learned scheduler in the table is measured against priority rules, partitioners or tuned
  heuristics. Decima is the one exception, and its exhaustive search covers the job-ordering part of the
  decision only; there, it matched the search.
- Exact batch optimisation exists as a hand technique (Firmament, SONIC). Our bar is not new as an
  optimisation idea. What is new is using it as the control for a learned placement policy, with the
  model's own information.
- Scope of the literature claim: these are the papers in the local mirror plus the earlier table
  (Gasse, GDP, Lachesis, WeCAN, EP-NCO, GRAF, Raith; scratchpad, not in the repo). "No published GNN
  scheduler" needs a systematic search before it can be written as such.

### What the repos ship (2026-10-12)

An agent cloned the open-source repos and read their code. I re-checked the table below against the
clones by grep and file listing; commits are in `positioning_exhaustive_bar_2026-10-12/README.md`.
"Reproducible < 1 day" is a yes only where something was actually run. "MP-off twin" means a trained
checkpoint or config with message passing disabled; a depth knob alone is not one.

| repo | strongest baseline in code | MP-off twin shipped? | costs deterministic? | reproducible < 1 day? |
|---|---|---|---|---|
| `decima-sim` (Mao, TF1) | dynamic-partition fair share with an executor cap (`heuristic_agent.py:8`, `:28`) | no (`--max_depth` knob only, `param.py:94`) | no: task durations drawn at random from profiled TPC-H samples (`spark_env/node.py:64–90`, seeded) | not run |
| `spark-sched-sim` (PyG port, with pretrained Decima) | round-robin fair / FIFO and random (`schedulers/heuristics/`) | no | no: durations sampled (`data_samplers/tpch.py:211`), Poisson arrivals (`:70`), seeded | **yes, with shims** (minutes per seed; see below) |
| `L2D` (Zhang+) | none runnable: PDRs are not shipped; `env_lab.py:63` imports an `ortools_baseline` module that is not in the repo | no | yes (static job-shop) | **yes** (pretrained 10×10, 100-instance test set, 21 s) |
| `L2S` (Zhang+ ICLR'24) | greedy local-search baselines from SPT / FDD÷MWKR / plist starts (`conventional_baselines_with_long-term-mem.py:90–95`, `:154`); OR-Tools CP-SAT as gap reference (`ortools_solver.py:3–68`) | no (GIN / DGHAN / GIN+DGHAN module ablations, not MP-off) | yes | not run |
| `ScheduleNet` (reimpl., zcaicaros) | none found in code | no | yes (job-shop) | not run |
| `park` (Mao+) | environments only (`park/envs/spark`, `spark_sim`, `load_balance`, `tf_placement`, …); no scheduler baselines found | no | per environment; `spark` needs a real Spark cluster | not run |
| `goal-co` (GOAL, Naver) | none in code; pretrained checkpoints and test sets only (README) | no | yes (static CO instances) | not run |
| `wheatley` (Jolibrain) | dispatching rules incl. MOPNR (`jssp/dispatching_rules/heuristics.py:88`); OR-Tools with a duration-estimate strategy (`args.py:253–381`) | no (`--n_layers_features_extractor` knob, `args.py:405`) | optional: `--duration_type stochastic` (`docs/JSSP.md:90–100`) | not run |

Not found as public code: Placeto, Spotlight, Lachesis, GDP, the official ScheduleNet.

**Cross-cutting.** No repo ships a baseline that does exhaustive search with the model's information,
and none ships an MP-off twin. The closest things are OR-Tools as a gap reference (L2S, wheatley) and
the depth knobs.

**Two measured lines, smoke scale, direction only.** Scripts and logs are in
`positioning_exhaustive_bar_2026-10-12/`.

- **(a) L2D, in the authors' own environment** (their left-shift insertion), with their pretrained
  10×10 model on their 100-instance test set (seed 200; L2D mean makespan 982.3). Hand dispatch rules
  tie or beat it:
  - MWKR 976.5, −0.59 % (53/100 instances better, 1 tie);
  - FDD/MWKR 952.8, −3.00 % (65/100);
  - MOPNR −0.20 % (43/100);
  - SPT +22.96 % (1/100);
  - a one-step lower-bound lookahead +26.68 % (2/100).

  The paper's 15–20 pp PDR deficit does not reproduce here, and the paper's PDR construction is not
  shipped. This agrees with `literature_reeval_v1` (FDD/MWKR beats the learned policy in their env,
  `literature_reeval_v1.md:89`, `:103`).
- **(b) Decima PyG port** (`spark-sched-sim`, pretrained weights, under the shims; see the README).
  - **The hand rule:** shortest-remaining-work job first, with a per-job executor cap (about 20 lines).
    A cap equal to the executor count means no cap.
  - **Decima repeats:** each seed runs Decima 3 times with torch, `random` and numpy seeded per repeat,
    and the table reports their mean. The policy samples its action with Python's global
    `random.choices` (`schedulers/decima/utils.py:21`), which the environment seed does not set. That
    is the source of the unseeded run-to-run difference seen earlier (610 vs 638 s).
  - **Spread across the 3 repeats:** 2–9 % of the mean (e.g. 587–637 s on seed 1234). With every RNG
    pinned to one value, 3 runs gave 600.84 s each, identical (`sss_handrule_seeds.log`, last line).
    The script for that pinned check is not among the attachments.

  **Table A: 10 executors, 50 jobs, 4 seeds** (`sss_handrule_seeds.log`; mean job duration in s; % vs
  Decima's mean).

  | seed | fair | Decima (3 repeats) | SJF cap 5 | SJF cap 10 (no cap) | Decima vs fair |
  |---|---|---|---|---|---|
  | 1234 | 988 | 611 (587–637) | 551 (−9.9 %) | 527 (−13.7 %) | −38.2 % |
  | 1 | 958 | 553 (537–585) | 447 (−19.2 %) | 490 (−11.3 %) | −42.3 % |
  | 2 | 842 | 506 (498–510) | 420 (−17.0 %) | 462 (−8.7 %) | −39.9 % |
  | 3 | 1124 | 827 (812–848) | 678 (−18.0 %) | 691 (−16.5 %) | −26.4 % |

  Both caps beat Decima on 4/4 seeds, by −8.7 to −19.2 %. Picking the better cap per seed would be post
  hoc, so quote each cap's own range. Decima beats fair by −26 to −42 %.

  **Table B: 50 executors, 200 jobs** (Decima's training environment, `config/decima_tpch.yaml:81–83`;
  `sss_handrule50_seeds.log`; 2 seeds only, about 225 s each; seed 2 produced an empty log).

  | seed | fair | Decima (3 repeats) | cap 5 | cap 10 | cap 25 | cap 50 (no cap) | Decima vs fair |
  |---|---|---|---|---|---|---|---|
  | 1234 | 62 | 54 (53–54) | 144 (+168.3 %) | 81 (+50.7 %) | 69 (+29.0 %) | 92 (+70.7 %) | −12.9 % |
  | 1 | 43 | 39 (39–40) | 107 (+171.8 %) | 62 (+56.4 %) | 48 (+22.2 %) | 59 (+49.9 %) | −9.3 % |

  - A variant that fills unknown stage durations with the job's mean known duration (`SJF2`,
    `sss_handrule_seed50b.py`) gives the same numbers as SJF at caps 10 / 25 / 50 on both seeds
    (`sss_handrule50b_seed_*.log`). Filling durations changes nothing here.
  - Every hand variant loses to Decima, by +22 to +172 %.

  **Reading.**
  - The 10-executor win is over a model served out of its training distribution (trained at 50
    executors, served at 10), in a small, heavily loaded pool (fair 842–1,124 s against 43–62 s at the
    training config). That is the load-contingency pattern our own record documents
    (`unsaturated_edge_v1`, `peer_affinity_v1`: live headlines that depend on load).
  - At the training config, a 20-line SJF rule does not transfer. The paper's tuned weighted-fair
    heuristic, which it reports 21 % behind Decima (`Decima_SIGCOMM19.txt:703–704`), is not shipped. So
    the right hand bar for Decima is still untested here.
  - **Decima is the one system we dissected where the shipped GNN beats the best shipped baseline in its
    own regime.** That baseline is fair share, and the hand bar we would hold it to (tuned weighted-fair
    or better, with Decima's information) has not been run.
  - Smoke scale, direction only: 4 seeds and 2 seeds, under the shims.

## 2. The measurements

All live results are in the accel seat (160 clients × 24 servers, R1.1 + WF1, warmth `node_disk_v2`,
`fastest_compatible`, moderate ×11.6139 / heavy ×27.6227). A negative % means the first arm is faster.

**Against the bar, every GNN arm loses.**

- `slate_fix_v1.md:6`: GNN retrained and served on the uncapped, correctly cold-priced slate vs
  `cd_exactS_ib0`: self-search +2.6 / **+5.3 %**, no-split +2.8 / **+7.4 %** (moderate / heavy; E-FASTER,
  Holm ≤ .03; 12 fresh topologies × g0–g1). The heavy gap grew by about 1 pp (`:8`).
- `selfsearch_confirm_v1.md:19`: earlier checkpoints, vs `cd_exactS` (the bar before `ib0`): +3.66 /
  +3.40 % at heavy (2/24), tie at moderate.
- Caveats: two training seeds per arm; whole-run means only, with no steady-state number for this gate
  (`slate_fix_v1.md:20`). The cells ran at two commits, joined by a 2-cell identity (`:21`).

**The bar is within 3 % of a per-batch oracle, live.**

- `ib0_confirm_v1.md:7`: the full-coverage label oracle beats `cd_exactS_ib0` by **−2.82 %** moderate
  (6/6, p .031) and **−1.96 %** heavy (4/6, p .31), on 6 topologies.
- This is a myopic per-batch oracle (the best plan for each batch on its co-simulated label), not a global
  optimum over the trace. "Within 3 % of optimal" must be written as "within 3 % of a per-batch label
  oracle".
- The bar itself beats `cd_exactS` by −4.73 / −2.27 % (12/12) and CD by −12.07 / −7.36 % (`ib0_confirm_v1.md:161`).

**The remaining room is mostly the deploy transient.**

- `ib0_confirm_v1/oracle_loss_diag.md:38`, `:55–56` (time split): 84 % (heavy) / 68 % (moderate) of the
  oracle's label gain over the bar sits in the first 5 % of each trace. That opening is the whole
  system's cold start (autoscale from zero, no warm-up discard; `docs/adr/0004-steady-state-primary-statistic.md`,
  PROPOSED). After it, the batch-level gain is 5.35 / 2.75 % of hand seconds, and less live.
- Caveats: within-cap batches only, 6 cells per rung; label seconds, not live latency.

**Information sparsity does not move it.**

- `sparse_obs_s0_v1.md:5–10` (HAND-RECOVERS-UNDER-SPARSITY):
  - queue counts instead of the exact drain walk cost the bar +0.7 / +0.3 % (DRAIN-NO-VOI);
  - a naive warmth estimate costs +11.1 / +6.4 %, but a per-platform-class prior recovers it
    (+0.1 / −1.8 %, n.s.); no residual is left for a learner;
  - the partner channel is not testable in this seat; exec was skipped by argument.
- Steady-state lines agree with the whole-run reads (`:11`).

**The GNN's wins are over weak or ablated baselines, the kind the literature reports.**

- **Over plain CD:**
  - `selfsearch_confirm_v1.md:12–13`: self-search −8.46 % moderate (23/24, Holm < .0001) and −3.09 %
    heavy; the physMP checkpoint −10.09 / −3.76 %. Confirmed at moderate; heavy is under the −5 % bar.
  - `accel_nosplit_v1.md:10`: the no-split decode −6.93 / −7.71 % at moderate (CONFIRMED).
  - Caveat: both are on the cold-mispriced declared slate (`declared_pruning_v1`) and not re-measured;
    `slate_fix_v1` re-measured against the hand bars, not against CD.
- **Over its MP-off twin:** `slate_fix_v1.md:11–13`: no-split −2.57 % heavy (12/12, Holm .002) and
  −1.55 % moderate (11/12), two seeds. That is DIRECTION-ONLY as an arm under the registered −2 % bar,
  self-search is n.s., and both modes lose to `cd_exactS_ib0`. The 4-seed replication is `slate_fix_v2`
  (REGISTERED, not read).
- **Matched slate:** `slate_bar_v1.md:5`: with the hand search restricted to the GNN's declared top-5, the
  GNN self-search is ahead by −4.85 / −1.94 % (21/24, 16/24). The restriction costs `cd_exactS`
  +4.9 / +5.0 %. This is a candidate-set handicap, not an MP win.

## 3. The claim

In a serverless placement environment like this one:

- deterministic costs;
- trace-grounded fan-out batches with a mean group size of about 4 tasks (3.91–4.09, max 10;
  `big_groups_s0_v1.md:211`);
- a few candidates per task (median eligible 4–6, p90 9–10, max 13–22 without a slate limit; served
  slate 5; `big_groups_s0_v1.md:205`).

**Under these conditions the batch placement problem is enumerable.** An exhaustive hand search with the
model's information comes within 3 % of a per-batch label oracle live. No GNN we trained beats it, with
or without message passing. GNN scheduling wins reported over heuristic baselines in comparable
settings (small batches, enumerable candidate sets, costs a hand model can price) should be re-read
against such a bar.

Out of scope (each is a separate claim):

- sequencing problems with a non-decomposable objective (makespan, critical path), where Decima and L2D
  live;
- hidden information that a class prior cannot recover;
- batches too large to enumerate. `big_groups_s0_v1` at G = 8: the hand search is still faster live.

**The small-batch condition is partly ours.** The workload mint caps every fan-out at 10 tasks. Over the
groups it draws, 7.5 % exceed 10, and they hold **37.5 %** of member tasks; 24.0 % of tasks sit in
groups larger than 20, and the largest is 2,876 (`grounded_workload_v1.md:109–124`; over the mint's draw,
not over the whole trace). The claim holds for the workload we served. It is not yet a claim about the
uncut trace.

## 4. When would a GNN be necessary?

_(SCIENTIST's answer, 2026-10-12, edited for voice. Probabilities are SCIENTIST's priors, not
measurements.)_

**Why exhaustive hand search is near-optimal here.** Nothing is broken. The accel seat is a small, fully
known, nearly separable assignment problem. On such a problem, search over a correct cost model is
near-optimal by construction, and a learned scorer can at best match it.

- **Near-additive objective.** Co-sim targets fit a sum of per-task terms (additive R² up to 1.00000,
  argmin exact; `program_verdict_v1.md:149`).
  - The one coupling is the pairwise exchange term. Hiding the batch plan costs the bar +113 / +170 %
    (`sparse_obs_s0_v1.md:10`). Search prices that term exactly.
  - Node-indexed contention is closed by the count theorem (`dag_fabric_contention_v1`).
  - Links never bind: at most 1.6 % of latency (`link_bound_s0_v1`).
- **Enumerable coupled set, but only below the 10-task cut** (above).
  - 59,908 of 73,279 moderate batches (82 %) and 56,700 of 73,281 heavy (77 %) are within the enumeration cap
    (`ib0_confirm_v1/oracle_loss_diag.md:16–17`).
  - Above the cap, hand local search is the better fallback. Top-k pruning beats the expansion fallback
    (`prune_g8_s0_v1`), and the GNN on over-cap batches is +0.2 / +2.5 % behind (`gnn_overcap_v1`).
- **Every cost is a deterministic table.**
  - Hidden execution time has no headroom (`hidden_exec_s0_v1.md:9`, an earlier seat).
  - The exact queue walk is worth under 1 %. Warmth is recovered by a per-class prior
    (`sparse_obs_s0_v1.md:6–7`).
- **No hidden state that only a learned model could predict.**
  - Partners arrive in the same batch, so the partner channel is not testable in this seat
    (`sparse_obs_s0_v1.md:9`).
  - Pricing partners that have not arrived yet: a hand self-predict rule recovers 93–95 % of an oracle
    (`lookahead_mp_v1`).
  - Within S's table, ΔS explains ≥ 98.9 % of the label's variance across plans. Plan-structure features
    add nothing on held-out cells (`oracle_loss_diag.md`, "Time split").
- **Nothing outside the batch changes the best per-batch choice.**
  - A cost-to-go re-rank ties live (`cost_to_go_v1`).
  - Rollout imitation wins offline (−12.7 %) and loses live (+14–28 %; `rollout_imitation_v1`).
  - The oracle's room is mostly the deploy transient (§2).

**The logic.** With a known deterministic model and enumerable batches, every training label comes from
search. A model trained on search labels can at best match that search's quality and can only win on
decision time. Every learned win in the record fits this, and none needs message passing:

- matched slate: the GNN beats a handicapped `cd_exactS` (`slate_bar_v1`);
- a GNN-seeded CD beats matched hand CD, but its MP-OFF twin beats the GNN seed (`gnn_seeded_cd_fixed_v1`,
  `gnn_seeded_cd_mp_ablation_v1`);
- the GNN proposal adds to a hand selector (`gnn_proposal_selector_v1`).

**Candidates, with SCIENTIST's prior for a GNN win of at least −5 % (Holm) against the named bar.**

| | change | what the record already says | bar | P(win) |
|---|---|---|---|---|
| A | coupled sets too large to enumerate (DAG stages, 30–100 operations) | timed hand search won every workflow / DAG / mixed lineage by 1.8–5 %, and the GNN tied MP-OFF | time-bounded `ib0` search | ~5 % (decision time free), ~8 % (charged) |
| B | stochastic exec or transfer times with learnable structure | `hidden_exec_s0_v1` NO-HEADROOM; per-input regression is pointwise (an MLP's job); co-location interference is node-indexed | hand rule with per-class means | ~3 % (~1 % needing MP) |
| C | closed-loop effects (per-batch best is wrong for the trajectory) | `cost_to_go_v1`, `rollout_imitation_v1`, `lookahead_mp_v1` negative; high ρ unreachable while stable (`high_rho_s0_v1`) | `ib0` + a hand value term | ~4 % |
| **D** | **remove the 10-task cut, serve real fan-outs as joint batches, and charge each arm's decision time as simulated time** | the cut is ours (37.5 % of tasks beyond it); `cd_exactS` already uses ~76 % of one scheduler thread at heavy (`decision_cost_s0_v1.md:13`) | `cd_exactS_ib0` and its time-bounded variant, both paying for decision time | ~10 % (~3 % also beating its MP-OFF twin) |

**SCIENTIST's pick: D.**

- It is the only option that breaks a property without inventing physics.
- It is the one pattern where learning has beaten an exact method in print: a learned component inside
  an exact search, winning on solve time (Gasse et al. NeurIPS'19).
- Real schedulers decide in milliseconds (Sparrow, Firmament).
- Kill checks:
  - Check 1, fan-out share beyond the cut ≥ ~5 %: **PASSED** (37.5 %).
  - Check 2, `cd_exactS` loses ≥ 3 % to its own decision time at k = 1: from `decision_cost_s0_v1`, not yet
    read. If it fails, stop.
- Cost if both pass, SCIENTIST's estimate:
  - hand arms live on an uncut mint, ~100–150 CPU-h;
  - labels by ILS replay (brute force cannot label these batches), ~0.5–1k CPU-h (from memory);
  - GNN + twin training and a live gate, ~150 CPU-h.
- The prior stays low because `big_groups_s0_v1`, `scale_sweep_v1` and `gnn_overcap_v1` lost or tied when
  decision time was free.

**Published GNN wins against an exact baseline with the same information.** SCIENTIST knows of none on
solution quality.

- L2D trails OR-Tools by 17.7–29.2 % (its Table 1).
- Kool et al. stay behind Concorde, LKH and Gurobi.
- Decima is compared against tuned heuristics. The one exhaustive search it faces covers job ordering
  only, and Decima matches it (§1).
- What survives is learning inside an exact search that wins on time, which is the reason for D.

## Caveats that travel with every quote

- **Declared-slate cold defect.** Every accel GNN number before `slate_fix_v1` is on the cold-mispriced
  declared slate and not re-measured.
- **Deploy transient.** Every live number is a whole-run mean including the scale-from-zero opening
  (about 40 % of heavy mean latency). Steady-state is a secondary line from 2026-10-12. Making it primary
  is the user's decision (ADR 0004, PROPOSED).
- **Seeds.** GNN contrasts use two training seeds per arm. `slate_fix_v2` adds seeds 3–4.
- **Not MP.** No message-passing win over any hand bar exists in the record.
- **One seat.** Every number is the accel seat; no claim transfers to another seat without a gate.
- **Oracle scope.** The oracle is per batch and myopic, on 6 topologies, with 96.0 % / 98.4 % coverage of seen tasks
  (`ib0_confirm_v1.md:7`).
