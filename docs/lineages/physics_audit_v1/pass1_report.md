# physics_audit_v1 -- instrumentation, checkers and the I11 replay fix (report)

Branch `rp/audit` (worktree `.claude/worktrees/rp-audit`), rebased onto `origin/reference-physics` `99fac565`, three commits on top
(`995444ee`, `14b6bd6d`, `2366ad31`). Nothing under `docs/lineages/` or `LINEAGES.md` was touched; `docs/gates/gate-tools.md` was not
edited either (the `env.step()` finding is drafted at the end for you to file).

R1 = `HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa`, peer exchange on, single origin, server-only replicas.

## What exists

| piece | file |
|---|---|
| opt-in trace (`HEROSIM_AUDIT_TRACE=<path>`) | `src/placement/physics_audit.py` + guarded hooks (`if _AUDIT is not None`) |
| snapshot fidelity block (`HEROSIM_SNAPSHOT_FIDELITY=1`) | `src/placement/snapshot_fidelity.py` + hooks in `live_audit`, `live_snapshot_seed`, `simulation`, `autoscaler`, `infrastructure`, GNN autoscaler, determined scheduler |
| checkers I1-I10, I12 | `scripts_cosim/physics_audit/check_invariants.py` |
| I11 replay harness | `scripts_cosim/physics_audit/i11_replay.py` |
| result comparator (masks wall-clock only) | `scripts_cosim/physics_audit/replay_identity.py` |
| I8 driver | `scripts_cosim/physics_audit/run_determinism.sh` |
| `env.step()` reproduction | `scripts_cosim/physics_audit/env_step_repro.py` |
| tests (29, incl. a seeded-defect negative control per checker) | `tests/test_physics_audit.py` |

## Identity checks (the default path replays byte for byte)

Compared with wall-clock fields masked (`gnn_decision_time`, `schedulingTime`, `total_inference_time`, ...) and, for old-vs-new code, the
`run_provenance.code` block (it correctly says "dirty"):

- original code (`99fac565`) vs this branch, all flags off: **36/36 cells IDENTICAL** (6 topologies x x2/x3/x5 x CD and Knative, 4,000 arrivals),
  R1 physics; and **6/6 IDENTICAL** on the legacy physics (store-and-forward, held replicas, legacy scale-out).
- trace on == trace off, and trace + snapshots + fidelity on == trace off: IDENTICAL (also asserted by `tests/test_physics_audit.py::test_default_path_and_trace_identity`
  when pointed at a cell via `HEROSIM_AUDIT_TEST_CFG/WORKLOAD`).
- repo suite: 26 failures on the original code and the same 26 on this branch (1,476 passed on both); no regression.

## I11 (blocking): FAILED before, now passes -- with qualifications

Bar: median <= 1 %, p95 <= 5 %. Failed replays count as errors of infinity, never excluded.

**Reference.** The earlier comparison used the *continuing* live run, which includes arrivals after the decision. Under R1 those matter (below), so
the reference is now the live run **cut at the decision** (events that arrived by the decision instant, none after): the simulator is deterministic,
so it follows the full run to the decision and then executes the chosen plan alone. This is "the same plan executed live from that state", and is what
a co-sim label means. The harness checks the cut run chose the same plan and scheduled the batch at the same instant (else the row is an error).

**Sample.** 216 states: 6 topologies (9483, 9491, 9506, 9533, 9550, 9568) x x2/x3/x5 x 12; per cell 6 evenly spaced over all decisions (uniform) and
6 drawn round-robin from the hard-state tags (targeted). First 3,000 arrivals of workload g0, CD arm.

| replay | failed | median | p95 | max | within 1 % |
|---|---|---|---|---|---|
| **with the fidelity block** | **0 / 216** | 0.00 % | 0.08 % | 0.98 % | **216 / 216** |
| original (oracle-style, same states, same reference) | 93 / 216 | 95.7 % | inf | inf | 41 / 216 (19 %) |

Fidelity replay by rung: x2 p95 0.09 % (max 0.98), x3 p95 0.04 % (max 0.45), x5 p95 0.28 % (max 0.59). By topology: p95 0.03-0.29 %, max <= 0.98 %.
Uniform half: 108/108 exact to < 0.01 %. Targeted half: p95 0.28 %. Composition (tags overlap): 38 clean, 78 with tasks held by released replicas,
89 with queued tasks (up to 23 replayed as real tasks), 83 with replicas still pulling, 132 with KPA in panic, **5 with a busy link pipe**.

### What was missing, and what closes it (the five you listed, plus four more found on the way)

1. KPA window history and panic state: samples, first/last traffic, panic times, replica birth causes, **and the tick phase**. The next tick is read from
   the Timeout the autoscaler process is suspended on (a process suspended on the mutex ticks after the decision in progress); `last_tick + interval` is wrong
   because of the `env.step()` finding below.
2. Live autoscaler settings: target concurrency 0.7 (the oracle's hard-coded 100), scaled keep-alive and tick (`snapshot_fidelity.live_run_params`);
   the co-sim `determined` arm runs the GNN-family autoscaler the live arms ran (A4) when the flag is on.
3. Tasks held by released replicas: ingress (remaining propagation, pipe wait/hold, pipes taken at apply time in the live queue order), cold start,
   rendezvous, input I/O, waiting for the compute lock, computing/output -- each resumed from the stage it was caught in. **Tasks in cold start/ingress get their
   input-stage length computed at capture.**
4. Replicas still pulling: tracked **per `initialize_replica` call** (the code can run twice for one platform; a per-platform record was overwritten).
   A pull holds the node's local storage for its whole duration, so every task on the node that needs storage (input, and the output stage) waits for it;
   queued pulls are priced by playing the node's FIFO queue forward, a later waiter finding the image cached by an earlier pull. Replay pulls hold the real storage and cache the image at pull start.
5. Per-replica warmth: `previous_task` per platform (the old seed marked every initialised replica warm for its function), plus idle/started/allocated/removed times and node image caches.
6. (found) Free platforms: the base seed triggered `initialized` on *every* platform in the queue snapshot, so a replica the replay's KPA created was ready at once and skipped its pull.
7. (found) Queued tasks: replayed as real tasks (forced onto their live platform, ahead of the batch) instead of the compressed virtual backlog, which under release blocks the pop loop for the whole drain time.
8. (found) Batch tasks' partners outside the batch: stubs that carry where they run; a partner not yet placed is a live rendezvous (see caveat).
9. (found) The first replayed KPA tick waits for the scheduler's first decision, because the live tick waited on the mutex behind it.

### Findings that are not bugs in the replay (they are properties of R1)

- **Under released replicas, the latency of a placed batch depends on arrivals after it.** Against the *continuing* live run the same replays miss in
  36 of 216 states (median 0, max 94 %); all 36 are in the first 27 s of the trace. Mechanism: a task takes the compute lock, then stalls in its output stage
  waiting for the node's local storage, which an image pull is holding (up to ~18 s); later-scheduled tasks on the same platform take the lock first. From 300 s on all 104 states
  match even the continuing run (max 0.59 %). A co-sim label (batch alone) therefore cannot equal the live latency of that batch in these states, whatever the snapshot holds.
- The oracle's `CosimOracleContext(seed=101)` builds a **different topology** from the live cell (316 vs 328 routes on 9483) and the simulator then hits
  `sys.exit(1)` on a missing connection; the harness now catches `SystemExit`.

## I1-I10, I12

Runs: 6 topologies x x2/x3/x5 x {CD, Knative `reactive`}, 12,000 arrivals each, trace on (36 runs); I8 on 12 cells x 2 runs.

| | result | notes |
|---|---|---|
| I1 Little's law | PASS 36/36 | counter vs rows at every KPA tick (2.9 M samples over the 36 runs, **0 violations**); L = lambda W worst gap 0.75 % (9-14 platforms/run with >= 200 tasks). Sampling the counter at ticks and averaging read ~1.5x too high (ticks queue behind decisions), so the counter is compared pointwise |
| I2 transfer time | PASS 36/36 | 1,000 ingress transfers/run, observed (arrive - pop - wait) vs route latency + size/bottleneck: max error 3e-11; peer sums agree |
| I3 store-and-forward off | PASS 36/36 | |
| I4 released replicas | PASS 36/36 | 0 compute overlaps, ~3,200 overlapping I/O pairs/run, 0 cold starts after the same function was popped |
| **I5 scale-out causality** | **FAIL 8 CD + 18 Knative; FAIL-WITH-CAUSE 10 CD** | CD: load share median 0.44 (8/18 runs > 50 %); Knative 0.77. Spearman(load-caused replicas, in-flight concurrency, 10 s bins): CD median -0.09, Knative +0.12 (bar 0.5). Information only: against KPA's own 60 s stable-window average the same correlation is +0.64 (CD) / +0.72 (Knative). Replicas follow the demand KPA measures, not instantaneous load |
| I6 memory | PASS 36/36 | **peak use is 4.5 % of a node at most and no refusal occurred: the refusal path is untested** |
| I7 conservation | PASS 36/36 | |
| I8 determinism | PASS, 12/12 cells identical | a third run with another `PYTHONHASHSEED` also identical except the recorded env var |
| I9 cold-start accounting | PASS 36/36 | cold starts == sandbox creations in pop order; no warm start charged |
| I10 decision time | PASS 36/36 for the two instrumented paths | decisions consume zero simulated seconds; scheduler paths without a hook report NOT-TESTED, not PASS |
| **I12 rung configuration** | **FAIL** | policy time scale 0.5 / 0.333 / 0.2 on x2/x3/x5, so keep-alive 15/10/6 s, KPA windows 30/20/12 s and tick 0.5/0.33/0.2 s differ per rung |

I12's question (why CD's latency fell from x2 to x5): in these runs (first 12,000 arrivals, mean of 6 topologies) CD latency is 0.678 / 0.649 / 0.650 s; cold starts per
task 0.229 / 0.206 / 0.182 and cold-start time per task 0.087 / 0.074 / 0.060 s, exchange per task 0.319 / 0.272 / 0.235 s, queueing flat. The cell is not the same measurement as
`transfer_physics_v1` (full traces, 19 topologies), so this does not settle it; the rung-dependent time scale and falling cold-start/exchange share are consistent with the explanations the node lists.

## The `env.step()` finding (for gate-tools.md)

`Autoscaler.autoscaler_process` and `_kpa_autoscaler_process` end each iteration with `self.env.step()` (src/placement/autoscaler.py, "Next event"). `Environment.step()` pops and
processes the next event in the queue, whatever it is and whenever it is due, and moves the clock to it. What the loop consumes therefore depends on what else is queued, so **any
additional scheduled event changes the run, even from a process that reads and writes nothing.** Also: autoscaler ticks are not `last_tick + interval`.

Minimal reproduction: `python scripts_cosim/physics_audit/env_step_repro.py` (standalone SimPy, one extra `Timeout`, prints two different sets of job completion times; exits 0 when they differ).
In the real simulator (`--sim CFG WORKLOAD --events 2000`): 0, 1 and 100 idle timeouts give total RTT 1288.517030 s (identical); **1,000 idle timeouts give 1311.276689 s (+1.8 %)**;
my first occupancy sampler (a 0.1 s process) moved the same cell from 1288.5 to 1077.7 s (-16 %). A single event is usually harmless in the full simulator (consumed early without
reordering anything that matters) and not harmless in the toy; a stream is not harmless in either. Consequence: instrumentation must schedule nothing (the recorder does not), and any new SimPy
process added to a run is a physics change.

## Limits (what this does not show)

- One workload (g0), the first 3,000 arrivals, CD arm; 18 cells x 12 states are **not 216 independent draws** (start-up states repeat across cells; the uniform picks line up across cells).
- No state had an unplaced partner outside the batch, so the oracle-supplied `future` placement path (`--future live`) was never used; open peer groups are unreproduced by construction.
- Only 5 states had a busy link pipe, all exact; the heavier-load behaviour of the pipe ghosts is thin.
- The Knative arm has no snapshot-capture path, so I11 covers CD only. Fidelity capture requires released replicas, one local storage per node, single-task applications (it raises otherwise).
- I1 and I2 use the simulator's own counters/records against independently derived values; they cannot catch a bug that is consistent across both (e.g. a wrong formula used everywhere).
- I5's verdict uses the literal pre-registered statistic; I did not change it.
