# route_c_link_transfer_v1 — REGISTERED

> **Status:** `REGISTERED` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Record spans:** 2026-08-26 → 2026-08-26

**Outcome.** Screen registered 2026-08-26 before generation; **name is reserved and is only claimed if the screen passes.** Asks whether an environment where link waiting is a material share of RTT resists a fairly-armed pointwise competitor.

**Related:** [link_contention_v1](link_contention_v1.md) · [route_b_v1](route_b_v1.md)

> **Split note.** This node was carved out of a single 4,995-line `LINEAGES.md` on
> 2026-08-27; the section bodies below are byte-for-byte as written. An *above* / *below*
> pointer inside one may refer to a section that now lives in another node — the
> **Related** links say which.

## Record

Newest first; the sections themselves are in chronological order below.

- [route_c_link_transfer_v1 — SCREEN REGISTRATION (2026-08-26, registered BEFORE generation)](#route-c-link-transfer-v1-screen-registration-2026-08-26-registered-before-generation)

---

### route_c_link_transfer_v1 — SCREEN REGISTRATION (2026-08-26, registered BEFORE generation)

**Question.** §9b/§9d say node-memory contention is pointwise-closable with the right
feature layout. The one mechanism pointwise link controls could NOT repair is link
contention (`link_contention_v1`, FALSIFIED on magnitude only: 0.08–0.35% regret). Can an
environment where link waiting is a *material* share of RTT resist a fairly-armed pointwise
competitor? If yes, the stage-2 re-registration is rewritten against that environment
(Branch B of the 2026-08-26 handover); if no, Branch A proceeds on the current corpus.
The lineage name is reserved; it is **named only if this screen passes**.

**Physics facts the screen rests on** (verified in code 2026-08-26): link waiting accrues
ONLY on the client→executing-node ingress transmission of a task's INPUT
(`infrastructure.py` store-and-forward loop; payload = `stateSize[app]["input"]` via
`scheduling_cost.transfer_time`). The parent→child dependency payload (`output`, Arm S's
800 MB lever) never touches the fabric — `_payload_transfer_time` charges hop-count time
with no pipes. Hence the new input-only lever `HEROSIM_INPUT_SIZE_BYTES`
(`executecosimulation.apply_state_size_override`; mutually exclusive with
`HEROSIM_STATE_SIZE_BYTES`), and hence the competitor block counts co-use on ingress
routes, not DAG-edge routes.

**Instruments** (all changes landed and validated before this registration):
- `score_route_b_contention.py` with the opt-in `linkrank` block: fixed-width order
  statistics of per-link co-use over the plan's ingress routes (top-4 counts, excess
  Σ(c−1)+ all/core, shared-link counts all/core; 8 columns, identity-free, poolable).
  Registered §9a statistics are proven unchanged (default blocks =
  `T1_REGISTERED_BLOCKS`; old-vs-new report diff identical on 24 Arm S datasets; one
  plan's co-use hand-verified against `infrastructure.json` routes). New per-dataset
  arms: `r_exact_repaired_lnk_pct` (all rows incl. unconstrained), `r_exact_repaired_t1lnk_pct`
  (constrained rows).
- `route_b_coefficient_transfer.py --add-linkrank`: extends ONLY the exploratory krank
  pooled arm with `linkrank` (registered §9b cells untouched; without the flag the frozen
  §9c artifacts reproduce exactly: pooled 0.7898, mean_tied 0.8242, R² 0.0138 — re-run
  2026-08-26).
- `HEROSIM_RETAIN_LINK_STATS=1`: every `placements.jsonl` row carries `link_wait_total`,
  `link_transfer_avg`, `fabric_link_wait_total` (fail-loud, `HEROSIM_RETAIN_TASK_TIMES`
  precedent) — the manipulation-check statistic.

**Registered reading, per rung.** Scorer alphas `2.0,6.0` (+ auto `None`); objective rtt.
Two channels: the **unconstrained row** is candidate 1 (pure link coupling — anchor: on
Arm S today unconstrained R_exact = 0.000 across 24/24 datasets, so any nonzero here is
the fabric), and **α=2.0** is candidate 2 (link + memory contention; α=6.0 is the
near-unconstrained cross-check for the pooled machinery, which needs finite caps).
A rung is GNN-promising on a channel iff ALL of:
1. *Manipulation check*: median over datasets of (Σ per-plan `link_wait_total` share of
   summed rtt) ≥ **10%** — otherwise the rung failed to make contention material and is
   INVALID (no verdict read from it, ladder continues).
2. *Firing*: ≥ **15%** of datasets at `r_exact_pct > 5.0` on that channel.
3. *Closure*: pooled anonymous closure (krank + dim36crk-expressible + linkrank, ONE
   coefficient set; `--add-linkrank`) median fraction < **0.5** on the firing set,
   reading `mean_tied`, with registered and optimistic readings agreeing within 0.1
   (disagreement ⇒ AMBIGUOUS: widen the ladder, never lower the bar). Per-dataset
   `r_exact_repaired_lnk_pct` / `t1lnk` are reported as diagnostics, no verdict.
Screen PASSES iff at least one VALID rung is GNN-promising on either channel. §9c's
write-up rule binds: a failed screen reads "this environment class is pointwise-closable;
the corpus is the limit", never "the GNN is falsified".

**Disclosure.** During instrument validation (before this registration was written) the
`--add-linkrank` pooled arm was run once on the existing Arm S corpus: pooled median
0.648 (mean_tied 0.790) vs 0.790 (0.824) without. No verdict is read from it — rung 0
has no manipulation check (old jsonl lacks link fields) and its bandwidth (1000 MB/s)
makes contention ~0 by the anchor above.

**Rungs** (4-task, local, `ROUTE_B_PILOT_V1_GRID` physics + Arm S env
`HEROSIM_DATA_LOCALITY=1 HEROSIM_OUTPUT_SIZE_BYTES=800000000` +
`HEROSIM_COSIM_KEEP_ALIVE=1000000 HEROSIM_RETAIN_TASK_TIMES=1 HEROSIM_RETAIN_LINK_STATS=1`,
24–48 datasets/rung; backbone via preset `backbone_defaults`):
- R1: n_core=4, attach_degree=1, chord_count=0 (the measured `link_contention_v1`
  coupling peak), bandwidth 1000 MB/s, `HEROSIM_INPUT_SIZE_BYTES=157286400` (150 MB).
- R2+: lower bandwidth (100, then 25 MB/s) at the same topology until the manipulation
  check passes; bandwidth moves link cost's share of RTT even though it cannot move the
  wait/transfer ratio (the 2026-08-18 null-lever result — the ratio lever is crossings,
  which n_core=4 maximizes).
- Contingency (only if AMBIGUOUS): 8-task (2× diamond4, α ladder ×2 per §9d equal
  tightness) at the best 4-task rung, on datalab via the `route_b_8task_probe.sbatch`
  pattern — concurrency is the known 7–14× amplifier (real-trace A/B).
- Control anchor per rung family: the existing Arm S corpus itself (bandwidth 1000,
  input 153,600 B) with its measured unconstrained 0.000.

**Decision gate** (write the outcome here): PASS → name the lineage, register its full
gate before any full corpus, rewrite the stage-2 registration against the new
environment; the GNN's claim-to-beat is the linkrank-augmented pointwise model. FAIL on
all valid rungs → Branch A on the current corpus with the honest sentence above.
AMBIGUOUS → widen the ladder.

**4-task ladder outcome (2026-08-26): ALL THREE RUNGS INVALID — AMBIGUOUS, contingency
invoked.** R1/R2/R3 (24 datasets each, local, corpora
`gnn_datasets_dag4_route_c_link_screen_r{1_bw1000,2_bw100,3_bw25}`): link-wait share of
rtt median 0.0007 / 0.0045 / 0.0104 against the 0.10 bar
(`simulation_data/route_c_screen_manipulation.json`). The failure is **structural, not a
tuning miss**: the bandwidth-free ceiling wait/(wait+transfer) — the share link waiting
reaches even if link cost consumed ALL of rtt — is median 4–6%, max 8.8%, because one
client and a diamond DAG cap concurrent transfers at 2 (root and sink transfer alone; the
two mids always share the client trunk). No bandwidth or payload value can pass the
manipulation check in this family — the 2026-08-18 null-lever result reappearing at the
rtt level. Corroborating diagnostic (no verdict — rungs invalid): unconstrained R_exact
is **exactly 0.000 on all 72 datasets across the 40× bandwidth range**
(`simulation_data/route_c_screen_4task_rtt.json`), i.e. at this concurrency link waiting
never moves the argmin at all; the α=2.0 firing seen is route_b's known memory channel.
Per the registered gate: widen the ladder along the only remaining lever, concurrency —
the 8-task contingency rung (grid `route_c_link_screen_8task`, 2 diamond4 instances from
independent clients, bw=25, input 150 MB, α ladder ×2), generated on datalab via the
`route_b_8task_probe.sbatch` pattern.

**8-task contingency rung outcome + SCREEN VERDICT (2026-08-26): INVALID —
FAIL-BY-EXHAUSTION.** Corpus: 24/24 datasets on datalab
(`gnn_datasets_route_c_link_screen_8task`, jobs 714729 gen + 714737 score; the first
attempt, 714278, was destroyed by a home-quota exhaustion that poisoned every shard —
Errno 122 in `placement_errors.log` even on shards that reported COMPLETED — and the
corpus was wiped and regenerated clean: 24/24 `sweep_complete`, zero error logs, zero
skips). Manipulation check: link-wait share of rtt **median 0.0129** (min 0.0054, max
0.0198) vs the 0.10 bar (`route_c_screen_8task_manipulation.json`) — doubling
concurrency moved the median only from 0.0104 (4-task R3). The bandwidth-free ceiling
wait/(wait+transfer) (at 8 tasks: Σ link_wait vs Σ 8·link_transfer_avg,
`route_c_screen_8task_ceiling.json`) is **median 0.0704, max 0.0999**: concurrency did
raise it (4-task: median 4–6%, max 8.8%) but **no dataset reaches 0.10 even if link cost
consumed ALL of rtt**, and concurrency was the ladder's last registered lever. Diagnostic
(no verdict — rung invalid): unconstrained R_exact > 1% on 4/24 datasets (max 142%,
`route_c_screen_8task_rtt.json`) is NOT attributable to the fabric — the 8-task Arm S
control (bw=1000, input 150 KB) already shows 14/204 = 6.9% > 1% (max 46.4%,
`route_b_8task_rtt.json` unconstrained row), and 4/24 against that base rate is within
binomial noise (P(≥4) ≈ 0.085). Note this corrects the registration's anchor sentence,
which cited the **4-task** Arm S 0.000: at 8 tasks a non-fabric co-batch coupling channel
already fires occasionally without any link lever. **Verdict, per the registered gate:
link contention cannot be made a material share of RTT in this simulator family at
enumerable-sweep concurrency — the corpus (not the architecture) is the limit. The
`route_c_link_transfer_v1` name is NOT granted. Proceed to Branch A** (stage-2
re-registration on the current corpus; recipe in
`handover-route-b-stage2-or-env-pivot.md` §2).
