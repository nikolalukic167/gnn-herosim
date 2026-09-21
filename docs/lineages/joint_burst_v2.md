# joint_burst_v2 — train on the served decision UNCAPPED: keep the loaded states the label-cap threw away

**Status:** `CLOSED` `GNN-BEATS-GREEDY / CD-STILL-AHEAD` (gate read 2026-09-20). The uncapped
served-distribution `gnnedge0` **beats the 1-pass greedy in its own seat (K1 −11.9 %, 13/13,
p = 0.0015)** — the exact contest v1 lost at +16.8 % — and beats reactive (K3 −34.7 %) and random
(~−48 %). It **loses to the coordinate-descent greedy (K2 +12.5 %, 0/13)**, the honest bar. The v1
loss was the **serving cap, not the model or the corpus**: uncapping the v1 weights alone already
beats the greedy (K6 −8.8 %, 16/16) and the loaded-state corpus adds nothing beyond it
(K5 CORPUS-NEUTRAL −3.9 %). Ties its pointwise twin (K4 −4.4 %, under the bar). Next lever to
reach CD: [`rollout_imitation_v1`](../../LINEAGES.md) (train the multi-pass optimum, registered).
Parent
[`joint_burst_v1`](joint_burst_v1.md) closed `NO-GNN-WIN`: trained on the served burst
distribution, `gnnedge0` beat reactive Knative (−12.5 %) and its cold twin (−7.8 %) but lost to
the batched greedy in its own seat (+16.8 %). This lineage tests **why**, and whether closing the
gap it names flips the sign.

**Parents:** [`joint_burst_v1`](joint_burst_v1.md) (the served-distribution corpus and gate),
[`peer_greedy_live_v1`](peer_greedy_live_v1.md) (the greedies that are the bars),
[`peer_affinity_warm_v1`](peer_affinity_warm_v1.md) (snapshot → dataset machinery).

## The question and what step 0 measured

`joint_burst_v1` trained the model under a **node cap** (`dag_alpha_key = 2.5`:
`cap = 2.5 × max single-candidate demand on a node`), and its corpus **rejected every snapshot
whose live slate admitted no plan under the tighter alpha = 2.0 cap** — 579 of 634 rejected
snapshots, i.e. the *loaded* states the served stream actually produces. So the served model met
loaded states it never trained on, and its decoder was forbidden the very co-location move the
label rewards. Step 0 measured both, on the 48 `joint_burst_v1` burst held-out groups, replaying
hand policies and checkpoints through the real engine (the path `peer_affinity_live_serve_check.py`
proved bit-identical to the sweep). Regret = group-summed elapsed over the sweep optimum (mean
46.0 s; random 48.9 s):

| policy | offline regret mean | median | frac of random | opt-hit |
|---|---|---|---|---|
| **coordinate-descent greedy** (2–3 pass) | **11.2 s** | 4.1 | **0.23** | 21 % |
| 1-pass batched greedy | 33.5 s | 19.8 | 0.69 | 10 % |
| immediate rule | 32.1 s | 20.2 | 0.66 | 12 % |
| burst-trained `gnnedge0` (v1, capped) | 27.5 s | 20.1 | 0.56 | 6 % |
| drain-only rule / reactive Knative | 41.8 s | 32–40 | 0.85 | 0–4 % |

**Two findings that shaped this lineage.**
1. **The cap was binding out the label's move.** Single cell (cc40s9101 w0), elapsed s/task,
   live: the SAME v1 `gnnedge0` seed-1 weights read **7.53 s capped vs 5.57 s uncapped** — and
   uncapped it *beats* the 1-pass greedy (6.41 s) on that cell. n = 1, unpaired, a direction only;
   the gate's `v1-served-uncapped` arm is the paired control.
2. **The room above the 1-pass greedy is joint structure a search captures, not yet the model's.**
   Coordinate descent on the *same score* (revisit each task with every partner's node known,
   2–3 passes) cuts regret 33.5 → 11.2 s. So the 1-pass greedy is myopic (id-order anchor, never
   revisits), CD is far better, and **CD is the honest bar; K1 vs the 1-pass greedy is the
   reachable one.** The x2-exchange probe read *faster* than x1 live on that cell (6.25 < 6.41),
   so more co-location than the rule is live-valid, not a group-local artifact.

## Design

- **Corpus (built).** `make_warm_corpus.py --no-cap-filter` on the same `joint_burst_v1`
  captures (batched rule behaviour policy, 20 finishing runs): every aligned snapshot offered,
  a snapshot needs only 20 % of its tasks to have a choice, the cap-infeasible loaded states are
  kept. **2,038 train + 480 held-out datasets** (v1 had 249 + 48). Label = the sweep optimum
  (enumerable at these slates; strictly dominates CD). Held-out = the 9222–9224 set-aside cells.
- **Cache and training.** The `joint_burst_v1` recipe with `DAG_PRIMARY_ALPHA_KEY=inf` (the
  UNCAPPED rung is primary; the tighter rungs carry an empty label set where no plan is feasible,
  recorded not raised). `experiments/joint_burst_v2_{gnnedge0,mpoff}.yaml` = the v1 configs with
  the cache/split swapped and `NEAR_RTT_DAG_ALPHA_KEY=inf`, 16 seeds each. Served uncapped
  (`GNN_PREFIX_ALPHA_KEY=inf`) — the greedy's own seat.
- **Gate (rule 6).** The SAME 16 study environments as `joint_burst_v1`, served identically
  under bursts. Arms: reactive, random, immediate rule, 1-pass batched greedy, **CD greedy**,
  jb2 `gnnedge0` × 16, jb2 `mpoff` × 16, **v1 `gnnedge0` served uncapped × 16** (isolates corpus
  from cap), the x2 probe (16 env). 848 arms.

## Bars (signed 2026-09-20, before the gate) — paired, n = 16, |median| ≥ 5 %, p < 0.05, signed-rank

| read | what | fires as |
|---|---|---|
| **K1** | jb2 `gnnedge0` vs 1-pass batched greedy, GRAPH FIRST (ckpt) | `GRAPH-BEATS-GREEDY` / `GREEDY-FASTER` / `NOT-SEPARATED` |
| **K2** | jb2 `gnnedge0` vs **CD greedy** (the honest bar, ckpt) | `GRAPH-BEATS-CD` / `CD-FASTER` / `NOT-SEPARATED` |
| K3 | jb2 `gnnedge0` vs reactive (ckpt) | `ARM-BEATS-REACTIVE` / … |
| K4 | jb2 `gnnedge0` vs jb2 `mpoff`, GRAPH FIRST (seed-paired) | `GRAPH-FASTER-THAN-TWIN` / … |
| K5 | jb2 vs v1-served-uncapped `gnnedge0` (env+seed) | `CORPUS-HELPS-BEYOND-UNCAPPING` / … |
| K6 | v1-served-uncapped vs 1-pass greedy (ckpt) | `UNCAPPING-ALONE-BEATS-GREEDY` / … |
| K7 | CD greedy vs 1-pass greedy (env) | `CD-BEATS-1PASS` (the joint-structure read) / … |
| K8 | x2 probe vs 1-pass greedy (env) | `MORE-COLOCATION-HELPS` / `-HURTS` (label validity) |
| **K9** | composite | `GNN-BEATS-GREEDY` iff K1 fires; `GNN-BEATS-CD` iff K2 fires; else `NO-GNN-WIN` |

**Registered expectation (signed 2026-09-20, before the gate).** K3 95 %; **K1 fires 45 %**,
NOT-SEP 35 %, greedy-faster 20 %; K6 fires 35 %; **K2 fires 5–10 %** (CD is a very strong bar);
K4 NOT-SEP 65 %; K7 CD-BEATS-1PASS fires (offline 33.5 → 11.2 s); K8 MORE-COLOCATION-HELPS ~55 %.
**Read:** the reachable headline is K1 (a learned decoder beats the one-pass physics greedy in its
own seat); K2 is expected to fail (CD beats both — the joint structure is in the search, not yet
the representation). Watch the starved-client / postponed counters: uncapped serving may stack
onto memory-full nodes and spin; a spun cell is a disclosed loss, never dropped.

**Consequences, signed in advance.**
- K1 fires (K2 does not): the paper claim is "matched seat + served-distribution corpus keeping
  the loaded states: a learned decoder beats the one-pass physics greedy; a 3-pass coordinate
  descent on the same score beats both" — a result about *where* the joint structure lives.
- K1 and K2 both fire: the first learned win over a rule with its own information at any depth —
  the program's target. Register the next rung (bigger topology) as confirmation.
- K1 does not fire but K6 does: the corpus was not the lever, the serving cap was; the deployable
  result is "serve the existing checkpoint uncapped".
- Nothing fires above reactive: `NO-GNN-WIN` stands and the 12-server rung (un-enumerable label,
  CD-imitation) is the remaining venue.

## Record (newest first)

- 2026-09-20 (night) — **Gate read (848 arms; 13/16 jb2 gnnedge0, 14/16 jb2 mpoff, 16/16
  v1-uncapped — the 3 gnnedge0 seeds safe-stopped at the trainer's end-of-run write are absent,
  reads disclosed on the complete checkpoints).** Per-task elapsed medians (s), best first:
  CD greedy **6.14**, v1-uncapped gnnedge0 6.97, **jb2 gnnedge0 7.05**, jb2 mpoff 7.45, x2 7.79,
  1-pass greedy 8.04, immediate rule 8.09, reactive 11.59, random 13.61.

  | read | verdict | median | p | ahead |
  |---|---|---|---|---|
  | K1 jb2 gnnedge0 vs 1-pass greedy | `GRAPH-BEATS-GREEDY` | −11.90 % | 0.0015 | 13/13 |
  | K2 jb2 gnnedge0 vs CD greedy | `CD-FASTER-THAN-GRAPH` | +12.48 % | 0.0015 | 0/13 |
  | K3 jb2 gnnedge0 vs reactive | `ARM-BEATS-REACTIVE` | −34.67 % | 0.0015 | 13/13 |
  | K4 jb2 gnnedge0 vs mpoff twin | `NOT-SEPARATED` | −4.44 % | 0.0033 | 11/11 |
  | K5 jb2 vs v1-uncapped | `CORPUS-NEUTRAL-BEYOND-UNCAPPING` | −3.88 % | 0.0015 | 13/13 |
  | K6 v1-uncapped vs 1-pass greedy | `UNCAPPING-ALONE-BEATS-GREEDY` | −8.81 % | 0.0004 | 16/16 |
  | K7 CD vs 1-pass greedy [env] | `CD-BEATS-1PASS` | −22.84 % | 0.0004 | 16/16 |
  | K8 x2 vs 1-pass greedy [env] | `MORE-COLOCATION-NEUTRAL` | −2.04 % | 0.0027 | 12/16 |

  **Outcome vs the pre-signed consequences: the "K1 fires, K2 does not" branch, sharpened by K6+K5.**
  The paper claim stands as signed — *matched seat + served-distribution corpus keeping the loaded
  states: a learned decoder beats the one-pass physics greedy in its own seat; a 3-pass coordinate
  descent on the same score beats both.* K6 firing and K5 reading NEUTRAL localise the v1 deficit to
  the **serving cap**, not the corpus: serving the existing v1 checkpoint uncapped already clears the
  1-pass greedy, and rebuilding the corpus to keep the loaded states buys nothing beyond that. K4
  under the bar keeps the model-class question a tie (this is a win over the baselines, not over the
  MLP). K9 composite printed `UNREADABLE` — a bookkeeping artifact of the disclosed-fallback path
  (it reads the pre-fallback primary verdict); the substance is `GNN-BEATS-GREEDY`, not `-CD`.
  CD remains the ceiling → [`rollout_imitation_v1`](../../LINEAGES.md) is the pre-signed next lever.
  Gate driven by a local throttle loop (`MaxSubmitJobs = 50`; the login-node reaper kills a detached
  submitter, so the refill loop must run off-cluster); read by
  `scripts_cosim/joint_burst_v2_gate_read.py` → `simulation_data/joint_burst_v2/read.json`.
- 2026-09-20 — Registered; step-0 diagnostics measured (above). Corpus built (2,038 + 480);
  cache and training launched. Apparatus: `make_warm_corpus.py --no-cap-filter`,
  `src/policy/peer_greedy_network/scheduler.py::PeerGreedyNetworkCDScheduler`, the
  `HEROSIM_PG_EXCHANGE_SCALE` probe knob, `scripts_cosim/datalab/joint_burst_v2_{corpus,cache,train,gate}.sbatch`,
  `scripts_cosim/joint_burst_v1_offline_rule_regret.py`.
