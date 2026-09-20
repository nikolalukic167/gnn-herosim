# corpus_matched_v1 — is the graph arm's win a model class, or a corpus?

**Superseded in part 2026-09-20 (`unsaturated_edge_v1`):** every "vs reactive" number on the client rungs was an unpaired ratio of medians over cells (two saturated) and flips sign when paired; the matched model-class contrasts (arm-vs-arm, paired) stand.

**Status:** `CLOSED` (2026-09-18) — **`MODEL-CLASS-EDGE-IS-CORPUS-CONTINGENT`.** Closed on a
live gate across the unsaturated client ladder (rule 6). Registered 2026-09-18; every bar below
was signed before its data, as a module constant in `scripts_cosim/corpus_matched_v1_read.py`
committed with its 14 tests before the reader was pointed at a results directory.

**Outcome. `best_arm_v1`'s corpus clause is discharged — and replaced by a narrower one.** With
the corpus held fixed at **1,670**, the graph arm beats the pointwise arm at 40 clients
(**−31.65 %, 16/16**) and 80 (**−17.12 %, 14/16**) and is **not behind at 20** (−0.73 %,
p = 0.88) ⇒ **G2 `MODEL-CLASS-EDGE-SURVIVES-MATCHING`**. Held fixed at **516**, the same
contrast wins only at 40 clients (−10.78 %, 12/16) and ties elsewhere ⇒ **H2
`MATCHED-EDGE-NOT-ESTABLISHED`**. The two disagree, so **G4 `MODEL-CLASS-EDGE-IS-CORPUS-
CONTINGENT`: name the corpus in every quote of either.**

**The finding that changes a reading: `best_arm_v1`'s only pointwise win was a corpus effect.**
F1 compared `gnnedge0` (1,670) with `mpoff_516` (516) and read **+15.71 %** for the pointwise
arm at 20 clients — the single loss that kept the ladder from overturning CLAUDE.md's pointwise
clause. **G3** measures the corpus term alone, same architecture: **+18.71 / +23.81 / +8.64 %**
in favour of 516 at the three rungs, `CONFOUND-IS-MATERIAL` at **all three**. At 20 clients the
corpus term is larger than the whole F1 gap, and the matched model-class term is −0.73 %.
(Medians of ratios do not add; this is a decomposition by magnitude and sign, not an identity.)

**What this does NOT do.** It does not overturn `best_arm_v1`'s F2 and is not read as if it did.
F2 was signed before its data and stands: `mpoff_516` is still the best *arm* the programme has,
and on the best-arm question the ladder is still split. G2 answers a different question — model
class at fixed corpus — and the two are complementary, not competing.

**Four clauses, none optional.**

1. **The contrast is the whole message-passing stack, not the bipartite conv.** `mpoff` has
   *all* message passing off, PeerConv included. That is exactly the contrast F1 made, which is
   what makes them comparable; `bipartite_edge_v1` D2/D3 is where the conv alone is isolated.
2. **At 20 clients both arms lose to reactive Knative catastrophically** (+85 % to +96 %, 0/16
   at every corpus). Both ties there are ties among losers on a rung where nothing the programme
   has built is deployable.
3. **At 40 clients the graph arm beats reactive at BOTH corpora** (−20.30 % at 1,670 on 16/16;
   −16.28 % at 516 on 14/16) while the pointwise arm manages +18.48 % and −7.24 %. That rung is
   where this effect lives — and it is the rung `bipartite_edge_v1` found the penalty on.
4. **`CORPUS-DOES-NOT-HELP` is sharper than B1 recorded.** 3.2× the data costs the *pointwise*
   arm 8.6–23.8 % of live elapsed at every unsaturated rung. Post-hoc and unregistered: it does
   **not** cost the graph arm the same way (+10.95 % at 20 clients, −16.27 % at 80). Recorded as
   a hypothesis needing its own registration, not folded in.

**Registered expectation, scored: WRONG on the one bar it predicted.** G2 was registered
`MATCHED-EDGE-NOT-ESTABLISHED`, on the assumption the 20-client pointwise win would survive
matching. It did not. G1 at 40/80 and G3 were predicted correctly; H2 and G4 were registered
UNCERTAIN.

**The question.** `best_arm_v1` closed `BEST-ARM-STILL-NOT-ESTABLISHED` and carried one clause
its own design could not discharge:

> It is not a GNN-vs-MLP claim, and it is **confounded with corpus**. `mpoff_516` trains on 516
> datasets and `gnnedge0` on 1,670, and B1 measured corpus as the largest lever in this
> programme (`CORPUS-DOES-NOT-HELP` 6/6 — more data made every arm *slower* live). Quote both
> arms' training caches or do not quote the comparison. **A corpus-matched successor is what
> would make this a model-class result.**

This is that successor.

**It asks a different question from `best_arm_v1`, and the difference is the point.** F1
compared *the best arm of each kind*, which is inherently cross-corpus because the better
pointwise arm is the 516 one. G1 holds the corpus fixed and varies only the model class.
**Neither read supersedes the other.** F2 remains the best-arm answer; G2 is the model-class
answer; G3 measures how far apart those two questions actually are on these cells.

**The G ladder needs no new runs.** `1670_mpoff` and `be1670_gnnedge0` are both already served
at 16+ checkpoints on all three unsaturated client rungs, from `peer_only_v1` B7 and
`best_arm_v1` F. That is why the bars are committed first and why this paragraph exists: **a
contrast that is free to compute is a contrast that is easy to compute twice and report once.**
The H ladder (the 516 graph arm) does need 16 training runs and 192 gate arms.

## The 2×2

|  | **graph arm** (`gnnedge0`) | **pointwise arm** (`mpoff`) |
|---|---|---|
| **1,670 datasets** | `be1670_gnnedge0` — served | `1670_mpoff` — served |
| **516 datasets** | `cm516_gnnedge0` — trained and served here | `516_mpoff` (`v3ext`) — served |

`best_arm_v1`'s F1 read the **off-diagonal**: `be1670_gnnedge0` vs `516_mpoff`. G1 reads the
top row, H1 the bottom row, and G3 the right-hand column.

## Registered reads

All at the three **unsaturated** client rungs `best_arm_v1` used — 20 (which **is** the R0
server cell, 6 servers / 20 clients), 40, 80. Bars reused **unchanged** from `peer_only_v1`'s
B4/B8 and `best_arm_v1`'s F, so G reads against the clause it is qualifying rather than against
a bar of its own choosing: **|median| ≥ 5 %, two-sided Wilcoxon p < 0.05, n ≥ 16 checkpoints**,
unit = checkpoint (`collapse_to_seed`, median over that checkpoint's cells), negative = the
graph arm faster.

- **G1** — `be1670_gnnedge0` vs `1670_mpoff`, per rung. **Corpus held fixed at 1,670.**
- **G2** — the composite, under **F2's rule unchanged**: two wins and no losses.
  Deliberately the same rule, so "did the verdict move once corpus was matched?" compares like
  with like. A rung that is UNREADABLE counts as **missing**, never as either side.
- **G3** — `1670_mpoff` vs `516_mpoff`, per rung. **Same architecture, corpus varies.** This is
  the confound itself, measured on the very cells it was flagged on.
- **H1/H2** — `cm516_gnnedge0` vs `516_mpoff`. **Corpus held fixed at 516.**
- **G4** — the 2×2: does the matched verdict agree at both corpora? **Descriptive on two
  corpora** — two points cannot establish corpus-independence in general, only that it did not
  move here.

## Consequences, signed in advance

1. **G2 = `MODEL-CLASS-EDGE-SURVIVES-MATCHING`** ⇒ `best_arm_v1`'s corpus clause is
   **discharged**: the graph arm's per-rung wins are a model-class effect and not a corpus
   artifact. CLAUDE.md's standing answer gains that, and its pointwise clause becomes a
   statement about **which arms exist**, not about model classes. The pointwise clause itself
   **still stands** — F2 is not re-read here.
2. **G2 = `POINTWISE-BETTER-AT-MATCHED-CORPUS`** ⇒ the pointwise clause **strengthens** from an
   arm comparison into a model-class one, and the graph arm's F1 wins at 40/80 clients are
   revealed as corpus effects. This is the outcome that would most change the programme.
3. **G2 = `MATCHED-EDGE-NOT-ESTABLISHED`** ⇒ nothing moves. The clause stays as written and the
   record gains the measurement that it could not be discharged, which is worth having.
4. **G3 = `CONFOUND-IS-MATERIAL`** (corpus separates the pointwise arm at ≥ 1 rung) ⇒
   `best_arm_v1`'s F1 numbers mix a model-class term with a corpus term of measurable size;
   **F2 may not be quoted without G2 beside it**, and the node says so.
5. **G3 = `CONFOUND-NOT-MATERIAL-HERE`** ⇒ the confound was correctly flagged and is not
   material on **these cells**; F2's ladder may be quoted on its own. That is a finding about
   this comparison only and **never a general licence to mix corpora**.
6. **G4 = `MODEL-CLASS-EDGE-IS-CORPUS-CONTINGENT`** ⇒ record it as a scope limit on whichever
   of G2/H2 is being quoted, and name the corpus in every future quote of either.

## Registered expectations

Scored after the read; being wrong is recorded, not rewritten.

- **G1 at 40 and 80 clients: `GRAPH-FASTER-AT-MATCHED-CORPUS`.** The graph arm already beat the
  *better* pointwise arm there (−14.76 % on 16/16; −11.32 %), and B1 says the 1,670 pointwise
  arm should be the *worse* one, so matching should widen the margin, not close it.
- **G1 at 20 clients: UNCERTAIN, leaning `POINTWISE-FASTER-AT-MATCHED-CORPUS`.** +15.71 % is a
  large gap to close, but it is also the rung where **both arms lose to reactive** and the
  ranking is among losers.
- **G2: `MATCHED-EDGE-NOT-ESTABLISHED`** — the same split ladder, one rung down.
- **G3: `CORPUS-516-FASTER` at ≥ 1 rung** (`CONFOUND-IS-MATERIAL`), because B1 read
  `CORPUS-DOES-NOT-HELP` 6/6 and more data made every arm slower live.
- **H2 and G4: UNCERTAIN.** Nothing in the record predicts how the repaired conv behaves on a
  third of the data.

## Method

- Training: `experiments/corpus_matched_v1_516_gnnedge0.yaml`, byte-identical to
  `experiments/bipartite_edge_v1_1670_gnnedge0.yaml` except the cache, the split artifact, the
  lineage and the wandb name — i.e. except the corpus. Run by
  `scripts_cosim/datalab/corpus_matched_v1_train.sbatch` (16 seeds), which **asserts
  `num_datasets == 516` and the T1b split's 386/96/34** before training a single epoch: *a
  corpus is its SPLIT, not its directory*, and rebuilding "516" from `--base-dirs` yields 1,670,
  which would turn the one term being controlled for into the contrast itself.
- Serving: appended blocks in the two existing gate scripts (server tasks 1300–1363, client
  tasks 1040–1167) so the H arm shares cells, workload, physics and summary schema with the
  three arms it is compared against. Both need `PO_TASK_OFFSET` (`MaxArraySize = 1001`).
- **New guard, both gates:** the split artifact's sha256 is now pinned **per corpus tag**. The
  corpus is the variable under test and it is invisible in the weights, so serving an arm from
  the wrong corpus would destroy the contrast in exactly the direction that flatters it.
  Verified against the served checkpoints: `1670`/`be1670`/`ba1670` carry `f9c79fd7bb77`,
  `v3`/`v3ext`/`516`/`cm516` carry `0d66d1c09276` — every arm already on disk passes it.
- Reading: `scripts_cosim/corpus_matched_v1_gate_read.py`, which reuses `peer_only_v1`'s
  loaders and `best_arm_v1`'s verified `_dedupe_r0_mpoff` (which **asserts** the duplicate R0
  `mpoff_516` runs are bit-identical before dropping any) rather than re-deriving either.

## Record

*(newest first; appended as the reads land)*

### 2026-09-18 — H1/H2/G4 read: the matched edge does NOT reproduce at 516, so it is corpus-contingent

Live gate, same three rungs, same bars. 16 `cm516_gnnedge0` checkpoints trained (job 788005,
16/16 COMPLETED) and 192 gate arms served (64 at R0, 128 on the client axis, all COMPLETED).

**H1 — `cm516_gnnedge0` vs `516_mpoff`, both trained on 516 datasets.**

| rung | H1 | median | p | ahead | vs reactive: graph / pointwise |
|---|---|---|---|---|---|
| 20 clients | `NOT-SEPARATED` | +3.00 % | 0.4691 | 7/16 | +84.97 % / +69.34 %, both 0/16 |
| 40 clients | **`GRAPH-FASTER-AT-MATCHED-CORPUS`** | **−10.78 %** | 0.0229 | 12/16 | **−16.28 %** (14/16) / −7.24 % |
| 80 clients | `NOT-SEPARATED` | −2.56 % | 0.4380 | 9/16 | −3.00 % / −4.84 % |

**H2 `MATCHED-EDGE-NOT-ESTABLISHED`** — one win, no losses. The ladder is split, not lost: the
40-client win is real under a registered bar and the graph arm is behind at no rung.

**G4 `MODEL-CLASS-EDGE-IS-CORPUS-CONTINGENT`.** G2 read `MODEL-CLASS-EDGE-SURVIVES-MATCHING` at
1,670 and H2 reads `MATCHED-EDGE-NOT-ESTABLISHED` at 516. **Signed consequence 6 fires: this is
a scope limit, and every future quote of either must name the corpus.** Descriptive on two
corpora — it moved, which is not a model of *how* it moves.

**The 2×2, complete** (median %, negative = the row-arm faster; unsaturated client rungs):

| | 20 clients | 40 clients | 80 clients |
|---|---|---|---|
| **graph vs pointwise @ 1,670** (G1) | −0.73 (tie) | **−31.65** | **−17.12** |
| **graph vs pointwise @ 516** (H1) | +3.00 (tie) | **−10.78** | −2.56 (tie) |
| **pointwise: 1,670 vs 516** (G3) | **+18.71** | **+23.81** | **+8.64** |

**Post-hoc, NOT a registered bar: the fourth edge.** G3 registered the pointwise arm across
corpora; its mirror — the *graph* arm across corpora — was not signed and is computed here only
because H2 came in weaker than G2 and the question is unavoidable. `gnnedge0_1670` vs
`gnnedge0_516`: **+10.95 % (3/16, p = 0.0151) at 20 clients, −3.74 % (tie) at 40, −16.27 %
(12/16, p = 0.0097) at 80.** So the corpus that costs the pointwise arm 8.6–23.8 % at *every*
rung **helps** the graph arm at the top rung and hurts it at the bottom one. That is a
coherent-sounding story — more data pays only where there is more load to reason about — and it
is exactly the kind of story that has failed five registered tests in this programme already.
**It is recorded as a hypothesis needing its own registration, not folded in.**

**Verification.** `cm516_gnnedge0` and `be1670_gnnedge0` are the same architecture and the same
flags, differing only in training corpus, and both are weight-invisible in the ways that matter
— so before believing any of the above: **0 of 192 (cell, seed) pairs read identically across
the two corpora.** The new per-corpus split-sha pin fired on every one of the 192 arms
(`corpus=cm516 ... ck=models/corpus-matched-v1-516-gnnedge0-lr2e3-seed1.pt` in the task logs).

**Registered expectations, scored.** H2 and G4 were both registered UNCERTAIN, so neither is
right or wrong. The one scored prediction of this lineage, G2, was **wrong** (recorded above).

**Two clauses.**

1. **The 40-client rung is the only one where the graph arm wins at both corpora.** It is also
   the rung where `bipartite_edge_v1` D1 found the bipartite penalty and where the repaired arm
   beats reactive by −20.3 %. Whatever this effect is, that rung is where it lives.
2. **At 516 the graph arm beats reactive at 40 clients (−16.28 %, 14/16) while its matched
   pointwise twin manages −7.24 %** — so even where the *pairwise* contrast is not established
   as a ladder, the graph arm is the better of the two against the baseline at that rung.

### 2026-09-18 — G1/G2/G3 read: `MODEL-CLASS-EDGE-SURVIVES-MATCHING`, and the confound was material at every rung

Live gate, the three unsaturated client rungs, every arm already on disk; read with
`scripts_cosim/corpus_matched_v1_gate_read.py`, bars from the module committed beforehand.

**G1 — `be1670_gnnedge0` vs `1670_mpoff`, both trained on 1,670 datasets.**

| rung | G1 | median | p | ahead | reactive queue |
|---|---|---|---|---|---|
| 20 clients (= R0 cells) | `NOT-SEPARATED` | −0.73 % | 0.8767 | 8/16 | 66 %, unsaturated |
| 40 clients | **`GRAPH-FASTER-AT-MATCHED-CORPUS`** | **−31.65 %** | 0.0004 | **16/16** | 77 %, unsaturated |
| 80 clients | **`GRAPH-FASTER-AT-MATCHED-CORPUS`** | **−17.12 %** | 0.0052 | 14/16 | 80 %, unsaturated |

**G2 `MODEL-CLASS-EDGE-SURVIVES-MATCHING`** — two wins, no losses, under F2's rule unchanged.
**Signed consequence 1 fires: `best_arm_v1`'s corpus clause is DISCHARGED.**

**G3 — `1670_mpoff` vs `516_mpoff`, same architecture, corpus varies.**

| rung | G3 | median | p | ahead |
|---|---|---|---|---|
| 20 clients | `CORPUS-516-FASTER` | +18.71 % | 0.0151 | 3/16 |
| 40 clients | `CORPUS-516-FASTER` | +23.81 % | 0.0027 | 2/16 |
| 80 clients | `CORPUS-516-FASTER` | +8.64 % | 0.0084 | 4/16 |

**G3 `CONFOUND-IS-MATERIAL` at all three rungs.** Signed consequence 4 fires. B1's
`CORPUS-DOES-NOT-HELP` does not merely reproduce here — **3.2× the data costs the pointwise arm
8.6–23.8 % of live elapsed, at every unsaturated rung, on 2–4 checkpoints out of 16.**

**What the two together say about `best_arm_v1`'s ladder.** F1 read `gnnedge0_1670` vs
`mpoff_516` — the off-diagonal — and its **only** pointwise win was **+15.71 % at 20 clients**.
At that same rung the corpus term alone is **+18.71 %** and the matched model-class term is
**−0.73 %, 8/16, p = 0.88**. **The one rung that kept `best_arm_v1` from overturning CLAUDE.md's
pointwise clause was a corpus effect.** (Medians of ratios do not add, so this is a decomposition
by magnitude and sign, not an identity.) The two graph wins survive matching and get *larger*:
−14.76 % → **−31.65 %** at 40 clients, −11.32 % → **−17.12 %** at 80, because the pointwise arm
it is now compared against is the worse of the two pointwise arms.

**Registered expectations, scored: RIGHT on four of five.** `GRAPH-FASTER` at 40 and 80 —
right. `CORPUS-516-FASTER` at ≥ 1 rung — right, and at all three. 20 clients was registered
UNCERTAIN leaning pointwise and came in **NOT-SEPARATED**, which is inside the registered
uncertainty. **G2 was registered `MATCHED-EDGE-NOT-ESTABLISHED` and is WRONG** — recorded as
wrong. The prediction assumed the 20-client rung would stay a pointwise win; matching removed it.

**Five clauses, none optional.**

1. **This does NOT overturn the pointwise clause, and it is not read as if it did.**
   `best_arm_v1`'s F2 was signed before its data and stands as read: `mpoff_516` is still the
   best pointwise arm the programme has, and on the best-arm question the ladder is still split.
   What G2 establishes is narrower and different — **that the graph arm's advantage is a
   property of the model class and not of its training cache.**
2. **The contrast is the whole message-passing stack, not the bipartite conv alone.** `mpoff`
   has *all* message passing off, PeerConv included. G1 is exactly the contrast F1 made, which
   is what makes them comparable, and it is not a claim about the bipartite stage by itself —
   `bipartite_edge_v1` D2/D3 is where that lives.
3. **At 20 clients both arms still lose to reactive Knative catastrophically** (+94.99 % and
   +95.72 %, 0/16). The `NOT-SEPARATED` there is a tie among losers on a rung where nothing the
   programme has built is deployable.
4. **At 40 and 80 clients the graph arm beats reactive and the matched pointwise arm does not**
   (−20.30 % on 16/16 and −13.48 % on 14/16, against +18.48 % and +0.93 %). On the matched
   corpus the graph arm is the only arm that beats the baseline at an unsaturated rung.
5. **The R0 duplicate count now reads 0, and that is correct.** `best_arm_v1` verified 16
   overlapping `mpoff_516` runs bit-identical and then archived the redundant copies with a
   README, so there is no longer an overlap to verify. The reader prints what it actually did.

**H is in flight.** 16 `gnnedge0` checkpoints on the 516 corpus (job 788005), then 192 gate
arms, to complete the 2×2 and read H2/G4. The lineage stays `ACTIVE` until then: G2 answers the
1,670 row, and a 2×2 is not read from one row.
