# accel_nosplit_v1 — does the accel GNN beat CD when split peer groups are served whole?

**Status:** `REGISTERED` (2026-10-10). Every bar below was signed before any data.

**Why.** In `accel_replica_v1` the GNN beat CD on every peer group of at most 8 tasks. It lost only on 9–10-task groups,
which the declared slate splits into blind sub-groups above 100k plans (S5, 855815). Serving those groups whole
(`GNN_SLATE_NO_SPLIT=1`, rp/accel-nosplit d4ec8688) helped the GNN in 6 of 6 diagnostic cells (855821). This node is
the live gate for that serving change. It needs no retraining: the checkpoints are those `accel_replica_v1` selected.

## Design (fixed now)

- **Environment:** the `accel_replica_v1` gate's, at both rungs (moderate m 11.6139, heavy m 27.6227),
  fastest_compatible, 4 WF1 windows.
- **Topologies:** 24 fresh ids, **16345–16368**. These are S5's next free pool; disjointness is re-verified before
  minting.
- **Code:** one commit for every arm, d4ec8688 or a descendant that adds only harness changes. Identity against the
  accel gate is shown on 4 cells before the full run.
- **Arms:**
  - gnn_eng and gnn_eng_physmp, served with `GNN_SLATE_NO_SPLIT=1`, seeds 1 and 2;
  - CD, the bar, seed 0.
  - **Descriptive:** the same GNN arms with the flag off, `cd_pull` (CD + node pull hold, rp/accel-pullhold 4ace56e0
    merged in or run at its own pin with identity shown), self-predict, and CD←GNN.
- **Bar rule:** plain CD is the bar, because it holds the model's information. `cd_pull` has pull-hold information the
  GNN lacks, so it is reported and never used for the verdict.
- **Statistic:** paired % vs CD per topology (median over windows and seeds), then the median over 24, with an exact
  two-sided Wilcoxon.
- **Primary family:** {gnn_eng, physmp} no-split vs CD × {moderate, heavy}, Holm over 4.
- **WIN:** an arm at median ≤ −5 % with Holm p < .05 at **both** rungs, and no rung with CD Holm-confirmed faster.
- **Per-rung labels:** CONFIRMED (median ≤ −5 % and Holm p < .05), DIRECTION-ONLY, NOT-SEPARATED or CD-FASTER.
- **Sealed:** SEALED=1, with the reader committed before the data. Stuck cells are killed by progress rate, never
  rerun, excluded and counted.
- **Scope of any claim:** a learned-scorer result with serving-side whole-group decoding. It is not a message-passing
  claim, since no MP-off twin was trained.

## Record (newest first)

- 2026-10-10 17:10 — **Amendment, before any data: gate pin** (coordinator). The pin is rp/accel-nosplit-gate acbea783:
  d4ec8688 plus the merge of rp/accel-pullhold 4ace56e0, which adds pull-ledger and cd_pull src code, off by default and
  identity-tested on its parent (855822), plus the no-split harness kinds.
  - This replaces the design's "d4ec8688 or a descendant that adds only harness changes".
  - The 4-cell identity at acbea783 against the accel gate still gates the run.
- 2026-10-10 16:55 — **Ids verified** (S5). 16345–16368 are free. Minted ids ≥ 16300 are 16301–16312 and
  16321–16344 only. The next free pool is 16369+.
- 2026-10-10 16:35 — **Registered** (coordinator). S4 mints and runs the gate, S5 re-verifies the ids, and S6 provides
  the harness kind for the no-split serving flag.
