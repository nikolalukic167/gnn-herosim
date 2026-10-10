# accel_nosplit_v1 — does the accel GNN beat CD when split peer groups are served whole?

**Status:** `CLOSED` (2026-10-10) — **DIRECTION-ONLY** (moderate CONFIRMED, heavy DIRECTION-ONLY; no WIN). Registered
2026-10-10; every bar below was signed before its data.

**Outcome.** Serving split groups whole makes the accel GNN beat plain CD at moderate, confirmed on 24 fresh
topologies. It also turns heavy from a loss into a small lead. It does not beat the stronger hand bar.
- **vs CD, moderate:** gnn_eng −6.93 % (23/24, Holm <.0001); physmp −7.71 % (22/24, Holm <.0001). Both CONFIRMED.
- **vs CD, heavy:** gnn_eng −1.65 % (15/24, Holm .079); physmp −1.05 % (Holm .32). DIRECTION-ONLY.
- **Flag off** on the same topologies: −4.3 / −4.8 % moderate, +4.1 / +4.3 % heavy. The sub-batch cut was the heavy
  loss.
- **vs cd_expand** (CD + α-expansion; descriptive, it holds the same information): −1.2 / −1.1 % moderate (n.s.),
  +2.7 / +3.5 % heavy (cd_expand faster, 5 and 3 of 24).
- **cd_pull** (CD + pull hold; information the GNN lacks): −4.2 / −8.9 % vs CD.
- **Do not quote** a win over the strongest hand bar, or any message-passing claim (no MP-off twin).

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

- 2026-10-11 07:20 — **Classical reference arms: every one is far slower than the no-split GNN and CD** (S4; 856147
  at acbea783, reader 06f09941 committed before the data, read once as 856169; 16345–16368 × 4 windows × 2 rungs,
  seed 0; descriptive addendum requested by PUBLISHER).
  - **Counts:** reactive, batched (one-pass greedy) and locality 192/192 each. Random 187/192: 5 watchdog kills,
    counted as failed, not rerun.
  - **Paired % vs gnn_eng no-split (positive = classical slower), moderate / heavy; Holm < .0001 and 0/24 topologies
    ahead in every family:**
    - random +160.7 / +224.3 %;
    - batched +23.45 / +13.85 %;
    - locality +22.63 / +13.18 %.
    - vs physmp no-split and vs CD the pattern is the same (CD vs batched +13.7 / +11.7 %).
  - **Reactive (Knative) collapses at both rungs; recorded, not dropped:**
    - moderate: p95 median 5,703 s, run end 2.01 × the last arrival, effective queue share 0.99;
    - heavy: p95 8,142 s, end 4.60 ×, share 1.00.
    - Its paired % (+18,621 / +31,042 %) is the queue blowing up, not a placement margin.
  - Random, batched and locality are healthy (end ≤ 1.04).
  - Sensitivity without the 4 topologies that had a failure: in `/share/nikola.lukic/accel_nosplit/read2_06f09941.json`.
- 2026-10-11 07:00 — **Registered diagnostic: the GNN searching on its own score ("gnn_selfsearch"), aimed at the heavy
  gap** (coordinator, on the user's "start A"; owner S4; descriptive; it cannot change this lineage's verdict).
  - **Why:** heavy is the missing rung vs CD: gnn_eng −1.65 %, while cd_exactS reaches −9.7 % on 16301–16312.
    Offline, the GNN's decoded plans have regret 21.7 % against argmin-S's 6.1 % (413 tables, `accel_replica_v1`).
    Hypothesis: part of the GNN's heavy error is its sequential one-pass decode, not its scores. Search on the model's
    own score could remove that and keep the policy learned-only. CD gained about 10 % from search on S.
  - **Arm:** the no-split GNN (ra_gnn_eng s1, the cdxapply checkpoint). Per batch, choose the slate plan with the
    highest GNN plan score: the sum over tasks of the model's logit for the task's choice, with every other task
    committed where the plan puts it (S5's cdxexg convention).
    - Exact enumeration when the batch has ≤ 625 slate plans.
    - Above that: coordinate ascent on the same score, starting from the no-split decode. Sweep tasks in id order and
      move each to its best slate candidate given the rest. Stop at no change or 5 sweeps.
    - No S, no CD and no hand term anywhere.
  - **Cells:** 16301–16312, g0, both rungs, seed 1. The plain no-split GNN runs in the same job as the paired
    reference. CD, cd_exactS and cdxapply cells are reused from 855894 / 855885 if their provenance matches. Flag-off
    identity is shown first on one cell vs the no-split path at acbea783.
  - **Bars, signed before any data:**
    - read per rung: paired median % vs CD, vs the plain no-split GNN, vs cd_exactS and vs cdxapply; sign counts;
      exact Wilcoxon;
    - **trigger:** heavy vs CD ≤ −5 % and moderate vs CD ≤ −5 % on these 12 topologies → register a sealed
      fresh-topology confirmation (new ids, accel setup) with gnn_selfsearch vs CD as the primary, plus the same search
      on the MP-OFF twin's score;
    - otherwise it is recorded as a diagnostic.
  - Also reported: counts of exact vs ascent batches, how often search changes the decoded plan, and decision time
    (not scored).
- 2026-10-10 21:00 — **Gate read: DIRECTION-ONLY, closed** (S4; one read, job 855897, reader ee821695; output
  `accel_nosplit/read_acbea783.{json,txt}`; 2,688 of 2,688 summaries: classical 576, learned 1,920, cd_expand 192;
  0 failed, none killed, none above 5× CD cost; the sensitivity pass is identical).

  | Arm vs CD | Moderate | Heavy |
  |---|---|---|
  | gnn_eng no-split (primary) | −6.93 % (23/24, Holm <.0001) | −1.65 % (15/24, p .040, Holm .079) |
  | physmp no-split (primary) | −7.71 % (22/24, Holm <.0001) | −1.05 % (15/24, Holm .32) |
  | gnn_eng flag off | −4.31 % (17/24) | +4.10 % (5/24) |
  | physmp flag off | −4.82 % (16/24) | +4.30 % (5/24) |
  | cd_pull | −4.18 % (22/24) | −8.89 % (24/24) |
  | self-predict | −1.09 % | −0.44 % |
  | CD←GNN (flag off) | −3.67 % (23/24) | −1.70 % (21/24) |

  - **No-split GNN vs cd_expand** (paired, descriptive):
    - gnn_eng: −1.20 % (15/24, p .17) / +2.68 % (5/24, p .0008);
    - physmp: −1.13 % (p .23) / +3.49 % (3/24, p .0001).
  - **Median decision cost per task:** no-split 6.1 / 5.6 ms, cd_expand 0.57 / 0.67 ms, CD 0.14 / 0.16 ms.
- 2026-10-10 19:40 — **Amendment, before any read: cd_expand added as a descriptive arm** (coordinator, on SCIENTIST's
  advice).
  - `cd_expand` (CD + exact α-expansion moves, rp/cd-expand 8e3d39c6, which is acbea783 + expansion, off by default and
    identity-tested) beat plain CD by −8.2 / −7.3 % on the accel gate topologies. It holds the GNN's information.
  - It runs on the same 24 topologies, both rungs, seed 0, with identity shown. It is paired descriptively against the
    no-split GNN arms.
  - The primary family and its verdict are unchanged. Any headline must also quote GNN vs cd_expand.
- 2026-10-10 17:50 — **Launched, sealed** (S4) at acbea783.
  - **Identity (855826):** 6 of 6 cells bit-identical against the accel gate (cd, selfpredict and flag-off ra_gnn_eng
    on 16301, g0, both rungs).
  - **Instrument checks:** no-split unsplit = sub_batched (677/677 moderate, 900/900 heavy); cd_pull charged
    6,769 / 24,923.
  - **Reader tests:** 9 passed at 8c5085ba.
  - **Arrays:** classical 855831 (cd, selfpredict and cd_pull; 576 cells) and learned 855832 (no-split and flag-off
    gnn_eng and physmp, cdapply flag-off; seeds 1–2; 1,920 cells).
- 2026-10-10 17:30 — **Inputs and reader ready** (S4).
  - Inputs 855824: 24 cfgs per rung for 16345–16368, fastest_compatible in every cfg, windows byte-identical to the
    accel gate's.
  - Reader committed before data at rp/accel-nosplit-run 8c5085ba.
  - Identity 855826 is running at acbea783 (10 cells on 16301). Submission is pre-approved on an identity pass.
- 2026-10-10 17:10 — **Amendment, before any data: gate pin** (coordinator). The pin is rp/accel-nosplit-gate acbea783:
  d4ec8688 plus the merge of rp/accel-pullhold 4ace56e0, which adds pull-ledger and cd_pull src code, off by default and
  identity-tested on its parent (855822), plus the no-split harness kinds.
  - This replaces the design's "d4ec8688 or a descendant that adds only harness changes".
  - The 4-cell identity at acbea783 against the accel gate still gates the run.
- 2026-10-10 16:55 — **Ids verified** (S5). 16345–16368 are free. Minted ids ≥ 16300 are 16301–16312 and
  16321–16344 only. The next free pool is 16369+.
- 2026-10-10 16:35 — **Registered** (coordinator). S4 mints and runs the gate, S5 re-verifies the ids, and S6 provides
  the harness kind for the no-split serving flag.
