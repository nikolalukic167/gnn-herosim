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
