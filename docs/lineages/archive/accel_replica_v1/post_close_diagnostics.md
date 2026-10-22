# accel_replica_v1 — post-close diagnostic chain (2026-10-10 15:15 → 2026-10-11 22:00)

> Attachment of [`accel_replica_v1`](../accel_replica_v1.md), moved verbatim from the node on 2026-10-12 (record consolidation). Newest first, as in the node.

- 2026-10-11 22:00 — **GNN-ranked pruning for over-cap batches finds better S plans than cost-ranked pruning, but not
  better live latency** (S5; cdxprune: over-cap batches shrink each task's slate to its top-k, ranked by (a) the GNN
  logit or (b) standalone cost, then enumerate exactly on S; cdxapply shell; rp/cd-exacts e1cbffa8; reruns
  856406/856407, 48/48 cells; 16301–16312, g0; descriptive).
  - **Disclosed:**
    - a first attempt was rejected wholesale by S5's own harness check (cancelled, nothing read);
    - an early auto-read on 23 of 24 arm-(a) cells was discarded before the full read.
  - **Counts:**
    - over-cap batches are 5.2 / 7.1 % of multi-task batches; k = 3 in about 80 %, k = 4 in about 20 %;
    - the GNN and cost top-k sets differ in 92–94 %;
    - S gain vs the expansion fallback per batch: (a) +1.81 / +1.00 s, (b) −0.76 / −0.45 s, i.e. cost pruning is worse
      than expansion on S;
    - better / worse than expansion: (a) 3,530/1,357 and 3,952/2,018; (b) 2,291/2,876 and 2,824/3,647.
  - **Live paired %, moderate / heavy:**
    - (a) vs (b): −0.22 / −0.10 %, a tie;
    - (a) vs cd_exactS: −0.09 / +0.45 %; vs cdxapply −0.44 / −1.30 %; vs cd_expand −2.39 / −2.15 %; vs CD −9.66 /
      −11.05 %.
  - **Read:** on the surrogate, the GNN ranking knows something standalone cost does not. On live latency it does not
    matter, because over-cap batches are only 5–7 % of decisions.
- 2026-10-11 05:30 — **A GNN tie-break and fallback seed inside cd_exactS add nothing measurable** (S5; cdxexg =
  cd_exactS with S-ties broken by the ra_gnn_eng s1 plan score, and over-cap fallbacks seeded from the no-split GNN
  plan; d2a980e1, flag-off identity with 72df02a0 shown; 856109, 24/24 cells; 16301–16312, g0; descriptive).
  - **Counts at moderate / heavy:**
    - ties in 20.5 / 31.0 % of exact batches; the tie-break changed the plan in 4.1 / 4.7 % of all exact batches;
    - fallback batches 5.1 / 7.2 %; the GNN seed changed 62 / 64 % of those.
  - **Paired %:**
    - vs cd_exactS: −1.04 % (7/12, p .34) / +0.40 % (5/12, p .34), a tie;
    - vs cdxapply: −1.02 % (9/12, .064) / −0.56 % (7/12, .79);
    - vs cd_expand: −2.99 / −1.87 %;
    - vs CD: −11.22 / −8.99 %.
  - Decision time per task, median 8.8 / 10.1 ms, about 6× cd_exactS.
  - **Read:** with the slate fixed, the GNN has little left to act on inside exact search. The GNN-plus-search stacks
    (cdxapply, cdxexg) and cd_exactS all sit at about −9 to −11 % vs CD.
- 2026-10-11 00:30 — **The top-5 slate rarely binds cd_exactS** (S5; 855941, log-only flag 804bd855, decisions
  identical to 855894; 16307 g0 seed 0, 2 cells; direction only).
  - **Kept-pass is mostly "already optimal":** 59.4 / 63.3 % of exact batches keep CD's pass plan, but in 57.3 / 59.1 %
    the pass equals the slate optimum. Only 2.1 / 4.2 % keep it because it is strictly cheaper.
  - **In those strict batches:**
    - the out-of-slate tasks mostly sit at standalone rank 6–7 (about 75 % at moderate, 46 % at heavy);
    - the S gap has median 1.0 s, p90 29 / 12 s;
    - in total that is 4.9 / 7.8 % of the S gained by moved batches.
  - cd_exactS already serves the pass plan in those batches. What remains is only a joint slate-plus-pass search.
  - **Ruling (coordinator):** slate widening is not pursued now. It is a small, tail-only lever on the surrogate,
    unmeasured in elapsed time. The cost_to_go_v1 plan set stays as registered: cd_exactS's own decision, which is the
    pass plan whenever the pass is kept, and the next 4 by S.
- 2026-10-10 23:00 — **Audit: the frozen autoscaler in fidelity replays leaves every reported offline number
  intact** (S6; fix 8e19fcab on rp/cost-to-go).
  - **The bug:** under non-determined schedulers, the replayed KPA's first tick waited on `_kpa_start_gate`, which only
    DeterminedScheduler released. The autoscaler therefore stayed frozen after the decision.
  - **What is shielded:** everything computed at the decision (chosen plans, S, argmin-S, ties) and every reported
    number scored from sweep-row labels (`label_of_plan`). This covers the 413-table reads above (argmin-S 79.4 %,
    cd_expand 64.6 %, oracle 5.67 %). The GNN numbers are S7's dataset decode, not a replay.
  - **Exposed:** only the diagnostic "CD replay differs from the label path by a few ms", which was never reported.
  - **Unaffected:** corpus labels, I11, label drift F→G (all on the determined arm), and live gates (no fidelity seed,
    gate None).
- 2026-10-10 22:45 — **Exact search on CD's surrogate is the strongest hand rule, level with the GNN-seeded
  search** (S5; cd_exactS = exhaustive enumeration over the top-5 slate on S, with a 100k cap and expansion fallback;
  rp/cd-exacts 72df02a0; 855894, 24/24 cells; seed 0; 16301–16312; g0; descriptive).
  - **Paired % at moderate / heavy (median, faster count, Wilcoxon p):**
    - vs CD: −10.34 % (11/12, .001) / −9.66 % (12/12, <.001);
    - vs cd_expand: −1.54 % (12/12, <.001) / −2.70 % (10/12, .034);
    - vs cdxapply: −0.10 % (6/12, .27) / −1.66 % (7/12, .13), not separated.
  - **Search:** 94.8 / 92.9 % of multi-task batches enumerated exactly, the rest fell back to expansion at the cap. Ties
    go to the lowest plan in CD's order (22 / 33 %).
  - **CD's pass plan was kept in 61 / 64 % of exact batches** (`pg_exact_kept_pass`: slate optimum ≥ pass S − 1e-9).
    This includes passes that sit inside the slate and equal its optimum, so it does **not** show that the slate binds
    (corrected 2026-10-10 23:30, S5 code read). An instrumented 2-cell check (855941, 16307) is measuring how many of
    the pass plan's tasks fall outside the slate.
  - **Decision time per task:** median 1.4 / 1.9 ms, mean 48 / 55 ms (CD 0.14 ms, GNN about 7 ms). Not scored (I10).
- 2026-10-10 22:30 — **Stakes-weighted training does not transfer live; physmp g3 stays the candidate** (S7; stakes
  physmp g3, 855813, 100 epochs, 3 seeds; live 855931, 48 cells; validation topologies 16251–16258, g0; selection
  evidence, descriptive only).
  - **Offline (best checkpoint, 438 validation datasets):** regret −3.2 % vs physmp g3, heavy −12.6 %, moderate +18 %
    (worse on all 3 seeds). The weights use the disclosed proxy, capped-sidecar mean RTT minus the optimum.
  - **Live, median paired % over 8 topologies:**
    - moderate: vs physmp g3 −0.52 % (7/8); vs CD −6.58 % (7/8);
    - heavy: vs physmp g3 −0.28 % (4/8); vs CD +1.78 % (3/8).
  - **Rule signed before the read:** swap only if heavy is ≥ 3 % ahead and moderate is no more than 2 % behind. Heavy
    is not met, so the swap is NOT made.
  - Offline regret predicted neither rung's live direction; it ranks epochs, not arms.
  - **Reuse:** 16 CD and 48 physmp g3 cells were taken from the livefb run. All 112 summaries carry the same
    run_provenance.code (9e5ee729, clean). Env flags are identical apart from GNN_MODEL_PATH.
- 2026-10-10 20:40 — **The GNN seed still helps the strongest search, slightly** (S5; cdxapply = the no-split GNN plan,
  seed 1, then CD refine with expansion; rp/cdx-apply f1dfc355; 855885, after the failed launch 855862 with 0 of 24
  usable; g0; 12 gate topologies; descriptive).
  - **Paired % at moderate / heavy:**
    - vs cd_expand: −1.84 % (10/12, Wilcoxon .042) / −1.44 % (9/12, .129);
    - vs plain cdapply: −3.82 / −4.87 %;
    - vs CD: −9.73 / −9.52 %;
    - vs the plain (split) GNN: −3.90 / −8.41 %.
  - Expansion does less work from the GNN seed (0.64 / 0.54 accepted moves per batch against 0.77 / 0.68). The seed
    starts nearer the search's optimum.
  - **Offline context** (S6, read-only, 413 held-out tables):
    - Exact argmin of CD's surrogate S over the slate hits the label optimum on 79.4 % (regret 6.1 % of Σ optimum),
      against cd_expand's 64.6 % (43.8 %) and the GNN's 64.8 % (21.7 %).
    - The GNN recovers up to 5.7 % of argmin-S's residual with an oracle selector, concentrated in a few large
      multi-node groups.
    - An earlier "S ties" finding was a tool artefact, retracted (S covered only the last decision on 25
      multi-decision datasets).
- 2026-10-10 19:00 — **Replay variant G fixes the opening-state I11 miss** (S6; rp/replay-g e787127a; 855829, 855845–6,
  855851, 855854).
  - **Miss at F:** heavy opening states (t < 120 s) failed I11, p95 1.57 % and max 3.68 %, with the replay always
    faster than live.
  - **Cause:** `snapshot_fidelity.ghost_order_key` created in-flight ghosts in (pop, tid) order, but a replica's compute
    lock is FIFO by request time. A lower-tid waiter therefore jumped the lock while a pull held the node's storage.
  - **G** orders in-flight ghosts by lock request, after ingress and cold ghosts.
  - **I11 at G:**
    - opening heavy: median 0.011 %, p95 0.200 %, max 0.617 %;
    - opening moderate: p95 0.041 %;
    - steady heavy: max 0.062 %.
    All pass.
  - **Label drift F → G** on 60 accel train datasets: 0 changed and 0 argmin flips. The existing scale160 and accel
    corpora are unaffected; only opening states need G.
  - **Opening label stability** (ε = 0.001 and 0.01 on every in-flight timer, 30 datasets): 0 argmin changes, the same
    as steady states.
- 2026-10-10 18:40 — **CD with α-expansion moves beats CD by 7–8 %** (S5; rp/cd-expand 8e3d39c6, flag
  `HEROSIM_PG_CD_EXPANSION`; 855850; seed 0, g0, 12 gate topologies; descriptive).
  - **Identity:** flag-unset CD is identical to `classical_fcd47841`.
  - **Paired % vs CD:**
    - moderate: −8.20 % (11/12 faster, Wilcoxon p .001);
    - heavy: −7.32 % (11/12, p .001).
    - Exchange falls 14–18 % at about +5 % queue.
  - **Decision cost per task:** median 0.50 / 0.59 ms, against CD's 0.14 ms and the GNN's about 7 ms.
  - **Origin:** suggested by SCIENTIST's classification (the batch problem is metric labelling; CD's single-task
    refine is ICM).
  - **Implication:** cd_expand holds the same information as the GNN, so under AGENTS' rule it is a stronger hand bar
    than CD. The accel GNN's moderate lead (−5.5 / −6.1 % vs CD) would read behind it. That is arithmetic, not paired.
- 2026-10-10 16:30 — **Serving split groups whole helps the GNN in 6 of 6 cells** (S6; diagnostic 855821 at
  rp/accel-nosplit d4ec8688, flag `GNN_SLATE_NO_SPLIT`; ra_gnn_eng seed 1, g0; descriptive).
  - **Identity:** 12 of 12 bit-identical with the flag off.
  - **Flag on vs off:** heavy 16302 −8.4 %, 16310 −6.4 %, 16311 −2.6 %, 16304 −1.3 %, 16309 −7.5 %; moderate 16302
    −8.3 %.
  - **No-split vs CD:** −2.8 / +10.0 / +2.5 / −5.9 / −10.0 % at heavy, and −6.9 % at moderate 16302.
  - **9–10-task groups:** no-split removes the deficit. The GNN beats CD in 5 of 6 cells and ties in 1; exchange falls
    and the same-node rate rises. The model decodes whole 10-task groups well, despite never training on them.
  - **Groups of ≤ 8 lose ground**, but the net is negative (faster) in all 6 cells. Inferred: the co-located large
    groups take the replicas the small groups used.
  - **Against cd_pull** (crossing commits; the shared CD cells are identical): +7.4 / +7.4 / +21.8 / −1.0 / −6.5 % at
    heavy, −4.5 % at moderate.
- 2026-10-10 16:00 — **CD + pull-hold term beats CD** (S6; diagnostic 855822 at rp/accel-pullhold 4ace56e0; g0; 12 gate
  topologies; descriptive).
  - **Term:** `cd_pull` adds max(0, node pull-hold − time-to-output) to CD's base cost, refine included. The live pull
    ledger is behind `HEROSIM_PULL_LEDGER`.
  - **Identity:** 28 of 28 cells bit-identical to `classical_fcd47841` (24 CD cells, plus 4 cells with the ledger on
    and the term off).
  - **Paired % vs CD:**
    - moderate: median −3.21 % (11/12 faster, range −7.3 to +0.6);
    - heavy: median −5.66 % (11/12, range −15.8 to +2.4).
    - Heavy queue time falls in every cell.
  - **Implication:** CD + pull has information the GNN lacks, a node's pull hold. Under AGENTS' rule (the bar is a rule
    with the model's information), plain CD remains the bar for today's GNN. CD + pull becomes the bar once a GNN gets
    pull-hold inputs. It is reported alongside, descriptively. GNN vs cd_pull was not paired.
- 2026-10-10 15:15 — **Post-close diagnostics: where the GNN loses** (descriptive; they close nothing and are inputs to
  the next registration).
  - **CD in the GNN's seat** (S4, 96 cells, f8284fe9, 12 gate topologies; `cd_declared` means the declared top-5 slate
    plus sub-batching at 100k plans): +7.00 % vs CD at moderate (0/12 faster, p .0005) and +7.92 % at heavy (0/12).
    The seat costs CD 7–8 %. Arm-vs-arm in the same seat was not paired.
  - **Slate instrument** (S4, f8284fe9, identity 10/10; heavy, g1, seed 1, 8 cells): 5.5–7.5 % of batches are split.
    These are about 10-task batches, cut into about 3 groups, holding 14–19 % of tasks.
  - **Regime split** (S5, 855803, 8 cells identical to the gate; 16301 and 16302, g0, seed 1):
    - The GNN is ahead of CD in the opening (t < 360 s) in 4 of 4 cells, by a lock-wait gain.
    - The heavy loss is in the steady state, by exchange. This falsifies the "missing opening states" hypothesis.
  - **Peer-group split** (S5, 855815, the same raws, steady state):
    - The GNN beats CD at every peer-group size ≤ 8 in all 4 cells: −8 to −32 % with 9–10 groups excluded.
    - The whole deficit is the 9–10-task groups, the split ones: 0 / +45 / +26 / +79 %, with a lower same-node rate and
      higher pair exchange.
    - Payloads and access classes do not differ.
  - **Fix A** (S6, rp/accel-seqgroups b46ad0ba; later groups see earlier groups as queued load): identity PASS, but the
    GNN got worse on 5 of 5 heavy cells (+0.7 to +14.2 %, paired on vs off). Queue fell and the outside-peer share was
    unchanged. The sub-batch cost is the peer cut, not load. Not registered.
  - **Hidden image pulls** (S6, 855795):
    - In-flight pulls hold a node's local storage and block every output write on that node.
    - Neither CD's cost nor the dim22 features see them.
    - Live, about 90 % of snapshots before 120 s carry them, under 2 % after 360 s.
    - The corpus starts at 361 s: 6 of 438 held-out datasets are affected, but ds_84201 alone is about 30 % of every
      checkpoint's validation regret.
  - **Checkpoint selection** (S7):
    - Final vs best is indistinguishable live (208 cells on validation topologies).
    - Validation regret is heavy-tailed: the worst 10 % of datasets carry 93–96 % of it.
  - **Opening corpus pricing** (S5, 855801): 0.35 / 0.41 worker-s per plan, about Phase 2's cost; supply-limited.
    Parked as hygiene.
