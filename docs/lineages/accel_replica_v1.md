# accel_replica_v1 — put the per-server replicas on the accelerators

**Status:** `CLOSED` (2026-10-10) — **NOT-SEPARATED** (no WIN). Registered 2026-10-10; every bar below was signed before
its data.

**Outcome.** Putting replicas on the fastest platform moves the GNN from behind CD to ahead of it at moderate, but not
at heavy, so it is no registered win.
- **Gate:** learned arms 855703 at 9e5ee729, classical 855432 at fcd47841, paired on identity. 12 fresh topologies;
  576 of 576 and 192 of 192 cells.
- **Paired % vs CD at moderate / heavy:**
  - gnn_eng: −5.45 % (9/12 faster, Holm .065) / +2.70 % (Holm .15).
  - physmp: −6.05 % (10/12, Holm .081) / +2.63 % (Holm .15).
  - The moderate medians pass −5 %, but Holm misses .05. Nothing is separated at heavy.
- **CD←GNN beats CD** by −5.03 / −3.40 % (12/12 at both rungs, uncorrected p .0005; descriptive). It is the
  learned-seed effect, and it is the largest of the three studies.
- **Self-predict** is at −1.1 / −1.6 %.
- **Unpaired side by side with `scale_160_v1`** (descriptive, no contrast across gates): gnn_eng goes from +1.2 % to
  −5.5 % at moderate, and from +4.0 % to +2.7 % at heavy.
- **Do not quote:** a GNN win, the moderate direction as confirmed, or "message passing helps". No MP twin was trained.
- A confirmation at moderate would need its own registration on fresh topologies.

**Why.** Under R1.1, `per_server = 1` places each type's server replica on the **first** compatible platform in the
node's list order (`src/generate_infrastructure.py:745–765`).
- On the 160 × 24 calibration topology 9905, every type sits on rpiCpu and xavierCpu. dnn1 also gets pynqFpga.
- **The 7 xavierGpu and 7 xavierDla host no replica at t = 0.** They are reachable only through KPA scale-out.
- CD's lock wait is 90 % on xavierCpu, and half of it is cnn, which runs 7× faster on xavierGpu (`scale_160_v1`,
  2026-10-10).
- The environment is therefore less heterogeneous than its hardware, which may also hide placement structure a graph
  model could use.

## Design (fixed now)

- **Physics change (the only one).** A generator flag `replicas.placement_rule`:
  - `first_compatible` is today's rule and the default, so R1.1 is unchanged.
  - `fastest_compatible` places each type's `per_server` replica on the compatible platform with the lowest
    exec time for that type, from the same exec table the simulator uses. Ties break deterministically by platform id.
    Everything else is R1.1 + the T-fix + replay F.
- **Size:** 160c × 24s, p = 0.6, top-5 slate, as in `scale_160_v1`, so the headroom compares directly with its 69 %.
- **Rungs:** re-derived by the `scale_160_v1` calibration protocol on the reserved seeds 9905–9908 (CD-steered,
  effective share moderate 0.20–0.30 and heavy 0.40–0.50, the same guards, at most 8 steps).

## Worth-it screen (all must pass before any corpus build)

- **W1, the change takes effect.** With CD at the calibrated heavy rung, the accelerator share under
  `fastest_compatible` is at least the share under `first_compatible` + 15 points, and xavierGpu hosts at least one
  type. Amended twice on 2026-10-10 (see the Record): the DLA clause was dropped, and the bar was tightened from an
  absolute 20 %.
- **W2, headroom.** S5's probe (`scale_probe`) on 4 fresh seeds (9909–9912), at the calibrated rungs: at least 3 of 4
  seeds have more than 50 % of batches above 1 % joint (non-pointwise) regret. The pooled share is also reported against
  `scale_160_v1`'s 69 %. This is descriptive and does not gate.
- **W3, replay.** The I11 replay at the calibrated heavy rung, 40 states: median ≤ 1 %, p95 ≤ 1 %, max ≤ 5 %. The
  `scale_160_v1` PASS-WITH-CAUSE ruling applies to sub-second batches with an absolute miss under 0.1 s.
- **W4, no hangs.** 0 captures hang, and every CD guard passes at both chosen rungs.
- **Any failure is a NO-GO.** It is recorded, and the calibration finals (CD, self-predict, batched greedy and
  Knative, run live at the calibrated rungs) stand as this lineage's live read. No corpus is built.

## If GO

The pipeline is identical to `scale_160_v1`, under its own IDs and stem `accel-replica-v1`:
- **Corpus:** replay F, about 50 train topologies and 8 held-out, exact top-5 labels.
- **Arms:** CD vs gnn_eng and gnn_eng_physmp. Each gets 6 configs × 3 seeds, selection on validation only, and the
  100-epoch cap. The MP-off twin is trained only if a GNN wins.
- **Gate:** 12 fresh test topologies × 2 rungs × 4 windows × seeds 1, 2.
- **Primary family:** Holm over 4. A win is median ≤ −5 % with Holm p < .05 at both rungs.
- **Secondary, descriptive:** the same arms against their `scale_160_v1` counterparts (first_compatible).
- **Scheduling:** the corpus build queues **behind** `scale_160_v1`'s build. The combined datalab cap of 44 still holds.

## Record (newest first)

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
- 2026-10-10 11:30 — **Gate read: NOT-SEPARATED, closed** (S4; one read, job 855719, `~/accel_read.sh` repointed to
  `learned_9e5ee729`, reader a95ce980; output `accel_gate/read_fcd47841.{json,txt}`. The file name keeps the
  classical pin; the learned arms ran at 9e5ee729. 768 summaries, 0 failed, 12 topologies; the sensitivity pass is
  identical; 855691 was a failed launch of 0 of 576).

  | Arm vs CD | Moderate | Heavy |
  |---|---|---|
  | gnn_eng | −5.45 % (9/12 faster, p .016, Holm .065) | +2.70 % (3/12, p .077, Holm .154) |
  | physmp | −6.05 % (10/12, p .027, Holm .081) | +2.63 % (4/12, p .151, Holm .154) |
  | self-predict (descriptive) | −1.14 % (9/12, p .027) | −1.64 % (8/12, p .077) |
  | CD←GNN (descriptive) | −5.03 % (12/12, p .0005) | −3.40 % (12/12, p .0005) |

  - **Labels (registered rule):** both arms NOT-SEPARATED; no WIN.
  - **Median decision cost per task:** CD 133 / 163 µs, GNN arms 6.8–7.2 ms.
  - **Spin:** one CD←GNN heavy cell has deferrals above 5× CD's.
  - **Side by side** (`scale_vs_accel_side_by_side.py`, unpaired, median [IQR]; s160 first_compatible vs accel):

    | Arm | Rung | scale_160 | accel |
    |---|---|---|---|
    | gnn_eng | moderate | +1.23 [−0.13, +2.49] | −5.45 [−9.23, −0.72] |
    | gnn_eng | heavy | +3.95 [+2.73, +5.66] | +2.70 [+0.97, +4.85] |
    | physmp | moderate | +1.13 [−2.04, +2.35] | −6.05 [−8.78, −0.23] |
    | physmp | heavy | +3.73 [+2.19, +6.10] | +2.63 [−1.24, +5.46] |
- 2026-10-10 10:40 — **Fix 9e5ee729; identity PASS; learned arms resubmitted as 855703** (S7 fix, S4 identity).
  - **Fix:** `rp/accel-gate-fix` 9e5ee729 on fcd47841, 3 files: `executesimulation.py` passes `space_config` and
    `prefix_serving.py` forwards it, plus tests. All four checkpoint × cell combinations are covered through
    `load_gnn_model`, and reverting the fix reproduces the failure.
  - **Callers left as they are:** `wf1_fidelity_parity.py:125`, `peer_affinity_live_serve_check.py:166` and
    `wide_choice_s0_v1_replay.py:73` still read the global. They refuse an accel checkpoint loudly.
  - **Identity at 9e5ee729:**
    - classical, 855697: 4 cells (cd and selfpredict; 16301, g0; moderate and heavy), 0 differences against
      `classical_fcd47841`;
    - learned, 855698: 2 of 2 ra_gnn_eng seed-1 cells loaded and finished.
  - **Resubmitted:** 855703 at 9e5ee729, SEALED=1, 576 cells; output `accel_gate/learned_9e5ee729`.
  - **Pairing:** the learned arms at 9e5ee729 are paired with the classical arms at fcd47841 on this identity.
- 2026-10-10 10:00 — **Learned launch 855691 failed at load; ruling: fix and resubmit** (S4 report; coordinator ruling).
  - **Failure:** 0 of 576 cells finished. All 576 exited rc 1 at checkpoint load with "physics environment mismatch
    (replica_placement_rule: … 'fastest_compatible', this run has 'first_compatible')". Slurm shows COMPLETED because
    the wrapper exits 0.
  - **Cause:** `prefix_serving.py:209` calls `require_matching_physics_env` without `space_config`. The check therefore
    reads the `replica_rule` module global, which is still the default, because the simulator sets the rule after the
    checkpoint loads. The cells do carry the rule, so it is a serving-check bug from the contract change 5b5dc011. The
    earlier learned-family identity did not exercise a fastest_compatible checkpoint through this loader. scale_160 and
    r1a are unaffected, because their default rule matches.
  - **Ruling:** a code fix, not a rerun. S7 fixes it on `rp/accel-gate-fix` off fcd47841, passing the cell's
    space_config into the check. Tests go through `load_gnn_model`, all four checkpoint × cell combinations.
  - **Before resubmitting**, S4 reviews the diff and runs identity at the new commit:
    - 4 classical cells, bit-identical to 855432;
    - 2 learned cells that must load and finish.
    Pairing against the classical arms at fcd47841 rests on that identity check, and it is disclosed.
  - Counted as a failed launch, not as excluded cells.
- 2026-10-10 09:50 — **Training done and checkpoints selected** (S7; array 855607 all COMPLETED; selection 855608;
  validation only).
  - **Rule:** lowest mean `val/regret_masked_topo` over 3 seeds. The split has 438 datasets on the held-out topologies
    16251–16258.

    | Arm | Chosen | Mean ± sd | Seeds | All configs g0–g5 |
    |---|---|---|---|---|
    | gnn_eng | g4 | 0.736 ± 0.036 | 0.781 / 0.692 / 0.736 | 0.927 / 0.776 / 0.765 / 0.922 / 0.736 / 0.939 |
    | gnn_eng_physmp | g3 | 0.633 ± 0.003 | 0.634 / 0.636 / 0.629 | 0.678 / 0.659 / 0.722 / 0.633 / 0.672 / 0.790 |

  - The corpus differs from scale_160's, so validation numbers are not comparable across the two.
  - Staged at `accel_prod/gate_inputs/models/` (6 .pt + 6 sidecars). The sentinel split sha is f5bc4f51…. Seed-1
    sidecars of both arms carry `replica_placement_rule = fastest_compatible`, and the loader refuses first_compatible
    cells.
- 2026-10-10 06:15 — **Training launched** (S7, `rp/accel-train` 26e4f5b2, pinned): cache 855606, then a 12-task array
  855607 (40G), then selection 855608.
  - The cache records and cross-checks `replica_placement_rule=fastest_compatible`.
  - Checkpoints are staged as `accel-replica-v1-<arm>-seed<N>.pt`. S4 arms the learned gate at fcd47841 with SEALED=1.
- 2026-10-10 — **Corpus PASS; headroom screen** (S5's build `rp/accel-corpus` e8cfc555; check 855578 exit 0; S6
  screen 855605, descriptive).
  - **Check:** 2,784 batches, 3,169 datasets (train 2,731, held-out 438), 0 problems, max |diff| 5.68e-14 over all datasets (S5 first quoted a single dataset's 0.0, corrected). The rule line is in 464 of 464 build logs.
  - **Screen**, train moderate / heavy:
    - share above 1 % joint regret: 48.6 / 61.0 %;
    - exact ties: 14.7 / 30.4 %;
    - multi-node optimum: 44.8 / 25.2 %;
    - single-candidate slots: 6.2 / 1.4 %.
    - Held-out: share above 1 % is 50.3 / 48.5 %.
  - The regret is bimodal, so medians are unstable; quote the share above 1 %.
  - **Against `scale_160_v1`:** fewer batches with headroom (55 % against 65 % on train). But labels are sharper, with
    fewer ties (15 % against 36 % at moderate), and more multi-node optima at moderate (45 % against 30 %).
- 2026-10-10 — **Gate commit fixed; classical arms sealed** (S4).
  - The gate commit is `rp/accel-gate` fcd47841: s160-gate + accel-calib + the 5b5dc011 contract.
  - Default-rule identity on full 50k-task cells at 9483: 24 of 24 cells identical to the 263dd915 / 335abdd1 results.
    They span CD, CD-declared, locality, one-pass greedy, self-predict, Knative, ra_gnn_eng and CD←GNN.
  - CD and self-predict were submitted as 855432 (192 cells), sealed. The learned arms run at fcd47841 on checkpoints
    whose contract says fastest_compatible.
  - Disclosed: S4 saw three CD moderate log lines (16301, 16302) while checking the array had started. Nothing was used.
    The selection is validation-only and the bars are signed, so this cannot steer the result.
- 2026-10-10 — **Contract gap closed; learned-family identity PASS** (S7, `rp/accel-train` 5b5dc011).
  - `replica_placement_rule` is now a cache_physics key; absent reads as first_compatible.
    - The cache cross-checks each dataset's replay infrastructure against its cell config, and refuses mixed or wrongly
      labelled data.
    - The sidecar carries the rule; both serving loaders refuse a mismatch, in both directions.
    - 67 tests pass.
  - **Default-rule identity, learned family:** ra_gnn_eng on 9101 moderate g0, served at base 315f660b and at 5b5dc011.
    The summaries are identical except `code`. The requirement is discharged.
  - Identity now holds for every family except random_network, which is nondeterministic at a single commit.
  - Gate (S4): inputs `rp/accel-gate` 41ef627b, byte-identical to the corpus chain. Every arm runs at one commit that
    merges 5b5dc011.
- 2026-10-10 — **Worth-it screen PASS → GO** (coordinator).
  - **W2** (S7, 855318 at 5e21b11f, post-fix, rule line in every build log): share of scored batches above 1 % joint
    regret.
    - Per seed: 9909 64.8 %, 9910 58.0 %, 9911 63.0 %, 9912 58.5 %, so 4 of 4 pass.
    - Pooled 61.1 % (124 of 203), against `scale_160_v1`'s 69 %. That comparison is descriptive, and S7's definition
      is not cross-checked against S5's.
    - Heavy 73–75 %; **moderate alone 42–57 %** (disclosed: the moderate rung carries less headroom).
    - Pre-fix (855276) pooled 59.6 %. The label-replay gap moved headroom by about seed noise (+1.5 points).
  - **W4:** 16 of 16 captures ok, 0 hung; CD guards 8 of 8 at both rungs.
  - With W1 (+28.8 points) and W3 (max 0.123 %), every bar passes.
  - **GO for the Phase 2 labels** from `rp/accel-corpus` 419dd11f (S5 launches; split 50 + 8 from 16201–16270), then
    training (S7) and the gate (S4, 16301–16312). scale_160 jobs keep priority.
- 2026-10-10 — **Dry run passed; speculative Phase 1 captures allowed** (S5, 855323/855337/855338; seed 16201).
  - 56 datasets. The rule line appears in 8/8 capture logs and 8/8 label-build logs, and in every dataset's
    `infrastructure.json`. The fidelity check finds 0 problems (max |diff| 1.8e-15). The inputs are byte-identical to
    S7's calibration inputs (8/8 windows).
  - **Ruling:** Phase 1 (inputs and captures, 16201–16270) may run before W2, because it is cheap; it submits only at
    ≤ 30 queued. Labels wait for W2. If W2 fails, the captures are scratch.
- 2026-10-10 — **Corpus staged, not launched** (S5, `rp/accel-corpus` 69d8b5f8: S6's scale160 pipeline merged onto
  S7's fixed code; 30 tests pass).
  - IDs: corpus pool 16201–16300 (train and held-out from 16201–16258); **gate test 16301–16312, spares to 16320**
    (signed by the coordinator).
  - Exact multipliers: moderate 11.6139, heavy 27.622665025860204. The rounded "27.6227" is not byte-identical.
  - Launch waits for W2 (rerun 855318) and the coordinator's GO. The byte-diff (855322) and the 1-topology dry run
    (855323) are in flight.
  - **Why CD is slower at moderate than at heavy** (S5, inferred from the finals' decomposition; no rendezvous counter
    was read). Under first_compatible, queue + lock wait rise by 0.30 s from moderate to heavy, while the rest falls by
    0.32 s.
    - The arrival-dependent term is peer rendezvous: a placed task holds its platform until its partners arrive, and
      partner gaps scale with 1/m.
    - This is the same mechanism `workload_fix_v1` measured at 40 × 6. It is larger under fastest_compatible because
      the rung ratio is wider (2.4×).
- 2026-10-10 — **Gap found in the labelling path; W2 rerun ordered** (S7). `make_warm_corpus`'s
  `cell_base_infrastructure` did not carry `replica_placement_rule`, so label sweeps scaled out under first_compatible.
  - Fixed at `rp/accel-calib` 7c789baf (with a unit test).
  - The W2 probe 855276 was launched before the fix. **Ruling (coordinator):** it cannot gate and is descriptive only
    (a pre-fix vs post-fix comparison). W2 reruns from 7c789baf on the same seeds and rungs; it counts only once its
    build logs show the rule line.
  - W3 stands: its replay path (`prepare_infrastructure_for_real_simulation`) was shown to apply the rule.
- 2026-10-10 — **W3 PASS; the replay applies the rule** (S7, `rp/accel-calib` c8b56d8e; I11 855277, check 855296).
  - **I11** at heavy ×27.6227, 9905, 40 states, CD capture: median 0.000 %, p95 0.098 %, max 0.123 %, 40 of 40
    replayed.
  - **Rule in replay:** the same 40 states were replayed under both rules.
    - 10 states create a replica inside the horizon.
    - In 7 of them, the end-of-replay platform mix differs; each new replica takes the faster available platform.
    - The replayed batch latencies are identical under the two rules, because placements are forced. So I11 alone
      cannot detect the rule, and this platform comparison is the evidence.
  - Still to verify before labels: the corpus-labelling path (`make_warm_corpus`) shows the rule line.
- 2026-10-10 — **Rungs calibrated; W1 PASS** (S7, `rp/accel-calib` bb26e296; seeds 9905–9908 × g0/g1; 8/8 guards).
  - **Rungs (fastest_compatible):** moderate ×11.6139 (CD effective share 0.220), heavy ×27.6227 (0.468). Against
    first_compatible's ×16.4245 / ×23.2278.
  - **Finals, mean latency moderate / heavy:**
    - CD 3.13 / 2.51 s
    - Self-predict 3.15 / 2.49 s (a tie with CD)
    - Batched greedy 3.67 / 2.82 s
    - Knative collapses (0.99 effective share).
  - **The congestion moves to the GPU.** CD's heavy lock wait is 47.8 % xavierGpu and 43.9 % xavierCpu, against 90 %
    xavierCpu before.
  - **W1** (job 855273, `accel_s7/w1/w1.json`): CD accelerator share at each rule's own heavy rung.
    - first_compatible: 5.2 % (cell range 1.8–8.3 %).
    - fastest_compatible: 34.0 % (31.2–45.0 %).
    - The difference is **+28.8 points**, against a bar of 15. xavierGpu hosts cnn and rf. **PASS.**
    - Equal load (×23.2278): +31.0 points, so this is the rule's effect, not the load's.
  - Descriptive: under both rules, CD's mean latency is higher at moderate than at heavy (first_compatible: 2.79 vs
    2.69 s). The cause has not been checked.
- 2026-10-10 — **Step 1 done; coordinator's preinit claim RETRACTED** (S7, `rp/accel-replica` 00a08d1a; 16 tests pass).
  - **Correction, measured:** live runs do **not** preinit at t = 0. A 4,000-event CD run on 9903 creates no initial
    replicas, and its first scale events are reachability 'up' events at t = 0.158 s.
    - `simulation.py:395–418` is the fallback branch of `precreate_replicas`, which needs
      `preinitialize_platforms` and a `replica_plan`. `executesimulation.py:200` builds neither.
    - Live replicas are created on demand. Under R1 (kpa) every arm uses `src/policy/gnn/autoscaler.py`
      (`create_first_replica`, `create_replica`).
    - My 2026-10-10 amendment read the code path without running it.
  - **Code:** one helper, `src/placement/replica_rule.py`, at the generator and precreate fallback, in
    `gnn/autoscaler.py`, and in `knative_network/autoscaler.py`. A non-default rule refuses any autoscaler without
    `supports_replica_rule`.
  - **Default-rule bit-identity vs R1.1** (4,000 events, same cell):
    - IDENTICAL for reactive, self-predict, batched greedy and CD.
    - random_network is nondeterministic at a single commit (two base runs differ), so it is not checkable. This is
      disclosed and predates the change.
    - The learned family is **required before the gate** (one ra_* run, default rule, identical to R1.1).
  - **Lever acts live** (CD, uncalibrated ×46.46, descriptive): accelerator share is 16.8 % under first_compatible and
    50.2 % under fastest_compatible. cnn moves to xavierGpu (681 of 1,044 tasks). Mean elapsed is 80.0 s against
    59.3 s.
  - **W1 tightened** (coordinator; the default already reaches 16.8 % at an uncalibrated rung, so 20 % is not a
    test). W1 is now:
    - accelerator share under fastest_compatible ≥ the share under first_compatible **+ 15 points**, both measured with
      CD at the calibrated heavy rung;
    - and xavierGpu hosts at least one type.
    - This is a tightening, disclosed after seeing the uncalibrated descriptive mix.
  - Pre-existing, disclosed: the precreate fallback crashes under `replica_overlap` (double `initialized.succeed()`).
    It is never reached on overlap cells.
- 2026-10-10 — **Amendment before any data (coordinator, on S7's step-1 audit).**
  - **Where the platform is chosen.** It is chosen in three places, and the rule must apply in all three:
    - (i) the generator (`generate_infrastructure.py:745–765`), which feeds co-sim capture and replay through
      `replica_placements`;
    - (ii) **live t = 0 preinit** (`src/placement/simulation.py:395–418`), which re-derives the replicas from the
      same `replicas` config by the same first-suitable rule over `node.platforms`. It does not read
      `infrastructure.json`. So live does create t = 0 replicas; S7's "live doesn't place replicas" covered only the
      network-generation path;
    - (iii) autoscaler scale-out (`create_first_replica` / `scale_up`, `sorted(available_hardware)`, alphabetical,
      e.g. `knative_network/autoscaler.py:210`).
  - **Ruling:** option 1. One shared helper orders compatible platforms by `preinit.replica_placement_rule`
    (`first_compatible` reproduces today's order at each site; `fastest_compatible` uses exec time, ties by name). It
    is used at (i), at (ii), and at (iii) in every autoscaler the gate arms use.
    - The flag lives in `preinit`, not `replicas`, because a string key in `replicas` would be read as a task type
      (`executecosimulation.py:1328`).
    - Required before calibration:
      - default-rule bit-identity of one live run per arm family against R1.1;
      - a test that capture-precreated and live-preinit replica sets are equal under both rules;
      - the run JSON's t = 0 platform mix under each rule.
  - **W1 amended.** Under `fastest_compatible` with overlap on, the exec table sends dnn1 to pynqFpga, dnn2 to
    xavierCpu, and rf and cnn to xavierGpu. xavierDla hosts no type, so "DLA used" cannot hold. W1 is now: at least
    20 % of CD's tasks run on an accelerator, **and** xavierGpu hosts at least one type. DLA use is descriptive.
- 2026-10-10 — Registered on the user's request ("send this study now … check if it's worth and then do if yes").
  Owner: S7.
