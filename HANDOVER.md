# 🚀 Session Handover (2026-08-24)

> ## 🔴 CLOSED 2026-08-24 — the reliability result was a training-draw lottery
>
> **Read this before citing any per-checkpoint number in this repo.**
>
> `p5b_draw_study` (jobs `711758` / `711774`, 480/480 COMPLETED, verdict
> `simulation_data/p5b_draw_study_verdict.json`; full record in `LINEAGES.md`):
>
> **The MLP trainer never seeded torch.** `--random-state` pinned the parent split and the
> batch order; the weight init came from OS entropy. **Every MLP checkpoint in this repo
> before 2026-08-24 is an unreproducible draw.** Fixed now (`torch.manual_seed`, and
> checkpoints carry `torch_seeded`). `src/notebooks/train_near_rtt.py:104-107` always
> seeded properly — the defect is MLP-only, which is why nobody caught it.
>
> With seeds actually controlled, collapse count over the same 30 cells:
>
> | condition | s1 | s2 | s3 | s4 | range |
> |---|---|---|---|---|---|
> | `dim14 / dim22` | 0 | 0 | 8 | 10 | **10** |
> | `dim14 / dim25cr` | 5 | 3 | 0 | 11 | **11** |
> | `tempfix / dim22` | 0 | 0 | 21 | 16 | **21** |
> | `tempfix / dim25cr` | 26 | 0 | 0 | 7 | **26** |
>
> **Retired by this:** "the MLP collapses 7/30" (that exact config gives 0, 0, 21, 16);
> "exactly 7/30 under each checkpoint ⇒ architectural" (two samples from a range-21
> distribution); and P5b's own 7→2 / 7→17 split. The candidate-relative feature's pooled
> median effect is **zero**. Stable at +30/+50/+100%.
>
> **What survives:** both GNN arms are 0/30 with mean margins −18.9% / −27.1% here. But
> that is 2–3 GNN draws against a measured MLP draw distribution in which 4 of 8 draws
> collapse — p ≈ 0.5³ = **0.125**. The GNN's reliability edge is **unfalsified, not
> established.**
>
> ### Start here
> 1. **≥ 8 seeded GNN draws × 30 cells, pre-registered**, if the reliability claim is to be
>    made at all. This is the only remaining route to it. The `p5b_draw_study_*.sbatch`
>    pair is the template — it needs the GNN trainer swapped in and its seeds varied.
> 2. **P5a as written is superseded.** Any reliability gate must compare draw
>    *distributions*, not one checkpoint per arm.
> 3. **Untouched:** the terminal negative from `program_verdict_v1` (single-batch co-sim
>    targets are pointwise-separable) never rested on a checkpoint and still stands. That
>    remains the paper's solid result.
>
> Everything below this box predates the draw study; per-checkpoint margins in it are each
> one draw.

---

> ## ✅ P5b DONE — verdict **INDETERMINATE**, and it changes the plan below
>
> Branch at **`b850da7`+, pushed**; datalab synced, clean tree. Pre-registration `2c5e676`
> (written before submission), implementation `886f559`, jobs `711675` (retrains, both
> validity gates passed) and `711679` (60/60 COMPLETED). Full record: `LINEAGES.md` →
> `### p5b_candidate_relative — OUTCOME`. Artifact: `simulation_data/p5b_verdict.json`.
>
> **The two cache arms moved in opposite directions**, robustly (survives dropping the
> registered detector for a plain RTT criterion):
>
> | arm | cache | collapses | mean margin vs Knative |
> |---|---|---|---|
> | `mlpcandrel` | `dim14` | 7/30 → **17/30** | blows out to +436% / +510% |
> | `mlpcandreltf` | `dim14_tempfix` | 7/30 → **2/30** | **negative in 4 of 6 conditions**, 26/30 cells |
>
> **What it killed:** the sentence "a pointwise scorer collapses *because* it cannot
> condition on where its peers are going." `mlpcandreltf` has exactly that conditioning,
> demonstrably uses it (28.8% of argmaxes move when the columns are ablated), and largely
> stops collapsing. **That mechanism claim must not go in the paper.** The GNN's 0/120
> record is untouched (no GNN arm was re-run) — what is gone is the *explanation*.
>
> **What it strengthened:** all four MLP checkpoints sit within ±0.004 test edge accuracy
> while their live collapse counts span 2/30 → 17/30. Supervised accuracy does not
> constrain live reliability at all.
>
> ### Start here: resolve the seed/cache confound (~half a day)
> Both candrel arms used `--random-state 42`, so cache and training draw are **perfectly
> confounded** — the exact confound `memory/herosim-live-quality-is-a-training-draw-lottery.md`
> describes. Retrain both at ≥3 seeds and re-gate; ~4 min per train, 30 gate runs each.
> Everything needed is in place — `fc_siv1_mlp_candrel.sbatch` and
> `mlp_candrel_arm_all_gates.sbatch` need only a seed/output loop.
> **No claim about which cache is "better" may be written until this is done.**
>
> **P5a (step 2 below) should NOT be run as written.** Its win condition assumed the MLP
> arm was the reliability foil; one MLP arm now beats Knative in 4 of 6 conditions.
>
> **Detector caveat, measured:** `chosen_queue_vs_min` p95 agrees with catastrophic RTT
> 29-30/30 on `mlp`/`mlptempfix`/`mlpcandreltf` but only **25/30 on `mlpcandrel`**, firing
> on five healthy cells including one **−2.5% win** — the CR feature makes non-min-queue
> choice deliberate, so the detector partly measures the intervention. Errors are
> one-directional in all 180 runs (never quiet-but-collapsed), so it stays valid as a
> negative test. **Score candidate-relative arms on RTT.**

**Status of the session below:** work committed on `feat/network-contention-v1`
(`6a2ec46`). This was a read-mostly *verdict* session: no retrains, no physics changes, no
new corpora — it closed the D3 fork left open by `cosim_deepdive_v1` and registered the
execution sequence for what comes next.

> Read first: `PROGRAM_VERDICT.md` (plain-language, 2 min) then `LINEAGES.md` → search
> **"program_verdict_v1"** (the authoritative entry: every path verdict with citations,
> costs, priors, and pre-registered thresholds).

## 0. The one-paragraph story

The question was: can a GNN trained on this repo's co-sim pipeline *ever* beat the MLP and
Knative on a live gate? The answer splits. On per-cell mean latency vs the MLP through any
single-batch supervised target: **no, terminally** — five physics mechanisms, the live
state distribution, and (new this session) the least-additive warmth stratum all measure
the target as pointwise-separable, and the spread-plans control that carries that
conclusion survived its own saturation audit with held-out R² = 1.0 to machine precision.
On the win condition the evidence actually supports — beats Knative everywhere + never
collapses — **the program has already worked**, but the 30/30 record is exploratory (cells
minted for the link A/B, win rule written at scoring time), so it needs one pre-registered
gate before it is publishable. The open paths are ranked and costed; the sequence is
registered at the end of the LINEAGES entry.

## 1. What this session did, in order

1. **Assembled the verdict** (`program_verdict_v1` in `LINEAGES.md`, commit `2ffa7be`):
   path table P1–P7 with mechanism → cheapest decisive test → cost → prior → verdict,
   plus flip conditions for every RULED OUT.
2. **P7 — ran the missing controls on the least-additive stratum** (pre-registered, then
   ~40 s of compute): `warmth_1060` / `sparse_warmth` (31% of the training cache, the one
   place the census showed real additive-argmin regret) go to **spread-plans additive
   R² = 1.00000 exactly, 0.00% regret, 100% optimal** — the non-additivity is entirely
   the collision term. Frozen reports:
   `simulation_data/separability_{warmth_1060,sparse_warmth}_n150{,_spread}.json`.
3. **Audited the spread-plans control itself for mechanical saturation** (commit
   `16ad458`) — it is load-bearing for the throughline's "no reservoir" sentence, and
   R² = 1.0 exactly is also the overfit signature. It survives: held-out R² (seed-0 half
   split) is exactly 1.00000 on 150/150 warmth_1060, **48/48 mh_off (the throughline's
   own base corpus)**, 137/144 sparse_warmth; the 7 failures are all near-saturated
   sweeps (rows/params ≤ 2.29, ≤ 16 spread rows), reported as *unresolvable*, excluded
   from the claim's basis. New tool: `scripts_cosim/audit_spread_fit_saturation.py`.
4. **Calibrated the P3 pilot cost — and found the rps naming bug** (commit `d6d5fa2`):
   `rps=150` is **per client node**; `workload-150-100` runs ~3,000 events/s steady state
   (t≈20–100 s, ~20 s ramps both ends). The first calibration slice sat in the ramp.
   Corrected: ~0.48 ms marginal per event ⇒ ~1.4 s per horizon-second per combo ⇒ h=10 s
   is ~300 CPU-h, not 47. Horizon length is now a registered parameter with a blocking
   in-harness calibration (which must also confirm combos run in-process).
5. **Hardened the gate designs**: P5a must co-register the MLP arm (collapse detector =
   `chosen_queue_vs_min` p95, 9.7× no-overlap gap) and mint fresh cells outside the A/B
   design; P3 got tail co-primaries (median-only can't resolve a link-shaped 3.3% tail);
   residency-hold pre-registers both the count column *and* the spread control.
6. **`PROGRAM_VERDICT.md`** — plain-language executive summary at repo root (`6a2ec46`).

## 2. Findings that correct earlier records

- **P4 was never "already falsified."** `node_contention_v3` holds the node slot only
  around exec (~0.024 s; `src/placement/infrastructure.py:1213-1218`); the residency-hold
  variant (cold starts reach 38 s, a ~1,500× longer hold) is **unbuilt**, and a longer
  hold flips overlap from *never* to *always* — a threshold, not the ratio-invariance
  that killed link bandwidth. It stays ruled out only via the empirical count-shaped
  rule; a 0.5-day pilot converts that into a measurement (registered as step 4 below).
- **The decode-conditioning phrasing in earlier entries is imprecise.** The gates' GNN
  and MLP arms run an *identical* decode (`mlp_scheduler.py:5-7` inherits it; in `argmax`
  mode `chosen_idx = gnn_idx` unconditionally, `seq_decode.py:719-728` — the queue
  roll-forward feeds stats only). Neither arm conditions on batch peers at decode time;
  the real separation is **score-side set-conditioning** (message passing sees the
  candidate context; a pointwise edge score cannot). This is why the P5b control below
  gates the paper.
- **Every "at rps=150" framing describes a system rate 20× the label** — the link A/B's
  "7–14× at rps=150" measurements are untouched, but external-realism judgements made
  against the label are off by 20×. Paper text must state the per-client convention.
- **The 30/30 GNN-vs-Knative record and the 14/120 collapse count are exploratory** — no
  pre-registration language exists anywhere in the backbone campaign span; the "< −0.4%"
  win rule was written at scoring time.

## 3. The registered execution sequence (start here next session)

1. **P5b — try to break our own result (1–2 days).** Retrain the MLP with one
   candidate-relative queue feature (rank/z-score within the candidate set), re-gate the
   30 cells. If it stops collapsing, the reliability separation is feature engineering,
   not architecture — and every P5a gate run before knowing that would be wasted. This
   gates the paper.
2. **P5a — the pre-registered reliability gate (2–3 days, CPU).** Win condition,
   thresholds, and the p95 collapse detector registered *before* running; arms = tempfix
   + MLP + Knative; traces `workload-125-225` + `workload-200-200`; fresh cells from a
   **new** minting script (not `make_backbone_gate_cells.py`). Then promote `tempfix`.
3. **P3 — in-horizon dynamics pilot (2–3 days).** Extend `live_snapshot_cosim_oracle`
   with in-horizon trace arrivals on backbone cells; M4 unmodified. Blocking
   preconditions: 3-snapshot in-harness calibration at h ∈ {2, 5, 10} confirming
   in-process amortization, horizon registered before the array. Co-primaries: median
   regret > 0.02 OR tail fraction (>2%) ≥ 15%, either only with node AND link repair
   < 0.5 on the affected stratum; n ≥ 300 snapshots. A null at h ≥ 5 s is terminal for
   the axis; at h ≤ 3 s it closes only short-horizon dynamics.
4. **Residency-hold pilot (0.5 day) if P3 nulls, before any P1 spend.** 16 datasets,
   both controls pre-registered. Expected outcome: sixth confirmation of the count rule —
   the point is converting the weakest ruling into a measured one for ~1/1000 of P1's
   cost. P1 (closed-loop RL/DAgger, 1–2 wk + 500–5,000 CPU-h, episode wall-clock
   measured at 754.9–1001.8 s) only if P3 or this pilot finds non-count signal.

## 4. Numbers worth keeping close

- Warmth stratum under `--spread-plans-only`: **R² = 1.00000 exactly, 0.00% regret, 100%
  optimal** (both collections; sparse 142/150 fitted). One-integer repair 51%/68%
  (secondary — lean on the spread control).
- Saturation audit basis: **137/144 + 150/150 + 48/48** held-out R² = 1.00000 exactly;
  the 7 exclusions all at ratio ≤ 2.29. Overfit cannot fake held-out exactness.
- Trace profile: ~3,000 events/s steady state, 461/s in the first 10 s (ramp), 20
  sources × ~15,067 events.
- P3 cost surface: ~1.4 s per horizon-second per combo (× assumed 1.5–2× backbone
  overhead): h=2/3/5/10 s ⇒ ~62/92/154/300 CPU-h at 300 × 256 combos.
- Episode wall-clock (P1 costing): GNN 754.9 s on 301k events, 1001.8 s on 352k,
  Knative 472.4 s; ~2.9 GB RSS/worker.

## 5. Environment gotchas (unchanged from last session — still all true)

```bash
PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH=/root/projects/my-herosim \
  pipenv run python3 scripts_cosim/<script>.py ...
# datalab: eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
# in an .sbatch: export HEROSIM_PY=python3 right after activation
# pin OMP/MKL/OPENBLAS/TORCH_NUM_THREADS=4 for ML runs
```

- Live-gate result JSONs are ~80 MB; read bounded prefixes
  (`extract_gate_stats_summary.py` / `extract_platform_dispersal.py` are the reusable
  patterns).
- `executesimulation --policy` takes registry names (`knative_network_batch`), **not**
  the `run_simulation.py` strategy strings (`kn_network_kn_network`) — costs a 5 s
  startup round-trip per wrong guess.
- Read `run_provenance.python_env` from result JSONs, not sbatch banners; grep both
  pipenv spellings when auditing the env leak.

## 6. Restore prompt for next session

```
[CONTEXT RESTORE] feat/network-contention-v1 at 6a2ec46, committed but NOT pushed --
push before any datalab work. Last session was the program_verdict_v1 verdict session
(read PROGRAM_VERDICT.md for the 2-minute version, LINEAGES.md "program_verdict_v1" for
the authoritative entry). Verdict: the supervised co-sim path to "GNN > MLP on latency"
is terminally closed -- five mechanisms + live states + the warmth stratum (P7: spread-
plans R^2 = 1.00000 exactly on both warmth collections) + a saturation audit showing the
spread control's conclusion is out-of-sample exact (137/144 + 150/150 + 48/48). The
reliability result (GNN beats Knative 30/30, 0/120 collapses vs MLP 14/120) is real but
exploratory -- no pre-registration existed. Also found: rps is PER CLIENT NODE (x20 =
~3,000 events/s system rate; every "at rps=150" framing understates concurrency 20x),
and the GNN/MLP arms share an identical decode (the separation is score-side set-
conditioning, not decode-time peer conditioning). Registered sequence, start at step 1:
(1) P5b MLP + candidate-relative queue feature retrain + 30-cell re-gate -- gates the
paper; (2) pre-registered P5a reliability gate (MLP arm included, fresh cells, p95
collapse detector, traces 125-225 + 200-200), then promote tempfix; (3) P3 in-horizon
dynamics pilot with blocking cost calibration (h registered, ~1.4 s per horizon-second
per combo); (4) residency-hold 0.5-day pilot if P3 nulls, before any P1 spend.
```
