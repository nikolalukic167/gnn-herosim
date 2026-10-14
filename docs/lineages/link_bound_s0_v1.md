# link_bound_s0_v1 — does a link-bound accel seat make CD's surrogate wrong in the typical decision?

**Status:** `CLOSED` (2026-10-11) — **NO-GO LINKS-DO-NOT-BIND; live HAND-SEARCH-FASTER**. Registered 2026-10-11;
every bar below was signed before any data.

> **Declared-slate caveat (resolved 2026-10-12):** the heavy deficit vs `cd_exactS` is a candidate-set handicap (`slate_bar_v1`: matched-slate, the GNN self-search wins −4.85 / −1.94 %); not an MP win. Its vs-CD numbers are on the cold-mispriced slate and not re-measured on the corrected slate; `slate_fix_v1` (CLOSED 2026-10-12, E-FASTER) retrained uncapped and still loses to the hand bar `cd_exactS_ib0` (self-search +2.6 % moderate / +5.3 % heavy, no-split +2.8 / +7.4 %).

**Outcome.** Slowing the core links does not make CD's surrogate wrong in the typical decision, and it opens no learned
gap.
- **Screen:** at the lowest CD-healthy core rate (3.9 MB/s), link wait is 1.2–1.6 % of CD's latency. It passes 5 % only
  where CD is unhealthy (≤ 1.95 MB/s), so the link is then a load lever.
- **Live gate at 3.9 MB/s** (288/288 cells, 12 fresh topologies 17007–17018, all 96 CD cells healthy), median paired %,
  moderate / heavy:
  - GNN self-search vs CD: −1.38 % (12/12) / +1.24 % (3/12);
  - self-search vs cd_exactS: +10.69 % / +11.14 % (0/12 each);
  - cd_exactS vs CD: −11.13 % / −8.02 % (12/12 each).
- Even in the gate, link wait stays ≤ 1.75 % of latency for every arm.
- **Quote as:** "links do not bind in the accel seat before CD overloads; hand search on S stays the bar". The
  self-search checkpoint was trained at 1000 MB/s and not retrained (as registered), so this is no claim about a GNN
  trained on link-bound data.

**Why.** After `cost_to_go_v1` (NO-WIN) and `big_groups_s0_v1` (live G8 HAND-SEARCH-FASTER), the accel seat's verdict is
that CD's surrogate S plus a good search is the bar.
- **S is nearly right:** argmin-S misses the optimum in 21 % of tables, and its regret is tail-only (median 0) even at
  G=8.
- **The platforms are 95–99 % idle.**
- **The one thing S misprices inside validated physics is shared-link exchange.** It prices each pair alone on its
  path.
- **This screen asks whether making the links, not the platforms, the bottleneck moves S's error from the tail to
  the median.** Load is held fixed and CD must stay healthy. It is the last cheap environment check before the record
  states "search is the bar" for this seat.

**Prior, disclosed: expect NO-GO.** Four earlier link-contention screens failed, on old physics and other seats.
- `g32_peer_fabric_s0_v1`: core wait was only 0.6–1.1 % of RTT, because staggered service de-overlaps the transfers.
- `g32_peer_prefetch_s0_v1`: the wait was mostly on access links (a count, not a graph effect).
- `g32_core_prefetch_s0_v1`: contention was real (13.7 % oracle gain), but a route-aware hand learner captured 72–100 %
  of it.
- `dag_fabric_contention_v1`: the optimum avoids link wait in 98–100 % of datasets.
- In `big_groups_s0_v1` the hand shared-link term S′ did not fix S (M4), but S's error there was tail-only.
- At 6 servers a slower backbone was a load lever first (`backbone_sparsity_v1`, `payload_scale_v1`), hence the load
  pre-check below.

## Design (fixed now)

- **Environment:** the accel seat unchanged (R1.1 + WF1, fastest_compatible, 160c × 24s, the current peer-group size,
  the same apps, payload model, pipelined transfers and KPA), moderate and heavy rungs.
- **The one change:** core-link bandwidth, as a cfg edit of `network.backbone.bandwidth_mbps` (Amendment 1; the env
  knob is a no-op here).
  - The ladder is 1000 / 250 / 125 / 62.5 / 31.25 MB/s. Access classes are unchanged.
  - The owner confirms the knob reaches the live path and the replay/label path identically before any screen read.
- **Load pre-check, before capture** (reactive is not the gate here; CD is the bar):
  - CD on the 8 development topologies × 4 windows × both rungs at each bandwidth;
  - a bandwidth is admissible only if CD's effective queue share is ≤ 0.80 and end/last-arrival ≤ 1.05 in every cell;
  - a hung, failed or missing run is not a pass;
  - median per-link utilisation is reported per cell;
  - the screen uses every admissible bandwidth below the default. If none is admissible: **NO-GO, LOAD-LEVER**.
- **Topologies:** 8 development ids from an unused range, disjoint from every gate pool (16251–16258, 16301–16399,
  16490–16523, 16369–16392, 16900–16911) and from the training corpora. The owner verifies the range is unused.
- **States:** about 50 per (bandwidth, rung), captured from CD runs with the fidelity replay at the same bandwidth
  (fixes daf74ffa, a96b8f27, 8e19fcab).
- **Reference plan:** the best-known plan by true replayed batch latency, by the `big_groups_s0_v1` iterated local search
  (start from argmin-S, the cd_exactS plan and the GNN self-search plan; budget B = 600 per state), identical at every
  bandwidth.
- **Measures per (bandwidth, rung):**
  - **L1, link-wait share:** link wait over batch latency **at the reference plan**, median over states. It must bind at
    the optimum, not only at bad plans.
  - **L2, typical S error:** the **median** per-state argmin-S regret vs the reference. This is the fix for
    `big_groups_s0_v1`, whose pooled regret was tail-only.
  - **L3, the fair hand bar:** the median per-state S′ regret (S plus S5's hand shared-link term).
  - **L4, landscape:** the median gap between the top two **latency-distinct** plans (M2's design flaw fixed).
- **GO** requires, at one admissible bandwidth and **both** rungs:
  - L1 ≥ 10 %;
  - L2 ≥ 2 %;
  - L3 ≥ ½ · L2, so the hand route-aware fix leaves at least half.
  Otherwise **NO-GO**, and the record states "search on S is the bar in the accel seat".
- **Live gate (rule 6, it runs either way):** at the lowest admissible bandwidth, on 12 fresh topologies × 4 windows ×
  both rungs.
  - Arms: the current GNN self-search (ra_gnn_eng s1, no retraining), CD and cd_exactS. CD is the bar.
  - The reader is committed before the data. A GO adds a retrained arm on a link-bound corpus as its own registered
    successor.
- **Owners:**
  - S7: the bandwidth ladder, the load pre-check and the topology range;
  - S4: knob parity (live vs replay at reduced bandwidth, 40 plans identical);
  - S6: capture, ILS reference and the screen read;
  - S5: S′ at reduced bandwidth.
- **Cost, estimated:**
  - pre-check about 1–1.5 h; capture about 1 h; ILS about 2–3 h, so the screen is about 4–5 h;
  - the live gate is about 5–6 h, because the cd_exactS cells are slow.

## Record (newest first)

- 2026-10-11 — **Live gate read once: HAND-SEARCH-FASTER; lineage CLOSED** (S7; array 858137, reader e02ad362,
  288/288 cells finished, 0 failed, 0 killed; `/share/nikola.lukic/link_bound/gate_3p9_read.json`).
  - **Statistic:** paired % of mean latency per topology (median over 4 windows), then the median over 12 topologies;
    exact Wilcoxon.

    | contrast | rung | median | first arm faster | p |
    |---|---|---|---|---|
    | self-search vs CD | moderate | −1.38 % | 12/12 | .0005 |
    | self-search vs CD | heavy | +1.24 % | 3/12 | .0093 |
    | self-search vs cd_exactS | moderate | +10.69 % | 0/12 | .0005 |
    | self-search vs cd_exactS | heavy | +11.14 % | 0/12 | .0005 |
    | cd_exactS vs CD | moderate | −11.13 % | 12/12 | .0005 |
    | cd_exactS vs CD | heavy | −8.02 % | 12/12 | .0005 |

  - **CD health, per cell:** 96/96 healthy. Queue share max 0.283 / 0.551, end/last-arrival max 1.019 / 1.044.
  - **Link telemetry** (median link-wait share, moderate / heavy): CD 1.15 / 1.59 %, cd_exactS 1.22 / 1.75 %,
    self-search 0.84 / 1.38 %. Core utilisation for CD is 3.8 / 8.0 %, and the busiest core link is ≤ 21 %.
  - **Search counters:** self-search is 80 / 77 % exact (the rest ascent) and changes 39 / 41 % of batches. cd_exactS
    falls back in 6.4 / 8.1 % of batches.
  - **Disclosures:**
    - the checkpoint (ra_gnn_eng s1) was trained on the 1000 MB/s corpus, as registered;
    - the smoke cell (17007 g0 heavy) is one of the 288 and was not rerun.
    - Compared with G8 (`big_groups_s0_v1`), the learned arm's lead over CD shrinks and its gap to cd_exactS widens.
      Those are different seats and topologies, so no causal claim is made.
  - **Ruling:** CLOSED. Three levers in a row (lookahead value, bigger groups, busy links) leave hand search on S as the
    bar in the accel seat.
- 2026-10-11 — **Live gate at 3.9 MB/s launched** (S7; array 858137, 12 tasks, 288 cells, code pinned e02ad362).
  - Topologies 17007–17018, verified unused by a token scan of /share and simulation_data plus a grep of docs, code and
    experiments. 17200–17211 was rejected because ds_172xx exists in a training corpus.
  - Reader `link_bound_gate_read.py` (e02ad362), committed before any data:
    - paired % per topology, median, sign count and exact Wilcoxon;
    - contrasts self-search vs CD, self-search vs cd_exactS, and cd_exactS vs CD;
    - per-cell CD health; unhealthy cells are listed, never dropped.
  - Smoke, one heavy g0 cell per arm on 17007, part of the 288: 0 failed. The core rate 3.90625 reached the live fabric,
    and CD's link wait is about 1.9 % of latency.
- 2026-10-11 — **Extension read: NO-GO, LINKS-DO-NOT-BIND by the signed stop rule; the ILS screen is not run; the live
  gate runs at 3.9 MB/s** (S7; 384 more cells, 0 failed or hung, pinned f9de3e8a;
  `/share/nikola.lukic/link_bound/precheck_read_ext.json`).
  - **CD link-wait share** (mean per-task link wait over mean latency, median over 32 cells), with admissibility:

    | MB/s | share mod / heavy | mean latency mod / heavy | inadmissible cells | admissible |
    |---|---|---|---|---|
    | 1000 (= 250 = 125) | 0.52 / 0.71 % | 3.31 / 2.67 s | 0/64 | yes |
    | 31.25 | 0.53 / 0.72 % | 3.39 / 2.71 s | 0/64 | yes |
    | 15.6 | 0.57 / 0.77 % | 3.43 / 2.74 s | 0/64 | yes |
    | 7.8 | 0.72 / 0.95 % | 3.49 / 2.74 s | 0/64 | yes |
    | **3.9** | **1.16 / 1.58 %** | 3.79 / 2.93 s | 0/64 | **yes (lowest admissible)** |
    | 1.95 | 2.19 / 3.51 % | 4.40 / 3.20 s | 2/64 | no |
    | 0.98 | 4.37 / 11.57 % | 5.00 / 4.64 s | 10/64 | no |
    | 0.49 | collapsed at heavy (2145 s) | — | 49/64 | no |

  - **The share passes 5 % only where CD is already unhealthy.** By then the core links are 12–43 % held and the link is
    the load lever, the confound the pre-check exists to exclude.
  - **Ruling:** NO-GO, LINKS-DO-NOT-BIND, as signed.
    - No intermediate rung is probed: the stop rule does not need one, and tuning the rung after the read is not allowed.
    - S6's capture and ILS screen are not run, and S4's second parity is not needed because the gate is live-only.
  - **Live gate (rule 6)** at 3.9 MB/s, set up by S7: 12 fresh topologies × 4 windows × 2 rungs; ra_gnn_eng self-search
    s1, CD and cd_exactS; the reader committed before data. Expected to repeat the G8 picture.
- 2026-10-11 — **Knob parity: ACCEPTED with a disclosed residual; a second parity at the lowest qualifying rung, with
  its bar signed now** (S4; topology 17100 moderate g0, 40 decisions with the live plan forced in the replay; 858098 /
  reader 858123 at 62.5 MB/s on S6's capture code; control 858124 at 1000 MB/s; failed launches 858095 and 858097 are
  unused).
  - **The knob reaches both paths:** the core rate is 62.5 in the live run and the replay, against 1000 in the default
    cfg, and the stamp check did not raise.
    - Plans match on 172/172 tasks (62.5) and 161/161 (1000). The median |Δ latency| is 3e-13 s.
    - Live linkWaitTime > 0 on 86 tasks, and the replay shows the same.
  - **Residual, always replay-shorter:**
    - **Exchange residual, present at both bandwidths:** 62.5 MB/s, 1 task up to −92.6 ms; 1000 MB/s, 2 tasks up to
      −69.4 ms. The tasks have no link-wait difference and carry peer exchange, in 7–10-task batches.
    - **Link-ordering residual, 62.5 MB/s only, one state (n = 1):** 3 tasks, −12.75 / −25.5 / −30.4 ms, accounted for
      by link wait.
    - The cause is not localised for either.
  - **Ruling:** accepted, disclosed. The exchange residual predates the knob.
  - **Signed bar for the second parity**, at the lowest rung that qualifies under Amendment 3, run before the screen
    read: plans match 100 %, ≥ 95 % of tasks within 1 ms, and the max residual ≤ 5 % of that task's latency.
    Otherwise the link-ordering residual is localised and fixed before any screen read.
- 2026-10-11 — **Pre-check read; Amendment 3 (before any S-regret is read): extend the ladder down, with a stop rule**
  (S7; 320/320 cells, 0 failed, 0 hung; telemetry added at rp/link-bound-telemetry f9de3e8a, because 79ea002c carried no
  link-wait or utilisation fields; reader `link_bound_precheck_read.py`; `/share/nikola.lukic/link_bound/precheck_read.json`).
  - **All five rungs are admissible** (worst queue share ≤ 0.63, end/last-arrival ≤ 1.042). **The core never binds.**

    | MB/s | rung | worst queue share | median busiest core link held | median per-task link wait | mean latency |
    |---|---|---|---|---|---|
    | 1000 = 250 = 125 | moderate / heavy | 0.39 / 0.63 | 1.8 / 3.9 % | 1.3 / 2.0 ms | 3.31 / 2.67 s |
    | 62.5 | moderate / heavy | 0.36 / 0.59 | 1.9 / 3.9 % | 2.0 / 2.3 ms | 3.41 / 2.73 s |
    | 31.25 | moderate / heavy | 0.34 / 0.59 | 1.9 / 4.1 % | 3.2 / 4.7 ms | 3.39 / 2.71 s |

  - **1000, 250 and 125 are bit-identical.** A pipelined transfer holds its route at the minimum link rate, and the access
    classes cap that at ≤ 117 MB/s. The mean link wait is 18–20 ms, about 0.6 % of mean latency at 31.25.
  - **Ruling, Amendment 3:** the premise, a link-bound seat, is not reached yet, so the ladder is extended downward
    before any screen data.
    - New rungs: 15.6 / 7.8 / 3.9 / 1.95 / 0.98 / 0.49 MB/s, CD only, the same 64 cells per rung and the same
      admissibility rule.
    - **Stop rule, signed now:** CD's link-wait share is mean per-task link wait over mean latency, per rung, at both
      load rungs.
      - The screen runs only at admissible rungs where that share is ≥ 5 % at both load rungs (half the L1 bar, as a
        margin for the reference plan differing from CD's), plus the 1000 reference.
      - If no admissible rung reaches 5 %: **NO-GO, LINKS-DO-NOT-BIND**, and the ILS screen is not run.
    - The live gate (rule 6) still runs, at the lowest admissible rung.
- 2026-10-11 — **Amendment 2, before data: screen details** (S6's reader and scorer committed at rp/link-bound-s0 67b855f0,
  pinned; coordinator rulings).
  - L1–L4 are as registered, from the label path's own replay taskResults. The GO rule is as registered, plus: any
    error row or empty cell cannot pass.
  - **Top-5 slate only** (the served one), at the current group size.
  - 2 states per capture cell, so up to 64 per (bandwidth, rung).
  - Capture cells are S7's pre-check cells, re-run with live-audit snapshots.
  - **The default 1000 MB/s IS screened, on the same 17100–17107 topologies,** as the descriptive reference. It cannot
    GO; it exists so the ladder is read on the same topologies, not against big_groups' different ones.
  - **Disclosed, as in big_groups:** the GNN self-search plan is not an ILS start, because it is unavailable offline.
  - **L3's S′ is re-targeted to input routes** (S5, rp/cd-exacts aa208b1d, flag `HEROSIM_PG_INPUT_LINK`).
    - For each task, the term adds the maximum, over the core links on its input route, of the summed holds of the
      batch's other inputs that share that link. It is order-free, so it is an upper bound against FIFO.
    - **Registered L3 uses `hold`:** each hold is input / route bottleneck, which is what the pipelined physics charges.
      The literal `core` variant (input / core bandwidth) is reported descriptively.
    - The old peer-pair S′ priced exchange contention that does not exist in this physics.
- 2026-10-11 — **Pre-check set up** (S7).
  - Development topologies 17100–17107, verified unused.
  - The ladder is minted into the cfg `backbone.bandwidth_mbps`; `core_bandwidth_mbps` defaults to it
    (`generate_infrastructure.py:421`).
  - Input build 858071 at G=0 (the current accel seat).
  - CD only: 5 bandwidths × 8 topologies × 4 windows × 2 rungs = 320 cells, as 20 jobs pinned at 79ea002c (the G8 gate
    harness).
  - S4's parity cell is 17100 / moderate / g0.
- 2026-10-11 — **Amendment 1, before any data: the knob is a config edit, core-only, with a deeper ladder; L1 is
  input-transfer wait only** (coordinator, on S6's code read, verified at `src/executesimulation.py:228–245` and
  `src/placement/infrastructure.py:1250–1275`).
  - **Finding 1: `HEROSIM_LINK_BANDWIDTH_MBPS` is a no-op in this seat.** It only synthesises a backbone when the config
    has none, and every accel cell config carries an explicit `network.backbone` (n_core 12, bandwidth_mbps 1000,
    access classes wired / wifi / cellular). **Ruling:** the ladder is new cfg files per bandwidth, editing
    `network.backbone.bandwidth_mbps`.
  - **Finding 2: `bandwidth_mbps` sets only the 12 core ring links.** Access links keep their class rates: wired 117,
    wifi 7–14, cellular out 4 / in 75 MB/s. At ⅛ (125 MB/s) the core is still faster than every access link.
    - **Ruling: core-only (option a).** Scaling access rates too would add node-indexed access-link wait, which
      `g32_peer_prefetch_s0_v1` already closed.
    - **The ladder becomes 1000 / 250 / 125 / 62.5 / 31.25 MB/s.** The last two rungs put the core below the wired
      access rate.
    - If the core-only ladder cannot make L1 bind, that is the NO-GO answer, not a reason to scale access links.
  - **Finding 3: in this physics only INPUT transfers queue on link pipes** (`linkWaitTime`, the pipelined branch).
    - Peer exchange (`_peer_exchange_time`) is an additive charge and never waits on a pipe.
    - L1 is therefore input-transfer link wait over batch latency at the reference plan. linkTransferTime and
      peerExchangeTime are recorded per state for the denominator split.
    - **Making exchange queue would be a physics change, out of scope here** (`g32_peer_fabric_s0_v1` is the prior).
  - **L3 consequence:** S′ must price the contention that exists, input routes sharing core links, not exchange
    sharing. S5 re-targets S′ before any read.
- 2026-10-11 — **Registered** (coordinator, on the user's GO). The design above was fixed before any data. Owners were
  assigned.
