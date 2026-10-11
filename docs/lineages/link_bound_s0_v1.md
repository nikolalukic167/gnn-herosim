# link_bound_s0_v1 — does a link-bound accel seat make CD's surrogate wrong in the typical decision?

**Status:** `ACTIVE` (2026-10-11). Registered 2026-10-11; every bar below was signed before any data.

**Now.** Registered. Owners assigned. No data yet.

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
- **The one change:** link bandwidth via `HEROSIM_LINK_BANDWIDTH_MBPS` (`src/executesimulation.py:231`).
  - The ladder is the default, ½, ¼ and ⅛ of the default backbone bandwidth.
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

- 2026-10-11 — **Registered** (coordinator, on the user's GO). The design above was fixed before any data. Owners were
  assigned.
