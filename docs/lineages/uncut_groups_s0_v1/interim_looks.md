# uncut_groups_s0_v1 — interim-looks log

Kept by S6 under amendments 70022963 and 190e5771. Every interim read is `INTERIM-DESCRIPTIVE`: direction, n ahead and mean Δ per rung, with no WIN/LOSS label and no Holm p. It is never quoted as a result and never reused by the final read. The final read reproduces this table.

**Freeze before the first interim cell** (190e5771):
- gate G3 `114cc0e7`;
- reader, stager and calibration tools at `557d84bf`;
- `cot_calib_v2` b46fd80f, `cot_hand_smt` 0082b2b9, `cot_hand_steps` 6ffc62b1.

**First interim cell:** job 871880 at `114cc0e7`, started 2026-10-14T17:36:14 (cluster clock). It is the 1b twin's PROVISIONAL-W calibration on 19461 g0. The first interim k = 1 cell is logged in its row below.

**Venue rule.** Interim cells run on GPU-node CPUs with no GPU while CPU-amd is reserved. Each row gives its host block. Calibration runs PAR = 1 with 1 thread; the cells run 1 thread with PAR ≤ physical cores − 4.

**B7.** The provisional-w calibration uses 19461 g0, the registered calibration topology. It was already used for w and K1, and it is never a gate topology or a fresh-topology draw. The expert ruled 2026-10-14 that this is fine and nothing new is seen. The k = 1 cells use only the 869828 set: 19537–19548 × g0–g3 × moderate/heavy.

| # | arm | date | cells | provisional w (ms/plan, mod / heavy; branch sha) | what was reported |
|---|---|---|---|---|---|
| 1 | Arm 1b twin_eng (`ra_twin_eng_arm1b_nosplit`), seeds 1, 2 | 2026-10-14 | Calibration 871880: 2 cells. k = 1 job 871907 at `f7f15566`: 192 cells, 0 failed, 0 kills, first cell 17:42:24. a-l40s-o-2, EPYC 9355, SMT on, 32 cores, PAR 28, 1 thread. | 2.286 / 2.354 pooled; by bucket le10 / 11_50 / gt50 = 2.936 / 1.068 / 1.295 (moderate), 3.007 / 1.072 / 1.555 (heavy). `rp/uncut-interim` f7f15566. Charge share 0.088 % / 0.117 %. | Seeds pooled vs `cd_ib0_k100` (869828), median / mean over 12 topologies, n ahead: moderate whole −3.33 / −4.85 %, 11/12; moderate after-360 −5.38 / −6.55 %, 12/12; heavy whole +1.11 / −0.00 %, 5/12; heavy after-360 −2.33 / −4.48 %, 8/12. Per seed also sent. `interim/read_1b_twin.json` md5 fceb64fb. |
