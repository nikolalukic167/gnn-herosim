# GNN Restructuring Roadmap

> **Started:** 2026-06-10  
> **Updated:** 2026-08-13  
> **Status:** Regime B **case study** closed on `oracle_split_v1`. Paper split into two regimes.

---

## Current goal

**Case study CLOSED; paper integration in progress.** Learned residual on `oracle_split_v1` closed: multiseed cold `ect_pull` distill + `seq_reforward_pull` → **31.66s** ≡ ect_pull/oracle **31.65s**.

Write it as **RQ9 / case study**, not a title-level win. Main story stays RQ3 (MLP 13/20 · Kn 6/20 · GNN 1/20). See `paper/case_study_regime_b.md`.

Claim is **bounded** to the intel cell — not a global replace of 873/v5.5.

---

## What is closed

| Track | Outcome |
|-------|---------|
| Regime B gate + intel margin (873) | **DONE** |
| Pull-obs CACHE 5.6 retrain | **NEGATIVE** |
| Physics `ect_pull` ceiling | **SHIPPED** — 31.65s |
| Oracle-split cosim 48 + CACHE 5.6 + retrain | **NEGATIVE** — MLP 125.28; GNN 187.93 |
| Phase 0–2 (parity / pull-decode / soft_combo_conc) | **DONE / REJECTED** (falsified) |
| Phase 3 single-stub distill | **PARTIAL** — 62.63s |
| **Phase 3.1 cold multiseed distill** | **DONE** — **31.66s** ≡ ceiling (`fb4e729`) |
| Set Transformer / tripartite HGT / N=4 static CE for Regime B | **REJECTED** — wrong tools for sequential FilterStore state |

---

## Next experiment queue (only)

1. **Transfer `/compare` (gate3)** — distill + `seq_reforward_pull` vs Kn / 873-argmax / `ect_pull` on 100-100. Expect a loss; needed so the paper can say so.
2. **Multi-cell Regime B DONE** — zero-jitter/latency/2nd burst = 31.66s; **jitter 0.5s = 125s, jitter 2.0s = 62.63s**. Write the fail.
3. Optional later: true **busy-without-warm** init. Do not reopen RQ3 / hub9 decode / Set Transformer / N=4 CE.

## Limitations (paper)

- Set-equivariant architectures not tested — open whether a ledger is necessary given sequential labels.
- Distill ckpt is dim24 + per-task re-forward; 873 deploy is dim14 argmax.
- Cold-only harvest; busy-as-warm poisons the teacher (375s).
- 31.66s is N=12, not 200k-task `/compare`.

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-08-13 | **Regime B CLOSED via Phase 3.1** | Cold multiseed ect_pull distill (6k frames) + seq_reforward_pull = 31.66s. PR #3 `fb4e729` merged. |
| 2026-08-12 | **Warm/busy v1 init poisons distill** | Busy ⇒ previous_task ⇒ warm ⇒ zero pull; teacher map wrong → 375.87s. Default harvest cold-only. |
| 2026-08-12 | **Hard-CE-only distill REJECTED** | α=0 → 125.28s; soft KL required. |
| 2026-08-12 | **Phase 1–2 REJECTED** | Pull-ledger alone / soft_combo_conc → pile; Set Transformer discarded. |
| 2026-08-12 | **N=4 oracle_split CE ≠ residual close** | MLP 125.28 (=873); GNN≈Kn; val joint acc 0% / MLP edge 25%. |
| 2026-08-12 | **Ship ect_pull physics ceiling** | 31.65s = oracle on intel cell. |
| 2026-08-12 | **Pull-obs ≠ residual close** | GNN 125; MLP 375. |
| 2026-08-11 | **Freeze regime_b; RQ3 ANSWERED** | Sprint = Regime B. |

---

## References

- Memory: `memory/memory.md` v0.39.1
- Phase 3.1 eval: `…/regime_b_phase3_ect_pull_distill_eval_multiseed_cold/summary.json`
- Ckpt: `models/near-rtt-v2-regime-b-oracle-split-v1-ect-pull-distill-multiseed.pt` (`f98f43f5…`)
- Ceiling: `…/oracle_split_v1_physics_ceiling_zeroshot/compare.json`
- PR #3: `cursor/regime-b-parity-phase0-af73` → main (`fb4e729`)
