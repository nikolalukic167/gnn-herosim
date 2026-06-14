# Archived Findings

These files contain experimental results, intermediate models, and analyses that are **not part of the publishable paper narrative**. They are kept here for:

- Internal reproducibility reference
- Debugging history
- Future work context

## Why Archived (Not Deleted)

| Reason | Description |
|---|---|
| Wrong model generation | Results based on 13-dim models (dg-26, clean-1230 argmax) with broken features — not comparable to 14-dim CE anchor |
| 150-150 / 450k dg-26 runs | Superseded; see `legacy_results.md` |
| Rejected interventions | Post-hoc fixes (seqblend, LQB) that were tested and rejected for deployment |
| Confounded ablations | Comparisons where multiple cache fixes happened simultaneously, preventing clean attribution |
| Incomplete / rejected experiments | Tests stopped early, lack cross-config coverage, or failed live gates (e.g. Track B r030 +1.9% vs CE post-fix) |

## What Is NOT Here

The following were dropped entirely — not archived, not published:

- `desert-galaxy-26.pt` (dg-26) results: 13-dim, broken cache, pre-dates all fixes
- `woven-totem-52.pt` results: 13-dim equivalent
- `good-plasma-43.pt`, `brisk-cosmos-41.pt`: seq-filtered training regime, different data pipeline
- `silvery-sun-4` near-RTT v1: loses 7/7 vs dg-26, superseded

## Contents

- `legacy_results.md` — 150-150 benchmark tables, seqblend/LQB ablation full data, historical model comparisons
