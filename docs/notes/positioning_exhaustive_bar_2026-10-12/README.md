# Attachments: repo smoke runs for the positioning note (2026-10-12)

The scripts and logs were run by an agent on 2026-10-12 against clones in a session scratch directory,
which is not kept. Absolute `/tmp/...` paths inside the scripts and logs point there. To re-run, clone
each repo at the commit below and adjust `sys.path`. Smoke scale, direction only.

| repo | commit |
|---|---|
| github.com/zcaicaros/L2D | 7b2efbb1ffc960260b16952f2bed68e500765bf0 |
| github.com/ArchieGertsman/spark-sched-sim | 2ba8d80c2ce57ccb4f20e753476a69295db223c9 |
| github.com/hongzimao/decima-sim | c010dd74ff4b7566bd0ac989c90a32cfbc630d84 |
| github.com/zcaicaros/L2S | 4deb8fcf344d0730049267ab66abc17bf469702d |
| github.com/zcaicaros/ScheduleNet | fba64f329f695a3cc02146001c8323c3b49ff327 |
| github.com/park-project/park | 08f8f7f0dea14e011af2d5ce2a72410084eb8713 |
| github.com/naver/goal-co | f9bd6ae5cc540d664f6da908d375a4d45ab82805 |
| github.com/jolibrain/wheatley | 6a9ca3d4686429cb6f775ebb5c0621ad8d1de095 |

## Files

- `l2d_smoke.log`: L2D's own `test_learned.py` on the pretrained 10×10 network and the
  `generatedData10_10_Seed200` test set. It writes `drlResult_10x10_10x10_Seed200.npy`, which
  `l2d_hand_rules.py` reads as the L2D column.
- `l2d_hand_rules.py`, `l2d_hand_rules.log`, `l2d_hand_rules2.log`: hand dispatch rules in L2D's
  `SJSSP` environment.
  - `l2d_hand_rules.log` came from an earlier version of the script that looped over SPT and MWKR. It
    then crashed on the lookahead rule: `copy.deepcopy` of a gym env.
  - The script as attached loops over MOPNR, FDD/MWKR and the lookahead, and produced
    `l2d_hand_rules2.log`. The SPT and MWKR branches are unchanged in it.
- `sss_smoke.py`, `sss_smoke.log`: FIFO, fair and pretrained Decima in `spark-sched-sim`, 10 executors,
  50 TPC-H jobs. Seeds 1234, 1 and 2 finished; seed 3 hit the timeout (exit 124).
- `sss_handrule.py`, `sss_handrule.log`: fair, the SJF-remaining-work rule at caps 2/3/5/10, and Decima.
  Only seed 1234 finished before the timeout (exit 124).
- `sss_handrule_seed.py`, `sss_handrule_seeds.log` (Table A): 10 executors, 50 jobs. Run once per
  seed (`python sss_handrule_seed.py <seed>`) for seeds 1234, 1, 2 and 3. It runs fair, SJF caps 5 and
  10, and Decima 3 times with torch, `random` and numpy seeded `1000·seed + repeat`. The log's last line
  (all RNGs pinned to 7, env seed 1234: 600.84 s three times) comes from a check whose script is not
  attached.
- `sss_handrule_seed50.py`, `sss_handrule50_seed_{1234,1}.log` (Table B): 50 executors, 200 jobs, the
  environment of `config/decima_tpch.yaml:81–83`. It execs the head of `sss_handrule_seed.py` from its
  scratch path, so both files are needed. Seed 2 produced an empty log. `sss_handrule50_seeds.log`
  concatenates both seeds and the SJF2 lines.
- `sss_handrule_seed50b.py`, `sss_handrule50b_seed_{1234,1}.log`: `SJF2`, SJF with unknown stage
  durations filled from the job's mean (else the global mean) of known durations, at caps 10/25/50.
- **Decima's run-to-run spread** without seeding comes from `random.choices` in
  `schedulers/decima/utils.py:21`, Python's global RNG, which the env seed does not set. The earlier
  `sss_smoke.log` (610.1 s) vs `sss_handrule.log` (638 s) difference on seed 1234 is this.
- `shims/torch_scatter.py`, `shims/torch_sparse.py`: **shims caveat.** `torch_scatter` and
  `torch_sparse` would not install, so these pure-PyTorch replacements (`segment_csr` via `index_add`,
  sparse matmul via `sparse_coo_tensor`) stand in for them. They are not the libraries the port was
  written against. Decima's numbers here are under the shims, and a shim bug would show up as a
  Decima-only error. Decima's run-to-run difference on seed 1234 (610.1 s vs 638 s) is policy
  sampling: with every RNG pinned, three runs are identical (see above).
