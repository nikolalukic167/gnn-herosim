# Joint id allocation (2026-10-25)

One rescan for four studies, nothing minted when filed. Rule: the lowest free run, in the order knative_stable (24), scale_d (24 study + 1 smoke per size), mp_confirm (24), quicklook (6). Exclusions: every token in the scan (all local and remote refs on docs, LINEAGES.md, scripts_cosim, experiments, tests and src; datalab share, simulation_data and worktree file names; sh and sbatch text; 5,562 ids once the sets below are removed), including 30327–30350, fresh6 30298–30303, the sealed 30285–30297 and the ids S6 reported (30383, 30386, 30398, 30400–30405). sha256 is of the comma-joined list.

| list | ids | n | sha256 |
|---|---|---|---|
| knative stable study | 30353–30376 | 24 | `fb6e6947402d694ea313b43025c0dda19ef3c8eddfa676a963641b0d14bf7072` |
| scale d study 64 | 30505–30522 | 18 | `42018603ebdcef25042b52fb03514e98164086bb45a380e34fdfb3f7aadad75e` |
| scale d study 128 | 30523–30528 | 6 | `fbd68ac720e2bf5de308cf1c517713f1d69f0c5b09f0b3fd5020df7cf099f8cb` |
| scale d smoke 64 (outside the study set) | 30529 | 1 | `88efe8995a716cb697930d8e7e069c7d4b9830857e58f8231891bf7080d94223` |
| scale d smoke 128 (outside the study set) | 30530 | 1 | `920087bbe109912470f3c75b82e0b8415d231ee8742ac045dcdca6cb039b781d` |
| mp confirm | 30542–30565 | 24 | `3d92a583a8e81e4bfd266cf876d884962ef7da46ebe2a944a2e83c707b269dfe` |
| quicklook disposable 64 (quicklook, disposable; never study ids) | 30566–30569 | 4 | `a41fef00380ffbd4ca532a98f6a9e56e049f33b1f1b377e2422454015a3cf6b3` |
| quicklook disposable 128 (quicklook, disposable; never study ids) | 30570–30571 | 2 | `ec37f4d608d992074d19d545b6fea52b001c366daa0300305960556682d6b17a` |
| scale d study all24 | 30505–30528 | 24 | `106b02af2f392924439a5aac77612dc9d65db4ca9daf2fc33cd8480546e8d5f1` |

Assumptions: scale_d's 24 study ids are 18 at 64 servers (first) then 6 at 128; the smoke id per size and the quicklook ids follow after all study sets. The ids are reserved here; minting happens when each study needs it.
