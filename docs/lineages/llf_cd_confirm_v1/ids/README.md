# llf_cd_confirm_v1 topology ids (amendment A1, n = 24)

- Ids: 30327–30350, `ids.txt`. sha256 of the comma-joined list: `8d9d17a1b7e5c45ea4eb8fa99e11ddca46ff3f32248f7ce1e727d44424d1c563`.
- Rescan (TE3 rule, all local and remote refs, datalab names and sh/sbatch text, 5,394 tokens): lowest free run of 24 at or above 30304. 30304 and 30326 are taken by incidental numbers in committed result JSONs; 30305–30325 is only 21 long.
- Mint: `cdc_mint.sh` (md5 3dff4b2380177deb95a5685143db6240), recipe 224dd02e (big_groups_inputs.sbatch db71ec73…, uncut_groups_group_inputs.py 9efb1da0…), then the cap200 clip. Jobs 905812, 905813, 905814, all COMPLETED.
- x1 inputs MD5SUMS md5 `f07dfdfb62b49038b4416e539d1ecf37`; x1 inputs_group MD5SUMS md5 `517cf7b1474e539fd393487bcd721b40`; cap200 inputs_group MD5SUMS md5 `8420ae6871af40e05b1c250745e594e6` (58 files, md5sum -c ok).
- cfg md5 (cat of the moderate and heavy cfg/*.json): `a0aec21247b68e79767ee20a457bb0c8`, identical before and after the clip.
- Data: /share/nikola.lukic/uncut_groups/llf_cd_conf/gate_inputs_cap200/inputs_group.
- Admission: `cdc_react.sh`, reactive s0, g0–g3 × both rungs, pin 0ec7cdcb (the reader's bar group).
