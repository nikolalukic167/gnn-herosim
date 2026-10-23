#!/bin/bash
# Own-state retest: snapshot the 12 best-val checkpoints (.pt + .contract.json) of uncut-dagger-full-{gnn,twin}-seed1..6, md5 both sides, write the md5 file (kind|sN -> md5, _provenance source = the snapshot path).
set -euo pipefail
U=/share/nikola.lukic/uncut_groups; D=$U/own24/ckpt_snapshot; SRC=/home/nikola.lukic/gnn-herosim-wt/dagfull_05f52abb/models
[ -e $D ] && { echo "FAIL LOUD: $D exists"; exit 2; }
mkdir -p $D
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
python3 - <<'P'
import hashlib, json, os, shutil, time
m = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
SRC = "/home/nikola.lukic/gnn-herosim-wt/dagfull_05f52abb/models"; D = "/share/nikola.lukic/uncut_groups/own24/ckpt_snapshot"
out = {"_provenance": {}, "_snapshot_time": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "_src_dir": SRC}
for kind, stem in (("ra_gnn_eng_combfl_dgu_nosplit", "uncut-dagger-full-gnn"), ("ra_twin_eng_combfl_dgu_nosplit", "uncut-dagger-full-twin")):
    for s in range(1, 7):
        for ext in ("pt", "contract.json"):
            f = f"{stem}-seed{s}.{ext}"; a = m(f"{SRC}/{f}"); shutil.copy2(f"{SRC}/{f}", f"{D}/{f}"); b = m(f"{D}/{f}")
            if a != b: raise SystemExit(f"FAIL LOUD: md5 {f} {a} != {b}")
            if ext == "pt": out[f"{kind}|s{s}"] = a; out["_provenance"][f"{kind}|s{s}"] = {"source": f"{D}/{f}", "src_mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(f"{SRC}/{f}")))}
            else: out["_provenance"][f"{kind}|s{s}"]["contract_md5"] = a
json.dump(out, open(D + "/md5_dgu.json", "w"), indent=1)
for k, v in out.items():
    if "|" in k: print(k, v, out["_provenance"][k]["src_mtime"])
P
