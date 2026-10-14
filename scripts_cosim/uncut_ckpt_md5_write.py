"""Write the ckpt md5 file the stager (uncut_arm1_gate_stage.py) reads, from finished -final checkpoints.

  uncut_ckpt_md5_write.py --dir <models dir> --name uncut-arm2k0 --prov prov.json --out experiments/uncut_arm2k0_ckpt_md5.json [--seeds 1,2] [--arms gnn_eng,twin_eng] [--batched true|false]

prov.json is {"<arm>|s<seed>": {"job": "870044_0", "host": "os-cpu-slurm-8", "cpu": "<lscpu Model name>"}}; every staged key needs all three, "file" is filled in
here. Checkpoints are {dir}/{name}-{arm with - }-seed{seed}-final.pt; each needs its -final.contract.json with physics_env.warmth_physics == node_disk_v2 and
batched_teacher_forced == --batched (default: absent or false). Prints the contract md5s (they are not in the md5 file; the stager rejects other _ keys).
"""
import argparse
import hashlib
import json
import os
import sys


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--prov", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--arms", default="gnn_eng,twin_eng")
    ap.add_argument("--batched", choices=("true", "false"), default="false")
    a = ap.parse_args(argv)
    prov_in = json.load(open(a.prov))
    out, prov, contract_md5 = {}, {}, {}
    for arm in a.arms.split(","):
        for seed in (int(s) for s in a.seeds.split(",")):
            key = f"{arm}|s{seed}"
            e = prov_in.get(key)
            if not (isinstance(e, dict) and all(isinstance(e.get(f), str) and e[f].strip() for f in ("job", "host", "cpu"))):
                raise SystemExit(f"FAIL LOUD: {a.prov}[{key!r}] needs non-empty job, host and cpu; got {e!r}")
            file = f"{a.name}-{arm.replace('_', '-')}-seed{seed}-final.pt"
            pt, side = os.path.join(a.dir, file), os.path.join(a.dir, file[:-3] + ".contract.json")
            for p in (pt, side):
                if not os.path.isfile(p):
                    raise SystemExit(f"FAIL LOUD: {key}: {p} missing")
            c = json.load(open(side))
            warmth = (c.get("physics_env") or {}).get("warmth_physics")
            if warmth != "node_disk_v2":
                raise SystemExit(f"FAIL LOUD: {key}: warmth_physics is {warmth!r}, not 'node_disk_v2'")
            btf = c.get("batched_teacher_forced")
            if (btf is not True) if a.batched == "true" else (btf not in (None, False)):
                raise SystemExit(f"FAIL LOUD: {key}: batched_teacher_forced is {btf!r}, expected {a.batched}")
            out[key] = md5(pt)
            prov[key] = {"job": e["job"], "host": e["host"], "cpu": e["cpu"], "file": file}
            contract_md5[key] = md5(side)
    if len(set(out.values())) != len(out):
        raise SystemExit("FAIL LOUD: two keys share one md5")
    for arm in a.arms.split(","):
        jobs = {prov[k]["job"].split("_")[0] for k in prov if k.split("|")[0] == arm}
        if len(jobs) > 1:
            print(f"WARNING: {arm}'s seeds come from different jobs {sorted(jobs)}; the stager's hedge rule will refuse this", file=sys.stderr)
    json.dump({"_ckpt_dir": os.path.abspath(a.dir), "_provenance": prov, **out}, open(a.out, "w"), indent=2)
    open(a.out, "a").write("\n")
    for k, v in contract_md5.items():
        print(f"contract md5 {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
