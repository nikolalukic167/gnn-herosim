"""Stage the uncut_groups_s0_v1 Arm 1 learned-gate inputs root. Checkpoints enter only through the md5 file.

  uncut_arm1_gate_stage.py --md5 experiments/uncut_arm1_ckpt_md5.json --src '<dir>/<name>-{arm}-seed{seed}.pt'
                           --inputs-group <gate_inputs_arm1/inputs_group> --split <split.json> --dest <IN root>
                           [--stem uncut-arm1] [--label descriptive-smoke]

--md5 holds {"<arm>|s<seed>": md5} for arm in gnn_eng, twin_eng, mlp_same (S7's training log). --src is a pattern with {arm}
(gnn-eng, twin-eng, mlp-same) and {seed}. Every listed checkpoint must exist with its .contract.json beside it and the md5 the
file gives, no two keys may share an md5, and every contract must carry physics_env.warmth_physics == node_disk_v2 (the bar's); each is copied to <dest>/models/<stem>-{arm}-seed{seed}.pt (+ .contract.json), the name the gate serves for
ra_<arm>_arm1, and its md5 is checked again after the copy. wf1_moderate / wf1_heavy are symlinked from --inputs-group; the
split is copied under its own name (pass its basename as R1A_SPLIT). <dest> must not exist. Writes <dest>/STAGE.json.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

ARMS = ("gnn_eng", "twin_eng", "mlp_same")
WARMTH = "node_disk_v2"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--md5", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--inputs-group", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--stem", default="uncut-arm1")
    ap.add_argument("--label", default="")
    a = ap.parse_args(argv)
    if os.path.exists(a.dest):
        raise SystemExit(f"FAIL LOUD: {a.dest} exists; stage into a fresh directory")
    want = json.load(open(a.md5))
    unknown = [k for k in want if k.split("|")[0] not in ARMS]
    if unknown:
        raise SystemExit(f"FAIL LOUD: {a.md5} has keys outside {ARMS}: {unknown}")
    dup = {m for m in want.values() if list(want.values()).count(m) > 1}
    if dup:
        raise SystemExit(f"FAIL LOUD: {a.md5} lists one md5 under several keys ({sorted(k for k, m in want.items() if m in dup)}); "
                         "a copy slip would make the two-seed mean a single seed")
    for r in ("wf1_moderate", "wf1_heavy"):
        if not os.path.isdir(os.path.join(a.inputs_group, r)):
            raise SystemExit(f"FAIL LOUD: {a.inputs_group}/{r} missing")
    plan = []
    for key, m in sorted(want.items()):
        arm, seed = key.split("|")
        seed = int(seed.lstrip("s"))
        src = a.src.format(arm=arm.replace("_", "-"), seed=seed)
        side = src[:-3] + ".contract.json"
        for p in (src, side):
            if not os.path.isfile(p):
                raise SystemExit(f"FAIL LOUD: {key}: {p} missing")
        if md5(src) != m:
            raise SystemExit(f"FAIL LOUD: {key}: {src} md5 {md5(src)} is not the training log's {m}")
        warmth = (json.load(open(side)).get("physics_env") or {}).get("warmth_physics")
        if warmth != WARMTH:
            raise SystemExit(f"FAIL LOUD: {key}: {side} physics_env.warmth_physics is {warmth!r}, not the gate's {WARMTH!r}")
        plan.append((key, m, src, side, f"{a.stem}-{arm.replace('_', '-')}-seed{seed}"))
    os.makedirs(os.path.join(a.dest, "models"))
    for r in ("wf1_moderate", "wf1_heavy"):
        os.symlink(os.path.realpath(os.path.join(a.inputs_group, r)), os.path.join(a.dest, r))
    shutil.copy2(a.split, os.path.join(a.dest, os.path.basename(a.split)))
    staged = {}
    for key, m, src, side, name in plan:
        pt = os.path.join(a.dest, "models", name + ".pt")
        shutil.copy2(src, pt)
        shutil.copy2(side, pt[:-3] + ".contract.json")
        if md5(pt) != m:
            raise SystemExit(f"FAIL LOUD: {key}: the copy {pt} does not have md5 {m}")
        staged[key] = {"src": src, "md5": m, "staged": pt, "contract_md5": md5(pt[:-3] + ".contract.json")}
    out = {"label": a.label, "md5_file": os.path.abspath(a.md5), "md5_file_md5": md5(a.md5), "inputs_group": os.path.realpath(a.inputs_group),
           "split": os.path.basename(a.split), "split_md5": md5(a.split), "stem": a.stem, "checkpoints": staged}
    json.dump(out, open(os.path.join(a.dest, "STAGE.json"), "w"), indent=1)
    print(json.dumps({k: v["staged"] for k, v in staged.items()}, indent=1))
    return out


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
