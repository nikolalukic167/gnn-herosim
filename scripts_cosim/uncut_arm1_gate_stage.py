"""Stage an uncut_groups_s0_v1 learned-gate inputs root (Arm 1, Arm 2, Arm 2-K0). Checkpoints enter only through the md5 file.

  uncut_arm1_gate_stage.py --md5 experiments/uncut_arm1_ckpt_md5.json --src '<dir>/<name>-{arm}-seed{seed}.pt'
                           --inputs-group <gate_inputs_arm1/inputs_group> --split <split.json> --dest <IN root>
                           [--stem uncut-arm1] [--arms gnn_eng,twin_eng,mlp_same] [--label descriptive-smoke]

--md5 holds {"<arm>|s<seed>": md5} (S7's training log, which may grow arm by arm); only the --arms keys are staged, every --arms arm must
have a key, and a key naming no known arm fails. It may name its checkpoint directory as "_ckpt_dir" (e.g. a relaunch's
/share/nikola.lukic/uncut_arm1/models_relaunch_x): --src may then use {dir}, and every resolved checkpoint must sit in that directory. Arm 2 is
--arms gnn_eng,twin_eng --stem uncut-arm2, Arm 2-K0 the same with --stem uncut-arm2k0 (the gate's R1A_ARM2_STEM / R1A_ARM2K0_STEM), Arm 1b
--arms gnn_eng,twin_eng --stem uncut-arm1b with its own md5 file. --src is a pattern with {arm}
(gnn-eng, twin-eng, mlp-same) and {seed}. Every listed checkpoint must exist with its .contract.json beside it and the md5 the
file gives, no two keys may share an md5, and every contract must carry physics_env.warmth_physics == node_disk_v2 (the bar's) and batched_teacher_forced true under --stem uncut-arm1b,
absent or false under any other stem (serving ignores the key, so this is where a 1b checkpoint is told apart); each is copied to <dest>/models/<stem>-{arm}-seed{seed}.pt (+ .contract.json), the name the gate serves for
ra_<arm>_arm1 (ra_<arm>_arm2 / _arm2k0 under their stems), and its md5 is checked again after the copy. wf1_moderate / wf1_heavy are symlinked from --inputs-group; the
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
ARM1B_STEM, ARM1B_ARMS = "uncut-arm1b", ("gnn_eng", "twin_eng")


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
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--label", default="")
    a = ap.parse_args(argv)
    arms = tuple(a.arms.split(","))
    if not arms or set(arms) - set(ARMS):
        raise SystemExit(f"FAIL LOUD: --arms {a.arms} is not a subset of {ARMS}")
    if a.stem == ARM1B_STEM and set(arms) - set(ARM1B_ARMS):
        raise SystemExit(f"FAIL LOUD: Arm 1b ({ARM1B_STEM}) has only {ARM1B_ARMS}, not {a.arms}")
    if os.path.exists(a.dest):
        raise SystemExit(f"FAIL LOUD: {a.dest} exists; stage into a fresh directory")
    want = json.load(open(a.md5))
    meta = {k: want.pop(k) for k in [k for k in want if k.startswith("_")]}
    if set(meta) - {"_ckpt_dir"}:
        raise SystemExit(f"FAIL LOUD: {a.md5} has metadata keys {sorted(set(meta) - {'_ckpt_dir'})}; only _ckpt_dir is known")
    ckpt_dir = meta.get("_ckpt_dir")
    if ckpt_dir is not None and not (isinstance(ckpt_dir, str) and os.path.isdir(ckpt_dir)):
        raise SystemExit(f"FAIL LOUD: {a.md5} _ckpt_dir {ckpt_dir!r} is not a directory")
    if "{dir}" in a.src and ckpt_dir is None:
        raise SystemExit(f"FAIL LOUD: --src uses {{dir}} but {a.md5} names no _ckpt_dir")
    unknown = [k for k in want if k.split("|")[0] not in ARMS]
    if unknown:
        raise SystemExit(f"FAIL LOUD: {a.md5} has keys outside {ARMS}: {unknown}")
    absent = [x for x in arms if not any(k.split("|")[0] == x for k in want)]
    if absent:
        raise SystemExit(f"FAIL LOUD: {a.md5} has no checkpoint for --arms {absent}")
    dup = {m for m in want.values() if list(want.values()).count(m) > 1}
    if dup:
        raise SystemExit(f"FAIL LOUD: {a.md5} lists one md5 under several keys ({sorted(k for k, m in want.items() if m in dup)}); "
                         "a copy slip would make the two-seed mean a single seed")
    for r in ("wf1_moderate", "wf1_heavy"):
        if not os.path.isdir(os.path.join(a.inputs_group, r)):
            raise SystemExit(f"FAIL LOUD: {a.inputs_group}/{r} missing")
    plan = []
    for key, m in sorted((k, m) for k, m in want.items() if k.split("|")[0] in arms):
        arm, seed = key.split("|")
        seed = int(seed.lstrip("s"))
        src = a.src.format(arm=arm.replace("_", "-"), seed=seed, dir=ckpt_dir)
        if ckpt_dir is not None and os.path.dirname(os.path.realpath(src)) != os.path.realpath(ckpt_dir):
            raise SystemExit(f"FAIL LOUD: {key}: {src} is not in the md5 file's _ckpt_dir {ckpt_dir}")
        side = src[:-3] + ".contract.json"
        for p in (src, side):
            if not os.path.isfile(p):
                raise SystemExit(f"FAIL LOUD: {key}: {p} missing")
        if md5(src) != m:
            raise SystemExit(f"FAIL LOUD: {key}: {src} md5 {md5(src)} is not the training log's {m}")
        contract = json.load(open(side))
        warmth = (contract.get("physics_env") or {}).get("warmth_physics")
        if warmth != WARMTH:
            raise SystemExit(f"FAIL LOUD: {key}: {side} physics_env.warmth_physics is {warmth!r}, not the gate's {WARMTH!r}")
        btf = contract.get("batched_teacher_forced")
        if (btf is not True) if a.stem == ARM1B_STEM else (btf is not None and btf is not False):
            raise SystemExit(f"FAIL LOUD: {key}: {side} batched_teacher_forced is {btf!r}; --stem {a.stem} needs "
                             + ("true (Arm 1b)" if a.stem == ARM1B_STEM else "it absent or false"))
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
           "split": os.path.basename(a.split), "split_md5": md5(a.split), "stem": a.stem, "arms": list(arms),
           "ckpt_dir": ckpt_dir, "checkpoints": staged}
    json.dump(out, open(os.path.join(a.dest, "STAGE.json"), "w"), indent=1)
    print(json.dumps({k: v["staged"] for k, v in staged.items()}, indent=1))
    return out


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
