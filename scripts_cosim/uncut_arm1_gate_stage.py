"""Stage an uncut_groups_s0_v1 learned-gate inputs root (Arm 1, Arm 2, Arm 2-K0). Checkpoints enter only through the md5 file.

  uncut_arm1_gate_stage.py --md5 experiments/uncut_arm1_ckpt_md5.json --src '<dir>/<name>-{arm}-seed{seed}.pt'
                           --inputs-group <gate_inputs_arm1/inputs_group> --split <split.json> --dest <IN root>
                           [--stem uncut-arm1] [--arms gnn_eng,twin_eng,mlp_same] [--label descriptive-smoke]

--md5 holds {"<arm>|s<seed>": md5} (S7's training log, which may grow arm by arm); only the --arms keys are staged, every --arms arm must
have a key, and a key naming no known arm fails (S7's schema). "_ckpt_dir" names the checkpoint directory, one string or {"<arm>": dir}
(arms may come from different jobs): --src may then use {dir} (the arm's), and every resolved checkpoint must sit in it.
"_provenance" {"<arm>|s<seed>": {"job", "host", "cpu", ["wandb_id", "file"]}} names where each checkpoint trained; required under
--stem uncut-arm1 (the hedge rule, 4213ac4e), optional otherwise, and when present it must cover every staged key. An arm's seeds must
name one job (array tasks 871420_0 and 871420_3 are one job; 869803 and 871420 are two). A chained run (Arm 2 / K0, uncut_ckpt_md5_write.py)
gives "links": [{job, host, cpu}, ...] and a "submission" label instead; its unit is the submission (ruling 943a53fd), so an arm's seeds
must all be chained and share one label; Arm 1 (--stem uncut-arm1) refuses links. With "file", the resolved checkpoint must
have that basename, and --src may be omitted ({dir}/{file}). Arm 2 is
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
    ap.add_argument("--src", default="{dir}/{file}")
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
    if set(meta) - {"_ckpt_dir", "_provenance"}:
        raise SystemExit(f"FAIL LOUD: {a.md5} has metadata keys {sorted(set(meta) - {'_ckpt_dir', '_provenance'})}; only _ckpt_dir and _provenance are known")
    ckpt_dir = meta.get("_ckpt_dir")
    dirs = None if ckpt_dir is None else {x: ckpt_dir for x in arms} if isinstance(ckpt_dir, str) else ckpt_dir if isinstance(ckpt_dir, dict) else {}
    if dirs is not None and not all(isinstance(dirs.get(x), str) and os.path.isdir(dirs[x]) for x in arms):
        raise SystemExit(f"FAIL LOUD: {a.md5} _ckpt_dir {ckpt_dir!r} does not name an existing directory for every arm of {arms}")
    if "{dir}" in a.src and dirs is None:
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
    prov = meta.get("_provenance")
    if prov is None and a.stem == "uncut-arm1":
        raise SystemExit(f"FAIL LOUD: {a.md5} has no _provenance; Arm 1 is staged only with each checkpoint's job, host and CPU model (hedge rule)")
    if prov is not None:
        staged_keys = [k for k in want if k.split("|")[0] in arms]
        for k in staged_keys:
            e = prov.get(k) if isinstance(prov, dict) else None
            if not (isinstance(e, dict) and all(isinstance(e.get(f), str) and e.get(f).strip() for f in ("job", "host", "cpu"))):
                raise SystemExit(f"FAIL LOUD: {a.md5} _provenance[{k!r}] needs non-empty job, host and cpu; got {e!r}")
        for arm in arms:
            keys = [k for k in staged_keys if k.split("|")[0] == arm]
            chained = [k for k in keys if "links" in prov[k]]
            if chained:
                # chained runs (Arm 2 / K0, ruling 943a53fd): the unit is the submission label, not the Slurm job
                if a.stem == "uncut-arm1":
                    raise SystemExit(f"FAIL LOUD: {a.md5} gives {arm} chained links; Arm 1's hedge rule is one Slurm job per arm")
                if len(chained) != len(keys):
                    raise SystemExit(f"FAIL LOUD: {a.md5} mixes chained and single-job provenance within {arm}: {sorted(chained)} of {sorted(keys)}")
                for k in keys:
                    links, sub = prov[k]["links"], prov[k].get("submission")
                    if not (isinstance(links, list) and links and all(
                            isinstance(l, dict) and all(isinstance(l.get(f), str) and l[f].strip() for f in ("job", "host", "cpu"))
                            for l in links)):
                        raise SystemExit(f"FAIL LOUD: {a.md5} _provenance[{k!r}].links must be non-empty {{job, host, cpu}} entries; got {links!r}")
                    if not (isinstance(sub, str) and sub.strip()):
                        raise SystemExit(f"FAIL LOUD: {a.md5} _provenance[{k!r}] has links but no submission label")
                subs = sorted({prov[k]["submission"] for k in keys})
                if len(subs) > 1:
                    raise SystemExit(f"FAIL LOUD: {arm}'s seeds come from different submissions {subs}; an arm's seeds share one submission label")
                continue
            jobs = sorted({prov[k]["job"].split("_")[0] for k in keys})
            if len(jobs) > 1:
                raise SystemExit(f"FAIL LOUD: {arm}'s seeds come from different jobs {jobs}; the hedge rule never splits an arm's seeds")
    if "{file}" in a.src and prov is None:
        raise SystemExit(f"FAIL LOUD: --src uses {{file}} but {a.md5} has no _provenance")
    plan = []
    for key, m in sorted((k, m) for k, m in want.items() if k.split("|")[0] in arms):
        arm, seed = key.split("|")
        seed = int(seed.lstrip("s"))
        file = (prov or {}).get(key, {}).get("file")
        if "{file}" in a.src and not file:
            raise SystemExit(f"FAIL LOUD: {key}: --src uses {{file}} but _provenance names no file")
        src = a.src.format(arm=arm.replace("_", "-"), seed=seed, dir=None if dirs is None else dirs[arm], file=file)
        if dirs is not None and os.path.dirname(os.path.realpath(src)) != os.path.realpath(dirs[arm]):
            raise SystemExit(f"FAIL LOUD: {key}: {src} is not in the md5 file's _ckpt_dir {dirs[arm]}")
        if file and os.path.basename(src) != file:
            raise SystemExit(f"FAIL LOUD: {key}: {src} is not the _provenance file {file}")
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
           "ckpt_dir": ckpt_dir, "provenance": {k: prov[k] for k in staged} if prov is not None else None, "checkpoints": staged}
    json.dump(out, open(os.path.join(a.dest, "STAGE.json"), "w"), indent=1)
    print(json.dumps({k: v["staged"] for k, v in staged.items()}, indent=1))
    return out


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
