#!/usr/bin/env python3
"""uncut_groups_s0_v1: derive the whole-group gate inputs from minted inputs (docs/lineages/uncut_groups_s0_v1.md, ruling C).

Copies <src> to <dst> byte for byte except each cell cfg's scheduler.batch_size, which goes from the minted 10 to "group"
(served by GNNOrchestrator.apply_cfg_batch_size as one batch per peer group). Writes an md5 manifest of every file in both
trees (<tree>/MD5SUMS, sorted relative paths) so the derived tree can be checked against the minted one.

  uncut_groups_group_inputs.py --src <gate_inputs>/inputs --dst <gate_inputs>/inputs_group
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

FROM, TO = 10, "group"


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(root: str) -> list:
    rows = []
    for d, _subdirs, files in os.walk(root):
        for f in files:
            rel = os.path.relpath(os.path.join(d, f), root)
            if rel not in ("MD5SUMS", "MD5SUMS.partial"):
                rows.append(f"{md5(os.path.join(root, rel))}  {rel}")
    return sorted(rows, key=lambda r: r.split("  ", 1)[1])


def write_manifest(root: str) -> None:
    rows = manifest(root)
    tmp = os.path.join(root, "MD5SUMS.partial")
    with open(tmp, "w") as fh:
        fh.write("\n".join(rows) + "\n")
    os.replace(tmp, os.path.join(root, "MD5SUMS"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.dst):
        raise SystemExit(f"FAIL LOUD: {a.dst} exists; derived inputs are frozen once written")
    tmp = a.dst + ".partial"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    shutil.copytree(a.src, tmp, ignore=shutil.ignore_patterns("MD5SUMS"))
    n = 0
    for d, _subdirs, files in os.walk(tmp):
        if os.path.basename(d) != "cfg":
            continue
        for f in sorted(files):
            p = os.path.join(d, f)
            cfg = json.load(open(p))
            got = (cfg.get("scheduler") or {}).get("batch_size")
            if got != FROM:
                raise SystemExit(f"FAIL LOUD: {p} has scheduler.batch_size {got!r}, want the minted {FROM}")
            cfg["scheduler"]["batch_size"] = TO
            with open(p, "w") as fh:
                json.dump(cfg, fh, indent=2)
            back = json.load(open(p))
            orig = json.load(open(os.path.join(a.src, os.path.relpath(p, tmp))))
            back["scheduler"]["batch_size"] = FROM
            if back != orig:
                raise SystemExit(f"FAIL LOUD: {p} differs from its source beyond scheduler.batch_size")
            n += 1
    if n == 0:
        raise SystemExit(f"FAIL LOUD: no cfg/*.json under {a.src}")
    os.replace(tmp, a.dst)
    write_manifest(a.src)
    write_manifest(a.dst)
    src_rows = dict(r.split("  ", 1)[::-1] for r in manifest(a.src))
    dst_rows = dict(r.split("  ", 1)[::-1] for r in manifest(a.dst))
    changed = sorted(k for k in src_rows if dst_rows.get(k) != src_rows[k])
    if set(src_rows) != set(dst_rows) or any(os.sep + "cfg" + os.sep not in os.sep + k for k in changed):
        raise SystemExit(f"FAIL LOUD: derived tree differs outside the cfg files: {changed[:5]}")
    print(json.dumps({"cfgs_rewritten": n, "files": len(src_rows), "changed": len(changed), "batch_size": [FROM, TO]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
