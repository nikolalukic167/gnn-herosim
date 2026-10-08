#!/usr/bin/env python3
"""workload_fix_v1 -- build gate inputs for one workload stage at arbitrary load rungs.

Follows the ladder protocol every earlier rung used, in the same order and with the same code:
  1. the x1 scattered grounded windows (grounded_g<i>_n50000.json) get the stage's payload sampler
     (`wf1_v1`: workload_payloads.resample_peer_exchange, seeded by the window's mint seed; `legacy`: untouched);
  2. cd_gap_v1_build_b.py scales every timestamp and the cells' batch_timeout by --rung factor (< 1 = faster);
  3. client_local_v1_single_origin.py gives every peer group one origin client.
`legacy` therefore reproduces the existing client_local_v1/wl_so/grounded_x* files byte for byte
(`--verify-against`). A rung is a multiplier m = 1 / factor on the x1 arrival rate.

  workload_fix_v1_build.py --grounded-wl <x1 wl dir> --cfg-dir <cfg dir> --topologies 9601 9602 \\
      --rung m2p0=0.5 --payload-sampler wf1_v1 --out <dir>

writes <dir>/wf1_<tag>/{cfg,wl} per rung and <dir>/manifest.json. The gate driver reads rungs through
WF1_RUNGS="<tag>,<tag>" (fresh_topo_burst_v1_gate.py, phases wf1 / wf1cal).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts_cosim"))

from src.placement.workload_payloads import (  # noqa: E402
    PAYLOAD_SAMPLERS, payload_sampler_meta, require_sampler, resample_peer_exchange,
)

WINDOWS = tuple(f"grounded_g{i}_n50000.json" for i in range(4))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def apply_sampler(src: Path, dst: Path, sampler: str) -> dict:
    """Copy the four x1 windows to ``dst``, resampling payloads when the sampler asks for it."""
    require_sampler(sampler)
    dst.mkdir(parents=True, exist_ok=True)
    seeds = {}
    for name in WINDOWS:
        wl = json.loads((src / name).read_text())
        seed = int(wl["grounded_workload_v1"]["seed"])
        seeds[name] = seed
        if sampler == "legacy":
            shutil.copyfile(src / name, dst / name)  # byte copy: build_b records the source's sha256
            continue
        wl["peer_exchange"] = resample_peer_exchange(wl["peer_exchange"], seed)
        wl["workload_fix_v1"] = {**payload_sampler_meta(), "seed": seed, "source_sha256": sha256(src / name)}
        with open(dst / name, "w") as fh:
            json.dump(wl, fh)
    return seeds


def fix_batch_timeout(cfg_dir: Path, seconds: float) -> int:
    """workload_fix_v1 amendment WB: hold ``scheduler.batch_timeout`` at ``seconds`` on a rung whose timestamps were
    scaled. The ladder value stays on record next to the fixed one. Returns the number of cells rewritten."""
    if seconds <= 0:
        raise SystemExit(f"FAIL LOUD: --batch-timeout-fixed must be > 0, got {seconds}")
    n = 0
    for path in sorted(cfg_dir.glob("cc40s*.json")):
        cfg = json.loads(path.read_text())
        ladder = float(cfg["scheduler"]["batch_timeout"])
        cfg["scheduler"]["batch_timeout"] = float(seconds)
        cfg["workload_fix_v1_batch_timeout"] = {"fixed_s": float(seconds), "ladder_value_s": ladder}
        with open(path, "w") as fh:
            json.dump(cfg, fh, indent=1)
        n += 1
    return n


def build_rung(x1_dir: Path, cfg_dir: Path, out: Path, tag: str, factor: float, topologies: list,
               batch_timeout_fixed: float = None) -> Path:
    rung_dir = out / f"wf1_{tag}"
    if rung_dir.exists():
        raise SystemExit(f"FAIL LOUD: {rung_dir} exists; a built rung is frozen")
    scattered = out / f"_scattered_{tag}"
    src = out / f"_src_{tag}"
    (src).mkdir(parents=True)
    os.symlink(x1_dir.resolve(), src / "wl")
    os.symlink(cfg_dir.resolve(), src / "cfg")
    subprocess.run([sys.executable, str(ROOT / "scripts_cosim" / "cd_gap_v1_build_b.py"), "--src", str(src),
                    "--out", str(scattered), "--factor", repr(factor), "--topologies", *map(str, topologies)],
                   check=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)})
    rung_dir.mkdir()
    subprocess.run([sys.executable, str(ROOT / "scripts_cosim" / "client_local_v1_single_origin.py"),
                    str(scattered / "wl"), str(rung_dir / "wl")], check=True, cwd=ROOT)
    shutil.copytree(scattered / "cfg", rung_dir / "cfg")
    if batch_timeout_fixed is not None:
        fix_batch_timeout(rung_dir / "cfg", batch_timeout_fixed)
    shutil.rmtree(scattered)
    shutil.rmtree(src)
    return rung_dir


def verify_against(rung_dir: Path, reference_wl_dir: Path) -> dict:
    result = {}
    for name in WINDOWS:
        a, b = sha256(rung_dir / "wl" / name), sha256(reference_wl_dir / name)
        result[name] = {"built": a, "reference": b, "identical": a == b}
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grounded-wl", type=Path, required=True, help="x1 scattered grounded windows")
    ap.add_argument("--cfg-dir", type=Path, required=True)
    ap.add_argument("--topologies", type=int, nargs="+", required=True)
    ap.add_argument("--rung", action="append", required=True, metavar="TAG=FACTOR")
    ap.add_argument("--payload-sampler", choices=PAYLOAD_SAMPLERS, default="legacy")
    ap.add_argument("--batch-timeout-fixed", type=float, default=None, metavar="SECONDS",
                    help="amendment WB: keep scheduler.batch_timeout at this value on every rung "
                         "(timestamps still scale); absent = the ladder protocol's scaled value")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--verify-against", type=Path, help="existing single-origin rung wl dir; its factor must be the only --rung")
    a = ap.parse_args()
    rungs = {}
    for spec in a.rung:
        tag, _, f = spec.partition("=")
        if not tag or not f or float(f) <= 0:
            raise SystemExit(f"FAIL LOUD: --rung {spec!r}; expected TAG=FACTOR with FACTOR > 0")
        rungs[tag] = float(f)
    if a.verify_against and len(rungs) != 1:
        raise SystemExit("FAIL LOUD: --verify-against needs exactly one --rung")
    a.out.mkdir(parents=True, exist_ok=True)
    x1 = a.out / "_x1_windows"
    if x1.exists():
        shutil.rmtree(x1)
    seeds = apply_sampler(a.grounded_wl, x1, a.payload_sampler)
    manifest = {"payload_sampler": a.payload_sampler, "batch_timeout_fixed_s": a.batch_timeout_fixed, "window_seeds": seeds, "topologies": a.topologies, "rungs": {}}
    for tag, factor in rungs.items():
        d = build_rung(x1, a.cfg_dir, a.out, tag, factor, a.topologies, a.batch_timeout_fixed)
        manifest["rungs"][tag] = {"factor": factor, "multiplier": 1.0 / factor,
                                  "wl_sha256": {n: sha256(d / "wl" / n) for n in WINDOWS}}
        if a.verify_against:
            v = verify_against(d, a.verify_against)
            manifest["rungs"][tag]["verify_against"] = v
            if not all(r["identical"] for r in v.values()):
                (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
                raise SystemExit(f"FAIL LOUD: built files differ from {a.verify_against}: {v}")
        print(f"built {d} (factor {factor}, m = {1.0 / factor:.4g})")
    shutil.rmtree(x1)
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
