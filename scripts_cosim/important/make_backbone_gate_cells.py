"""Mint parity-exact backbone gate cells at an arbitrary backbone configuration.

The `a1_backbone_bw1p5` cells were built before `network.backbone.rng_stream` existed, so
they carry the `legacy_v0` jitter stream and their preflight needs
`--allow-backbone-latency-divergence` on every cell with a non-empty replica-reachability
repair set. Cells minted here declare `rng_stream: independent_v1`, which draws the jitter
from `Random(f"{seed}:backbone_v1")` instead of the stream the repair already consumed —
so the corpus-side and live-side generators agree exactly and the preflight passes with no
waiver.

Takes an existing non-backbone cell collection as the source of truth for topology (so a
new backbone config differs from the recorded arms in the backbone block *only*) and
writes `configs/<cell>.json` + `cell_infrastructure/<cell>/{space_with_network,
infrastructure}.json` in the layout the live-gate runner expects.

Usage:
  make_backbone_gate_cells.py --sweep-dir simulation_data/normal_sim_sweeps/bb_core8 \\
      --n-core 8 --bandwidth-mbps 1.5
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.generate_infrastructure import generate_deterministic_infrastructure  # noqa: E402

DEFAULT_SOURCE = (
    REPO / "simulation_data/normal_sim_sweeps/full_corpus_siv1_live_gate_20260820"
)
DEFAULT_SIM_INPUT = REPO / "data/nofs-ids"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=Path, required=True)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                    help="non-backbone cell collection to derive topology from")
    ap.add_argument("--sim-input", type=Path, default=DEFAULT_SIM_INPUT)
    ap.add_argument("--n-core", type=int, required=True)
    ap.add_argument("--bandwidth-mbps", type=float, required=True)
    ap.add_argument("--attach-degree", type=int, default=1)
    ap.add_argument("--chord-count", type=int, default=0)
    ap.add_argument("--core-link-latency-ms", type=float, default=4.0)
    ap.add_argument("--access-link-latency-ms", type=float, default=20.0)
    ap.add_argument("--rng-stream", choices=("independent_v1", "legacy_v0"),
                    default="independent_v1")
    args = ap.parse_args()

    src_cfg_dir = args.source / "configs"
    if not src_cfg_dir.is_dir():
        raise SystemExit(f"FAIL LOUD: no configs under {args.source}")
    src_configs = sorted(src_cfg_dir.glob("*.json"))
    if not src_configs:
        raise SystemExit(f"FAIL LOUD: no cell configs in {src_cfg_dir}")

    cfg_out = args.sweep_dir / "configs"
    infra_out = args.sweep_dir / "cell_infrastructure"
    cfg_out.mkdir(parents=True, exist_ok=True)
    infra_out.mkdir(parents=True, exist_ok=True)

    backbone = {
        "n_core": args.n_core,
        "attach_degree": args.attach_degree,
        "chord_count": args.chord_count,
        "core_link_latency_ms": args.core_link_latency_ms,
        "access_link_latency_ms": args.access_link_latency_ms,
        "bandwidth_mbps": args.bandwidth_mbps,
        "rng_stream": args.rng_stream,
    }
    print(f"backbone: {backbone}")

    for src in src_configs:
        cell = src.stem
        cfg = json.loads(src.read_text())
        seed = cfg.get("network", {}).get("topology", {}).get("seed")
        if seed is None:
            raise SystemExit(f"FAIL LOUD: {src} declares no network.topology.seed")
        if cfg.get("network", {}).get("backbone"):
            raise SystemExit(
                f"FAIL LOUD: {src} already declares a backbone — the source collection "
                "must be backbone-free so the new cells differ in that block only"
            )
        cfg["network"]["backbone"] = dict(backbone)

        (cfg_out / f"{cell}.json").write_text(json.dumps(cfg, indent=2))
        cell_dir = infra_out / cell
        cell_dir.mkdir(parents=True, exist_ok=True)
        space = cell_dir / "space_with_network.json"
        shutil.copyfile(cfg_out / f"{cell}.json", space)
        generate_deterministic_infrastructure(
            str(space), args.sim_input, str(cell_dir / "infrastructure.json"), int(seed)
        )
        print(f"  minted {cell} (seed={seed})")

    print(f"\nminted {len(src_configs)} cells under {args.sweep_dir}")
    print("Verify with (NO waiver flag should be needed for rng_stream=independent_v1):")
    print(f"  verify_live_infra_parity.py "
          f"{' '.join('--dataset ' + str(infra_out / c.stem) for c in src_configs)} -v")
    return 0


if __name__ == "__main__":
    sys.exit(main())
