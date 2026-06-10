#!/usr/bin/env python3
"""Generate tiered-hub (degree_skewed_core) JSON configs for GNN vs MLP sweeps."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = PROJECT_ROOT / "simulation_data" / "normal_sim_sweeps" / "atomic21_skew_configs" / "default_20_20_degree_skew.json"

DEFAULT_K_CORE_VALUES = (2, 4, 6, 8)
DEFAULT_HUB_SEEKER_FRACTIONS = (0.3, 0.5, 0.8)

CONTROL_CONFIGS = (
    ("default_20_20_p50", "simulation_data/space_with_network.json"),
    (
        "05_sparse_40_40_p25",
        "simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/05_sparse_40_40_p25.json",
    ),
)

POLICY_SETS = {
    "all": (
        ("gnn_dim22", "gpu"),
        ("gnn_atomic21", "gpu"),
        ("mlp_dim22", "gpu"),
        ("mlp_atomic21", "gpu"),
        ("knative", "cpu"),
    ),
    "dim22": (
        ("gnn_dim22", "gpu"),
        ("mlp_dim22", "gpu"),
    ),
}


def seek_suffix(fraction: float) -> str:
    return f"seek{int(round(fraction * 100)):02d}"


def config_name(k_core: int, hub_seeker_fraction: float) -> str:
    return f"hub_k{k_core}_{seek_suffix(hub_seeker_fraction)}"


def build_config(
    base: dict,
    k_core: int,
    hub_seeker_fraction: float,
    seed: int,
    *,
    latency_core_ms: float,
    latency_periphery_ms: float,
) -> dict:
    cfg = deepcopy(base)
    cfg.setdefault("network", {}).setdefault("topology", {})
    topo = cfg["network"]["topology"]
    topo.update(
        {
            "type": "degree_skewed_core",
            "k_core": k_core,
            "hub_seeker_fraction": hub_seeker_fraction,
            "p_core": 0.95,
            "p_periphery": 0.15,
            "latency_core_ms": latency_core_ms,
            "latency_periphery_ms": latency_periphery_ms,
            "seed": seed,
        }
    )
    cfg.setdefault("nodes", {})
    cfg["nodes"]["client_nodes"] = {"count": 20}
    cfg["nodes"]["server_nodes"] = {"count": 20}
    return cfg


def parse_k_core_values(raw: str) -> tuple[int, ...]:
    values = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    if not values:
        raise ValueError("k-core list must not be empty")
    if any(k < 1 for k in values):
        raise ValueError(f"k_core values must be >= 1, got {values}")
    return values


def parse_seek_fractions(raw: str) -> tuple[float, ...]:
    values = tuple(float(x.strip()) for x in raw.split(",") if x.strip())
    if not values:
        raise ValueError("seek-fraction list must not be empty")
    if any(v <= 0 or v >= 1 for v in values):
        raise ValueError(f"seek fractions must be in (0, 1), got {values}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "simulation_data" / "normal_sim_sweeps" / "tiered_hub_gnn_mlp_20260610" / "configs",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--k-core-values",
        type=str,
        default=",".join(str(k) for k in DEFAULT_K_CORE_VALUES),
        help="Comma-separated hub counts (default: 2,4,6,8)",
    )
    parser.add_argument(
        "--seek-fractions",
        type=str,
        default=",".join(str(f) for f in DEFAULT_HUB_SEEKER_FRACTIONS),
        help="Comma-separated hub seeker fractions (default: 0.3,0.5,0.8)",
    )
    parser.add_argument(
        "--latency-core-ms",
        type=float,
        default=5.0,
        help="Client↔core-hub link latency in ms (default: 5.0)",
    )
    parser.add_argument(
        "--latency-periphery-ms",
        type=float,
        default=5.0,
        help="Client↔periphery link latency in ms (default: 5.0, same as core for backward compat)",
    )
    parser.add_argument(
        "--with-controls",
        action="store_true",
        help="Add default_20_20_p50 and 05_sparse_40_40_p25 control configs",
    )
    parser.add_argument(
        "--policies",
        choices=sorted(POLICY_SETS),
        default="all",
        help="Policy manifest set (dim22 = gnn_dim22 + mlp_dim22 only)",
    )
    args = parser.parse_args()

    if args.latency_core_ms <= 0 or args.latency_periphery_ms <= 0:
        raise ValueError("latencies must be > 0 ms")

    k_core_values = parse_k_core_values(args.k_core_values)
    seek_fractions = parse_seek_fractions(args.seek_fractions)

    if not BASE_CONFIG.is_file():
        raise FileNotFoundError(f"Missing base config: {BASE_CONFIG}")

    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_root = args.out_dir.resolve()
    project_root = PROJECT_ROOT.resolve()

    policies = POLICY_SETS[args.policies]
    config_entries: list[tuple[str, str]] = []

    def config_rel_path(path: Path) -> str:
        try:
            return path.relative_to(project_root).as_posix()
        except ValueError:
            return path.as_posix()

    for k_core in k_core_values:
        if k_core > 20:
            raise ValueError(f"k_core={k_core} exceeds server count (20)")
        for hub_frac in seek_fractions:
            name = config_name(k_core, hub_frac)
            cfg = build_config(
                base,
                k_core,
                hub_frac,
                args.seed,
                latency_core_ms=args.latency_core_ms,
                latency_periphery_ms=args.latency_periphery_ms,
            )
            out_path = out_root / f"{name}.json"
            out_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
            rel = config_rel_path(out_path)
            config_entries.append((name, rel))

    if args.with_controls:
        for name, rel in CONTROL_CONFIGS:
            full = PROJECT_ROOT / rel
            if not full.is_file():
                raise FileNotFoundError(f"Missing control config: {full}")
            config_entries.append((name, rel))

    manifest_lines = ["policy\tconfig_name\tconfig_path\tpartition"]
    for name, rel in config_entries:
        for policy, partition in policies:
            manifest_lines.append(f"{policy}\t{name}\t{rel}\t{partition}")

    manifest_name = "jobs_dim22.tsv" if args.policies == "dim22" else "jobs.tsv"
    manifest_path = out_root / manifest_name
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    gpu_jobs = sum(1 for line in manifest_lines[1:] if line.endswith("\tgpu"))
    cpu_jobs = sum(1 for line in manifest_lines[1:] if line.endswith("\tcpu"))
    hub_count = len(k_core_values) * len(seek_fractions)
    control_count = len(CONTROL_CONFIGS) if args.with_controls else 0
    print(f"Wrote {hub_count} hub configs (+{control_count} controls) to {out_root}")
    print(
        f"Latencies: core={args.latency_core_ms}ms periphery={args.latency_periphery_ms}ms"
    )
    print(
        f"Jobs manifest: {manifest_path} ({gpu_jobs} gpu + {cpu_jobs} cpu = {gpu_jobs + cpu_jobs} total)"
    )


if __name__ == "__main__":
    main()
