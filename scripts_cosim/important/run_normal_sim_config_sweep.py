#!/usr/bin/env python3
"""
Run normal simulation sweeps (not co-sim) with varied infrastructure configs.

This script:
1) Clones simulation_data/space_with_network.json into per-scenario configs
2) Tweaks topology/size knobs (clients, servers, connectivity, preinit)
3) Calls scripts_cosim/run_simulation.py for each scenario
"""

import argparse
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = PROJECT_ROOT / "simulation_data" / "space_with_network.json"
DEFAULT_WORKLOAD = PROJECT_ROOT / "data" / "nofs-ids" / "traces" / "workload-100-100.json"


@dataclass
class Scenario:
    name: str
    clients: int
    servers: int
    connection_probability: float
    client_preinit: float
    server_preinit: float


SCENARIOS: List[Scenario] = [
    Scenario("balanced_30_30_p35", 30, 30, 0.35, 0.4, 0.6),
    Scenario("balanced_40_40_p50", 40, 40, 0.50, 0.4, 0.6),
    Scenario("balanced_50_50_p60", 50, 50, 0.60, 0.4, 0.6),
    Scenario("client_heavy_50_35_p50", 50, 35, 0.50, 0.4, 0.6),
    Scenario("server_heavy_35_50_p50", 35, 50, 0.50, 0.4, 0.6),
    Scenario("sparse_40_40_p25", 40, 40, 0.25, 0.4, 0.6),
]


def _load_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _save_json(path: Path, payload: Dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _build_config(base: Dict, scenario: Scenario) -> Dict:
    cfg = json.loads(json.dumps(base))
    cfg["nodes"]["client_nodes"]["count"] = scenario.clients
    cfg["nodes"]["server_nodes"]["count"] = scenario.servers
    cfg["network"]["topology"]["connection_probability"] = scenario.connection_probability
    cfg.setdefault("preinit", {})
    cfg["preinit"]["client_percentage"] = scenario.client_preinit
    cfg["preinit"]["server_percentage"] = scenario.server_preinit
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run normal simulation config sweep")
    parser.add_argument(
        "--policy",
        choices=[
            "knative_network",
            "roundrobin",
            "random_network",
            "herocache_network",
            "offload_network",
            "gnn",
        ],
        default="knative_network",
        help="Policy flag passed to scripts_cosim/run_simulation.py",
    )
    parser.add_argument("--timeout", type=int, default=3600, help="Per-run timeout (seconds)")
    parser.add_argument("--seed", type=int, default=42, help="Simulation seed")
    parser.add_argument("--workload", type=str, default=str(DEFAULT_WORKLOAD), help="Workload JSON path")
    parser.add_argument("--max-scenarios", type=int, default=6, help="Run first N scenarios")
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Reuse an existing sweep root directory (for resume)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip scenarios whose output already exists",
    )
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = (
        Path(args.output_root)
        if args.output_root
        else PROJECT_ROOT / "simulation_data" / "normal_sim_sweeps" / f"{args.policy}_{ts}"
    )
    cfg_dir = run_root / "configs"
    out_dir = run_root / "results"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    progress_log = PROJECT_ROOT / "logs" / f"normal_sim_sweep_{args.policy}_{ts}.log"
    base = _load_json(BASE_CONFIG)
    selected = SCENARIOS[: max(1, args.max_scenarios)]

    print("=== Normal Simulation Config Sweep ===")
    print(f"Policy: {args.policy}")
    print(f"Workload: {args.workload}")
    print(f"Scenarios: {len(selected)}")
    print(f"Output root: {run_root}")

    for idx, s in enumerate(selected):
        config_payload = _build_config(base, s)
        config_path = cfg_dir / f"{idx:02d}_{s.name}.json"
        output_path = out_dir / f"{idx:02d}_{s.name}.json"
        if not config_path.exists():
            _save_json(config_path, config_payload)

        if args.resume and output_path.exists():
            line = (
                f"{idx:02d} {s.name} SKIPPED_EXISTS 0.0s "
                f"clients={s.clients} servers={s.servers} p={s.connection_probability}\n"
            )
            with open(progress_log, "a") as f:
                f.write(line)
            print(line.strip())
            continue

        # See the HEROSIM_PY note in scripts_cosim/run_simulation.py: an argv list is
        # invisible to a grep for the shell spelling, which is how this call site kept
        # re-spawning under pipenv after the 2026-08-21 sweep.
        cmd = shlex.split(os.environ.get("HEROSIM_PY") or "pipenv run python3") + [
            "scripts_cosim/run_simulation.py",
            f"--{args.policy}",
            "--config",
            str(config_path),
            "--workload",
            str(args.workload),
            "--output",
            str(output_path),
            "--timeout",
            str(args.timeout),
            "--seed",
            str(args.seed),
        ]

        started = time.time()
        status = "SUCCESS"
        try:
            subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
        except subprocess.CalledProcessError as exc:
            status = f"FAILED(exit={exc.returncode})"
        elapsed = time.time() - started

        line = (
            f"{idx:02d} {s.name} {status} {elapsed:.1f}s "
            f"clients={s.clients} servers={s.servers} p={s.connection_probability}\n"
        )
        with open(progress_log, "a") as f:
            f.write(line)
        print(line.strip())

    print(f"\nProgress log: {progress_log}")
    print(f"Results root: {run_root}")


if __name__ == "__main__":
    main()
