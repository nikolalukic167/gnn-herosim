#!/usr/bin/env python3
"""Verify that a live simulation regenerates the topology a co-sim dataset was built on.

`src/executesimulation.py` never loads a co-sim `infrastructure.json`. It regenerates the
topology from a space config plus a seed (`prepare_infrastructure_for_real_simulation`),
and nothing cross-checks the result against the corpus a checkpoint trained on. This tool
is that cross-check: it regenerates from a dataset's *own* `space_with_network.json` and
diffs the result against that dataset's `infrastructure.json`.

Two divergence classes, treated very differently:

``replica_reachability_repair``
    Expected and benign-by-construction. `generate_infrastructure.py` (step 2b,
    ~lines 712-778) adds client->server edges after replica placement so every client
    reaches at least ``MIN_REPLICA_SERVERS`` replica hosts per task type. A live run has
    no replica placements — it autoscales from zero — so it cannot and should not
    reproduce them. Every such edge is client<->server, absent from live, and carries
    exactly ``network.latency.base_latency``. Reported with its magnitude, not fatal.

anything else
    A real train/serve infrastructure mismatch. Fatal.

The live topology must always be a **subgraph** of the corpus topology: a live-only edge
means the two generators have genuinely diverged, and is fatal regardless of latency.

Usage:
    verify_live_infra_parity.py --dataset simulation_data/<coll>/ds_00000
    verify_live_infra_parity.py --collection simulation_data/<coll> --sample 20
    verify_live_infra_parity.py --dataset <ds> --config <other_space_config.json>

Run with the repo root on PYTHONPATH (see HANDOVER.md §0):
    PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH=$(pwd) \
      pipenv run python3 scripts_cosim/verify_live_infra_parity.py ...
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.executesimulation import prepare_infrastructure_for_real_simulation  # noqa: E402
from src.placement.topology_features import CLIENT_NODE_PREFIX  # noqa: E402
from src.placement.network_fabric import link_key  # noqa: E402

# Latency equality tolerance. Both sides come from the same seeded RNG and the same
# formula, so agreement is exact in practice; this only guards float round-tripping
# through JSON.
LATENCY_TOL = 1e-9

DEFAULT_SIM_INPUT = REPO_ROOT / "data" / "nofs-ids"


def is_client(node_name: str) -> bool:
    return str(node_name).startswith(CLIENT_NODE_PREFIX)


@dataclass
class ParityResult:
    """Outcome of comparing one live regeneration against one corpus dataset."""

    dataset: str
    ok: bool
    findings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.findings.append(msg)

    def note(self, msg: str) -> None:
        self.notes.append(msg)


def _load_json(path: Path) -> Any:
    with open(path, "r") as handle:
        return json.load(handle)


def regenerate_live_topology(
    space_config: Dict[str, Any],
    sim_input_path: Path,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the live infra path exactly as `executesimulation.main()` would.

    `seed=None` is deliberate and load-bearing: it makes
    `prepare_infrastructure_for_real_simulation` fall back to the space config's own
    `network.topology.seed`, which is the seed the corpus was generated with. Passing
    `--seed` on the live CLI *overrides* that seed and produces a different topology.
    """
    # The live path also seeds the global RNG before generating (executesimulation.py
    # :842-845). Mirror that so replica-independent draws line up.
    placement_seed = seed
    if placement_seed is None:
        placement_seed = space_config.get("network", {}).get("topology", {}).get("seed", 42)
    random.seed(placement_seed)

    # The generator is chatty; its progress output is not this tool's output.
    with contextlib.redirect_stdout(io.StringIO()):
        return prepare_infrastructure_for_real_simulation(
            space_config, seed=seed, sim_input_path=sim_input_path
        )


def _edge_set(maps: Dict[str, Dict[str, float]]) -> Dict[Tuple[str, str], float]:
    """Directed adjacency as an identity-keyed dict, never positional."""
    return {
        (src, dst): float(latency)
        for src, neighbours in maps.items()
        for dst, latency in neighbours.items()
    }


def _classify_corpus_only_edges(
    corpus_only: Dict[Tuple[str, str], float],
    base_latency: float,
    backbone: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    """Split corpus-only edges into the expected repair class and everything else.

    Without a backbone a repair edge is recognizable by its latency sitting at exactly
    `base_latency`. Under a backbone that signature is gone — `build_core_backbone`
    rewrites every logical edge's latency as the sum along its core route — so the same
    genuine repair edges look "unexplained". When the backbone is available we therefore
    check the *structural* property instead: the edge crosses tiers and its latency equals
    the path sum over the recorded route. That is a stricter test than the base_latency
    one, not a relaxation: an edge whose latency does not match its own route is still
    reported.
    """
    routes = (backbone or {}).get("routes") or {}
    links = (backbone or {}).get("links") or {}

    def route_latency(src: str, dst: str) -> Optional[float]:
        path = routes.get(src, {}).get(dst) or routes.get(dst, {}).get(src)
        if not path:
            return None
        total = 0.0
        for i in range(len(path) - 1):
            attrs = links.get(link_key(path[i], path[i + 1]))
            if attrs is None:
                return None
            total += float(attrs["latency"])
        return total

    repair: Dict[Tuple[str, str], float] = {}
    unexplained: Dict[Tuple[str, str], float] = {}
    for edge, latency in corpus_only.items():
        src, dst = edge
        crosses_tiers = is_client(src) != is_client(dst)
        at_base = math.isclose(latency, base_latency, rel_tol=0.0, abs_tol=LATENCY_TOL)
        on_route = False
        if backbone:
            expected = route_latency(src, dst)
            on_route = expected is not None and math.isclose(
                latency, expected, rel_tol=0.0, abs_tol=LATENCY_TOL
            )
        if crosses_tiers and (at_base or on_route):
            repair[edge] = latency
        else:
            unexplained[edge] = latency
    return repair, unexplained


def compare_topology(
    dataset_dir: Path,
    space_config: Dict[str, Any],
    corpus_infra: Dict[str, Any],
    sim_input_path: Path,
    seed: Optional[int] = None,
    allow_backbone_latency_divergence: bool = False,
) -> ParityResult:
    result = ParityResult(dataset=str(dataset_dir), ok=True)

    corpus_maps: Dict[str, Dict[str, float]] = corpus_infra["network_maps"]
    live_infra = regenerate_live_topology(space_config, sim_input_path, seed=seed)
    live_maps = {node["node_name"]: node.get("network_map", {}) for node in live_infra["nodes"]}

    # `build_core_backbone` draws its access-link jitter (generate_infrastructure.py:372-375)
    # from the SAME rng the replica-reachability repair has already consumed
    # (generate_infrastructure.py:768, `rng.shuffle`), and the backbone is overlaid *after*
    # that repair. A live run autoscales from zero, performs no repair, and therefore reaches
    # the backbone build at a different position in the rng stream -- so every access-link
    # latency diverges on exactly those cells whose repair set is non-empty. Measured on the
    # siv1 gate cells: p=0.35 and p=0.50 (repair 0/282 and 0/380) reproduce exactly, while
    # p=0.15/0.20/0.25 (repair 34/12/14) diverge on 100% of shared edges.
    #
    # This flag exists for a live-vs-live matched A/B (backbone on vs off, same cells, same
    # trace, same checkpoint), where the corpus-side artifact is only a preflight fixture and
    # both live arms are self-consistent with each other. It is NOT a general escape hatch:
    # it downgrades exactly two finding classes, and only when a backbone is present on BOTH
    # sides, so it cannot silently pass a backbone-less corpus.
    backbone_relaxation = allow_backbone_latency_divergence and (
        (corpus_infra.get("link_topology") or {}).get("links") is not None
        and (live_infra.get("link_topology") or {}).get("links") is not None
    )

    # --- node identity -----------------------------------------------------------
    corpus_names, live_names = set(corpus_maps), set(live_maps)
    if corpus_names != live_names:
        missing = sorted(corpus_names - live_names)[:8]
        extra = sorted(live_names - corpus_names)[:8]
        result.fail(
            f"node set mismatch: corpus has {len(corpus_names)} nodes, live has "
            f"{len(live_names)}; missing_from_live={missing} live_only={extra}"
        )
        return result  # every downstream comparison would be noise

    declared_clients = space_config["nodes"]["client_nodes"]["count"]
    declared_servers = space_config["nodes"]["server_nodes"]["count"]
    actual_clients = sum(1 for n in corpus_names if is_client(n))
    actual_servers = len(corpus_names) - actual_clients
    if (actual_clients, actual_servers) != (declared_clients, declared_servers):
        result.fail(
            f"space config declares {declared_clients} clients / {declared_servers} "
            f"servers but the corpus infrastructure has {actual_clients} / {actual_servers}"
        )

    # --- adjacency, by identity --------------------------------------------------
    corpus_edges = _edge_set(corpus_maps)
    live_edges = _edge_set(live_maps)

    live_only = {e: v for e, v in live_edges.items() if e not in corpus_edges}
    corpus_only = {e: v for e, v in corpus_edges.items() if e not in live_edges}

    base_latency = float(
        space_config.get("network", {}).get("latency", {}).get("base_latency", 0.1)
    )
    repair_edges, unexplained = _classify_corpus_only_edges(
        corpus_only, base_latency, backbone=corpus_infra.get("link_topology")
    )

    if live_only:
        sample = sorted(live_only)[:6]
        result.fail(
            f"{len(live_only)} live-only edge(s): the live topology is NOT a subgraph of "
            f"the corpus topology, so the two generators have diverged. e.g. {sample}"
        )
    if unexplained:
        sample = [(e, round(v, 6)) for e, v in sorted(unexplained.items())[:6]]
        msg = (
            f"{len(unexplained)} corpus-only edge(s) that do not match the "
            f"replica-reachability signature (client<->server at base_latency="
            f"{base_latency}). e.g. {sample}"
        )
        if backbone_relaxation:
            # Under a backbone these edges are real repair edges; they simply no longer sit
            # at exactly base_latency because their latency became a path sum over the core.
            result.notes.append(f"[backbone-relaxed] {msg}")
        else:
            result.fail(msg)

    # --- latency on shared edges -------------------------------------------------
    shared = [e for e in corpus_edges if e in live_edges]
    lat_mismatch = [
        e for e in shared
        if not math.isclose(corpus_edges[e], live_edges[e], rel_tol=0.0, abs_tol=LATENCY_TOL)
    ]
    if lat_mismatch:
        worst = max(lat_mismatch, key=lambda e: abs(corpus_edges[e] - live_edges[e]))
        msg = (
            f"{len(lat_mismatch)}/{len(shared)} shared edge(s) disagree on latency; "
            f"worst {worst}: corpus={corpus_edges[worst]:.9f} live={live_edges[worst]:.9f}"
        )
        if backbone_relaxation:
            result.notes.append(f"[backbone-relaxed] {msg} (access-link jitter rng offset)")
        else:
            result.fail(msg)

    # --- backbone ----------------------------------------------------------------
    corpus_links = (corpus_infra.get("link_topology") or {}).get("links")
    live_link_topology = live_infra.get("link_topology")
    live_links = (live_link_topology or {}).get("links")
    if (corpus_links is None) != (live_links is None):
        result.fail(
            f"link_topology presence differs: corpus="
            f"{'present' if corpus_links is not None else 'absent'} live="
            f"{'present' if live_links is not None else 'absent'}"
        )
    elif corpus_links is not None and len(corpus_links) != len(live_links):
        result.fail(
            f"backbone link count differs: corpus={len(corpus_links)} live={len(live_links)}"
        )

    # --- seed provenance ---------------------------------------------------------
    config_seed = space_config.get("network", {}).get("topology", {}).get("seed")
    infra_seed = (corpus_infra.get("metadata") or {}).get("seed")
    if infra_seed is not None and config_seed is not None and infra_seed != config_seed:
        result.fail(
            f"seed provenance mismatch: space config says {config_seed}, the corpus "
            f"infrastructure was generated with {infra_seed}"
        )

    if repair_edges:
        pct = 100.0 * len(repair_edges) / max(len(corpus_edges), 1)
        clients_touched = sorted({s if is_client(s) else d for s, d in repair_edges})
        result.note(
            f"replica_reachability_repair: {len(repair_edges)}/{len(corpus_edges)} "
            f"directed edges ({pct:.2f}%) exist in the corpus but not live, on "
            f"{len(clients_touched)} client(s) — expected, see module docstring"
        )

    result.stats = {
        "n_nodes": len(corpus_names),
        "n_clients": actual_clients,
        "n_servers": actual_servers,
        "topology_type": space_config.get("network", {}).get("topology", {}).get("type"),
        "connection_probability": space_config.get("network", {})
        .get("topology", {})
        .get("connection_probability"),
        "topology_seed": config_seed,
        "corpus_directed_edges": len(corpus_edges),
        "live_directed_edges": len(live_edges),
        "repair_edges": len(repair_edges),
        "repair_edge_fraction": (
            len(repair_edges) / len(corpus_edges) if corpus_edges else 0.0
        ),
        "unexplained_corpus_only_edges": len(unexplained),
        "live_only_edges": len(live_only),
        "latency_mismatches": len(lat_mismatch),
        "backbone_links": len(corpus_links) if corpus_links is not None else None,
    }
    return result


def verify_dataset(
    dataset_dir: Path,
    sim_input_path: Path,
    config_override: Optional[Path] = None,
    seed: Optional[int] = None,
    allow_backbone_latency_divergence: bool = False,
) -> ParityResult:
    infra_path = dataset_dir / "infrastructure.json"
    config_path = config_override or (dataset_dir / "space_with_network.json")
    for path in (infra_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing required file: {path}")
    return compare_topology(
        dataset_dir,
        _load_json(config_path),
        _load_json(infra_path),
        sim_input_path,
        seed=seed,
        allow_backbone_latency_divergence=allow_backbone_latency_divergence,
    )


def _print_result(result: ParityResult, verbose: bool) -> None:
    status = "PASS" if result.ok else "FAIL"
    name = Path(result.dataset).name
    stats = result.stats
    summary = ""
    if stats:
        summary = (
            f"  [{stats['n_clients']}c/{stats['n_servers']}s "
            f"{stats['topology_type']} p={stats['connection_probability']} "
            f"seed={stats['topology_seed']}  repair={stats['repair_edges']}"
            f"/{stats['corpus_directed_edges']}]"
        )
    print(f"{status}  {name}{summary}")
    for finding in result.findings:
        print(f"      !! {finding}")
    if verbose:
        for note in result.notes:
            print(f"      note: {note}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", action="append", type=Path, help="co-sim dataset dir (repeatable)")
    src.add_argument("--collection", type=Path, help="collection dir containing ds_* subdirs")
    parser.add_argument("--sample", type=int, default=0, help="with --collection: check the first N datasets (0 = all)")
    parser.add_argument(
        "--config",
        type=Path,
        help="space config to regenerate from instead of the dataset's own "
        "(use to prove a mismatched config FAILS)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="override the topology seed, mirroring the live CLI's --seed. Omit for "
        "parity: the live --seed overrides the config seed and changes the topology.",
    )
    parser.add_argument("--sim-input", type=Path, default=DEFAULT_SIM_INPUT)
    parser.add_argument(
        "--allow-backbone-latency-divergence",
        action="store_true",
        help="Downgrade access-link latency divergence (and repair edges no longer at "
        "base_latency) from findings to notes, ONLY when a backbone is present on both "
        "sides. build_core_backbone's jitter rng is offset by the replica-reachability "
        "repair, which live runs never perform, so these diverge on any cell with a "
        "non-empty repair set. Use for a live-vs-live matched A/B where the corpus-side "
        "artifact is only a preflight fixture -- never to wave through a real mismatch.",
    )
    parser.add_argument("--json-out", type=Path, help="write per-dataset results here")
    parser.add_argument("-v", "--verbose", action="store_true", help="print notes as well as findings")
    args = parser.parse_args()

    if args.dataset:
        datasets = list(args.dataset)
    else:
        datasets = sorted(d for d in args.collection.glob("ds_*") if d.is_dir())
        if not datasets:
            print(f"ERROR: no ds_* directories under {args.collection}", file=sys.stderr)
            return 2
        if args.sample:
            datasets = datasets[: args.sample]

    if args.config and len(datasets) > 1:
        print("ERROR: --config applies to a single --dataset", file=sys.stderr)
        return 2

    results: List[ParityResult] = []
    for dataset in datasets:
        result = verify_dataset(
            dataset,
            args.sim_input,
            config_override=args.config,
            seed=args.seed,
            allow_backbone_latency_divergence=args.allow_backbone_latency_divergence,
        )
        results.append(result)
        _print_result(result, args.verbose)

    n_fail = sum(1 for r in results if not r.ok)
    passed = len(results) - n_fail
    print(f"\n{passed}/{len(results)} datasets PASS infra parity")
    if results and all(r.stats for r in results):
        fractions = [r.stats["repair_edge_fraction"] for r in results if r.stats]
        if fractions:
            print(
                f"replica_reachability_repair edges: mean "
                f"{100 * sum(fractions) / len(fractions):.2f}%, max "
                f"{100 * max(fractions):.2f}% of corpus edges"
            )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as handle:
            json.dump(
                {
                    "n_datasets": len(results),
                    "n_pass": passed,
                    "n_fail": n_fail,
                    "results": [
                        {
                            "dataset": r.dataset,
                            "ok": r.ok,
                            "findings": r.findings,
                            "notes": r.notes,
                            "stats": r.stats,
                        }
                        for r in results
                    ],
                },
                handle,
                indent=2,
            )
        print(f"wrote {args.json_out}")

    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
