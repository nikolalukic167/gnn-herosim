#!/usr/bin/env python3
"""Measure whether two venues (local vs datalab) produce the same GNN decisions.

Why this exists
---------------
Live-gate verdicts are cross-policy `total_rtt` comparisons, so a GNN arm that shifts with
the machine while Knative does not makes any mixed-venue gate a measurement of the venue.
`verify_live_infra_parity.py` proves the two venues simulate the same *topology*; nothing
proved they run the same *numerics*. This is that check.

The mechanism it is looking for
-------------------------------
`total_rtt` only moves when an **argmax flips** in `src/policy/gnn/seq_decode.py`. Kernel
differences across torch/PyG versions perturb logits by ~1e-6 relative — far below the
measured top-2 margins (median 0.058). So flips should be rare, and crucially *unbiased in
sign*: a coin-flip perturbation helps as often as it hurts. A one-directional gap on every
cell is therefore NOT an env signature; it is a biased-estimator signature (which is what the
dims 9-11 temporal-remainder bug turned out to be). This tool exists to hold that reasoning to
a number instead of an argument.

Two modes
---------
``--mode logits`` (seconds)
    Forward a committed frozen batch fixture and diff against a committed reference.
    Reports max|Δ|, p99.9|Δ| and the argmax flip count. Cheap enough to run as a gate
    preflight, and cheap enough to run on a login node (datalab-pitfalls #2 allows a few
    seconds of `python3 -c`; this is that budget).

``--mode run`` (minutes)
    Simulate one gate cell and diff `total_rtt` against a committed reference. Keeps both a
    `knative` and a `gnn` arm on purpose: "Knative reproduces while GNN does not" is the
    fingerprint of this whole bug class, and a Knative-only cross-check is structurally
    incapable of detecting it (Knative never touches `feature_builder.py`).

The fixture is stored as plain numpy arrays, never a pickled PyG ``Data``
------------------------------------------------------------------------
The entire point is to run the same bytes under different `torch_geometric` versions. A
pickled `Data` is a version-coupled artifact and would either fail to load or silently
migrate, which is precisely the failure this tool is supposed to detect.

Usage
-----
    # one-time, on the blessed interpreter: mint the fixture and the reference
    verify_venue_parity.py --capture --cache-dir simulation_data/graphs_cache_full_corpus_siv1_dim14
    verify_venue_parity.py --mode logits --write-reference

    # thereafter, in either venue
    verify_venue_parity.py --mode logits --assert

Run with the repo root on PYTHONPATH (see HANDOVER.md §0):
    PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH=$(pwd) \
      pipenv run python3 scripts_cosim/verify_venue_parity.py --mode logits
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.placement.env_fingerprint import (  # noqa: E402
    describe_python_env,
    env_fingerprint,
    format_env_banner,
)

DEFAULT_FIXTURE = REPO_ROOT / "tests/fixtures/venue_parity/batch_fixture.npz"
DEFAULT_REFERENCE = REPO_ROOT / "tests/fixtures/venue_parity/reference_logits.npz"
DEFAULT_MODEL = REPO_ROOT / "models/near-rtt-v2-full-corpus-siv1-dim14-ce-only.pt"

# Tensor fields copied out of a cached graph. `y` and the dict sidecars are deliberately
# excluded: the forward pass never reads them, so carrying them would only bloat the
# fixture and couple it to cache-format changes.
GRAPH_TENSOR_FIELDS = (
    "task_features",
    "platform_features",
    "edge_index",
    "edge_attr",
    "node_edge_index",
)
GRAPH_SCALAR_FIELDS = ("n_tasks", "n_platforms")


# --------------------------------------------------------------------------------------
# fixture capture
# --------------------------------------------------------------------------------------


def capture_fixture(cache_dir: Path, n_graphs: int, out_path: Path, stride: int) -> None:
    """Export `n_graphs` cached graphs to a version-agnostic .npz.

    Runs only on the blessed interpreter, and only when the fixture is (re)minted — the
    pickle read here is the version coupling this tool otherwise refuses to have.
    """
    import pickle

    graphs_pkl = cache_dir / "graphs.pkl"
    if not graphs_pkl.is_file():
        raise FileNotFoundError(f"no graphs.pkl under {cache_dir}")

    with graphs_pkl.open("rb") as handle:
        graphs = pickle.load(handle)
    if not isinstance(graphs, list) or not graphs:
        raise ValueError(f"{graphs_pkl} did not contain a non-empty list of graphs")

    # Strided sampling rather than the first N: cached graphs are written in dataset order,
    # so a head slice would be one corner of the corpus (one collection, one density) and a
    # flip rate measured there would not generalise to the gate's five topology cells.
    selected_indices = list(range(0, len(graphs), max(1, stride)))[:n_graphs]
    if len(selected_indices) < n_graphs:
        raise ValueError(
            f"cache holds {len(graphs)} graphs; stride={stride} yields only "
            f"{len(selected_indices)} of the requested {n_graphs}. Lower --stride."
        )

    payload: Dict[str, np.ndarray] = {}
    for slot, graph_index in enumerate(selected_indices):
        graph = graphs[graph_index]
        for field in GRAPH_TENSOR_FIELDS:
            value = getattr(graph, field, None)
            if value is None:
                raise ValueError(
                    f"graph {graph_index} is missing tensor field {field!r}; the cache at "
                    f"{cache_dir} is not the layout this fixture expects"
                )
            payload[f"g{slot}/{field}"] = value.detach().cpu().numpy()
        for field in GRAPH_SCALAR_FIELDS:
            payload[f"g{slot}/{field}"] = np.asarray(int(getattr(graph, field)), dtype=np.int64)

    payload["n_graphs"] = np.asarray(len(selected_indices), dtype=np.int64)
    payload["source_cache"] = np.asarray(str(cache_dir))
    payload["source_indices"] = np.asarray(selected_indices, dtype=np.int64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    total_edges = sum(
        int(payload[f"g{slot}/edge_index"].shape[1]) for slot in range(len(selected_indices))
    )
    print(
        f"[capture] wrote {out_path} — {len(selected_indices)} graphs, {total_edges} scored "
        f"edges, {out_path.stat().st_size / 1024:.1f} KiB",
        flush=True,
    )


def load_fixture(fixture_path: Path) -> List[Any]:
    """Rebuild PyG `Data` objects from the version-agnostic fixture."""
    import torch
    from torch_geometric.data import Data

    if not fixture_path.is_file():
        raise FileNotFoundError(
            f"no fixture at {fixture_path}. Mint one on the blessed interpreter with "
            f"--capture --cache-dir <graphs cache>."
        )
    archive = np.load(fixture_path, allow_pickle=False)
    n_graphs = int(archive["n_graphs"])

    graphs: List[Any] = []
    for slot in range(n_graphs):
        data = Data()
        for field in GRAPH_TENSOR_FIELDS:
            array = archive[f"g{slot}/{field}"]
            setattr(data, field, torch.from_numpy(np.ascontiguousarray(array)))
        for field in GRAPH_SCALAR_FIELDS:
            setattr(data, field, int(archive[f"g{slot}/{field}"]))
        graphs.append(data)
    return graphs


# --------------------------------------------------------------------------------------
# logits mode
# --------------------------------------------------------------------------------------


def forward_fixture(graphs: List[Any], model_path: Path, device_name: str) -> Dict[str, np.ndarray]:
    """Run the checkpoint over every fixture graph; return flat logits + per-decision index.

    Loads through `src/executesimulation.py::load_gnn_model` rather than reconstructing the
    architecture here, so the probe exercises the same contract checks, layout resolution and
    message-passing configuration the gate does. A probe that built the model its own way
    could pass while the gate served something else.
    """
    import torch

    from src.executesimulation import load_gnn_model

    model, auto_device = load_gnn_model(model_path)
    device = auto_device if device_name == "auto" else torch.device(device_name)
    model = model.to(device)
    model.eval()

    flat_logits: List[np.ndarray] = []
    decision_offsets: List[int] = [0]
    decision_graph: List[int] = []
    decision_task: List[int] = []

    with torch.no_grad():
        for graph_index, data in enumerate(graphs):
            moved = data.clone() if hasattr(data, "clone") else data
            for field in GRAPH_TENSOR_FIELDS:
                setattr(moved, field, getattr(moved, field).to(device))
            logits_per_task = model(moved)
            for task_index, logits_t in enumerate(logits_per_task):
                values = logits_t.detach().cpu().numpy().astype(np.float32, copy=True)
                # A task with no feasible platform produces an empty logit vector. It is a
                # real state (sparse topologies do it), but it is not a decision, so it must
                # not enter the flip denominator.
                if values.size == 0:
                    continue
                flat_logits.append(values)
                decision_offsets.append(decision_offsets[-1] + values.size)
                decision_graph.append(graph_index)
                decision_task.append(task_index)

    if not flat_logits:
        raise ValueError("fixture produced zero scored decisions — the fixture is unusable")

    return {
        "logits": np.concatenate(flat_logits),
        "offsets": np.asarray(decision_offsets, dtype=np.int64),
        "decision_graph": np.asarray(decision_graph, dtype=np.int64),
        "decision_task": np.asarray(decision_task, dtype=np.int64),
    }


def _per_decision_argmax(logits: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """argmax within each decision's slice — the quantity that actually changes placements."""
    return np.asarray(
        [
            int(np.argmax(logits[offsets[i] : offsets[i + 1]]))
            for i in range(len(offsets) - 1)
        ],
        dtype=np.int64,
    )


def _top2_margins(logits: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """|top1 - top2| per decision; single-candidate decisions contribute +inf.

    This is the population any `GNN_LOGIT_EPS` would have to sit below. Recorded here so the
    eps in Phase 4b is chosen from a measurement rather than from `logit_tied_rate`, which is
    a `< 0.1` band and not a float-tie rate at all.
    """
    margins: List[float] = []
    for i in range(len(offsets) - 1):
        block = logits[offsets[i] : offsets[i + 1]]
        if block.size < 2:
            margins.append(float("inf"))
            continue
        top2 = np.partition(block, -2)[-2:]
        margins.append(float(abs(top2[1] - top2[0])))
    return np.asarray(margins, dtype=np.float64)


def compare_logits(
    actual: Dict[str, np.ndarray], reference_path: Path
) -> Dict[str, Any]:
    """Diff a fresh forward against the committed reference."""
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"no reference at {reference_path}. Mint one on the blessed interpreter with "
            f"--mode logits --write-reference."
        )
    archive = np.load(reference_path, allow_pickle=False)
    ref_logits = archive["logits"]
    ref_offsets = archive["offsets"]

    if ref_logits.shape != actual["logits"].shape or not np.array_equal(
        ref_offsets, actual["offsets"]
    ):
        # Not a numerics difference — the graph itself changed shape. Comparing values
        # across differing decision layouts would silently align the wrong elements.
        raise ValueError(
            "reference and current forward disagree on SHAPE, not just values "
            f"(reference {ref_logits.shape} / {len(ref_offsets) - 1} decisions vs current "
            f"{actual['logits'].shape} / {len(actual['offsets']) - 1} decisions). "
            "The fixture, the checkpoint or the graph builder changed — re-mint both."
        )

    delta = np.abs(actual["logits"].astype(np.float64) - ref_logits.astype(np.float64))
    ref_argmax = _per_decision_argmax(ref_logits, ref_offsets)
    cur_argmax = _per_decision_argmax(actual["logits"], actual["offsets"])
    flipped = np.flatnonzero(ref_argmax != cur_argmax)
    margins = _top2_margins(ref_logits, ref_offsets)
    finite_margins = margins[np.isfinite(margins)]

    reference_env = json.loads(str(archive["env"])) if "env" in archive else None

    report: Dict[str, Any] = {
        "n_decisions": int(len(ref_argmax)),
        "n_scored_edges": int(ref_logits.size),
        "max_abs_delta": float(delta.max()),
        "p99_9_abs_delta": float(np.percentile(delta, 99.9)),
        "mean_abs_delta": float(delta.mean()),
        "argmax_flips": int(flipped.size),
        "argmax_flip_rate": float(flipped.size / max(1, len(ref_argmax))),
        "flipped_decisions": [
            {
                "graph": int(actual["decision_graph"][i]),
                "task": int(actual["decision_task"][i]),
                "reference_choice": int(ref_argmax[i]),
                "current_choice": int(cur_argmax[i]),
                "top2_margin": float(margins[i]),
            }
            for i in flipped[:20]
        ],
        "top2_margin": {
            "median": float(np.median(finite_margins)) if finite_margins.size else None,
            "p05": float(np.percentile(finite_margins, 5)) if finite_margins.size else None,
            "min": float(finite_margins.min()) if finite_margins.size else None,
            "n_single_candidate": int(np.count_nonzero(~np.isfinite(margins))),
        },
        "reference_env": reference_env,
        "current_env": describe_python_env(),
    }
    report["env_fingerprint_matches"] = bool(
        reference_env is not None
        and env_fingerprint(reference_env) == env_fingerprint(report["current_env"])
    )
    return report


def write_reference(actual: Dict[str, np.ndarray], reference_path: Path) -> None:
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        reference_path,
        logits=actual["logits"],
        offsets=actual["offsets"],
        decision_graph=actual["decision_graph"],
        decision_task=actual["decision_task"],
        env=np.asarray(json.dumps(describe_python_env(), sort_keys=True)),
    )
    print(
        f"[reference] wrote {reference_path} — {len(actual['offsets']) - 1} decisions, "
        f"{actual['logits'].size} scored edges",
        flush=True,
    )


def print_logits_report(report: Dict[str, Any]) -> None:
    print("")
    print("=== venue parity: logits ===")
    print(f"  decisions            : {report['n_decisions']}")
    print(f"  scored edges         : {report['n_scored_edges']}")
    print(f"  max |delta|          : {report['max_abs_delta']:.6e}")
    print(f"  p99.9 |delta|        : {report['p99_9_abs_delta']:.6e}")
    print(f"  mean |delta|         : {report['mean_abs_delta']:.6e}")
    print(
        f"  argmax flips         : {report['argmax_flips']} "
        f"({100.0 * report['argmax_flip_rate']:.4f}%)"
    )
    margin = report["top2_margin"]
    if margin["median"] is not None:
        print(
            f"  ref top-2 margin     : median={margin['median']:.4f} "
            f"p05={margin['p05']:.4f} min={margin['min']:.3e} "
            f"(single-candidate: {margin['n_single_candidate']})"
        )
    reference_env = report.get("reference_env") or {}
    current_env = report["current_env"]
    print(
        f"  reference env        : torch={reference_env.get('torch')} "
        f"numpy={reference_env.get('numpy')} pyg={reference_env.get('torch_geometric')}"
    )
    print(
        f"  current env          : torch={current_env.get('torch')} "
        f"numpy={current_env.get('numpy')} pyg={current_env.get('torch_geometric')}"
    )
    print(f"  env fingerprint match: {report['env_fingerprint_matches']}")
    for flip in report["flipped_decisions"]:
        print(
            f"    FLIP graph={flip['graph']} task={flip['task']} "
            f"{flip['reference_choice']} -> {flip['current_choice']} "
            f"(ref top-2 margin {flip['top2_margin']:.3e})"
        )
    print("")


# --------------------------------------------------------------------------------------
# run mode
# --------------------------------------------------------------------------------------


def run_mode(
    config_path: Path,
    workload_path: Path,
    policies: List[str],
    reference_path: Path,
    write_ref: bool,
    timeout: int,
    tolerance: float,
) -> Dict[str, Any]:
    """Simulate one cell per policy and diff `total_rtt` against a committed reference.

    Both `knative` and `gnn` are kept by default. Knative is the control: it never touches
    `build_inference_feature_bundle`, so if Knative reproduces and GNN does not, the
    divergence is in the learned path — the exact signature this whole lineage chased.
    """
    import subprocess
    import tempfile

    interpreter = os.environ.get("HEROSIM_PY", "").strip()
    base_cmd = interpreter.split() if interpreter else [sys.executable]

    policy_args = {
        "knative": ["--knative_network"],
        "gnn": ["--gnn"],
        "mlp": ["--mlp_batch"],
    }

    results: Dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for policy in policies:
            if policy not in policy_args:
                raise ValueError(f"unknown policy {policy!r}; expected one of {sorted(policy_args)}")
            output = Path(tmpdir) / f"{policy}.json"
            cmd = base_cmd + [
                str(REPO_ROOT / "scripts_cosim/run_simulation.py"),
                "--config", str(config_path),
                "--workload", str(workload_path),
                "--output", str(output),
                "--timeout", str(timeout),
                *policy_args[policy],
            ]
            print(f"[run] {policy}: {' '.join(cmd)}", flush=True)
            # No capture: a multi-minute simulation with its output swallowed is how a
            # silent failure becomes a mystery an hour later.
            completed = subprocess.run(cmd, cwd=str(REPO_ROOT))
            if completed.returncode != 0:
                raise RuntimeError(f"{policy} arm exited {completed.returncode}")
            payload = json.loads(output.read_text())
            results[policy] = {
                "total_rtt": float(payload.get("total_rtt", float("nan"))),
                "provenance": payload.get("run_provenance"),
            }

    report: Dict[str, Any] = {
        "config": str(config_path),
        "workload": str(workload_path),
        "results": results,
        "current_env": describe_python_env(),
    }

    if write_ref:
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        reference_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"[reference] wrote {reference_path}", flush=True)
        report["comparison"] = None
        return report

    if not reference_path.is_file():
        raise FileNotFoundError(
            f"no reference at {reference_path}. Mint one on the blessed interpreter with "
            f"--mode run --write-reference."
        )
    reference = json.loads(reference_path.read_text())
    comparison: Dict[str, Any] = {}
    for policy, observed in results.items():
        expected = reference.get("results", {}).get(policy)
        if expected is None:
            comparison[policy] = {"status": "MISSING_IN_REFERENCE"}
            continue
        ref_rtt = float(expected["total_rtt"])
        rel = abs(observed["total_rtt"] - ref_rtt) / max(abs(ref_rtt), 1e-12)
        comparison[policy] = {
            "reference_total_rtt": ref_rtt,
            "total_rtt": observed["total_rtt"],
            "relative_delta": rel,
            "bit_identical": observed["total_rtt"] == ref_rtt,
            "status": "PASS" if rel <= tolerance else "FAIL",
        }
    report["comparison"] = comparison
    report["reference_env"] = reference.get("current_env")
    return report


def print_run_report(report: Dict[str, Any], tolerance: float) -> None:
    print("")
    print("=== venue parity: run ===")
    print(f"  config  : {report['config']}")
    print(f"  workload: {report['workload']}")
    for policy, entry in (report.get("comparison") or {}).items():
        if entry.get("status") == "MISSING_IN_REFERENCE":
            print(f"  {policy:8s}: no reference arm to compare against")
            continue
        print(
            f"  {policy:8s}: {entry['total_rtt']:.6f} vs ref {entry['reference_total_rtt']:.6f} "
            f"rel={entry['relative_delta']:.3e} "
            f"{'bit-identical' if entry['bit_identical'] else ''} [{entry['status']}]"
        )
    print(f"  tolerance: {tolerance:.3e}")
    print("")


# --------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify two venues produce the same GNN decisions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("logits", "run"), default="logits")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--reference", type=Path, default=None,
                        help="defaults to reference_logits.npz / reference_run.json beside the fixture")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu",
                        help="cpu|cuda|auto. Defaults to cpu so the probe isolates the library "
                             "stack rather than the accelerator.")
    parser.add_argument("--threads", type=int, default=1,
                        help="torch intra-op threads. 1 by default: thread count changes "
                             "reduction order, which is a per-job knob, not an env property.")
    parser.add_argument("--write-reference", action="store_true",
                        help="mint the reference instead of comparing against it")
    parser.add_argument("--assert", dest="do_assert", action="store_true",
                        help="exit non-zero on any argmax flip (logits) or tolerance breach (run)")
    parser.add_argument("--json-out", type=Path, default=None)

    capture = parser.add_argument_group("fixture capture (blessed interpreter only)")
    capture.add_argument("--capture", action="store_true")
    capture.add_argument("--cache-dir", type=Path, default=None)
    capture.add_argument("--n-graphs", type=int, default=64)
    capture.add_argument("--stride", type=int, default=37,
                         help="stride through the cache so the sample spans collections and "
                              "densities rather than one corner of the corpus")

    run_group = parser.add_argument_group("run mode")
    run_group.add_argument("--config", type=Path, default=None)
    run_group.add_argument("--workload", type=Path,
                           default=REPO_ROOT / "data/nofs-ids/traces/workload-125-225.json")
    run_group.add_argument("--policies", default="knative,gnn")
    run_group.add_argument("--timeout", type=int, default=18000)
    run_group.add_argument("--tolerance", type=float, default=0.0038,
                           help="relative total_rtt tolerance; default is the measured 0.38%% "
                                "GNN noise floor")

    args = parser.parse_args()

    print(format_env_banner(), flush=True)

    if args.capture:
        if args.cache_dir is None:
            parser.error("--capture requires --cache-dir")
        capture_fixture(args.cache_dir, args.n_graphs, args.fixture, args.stride)
        return 0

    if args.mode == "logits":
        import torch

        torch.set_num_threads(max(1, args.threads))
        reference_path = args.reference or DEFAULT_REFERENCE
        graphs = load_fixture(args.fixture)
        actual = forward_fixture(graphs, args.model, args.device)

        if args.write_reference:
            write_reference(actual, reference_path)
            return 0

        report = compare_logits(actual, reference_path)
        print_logits_report(report)
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if args.do_assert and report["argmax_flips"] > 0:
            print(
                f"FAIL: {report['argmax_flips']} argmax flip(s) vs the reference. This venue "
                f"does not decide identically; its total_rtt is not comparable.",
                file=sys.stderr,
            )
            return 1
        return 0

    # run mode
    if args.config is None:
        parser.error("--mode run requires --config <cell config json>")
    reference_path = args.reference or (args.fixture.parent / "reference_run.json")
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    report = run_mode(
        args.config, args.workload, policies, reference_path,
        args.write_reference, args.timeout, args.tolerance,
    )
    if not args.write_reference:
        print_run_report(report, args.tolerance)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.do_assert and not args.write_reference:
        comparison = report.get("comparison")
        if not comparison:
            print(
                "FAIL: --assert requested but nothing was compared "
                "(empty --policies or no results)",
                file=sys.stderr,
            )
            return 1
        failures = [p for p, e in comparison.items() if e.get("status") != "PASS"]
        if failures:
            print(f"FAIL: arms outside tolerance: {', '.join(failures)}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
