#!/usr/bin/env python3
"""scheduler_residence_v1 R3 -- mint cells that differ ONLY in their topology seed, and
measure the structure each seed actually draws.

R1 found that `cell_s9001` -- the one cell of three with no queue blow-up -- is separated from
the other two on two structural statistics with no overlap: every client can reach at least
2 servers (the others have a client that can reach exactly 1), and its clients-per-server
imbalance is 3 against 8 and 7. The three cells' configs are byte-identical in 149 of 150
fields; the only difference is `network.topology.seed`.

Three cells is three cells. This mints more of them so the relationship can be tested at a
sample size, and it does the selection on the INDEPENDENT variable only: cells are chosen to
span the structural range, before any of them has been served a single task. That ordering is
the point -- picking cells after seeing their latency would be choosing the answer.

    # measure 40 seeds, select 12 spanning the range, write their configs
    python3 scripts_cosim/scheduler_residence_v1_r3_cells.py \
        --base .../configs/cell_s9001_f4000_pg16.json --sim-inputs data/nofs-ids \
        --seeds 9100-9139 --select 12 --out-dir .../configs --manifest .../r3_cells.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.scheduler_residence_v1_read import cell_structure  # noqa: E402

# The statistic cells are selected to span. R1's cleanest separator: the worst-off client's
# number of reachable servers. Ties are broken by the clients-per-server imbalance, R1's other.
SELECT_KEY = "min_reachable_servers"
TIEBREAK_KEY = "hosting_node_spread"
SEED_FIELD = ("network", "topology", "seed")


class CellMintError(RuntimeError):
    """Fail loud: a cell that differs from its base in more than the seed is a different
    experiment wearing the same name."""


def parse_seeds(spec: str) -> List[int]:
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    if not out:
        raise CellMintError(f"no seeds parsed from {spec!r}")
    return out


def with_seed(base: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """The base config with exactly one field changed. Verified field-by-field below."""
    cfg = json.loads(json.dumps(base))
    node = cfg
    for key in SEED_FIELD[:-1]:
        node = node.setdefault(key, {})
    node[SEED_FIELD[-1]] = int(seed)
    _assert_only_seed_differs(base, cfg, seed)
    return cfg


def _flatten(doc: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(doc, dict):
        for k, v in doc.items():
            out.update(_flatten(v, f"{prefix}/{k}"))
    elif isinstance(doc, list):
        out[prefix] = json.dumps(doc)
    else:
        out[prefix] = doc
    return out


def _assert_only_seed_differs(base: Dict[str, Any], cfg: Dict[str, Any], seed: int) -> None:
    fb, fc = _flatten(base), _flatten(cfg)
    seed_path = "/" + "/".join(SEED_FIELD)
    differing = sorted(
        k for k in set(fb) | set(fc) if fb.get(k) != fc.get(k)
    )
    if differing != [seed_path]:
        raise CellMintError(
            f"seed {seed}: minted config differs from its base in {differing} -- expected "
            f"only {seed_path}. A cell that changes anything else is not a topology draw."
        )
    if fc[seed_path] != seed:
        raise CellMintError(f"seed {seed}: seed field reads {fc[seed_path]!r}")


def select_spanning(
    rows: List[Dict[str, Any]], k: int
) -> List[Dict[str, Any]]:
    """Pick k cells spanning the structural range, on the INDEPENDENT variable only.

    Sorted by the separator then the tiebreak, taking evenly spaced ranks: this keeps the
    extremes (which is where the relationship, if any, lives) without over-sampling the mode.
    """
    usable = [r for r in rows if r["structure"].get(SELECT_KEY) is not None]
    if len(usable) < k:
        raise CellMintError(
            f"only {len(usable)} of {len(rows)} seeds gave a measurable "
            f"{SELECT_KEY}; cannot select {k}"
        )
    usable.sort(key=lambda r: (r["structure"][SELECT_KEY],
                               r["structure"].get(TIEBREAK_KEY) or 0.0,
                               r["seed"]))
    if k == 1:
        return [usable[0]]
    idx = [round(i * (len(usable) - 1) / (k - 1)) for i in range(k)]
    seen, picked = set(), []
    for i in idx:
        while i in seen and i + 1 < len(usable):
            i += 1
        seen.add(i)
        picked.append(usable[i])
    return picked


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--sim-inputs", type=Path, required=True)
    ap.add_argument("--seeds", required=True, help="e.g. 9100-9139 or 1,2,3")
    ap.add_argument("--select", type=int, default=0, help="write this many spanning cells")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--prefix", default="cell_r3s")
    ap.add_argument("--suffix", default="_f4000_pg16")
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args(argv)

    base = json.loads(args.base.read_text())
    rows: List[Dict[str, Any]] = []
    for seed in parse_seeds(args.seeds):
        cfg = with_seed(base, seed)
        rows.append({"seed": seed,
                     "structure": cell_structure(cfg, sim_input_path=args.sim_inputs)})

    print(f"\n[R3-cells] {len(rows)} topology draws from base {args.base.name}")
    print(f"  {'seed':>6s} {'min_reach':>10s} {'mean_reach':>11s} {'imbalance':>10s} "
          f"{'clients/server':>15s}")
    for r in rows:
        s = r["structure"]
        def f(k):
            v = s.get(k)
            return "  n/a" if v is None else f"{v:.3f}"
        print(f"  {r['seed']:6d} {f('min_reachable_servers'):>10s} "
              f"{f('mean_reachable_servers'):>11s} {f('hosting_node_spread'):>10s} "
              f"{f('replicas_per_task_type'):>15s}")

    out: Dict[str, Any] = {"lineage": "scheduler_residence_v1", "stage": "R3-cells",
                           "base": str(args.base), "select_key": SELECT_KEY,
                           "tiebreak_key": TIEBREAK_KEY, "draws": rows}

    if args.select:
        if not args.out_dir:
            raise CellMintError("--select needs --out-dir")
        picked = select_spanning(rows, args.select)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        names = []
        for r in picked:
            name = f"{args.prefix}{r['seed']}{args.suffix}"
            (args.out_dir / f"{name}.json").write_text(
                json.dumps(with_seed(base, r["seed"]), indent=1))
            names.append(name)
        out["selected"] = [{"cell": n, **r} for n, r in zip(names, picked)]
        print(f"\n[R3-cells] selected {len(picked)} spanning "
              f"{SELECT_KEY} "
              f"{picked[0]['structure'][SELECT_KEY]:.3f}..{picked[-1]['structure'][SELECT_KEY]:.3f}")
        for n, r in zip(names, picked):
            s = r["structure"]
            print(f"  {n:28s} min_reach {s[SELECT_KEY]:.3f}  imbalance "
                  f"{s.get(TIEBREAK_KEY)}")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(out, indent=2))
    print(f"\n[wrote] {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
