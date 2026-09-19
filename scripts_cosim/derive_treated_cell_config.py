#!/usr/bin/env python3
"""Apply the delta between a base cell config and its treated variant to another base cell.

serving_stability_v1 replicates across CELLS -- independent topology/placement-seed draws --
so C2 and C3 must receive exactly the treatment C1 received, and nothing else. Copying a
config by hand is how a replication quietly becomes a different experiment.

This reads (base, treated) as a worked example of the treatment, extracts the flattened key
delta, applies it to a new base, and REFUSES unless:

  * the example delta is non-empty (otherwise there is no treatment to copy);
  * the new base differs from the example base only in keys named by --allow-differ
    (topology seed, by default) -- so the two cells really are the same cell modulo the draw;
  * the new base does not already carry any treated value;
  * the output differs from the new base in exactly the delta's keys, and from the treated
    example in exactly the allowed keys.

Usage:
    python3 scripts_cosim/derive_treated_cell_config.py \
        --example-base   configs/cell_s7901.json \
        --example-treated configs/cell_s7901_f4000_pg16.json \
        --new-base       configs/cell_s9001.json \
        --out            configs/cell_s9001_f4000_pg16.json
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class ConfigDeriveError(RuntimeError):
    """Fail loud: a silently different cell is a silently different experiment."""


def flatten(doc: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, val in doc.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict):
            out.update(flatten(val, path))
        else:
            out[path] = val
    return out


_MISSING = object()


def delta(base: Dict[str, Any], treated: Dict[str, Any]) -> Dict[str, Any]:
    fb, ft = flatten(base), flatten(treated)
    return {k: ft.get(k, _MISSING) for k in sorted(set(fb) | set(ft))
            if fb.get(k, _MISSING) != ft.get(k, _MISSING)}


def set_path(doc: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node = doc
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            raise ConfigDeriveError(f"cannot set {path}: {part} is not a mapping")
        node = nxt
    node[parts[-1]] = value


def derive(
    example_base: Dict[str, Any],
    example_treated: Dict[str, Any],
    new_base: Dict[str, Any],
    allow_differ: List[str],
) -> Dict[str, Any]:
    treatment = delta(example_base, example_treated)
    if not treatment:
        raise ConfigDeriveError(
            "the example base and treated config are identical -- there is no treatment to copy"
        )
    if any(v is _MISSING for v in treatment.values()):
        raise ConfigDeriveError(
            f"the treatment REMOVES keys {sorted(k for k, v in treatment.items() if v is _MISSING)};"
            " copying a removal is not supported, do it by hand and say so"
        )

    flat_new = flatten(new_base)
    already = {k: flat_new[k] for k, v in treatment.items()
               if k in flat_new and flat_new[k] == v}
    if already:
        raise ConfigDeriveError(
            f"the new base already carries the treated value for {sorted(already)} -- it is "
            "already treated, or the example delta is not the treatment you think it is"
        )

    drift = delta(example_base, new_base)
    unexpected = sorted(set(drift) - set(allow_differ))
    if unexpected:
        raise ConfigDeriveError(
            f"the new base differs from the example base in {unexpected}, which is not in "
            f"--allow-differ {allow_differ}. These are not the same cell modulo its draw, so "
            "the treatment would not be the only thing that changed."
        )

    out = copy.deepcopy(new_base)
    for key, value in treatment.items():
        set_path(out, key, value)

    applied = delta(new_base, out)
    if set(applied) != set(treatment):
        raise ConfigDeriveError(
            f"applied {sorted(applied)} but the treatment is {sorted(treatment)}"
        )
    residual = sorted(set(delta(example_treated, out)) - set(allow_differ))
    if residual:
        raise ConfigDeriveError(
            f"the derived config differs from the treated example in {residual}, beyond the "
            f"allowed {allow_differ}"
        )
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--example-base", type=Path, required=True)
    ap.add_argument("--example-treated", type=Path, required=True)
    ap.add_argument("--new-base", type=Path, required=True)
    ap.add_argument("--allow-differ", nargs="*", default=["network.topology.seed"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    eb = json.loads(args.example_base.read_text())
    et = json.loads(args.example_treated.read_text())
    nb = json.loads(args.new_base.read_text())
    out = derive(eb, et, nb, list(args.allow_differ))

    treatment = delta(eb, et)
    print(f"[treatment] {len(treatment)} key(s): "
          + "; ".join(f"{k} = {v!r}" for k, v in treatment.items()))
    print(f"[draw] {args.new_base.name} differs from {args.example_base.name} only in "
          + str(sorted(delta(eb, nb))))
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[wrote] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
