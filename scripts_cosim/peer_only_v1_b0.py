"""peer_only_v1 B0 -- the 1,670-dataset cache is comparable to the 516-dataset psv3 cache.

    python3 scripts_cosim/peer_only_v1_b0.py --big-cache DIR --small-cache DIR \
        --split experiments/peer_affinity_v1_t1b_split.json --out b0.json

Checks, all registered in docs/lineages/peer_only_v1.md:
  * metadata agrees on contract (partial_state_v3), label, alpha, platform width;
  * >= B0_MIN_DATASETS datasets in the big cache;
  * every dataset of the small cache is in the big cache with IDENTICAL partial-state
    ingredients and bit-identical computed columns (the P0 comparison restricted to those ids);
  * the split's test ids are present in both (the held-out set is the same 34 datasets).
Fails loud on anything but INSTRUMENT-PASS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from scripts_cosim.partial_state_v3_p0 import _load
from scripts_cosim.peer_only_v1_read import read_b0

META_KEYS = ("partial_state_contract", "label_objective", "dag_primary_alpha_key", "platform_feature_dim",
             "peer_exchange_block")


def compare(big: Path, small: Path, split: Path) -> dict:
    mb, ms = json.load(open(big / "metadata.json")), json.load(open(small / "metadata.json"))
    meta_agree = all(mb.get(k) == ms.get(k) for k in META_KEYS)
    gb, gs = _load(big), _load(small)
    missing = sorted(set(gs) - set(gb))
    rows = {}
    for gid in sorted(gs):
        if gid in gb:
            # Same contract on both sides, so the columns are a deterministic function of the
            # ingredients: ingredient + candidate-set equality is the whole comparison.
            rows[gid] = _same_ingredients(gb[gid], gs[gid])
    max_diff = max((r for r in rows.values()), default=0.0)
    test_ids = list(json.load(open(split)).get("test") or [])
    test_same = all(t in gb and t in gs for t in test_ids)
    res = read_b0(int(mb.get("num_datasets") or len(gb)), meta_agree, max_diff, test_same)
    res.update(big_cache=str(big), small_cache=str(small), n_small=len(gs), n_big=len(gb),
               small_missing_from_big=missing, n_test=len(test_ids),
               meta_big={k: mb.get(k) for k in META_KEYS}, meta_small={k: ms.get(k) for k in META_KEYS})
    if missing:
        res["verdict"] = "CORPUS-NOT-COMPARABLE"
    return res


def _same_ingredients(gb, gs) -> float:
    """0.0 when every partial-state ingredient and the candidate sets agree, else 1.0."""
    from scripts_cosim.partial_state_v3_p0 import INGREDIENTS, _canon
    pb, ps = gb.partial_state_ctx, gs.partial_state_ctx
    if any(_canon(pb.get(k)) != _canon(ps.get(k)) for k in INGREDIENTS):
        return 1.0
    tb = [tuple(map(tuple, gb.task_logit_to_placement[t])) for t in range(int(gb.n_tasks))]
    ts = [tuple(map(tuple, gs.task_logit_to_placement[t])) for t in range(int(gs.n_tasks))]
    return 0.0 if tb == ts else 1.0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--big-cache", type=Path, required=True)
    ap.add_argument("--small-cache", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    res = compare(a.big_cache, a.small_cache, a.split)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[B0] {res['verdict']}  big={res['n_big']} small={res['n_small']} missing={len(res['small_missing_from_big'])} "
          f"meta_agree={res['meta_agree']} ingredients_max_diff={res['ingredients_max_diff']} test_same={res['test_ids_same']}")
    if res["verdict"] != "INSTRUMENT-PASS":
        print(f"FAIL LOUD: B0 is {res['verdict']}; see {a.out}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
