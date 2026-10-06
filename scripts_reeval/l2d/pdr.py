"""The four priority dispatching rules L2D compares against (paper Table 1), run through the same
env so semantics (semi-active schedule with permissible left shift) match the learned arms.
Ties -> lowest operation index (deterministic)."""
from __future__ import annotations

import numpy as np

PDRS = ("SPT", "MWKR", "FDD_MWKR", "MOPNR")


def pdr_action(rule: str, env, candidate: np.ndarray, mask: np.ndarray) -> int:
    cand = candidate[~mask]
    n_m = env.number_of_machines
    rows, cols = cand // n_m, cand % n_m
    dur = env.dur
    if rule == "SPT":
        key = dur[rows, cols]
    elif rule == "MWKR":
        key = -np.array([dur[r, c:].sum() for r, c in zip(rows, cols)])
    elif rule == "MOPNR":
        key = -(n_m - cols).astype(float)
    elif rule == "FDD_MWKR":
        fdd = np.array([dur[r, : c + 1].sum() for r, c in zip(rows, cols)], dtype=float)
        mwkr = np.array([dur[r, c:].sum() for r, c in zip(rows, cols)], dtype=float)
        key = fdd / mwkr
    else:
        raise ValueError(rule)
    return int(cand[int(np.argmin(key))])


def run_pdr(rule: str, env, data) -> float:
    _, _, candidate, mask = env.reset(data)
    total = -env.initQuality
    while True:
        a = pdr_action(rule, env, candidate, mask)
        _, _, r, done, candidate, mask = env.step(a)
        total += r
        if done:
            break
    return float(-(total - env.posRewards))
