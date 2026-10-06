"""literature_reeval_v1 -- PDR schedule generators other than the L2D env's insertion generator,
plus an independent schedule validator.

Why: run through the upstream env (semi-active with permissible left shift), MWKR / MOPNR / FDD-MWKR
land ~10-15% *below* the paper's Table 1 numbers -- at 6x6 they tie or beat the learned policy. The
paper defines the priority indices (supplement §2) but not the schedule generator its baselines used.
Two classical generators are implemented here so the discrepancy can be attributed:

  append    -- semi-active, no insertion: op starts at max(job_ready, machine_free); the rule ranks
               every ready op (one per job) -- same candidate set as the env, weaker generator.
  nondelay  -- non-delay (Giffler-Thompson style): only ops that can start at the earliest possible
               start time t* are candidates; the rule breaks ties among them.

`validate_env_schedule` recomputes start times from the env's machine sequences and checks machine
non-overlap and job precedence -- independent of the env's own bookkeeping.
"""
from __future__ import annotations

import numpy as np

from pdr import PDRS


def _priority(rule, dur, rows, cols):
    n_m = dur.shape[1]
    if rule == "SPT":
        return dur[rows, cols].astype(float)
    if rule == "MWKR":
        return -np.array([dur[r, c:].sum() for r, c in zip(rows, cols)], dtype=float)
    if rule == "MOPNR":
        return -(n_m - cols).astype(float)
    if rule == "FDD_MWKR":
        fdd = np.array([dur[r, : c + 1].sum() for r, c in zip(rows, cols)], dtype=float)
        mwkr = np.array([dur[r, c:].sum() for r, c in zip(rows, cols)], dtype=float)
        return fdd / mwkr
    raise ValueError(rule)


def run_pdr_generator(rule: str, data, generator: str, validate: bool = True) -> float:
    dur, mch = data
    dur = np.asarray(dur); mch = np.asarray(mch) - 1
    n_j, n_m = dur.shape
    next_col = np.zeros(n_j, dtype=int)
    job_ready = np.zeros(n_j)
    mach_free = np.zeros(n_m)
    end = np.zeros((n_j, n_m))
    seq = [[] for _ in range(n_m)]
    for _ in range(n_j * n_m):
        rows = np.where(next_col < n_m)[0]
        cols = next_col[rows]
        est = np.maximum(job_ready[rows], mach_free[mch[rows, cols]])
        if generator == "nondelay":
            keep = est <= est.min() + 1e-9
            rows, cols, est = rows[keep], cols[keep], est[keep]
        elif generator != "append":
            raise ValueError(generator)
        k = int(np.argmin(_priority(rule, dur, rows, cols)))
        r, c, s = rows[k], cols[k], est[k]
        e = s + dur[r, c]
        end[r, c] = e
        job_ready[r] = e
        mach_free[mch[r, c]] = e
        next_col[r] += 1
        seq[mch[r, c]].append(int(r * n_m + c))
    ms = float(end.max())
    if validate:
        chk = validate_sequences(np.array(seq), dur)
        if abs(chk - ms) > 1e-6:
            raise AssertionError(f"{generator}/{rule}: generator makespan {ms} != validator {chk}")
    return ms


def validate_env_schedule(env) -> float:
    """Rebuild the schedule from env.opIDsOnMchs / env.dur and return its makespan; raises on any
    machine overlap or job-precedence violation. Uses only the machine sequences, not env start times."""
    return validate_sequences(env.opIDsOnMchs, env.dur)


def validate_sequences(seq, dur) -> float:
    """seq: (n_m, n_j) op ids in processing order per machine (negative = empty). Fixed-point start
    times from precedence + machine order, then explicit overlap / precedence / completeness checks."""
    n_j, n_m = dur.shape
    start = np.full((n_j, n_m), np.nan)
    # iterate to a fixed point: start = max(end of job pred, end of machine pred)
    for _ in range(n_j * n_m + 1):
        changed = False
        for k in range(n_m):
            prev_end = 0.0
            for op in seq[k]:
                if op < 0:
                    continue
                r, c = op // n_m, op % n_m
                jp_end = 0.0 if c == 0 else (start[r, c - 1] + dur[r, c - 1])
                if np.isnan(jp_end):
                    prev_end = np.nan
                    continue
                s = max(prev_end, jp_end)
                if np.isnan(start[r, c]) or abs(start[r, c] - s) > 1e-9:
                    start[r, c] = s
                    changed = True
                prev_end = s + dur[r, c]
        if not changed:
            break
    if np.isnan(start).any():
        raise AssertionError("schedule has a precedence cycle / unscheduled op")
    for k in range(n_m):
        ops = [op for op in seq[k] if op >= 0]
        if len(ops) != n_j or len(set(ops)) != n_j:
            raise AssertionError(f"machine {k} does not hold exactly one op per job")
        for a, b in zip(ops, ops[1:]):
            ra, ca, rb, cb = a // n_m, a % n_m, b // n_m, b % n_m
            if start[ra, ca] + dur[ra, ca] > start[rb, cb] + 1e-9:
                raise AssertionError(f"machine {k}: overlap between ops {a} and {b}")
    for r in range(n_j):
        for c in range(1, n_m):
            if start[r, c - 1] + dur[r, c - 1] > start[r, c] + 1e-9:
                raise AssertionError(f"job {r}: op {c} starts before op {c-1} ends")
    return float((start + dur).max())


def paper_variant_probe(L, data, env):
    """Try to reproduce the paper's Table-1 baseline numbers: rules with 'remaining' excluding the
    current op, and random tie-breaking (5 draws). Diagnostic only."""
    import random
    rng = random.Random(0)
    res = {}
    for rule in ("MWKR", "MOPNR", "FDD_MWKR"):
        def pick(env, cand, mask, excl, randtie):
            c = cand[~mask]; n_m = env.number_of_machines; rows, cols = c // n_m, c % n_m; dur = env.dur
            off = 1 if excl else 0
            if rule == "MWKR": key = -np.array([dur[r, cc + off:].sum() for r, cc in zip(rows, cols)], float)
            elif rule == "MOPNR": key = -(n_m - cols - off).astype(float)
            else:
                fdd = np.array([dur[r, :cc + 1].sum() for r, cc in zip(rows, cols)], float)
                mw = np.array([dur[r, cc + off:].sum() for r, cc in zip(rows, cols)], float)
                key = fdd / np.maximum(mw, 1e-9)
            if randtie:
                best = np.where(key <= key.min() + 1e-9)[0]; return int(c[rng.choice(list(best))])
            return int(c[int(np.argmin(key))])
        for excl in (False, True):
            for randtie in (False, True):
                ms = []
                for x in data:
                    _, _, cand, mask = env.reset(x); tot = -env.initQuality
                    while True:
                        _, _, r, done, cand, mask = env.step(pick(env, cand, mask, excl, randtie)); tot += r
                        if done: break
                    ms.append(-(tot - env.posRewards))
                res[f"{rule} excl_current={excl} random_ties={randtie}"] = float(np.mean(ms))
    return res


if __name__ == "__main__":
    import argparse, os, sys, json
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from l2d_bridge import load_l2d
    from pdr import run_pdr
    ap = argparse.ArgumentParser(); ap.add_argument("--n_j", type=int, default=6); ap.add_argument("--n_m", type=int, default=6)
    a = ap.parse_args()
    L = load_l2d(a.n_j, a.n_m, "cpu")
    d = np.load(os.path.join(L.root, f"DataGen/generatedData{a.n_j}_{a.n_m}_Seed200.npy"))
    data = [(d[i][0], d[i][1]) for i in range(d.shape[0])]
    env = L.SJSSP(n_j=a.n_j, n_m=a.n_m)
    out = {}
    for rule in PDRS:
        env_ms, env_chk = [], []
        for x in data:
            ms = run_pdr(rule, env, x)
            chk = validate_env_schedule(env)
            if abs(chk - ms) > 1e-6:
                raise AssertionError(f"{rule}: env makespan {ms} != validator {chk}")
            env_ms.append(ms)
        out[rule] = {"env_insertion": float(np.mean(env_ms)),
                     "append": float(np.mean([run_pdr_generator(rule, x, "append") for x in data])),
                     "nondelay": float(np.mean([run_pdr_generator(rule, x, "nondelay") for x in data]))}
        print(rule, json.dumps(out[rule]))
    for k, v in paper_variant_probe(L, data, env).items():
        print("probe", k, round(v, 2))
