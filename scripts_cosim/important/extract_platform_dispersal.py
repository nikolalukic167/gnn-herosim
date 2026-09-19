"""Per-platform utilisation from a live-gate result, as a dispersal measure.

Task B asked whether the cells where the MLP collapses share a *structure* the GNN can see
and the MLP cannot. They do not -- adjacency is byte-identical across the four cell sets and
no degree/choice-set/concentration statistic separates collapse from healthy. What separates
them is what the scheduler *did*: how widely it spread load over the platforms available.

`stats.nodeResults[].platformResults[].idleProportion` carries that directly, but it sits
inside the ~58MB nodeResults block, so this streams the block and pairs each idleProportion
with the platformId that most recently preceded it rather than parsing the document.

Reported per run:
  * `n_busy_gt1pct` -- platforms doing more than 1% of wall time. The dispersal measure.
  * `top1_share` / `top3_share` -- share of all busy time held by the busiest platform(s).
  * `max_busy_pct` -- utilisation of the single busiest platform.

The last one is what tells the two collapse MECHANISMS apart. A platform-side collapse packs
a few platforms hot (high max_busy, low n_busy). A link-side collapse leaves every platform
nearly idle (max_busy ~5%) while RTT still blows up, because the link wait is taken inside
the replica's serving loop (infrastructure.py:1082) and so surfaces as queue time, never as
averageCommunicationsTime.

Usage:
  extract_platform_dispersal.py --out simulation_data/platform_dispersal.json \\
      --arms mlp,mlptempfix,tempfix,knative
"""

import argparse
import bisect
import json
import re
import sys
from pathlib import Path

PID = re.compile(rb'"platformId":\s*(\d+)')
IDLE = re.compile(rb'"idleProportion":\s*([\d.eE+-]+)')

GATES = [
    ("drawgate", "backbone"), ("drawgate", "nobackbone"),
    ("promo175", "backbone"), ("promo175", "nobackbone"),
    ("bbrob", "bb_core8_bw1p5"), ("bbrob", "bb_core4_bw0p5"),
]
ARM_SUFFIX = {"knative": "knative", "mlp": "mlp_dim22", "mlptempfix": "mlp_dim22"}


def platform_utilisation(path: Path) -> dict[int, float]:
    """{platformId: busy_pct}. Reads the whole file once; the region of interest is bounded."""
    data = path.read_bytes()
    start = data.find(b'"nodeResults"')
    end = data.find(b'"taskResults"')
    if start < 0 or end < 0 or end <= start:
        raise SystemExit(
            f"FAIL LOUD: {path} has no nodeResults..taskResults region "
            "(start={start}, end={end}) -- the result layout changed."
        )
    region = data[start:end]
    pids = [(m.start(), int(m.group(1))) for m in PID.finditer(region)]
    if not pids:
        raise SystemExit(f"FAIL LOUD: no platformId entries in {path}")
    offsets = [p[0] for p in pids]
    out: dict[int, float] = {}
    for m in IDLE.finditer(region):
        i = bisect.bisect_left(offsets, m.start()) - 1
        if i >= 0:
            out[pids[i][1]] = 100.0 - float(m.group(1))
    return out


def summarise(util: dict[int, float]) -> dict:
    busy = sorted(util.values(), reverse=True)
    total = sum(busy)
    if total <= 0:
        raise SystemExit("FAIL LOUD: every platform reports 100% idle")
    return {
        "n_platforms": len(busy),
        "n_busy_gt1pct": sum(1 for b in busy if b > 1.0),
        "max_busy_pct": round(busy[0], 3),
        "top1_share_pct": round(100.0 * busy[0] / total, 2),
        "top3_share_pct": round(100.0 * sum(busy[:3]) / total, 2),
        "total_busy_pct": round(total, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("simulation_data/normal_sim_sweeps"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", default="knative,tempfix,mlp,mlptempfix")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    out: dict = {}
    for prefix, cond in GATES:
        for arm in arms:
            d = args.root / f"{prefix}_{cond}_{arm}" / "results"
            if not d.is_dir():
                continue
            suffix = ARM_SUFFIX.get(arm, "gnn")
            for p in sorted(d.glob(f"*_s0_{suffix}.json")):
                cell = p.name.replace(f"_s0_{suffix}.json", "")
                rec = summarise(platform_utilisation(p))
                out.setdefault(f"{prefix}/{cond}", {}).setdefault(cell, {})[arm] = rec
                print(f"{prefix}/{cond} {cell} {arm}: "
                      f"n_busy>1%={rec['n_busy_gt1pct']:3d} "
                      f"max_busy={rec['max_busy_pct']:6.2f}% "
                      f"top3_share={rec['top3_share_pct']:5.1f}%", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    n = sum(len(a) for c in out.values() for a in c.values())
    print(f"wrote {args.out}: {n} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
