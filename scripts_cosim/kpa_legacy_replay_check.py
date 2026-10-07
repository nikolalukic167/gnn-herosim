#!/usr/bin/env python3
"""kpa_scaleout_v1 replay check: the legacy path with the kpa code must reproduce transfer_physics_v1.

  kpa_legacy_replay_check.py <old gate dir> <new gate dir> [--out check.json]

Compares every run summary present in both directories, field for field, on every simulated quantity.
Wall-clock fields (wallclock_s, total_rtt_plus_inference) and provenance blocks (env, code) are excluded;
`env` must differ only by HEROSIM_SCALEOUT. Runs failed or missing on either side are listed by name,
never imputed. Exit 1 if any common run differs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXCLUDED = {"wallclock_s", "total_rtt_plus_inference", "env", "code"}


def load(d: Path) -> dict:
    return {p.name[: -len(".summary.json")]: p for p in d.glob("*.summary.json")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    old, new = load(a.old), load(a.new)
    common = sorted(set(old) & set(new))
    diffs, env_diffs = {}, {}
    for name in common:
        o, n = json.loads(old[name].read_text()), json.loads(new[name].read_text())
        bad = sorted(k for k in (set(o) | set(n)) - EXCLUDED if o.get(k) != n.get(k))
        if bad:
            diffs[name] = {k: [o.get(k), n.get(k)] for k in bad[:6]}
        oe, ne = dict(o.get("env") or {}), dict(n.get("env") or {})
        ne.pop("HEROSIM_SCALEOUT", None)
        oe.pop("HEROSIM_SCALEOUT", None)
        if oe != ne:
            env_diffs[name] = {"old": oe, "new": ne}
    report = {
        "old": str(a.old), "new": str(a.new), "common": len(common), "identical": len(common) - len(diffs),
        "differ": diffs, "env_differ": env_diffs,
        "only_old": sorted(set(old) - set(new)), "only_new": sorted(set(new) - set(old)),
        "failed_old": sorted(p.name for p in a.old.glob("*.failed.json")),
        "failed_new": sorted(p.name for p in a.new.glob("*.failed.json")),
    }
    text = json.dumps(report, indent=1)
    if a.out:
        a.out.write_text(text)
    print(f"{a.new.name}: {report['identical']}/{len(common)} identical, {len(diffs)} differ, "
          f"{len(env_diffs)} env differ, only_old {len(report['only_old'])}, only_new {len(report['only_new'])}")
    return 1 if diffs or env_diffs else 0


if __name__ == "__main__":
    sys.exit(main())
