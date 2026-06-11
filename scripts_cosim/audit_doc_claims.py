#!/usr/bin/env python3
"""Verify quantitative claims against on-disk sweep JSONs and co-sim corpora."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
SWEEPS = ROOT / "simulation_data/normal_sim_sweeps"
COSIM = ROOT / "simulation_data/artifacts/run_queue_big"
RESULTS = ROOT / "simulation_data/results"

CONFIGS = [
    ("default", "default_20_20_p50.json"),
    ("00", "00_balanced_30_30_p35.json"),
    ("01", "01_balanced_40_40_p50.json"),
    ("02", "02_balanced_50_50_p60.json"),
    ("03", "03_client_heavy_50_35_p50.json"),
    ("04", "04_server_heavy_35_50_p50.json"),
    ("05", "05_sparse_40_40_p25.json"),
]


def load_json(p: Path) -> Any:
    with open(p) as f:
        return json.load(f)


def rtt_m(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    v = load_json(path).get("total_rtt")
    return float(v) / 1e6 if v is not None else None


def pct_delta(a: float, b: float) -> float:
    return 100.0 * (a - b) / b


def cold_task_rate(stats: Dict) -> Optional[float]:
    trs = stats.get("taskResults") or []
    if not trs:
        return None
    cold = sum(1 for t in trs if float(t.get("coldStartTime") or 0) > 0)
    return 100.0 * cold / len(trs)


def cosim_cold_task_rates() -> Tuple[Optional[float], int]:
    root = COSIM / "gnn_datasets_4tasks_1060"
    rates: List[float] = []
    for ds in sorted(root.glob("ds_*")):
        opt = ds / "optimal_result.json"
        if not opt.exists():
            continue
        r = cold_task_rate(load_json(opt).get("stats", {}))
        if r is not None:
            rates.append(r)
    if not rates:
        return None, 0
    return statistics.mean(rates), len(rates)


def cosim_cold_platform_rates() -> Tuple[Optional[float], int]:
    root = COSIM / "gnn_datasets_4tasks_1060"
    rates: List[float] = []
    for ds in sorted(root.glob("ds_*")):
        ssc = ds / "system_state_captured_unique.json"
        if not ssc.exists():
            continue
        init = load_json(ssc).get("initialized_snapshot") or {}
        if not init:
            continue
        cold = sum(1 for v in init.values() if not v)
        rates.append(100.0 * cold / len(init))
    if not rates:
        return None, 0
    return statistics.mean(rates), len(rates)


def sum_rtts(dir_path: Path, pattern: str) -> Tuple[Optional[float], int]:
    if not dir_path.exists():
        return None, 0
    files = [f for f in dir_path.glob(pattern) if "decode_stats" not in f.name]
    if not files:
        return None, 0
    return sum(load_json(f)["total_rtt"] for f in files) / 1e6, len(files)


def main() -> None:
    report: Dict[str, Any] = {"fixes_needed": [], "verified": {}}

    ce_dir = SWEEPS / "gnn_near_rtt_v2_dim14_ce_only_20260609/results"
    kn_dir = SWEEPS / "knative_network_20260606_192413/results"
    kn_def = SWEEPS / "baseline_default_100100/results/knative_default_20_20_p50.json"
    hrc_dir = SWEEPS / "herocache_network_20260606_205112/results"
    pf_dir = SWEEPS / "dim14_3model_3cfg_queuefix_20260609/results"
    rr_dir = SWEEPS / "random_rr_3cfg_20260609/results"

    ce_rtts = {s: rtt_m(ce_dir / fn) for s, fn in CONFIGS}
    ce_rtts = {k: v for k, v in ce_rtts.items() if v is not None}
    report["verified"]["ce_only_7cfg_m"] = {k: round(v, 3) for k, v in ce_rtts.items()}
    report["verified"]["ce_only_sum_m"] = round(sum(ce_rtts.values()), 3)

    wins_kn = wins_hrc = 0
    for short, fn in CONFIGS:
        ce = ce_rtts.get(short)
        if ce is None:
            continue
        kn_p = kn_def if short == "default" else kn_dir / fn
        kn = rtt_m(kn_p)
        hrc = rtt_m(hrc_dir / fn)
        if kn and ce < kn:
            wins_kn += 1
        if hrc and ce < hrc:
            wins_hrc += 1
    report["verified"]["ce_wins_vs_knative"] = f"{wins_kn}/7"
    report["verified"]["ce_wins_vs_hrc"] = f"{wins_hrc}/7"

    pf_vals = [
        rtt_m(pf_dir / f"ce-only_{c}.json")
        for c in ["default_20_20_p50", "02_balanced_50_50_p60", "04_server_heavy_35_50_p50"]
    ]
    report["verified"]["post_fix_ce_3cfg_sum_m"] = round(sum(v for v in pf_vals if v), 3)

    bip = SWEEPS / "sweep_bipartite_coordination_v1/results"
    for label, pat in [("gnn", "*_gnn_dim22.json"), ("mlp", "*_mlp_dim22.json"), ("kn", "*_knative.json")]:
        s, n = sum_rtts(bip, pat)
        report["verified"][f"bipartite_{label}_sum_m"] = round(s, 3) if s else None
        report["verified"][f"bipartite_{label}_n"] = n

    p150 = RESULTS / "150-150"
    kn150 = rtt_m(p150 / "simulation_result_knative_150-150.json")
    seq150 = rtt_m(p150 / "simulation_result_gnn_seqblend_150-150.json")
    report["verified"]["150_150"] = {
        "knative_m": round(kn150, 3) if kn150 else None,
        "seqblend_m": round(seq150, 3) if seq150 else None,
        "seqblend_pct_vs_kn": round(pct_delta(seq150, kn150), 1) if kn150 and seq150 else None,
    }

    ct, cn = cosim_cold_task_rates()
    cp, cpn = cosim_cold_platform_rates()
    live_csp = []
    for _, fn in CONFIGS:
        p = ce_dir / fn
        if p.exists():
            csp = load_json(p).get("stats", {}).get("coldStartProportion")
            if csp is not None:
                live_csp.append(100 * float(csp))
    report["verified"]["cold_start"] = {
        "cosim_task_cold_pct_mean": round(ct, 2) if ct is not None else None,
        "cosim_task_n": cn,
        "cosim_platform_cold_pct_mean": round(cp, 1) if cp is not None else None,
        "cosim_platform_n": cpn,
        "live_ce_coldStartProportion_pct_mean": round(statistics.mean(live_csp), 1) if live_csp else None,
        "live_ce_coldStartProportion_pct_range": [round(min(live_csp), 1), round(max(live_csp), 1)] if live_csp else None,
    }

    kn_d, rnd_d, rr_d = [], [], []
    for short, fn in CONFIGS:
        ce = ce_rtts.get(short)
        if ce is None:
            continue
        kn_p = kn_def if short == "default" else kn_dir / fn
        kn = rtt_m(kn_p)
        rnd = rtt_m(rr_dir / f"random_network_{fn}")
        rb = rtt_m(rr_dir / f"roundrobin_{fn}")
        if kn:
            kn_d.append(pct_delta(kn, ce))
        if rnd:
            rnd_d.append(pct_delta(rnd, ce))
        if rb:
            rr_d.append(pct_delta(rb, ce))
    report["verified"]["weak_baselines_delta_pct"] = {
        "knative": [round(min(kn_d), 0), round(max(kn_d), 0)] if kn_d else None,
        "random": [round(min(rnd_d), 0), round(max(rnd_d), 0)] if rnd_d else None,
        "roundrobin": [round(min(rr_d), 0), round(max(rr_d), 0)] if rr_d else None,
    }

    out = ROOT / "logs/doc_claims_audit.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
