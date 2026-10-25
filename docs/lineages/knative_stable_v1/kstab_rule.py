#!/usr/bin/env python3
"""knative_stable_v1 calibration read: ONLY the three rule fields of each reactive s0 summary (docs/lineages/knative_stable_v1.md, 'Stability rule').
stable = drain (endTime - arrival_end.last_arrival_s)/last_arrival_s <= 0.10 AND peerExchangeFairshare.flow_seconds/(flow_seconds - extra_seconds) <= 3 AND queue_share <= 0.80.
Unknown (missing file, failed file, missing field) is not a pass. A topology passes a rung if its g0 and g1 cells are both stable; the chosen rung is the highest f with >= 11 of 12 topologies passing.
usage: kstab_rule.py <kstab_calib dir> <out.json>"""
import json, os, sys
IDS = (30285, 30286, 30287, 30289, 30290, 30291, 30292, 30293, 30294, 30295, 30296, 30297)
FS = ("0.5", "0.25", "0.125", "0.0625", "0.03125")


def cell(root, f, t, w):
    p = os.path.join(root, f"m{f}", "cells", f"cc40s{t}__{w}f{f}__reactive_s0.summary.json")
    if not os.path.exists(p):
        fp = p.replace(".summary.json", ".failed.json")
        return {"state": "FAILED" if os.path.exists(fp) else "MISSING", "stable": False}
    try:
        s = json.load(open(p))
        end, last = float(s["endTime"]), float(s["arrival_end"]["last_arrival_s"])
        fl, ex = float(s["peerExchangeFairshare"]["flow_seconds"]), float(s["peerExchangeFairshare"]["extra_seconds"])
        qs = float(s["queue_share"])
        drain, slow = (end - last) / last, fl / (fl - ex)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, json.JSONDecodeError) as e:
        return {"state": f"UNKNOWN({type(e).__name__})", "stable": False}
    return {"state": "OK", "drain": drain, "slowdown": slow, "queue_share": qs, "ok_drain": drain <= 0.10, "ok_slow": slow <= 3, "ok_queue": qs <= 0.80,
            "stable": drain <= 0.10 and slow <= 3 and qs <= 0.80}


def main(root, out):
    res, chosen = {}, None
    for f in FS:
        rows = {t: {w: cell(root, f, t, w) for w in ("g0", "g1")} for t in IDS}
        npass = sum(1 for t in IDS if all(rows[t][w]["stable"] for w in rows[t]))
        res[f] = {"topologies_passing": npass, "rung_passes": npass >= 11, "cells": {str(t): rows[t] for t in IDS}}
    ok = [f for f in FS if res[f]["rung_passes"]]
    chosen = max(ok, key=float) if ok else "STOP-NO-STABLE-RATE"
    with open(out, "w") as fh:
        json.dump({"rule": "drain<=0.10, slowdown<=3, queue_share<=0.80; topology passes if g0 and g1 stable; chosen = highest f with >=11/12", "chosen": chosen, "levels": res}, fh, indent=1)
    for f in FS:
        print(f"f{f}: {res[f]['topologies_passing']}/12 topologies pass")
        for t in IDS:
            print("  ", t, " ".join(f"{w}:{'PASS' if c['stable'] else 'fail'}" + (f"(d{c['drain']:.2f} s{c['slowdown']:.1f}x q{c['queue_share']:.2g})" if c["state"] == "OK" else f"({c['state']})") for w, c in res[f]["cells"][str(t)].items()))
    print("CHOSEN", chosen)


if __name__ == "__main__":
    main(*sys.argv[1:3])
