#!/usr/bin/env python3
"""KPA conformance (ii), the comparison. Open loop: every decider gets the traced (t, obs, cur, ready) per function per tick.

  ours   src/placement/scaleout.py KpaScaler (origin/reference-physics; unchanged since the cells' pin a459dd98), replayed.
  go     Knative Serving's decider (pkg/autoscaler/scaling) over its TimedFloat64Buckets, from the Go test (per-tick CSV).
  model  a parametrised port: every switch at "ours" must reproduce KpaScaler on every tick, every switch at "kn" must
         reproduce the Go series on every tick (the joint residual). One switch at a time to "kn" attributes a difference:
         a tick is explained by Dk only if it differs ours-vs-go and the Dk-aligned model equals go there. Nothing is fitted.

  D1 panic entry on ceil(panic/target)/max(1,ready) (kn) vs unceiled panic/target/max(1,ready) (ours)
  D2 desired in panic: max(stable, panic) then the running max (kn) vs max(panic, running max, current) (ours)
  D3 un-panic strictly after last-over + stable window (kn) vs at >= (ours)
  D4 hold of the last replica until a quiet stable window (ours only; Knative leaves it to the reconciler)
  D5 averaging: 1 s buckets, partial-window denominator, 6-digit rounding (kn) vs mean of samples in (now - W, now] (ours)
  D6 ceil without the 1e-9 epsilon (kn) vs ceil(x - 1e-9) (ours)
  D7 no decision while the stable buckets are empty (kn ScaleValid false)
usage: kpa_conformance.py <scaleout.py> <trace.kpa.jsonl> <go.csv> <label>   (prints the report; per-tick CSV next to go.csv)"""
import csv, importlib.util, itertools, json, math, sys
from collections import Counter, defaultdict

TARGET, SW, PW, THR, UP, DOWN = 0.7, 60.0, 6.0, 2.0, 1000.0, 2.0
CAP = 120
EPOCH_S = 1_000_000_000
D = ("D1", "D2", "D3", "D4", "D5", "D6", "D7")


def go_round6(x):
    p = 1e6
    y = x * p
    return math.copysign(math.floor(abs(y) + 0.5), y) / p


class Buckets:
    """aggregation.TimedFloat64Buckets, 1 s granularity, times as absolute integer seconds."""

    def __init__(self, window_s):
        self.n = int(window_s); self.w = int(window_s)
        self.b = [0.0] * self.n; self.first = None; self.last = None; self.total = 0.0

    def record(self, sec, v):
        if self.last != sec:
            if self.last is None or sec + self.w > self.last:
                if self.first is None or self.first > sec:
                    self.first = sec
                if self.last is None or sec > self.last:
                    if self.last is None or sec - self.last >= self.w:
                        self.first = sec; self.b = [0.0] * self.n; self.total = 0.0
                    else:
                        for i in range(self.last + 1, sec + 1):
                            idx = i % self.n; self.total -= self.b[idx]; self.b[idx] = 0.0
                    self.last = sec
            else:
                return
        self.b[sec % self.n] += v; self.total += v

    def empty(self, sec):
        return self.last is None or sec - self.last > self.w

    def avg(self, sec):
        d = sec - self.last
        if d <= 0:
            return go_round6(self.total / min(self.last - self.first + 1, self.n))
        if d < self.w:
            ret = self.total
            for i in range(self.last + 1, sec + 1):
                ret -= self.b[i % self.n]
            return go_round6(ret / min(self.last - self.first + 1, self.n - (sec - self.last)))
        return 0.0


class Model:
    def __init__(self, kn):
        self.kn = set(kn)
        self.samples = []; self.first = None; self.last_traffic = -math.inf
        self.panic_since = None; self.last_over = None; self.maxp = 0
        self.sb, self.pb = Buckets(SW), Buckets(PW)
        self.k_panic_ns = None  # Knative panicTime (ns), zero = None

    def step(self, t, obs, cur, ready):
        ns = int(t * 1e9); sec = EPOCH_S + ns // 1_000_000_000
        if self.first is None:
            self.first = t
        self.samples.append((t, obs))
        if obs > 0:
            self.last_traffic = t
        self.samples = [x for x in self.samples if not x[0] <= t - SW]
        self.sb.record(sec, obs); self.pb.record(sec, obs)
        if "D7" in self.kn and self.sb.empty(sec):
            return None
        if "D5" in self.kn:
            sa, pa = self.sb.avg(sec), self.pb.avg(sec)
        else:
            sv = [v for tt, v in self.samples if tt > t - SW]; pv = [v for tt, v in self.samples if tt > t - PW]
            sa = sum(sv) / len(sv) if sv else 0.0; pa = sum(pv) / len(pv) if pv else 0.0
        c = (lambda x: math.ceil(x)) if "D6" in self.kn else (lambda x: math.ceil(x - 1e-9))
        rs, rp = c(sa / TARGET), c(pa / TARGET)
        rdy = max(1, ready)
        lo = math.floor(rdy / DOWN); hi = math.ceil(UP * rdy)
        ds, dp = min(max(rs, lo), hi), min(max(rp, lo), hi)
        over = (rp / rdy >= THR) if "D1" in self.kn else ((pa / TARGET) / max(1, ready) >= THR)
        if "D3" in self.kn:
            if over:
                self.k_panic_ns = ns
            elif self.k_panic_ns is not None and self.k_panic_ns + int(SW * 1e9) < ns:
                self.k_panic_ns = None
            panicking = self.k_panic_ns is not None
        else:
            if over:
                if self.panic_since is None:
                    self.panic_since = t
                self.last_over = t
            elif self.panic_since is not None and t - self.last_over >= SW:
                self.panic_since = None; self.last_over = None
            panicking = self.panic_since is not None
        if panicking:
            if "D2" in self.kn:
                d = max(ds, dp)
                if d > self.maxp:
                    self.maxp = d
                desired = self.maxp
            else:
                desired = max(dp, self.maxp, cur); self.maxp = desired
        else:
            self.maxp = 0; desired = ds
            if "D4" not in self.kn and desired == 0 and cur > 0:
                covered = t - self.first >= SW; quiet = t - self.last_traffic >= SW
                if not (covered and quiet):
                    desired = 1
        return desired


def main():
    so_path, trace, gocsv, label = sys.argv[1:5]
    spec = importlib.util.spec_from_file_location("scaleout_rp", so_path); so = importlib.util.module_from_spec(spec); sys.modules["scaleout_rp"] = so; spec.loader.exec_module(so)
    rows = [json.loads(l) for l in open(trace)]
    hdr = rows[0]; assert hdr["k"] == "header"
    kc = hdr["kpa"]
    assert (kc["target"], kc["stable_window_s"], kc["panic_window_s"], kc["panic_threshold"], kc["max_scale_up_rate"], kc["max_scale_down_rate"]) == (TARGET, SW, PW, THR, UP, DOWN), kc
    rows = [r for r in rows if r["k"] == "kpa"]
    go = list(csv.DictReader(open(gocsv)))
    assert len(go) == len(rows)
    real = so.KpaScaler(so.KpaConfig(TARGET, SW, PW))
    variants = {"ours": (), "joint": D, **{k: (k,) for k in D},
                **{"+".join(c): c for n in (2, 3) for c in itertools.combinations(D[:6], n)}}
    models = {v: defaultdict(lambda kn=kn: Model(kn)) for v, kn in variants.items()}
    out = []
    first = {}
    for i, (r, g) in enumerate(zip(rows, go)):
        fn, t = r["fn"], r["t"]
        assert g["fn"] == fn and float(g["t"]) == t and float(g["obs"]) == r["obs"] and int(g["ready"]) == r["ready"], i
        first.setdefault(fn, t)
        real.observe(fn, t, r["obs"]); dec = real.decide(fn, t, r["cur"], r["ready"])
        assert dec.desired == r["desired"], f"replay != trace at row {i}"
        m = {v: models[v][fn].step(t, r["obs"], r["cur"], r["ready"]) for v in variants}
        assert m["ours"] == dec.desired, f"model(ours) != KpaScaler at row {i}"
        gd = int(g["desired"]) if g["valid"] == "true" else None
        out.append(dict(fn=fn, t=t, cur=r["cur"], ready=r["ready"], obs=r["obs"], occ_idle=sum(1 for v in r["occ"].values() if v == 0),
                        ours=dec.desired, go=gd, **{f"m_{v}": m[v] for v in variants if v != "ours"},
                        full=t - first[fn] >= SW))
    # tick groups (one autoscaler pass = one row per function, in order) for the cap check
    nf = len(first); tot_cur = []
    for j in range(0, len(out), nf):
        s = sum(x["cur"] for x in out[j:j + nf]); tot_cur += [s] * len(out[j:j + nf])
    rep = [f"=== {label}: {len(out)} ticks, {nf} functions; trace {trace}"]
    rep.append("  self-check: replayed KpaScaler == traced desired on 100 % of ticks; model(all ours) == KpaScaler on 100 %")
    joint_bad = [x for x in out if x["full"] and x["m_joint"] != x["go"]]
    rep.append(f"  joint alignment (D1-D7 all to kn) vs go, window full: residual {len(joint_bad)} ticks"
               + (f"; all ticks incl. filling: {sum(1 for x in out if x['m_joint'] != x['go'])}" ))
    keep = ["fn", "t", "cur", "ready", "obs", "occ_idle", "ours", "go", "full"] + [f"m_{k}" for k in ("joint",) + D]
    with open(gocsv.replace(".csv", ".compare.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keep + ["tot_cur"]); w.writeheader()
        for x, s in zip(out, tot_cur):
            w.writerow({**{k: x[k] for k in keep}, "tot_cur": s})
    rep.append(f"  {'fn':5s} {'full':>6s} {'same':>6s} {'differ':>6s} " + " ".join(f"{k:>5s}" for k in D)
               + f" {'multi':>5s} {'joint':>5s} {'UNEXP':>5s} {'|diff| hist':>s}")
    combo_by = defaultdict(Counter)
    for fn in sorted(first):
        xs = [x for x in out if x["fn"] == fn and x["full"]]
        diff = [x for x in xs if x["ours"] != x["go"]]
        by = Counter(); multi = joint_only = unexp = 0; mag = Counter()
        for x in diff:
            hit = [k for k in D if x[f"m_{k}"] == x["go"]]
            for k in hit:
                by[k] += 1
            multi += len(hit) > 1
            if not hit:
                if x["m_joint"] == x["go"]:
                    joint_only += 1
                    combos = [v for v in variants if "+" in v and x[f"m_{v}"] == x["go"]]
                    n0 = min((v.count("+") for v in combos), default=None)
                    combo_by[fn][" | ".join(v for v in combos if v.count("+") == n0) if combos else "needs >3"] += 1
                else:
                    unexp += 1
            mag[(x["go"] if x["go"] is not None else 0) - x["ours"]] += 1
        rep.append(f"  {fn:5s} {len(xs):6d} {len(xs) - len(diff):6d} {len(diff):6d} " + " ".join(f"{by[k]:5d}" for k in D)
                   + f" {multi:5d} {joint_only:5d} {unexp:5d} " + " ".join(f"{d:+d}:{n}" for d, n in sorted(mag.items())))
    for fn in sorted(combo_by):
        rep.append(f"  joint-only ticks {fn}, smallest aligning combination(s): " + "; ".join(f"{k}: {v}" for k, v in combo_by[fn].most_common()))
    # realised effect (approximate): requested change; ups capped by 120 - total live replicas, downs by idle replicas
    def realised(x, d, s):
        if d is None:
            return 0
        ch = d - x["cur"]
        return min(ch, max(0, CAP - s)) if ch > 0 else -min(-ch, x["occ_idle"]) if ch < 0 else 0
    rq = rl = 0; rl_by = Counter()
    for x, s in zip(out, tot_cur):
        if x["full"] and x["ours"] != x["go"]:
            rq += 1
            a, b = realised(x, x["ours"], s), realised(x, x["go"], s)
            if a != b:
                rl += 1; rl_by[x["fn"]] += 1
    rep.append(f"  window-full differing ticks {rq}; realised change would differ on {rl} (ups capped at {CAP} - live total, downs at idle"
               f" replicas from the trace's occupancy; approximate: memory refusals and A3-protected replicas not modelled) " + str(dict(rl_by)))
    # D5 bookkeeping: tick spacing and bucket sharing
    for fn in sorted(first):
        ts = [x["t"] for x in out if x["fn"] == fn]
        sp = Counter(round(b - a, 3) for a, b in zip(ts, ts[1:]))
        secs = [int(t * 1e9) // 1_000_000_000 for t in ts]
        shared = sum(1 for a, b in zip(secs, secs[1:]) if a == b)
        empty = sum(max(0, b - a - 1) for a, b in zip(secs, secs[1:]))
        top = ", ".join(f"{k}s:{v}" for k, v in sp.most_common(5))
        frac = sum(1 for t in ts if t != int(t))
        rep.append(f"  D5 {fn}: spacing {top} (distinct {len(sp)}); non-integer tick times {frac}; shared buckets {shared}; empty buckets {empty}")
    print("\n".join(rep))


main()
