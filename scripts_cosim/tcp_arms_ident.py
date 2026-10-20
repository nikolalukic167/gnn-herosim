#!/usr/bin/env python3
"""TCP-on arm study, identity cells (registration scientist/tcp-arms-reg, 80ad5181; cell list md5 3e94b9af3bcba01dbe87bfec79fce174).
One switch-off cell per arm whose filed switch-off cells sit at another pin (8: 24 FL MLP; 64 bar, FL gnn, FL MLP, twin; 128 bar, FL gnn, FL MLP),
rerun at e8b3f53c with the switch unset, against the filed cell of the same name. PASS per cell = the three per-task arrays (tasks, tasks_bsz,
tasks_lock) byte-equal, and every summary field of the filed cell equal in the rerun except provenance / wall clock / decision timing
(code, host, python_env, env_fingerprint, wallclock_s, decisionTiming, total_rtt_plus_inference). A field only the rerun carries must be one of the two
named additions (peerExchangeOverlap; `bytes` in a peerExchangeByAccessClass pair); anything else fails loud (unknown is not a pass). A field only the
filed cell carries fails. The rerun must be at e8b3f53c, clean, with no peerExchangeTcp block and no TCP switch in its env, and, for a learned cell, the
served checkpoint's md5 (comb/md5_comball.json or fl3/md5_twin.json). This script opens only the identity cells and their filed twins. A size passes only if all its cells do."""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PIN_ON = "e8b3f53c8b2e1e3d8d5a586064fc4f15b39b2e50"
CELLS_MD5 = "3e94b9af3bcba01dbe87bfec79fce174"
VOLATILE = ("wallclock_s", "code", "python_env", "host", "env_fingerprint", "decisionTiming", "total_rtt_plus_inference")
ARRAYS = (".tasks.npy", ".tasks_bsz.npy", ".tasks_lock.npy")
CKPT = {"ra_gnn_eng_combfl_nosplit": "uncut-comb-fl-gnn-eng-seed{}.pt", "ra_mlp_same_combfl_nosplit": "uncut-comb-fl-mlp-same-seed{}.pt",
        "ra_twin_eng_combfl_nosplit": "uncut-comb-fl-twin-seed{}.pt"}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def walk(a, b, p, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append((p + [k], "new", b[k]))
            elif k not in b:
                out.append((p + [k], "gone", a[k]))
            else:
                walk(a[k], b[k], p + [k], out)
    elif a != b:
        out.append((p, "diff", (a, b)))


def allowed_new(path):
    return path == ["peerExchangeOverlap"] or (len(path) == 3 and path[0] == "peerExchangeByAccessClass" and path[2] == "bytes")


def judge(new_stem, old_stem, md5s, ckpt_seen):
    s, o = json.load(open(new_stem + ".summary.json")), json.load(open(old_stem + ".summary.json"))
    why = []
    c = s["code"]
    if c["commit"] != PIN_ON or c["dirty"]:
        why.append(f"rerun pin {c['commit'][:8]} dirty={c['dirty']}")
    if "peerExchangeTcp" in s or any("TCP" in k for k in s["env"]):
        why.append("TCP block or switch in the rerun")
    arm = os.path.basename(new_stem).split("__")[2].rsplit("_k100_s", 1)
    if arm[0] in CKPT:
        path = s["env"].get("GNN_MODEL_PATH") or ""
        sd = arm[1]
        if os.path.basename(path) != CKPT[arm[0]].format(sd):
            why.append(f"model path {path}")
        else:
            ckpt_seen.setdefault(path, md5(path))
            if ckpt_seen[path] != md5s[f"{arm[0]}|s{sd}"]:
                why.append("checkpoint md5")
    out = []
    walk(o, s, [], out)
    news, other = [], []
    for p, kind, v in out:
        if p[0] in VOLATILE:
            continue
        if kind == "new" and allowed_new(p):
            news.append(".".join(map(str, p)))
        else:
            other.append(f"{kind} {'.'.join(map(str, p))}")
    if other:
        why.append(f"{len(other)} fields: {other[:5]}")
    bad = [e for e in ARRAYS if md5(new_stem + e) != md5(old_stem + e)]
    if bad:
        why.append(f"arrays differ: {bad}")
    return why, sorted(set(news)), {e: md5(new_stem + e)[:8] for e in ARRAYS}


def main(argv):
    U = os.environ.get("TCP_ARMS_U", "/share/nikola.lukic/uncut_groups")
    cells_file = argv[1]
    out_path = argv[2] if len(argv) > 2 else f"{U}/anomaly/tcp_arms_ident_check.txt"
    if hashlib.md5(open(cells_file, "rb").read()).hexdigest() != CELLS_MD5:
        raise SystemExit("FAIL LOUD: cell list md5")
    md5s = {**json.load(open(f"{U}/comb/md5_comball.json")), **json.load(open(f"{U}/fl3/md5_twin.json"))}
    rows = [ln.split("\t") for ln in open(cells_file).read().splitlines() if ln.startswith("tcp_arms/ident")]
    assert len(rows) == 8, f"FAIL LOUD: {len(rows)} identity entries, want 8"
    L, seen, size_ok = [__doc__.strip(), ""], {}, {}
    for root, name, flag, vs in rows:
        assert flag == "TCP=0" and vs.startswith("vs "), f"FAIL LOUD: {root} {name} {flag} {vs}"
        size = root.replace("tcp_arms/ident", "")
        why, news, arrs = judge(f"{U}/anomaly/{root}/{name}", f"{U}/{vs[3:]}/{name}", md5s, seen)
        size_ok[size] = size_ok.get(size, True) and not why
        L.append(f"  {'PASS' if not why else 'FAIL'}  {size:>3s}  {name}  vs {vs[3:]}  rerun-only fields {news}  arrays {arrs}" + (f"  WHY {why}" if why else ""))
    for size, ok in size_ok.items():
        L.append(f"=== size {size}: {'PASS' if ok else 'FAIL, stop this size'}")
    txt = "\n".join(L)
    print(txt)
    if os.path.exists(out_path):
        raise SystemExit("FAIL LOUD: output exists, written once")
    open(out_path + ".partial", "w").write(txt + "\n")
    os.replace(out_path + ".partial", out_path)
    if not all(size_ok.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv)
