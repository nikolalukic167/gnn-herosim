#!/usr/bin/env python3
"""uncut_groups_s0_v1 Arm 1 / Arm 2 live-gate read (uncut_arm1 / uncut_arm2), against the filed gate text (ad1f728a), its addenda (dcaa41d9
multiplicity and ids, 1c7c8682 MLP trained) and SCIENTIST's gate-reader checklist (items 1-17).

  uncut_arm_gate_read.py --gate <OUT>[,<OUT>] --topos 19537,...,19548 --topos-sha256 H --pin P
                         --primary uncut_arm1_gnn_eng --twin uncut_arm1_twin_eng [--mlp ra_mlp_same] [--reference ra_gnn_eng_nosplit]
                         --checkpoint-md5 ckpt_md5.json[,ref_md5.json] [--suffix _k100] [--out read.json]
                         [--repo <git checkout> --w-pins W1[,W2]]

Cells: <gate>/cc40s<topo>__g<w><rung>__<arm><suffix>_s<seed>.summary.json, windows g0-g3, rungs moderate/heavy. Learned arms at seeds 1 and 2;
the slate_fix_v1 reference at seed 1; hand arms (cd_ib0, cd) at seed 0. Beside each summary: .log, .tasks.npy, .tasks_bsz.npy and .tasks_lock.npy.

Asserts (fail loud): every cell's code.commit is --pin and code.dirty is false, and so is its log's [CODE] banner;
w-only pins (pre-registered amendment, 2026-10-14): a learned cell may instead run at one of --w-pins, which must form an ancestry chain
--pin -> W1 -> W2 whose every `git diff --name-only --pin..Wi` is exactly scripts_cosim/decision_cost_w.json; every key already in the w file
is identical at the next pin (frozen once committed), every added key is an Arm 1 learned arm's (ARM1_W_KINDS, in w_ms_per_plan /
w_ms_per_plan_by_bucket, or a top-level _source_<kind> note), and an Arm 1 learned cell's own w key exists at its pin; hand cells stay at --pin; sha256 of the comma-joined --topos equals --topos-sha256;
every learned cell's checkpoint (env.GNN_MODEL_PATH) has the md5 the training log gives in --checkpoint-md5 ({"<arm>|s<seed>": md5};
the key is the gate arm, else the training log's name, ra_gnn_eng_arm1_nosplit -> gnn_eng; several files are merged and must agree);
every cell has a .tasks_bsz.npy with a groupSize and servedBatchSize per task (never filled in), and a .tasks_lock.npy on the same
rows with no unstamped (NaN) task (never counted as 0 lock wait).

Cell value: a learned arm's cell mean = the mean of its two seeds' cell means (whole-run averageElapsedTime, after-360 s steady_state.after_360s.mean);
a cell with either seed missing is missing for the arm. Each seed is also read alone. --seeds (default 1,2; the seed extension of 2026-10-17
reads its new seeds 11-18 with it; --bar cd_ib0_inflightfull, default cd_ib0, names the hand bar) replaces the learned arms' seed set everywhere 'two seeds' is said here; a learned cell at any other seed fails loud.
Statistic: per topology the median over the windows present of 100 (arm - bar) / bar; then the median over topologies, the sign count, an exact
two-sided Wilcoxon (r1_attribution_v1_read.exact_wilcoxon) and Holm (r1_attribution_v1_read.holm).

F1 (Holm over 4: primary vs cd_ib0 x 2 rungs x {whole, after360}):
  WIN-ARM: median <= -2 % and Holm p < .05 on all four, each seed alone median <= -1 % on all four, the primary's collapse count <= cd_ib0's at each rung.
  Program-level WIN additionally needs Holm p < .025 on all four; otherwise WIN-ARM-ONLY (never quoted as the program result).
  Anchor: the primary's decision share must be >= cd_ib0's at each rung, else "F3 at k = 0.25 required" and no WIN label.
  Robust bar: if cd_ib0 collapses at a rung, the WIN rule is re-read against cd: WIN-ROBUST or WIN-OVER-BRITTLE-BAR.
  Otherwise TIES (no Holm-significant test), E-FASTER (cd_ib0 Holm-faster on all four), MIXED (anything else; the passing tests listed).
F2 (Holm over 8 with the MLP, 4 without): primary vs twin (and vs MLP) x 2 x 2. MP-ATTRIBUTED only if F1 is a program-level WIN and primary vs
  twin is <= -2 % with Holm p < .025 on all four; else MP-DIRECTION (median negative on all four) or NOT-SEPARATED. MODEL-CLASS (vs MLP) the same,
  labelled separately and naming the corpus.
Co-tenancy verdict (pre-registered 2026-10-14, --cotenancy required): every learned Arm 1 arm also runs as <arm>_kr, charged k_cotenancy x its
  frozen w (k_cotenancy = max(1, r_hand, r_learned), frozen in decision_cost_w.json with the learned w). F1/F2 are computed again on those cells
  against the same hand cells; the combined F1 is the weaker of the two under an explicit order over every F1 label (two WIN labels
  combine axis by axis, program-level and bar qualifier, keeping both qualifiers); a learned WIN counts only if both are WIN-class
  (NOT-COTENANCY-ROBUST flags a frozen-only WIN, COTENANCY-ANOMALY a co-tenancy-only one); F2 combined = the weaker of the two. Each learned cell
  must have served exactly k x the frozen w at its pin (k = 1, or k_cotenancy for _kr).
Arm 1b (pre-registered f6c1b85e; the MLP comparator added by the amendment of 2026-10-15): --primary ra_gnn_eng_arm1b_nosplit --twin ra_twin_eng_arm1b_nosplit
  [--mlp ra_mlp_same_arm1b_nosplit: F2 is then a Holm family of 8, as Arm 1's, and its w is guarded against Arm 1's ra_mlp_same_arm1_nosplit], both verdicts, read after
  Arm 1 with --after-read <Arm 1's read JSON> (fixed sequence: a program-level label only if Arm 1's combined F1 is program-level, else at best
  WIN-ARM-ONLY) and --compare ra_gnn_eng_arm1_nosplit (direction only: sign, topologies ahead, unadjusted p, descriptive). Its checkpoints
  come from its own md5 file, passed as arm1b=<path>; its w (ARM1B_W_KINDS) must sit within W_GUARD_X of Arm 1's at every pin (w_guard).
  The gate changes are --gate-change bf79865a,<Arm 1b kinds>.
Arm 3 (the expected move-regret lever, gate change G6): --primary ra_gnn_eng_arm3_nosplit --twin ra_twin_eng_arm3_nosplit, no MLP, both verdicts, read after
  Arm 1b with --after-read <Arm 1b's read JSON> (its primary must be ra_gnn_eng_arm1b_nosplit) and --compare ra_gnn_eng_arm1b_nosplit (direction only). B6: the lever
  was conceived after interim looks, so the program-level label is never open: at best WIN-ARM-ONLY. With no MP-OFF twin trained (wave 2 held) the read omits --twin: F1 against the bar is read as filed, F2 is skipped and says so ('no MP attribution'). Its checkpoints come from its own md5 file (arm3=<path>); its w
  (ARM3_W_KINDS) must sit within W_GUARD_X of Arm 1b's same arm at every pin. The gate changes are --gate-change <...>,8dfd7e0c,<G6>.
Descriptive: UNCUT-ATTRIBUTED (share of the task-weighted primary - cd_ib0 latency gap carried by tasks whose original group > 10, from the per-task
  sidecars, tasks matched by taskId, the primary per task = the mean of its two seeds; >= 50 % UNCUT-ATTRIBUTED else WIN-NOT-UNCUT; by bucket),
  OPENING share (tasks dispatched at t < 360 s; dispatchedTime is the arrival proxy), collapse (per seed cell, > 3 x cd's mean, either statistic; the arm's count is the worse seed's), tail flag
  (end - last arrival > 120 s), WALL-INFEASIBLE, decision share ((sched_queue_wait + sched_busy) / tasks / mean), and the training effect
  (primary vs the slate_fix_v1 reference).
"""
import argparse
import hashlib
import json
import os
import re
import statistics as st
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r1_attribution_v1_read import exact_wilcoxon, holm  # noqa: E402

FILED = {"gate_text": "ad1f728a", "addenda": ["dcaa41d9", "1c7c8682"]}
RUNGS = ("moderate", "heavy")
WINDOWS = ("g0", "g1", "g2", "g3")
STATS = ("whole", "after360")
BAR, CD = "cd_ib0", "cd"
SEEDS = (1, 2)
WIN_MED, SEED_MED, ALPHA, PROG_ALPHA = -2.0, -1.0, 0.05, 0.025
COLLAPSE_X, TAIL_S, OPEN_T = 3.0, 120.0, 360.0
BUCKETS = (("le10", 0, 10), ("11_50", 11, 50), ("51_100", 51, 100), ("gt100", 101, 10 ** 9))
W_FILE = "scripts_cosim/decision_cost_w.json"
ARM1_W_KINDS = tuple(f"ra_{a}_arm1_nosplit" for a in ("gnn_eng", "twin_eng", "mlp_same"))
# Arm 1b (pre-registered f6c1b85e): its own w, each within W_GUARD_X of Arm 1's same arm (scalar and every bucket, both rungs), else stop
ARM1B_W_KINDS = tuple(f"ra_{a}_arm1b_nosplit" for a in ("gnn_eng", "twin_eng", "mlp_same"))
# the expected move-regret lever (Arm 3, gate change G6): gnn_eng and its MP-OFF twin, no MLP; each within W_GUARD_X of Arm 1b's same arm
ARM3_W_KINDS = tuple(f"ra_{a}_arm3_nosplit" for a in ("gnn_eng", "twin_eng"))
LEARNED_W_KINDS = ARM1_W_KINDS + ARM1B_W_KINDS + ARM3_W_KINDS
W_GUARD_X = 1.5
# co-tenancy amendment (2026-10-14) and Arm 1b: the gate-code commits between --pin and the w-only pins (bf79865a the _kr tag, then the
# Arm 1b kinds) each touch the gate and its test, nothing else
GATE_CHANGE_FILES = ("scripts_cosim/fresh_topo_burst_v1_gate.py", "tests/test_uncut_arm1_gate_stage.py")
# the K0 / Arm 2 chain (B3, the expert's ruling (c1), 2026-10-17): gate change G4 69354280 (the Arm 2-K0 kinds, ported from c34b17e3) touches
# the gate, the sidecheck and its own test. That exact set is admitted for that one commit only; every other gate change keeps GATE_CHANGE_FILES.
G4 = "6935428088592fad78ee2a786ad7d5fc1388e30b"
GATE_CHANGE_FILES_BY_COMMIT = {G4: ("scripts_cosim/fresh_topo_burst_v1_gate.py", "scripts_cosim/joint_burst_v2_sidecheck.py", "tests/test_arm2_gate.py")}
# Arm 2 and Arm 2-K0 (read under ruling 2, on the G4 chain): their w keys may enter a w-only pin only on a chain through G4
ARM2_W_KINDS = tuple(f"ra_{a}_eng_{k}_nosplit" for k in ("arm2", "arm2k0") for a in ("gnn", "twin"))
COT_KEYS = ("k_cotenancy", "_source_k_cotenancy")
KR = "_kr"
NAME = re.compile(r"^cc40s(\d+)__(g\d)(moderate|heavy)__(.+)_s(\d+)\.(summary|failed)\.json$")


def load(gates, suffix):
    cells, failed = {}, {}
    for g in [x for x in gates.split(",") if x]:
        for f in sorted(os.listdir(g)):
            m = NAME.match(f)
            if not m:
                continue
            arm = m.group(4)
            if suffix and arm.endswith(suffix):
                arm = arm[: -len(suffix)]
            key = (arm, int(m.group(5)), int(m.group(1)), m.group(2), m.group(3))
            if key in cells or key in failed:
                raise SystemExit(f"FAIL LOUD: duplicate cell {key}")
            p = os.path.join(g, f)
            with open(p) as fh:
                d = json.load(fh)
            if m.group(6) == "summary":
                d["_path"] = p
                cells[key] = d
            else:
                failed[key] = d
    return cells, failed


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log_banner(path):
    """(commit prefix, dirty) from the cell log's `[CODE] commit=<sha> ... dirty=<bool>` line; None without one. A second
    pin source beside the summary's `code` block (run_provenance.code is None in these summaries)."""
    if not os.path.exists(path):
        return None
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith("[CODE]"):
                f = dict(p.split("=", 1) for p in line.split()[1:] if "=" in p)
                return f.get("commit", ""), f.get("dirty")
    return None


def git(repo, *args):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"FAIL LOUD: git {' '.join(args)} in {repo}: {r.stderr.strip()}")
    return r.stdout


def w_frozen(prev, cur, sha, kinds=LEARNED_W_KINDS):
    """Every key of `prev` identical in `cur`; every key `cur` adds belongs to a learned arm of `kinds` (Arm 1, 1b and 3; Arm 2 and K0 on the G4 chain)."""
    for top, v in prev.items():
        if top not in cur:
            raise SystemExit(f"FAIL LOUD: {sha} drops {W_FILE} key {top!r}")
        if isinstance(v, dict):
            for k, x in v.items():
                if json.dumps(cur[top].get(k), sort_keys=True) != json.dumps(x, sort_keys=True):
                    raise SystemExit(f"FAIL LOUD: {sha} changes frozen {W_FILE} key {top}[{k!r}]")
            extra = sorted(set(cur[top]) - set(v) - set(kinds))
        else:
            extra = [] if json.dumps(cur[top], sort_keys=True) == json.dumps(v, sort_keys=True) else [top]
        if extra:
            raise SystemExit(f"FAIL LOUD: {sha} changes or adds {W_FILE} keys {top}{extra} outside the learned arms {kinds}")
    extra = sorted(set(cur) - set(prev) - {f"_source_{k}" for k in kinds} - set(COT_KEYS))
    if extra:
        raise SystemExit(f"FAIL LOUD: {sha} adds top-level {W_FILE} keys {extra}")
    k = cur.get("k_cotenancy")
    if "k_cotenancy" in cur and (type(k) not in (int, float) or not k >= 1.0):
        raise SystemExit(f"FAIL LOUD: {sha} k_cotenancy {k!r} is not a number >= 1")


def w_guard(w, sha):
    """Arm 1b's w (each arm, rung, and bucket where both have buckets) within W_GUARD_X of Arm 1's same arm, and Arm 3's of Arm 1b's, which must be in the same file."""
    for b in ARM1B_W_KINDS + ARM3_W_KINDS:
        if b not in w.get("w_ms_per_plan", {}):
            continue
        a = b.replace("_arm1b_", "_arm1_") if b in ARM1B_W_KINDS else b.replace("_arm3_", "_arm1b_")
        who, ref = ("Arm 1b", "Arm 1") if b in ARM1B_W_KINDS else ("Arm 3", "Arm 1b")
        if a not in w.get("w_ms_per_plan", {}):
            raise SystemExit(f"FAIL LOUD: {sha} has {who} w for {b} without {ref}'s {a} to guard it against")
        pairs = [(f"{r}", w["w_ms_per_plan"][b].get(r), w["w_ms_per_plan"][a].get(r)) for r in RUNGS]
        bb, ab = w.get("w_ms_per_plan_by_bucket", {}).get(b) or {}, w.get("w_ms_per_plan_by_bucket", {}).get(a) or {}
        for r in RUNGS:
            if (r in bb) != (r in ab) or (r in bb and set(bb[r]) != set(ab[r])):
                raise SystemExit(f"FAIL LOUD: {sha} {b} and {a} differ in their {r} buckets")
            pairs += [(f"{r}/{k}", bb[r][k], ab[r][k]) for k in (bb.get(r) or {})]
        for where, x, y in pairs:
            if x is None or y is None or not (float(x) > 0 and float(y) > 0) or max(x / y, y / x) > W_GUARD_X * (1 + 1e-9):
                raise SystemExit(f"FAIL LOUD: {sha} {who} w {b}[{where}] = {x!r} is not within {W_GUARD_X}x of {ref}'s {y!r}: stop")


def w_pin_chain(repo, base, w_pins, gate_change=""):
    """{full sha: the w file at it} for --pin, the gate-change commits (if any) and the w-only pins, after the amendments' ancestry,
    diff, freeze and Arm 1b guard checks. Each gate change is a child of the one before (the first of --pin) touching exactly
    GATE_CHANGE_FILES; every w-only pin descends from the last and differs from it only in the w file."""
    rev = lambda p: git(repo, "rev-parse", "--verify", p + "^{commit}").strip()
    changes = [rev(g) for g in gate_change.split(",") if g]
    shas = [rev(base)] + changes + [rev(p) for p in w_pins]
    root = shas[len(changes)]
    for parent, g in zip(shas, changes):
        if git(repo, "rev-parse", g + "^").strip() != parent:
            raise SystemExit(f"FAIL LOUD: the gate change {g} is not a child of {parent}")
        names = sorted(git(repo, "diff", "--name-only", parent, g).split())
        allowed = GATE_CHANGE_FILES_BY_COMMIT.get(g, GATE_CHANGE_FILES)
        if names != sorted(allowed):
            raise SystemExit(f"FAIL LOUD: the gate change {g} touches {names}, not exactly {list(allowed)}")
    for prev, cur in zip(shas, shas[1:]):
        if subprocess.run(["git", "-C", repo, "merge-base", "--is-ancestor", prev, cur]).returncode:
            raise SystemExit(f"FAIL LOUD: w-only pins are not a chain from {base}: {prev} is not an ancestor of {cur}")
    for cur in shas[shas.index(root) + 1:]:
        names = git(repo, "diff", "--name-only", root, cur).split()
        if names != [W_FILE]:
            raise SystemExit(f"FAIL LOUD: {cur} differs from {root} in {names}, not only {W_FILE}")
    ws = {sha: json.loads(git(repo, "show", f"{sha}:{W_FILE}")) for sha in shas}
    for prev, cur in zip(shas, shas[1:]):
        w_frozen(ws[prev], ws[cur], cur, LEARNED_W_KINDS + ARM2_W_KINDS if G4 in changes else LEARNED_W_KINDS)
    for sha in shas:
        w_guard(ws[sha], sha)
    return ws


def pin_match(commit, pins):
    """The pin a cell's commit (summary or banner, >= 7 characters) names, prefix either way; None for none."""
    c = str(commit or "")
    hit = [p for p in pins if len(c) >= 7 and (c.startswith(p) or p.startswith(c))]
    return hit[0] if len(hit) == 1 else None


def ckpt_arm(arm):
    """The training log's arm name for a gate arm: ra_gnn_eng_arm1_nosplit -> gnn_eng (S7's keys); an exact key wins."""
    arm = arm[3:] if arm.startswith("ra_") else arm
    for sfx in ("_nosplit", "_arm3", "_arm2k0", "_arm2", "_arm1b", "_arm1"):
        arm = arm[: -len(sfx)] if arm.endswith(sfx) else arm
    return arm


def ckpt_tag(arm):
    """The md5 namespace a gate arm's checkpoint is looked up in: "arm1b" for Arm 1b and "arm3" for Arm 3 (their own md5 files, passed as arm1b=<path> / arm3=<path>), else ""."""
    a = arm.removesuffix("_nosplit")
    return "arm1b" if a.endswith("_arm1b") else "arm3" if a.endswith("_arm3") else ""


def ckpt_md5s(spec):
    """--checkpoint-md5: comma-separated [tag=]path; a tagged file's keys become "<arm>@<tag>|s<seed>". Files must agree on shared keys."""
    ckpt = {}
    for item in [x for x in spec.split(",") if x]:
        tag, _, path = item.rpartition("=")
        with open(path) as fh:
            for k, v in json.load(fh).items():
                if k.startswith("_"):
                    continue
                if tag:
                    arm, _, sd = k.partition("|")
                    k = f"{arm}@{tag}|{sd}"
                if k in ckpt and ckpt[k] != v:
                    raise SystemExit(f"FAIL LOUD: checkpoint md5 files disagree on {k}")
                ckpt[k] = v
    return ckpt


def charge_check(key, s, w, k):
    """The cell served k x its arm's frozen w at its pin: the by-bucket rates when the arm has them at that rung, else the scalar."""
    env, rung = s.get("env") or {}, key[4]
    byb = (w.get("w_ms_per_plan_by_bucket", {}).get(key[0]) or {}).get(rung)
    close = lambda x, y: abs(float(x) - y) <= 1e-9 * max(1.0, abs(y))
    if byb is not None:
        got = json.loads(env.get("HEROSIM_DECISION_COST_MS_BY_BUCKET") or "null")
        want = {b: k * float(v) for b, v in byb.items()}
        ok = isinstance(got, dict) and set(got) == set(want) and all(close(got[b], want[b]) for b in want) and not env.get("HEROSIM_DECISION_COST_MS")
    else:
        got = env.get("HEROSIM_DECISION_COST_MS")
        ok = got is not None and close(got, k * float(w["w_ms_per_plan"][key[0]][rung]))
    if not ok:
        raise SystemExit(f"FAIL LOUD: {key} served decision cost {got!r}, not k = {k} x its frozen w")


def asserts(cells, pin, learned, ckpt, w_pins=None, kr=False):
    """w_pins: {full sha: w file} from w_pin_chain, --pin's own entry first; None without --w-pins (every cell at --pin).
    kr: these are the co-tenancy cells, charged at the pin's k_cotenancy instead of 1."""
    warmth = {}
    base = pin if w_pins is None else next(iter(w_pins))
    for key, s in cells.items():
        warmth.setdefault((s.get("env") or {}).get("HEROSIM_WARMTH_PHYSICS"), []).append(key)
        code = s.get("code") or {}
        allowed = list(w_pins) if (w_pins and key[0] in learned) else [base]
        at = pin_match(code.get("commit"), allowed)
        if at is None or code.get("dirty") is not False:
            raise SystemExit(f"FAIL LOUD: {key} code {code.get('commit')} dirty={code.get('dirty')} is not the gate pin {pin}"
                             + (f" or a w-only pin" if allowed != [base] else ""))
        if w_pins and key[0] in LEARNED_W_KINDS + ARM2_W_KINDS and key[0] not in w_pins[at].get("w_ms_per_plan", {}):
            raise SystemExit(f"FAIL LOUD: {key} ran at {at}, whose {W_FILE} has no calibrated w for {key[0]}")
        if w_pins and key[0] in LEARNED_W_KINDS + ARM2_W_KINDS:
            k = w_pins[at].get("k_cotenancy") if kr else 1.0
            if k is None:
                raise SystemExit(f"FAIL LOUD: {key} is a co-tenancy cell at {at}, whose {W_FILE} has no k_cotenancy")
            charge_check(key, s, w_pins[at], float(k))
        bsz = s["_path"].replace(".summary.json", ".tasks_bsz.npy")
        if not os.path.exists(bsz):
            raise SystemExit(f"FAIL LOUD: {key} has no per-task batch-size sidecar {bsz}")
        a = np.load(bsz)
        if a.ndim != 2 or a.shape[1] != 5 or (a[:, 3] < 0).any() or (a[:, 4] < 0).any():
            raise SystemExit(f"FAIL LOUD: {key} per-task sidecar lacks groupSize/servedBatchSize on some task")
        banner = log_banner(s["_path"].replace(".summary.json", ".log"))
        if banner is None or pin_match(banner[0], [at]) is None or banner[1] != "False":
            raise SystemExit(f"FAIL LOUD: {key} log banner {banner} is not the cell's pin {at} with dirty=False")
        lk_path = s["_path"].replace(".summary.json", ".tasks_lock.npy")
        if not os.path.exists(lk_path):
            raise SystemExit(f"FAIL LOUD: {key} has no per-task lock-wait sidecar {lk_path}")
        lk = np.load(lk_path)
        if lk.ndim != 2 or lk.shape[1] != 4 or lk.shape[0] != a.shape[0] or not np.array_equal(lk[:, 0], a[:, 0]):
            raise SystemExit(f"FAIL LOUD: {key} lock-wait sidecar rows do not follow the batch-size sidecar")
        n_unstamped = int(np.isnan(lk[:, 1:]).any(axis=1).sum())
        if n_unstamped:
            raise SystemExit(f"FAIL LOUD: {key} lock-wait sidecar has {n_unstamped} unstamped task(s); never counted as 0")
        if key[0] in learned:
            h, th = s.get("host"), (s.get("python_env") or {}).get("torch_num_threads")
            if not isinstance(h, dict) or not h.get("cpu_model") or not h.get("nproc") or th != 1:
                raise SystemExit(f"FAIL LOUD: {key} needs the gate's host block and 1 torch thread (calibration protocol); host={h!r} threads={th!r}")
            tag = ckpt_tag(key[0])
            want = ckpt.get(f"{ckpt_arm(key[0])}@{tag}|s{key[1]}") if tag else ckpt.get(f"{key[0]}|s{key[1]}", ckpt.get(f"{ckpt_arm(key[0])}|s{key[1]}"))
            path = (s.get("env") or {}).get("GNN_MODEL_PATH")
            if want is None or path is None or md5(path) != want:
                raise SystemExit(f"FAIL LOUD: {key} checkpoint {path} md5 does not match the training log ({want})")
    if None in warmth or len(warmth) != 1:
        raise SystemExit(f"FAIL LOUD: cells do not all record one HEROSIM_WARMTH_PHYSICS: "
                         f"{ {w: len(k) for w, k in warmth.items()} }, unrecorded e.g. {warmth.get(None, [])[:3]}")
    return next(iter(warmth))


def hosts(cells):
    """Per arm, the cells per (CPU model, SMT state, nproc) they ran on; hand cells filed before the host block count as unrecorded."""
    out = {}
    for key, s in cells.items():
        h = s.get("host") or {}
        v = f"{h.get('cpu_model', 'unrecorded')} | smt {h.get('smt_control', '?')} | nproc {h.get('nproc', '?')}"
        out.setdefault(key[0], {}).setdefault(v, 0)
        out[key[0]][v] += 1
    return out


def means(s):
    a = ((s.get("steady_state") or {}).get("after_360s") or {}).get("mean")
    if a is None:
        raise SystemExit(f"FAIL LOUD: {s['_path']} has no steady_state.after_360s.mean")
    return {"whole": float(s["averageElapsedTime"]), "after360": float(a)}


def cell_mean(cells, arm, seeds, t, w, r):
    ss = [cells.get((arm, sd, t, w, r)) for sd in seeds]
    if any(x is None for x in ss):
        return None
    ms = [means(x) for x in ss]
    return {k: sum(m[k] for m in ms) / len(ms) for k in STATS}


def contrast(cells, a, a_seeds, b, b_seeds, rung, topos, stat):
    per = {}
    for t in topos:
        v = []
        for w in WINDOWS:
            ma, mb = cell_mean(cells, a, a_seeds, t, w, rung), cell_mean(cells, b, b_seeds, t, w, rung)
            if ma is None or mb is None:
                continue
            v.append(100.0 * (ma[stat] - mb[stat]) / mb[stat])
        if v:
            per[t] = (st.median(v), len(v))
    vals = [x[0] for x in per.values()]
    return {"a": a, "a_seeds": list(a_seeds), "b": b, "rung": rung, "stat": stat, "n_topologies": len(vals), "n_cells": sum(x[1] for x in per.values()),
            "median_pct": st.median(vals) if vals else None, "faster": sum(1 for x in vals if x < 0), "p": exact_wilcoxon(vals) if vals else None,
            "per_topology": {str(t): round(x[0], 3) for t, x in sorted(per.items())}}


def family(entries):
    adj = holm({str(i): e["p"] for i, e in enumerate(entries)})
    for i, e in enumerate(entries):
        e["p_holm"] = 1.0 if adj[str(i)] is None else adj[str(i)]
    return entries


def four(cells, a, a_seeds, b, b_seeds, topos):
    return [contrast(cells, a, a_seeds, b, b_seeds, r, topos, s) for r in RUNGS for s in STATS]


def win_four(es, med, alpha):
    return all(e["median_pct"] is not None and e["median_pct"] <= med and e["p_holm"] < alpha for e in es)


def health(cells, failed, arm, seeds, topos):
    """Collapse is tested per seed cell against cd (the expert's ruling: a seed mean can hide one collapsing seed). Counts are kept per seed;
    the arm's collapse count for the health clause is the worse seed's."""
    out = {}
    for r in RUNGS:
        row = {"cells": 0, "missing": [], "per_seed": {sd: {"collapse": {k: 0 for k in STATS}, "collapse_any": 0} for sd in seeds},
               "tail_flag": 0, "wall_infeasible": 0, "other_failed": 0}
        for t in topos:
            for w in WINDOWS:
                ref = cell_mean(cells, CD, (0,), t, w, r)
                for sd in seeds:
                    f = failed.get((arm, sd, t, w, r))
                    if f is not None:
                        if str(f.get("why", "")).startswith("watchdog: projected"):
                            row["wall_infeasible"] += 1
                        else:
                            row["other_failed"] += 1
                    s = cells.get((arm, sd, t, w, r))
                    if s is None:
                        continue
                    m = means(s)
                    c = {k: bool(ref and m[k] > COLLAPSE_X * ref[k]) for k in STATS}
                    for k in STATS:
                        row["per_seed"][sd]["collapse"][k] += c[k]
                    row["per_seed"][sd]["collapse_any"] += any(c.values())
                    row["tail_flag"] += float(s["endTime"]) - float(s["arrival_end"]["last_arrival_s"]) > TAIL_S
                if cell_mean(cells, arm, seeds, t, w, r) is None:
                    row["missing"].append(f"{t}{w}")
                else:
                    row["cells"] += 1
        row["collapse_any"] = max(v["collapse_any"] for v in row["per_seed"].values())
        out[r] = row
    return out


def decision_share(cells, arm, seeds, topos):
    out = {}
    for r in RUNGS:
        v = []
        for t in topos:
            for w in WINDOWS:
                for sd in seeds:
                    s = cells.get((arm, sd, t, w, r))
                    if s is None:
                        continue
                    c = s["schedulerCounters"]
                    v.append(100.0 * (float(c["sched_queue_wait_seconds"]) + float(c["sched_busy_seconds"])) / int(s["num_tasks"]) / means(s)["whole"])
        out[r] = st.median(v) if v else None
    return out


def per_task(s):
    a = np.load(s["_path"].replace(".summary.json", ".tasks_bsz.npy"))
    return {int(r[0]): (float(r[2] - r[1]), float(r[1]), int(r[3])) for r in a}


def gap_attribution(cells, primary, topos):
    out = {}
    for r in RUNGS:
        tot = 0.0
        by = {b[0]: 0.0 for b in BUCKETS}
        opening = 0.0
        for t in topos:
            for w in WINDOWS:
                ps = [cells.get((primary, sd, t, w, r)) for sd in SEEDS]
                bs = cells.get((BAR, 0, t, w, r))
                if any(x is None for x in ps) or bs is None:
                    continue
                p1, p2, bb = per_task(ps[0]), per_task(ps[1]), per_task(bs)
                if not (p1.keys() == p2.keys() == bb.keys()):
                    raise SystemExit(f"FAIL LOUD: task ids differ between {primary} seeds and {BAR} at {t} {w} {r}")
                for tid, (lb, arr, g) in bb.items():
                    d = (p1[tid][0] + p2[tid][0]) / 2.0 - lb
                    tot += d
                    for name, lo, hi in BUCKETS:
                        if lo <= g <= hi:
                            by[name] += d
                    if arr < OPEN_T:
                        opening += d
        gt10 = sum(v for k, v in by.items() if k != "le10")
        share = gt10 / tot if tot else None
        out[r] = {"gap_seconds": tot, "by_bucket_seconds": by, "share_gt10": share,
                  "label": None if share is None else ("UNCUT-ATTRIBUTED" if share >= 0.5 else "WIN-NOT-UNCUT"),
                  "opening_share": opening / tot if tot else None}
    return out


def f1_label(f1, seeds_alone, h_primary, h_bar, anchor_ok, robust, prog_allowed=True):
    if win_four(f1, WIN_MED, ALPHA):
        seed_ok = all(all(e["median_pct"] is not None and e["median_pct"] <= SEED_MED for e in es) for es in seeds_alone.values())
        col_ok = all(h_primary[r]["collapse_any"] <= h_bar[r]["collapse_any"] for r in RUNGS)
        if not (seed_ok and col_ok):
            return "MIXED", f"WIN line met but seeds_alone_ok={seed_ok} collapse_ok={col_ok}"
        if not all(anchor_ok.values()):
            return "F3 at k = 0.25 required", "anchor clause: the primary's decision share is below cd_ib0's at a rung"
        prog = prog_allowed and win_four(f1, WIN_MED, PROG_ALPHA)
        base = "WIN" if prog else "WIN-ARM-ONLY"
        if any(h_bar[r]["collapse_any"] > 0 for r in RUNGS):
            base = ("WIN-ROBUST" if win_four(robust, WIN_MED, PROG_ALPHA if prog else ALPHA) else "WIN-OVER-BRITTLE-BAR") + ("" if prog else " (ARM-ONLY)")
        if not prog_allowed:
            return base, "fixed sequence: Arm 1 did not win at program level, so this arm's best label is ARM-ONLY"
        return base, "program-level (Holm < .025 on all four)" if prog else "clears .05 but not .025: never quoted as the program result"
    sig = [e for e in f1 if e["p_holm"] < ALPHA]
    if not sig:
        return "TIES", "no Holm-significant test"
    if all(e["median_pct"] is not None and e["median_pct"] > 0 and e["p_holm"] < ALPHA for e in f1):
        return "E-FASTER", "cd_ib0 Holm-faster on all four"
    return "MIXED", "passing: " + ", ".join(f"{e['rung']}/{e['stat']} {e['median_pct']:+.2f}% p_holm={e['p_holm']:.3g}" for e in sig)


def f2_label(es, f1_program_win):
    if f1_program_win and win_four(es, WIN_MED, PROG_ALPHA):
        return "ATTRIBUTED"
    if all(e["median_pct"] is not None and e["median_pct"] < 0 for e in es):
        return "DIRECTION"
    return "NOT-SEPARATED"


def direction(cells, a, b, topos):
    """Direction only (Arm 1b vs Arm 1, pre-registered): per rung x statistic the sign of the median, the topologies ahead and an
    unadjusted sign-flip p, labelled descriptive; no magnitude is reported."""
    out = []
    for e in four(cells, a, SEEDS, b, SEEDS, topos):
        m = e["median_pct"]
        out.append({"a": a, "b": b, "rung": e["rung"], "stat": e["stat"], "n_topologies": e["n_topologies"], "a_faster": e["faster"],
                    "direction": None if m is None else ("a faster" if m < 0 else "a slower" if m > 0 else "tie"),
                    "p_unadjusted_descriptive": e["p"]})
    return out


def verdict(cells, failed, a, topos, prog_allowed=True):
    """F1, F2, health, decision share and the descriptive block on one cell set. prog_allowed False: the fixed sequence closes the
    program-level label (Arm 1b when Arm 1 did not win at program level)."""
    ref_seeds = (1,)
    res = {}
    res["health"] = {arm: health(cells, failed, arm, sd, topos) for arm, sd in
                     [(a.primary, SEEDS)] + ([(a.twin, SEEDS)] if a.twin else []) + ([(a.mlp, SEEDS)] if a.mlp else []) + ([(a.reference, ref_seeds)] if a.reference else [])
                     + [(BAR, (0,)), (CD, (0,))]}
    res["decision_share_pct"] = {arm: decision_share(cells, arm, sd, topos) for arm, sd in
                                 [(a.primary, SEEDS)] + ([(a.twin, SEEDS)] if a.twin else []) + [(BAR, (0,)), (CD, (0,))] + ([(a.mlp, SEEDS)] if a.mlp else [])}
    f1 = family(four(cells, a.primary, SEEDS, BAR, (0,), topos))
    seeds_alone = {sd: four(cells, a.primary, (sd,), BAR, (0,), topos) for sd in SEEDS}
    robust = family(four(cells, a.primary, SEEDS, CD, (0,), topos))
    anchor = {r: (res["decision_share_pct"][a.primary][r] or 0) >= (res["decision_share_pct"][BAR][r] or 0) for r in RUNGS}
    lab, why = f1_label(f1, seeds_alone, res["health"][a.primary], res["health"][BAR], anchor, robust, prog_allowed)
    res["F1"] = {"tests": f1, "seeds_alone": seeds_alone, "vs_cd_robust": robust, "anchor_ok": anchor, "label": lab, "why": why}
    prog = prog_win(lab)
    if a.twin:
        f2 = four(cells, a.primary, SEEDS, a.twin, SEEDS, topos) + (four(cells, a.primary, SEEDS, a.mlp, SEEDS, topos) if a.mlp else [])
        family(f2)
        res["F2"] = {"holm_family": len(f2), "tests": f2, "MP": "MP-" + f2_label(f2[:4], prog)}
        if a.mlp:
            res["F2"]["MODEL-CLASS"] = f"MODEL-CLASS-{f2_label(f2[4:], prog)} (corpus: {a.corpus})"
    else:
        res["F2"] = {"holm_family": 0, "tests": [], "MP": NO_TWIN if SEEDS == (1, 2) else NO_TWIN_EXT}
    res["descriptive"] = {"gap_attribution": gap_attribution(cells, a.primary, topos)}
    if a.reference:
        res["descriptive"]["training_effect_vs_reference"] = four(cells, a.primary, SEEDS, a.reference, ref_seeds, topos)
    if a.compare:
        res["descriptive"]["direction_vs_compare"] = direction(cells, a.primary, a.compare, topos)
    return res


F2_ORDER = ("ATTRIBUTED", "DIRECTION", "NOT-SEPARATED")
NO_TWIN = "F2 skipped: no MP-OFF twin trained (wave 2 held); no MP attribution"
NO_TWIN_EXT = "F2 skipped: seed-extension read of an arm with no new-seed twin (registered 2026-10-17); MP is read in the extension's own family E2"


def weaker(x, y):
    """The weaker of two F2 labels (ATTRIBUTED > DIRECTION > NOT-SEPARATED), prefix and corpus kept from x."""
    rank = lambda lab: next(i for i, t in enumerate(F2_ORDER) if t in lab)
    return x if rank(x) >= rank(y) else x.replace(next(t for t in F2_ORDER if t in x), next(t for t in F2_ORDER if t in y))


# Every F1 label f1_label can emit, as (program-level?, bar qualifier) for the WIN class and as a rank for the rest. Strength order (strongest first):
# WIN-class above every non-WIN label; within WIN, program-level above ARM-ONLY and, on the bar axis, no qualifier > ROBUST > OVER-BRITTLE-BAR;
# non-WIN: "F3 at k = 0.25 required" > MIXED > TIES > E-FASTER.
BAR_Q = ("", "ROBUST", "OVER-BRITTLE-BAR")
NON_WIN = ("F3 at k = 0.25 required", "MIXED", "TIES", "E-FASTER")


def f1_parts(lab):
    """("win", program-level, bar-qualifier index) or ("non", rank); fails loud on a label it does not know."""
    if lab in NON_WIN:
        return ("non", NON_WIN.index(lab))
    if lab.startswith("WIN"):
        q = 2 if "OVER-BRITTLE-BAR" in lab else 1 if "ROBUST" in lab else 0
        known = {"WIN", "WIN-ARM-ONLY", "WIN-ROBUST", "WIN-OVER-BRITTLE-BAR", "WIN-ROBUST (ARM-ONLY)", "WIN-OVER-BRITTLE-BAR (ARM-ONLY)",
                 "WIN-ARM-ONLY; ROBUST", "WIN-ARM-ONLY; OVER-BRITTLE-BAR"}
        if lab in known:
            return ("win", "ARM-ONLY" not in lab, q)
    raise SystemExit(f"FAIL LOUD: unknown F1 label {lab!r}")


def win_label(prog, q):
    if prog:
        return "WIN" + (f"-{BAR_Q[q]}" if q else "")
    return "WIN-ARM-ONLY" + (f"; {BAR_Q[q]}" if q else "")


def prog_win(lab):
    p = f1_parts(lab)
    return p[0] == "win" and p[1] and p[2] != 2


def weaker_f1(x, y):
    """The weaker of two F1 labels under the order above; two WIN labels combine axis by axis, keeping both qualifiers."""
    px, py = f1_parts(x), f1_parts(y)
    if px[0] == py[0] == "win":
        return win_label(px[1] and py[1], max(px[2], py[2]))
    if px[0] == "win":
        return y
    if py[0] == "win":
        return x
    return x if px[1] >= py[1] else y


def combine(main, cot, a):
    """Pre-registered (2026-10-14, amended by SCIENTIST's sign-off conditions): the combined F1 is the weaker of the frozen and the
    co-tenancy verdicts; a learned win counts only if both are WIN-class. COTENANCY-ANOMALY flags a co-tenancy WIN without a frozen one
    (the label stays the frozen one)."""
    m, c = main["F1"]["label"], cot["F1"]["label"]
    f1 = weaker_f1(m, c)
    mw, cw = f1_parts(m)[0] == "win", f1_parts(c)[0] == "win"
    flags = (["NOT-COTENANCY-ROBUST"] if mw and not cw else []) + (["COTENANCY-ANOMALY"] if cw and not mw else [])
    why = f"weaker of frozen {m!r} and co-tenancy {c!r}" + (f"; {', '.join(flags)}" if flags else "")
    out = {"F1": f1, "why": why, "flags": flags, "MP": NO_TWIN if NO_TWIN in (main["F2"]["MP"], cot["F2"]["MP"]) else weaker(main["F2"]["MP"], cot["F2"]["MP"])}
    if a.mlp:
        out["MODEL-CLASS"] = weaker(main["F2"]["MODEL-CLASS"], cot["F2"]["MODEL-CLASS"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--topos", required=True)
    ap.add_argument("--topos-sha256", required=True)
    ap.add_argument("--pin", required=True)
    ap.add_argument("--primary", required=True)
    ap.add_argument("--twin", default="", help="required, except for an Arm 3 read before its MP-OFF twin is trained (F2 is then skipped and says so)")
    ap.add_argument("--mlp", default="")
    ap.add_argument("--reference", default="")
    ap.add_argument("--checkpoint-md5", required=True)
    ap.add_argument("--suffix", default="_k100")
    ap.add_argument("--corpus", default="uncut_arm1, 12,000 states")
    ap.add_argument("--out")
    ap.add_argument("--repo", default="")
    ap.add_argument("--w-pins", default="")
    ap.add_argument("--gate-change", default="", help="the gate-code commits between --pin and the w-only pins, in order (bf79865a the _kr tag, then the Arm 1b kinds)")
    ap.add_argument("--after-read", default="", help="Arm 1b: Arm 1's gate read JSON (--out of its own read); its combined F1 opens or closes the program-level label")
    ap.add_argument("--compare", default="", help="Arm 1b: Arm 1's primary, read direction-only against this primary")
    ap.add_argument("--bar", default="cd_ib0", help="the hand bar kind (the combined lever reads against cd_ib0_inflightfull); its cells sit at --pin")
    ap.add_argument("--seeds", default="1,2", help="the learned arms' checkpoint seeds (seed extension: 11-18 or 11-16; the pooled line lists all)")
    ap.add_argument("--cotenancy", choices=("required", "off"), default="required",
                    help="required (the gate read): the learned arms' <arm>_kr cells give the second verdict; off only for smokes")
    a = ap.parse_args(argv)
    global SEEDS, BAR
    if a.bar not in ("cd_ib0", "cd_ib0_inflightfull"):
        raise SystemExit(f"FAIL LOUD: --bar {a.bar!r}: cd_ib0 (the filed bar) or cd_ib0_inflightfull (the combined lever's)")
    BAR = a.bar
    seeds = tuple(int(x) for x in a.seeds.split(",") if x.isdigit())
    if len(seeds) != len([x for x in a.seeds.split(",") if x]) or not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 1:
        raise SystemExit(f"FAIL LOUD: --seeds {a.seeds!r}: distinct checkpoint seeds >= 1")
    SEEDS = seeds
    print(f"filed gate text {FILED['gate_text']}; addenda {', '.join(FILED['addenda'])}; gate pin {a.pin}")
    if hashlib.sha256(a.topos.encode()).hexdigest() != a.topos_sha256:
        raise SystemExit("FAIL LOUD: the gate topology list hash differs from the filed one")
    topos = [int(x) for x in a.topos.split(",")]
    cells, failed = load(a.gate, a.suffix)
    learned = [x for x in (a.primary, a.twin, a.mlp, a.reference) if x]
    stray = sorted({(k[0], k[1]) for k in list(cells) + list(failed) if (k[0][:-len(KR)] if k[0].endswith(KR) else k[0]) in (a.primary, a.twin, a.mlp) and k[1] not in SEEDS})
    if stray:
        raise SystemExit(f"FAIL LOUD: learned cells at seeds outside --seeds {list(SEEDS)}: {stray[:6]}")
    if not a.twin and ckpt_tag(a.primary) != "arm3" and SEEDS == (1, 2):
        raise SystemExit("FAIL LOUD: --twin is required (only an Arm 3 read, or a seed-extension read of an arm with no new-seed twin, may go without one)")
    ckpt = ckpt_md5s(a.checkpoint_md5)
    prog_allowed = True
    tags = {ckpt_tag(x) for x in (a.primary, a.twin) if x}
    if len(tags) > 1 or "arm3" in tags and a.mlp:
        raise SystemExit("FAIL LOUD: --primary and --twin must be one trainer's (Arm 3 has no MLP comparator)")
    arm1b, arm3 = "arm1b" in tags, "arm3" in tags
    if (arm1b or arm3) != bool(a.after_read) or (a.compare and not (arm1b or arm3)):
        raise SystemExit("FAIL LOUD: --after-read (and --compare) go with an Arm 1b or Arm 3 read and only with it; Arm 1b is read after Arm 1, Arm 3 after Arm 1b")
    if a.mlp and (ckpt_tag(a.mlp) == "arm1b") != arm1b:
        raise SystemExit("FAIL LOUD: --mlp must be the same trainer's MLP: ra_mlp_same_arm1b_nosplit for an Arm 1b read, an Arm 1 or RA MLP otherwise")
    if arm1b or arm3:
        if a.cotenancy != "required":
            raise SystemExit("FAIL LOUD: Arm 1b and Arm 3 are read under both verdicts")
        with open(a.after_read) as fh:
            prev = json.load(fh)
        before = "ra_gnn_eng_arm1b_nosplit" if arm3 else "ra_gnn_eng_arm1_nosplit"
        if prev.get("primary") != before or not isinstance(prev.get("combined"), dict):
            raise SystemExit(f"FAIL LOUD: {a.after_read} is not {'Arm 1b' if arm3 else 'Arm 1'}'s gate read under both verdicts")
        # B6: the lever was conceived after interim looks and gets no share of the program alpha: never a program-level label
        prog_allowed = prog_win(prev["combined"]["F1"]) and not arm3
        if a.compare:
            learned.append(a.compare)
    w_pins = None
    if a.w_pins:
        if not a.repo:
            raise SystemExit("FAIL LOUD: --w-pins needs --repo, a git checkout holding --pin and every w-only pin")
        w_pins = w_pin_chain(a.repo, a.pin, a.w_pins.split(","), a.gate_change)
    kr_keys = [k for k in list(cells) + list(failed) if k[0].endswith(KR)]
    bad = sorted({k[0] for k in kr_keys if k[0][:-len(KR)] not in learned})
    if bad:
        raise SystemExit(f"FAIL LOUD: co-tenancy cells {bad} are not of a learned arm")
    cells_kr = {(k[0][:-len(KR)],) + k[1:]: v for k, v in cells.items() if k[0].endswith(KR)}
    failed_kr = {(k[0][:-len(KR)],) + k[1:]: v for k, v in failed.items() if k[0].endswith(KR)}
    cells = {k: v for k, v in cells.items() if not k[0].endswith(KR)}
    failed = {k: v for k, v in failed.items() if not k[0].endswith(KR)}
    if a.cotenancy == "required" and not (cells_kr or failed_kr):
        raise SystemExit("FAIL LOUD: no co-tenancy (_kr) cells; the pre-registered read needs both verdicts")
    warmth = asserts(cells, a.pin, learned, ckpt, w_pins)
    if cells_kr:
        if asserts(cells_kr, a.pin, learned, ckpt, w_pins, kr=True) != warmth:
            raise SystemExit("FAIL LOUD: the co-tenancy cells record another HEROSIM_WARMTH_PHYSICS")
    cells_kr.update({k: v for k, v in cells.items() if k[0] not in learned})
    failed_kr.update({k: v for k, v in failed.items() if k[0] not in learned})
    res = {"filed": FILED, "pin": a.pin, "gate_change": a.gate_change, "topos": topos, "warmth_physics": warmth, "primary": a.primary, "twin": a.twin,
           "w_pins": {sha: sorted(k for k in LEARNED_W_KINDS + ARM2_W_KINDS if k in w.get("w_ms_per_plan", {})) for sha, w in (w_pins or {}).items()}}
    if SEEDS != (1, 2):
        res["seeds"] = list(SEEDS)
    if BAR != "cd_ib0":
        res["bar"] = BAR
    if arm1b or arm3:
        res["sequence"] = {"after_read": os.path.abspath(a.after_read), "after_read_md5": md5(a.after_read), ("arm1b_combined_F1" if arm3 else "arm1_combined_F1"): prev["combined"]["F1"],
                           "program_label_open": prog_allowed}
    res["hosts"] = hosts(cells)
    res.update(verdict(cells, failed, a, topos, prog_allowed))
    if a.cotenancy == "required":
        res["cotenancy"] = verdict(cells_kr, failed_kr, a, topos, prog_allowed)
        res["combined"] = combine(res, res["cotenancy"], a)
        lab, why, f2 = res["combined"]["F1"], res["combined"]["why"], res["combined"]
    else:
        res["cotenancy"] = "OFF (not the gate read)"
        lab, why, f2 = res["F1"]["label"], res["F1"]["why"], res["F2"]
    f1 = res["F1"]["tests"]
    txt = json.dumps(res, indent=1, default=str)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(txt)
    print(f"F1 {lab} ({why}); F2 {f2['MP']}" + (f"; {f2['MODEL-CLASS']}" if a.mlp else ""))
    for e in f1:
        print(f"  F1 {e['rung']:8s} {e['stat']:8s} median {e['median_pct']} faster {e['faster']}/{e['n_topologies']} p {e['p']} holm {e['p_holm']}")
    for r in RUNGS:
        g = res["descriptive"]["gap_attribution"][r]
        print(f"  {r}: gap {g['gap_seconds']:.1f} s, share in groups > 10 {g['share_gt10']}, opening {g['opening_share']} -> {g['label']}")
    return res


if __name__ == "__main__":
    main(sys.argv[1:])
