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
with --w-pins, cells of a learned kind that is not one of the read's arms (several kinds in one cell directory) pass the same learned-cell
asserts (w-pin chain, w charge, checkpoint md5, host block, warmth) and are then left out of everything the read computes and writes;
--confirm (FL confirmation, registered 2026-10-18): FL gnn [+ the FL MLP], both verdicts, fresh topologies only, B6 lifted for this family:
CONFIRMED iff the combined F1 is a program-level WIN. --stage primary computes F1 under both verdicts only (the MLP's cells are checked as
another learned kind) and writes once; --stage secondary adds F2-MC (FL gnn vs FL MLP, Holm 4, MODEL-CLASS rule) and the descriptive block
only once the primary read's md5 is in the lineage node at --record on --filed-ref, and only if its primary parts equal the filed ones;
--confirm2 (FL confirmation 2, registered 2026-10-18 at 2ea36408): as --confirm, but FL gnn seeds 3-6 only (--seeds 3,4,5,6, nothing else), the
program threshold Holm < .0125 (alpha split over the two confirmation families), topologies also fresh of confirmation 1 and the smoke id, and
no MLP cell in --gate; --stage secondary reads --desc-gate (FL gnn seeds 1-2 and the FL MLP seeds 1-2, checked as learned cells) for the
descriptive seeds-1-2-vs-bar line and F2-MC, FL gnn s3-6 vs FL MLP s1-2, p unadjusted and labelled DESCRIPTIVE (never a label);
--confirm-size64 (FL size-64, registered 2026-10-18): as --confirm2's staging and grid, but FL gnn seeds 1-2 (--seeds 1,2) on 64-server topologies,
Holm < .025, the guard also fresh of confirmation 2, the pilot 30011-30123 and the retired 20021-20032, label SIZE-64-CONFIRMED, every loaded cell recording servers = 64; --stage secondary reads --desc-gate (the
uncharged k = 0 cells of FL gnn s1-2 and the bar as bare kinds, the FL MLP s1-2 _k100) and --reactive-gate (reactive s0), descriptive only;
--confirm-fl-lambda (FL + lambda, registered 2026-10-19): a learned-vs-learned family, FL + lambda (ra_gnn_eng_flreg_nosplit) against its matched
lambda = 0 control FL (ra_gnn_eng_combfl_nosplit), seeds 1-2, Holm < .025, both verdicts, 24-server fresh topologies (every loaded cell records
servers = 24), --gate exactly the bars, FL + lambda and FL (both charges); the per-seed clause and the collapse clause are seed-matched (FL + lambda
seed s vs FL seed s); label LAMBDA-WIN or NOT-WIN; --stage secondary reads --desc-gate (FL MLP + lambda and the FL MLP, both charges) for the
descriptive rows (FL + lambda vs the bars, MLP + lambda vs MLP, FL + lambda vs MLP + lambda, MLP + lambda vs FL gnn under _k100), never a label;
--confirm-exchange-trend (exchange-model sensitivity, registered 2026-10-19 at 4721d545): --xs-root holds one directory per payload variant
(x0 x0p1 x0p3 x1 x3 tier0 azblob), each with cells_reactive, cells_bar and cells_learned; a topology is admitted for a variant iff all 8 of its
reactive cells record queue_share <= 0.80 (unknown fails); each variant's cells must be exactly the bar s0 and FL gnn / FL MLP s1-2 (_k100) on
its admitted ids, servers = 24; the primary is Page's L of the paired FL gnn vs bar % over x0.1 < x0.3 < x1 < x3 on the ids admitted at every
trend level (x3 dropped if > 4 fail; VOID if x0.1 / x0.3 / x1 loses > 4), exact by convolution (normal approximation if a topology ties),
one-sided upper, Holm 4; EXCHANGE-TREND iff Holm < .025 on both moderate tests; the secondary adds the descriptive rows;
--confirm-size128 (FL size-128, registered 2026-10-19 with SCIENTIST's addendum B): --stage primary is F1 exactly as --confirm-size64 (FL gnn s1-2 vs
cd_ib0_inflightfull, Holm 4 per charge, the program's win rule under both verdicts) with every cell at servers = 128, label SIZE-128-CONFIRMED (size
chosen after an exploratory pilot) or NOT-CONFIRMED; --stage secondary (after F1's md5 is filed) reads --mlp-gate (FL MLP s1-2, _k100 and _kr),
--desc-gate (the uncharged k = 0 FL gnn s1-2 and bar) and --reactive-gate, then F2 (FL gnn vs FL MLP, Holm 4 per charge): GNN-AHEAD-OF-MLP-128 iff
median < 0 and Holm < .025 on all four under both charges and, per test, the median over topologies of (gap_t + d_t) < 0, d_t the median over
windows of the seed-mean decision-charge miss (measured - charged decision seconds, % of task latency; whole run) of FL gnn minus FL MLP; else
DIRECTION-ONLY; plus a decision-charge row per charged cell (flag > 0.5 pp) and the descriptive rows;
--confirm-degree-trend (partner structure, registered 2026-10-19 with SCIENTIST's addendum B): --deg-root holds k1 k2 k4 k8, each with
inputs_group/wf1_<rung>/wl/grounded_<window>_n50000.json and cells_reactive, cells_bar, cells_learned (24 servers); VOID unless every level's
workloads carry the signed mint's grounded_workload_v1.partner_nested with the level's k (= peer_rule.partners), the minted nonpair_sha256 and the
built {rps, duration, events} hash are each equal across levels and every group holds its k = 2 bytes; admission as
P10 (a rung saturated on every topology is excluded and listed); the primary is Page's L of the paired FL gnn vs FL MLP % over k = 1 < 2 < 4 < 8 at
moderate, whole and after-360, Holm 2, on the ids admitted at every level (VOID if a level loses > 4); DEGREE-TREND iff Holm < .025 on both;
the secondary adds per-level descriptive rows (vs the bar, realised structure, exchange saved / net gained, decision-charge miss per arm);
--confirm-mp-twin (FL MP-OFF twin, registered 2026-10-19 at 0a251dd9): read only after FL + lambda's primary md5 is filed (--fll-primary-read,
--fll-record); --gate is FL + lambda's (only FL gnn s1-2 _k100 / _kr, cd_ib0_inflightfull and cd are taken, as their exact grid) and --twin-gate the
twin's own 384 cells (ra_twin_eng_combfl_nosplit s1-2 _k100 / _kr), every cell at servers = 24 on fl3 30149-30160; --w-pins ends P7, P7b and the
link P7 -> P7b may add only the twin's keys; every twin cell ran at P7b and served k x its w there; F2 = FL gnn vs the twin, Holm 4 per charge,
seed-matched: TWIN-FASTER / MP-ATTRIBUTED / MP-DIRECTION / TIE, the weaker of the two charges; the secondary adds the twin vs FL MLP and vs the bar
(--desc-gate, FL + lambda's FL MLP cells) and the twin at 64 servers (--desc64-gate, vs --s64-gate's bar and FL gnn), servers = 64;
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
  sidecars, tasks matched by taskId, the primary per task = the mean of its seeds (every served seed; through 4441fa39 only the first two); >= 50 % UNCUT-ATTRIBUTED else WIN-NOT-UNCUT; by bucket),
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
PROG_ALPHA_FILED = PROG_ALPHA
COLLAPSE_X, TAIL_S, OPEN_T = 3.0, 120.0, 360.0
BUCKETS = (("le10", 0, 10), ("11_50", 11, 50), ("51_100", 51, 100), ("gt100", 101, 10 ** 9))
W_FILE = "scripts_cosim/decision_cost_w.json"
ARM1_W_KINDS = tuple(f"ra_{a}_arm1_nosplit" for a in ("gnn_eng", "twin_eng", "mlp_same"))
# Arm 1b (pre-registered f6c1b85e): its own w, each within W_GUARD_X of Arm 1's same arm (scalar and every bucket, both rungs), else stop
ARM1B_W_KINDS = tuple(f"ra_{a}_arm1b_nosplit" for a in ("gnn_eng", "twin_eng", "mlp_same"))
# the expected move-regret lever (Arm 3, gate change G6): gnn_eng and its MP-OFF twin, no MLP; each within W_GUARD_X of Arm 1b's same arm
ARM3_W_KINDS = tuple(f"ra_{a}_arm3_nosplit" for a in ("gnn_eng", "twin_eng"))
# combined lever (G8 / G8b): F, L, FL, FL's MLP and the two FAST arms; their w sits at w-only pins on G8 / G8b, never on the older chains
COMB_W_KINDS = tuple(f"{k}_nosplit" for k in ("ra_gnn_eng_combf", "ra_gnn_eng_combl", "ra_gnn_eng_combfl", "ra_mlp_same_combfl", "ra_gnn_eng_a3l", "ra_gnn_eng_a3lft"))
# FL + lambda (G8c b8909f58): its two kinds' w enters a w-only pin on G8c; kept out of COMB_W_KINDS so the combined lever's B6 set is unchanged
FLREG_W_KINDS = ("ra_gnn_eng_flreg_nosplit", "ra_mlp_same_flreg_nosplit")
LEARNED_W_KINDS = ARM1_W_KINDS + ARM1B_W_KINDS + ARM3_W_KINDS + COMB_W_KINDS + FLREG_W_KINDS
W_GUARD_X = 1.5
# co-tenancy amendment (2026-10-14) and Arm 1b: the gate-code commits between --pin and the w-only pins (bf79865a the _kr tag, then the
# Arm 1b kinds) each touch the gate and its test, nothing else
GATE_CHANGE_FILES = ("scripts_cosim/fresh_topo_burst_v1_gate.py", "tests/test_uncut_arm1_gate_stage.py")
# the K0 / Arm 2 chain (B3, the expert's ruling (c1), 2026-10-17): gate change G4 69354280 (the Arm 2-K0 kinds, ported from c34b17e3) touches
# the gate, the sidecheck and its own test. That exact set is admitted for that one commit only; every other gate change keeps GATE_CHANGE_FILES.
G4 = "6935428088592fad78ee2a786ad7d5fc1388e30b"
# G8d 05ec45c1 (the FL MP-OFF twin kind, on G8c, before P7): the gate, the sidecheck (its v6 twin arm) and the gate test; admitted for that commit only
G8D = "05ec45c103214f7fa8d0617479a8aabeaeb2d444"
# P8.1 (FL + lambda): G8c was built on the P2 w-only pin 74ddf70e, so the FL + lambda chain is a459dd98 -> 095948ea -> f6818d09 -> 74ddf70e
# -> b8909f58 -> 05ec45c1 -> P7: a w-only link inside the gate-change sequence. 74ddf70e is admitted there with the w file as its only file,
# for that commit only; w_frozen still checks it against its parent like every other link.
P2 = "74ddf70e1effbdc1eea0a69b8573c81a5f23384e"
GATE_CHANGE_FILES_BY_COMMIT = {G4: ("scripts_cosim/fresh_topo_burst_v1_gate.py", "scripts_cosim/joint_burst_v2_sidecheck.py", "tests/test_arm2_gate.py"),
                               G8D: ("scripts_cosim/fresh_topo_burst_v1_gate.py", "scripts_cosim/joint_burst_v2_sidecheck.py", "tests/test_uncut_arm1_gate_stage.py"),
                               P2: (W_FILE,)}
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


# P5.1 (job 885487): the per-arrival reactive arm (knative_network) writes no groupSize / servedBatchSize to its sidecar (every task < 0)
PER_ARRIVAL_KINDS = ("reactive",)


def asserts(cells, pin, learned, ckpt, w_pins=None, kr=False, k0=False):
    """w_pins: {full sha: w file} from w_pin_chain, --pin's own entry first; None without --w-pins (every cell at --pin).
    kr: these are the co-tenancy cells, charged at the pin's k_cotenancy instead of 1. k0: uncharged cells (the bare kind), which must
    serve no decision-cost rate at all. A per-arrival kind (PER_ARRIVAL_KINDS) is not held to groupSize / servedBatchSize, which it never
    writes; its sidecar must still exist with 5 columns and one row per task, with both fields unset on every row (a filled one means a
    batched path served it), and every other assert holds."""
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
        if k0:
            env = s.get("env") or {}
            if env.get("HEROSIM_DECISION_COST_MS") or env.get("HEROSIM_DECISION_COST_MS_BY_BUCKET"):
                raise SystemExit(f"FAIL LOUD: {key} is a k = 0 cell but served a decision-cost rate")
        elif w_pins and key[0] in LEARNED_W_KINDS + ARM2_W_KINDS:
            k = w_pins[at].get("k_cotenancy") if kr else 1.0
            if k is None:
                raise SystemExit(f"FAIL LOUD: {key} is a co-tenancy cell at {at}, whose {W_FILE} has no k_cotenancy")
            charge_check(key, s, w_pins[at], float(k))
        bsz = s["_path"].replace(".summary.json", ".tasks_bsz.npy")
        if not os.path.exists(bsz):
            raise SystemExit(f"FAIL LOUD: {key} has no per-task batch-size sidecar {bsz}")
        a = np.load(bsz)
        if key[0] in PER_ARRIVAL_KINDS:
            if a.ndim != 2 or a.shape[1] != 5 or a.shape[0] != int(s["num_tasks"]):
                raise SystemExit(f"FAIL LOUD: {key} per-task sidecar is not 5 columns x one row per task ({a.shape}, num_tasks {s.get('num_tasks')})")
            if (a[:, 3] >= 0).any() or (a[:, 4] >= 0).any():
                raise SystemExit(f"FAIL LOUD: {key} is a per-arrival cell but its sidecar has a filled groupSize/servedBatchSize (a batched path served it)")
        elif a.ndim != 2 or a.shape[1] != 5 or (a[:, 3] < 0).any() or (a[:, 4] < 0).any():
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
                pts, bb = [per_task(x) for x in ps], per_task(bs)
                if any(pt.keys() != bb.keys() for pt in pts):
                    raise SystemExit(f"FAIL LOUD: task ids differ between {primary} seeds and {BAR} at {t} {w} {r}")
                for tid, (lb, arr, g) in bb.items():
                    # the mean over every served seed; for two seeds (0 + a) + b == a + b, so the filed two-seed reads are unchanged
                    d = sum(pt[tid][0] for pt in pts) / len(pts) - lb
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


FIXED_SEQ_WHY = "fixed sequence: Arm 1 did not win at program level, so this arm's best label is ARM-ONLY"
COMB_B6_WHY = "B6: the combined lever was conceived after interim looks, so its best label is ARM-ONLY"


def f1_label(f1, seeds_alone, h_primary, h_bar, anchor_ok, robust, prog_allowed=True, closed_why=FIXED_SEQ_WHY):
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
            return base, closed_why
        if PROG_ALPHA != PROG_ALPHA_FILED:
            return base, f"program-level (Holm < {PROG_ALPHA:g} on all four)" if prog else f"clears .05 but not {PROG_ALPHA:g}: never quoted as the program result"
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


def verdict(cells, failed, a, topos, prog_allowed=True, closed_why=FIXED_SEQ_WHY):
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
    lab, why = f1_label(f1, seeds_alone, res["health"][a.primary], res["health"][BAR], anchor, robust, prog_allowed, closed_why)
    res["F1"] = {"tests": f1, "seeds_alone": seeds_alone, "vs_cd_robust": robust, "anchor_ok": anchor, "label": lab, "why": why}
    prog = prog_win(lab)
    if a.twin:
        f2 = four(cells, a.primary, SEEDS, a.twin, SEEDS, topos) + (four(cells, a.primary, SEEDS, a.mlp, SEEDS, topos) if a.mlp else [])
        family(f2)
        res["F2"] = {"holm_family": len(f2), "tests": f2, "MP": "MP-" + f2_label(f2[:4], prog)}
        if a.mlp:
            res["F2"]["MODEL-CLASS"] = f"MODEL-CLASS-{f2_label(f2[4:], prog)} (corpus: {a.corpus})"
    elif a.mlp and getattr(a, "confirm", False):
        # F2-MC (FL confirmation): FL gnn vs FL MLP, its own Holm family of 4, no twin; ATTRIBUTED needs the primary program-level WIN
        mc = family(four(cells, a.primary, SEEDS, a.mlp, SEEDS, topos))
        res["F2"] = {"holm_family": len(mc), "tests": mc, "MP": NO_TWIN_COMB, "MODEL-CLASS": f"MODEL-CLASS-{f2_label(mc, prog)} (corpus: {a.corpus})"}
    else:
        res["F2"] = {"holm_family": 0, "tests": [], "MP": NO_TWIN_COMB if a.primary in COMB_W_KINDS else NO_TWIN if SEEDS == (1, 2) else NO_TWIN_EXT}
    res["descriptive"] = {"gap_attribution": gap_attribution(cells, a.primary, topos)}
    if a.reference:
        res["descriptive"]["training_effect_vs_reference"] = four(cells, a.primary, SEEDS, a.reference, ref_seeds, topos)
    if a.compare:
        res["descriptive"]["direction_vs_compare"] = direction(cells, a.primary, a.compare, topos)
    return res


F2_ORDER = ("ATTRIBUTED", "DIRECTION", "NOT-SEPARATED")
NO_TWIN = "F2 skipped: no MP-OFF twin trained (wave 2 held); no MP attribution"
NO_TWIN_COMB = "F2 skipped: the combined lever trains no MP-OFF twin (the user's ruling); no MP attribution"
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
    out = {"F1": f1, "why": why, "flags": flags, "MP": next((x for x in (main["F2"]["MP"], cot["F2"]["MP"]) if x.startswith("F2 skipped")), None) or weaker(main["F2"]["MP"], cot["F2"]["MP"])}
    if a.mlp:
        out["MODEL-CLASS"] = weaker(main["F2"]["MODEL-CLASS"], cot["F2"]["MODEL-CLASS"])
    return out


# FL confirmation (uncut_groups_s0_v1, registered 2026-10-18): FL gnn on fresh topologies, never pooled with the 12 gate or the 36 TE3 ids
CONFIRM_PRIMARY, CONFIRM_MLP = "ra_gnn_eng_combfl_nosplit", "ra_mlp_same_combfl_nosplit"
CONFIRM_USED_TOPOS = frozenset(range(19537, 19549)) | frozenset(range(19554, 19590))
CONFIRM_CORPUS = "11,073 states, 49 topologies, simulator labels"
CONFIRM_WHY = ("FL confirmation: CONFIRMED (program-level, B6 lifted for FL gnn and this family only) iff the program's win rule holds under "
               "both verdicts; else NOT-CONFIRMED and FL stays WIN-ARM-ONLY")


# FL confirmation 2 (registered 2026-10-18, 2ea36408; SCIENTIST's C1-C4): new seeds 3-6, alpha split over the two confirmation families
CONFIRM2_SEEDS = (3, 4, 5, 6)
CONFIRM2_MLP_SEEDS = (1, 2)
CONFIRM2_ALPHA = 0.0125
CONFIRM2_USED_TOPOS = CONFIRM_USED_TOPOS | frozenset(range(19707, 19731)) | frozenset({19461})
CONFIRM2_YES = "CONFIRMED (second registered confirmation; the first, 83fa5b8e, was NOT-CONFIRMED on one per-seed clause)"
CONFIRM2_ALPHA_LINE = "Holm < .0125 (\u03b1 split over two confirmation families)"
CONFIRM2_MC = "DESCRIPTIVE, unequal and different seeds (GNN s3\u20136 vs MLP s1\u20132)"
CONFIRM2_WHY = ("FL confirmation 2: " + CONFIRM2_YES + " iff the program's win rule holds under both verdicts at " + CONFIRM2_ALPHA_LINE
                + ", each of seeds 3-6 alone <= -1 % on all four; else NOT-CONFIRMED and FL stays WIN-ARM-ONLY")


# FL size-64 (registered 2026-10-18): FL gnn seeds 1-2, trained at 24 servers, against the in-flight bar at 64 servers
# 20021-20032: the first size-64 mint, rejected (floor 19976, not the registered 30123) and retired unread
SIZE64_USED_TOPOS = CONFIRM2_USED_TOPOS | frozenset(range(19953, 19977)) | frozenset(range(30011, 30124)) | frozenset(range(20021, 20033))
SIZE64_YES = "SIZE-64-CONFIRMED (size chosen after an exploratory pilot)"
SIZE64_WHY = ("FL size-64: " + SIZE64_YES + " iff the program's win rule (Holm < .025, each seed alone <= -1 %, collapses <= the bar's) holds "
              "under both verdicts; else NOT-CONFIRMED. A claim about 64 servers only; nothing at 24 servers changes")
SIZE64_REACTIVE = "reactive"
SIZE64_SERVERS = 64


def servers_or_fail(cells, where):
    """--confirm-size64 (SCIENTIST): every loaded cell records servers = 64 in its summary; another count or none fails loud."""
    bad = sorted((k, s.get("servers")) for k, s in cells.items() if s.get("servers") != SIZE64_SERVERS or type(s.get("servers")) is not int)
    if bad:
        raise SystemExit(f"FAIL LOUD: {where}: {len(bad)} cell(s) do not record servers = {SIZE64_SERVERS}, e.g. {bad[:3]}")


def confirm_args(a):
    """--confirm: the registered read's fixed arms, both verdicts, a staged read, fresh topologies only. --confirm2 as --confirm, at seeds
    3-6 only, fresh of confirmation 1 and the smoke id too; the MLP is not in --gate but in --desc-gate (--stage secondary)."""
    if sum(map(bool, (a.confirm, a.confirm2, a.confirm_size64))) > 1:
        raise SystemExit("FAIL LOUD: --confirm, --confirm2 and --confirm-size64 are separate families; pass one")
    if a.reactive_gate and not (a.confirm_size64 and a.stage == "secondary"):
        raise SystemExit("FAIL LOUD: --reactive-gate belongs to --confirm-size64 --stage secondary")
    if a.confirm2:
        if a.seeds != ",".join(map(str, CONFIRM2_SEEDS)):
            raise SystemExit(f"FAIL LOUD: --confirm2 reads seeds {list(CONFIRM2_SEEDS)} only (--seeds 3,4,5,6), not {a.seeds!r}")
        a.seeds = "1,2"  # the shared checks below; restored after them
    elif a.desc_gate and not a.confirm_size64:
        raise SystemExit("FAIL LOUD: --desc-gate belongs to --confirm2 / --confirm-size64 --stage secondary")
    if a.primary != CONFIRM_PRIMARY or a.twin or a.mlp not in ("", CONFIRM_MLP) or a.reference or a.compare or a.after_read:
        raise SystemExit(f"FAIL LOUD: --confirm reads --primary {CONFIRM_PRIMARY} [--mlp {CONFIRM_MLP}], no twin, reference, compare or after-read")
    if a.cotenancy != "required" or a.bar != "cd_ib0_inflightfull" or a.stage == "full" or a.seeds != "1,2":
        raise SystemExit("FAIL LOUD: --confirm needs --cotenancy required, --bar cd_ib0_inflightfull, --seeds 1,2 and --stage primary or secondary")
    if a.corpus not in ("uncut_arm1, 12,000 states", CONFIRM_CORPUS):
        raise SystemExit(f"FAIL LOUD: --confirm names the corpus {CONFIRM_CORPUS!r}")
    a.corpus = CONFIRM_CORPUS
    guard = SIZE64_USED_TOPOS if a.confirm_size64 else CONFIRM2_USED_TOPOS if a.confirm2 else CONFIRM_USED_TOPOS
    used = sorted(set(int(x) for x in a.topos.split(",")) & guard)
    if used:
        raise SystemExit(f"FAIL LOUD: --confirm{'-size64' if a.confirm_size64 else '2' if a.confirm2 else ''} topologies must be fresh; {used[:6]} are gate, TE3"
                         + (", confirmation, smoke or pilot ids" if a.confirm_size64 else ", confirmation 1 or smoke ids" if a.confirm2 else " ids"))
    if a.confirm_size64:
        if a.stage == "secondary" and not (a.desc_gate and a.reactive_gate):
            raise SystemExit("FAIL LOUD: --confirm-size64 --stage secondary needs --desc-gate (k = 0 cells, FL MLP) and --reactive-gate")
        if a.stage == "primary" and a.desc_gate:
            raise SystemExit("FAIL LOUD: --desc-gate is not read before the primary md5 is filed (seal)")
    if a.confirm2:
        a.seeds = ",".join(map(str, CONFIRM2_SEEDS))
        if a.stage == "secondary" and not a.desc_gate:
            raise SystemExit("FAIL LOUD: --confirm2 --stage secondary needs --desc-gate (FL gnn s1-2 and FL MLP s1-2)")
        if a.stage == "primary" and a.desc_gate:
            raise SystemExit("FAIL LOUD: --desc-gate is not read before the primary md5 is filed (C3)")
    if a.stage == "secondary" and not (a.primary_read and a.record and a.repo and a.mlp):
        raise SystemExit("FAIL LOUD: --stage secondary needs --primary-read, --record, --repo and --mlp")
    if a.stage == "primary" and (a.primary_read or a.record):
        raise SystemExit("FAIL LOUD: --primary-read / --record belong to --stage secondary")


def check_against_filed_primary(a, res, node="docs/lineages/uncut_groups_s0_v1.md"):
    """--stage secondary: the primary read's full md5 is in the lineage node at --record, --record is on --filed-ref, and this run's primary
    parts (F1 under both verdicts, the combined F1, the confirmation label) equal the filed primary read's."""
    m = md5(a.primary_read)
    body = git(a.repo, "show", f"{a.record}:{node}")
    if m not in body:
        raise SystemExit(f"FAIL LOUD: the primary read {a.primary_read} (md5 {m}) is not filed in {node} at {a.record}")
    if subprocess.run(["git", "-C", a.repo, "merge-base", "--is-ancestor", a.record, a.filed_ref]).returncode:
        raise SystemExit(f"FAIL LOUD: {a.record} is not on {a.filed_ref}: the primary read is not filed")
    with open(a.primary_read) as fh:
        p = json.load(fh)
    if p.get("confirmation", {}).get("stage") != "primary":
        raise SystemExit(f"FAIL LOUD: {a.primary_read} is not a primary-stage confirmation read")
    same = lambda x, y: json.dumps(x, sort_keys=True, default=str) == json.dumps(y, sort_keys=True, default=str)
    pairs = [(res["F1"], p["F1"]), (res["cotenancy"]["F1"], p["cotenancy"]["F1"]), (res["combined"]["F1"], p["combined"]["F1"]),
             (res["confirmation"]["label"], p["confirmation"]["label"]), (res["topos"], p["topos"])]
    if not all(same(x, y) for x, y in pairs):
        raise SystemExit(f"FAIL LOUD: the secondary run's primary parts differ from the filed primary read {a.primary_read}")
    res["confirmation"]["primary_read"] = {"path": os.path.abspath(a.primary_read), "md5": m, "record": a.record}


def grid_or_void(cells, failed, arms, topos, where):
    """--confirm2 (R1, the registration's VOID rule): the cell set is exactly the registered grid, every (kind, seed) of arms x topos x
    g0-g3 x both rungs present as a summary, nothing failed and nothing else in the directory; else the family is VOID and the read stops."""
    want = {(k, sd, t, w, r) for k, seeds in arms for sd in seeds for t in topos for w in WINDOWS for r in RUNGS}
    have = set(cells)
    missing, extra = sorted(want - have), sorted(have - want)
    if failed or missing or extra:
        raise SystemExit(f"FAIL LOUD: VOID: {where} is not the registered grid ({len(want)} cells): {len(have & want)} present, "
                         f"{len(missing)} missing e.g. {missing[:3]}, {len(failed)} failed e.g. {sorted(failed)[:3]}, {len(extra)} extra e.g. {extra[:3]}")
    return len(want)


def desc_confirm2(a, res, cells, cells_kr, topos, ckpt, w_pins, warmth, mlp):
    """--confirm2 --stage secondary: --desc-gate's FL gnn s1-2 and FL MLP s1-2 cells (both charges) pass the learned-cell asserts, then two
    descriptive lines per verdict: FL gnn s1-2 vs the bar, and F2-MC (FL gnn s3-6 vs FL MLP s1-2). p unadjusted; never a label."""
    dc, df = load(a.desc_gate, a.suffix)
    want = {(a.primary, sd) for sd in CONFIRM2_MLP_SEEDS} | {(mlp, sd) for sd in CONFIRM2_MLP_SEEDS}
    base_kind = lambda k: k[:-len(KR)] if k.endswith(KR) else k
    bad = sorted({(k[0], k[1]) for k in dc if (base_kind(k[0]), k[1]) not in want})
    if bad:
        raise SystemExit(f"FAIL LOUD: --desc-gate holds cells outside FL gnn / FL MLP seeds 1-2: {bad[:4]}")
    two = CONFIRM2_MLP_SEEDS
    res["confirmation"]["desc_grid_cells"] = grid_or_void(dc, df, [(a.primary, two), (a.primary + KR, two), (mlp, two), (mlp + KR, two)],
                                                          topos, "--desc-gate")
    d100 = {k: v for k, v in dc.items() if not k[0].endswith(KR)}
    dkr = {(k[0][:-len(KR)],) + k[1:]: v for k, v in dc.items() if k[0].endswith(KR)}
    if not d100 or not dkr:
        raise SystemExit("FAIL LOUD: --desc-gate needs both _k100 and _kr cells")
    for part, kr in ((d100, False), (dkr, True)):
        if asserts(part, a.pin, [a.primary, mlp], ckpt, w_pins, kr=kr) != warmth:
            raise SystemExit("FAIL LOUD: the --desc-gate cells record another HEROSIM_WARMTH_PHYSICS")
    for v, base in ((res, cells), (res["cotenancy"], cells_kr)):
        dd = d100 if v is res else dkr
        merged = {**base, **{k: x for k, x in dd.items() if k[0] == a.primary}}
        withm = {**base, **{k: x for k, x in dd.items() if k[0] == mlp}}
        v.setdefault("descriptive", {})
        v["descriptive"]["seeds_1_2_vs_bar"] = {"note": "descriptive, p unadjusted; FL gnn seeds 1-2 on the confirmation-2 topologies",
                                                "tests": four(merged, a.primary, CONFIRM2_MLP_SEEDS, BAR, (0,), topos)}
        v["descriptive"]["F2_MC"] = {"label": CONFIRM2_MC, "note": "p unadjusted, never a label",
                                     "tests": four(withm, a.primary, SEEDS, mlp, CONFIRM2_MLP_SEEDS, topos)}


def desc_size64(a, res, cells, topos, ckpt, w_pins, warmth, mlp):
    """--confirm-size64 --stage secondary: --desc-gate holds exactly the uncharged k = 0 cells (FL gnn s1-2 and the bar, bare kinds) and the
    FL MLP s1-2 _k100 (480 cells for 12 topologies), --reactive-gate exactly reactive s0; each passes the cell asserts, then descriptive lines
    (p unadjusted, never a label): k = 0 FL vs k = 0 bar; FL MLP vs the bar and FL gnn vs FL MLP (_k100); reactive vs the bar."""
    dc, df = load(a.desc_gate, a.suffix)
    servers_or_fail(dc, "--desc-gate")
    for key, s in dc.items():
        charged = re.search(r"__.+_(k\d+|kr)_s\d+\.summary\.json$", os.path.basename(s["_path"]))
        if (key[0] == mlp) != bool(charged and charged.group(1) == a.suffix[1:]):
            raise SystemExit(f"FAIL LOUD: --desc-gate cell {os.path.basename(s['_path'])}: the k = 0 cells are bare kinds and the MLP is {a.suffix}")
    res["confirmation"]["desc_grid_cells"] = grid_or_void(dc, df, [(a.primary, SEEDS), (BAR, (0,)), (mlp, SEEDS)], topos, "--desc-gate")
    k0 = {k: v for k, v in dc.items() if k[0] != mlp}
    mc = {k: v for k, v in dc.items() if k[0] == mlp}
    if asserts(k0, a.pin, [a.primary], ckpt, w_pins, k0=True) != warmth or asserts(mc, a.pin, [mlp], ckpt, w_pins) != warmth:
        raise SystemExit("FAIL LOUD: the --desc-gate cells record another HEROSIM_WARMTH_PHYSICS")
    rc, rf = load(a.reactive_gate, a.suffix)
    servers_or_fail(rc, "--reactive-gate")
    res["confirmation"]["reactive_grid_cells"] = grid_or_void(rc, rf, [(SIZE64_REACTIVE, (0,))], topos, "--reactive-gate")
    if asserts(rc, a.pin, [], ckpt, w_pins) != warmth:
        raise SystemExit("FAIL LOUD: the --reactive-gate cells record another HEROSIM_WARMTH_PHYSICS")
    withm = {**cells, **mc}
    note = "descriptive, p unadjusted, never a label"
    res["descriptive"]["k0_fl_vs_bar"] = {"note": note + "; both arms uncharged (k = 0)", "tests": four(k0, a.primary, SEEDS, BAR, (0,), topos)}
    res["descriptive"]["mlp_vs_bar"] = {"note": note + "; _k100", "tests": four(withm, mlp, SEEDS, BAR, (0,), topos)}
    res["descriptive"]["fl_vs_mlp"] = {"note": note + "; _k100", "tests": four(withm, a.primary, SEEDS, mlp, SEEDS, topos)}
    res["descriptive"]["reactive_vs_bar"] = {"note": note + "; reactive uncharged, the bar _k100", "tests": four({**cells, **rc}, SIZE64_REACTIVE, (0,), BAR, (0,), topos)}


# FL + lambda (uncut_groups_s0_v1, registered 2026-10-19, signed against b6c57aaa): FL + lambda vs its lambda = 0 control FL, seed-matched clauses
FLL_PRIMARY, FLL_CONTROL = "ra_gnn_eng_flreg_nosplit", CONFIRM_PRIMARY
FLL_MLP, FLL_MLP_CONTROL = "ra_mlp_same_flreg_nosplit", CONFIRM_MLP
FLL_SEEDS = (1, 2)
FLL_SERVERS = 24
FLL_USED_TOPOS = SIZE64_USED_TOPOS | frozenset(range(30129, 30141))
FLL_YES = "LAMBDA-WIN"
FLL_WHY = ("FL + lambda: " + FLL_YES + " iff, under both verdicts, FL + lambda vs FL has median <= -2 % and Holm < .025 on all four, each seed s alone "
           "(FL + lambda s vs FL s) <= -1 % on all four, FL + lambda's collapses <= FL's seed by seed at each rung, and the anchor clause holds; else NOT-WIN")
FLL_ORDER = ("WIN", "MIXED", "TIES", "FL-FASTER")
FLL_DESC = "descriptive, p unadjusted, never a label"


def fll_args(a):
    """--confirm-fl-lambda: the registered arms, both verdicts, seeds 1-2, a staged read, fresh 24-server topologies."""
    if a.confirm or a.confirm2 or a.confirm_size64:
        raise SystemExit("FAIL LOUD: --confirm-fl-lambda is its own family; pass no other --confirm flag")
    if a.primary != FLL_PRIMARY or a.twin or a.reference or a.compare or a.after_read or a.reactive_gate:
        raise SystemExit(f"FAIL LOUD: --confirm-fl-lambda reads --primary {FLL_PRIMARY}; no twin, reference, compare, after-read or reactive-gate")
    if a.mlp not in ("", FLL_MLP):
        raise SystemExit(f"FAIL LOUD: --confirm-fl-lambda's --mlp is {FLL_MLP} (secondary only)")
    if a.cotenancy != "required" or a.bar != "cd_ib0_inflightfull" or a.stage not in ("primary", "secondary") or a.seeds != "1,2":
        raise SystemExit("FAIL LOUD: --confirm-fl-lambda needs --cotenancy required, --bar cd_ib0_inflightfull, --seeds 1,2 and --stage primary or secondary")
    if not (a.repo and a.w_pins):
        raise SystemExit("FAIL LOUD: --confirm-fl-lambda needs --repo and --w-pins (the G8c chain)")
    used = sorted(set(int(x) for x in a.topos.split(",")) & FLL_USED_TOPOS)
    if used:
        raise SystemExit(f"FAIL LOUD: --confirm-fl-lambda topologies must be fresh; {used[:6]} are gate, TE3, confirmation, smoke, pilot or size-64 ids")
    if a.stage == "primary" and (a.desc_gate or a.primary_read or a.record or a.mlp):
        raise SystemExit("FAIL LOUD: --desc-gate / --mlp / --primary-read / --record belong to --stage secondary (seal)")
    if a.stage == "secondary" and not (a.desc_gate and a.primary_read and a.record and a.mlp == FLL_MLP):
        raise SystemExit(f"FAIL LOUD: --confirm-fl-lambda --stage secondary needs --desc-gate, --primary-read, --record and --mlp {FLL_MLP}")


def servers_eq(cells, n, where):
    bad = sorted((k, s.get("servers")) for k, s in cells.items() if s.get("servers") != n or type(s.get("servers")) is not int)
    if bad:
        raise SystemExit(f"FAIL LOUD: {where}: {len(bad)} cell(s) do not record servers = {n}, e.g. {bad[:3]}")


def split_kr(cells):
    """(the _k100 cells, the _kr cells keyed by their base kind)."""
    return ({k: v for k, v in cells.items() if not k[0].endswith(KR)},
            {(k[0][:-len(KR)],) + k[1:]: v for k, v in cells.items() if k[0].endswith(KR)})


def fll_label(f1, seeds_alone, h_p, h_c, anchor_ok):
    """The registered win rule on FL + lambda vs FL; seed-matched per-seed and collapse clauses."""
    if win_four(f1, WIN_MED, PROG_ALPHA):
        seed_ok = all(e["median_pct"] is not None and e["median_pct"] <= SEED_MED for es in seeds_alone.values() for e in es)
        col_ok = all(h_p[r]["per_seed"][sd]["collapse_any"] <= h_c[r]["per_seed"][sd]["collapse_any"] for r in RUNGS for sd in FLL_SEEDS)
        if not (seed_ok and col_ok):
            return "MIXED", f"WIN line met but seeds_alone_ok={seed_ok} collapse_ok={col_ok} (seed-matched)"
        if not all(anchor_ok.values()):
            return "MIXED", "WIN line met but the anchor clause fails: FL + lambda's decision share is below the bar's at a rung"
        return "WIN", "every clause holds"
    sig = [e for e in f1 if e["p_holm"] < PROG_ALPHA]
    if not sig:
        return "TIES", "no Holm-significant test"
    if all(e["median_pct"] is not None and e["median_pct"] > 0 and e["p_holm"] < PROG_ALPHA for e in f1):
        return "FL-FASTER", "FL (lambda = 0) Holm-faster on all four"
    return "MIXED", "Holm-significant: " + ", ".join(f"{e['rung']}/{e['stat']} {e['median_pct']:+.2f}% p_holm={e['p_holm']:.3g}" for e in sig)


def fll_verdict(cells, failed, topos):
    """One charge: health, decision share, F1-lambda (Holm 4), the seed-matched seeds alone, robust rows are not part of this family."""
    arms = [(FLL_PRIMARY, FLL_SEEDS), (FLL_CONTROL, FLL_SEEDS), (BAR, (0,)), (CD, (0,))]
    res = {"health": {arm: health(cells, failed, arm, sd, topos) for arm, sd in arms},
           "decision_share_pct": {arm: decision_share(cells, arm, sd, topos) for arm, sd in arms}}
    f1 = family(four(cells, FLL_PRIMARY, FLL_SEEDS, FLL_CONTROL, FLL_SEEDS, topos))
    seeds_alone = {sd: four(cells, FLL_PRIMARY, (sd,), FLL_CONTROL, (sd,), topos) for sd in FLL_SEEDS}
    anchor = {r: (res["decision_share_pct"][FLL_PRIMARY][r] or 0) >= (res["decision_share_pct"][BAR][r] or 0) for r in RUNGS}
    lab, why = fll_label(f1, seeds_alone, res["health"][FLL_PRIMARY], res["health"][FLL_CONTROL], anchor)
    res["F1"] = {"contrast": f"{FLL_PRIMARY} vs {FLL_CONTROL} (seed-matched clauses)", "tests": f1, "seeds_alone": seeds_alone,
                 "anchor_ok": anchor, "label": lab, "why": why}
    return res


def fll_desc(a, res, cells, cells_kr, topos, ckpt, w_pins, warmth):
    """--stage secondary: --desc-gate is exactly FL MLP + lambda and the FL MLP, seeds 1-2, both charges (R1); each passes the learned-cell
    asserts; then the registered descriptive rows, p unadjusted, never a label."""
    dc, df = load(a.desc_gate, a.suffix)
    servers_eq(dc, FLL_SERVERS, "--desc-gate")
    arms = [(k, FLL_SEEDS) for k in (FLL_MLP, FLL_MLP + KR, FLL_MLP_CONTROL, FLL_MLP_CONTROL + KR)]
    res["confirmation"]["desc_grid_cells"] = grid_or_void(dc, df, arms, topos, "--desc-gate")
    d100, dkr = split_kr(dc)
    for part, kr in ((d100, False), (dkr, True)):
        if asserts(part, a.pin, [FLL_MLP, FLL_MLP_CONTROL], ckpt, w_pins, kr=kr) != warmth:
            raise SystemExit("FAIL LOUD: the --desc-gate cells record another HEROSIM_WARMTH_PHYSICS")
    for v, base, dd in ((res, cells, d100), (res["cotenancy"], cells_kr, dkr)):
        both = {**base, **dd}
        v["descriptive"] = {
            "fl_lambda_vs_bar": {"note": FLL_DESC, "tests": four(both, FLL_PRIMARY, FLL_SEEDS, BAR, (0,), topos)},
            "fl_lambda_vs_cd": {"note": FLL_DESC, "tests": four(both, FLL_PRIMARY, FLL_SEEDS, CD, (0,), topos)},
            "mlp_lambda_vs_mlp": {"note": FLL_DESC, "tests": four(both, FLL_MLP, FLL_SEEDS, FLL_MLP_CONTROL, FLL_SEEDS, topos)},
            "fl_lambda_vs_mlp_lambda": {"note": FLL_DESC + "; model class under lambda, no MP-OFF twin, so no MP claim",
                                        "tests": four(both, FLL_PRIMARY, FLL_SEEDS, FLL_MLP, FLL_SEEDS, topos)},
            "health": {arm: health(both, {}, arm, FLL_SEEDS, topos) for arm in (FLL_MLP, FLL_MLP_CONTROL)}}
    res["descriptive"]["mlp_lambda_vs_fl_gnn"] = {"note": FLL_DESC + "; _k100 only (SCIENTIST's prior 2: can the cost term let the pointwise model catch the current GNN)",
                                                  "tests": four({**cells, **d100}, FLL_MLP, FLL_SEEDS, FLL_CONTROL, FLL_SEEDS, topos)}


def read_fl_lambda(a):
    """--confirm-fl-lambda: the registered FL + lambda read (primary stage: F1-lambda under both verdicts only; secondary: the descriptive rows)."""
    global SEEDS, BAR, PROG_ALPHA
    fll_args(a)
    SEEDS, BAR, PROG_ALPHA = FLL_SEEDS, a.bar, PROG_ALPHA_FILED
    print(f"filed gate text {FILED['gate_text']}; addenda {', '.join(FILED['addenda'])}; gate pin {a.pin}")
    if hashlib.sha256(a.topos.encode()).hexdigest() != a.topos_sha256:
        raise SystemExit("FAIL LOUD: the gate topology list hash differs from the filed one")
    topos = [int(x) for x in a.topos.split(",")]
    cells, failed = load(a.gate, a.suffix)
    mc = sorted({k[0] for k in list(cells) + list(failed) if k[0].removesuffix(KR) in (FLL_MLP, FLL_MLP_CONTROL)})
    if mc:
        raise SystemExit(f"FAIL LOUD: --confirm-fl-lambda --gate holds MLP cells {mc[:3]}; they belong in --desc-gate (seal)")
    arms = [(FLL_PRIMARY, FLL_SEEDS), (FLL_PRIMARY + KR, FLL_SEEDS), (FLL_CONTROL, FLL_SEEDS), (FLL_CONTROL + KR, FLL_SEEDS), (BAR, (0,)), (CD, (0,))]
    grid_n = grid_or_void(cells, failed, arms, topos, "--gate")
    servers_eq(cells, FLL_SERVERS, "--gate")
    ckpt = ckpt_md5s(a.checkpoint_md5)
    w_pins = w_pin_chain(a.repo, a.pin, a.w_pins.split(","), a.gate_change)
    learned = [FLL_PRIMARY, FLL_CONTROL]
    cells, cells_kr = split_kr(cells)
    warmth = asserts(cells, a.pin, learned, ckpt, w_pins)
    if asserts(cells_kr, a.pin, learned, ckpt, w_pins, kr=True) != warmth:
        raise SystemExit("FAIL LOUD: the co-tenancy cells record another HEROSIM_WARMTH_PHYSICS")
    cells_kr.update({k: v for k, v in cells.items() if k[0] not in learned})
    res = {"filed": FILED, "family": "FL + lambda", "pin": a.pin, "gate_change": a.gate_change, "topos": topos, "warmth_physics": warmth,
           "primary": FLL_PRIMARY, "control": FLL_CONTROL, "seeds": list(FLL_SEEDS), "bar": BAR, "servers": FLL_SERVERS,
           "w_pins": {sha: sorted(k for k in LEARNED_W_KINDS + ARM2_W_KINDS if k in w.get("w_ms_per_plan", {})) for sha, w in w_pins.items()},
           "hosts": hosts({**cells, **{(k[0] + KR,) + k[1:]: v for k, v in cells_kr.items() if k[0] in learned}})}
    res.update(fll_verdict(cells, {}, topos))
    res["cotenancy"] = fll_verdict(cells_kr, {}, topos)
    m, c = res["F1"]["label"], res["cotenancy"]["F1"]["label"]
    lab = max(m, c, key=FLL_ORDER.index)
    res["combined"] = {"F1": lab, "why": f"weaker of frozen {m!r} and co-tenancy {c!r}"}
    res["confirmation"] = {"label": FLL_YES if lab == "WIN" else "NOT-WIN", "reader_label": lab, "stage": a.stage, "rule": FLL_WHY,
                           "family": "FL + lambda", "grid_cells": grid_n}
    if a.stage == "secondary":
        check_against_filed_primary(a, res)
        fll_desc(a, res, cells, cells_kr, topos, ckpt, w_pins, warmth)
    txt = json.dumps(res, indent=1, default=str)
    if a.out:
        if os.path.exists(a.out):
            raise SystemExit(f"FAIL LOUD: {a.out} exists; a confirmation read is written once")
        with open(a.out + ".tmp", "w") as fh:
            fh.write(txt)
        os.replace(a.out + ".tmp", a.out)
    print(f"FL + LAMBDA {res['confirmation']['label']} (reader: {lab}; {res['combined']['why']})")
    for tag, v in (("", res), ("_kr ", res["cotenancy"])):
        for e in v["F1"]["tests"]:
            print(f"  F1-lambda {tag}{e['rung']:8s} {e['stat']:8s} median {e['median_pct']} faster {e['faster']}/{e['n_topologies']} p {e['p']} holm {e['p_holm']}")
    return res


# Exchange-model sensitivity (uncut_groups_s0_v1, registered 2026-10-19, signed against 4721d545): FL gnn vs the in-flight bar across payload variants
XS_VARIANTS = ("x0", "x0p1", "x0p3", "x1", "x3", "tier0", "azblob")
XS_TREND = ("x0p1", "x0p3", "x1", "x3")
XS_LABEL = {"x0": "x0 (zero payload; latency and rendezvous remain)", "x0p1": "x0.1", "x0p3": "x0.3", "x1": "x1", "x3": "x3",
            "tier0": "heavy-tier 0 % at x1", "azblob": "azblob (per-access blob size as a proxy, not function-to-function exchange)"}
XS_FL, XS_MLP, XS_REACTIVE = CONFIRM_PRIMARY, CONFIRM_MLP, "reactive"
XS_QUEUE_MAX, XS_DROP_OVER = 0.80, 4
XS_USED_TOPOS = FLL_USED_TOPOS | frozenset(range(30149, 30161))
XS_YES, XS_NO = "EXCHANGE-TREND", "NO-TREND"


def xs_args(a):
    if a.confirm or a.confirm2 or a.confirm_size64 or a.confirm_fl_lambda:
        raise SystemExit("FAIL LOUD: --confirm-exchange-trend is its own family; pass no other --confirm flag")
    if a.primary != XS_FL or a.twin or a.reference or a.compare or a.after_read or a.reactive_gate or a.desc_gate or a.mlp:
        raise SystemExit(f"FAIL LOUD: --confirm-exchange-trend reads --primary {XS_FL} only; the variants' cells come from --xs-root")
    if a.bar != "cd_ib0_inflightfull" or a.stage not in ("primary", "secondary") or a.seeds != "1,2" or a.suffix != "_k100":
        raise SystemExit("FAIL LOUD: --confirm-exchange-trend needs --bar cd_ib0_inflightfull, --seeds 1,2, --suffix _k100 and --stage primary or secondary")
    if not (a.repo and a.w_pins and a.xs_root):
        raise SystemExit("FAIL LOUD: --confirm-exchange-trend needs --repo, --w-pins and --xs-root")
    used = sorted(set(int(x) for x in a.topos.split(",")) & XS_USED_TOPOS)
    if used:
        raise SystemExit(f"FAIL LOUD: --confirm-exchange-trend topologies must be fresh; {used[:6]} are used ids")
    if a.stage == "primary" and (a.primary_read or a.record):
        raise SystemExit("FAIL LOUD: --primary-read / --record belong to --stage secondary (seal)")
    if a.stage == "secondary" and not (a.primary_read and a.record):
        raise SystemExit("FAIL LOUD: --confirm-exchange-trend --stage secondary needs --primary-read and --record")


def xs_admission(rc, rf, topos):
    """Per topology: admitted iff all 8 reactive cells (g0-g3 x both rungs) are summaries whose queue_share is a number <= 0.80; a missing or
    failed cell, or one without the field, fails it. Returns (admitted ids, {id: reasons})."""
    bad = {}
    for t in topos:
        why = []
        for w in WINDOWS:
            for r in RUNGS:
                s = rc.get((XS_REACTIVE, 0, t, w, r))
                q = None if s is None else s.get("queue_share")
                if type(q) not in (int, float) or q > XS_QUEUE_MAX:
                    why.append(f"{w}{r}={'failed' if (XS_REACTIVE, 0, t, w, r) in rf else q}")
        if why:
            bad[t] = why
    return [t for t in topos if t not in bad], bad


def xs_topo_values(cells, rung, stat, topos):
    """{topology: median over g0-g3 of 100 (FL gnn seed mean - bar) / bar}, unrounded; every window must be present (the grid is exact)."""
    out = {}
    for t in topos:
        v = []
        for w in WINDOWS:
            ma, mb = cell_mean(cells, XS_FL, SEEDS, t, w, rung), cell_mean(cells, BAR, (0,), t, w, rung)
            if ma is None or mb is None:
                raise SystemExit(f"FAIL LOUD: {t}{w}{rung} has no FL gnn or bar cell mean")
            v.append(100.0 * (ma[stat] - mb[stat]) / mb[stat])
        out[t] = st.median(v)
    return out


def _midranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def page_trend(rows):
    """Page's L for rows of k values in the hypothesised order (level 1 .. k); the registered direction is 'more negative as the level grows',
    so within a row the most negative value gets the highest rank (ranks of -x). L = sum_j j * R_j. One-sided upper p: exact by convolution over
    rows (each row's k! orderings equally likely) when no row ties; with any tie, midranks and the normal approximation from the per-row
    permutation variance."""
    import itertools
    import math
    k, n = len(rows[0]), len(rows)
    ranks = [_midranks([-v for v in row]) for row in rows]
    L = sum((j + 1) * r[j] for r in ranks for j in range(k))
    tied = any(len(set(r)) < k for r in ranks)
    if not tied:
        one = {}
        for perm in itertools.permutations(range(1, k + 1)):
            s_ = sum((j + 1) * perm[j] for j in range(k))
            one[s_] = one.get(s_, 0) + 1
        dist = {0: 1}
        for _ in range(n):
            nxt = {}
            for a_, ca in dist.items():
                for b_, cb in one.items():
                    nxt[a_ + b_] = nxt.get(a_ + b_, 0) + ca * cb
            dist = nxt
        total = math.factorial(k) ** n
        p = sum(c for v, c in dist.items() if v >= round(L)) / total
        method = "exact"
    else:
        jbar = (k + 1) / 2.0
        mean = sum(k * jbar * (sum(r) / k) for r in ranks)
        sjj = sum((j + 1 - jbar) ** 2 for j in range(k))
        var = sum(sjj * sum((x - sum(r) / k) ** 2 for x in r) / (k - 1) for r in ranks)
        z = (L - mean) / math.sqrt(var) if var > 0 else 0.0
        p = 0.5 * math.erfc(z / math.sqrt(2.0))
        method = "normal approximation (a topology ties; midranks)"
    return {"L": L, "k": k, "n": n, "p": p, "method": method}


def xs_load_variant(a, v, topos, ckpt, w_pins, open_cells):
    """One variant: the reactive admission, and (open_cells) its bar and learned cells checked as the exact grid on its admitted ids."""
    d = os.path.join(a.xs_root, v)
    rc, rf = load(os.path.join(d, "cells_reactive"), a.suffix) if os.path.isdir(os.path.join(d, "cells_reactive")) else ({}, {})
    stray = sorted({k[0] for k in list(rc) + list(rf) if k[0] != XS_REACTIVE or k[1] != 0})
    if stray:
        raise SystemExit(f"FAIL LOUD: {v}/cells_reactive holds cells other than reactive s0: {stray[:3]}")
    servers_eq(rc, FLL_SERVERS, f"{v}/cells_reactive")
    admitted, bad = xs_admission(rc, rf, topos)
    out = {"admitted": admitted, "failing": {str(t): w for t, w in sorted(bad.items())}, "reactive_cells": len(rc), "reactive_failed": len(rf)}
    if not open_cells or not admitted:
        return out, None
    paths = [os.path.join(d, x) for x in ("cells_bar", "cells_learned")]
    if not all(os.path.isdir(p) for p in paths):
        raise SystemExit(f"FAIL LOUD: VOID: variant {v} has admitted ids {admitted} but no cells_bar / cells_learned")
    cells, failed = load(",".join(paths), a.suffix)
    out["grid_cells"] = grid_or_void(cells, failed, [(BAR, (0,)), (XS_FL, SEEDS), (XS_MLP, SEEDS)], admitted, f"variant {v} (its admitted ids)")
    servers_eq(cells, FLL_SERVERS, f"variant {v}")
    out["warmth"] = asserts(cells, a.pin, [XS_FL, XS_MLP], ckpt, w_pins)
    return out, cells


def xs_trend_family(adm, cells_by_v):
    """The registered trend: the levels kept (x3 dropped if > 4 fail), the trend ids, Page's L per rung x statistic, Holm 4, the label."""
    fails = {v: len(adm[v]["failing"]) for v in XS_TREND}
    levels = [v for v in XS_TREND if not (v == "x3" and fails[v] > XS_DROP_OVER)]
    if any(fails[v] > XS_DROP_OVER for v in XS_TREND if v != "x3"):
        return {"label": "VOID", "why": f"a trend level below x3 lost more than {XS_DROP_OVER} ids: {fails}", "failing_per_level": fails}
    ids = sorted(set.intersection(*[set(adm[v]["admitted"]) for v in levels]))
    tests = []
    for r in RUNGS:
        for stat in STATS:
            vals = {v: xs_topo_values(cells_by_v[v], r, stat, ids) for v in levels}
            pt = page_trend([[vals[v][t] for v in levels] for t in ids])
            pt.update({"rung": r, "stat": stat, "levels": levels,
                       "median_pct_by_level": {v: st.median(vals[v].values()) for v in levels},
                       "per_topology": {str(t): {v: round(vals[v][t], 3) for v in levels} for t in ids}})
            tests.append(pt)
    family(tests)
    mod = [e for e in tests if e["rung"] == "moderate"]
    yes = all(e["p_holm"] < PROG_ALPHA for e in mod)
    return {"label": XS_YES if yes else XS_NO, "why": "Holm < .025 on both moderate tests" if yes else "not Holm < .025 on both moderate tests",
            "levels": levels, "x3_dropped": "x3" not in levels, "failing_per_level": fails, "trend_ids": ids, "n": len(ids), "tests": tests}


def xs_gap(cells, topos):
    """Descriptive mechanism row: pooled over the variant's cells, the exchange seconds FL gnn saves against the bar vs its total latency gain."""
    ex = tot = 0.0
    for t in topos:
        for w in WINDOWS:
            for r in RUNGS:
                b = cells[(BAR, 0, t, w, r)]
                for sd in SEEDS:
                    f = cells[(XS_FL, sd, t, w, r)]
                    ex += (float(b["totalPeerExchangeTime"]) - float(f["totalPeerExchangeTime"])) / len(SEEDS)
                    tot += (float(b["averageElapsedTime"]) * int(b["num_tasks"]) - float(f["averageElapsedTime"]) * int(f["num_tasks"])) / len(SEEDS)
    return {"exchange_seconds_saved": ex, "total_latency_seconds_gained": tot, "share": ex / tot if tot else None}


def read_exchange_trend(a):
    """--confirm-exchange-trend: primary = admission + the trend family only (the trend levels' cells opened); secondary = the descriptive rows
    of every variant, after the primary read's md5 is filed."""
    global SEEDS, BAR, PROG_ALPHA
    xs_args(a)
    SEEDS, BAR, PROG_ALPHA = (1, 2), a.bar, PROG_ALPHA_FILED
    print(f"filed gate text {FILED['gate_text']}; addenda {', '.join(FILED['addenda'])}; gate pin {a.pin}")
    if hashlib.sha256(a.topos.encode()).hexdigest() != a.topos_sha256:
        raise SystemExit("FAIL LOUD: the topology list hash differs from the filed one")
    topos = [int(x) for x in a.topos.split(",")]
    extra = sorted(set(os.listdir(a.xs_root)) & set(XS_VARIANTS)) != sorted(XS_VARIANTS)
    if extra:
        raise SystemExit(f"FAIL LOUD: --xs-root must hold every variant directory {list(XS_VARIANTS)}")
    ckpt = ckpt_md5s(a.checkpoint_md5)
    w_pins = w_pin_chain(a.repo, a.pin, a.w_pins.split(","), a.gate_change)
    adm, cells_by_v = {}, {}
    for v in XS_VARIANTS:
        adm[v], _ = xs_load_variant(a, v, topos, ckpt, w_pins, open_cells=False)
    fails = {v: len(adm[v]["failing"]) for v in XS_TREND}
    trend_levels = [v for v in XS_TREND if not (v == "x3" and fails[v] > XS_DROP_OVER)]
    void = any(fails[v] > XS_DROP_OVER for v in XS_TREND if v != "x3")
    warm = set()
    for v in ([] if void else trend_levels):
        adm[v], cells_by_v[v] = xs_load_variant(a, v, topos, ckpt, w_pins, open_cells=True)
        warm.add(adm[v]["warmth"])
    if len(warm) > 1:
        raise SystemExit(f"FAIL LOUD: the variants record different HEROSIM_WARMTH_PHYSICS {sorted(warm)}")
    res = {"filed": FILED, "family": "exchange-model sensitivity", "pin": a.pin, "gate_change": a.gate_change, "topos": topos,
           "primary": XS_FL, "bar": BAR, "seeds": list(SEEDS), "servers": FLL_SERVERS, "variant_labels": XS_LABEL,
           "w_pins": {sha: sorted(k for k in LEARNED_W_KINDS + ARM2_W_KINDS if k in w.get("w_ms_per_plan", {})) for sha, w in w_pins.items()},
           "admission": adm, "trend": xs_trend_family(adm, cells_by_v)}
    res["confirmation"] = {"label": res["trend"]["label"], "stage": a.stage, "family": "exchange-model sensitivity",
                           "rule": "EXCHANGE-TREND iff Page's L (one-sided upper, Holm 4) has Holm < .025 on both moderate tests; else NO-TREND; VOID if a trend level below x3 loses > 4 ids"}
    if a.stage == "secondary":
        m = md5(a.primary_read)
        node = "docs/lineages/uncut_groups_s0_v1.md"
        if m not in git(a.repo, "show", f"{a.record}:{node}"):
            raise SystemExit(f"FAIL LOUD: the primary read {a.primary_read} (md5 {m}) is not filed in {node} at {a.record}")
        if subprocess.run(["git", "-C", a.repo, "merge-base", "--is-ancestor", a.record, a.filed_ref]).returncode:
            raise SystemExit(f"FAIL LOUD: {a.record} is not on {a.filed_ref}: the primary read is not filed")
        with open(a.primary_read) as fh:
            p = json.load(fh)
        same = lambda x, y: json.dumps(x, sort_keys=True, default=str) == json.dumps(y, sort_keys=True, default=str)
        if p.get("confirmation", {}).get("stage") != "primary" or not all(same(res[k], p[k]) for k in ("trend", "admission", "topos")) \
                or res["confirmation"]["label"] != p["confirmation"]["label"]:
            raise SystemExit(f"FAIL LOUD: the secondary run's primary parts differ from the filed primary read {a.primary_read}")
        res["confirmation"]["primary_read"] = {"path": os.path.abspath(a.primary_read), "md5": m, "record": a.record}
        desc, note = {}, "descriptive, p unadjusted, never a label"
        for v in XS_VARIANTS:
            if v in cells_by_v:
                cells, ids = cells_by_v[v], adm[v]["admitted"]
            elif adm[v]["admitted"] and len(adm[v]["failing"]) <= XS_DROP_OVER:
                adm_v, cells = xs_load_variant(a, v, topos, ckpt, w_pins, open_cells=True)
                if warm and adm_v["warmth"] not in warm:
                    raise SystemExit(f"FAIL LOUD: variant {v} records another HEROSIM_WARMTH_PHYSICS")
                cells_by_v[v], ids = cells, adm_v["admitted"]
                desc.setdefault("_grid_cells", {})[v] = adm_v["grid_cells"]
            else:
                desc[v] = {"note": "no cells read: " + (f"more than {XS_DROP_OVER} ids failed admission (not queued; disclosed, never re-scaled)"
                                                       if adm[v]["admitted"] else "no admitted id")}
                continue
            desc[v] = {"label": XS_LABEL[v], "n_topologies": len(ids), "note": note,
                       "fl_vs_bar": four(cells, XS_FL, SEEDS, BAR, (0,), ids), "mlp_vs_bar": four(cells, XS_MLP, SEEDS, BAR, (0,), ids),
                       "fl_vs_mlp": four(cells, XS_FL, SEEDS, XS_MLP, SEEDS, ids), "gap": xs_gap(cells, ids)}
        if "x0" in cells_by_v:
            meds = [e["median_pct"] for e in desc["x0"]["fl_vs_bar"]]
            desc["x0_prediction"] = {"rule": "|median FL gnn vs bar| <= 1 % on all four (registered prediction, not a gate)",
                                     "medians": meds, "holds": all(m_ is not None and abs(m_) <= 1.0 for m_ in meds)}
        if "tier0" in cells_by_v and "x1" in cells_by_v:
            ids = sorted(set(adm["tier0"]["admitted"]) & set(adm["x1"]["admitted"]))
            rows = []
            for r in RUNGS:
                for stat in STATS:
                    a0, a1 = xs_topo_values(cells_by_v["tier0"], r, stat, ids), xs_topo_values(cells_by_v["x1"], r, stat, ids)
                    d_ = [a0[t] - a1[t] for t in ids]
                    rows.append({"rung": r, "stat": stat, "n_topologies": len(ids), "median_pp_tier0_minus_x1": st.median(d_),
                                 "tier0_more_negative": sum(1 for x in d_ if x < 0), "p": exact_wilcoxon(d_)})
            desc["tier0_vs_x1"] = {"note": note + "; the paired FL gnn vs bar margin, tier0 minus x1, in percentage points", "tests": rows}
        res["descriptive"] = desc
    txt = json.dumps(res, indent=1, default=str)
    if a.out:
        if os.path.exists(a.out):
            raise SystemExit(f"FAIL LOUD: {a.out} exists; a confirmation read is written once")
        with open(a.out + ".tmp", "w") as fh:
            fh.write(txt)
        os.replace(a.out + ".tmp", a.out)
    tr = res["trend"]
    print(f"EXCHANGE {tr['label']} ({tr['why']})")
    for e in tr.get("tests", []):
        print(f"  Page {e['rung']:8s} {e['stat']:8s} L {e['L']} k {e['k']} n {e['n']} p {e['p']:.3g} holm {e['p_holm']:.3g} ({e['method']}) "
              f"medians {', '.join(f'{v} {m_:+.2f}' for v, m_ in e['median_pct_by_level'].items())}")
    return res


# FL size-128 (uncut_groups_s0_v1, registered 2026-10-19 with SCIENTIST's addendum B): FL gnn s1-2, trained at 24 servers, vs the in-flight bar at 128
S128_SERVERS = 128
# every id the uncut_groups node names at reference-physics 0f3d0de1 (5-digit tokens and ranges <= 120 wide in 19000-30999, md5/hex fragments
# excluded, minus the two runs it names as free, 30163-30174 and 30175-30186), extracted by script, not typed (SCIENTIST's ruling (e)); a superset
S128_NODE_IDS = frozenset().union(*(range(a, b + 1) for a, b in (
    (19206, 19217), (19300, 19305), (19400, 19432), (19440, 19452), (19461, 19549), (19553, 19590), (19627, 19627), (19635, 19658), (19692, 19692),
    (19695, 19695), (19707, 19730), (19800, 19805), (19807, 19807), (19815, 19815), (19820, 19820), (19843, 19843), (19855, 19855), (19862, 19862),
    (19868, 19868), (19879, 19879), (19884, 19884), (19898, 19898), (19953, 19976), (20021, 20032), (20118, 20158), (20182, 20182), (20400, 20439),
    (20450, 20461), (24617, 24617), (24890, 24890), (25845, 25868), (28090, 28090), (30011, 30124), (30128, 30141), (30147, 30160), (30162, 30163),
    (30175, 30188), (30200, 30205), (30212, 30213), (30221, 30239), (30253, 30253), (30258, 30258))))
S128_USED_TOPOS = (SIZE64_USED_TOPOS | frozenset(range(30129, 30141)) | frozenset(range(30149, 30161)) | frozenset(range(30176, 30188))
                   | frozenset(range(30200, 30240)) | S128_NODE_IDS)
S128_YES = "SIZE-128-CONFIRMED (size chosen after an exploratory pilot)"
S128_WHY = ("FL size-128: " + S128_YES + " iff the program's win rule (median <= -2 % and Holm < .025 on all four, each seed alone <= -1 %, collapses "
            "<= the bar's, the anchor clause) holds under both verdicts; else NOT-CONFIRMED. A claim about 128 servers only")
S128_F2_YES, S128_F2_NO = "GNN-AHEAD-OF-MLP-128", "DIRECTION-ONLY"
S128_F2_WHY = (S128_F2_YES + " iff, under both charges, FL gnn vs FL MLP has median < 0 and Holm < .025 (Holm 4 per charge) on all four, and on every "
               "test the median over topologies of (gap_t + d_t) < 0, d_t = the median over windows of the seed-mean decision-charge miss of FL gnn minus "
               "FL MLP in pp; else " + S128_F2_NO + ". Never 'message passing'")
S128_MISS_FLAG_PP = 0.5
S128_MISS_NOTE = ("decisionTiming is recorded for the whole run only, so the whole-run miss is used for both statistics; the charge is the 24-server "
                  "per-evaluation w, unchanged at 128 servers")


def s128_args(a):
    if a.confirm or a.confirm2 or a.confirm_size64 or a.confirm_fl_lambda or a.confirm_exchange_trend or a.confirm_degree_trend:
        raise SystemExit("FAIL LOUD: --confirm-size128 is its own family; pass no other --confirm flag")
    if a.primary != CONFIRM_PRIMARY or a.twin or a.reference or a.compare or a.after_read or a.xs_root:
        raise SystemExit(f"FAIL LOUD: --confirm-size128 reads --primary {CONFIRM_PRIMARY}; no twin, reference, compare, after-read or xs-root")
    if a.cotenancy != "required" or a.bar != "cd_ib0_inflightfull" or a.stage not in ("primary", "secondary") or a.seeds != "1,2" or a.suffix != "_k100":
        raise SystemExit("FAIL LOUD: --confirm-size128 needs --cotenancy required, --bar cd_ib0_inflightfull, --seeds 1,2, --suffix _k100 and --stage primary or secondary")
    if not (a.repo and a.w_pins):
        raise SystemExit("FAIL LOUD: --confirm-size128 needs --repo and --w-pins (the 24-server frozen w chain)")
    used = sorted(set(int(x) for x in a.topos.split(",")) & S128_USED_TOPOS)
    if used:
        raise SystemExit(f"FAIL LOUD: --confirm-size128 topologies must be fresh; {used[:6]} are ids another family, scan or token in the node uses")
    if a.stage == "primary" and (a.desc_gate or a.reactive_gate or a.mlp_gate or a.mlp or a.primary_read or a.record):
        raise SystemExit("FAIL LOUD: --mlp / --mlp-gate / --desc-gate / --reactive-gate / --primary-read / --record belong to --stage secondary (seal)")
    if a.stage == "secondary" and not (a.mlp == CONFIRM_MLP and a.mlp_gate and a.desc_gate and a.reactive_gate and a.primary_read and a.record):
        raise SystemExit(f"FAIL LOUD: --confirm-size128 --stage secondary needs --mlp {CONFIRM_MLP}, --mlp-gate, --desc-gate, --reactive-gate, --primary-read and --record")


def charge_miss(s, where):
    """One charged cell: the simulated decision seconds it was charged (its served rate(s) x plan evaluations, by bucket when served by bucket),
    the measured decision wall (decisionTiming.total_s), and the excess as % of the cell's task latency (num_tasks x mean)."""
    dt, env = s.get("decisionTiming") or {}, s.get("env") or {}
    bb = dt.get("by_bucket") or {}
    if type(dt.get("total_s")) not in (int, float) or not bb:
        raise SystemExit(f"FAIL LOUD: {where} has no decisionTiming total_s / by_bucket")
    if env.get("HEROSIM_DECISION_COST_MS_BY_BUCKET"):
        rates = json.loads(env["HEROSIM_DECISION_COST_MS_BY_BUCKET"])
        if not set(bb) <= set(rates):
            raise SystemExit(f"FAIL LOUD: {where} timed buckets {sorted(bb)} the served rates {sorted(rates)} do not cover")
        charged = sum(float(rates[b]) * int(v.get("plan_evals", 0)) for b, v in bb.items()) / 1e3
    elif env.get("HEROSIM_DECISION_COST_MS"):
        charged = float(env["HEROSIM_DECISION_COST_MS"]) * sum(int(v.get("plan_evals", 0)) for v in bb.values()) / 1e3
    else:
        raise SystemExit(f"FAIL LOUD: {where} is a charged cell but served no decision-cost rate")
    measured, lat = float(dt["total_s"]), int(s["num_tasks"]) * float(s["averageElapsedTime"])
    return {"charged_s": charged, "measured_s": measured, "excess_pp": 100.0 * (measured - charged) / lat}


def topo_pct(cells, a, a_seeds, b, b_seeds, rung, stat, topos):
    """{topology: median over g0-g3 of 100 (a - b) / b}, unrounded (the per-topology value contrast() rounds for display); every window present."""
    out = {}
    for t in topos:
        v = []
        for w in WINDOWS:
            ma, mb = cell_mean(cells, a, a_seeds, t, w, rung), cell_mean(cells, b, b_seeds, t, w, rung)
            if ma is None or mb is None:
                raise SystemExit(f"FAIL LOUD: {t}{w}{rung} has no {a} or {b} cell mean")
            v.append(100.0 * (ma[stat] - mb[stat]) / mb[stat])
        out[t] = st.median(v)
    return out


def s128_f2(cells, a, b, topos):
    """One charge: F2 (FL gnn vs FL MLP, Holm 4) and the signed decision-charge clause per test."""
    tests = family(four(cells, a, SEEDS, b, SEEDS, topos))
    miss = lambda arm, t, w, r: sum(charge_miss(cells[(arm, sd, t, w, r)], f"{arm} s{sd} {t}{w}{r}")["excess_pp"] for sd in SEEDS) / len(SEEDS)
    for e in tests:
        gap = topo_pct(cells, a, SEEDS, b, SEEDS, e["rung"], e["stat"], topos)
        d = {t: st.median([miss(a, t, w, e["rung"]) - miss(b, t, w, e["rung"]) for w in WINDOWS]) for t in topos}
        corr = [gap[t] + d[t] for t in topos]
        e["charge_clause"] = {"median_corrected_pct": st.median(corr), "median_miss_diff_pp": st.median(d.values()),
                              "holds": st.median(corr) < 0, "per_topology_miss_diff_pp": {str(t): round(d[t], 4) for t in topos}}
    ok = all(e["median_pct"] is not None and e["median_pct"] < 0 and e["p_holm"] < PROG_ALPHA and e["charge_clause"]["holds"] for e in tests)
    return {"tests": tests, "all_clauses": ok}


def s128_charge_rows(cells_by_charge):
    """Per charged cell: charged, measured, excess (pp of latency), flagged above 0.5 pp; and per arm x charge the measured/charged range."""
    rows, by = [], {}
    for charge, cells in cells_by_charge:
        for key in sorted(cells):
            s = cells[key]
            m = charge_miss(s, f"{key}")
            rows.append({"cell": os.path.basename(s["_path"]), "charge": charge, **m, "flag_over_0_5pp": m["excess_pp"] > S128_MISS_FLAG_PP})
            f = m["measured_s"] / m["charged_s"] if m["charged_s"] else None
            by.setdefault(f"{key[0]} {charge}", []).append((m["excess_pp"], f))
    summ = {k: {"cells": len(v), "excess_pp_max": max(x[0] for x in v), "excess_pp_median": st.median(x[0] for x in v),
                "measured_over_charged_min": min(x[1] for x in v if x[1] is not None), "measured_over_charged_max": max(x[1] for x in v if x[1] is not None)}
            for k, v in sorted(by.items())}
    return {"note": S128_MISS_NOTE, "flag_pp": S128_MISS_FLAG_PP, "flagged": sum(r["flag_over_0_5pp"] for r in rows), "by_arm": summ, "cells": rows}


def s128_secondary(a, res, cells, cells_kr, topos, ckpt, w_pins, warmth):
    """--stage secondary, opened only after the primary md5 is shown filed: --mlp-gate is exactly FL MLP s1-2 _k100 and _kr (384), --desc-gate
    exactly the uncharged k = 0 cells (FL gnn s1-2 and the bar, bare kinds; 288), --reactive-gate exactly reactive s0 (96); every cell at
    servers = 128 and through the asserts. Then F2 with its label, the per-cell decision-charge rows and the descriptive rows."""
    mlp = a.mlp
    mc, mf = load(a.mlp_gate, a.suffix)
    servers_eq(mc, S128_SERVERS, "--mlp-gate")
    res["confirmation"]["mlp_grid_cells"] = grid_or_void(mc, mf, [(mlp, SEEDS), (mlp + KR, SEEDS)], topos, "--mlp-gate")
    m100, mkr = split_kr(mc)
    if asserts(m100, a.pin, [mlp], ckpt, w_pins) != warmth or asserts(mkr, a.pin, [mlp], ckpt, w_pins, kr=True) != warmth:
        raise SystemExit("FAIL LOUD: the --mlp-gate cells record another HEROSIM_WARMTH_PHYSICS")
    dc, df = load(a.desc_gate, a.suffix)
    servers_eq(dc, S128_SERVERS, "--desc-gate")
    for key, s in dc.items():
        if re.search(r"__.+_(k\d+|kr)_s\d+\.summary\.json$", os.path.basename(s["_path"])):
            raise SystemExit(f"FAIL LOUD: --desc-gate cell {os.path.basename(s['_path'])}: the k = 0 cells are bare kinds")
    res["confirmation"]["desc_grid_cells"] = grid_or_void(dc, df, [(a.primary, SEEDS), (BAR, (0,))], topos, "--desc-gate")
    if asserts(dc, a.pin, [a.primary], ckpt, w_pins, k0=True) != warmth:
        raise SystemExit("FAIL LOUD: the --desc-gate cells record another HEROSIM_WARMTH_PHYSICS")
    rc, rf = load(a.reactive_gate, a.suffix)
    servers_eq(rc, S128_SERVERS, "--reactive-gate")
    res["confirmation"]["reactive_grid_cells"] = grid_or_void(rc, rf, [(SIZE64_REACTIVE, (0,))], topos, "--reactive-gate")
    if asserts(rc, a.pin, [], ckpt, w_pins) != warmth:
        raise SystemExit("FAIL LOUD: the --reactive-gate cells record another HEROSIM_WARMTH_PHYSICS")
    f100, fkr = s128_f2({**cells, **m100}, a.primary, mlp, topos), s128_f2({**cells_kr, **mkr}, a.primary, mlp, topos)
    res["F2_size128"] = {"family": "FL gnn vs FL MLP (model class), Holm 4 per charge, outside F1's Holm set", "rule": S128_F2_WHY, "miss_note": S128_MISS_NOTE,
                         "label": S128_F2_YES if f100["all_clauses"] and fkr["all_clauses"] else S128_F2_NO, "frozen": f100, "cotenancy": fkr}
    learned_bar = lambda c: {k: v for k, v in c.items() if k[0] in (a.primary, BAR, CD)}
    res["decision_charge"] = s128_charge_rows([("_k100", learned_bar(cells)), ("_k100", m100), ("_kr", {k: v for k, v in cells_kr.items() if k[0] == a.primary}),
                                               ("_kr", mkr)])
    note = "descriptive, p unadjusted, never a label"
    withm = {**cells, **m100}
    res["descriptive"]["k0_fl_vs_bar"] = {"note": note + "; both arms uncharged (k = 0)", "tests": four(dc, a.primary, SEEDS, BAR, (0,), topos)}
    res["descriptive"]["mlp_vs_bar"] = {"note": note + "; _k100", "tests": four(withm, mlp, SEEDS, BAR, (0,), topos)}
    res["descriptive"]["reactive_vs_bar"] = {"note": note + "; reactive uncharged, the bar _k100", "tests": four({**cells, **rc}, SIZE64_REACTIVE, (0,), BAR, (0,), topos)}


def read_size128(a):
    """--confirm-size128: --stage primary = F1 under both verdicts (FL gnn vs the in-flight bar, the size-64 rule at servers = 128); --stage secondary
    = F2 (FL gnn vs FL MLP) with its label, the per-cell decision-charge rows and the descriptive block, after F1's md5 is filed."""
    global SEEDS, BAR, PROG_ALPHA
    s128_args(a)
    SEEDS, BAR, PROG_ALPHA = (1, 2), a.bar, PROG_ALPHA_FILED
    print(f"filed gate text {FILED['gate_text']}; addenda {', '.join(FILED['addenda'])}; gate pin {a.pin}")
    if hashlib.sha256(a.topos.encode()).hexdigest() != a.topos_sha256:
        raise SystemExit("FAIL LOUD: the gate topology list hash differs from the filed one")
    topos = [int(x) for x in a.topos.split(",")]
    cells, failed = load(a.gate, a.suffix)
    mc = sorted({k[0] for k in list(cells) + list(failed) if k[0].removesuffix(KR) == CONFIRM_MLP})
    if mc:
        raise SystemExit(f"FAIL LOUD: --confirm-size128 --gate holds FL MLP cells {mc[:3]}; they belong in --mlp-gate (seal)")
    grid_n = grid_or_void(cells, failed, [(a.primary, SEEDS), (a.primary + KR, SEEDS), (BAR, (0,)), (CD, (0,))], topos, "--gate")
    servers_eq(cells, S128_SERVERS, "--gate")
    ckpt = ckpt_md5s(a.checkpoint_md5)
    w_pins = w_pin_chain(a.repo, a.pin, a.w_pins.split(","), a.gate_change)
    learned = [a.primary]
    cells, cells_kr = split_kr(cells)
    warmth = asserts(cells, a.pin, learned, ckpt, w_pins)
    if asserts(cells_kr, a.pin, learned, ckpt, w_pins, kr=True) != warmth:
        raise SystemExit("FAIL LOUD: the co-tenancy cells record another HEROSIM_WARMTH_PHYSICS")
    cells_kr.update({k: v for k, v in cells.items() if k[0] not in learned})
    v = argparse.Namespace(**{**vars(a), "mlp": "", "twin": "", "reference": "", "compare": "", "confirm": True})
    res = {"filed": FILED, "family": "FL size-128", "pin": a.pin, "gate_change": a.gate_change, "topos": topos, "warmth_physics": warmth,
           "primary": a.primary, "seeds": list(SEEDS), "bar": BAR, "servers": S128_SERVERS,
           "w_pins": {sha: sorted(k for k in LEARNED_W_KINDS + ARM2_W_KINDS if k in w.get("w_ms_per_plan", {})) for sha, w in w_pins.items()},
           "hosts": hosts({**cells, **{(k[0] + KR,) + k[1:]: x for k, x in cells_kr.items() if k[0] in learned}})}
    res.update(verdict(cells, {}, v, topos, True, CONFIRM_WHY))
    res["cotenancy"] = verdict(cells_kr, {}, v, topos, True, CONFIRM_WHY)
    res["combined"] = combine(res, res["cotenancy"], v)
    lab = res["combined"]["F1"]
    res["confirmation"] = {"label": S128_YES if prog_win(lab) else "NOT-CONFIRMED", "reader_label": lab, "stage": a.stage, "rule": S128_WHY,
                           "family": "FL size-128", "grid_cells": grid_n}
    if a.stage == "primary":
        for x in (res, res["cotenancy"]):
            x.pop("descriptive", None)
    else:
        check_against_filed_primary(a, res)
        s128_secondary(a, res, cells, cells_kr, topos, ckpt, w_pins, warmth)
    txt = json.dumps(res, indent=1, default=str)
    if a.out:
        if os.path.exists(a.out):
            raise SystemExit(f"FAIL LOUD: {a.out} exists; a confirmation read is written once")
        with open(a.out + ".tmp", "w") as fh:
            fh.write(txt)
        os.replace(a.out + ".tmp", a.out)
    print(f"SIZE-128 {res['confirmation']['label']} (reader: {lab}; {res['combined']['why']})")
    for tag, x in (("", res), ("_kr ", res["cotenancy"])):
        for e in x["F1"]["tests"]:
            print(f"  F1 {tag}{e['rung']:8s} {e['stat']:8s} median {e['median_pct']} faster {e['faster']}/{e['n_topologies']} p {e['p']} holm {e['p_holm']}")
    if "F2_size128" in res:
        f2 = res["F2_size128"]
        print(f"F2 {f2['label']} ({S128_MISS_NOTE})")
        for tag in ("frozen", "cotenancy"):
            for e in f2[tag]["tests"]:
                c = e["charge_clause"]
                print(f"  F2 {tag:9s} {e['rung']:8s} {e['stat']:8s} median {e['median_pct']} faster {e['faster']}/{e['n_topologies']} holm {e['p_holm']} "
                      f"miss diff {c['median_miss_diff_pp']:+.4f} pp corrected {c['median_corrected_pct']:+.3f} holds {c['holds']}")
        dcg = res["decision_charge"]
        print(f"  decision charge: {dcg['flagged']} cell(s) > {S128_MISS_FLAG_PP} pp; " + "; ".join(
            f"{k} max {x['excess_pp_max']:+.3f} pp, measured/charged {x['measured_over_charged_min']:.2f}-{x['measured_over_charged_max']:.2f}" for k, x in dcg["by_arm"].items()))
    return res


# Partner structure (uncut_groups_s0_v1, registered 2026-10-19 with SCIENTIST's addendum B): FL gnn vs FL MLP over partners drawn per task k
DEG_LEVELS = ("k1", "k2", "k4", "k8")
DEG_K = {"k1": 1, "k2": 2, "k4": 4, "k8": 8}
DEG_TOPOS_SHA256 = "dd359919d7fc42c5e0bec1f40d934edb6192b0993fa542cd08608b56e59cceb7"
DEG_META = "grounded_workload_v1"
DEG_BIG_BYTES = 50e6
DEG_YES, DEG_NO = "DEGREE-TREND", "NO-TREND"
DEG_WHY = (DEG_YES + " iff Page's L (one-sided upper over k = 1 < 2 < 4 < 8, the GNN's lead over the MLP growing with k; Holm 2) has Holm < .025 on both "
           "moderate tests; else " + DEG_NO + "; VOID if any level loses > 4 ids (a partially saturated heavy rung still removes topologies under P10's "
           "rule) or the levels' workloads are not identical apart from the pairs, or a group's bytes differ from its k = 2 total. Both "
           "models were trained at k = 2: k = 4 and 8 are out of distribution, which biases against the registered direction. Neither label says "
           "'message passing'")
DEG_IDENTITY = ("the mint's own grounded_workload_v1.partner_nested.nonpair_sha256 equal across the four levels (mint time), and the built file's "
                "{rps, duration, events} hashed with the minter's serialization (json.dumps sort_keys, default separators) equal across them (build "
                "time); the builder rewrites events (rate scale, task mix, single origin, client remap), so the built hash is not the minted one")
DEG_BYTES_RTOL = 1e-9


def deg_args(a):
    if a.confirm or a.confirm2 or a.confirm_size64 or a.confirm_fl_lambda or a.confirm_exchange_trend or a.confirm_size128:
        raise SystemExit("FAIL LOUD: --confirm-degree-trend is its own family; pass no other --confirm flag")
    if a.primary != CONFIRM_PRIMARY or a.mlp != CONFIRM_MLP or a.twin or a.reference or a.compare or a.after_read or a.reactive_gate or a.desc_gate \
            or a.mlp_gate or a.xs_root:
        raise SystemExit(f"FAIL LOUD: --confirm-degree-trend reads --primary {CONFIRM_PRIMARY} --mlp {CONFIRM_MLP}; the levels' cells come from --deg-root")
    if a.bar != "cd_ib0_inflightfull" or a.stage not in ("primary", "secondary") or a.seeds != "1,2" or a.suffix != "_k100":
        raise SystemExit("FAIL LOUD: --confirm-degree-trend needs --bar cd_ib0_inflightfull, --seeds 1,2, --suffix _k100 and --stage primary or secondary")
    if not (a.repo and a.w_pins and a.deg_root):
        raise SystemExit("FAIL LOUD: --confirm-degree-trend needs --repo, --w-pins and --deg-root")
    if a.topos_sha256 != DEG_TOPOS_SHA256:
        raise SystemExit("FAIL LOUD: --confirm-degree-trend reads the exchange-study ids 30176-30187 (sha256 dd359919...) only")
    if a.stage == "primary" and (a.primary_read or a.record):
        raise SystemExit("FAIL LOUD: --primary-read / --record belong to --stage secondary (seal)")
    if a.stage == "secondary" and not (a.primary_read and a.record):
        raise SystemExit("FAIL LOUD: --confirm-degree-trend --stage secondary needs --primary-read and --record")


def deg_workload(path):
    """One level's built workload: the signed mint's fields (grounded_workload_v1.partner_nested: level, connector_pairs, nonpair_sha256;
    peer_rule.partners), the build-time identity hash, the bytes per group and the realised structure."""
    with open(path) as fh:
        w = json.load(fh)
    g = w.get(DEG_META)
    pn = (g or {}).get("partner_nested")
    if not isinstance(pn, dict) or type(pn.get("level")) is not int or type(pn.get("connector_pairs")) is not int \
            or not isinstance(pn.get("nonpair_sha256"), str) or (g.get("peer_rule") or {}).get("partners") != pn["level"]:
        raise SystemExit(f"FAIL LOUD: {path} has no {DEG_META}.partner_nested (integer level and connector_pairs, nonpair_sha256) with "
                         f"peer_rule.partners equal to its level")
    built = hashlib.sha256(json.dumps({k: w[k] for k in ("rps", "duration", "events")}, sort_keys=True).encode()).hexdigest()
    ev = w["events"]
    n = len(ev)
    nb, pairs, total, big, by_group = {}, set(), 0.0, 0.0, {}
    for p in w["peer_exchange"]:
        i, j, b = int(p[0]), int(p[1]), float(p[2])
        e = (min(i, j), max(i, j))
        if i == j or e in pairs:
            raise SystemExit(f"FAIL LOUD: {path} peer_exchange has a self or duplicate pair {e}")
        if ev[i]["peer_group"] != ev[j]["peer_group"]:
            raise SystemExit(f"FAIL LOUD: {path} peer_exchange pair {e} crosses groups")
        pairs.add(e)
        nb.setdefault(i, set()).add(j)
        nb.setdefault(j, set()).add(i)
        total += b
        big += b if b > DEG_BIG_BYTES else 0.0
        by_group[ev[i]["peer_group"]] = by_group.get(ev[i]["peer_group"], 0.0) + b
    size = {}
    for e in ev:
        size[e["peer_group"]] = size.get(e["peer_group"], 0) + 1
    k = pn["level"]
    return {"minted": pn["nonpair_sha256"], "built": built, "bytes_by_group": by_group}, {
        "k": k, "connector_pairs": pn["connector_pairs"], "pairs": len(pairs), "pairs_per_task": len(pairs) / n,
        "mean_partners_per_task_with_partners": sum(len(v) for v in nb.values()) / len(nb) if nb else 0.0,
        "share_tasks_no_partner": (n - len(nb)) / n, "byte_share_over_50MB": big / total if total else None,
        "share_tasks_in_groups_le_k_plus_1": sum(1 for e in ev if size[e["peer_group"]] <= k + 1) / n, "total_bytes": total}


def deg_structure(a):
    """Every level's workloads (both rungs x g0-g3): the mint's level is the directory's, the minted and the built identity hashes are each
    equal across the four levels, and every group carries its k = 2 bytes at every level; else VOID."""
    ids, out = {}, {}
    for lv in DEG_LEVELS:
        rows = {}
        for r in RUNGS:
            for w in WINDOWS:
                p = os.path.join(a.deg_root, lv, "inputs_group", f"wf1_{r}", "wl", f"grounded_{w}_n50000.json")
                if not os.path.exists(p):
                    raise SystemExit(f"FAIL LOUD: VOID: level {lv} has no workload {p}")
                h, s = deg_workload(p)
                if s["k"] != DEG_K[lv]:
                    raise SystemExit(f"FAIL LOUD: VOID: {p} records level {s['k']}, the directory is {lv}")
                ids.setdefault((r, w), {})[lv] = h
                rows[f"{r}/{w}"] = s
        agg = {f: st.mean(x[f] for x in rows.values()) for f in ("pairs_per_task", "mean_partners_per_task_with_partners", "share_tasks_no_partner",
                                                                    "byte_share_over_50MB", "share_tasks_in_groups_le_k_plus_1")}
        out[lv] = {"mean_over_workloads": agg, "connector_pairs_total": sum(x["connector_pairs"] for x in rows.values()), "per_workload": rows}
    for kind in ("minted", "built"):
        bad = sorted(f"{r}/{w}" for (r, w), h in ids.items() if len({x[kind] for x in h.values()}) != 1)
        if bad:
            raise SystemExit(f"FAIL LOUD: VOID: the levels' {kind} identity hashes differ at {bad[:4]} ({DEG_IDENTITY})")
    for (r, w), h in ids.items():
        ref = h["k2"]["bytes_by_group"]
        for lv in DEG_LEVELS:
            got = h[lv]["bytes_by_group"]
            off = [gid for gid in set(ref) | set(got) if abs(got.get(gid, 0.0) - ref.get(gid, 0.0)) > DEG_BYTES_RTOL * max(1.0, abs(ref.get(gid, 0.0)))]
            if off:
                raise SystemExit(f"FAIL LOUD: VOID: {lv} {r}/{w}: {len(off)} group(s) do not carry their k = 2 bytes, e.g. group {sorted(off)[0]} "
                                 f"({got.get(sorted(off)[0], 0.0):.6g} vs {ref.get(sorted(off)[0], 0.0):.6g}); was the build's payload sampler legacy?")
    return {"identity": DEG_IDENTITY, "minted_nonpair_sha256": {f"{r}/{w}": h["k2"]["minted"] for (r, w), h in sorted(ids.items())},
            "built_sha256": {f"{r}/{w}": h["k2"]["built"] for (r, w), h in sorted(ids.items())},
            "bytes_per_group_held_at_k2": True, "levels": out}


def deg_admission(rc, rf, topos):
    """P10's per-topology admission over every reactive cell (queue_share <= 0.80; unknown fails), except that a rung saturated on every topology is
    excluded and listed, and admission is then read on the other rung's cells."""
    sat = lambda t, w, r: type((rc.get((XS_REACTIVE, 0, t, w, r)) or {}).get("queue_share")) not in (int, float) \
        or rc[(XS_REACTIVE, 0, t, w, r)]["queue_share"] > XS_QUEUE_MAX
    whole = [r for r in RUNGS if all(any(sat(t, w, r) for w in WINDOWS) for t in topos)]
    rungs = [r for r in RUNGS if r not in whole]
    bad = {}
    for t in topos:
        why = [f"{w}{r}={'failed' if (XS_REACTIVE, 0, t, w, r) in rf else (rc.get((XS_REACTIVE, 0, t, w, r)) or {}).get('queue_share')}"
               for w in WINDOWS for r in rungs if sat(t, w, r)]
        if why or not rungs:
            bad[t] = why or ["every rung saturated"]
    return [t for t in topos if t not in bad], bad, whole


def deg_load_level(a, lv, topos, ckpt, w_pins):
    d = os.path.join(a.deg_root, lv)
    rc, rf = load(os.path.join(d, "cells_reactive"), a.suffix) if os.path.isdir(os.path.join(d, "cells_reactive")) else ({}, {})
    stray = sorted({k[0] for k in list(rc) + list(rf) if k[0] != XS_REACTIVE or k[1] != 0})
    if stray:
        raise SystemExit(f"FAIL LOUD: {lv}/cells_reactive holds cells other than reactive s0: {stray[:3]}")
    servers_eq(rc, FLL_SERVERS, f"{lv}/cells_reactive")
    admitted, bad, whole = deg_admission(rc, rf, topos)
    out = {"admitted": admitted, "failing": {str(t): x for t, x in sorted(bad.items())}, "rungs_wholly_saturated_excluded": whole,
           "reactive_cells": len(rc), "reactive_failed": len(rf)}
    if not admitted:
        return out, None
    paths = [os.path.join(d, x) for x in ("cells_bar", "cells_learned")]
    if not all(os.path.isdir(p) for p in paths):
        raise SystemExit(f"FAIL LOUD: VOID: level {lv} has admitted ids but no cells_bar / cells_learned")
    cells, failed = load(",".join(paths), a.suffix)
    out["grid_cells"] = grid_or_void(cells, failed, [(BAR, (0,)), (CONFIRM_PRIMARY, SEEDS), (CONFIRM_MLP, SEEDS)], admitted, f"level {lv} (its admitted ids)")
    servers_eq(cells, FLL_SERVERS, f"level {lv}")
    out["warmth"] = asserts(cells, a.pin, [CONFIRM_PRIMARY, CONFIRM_MLP], ckpt, w_pins)
    rz = [k for k, s in cells.items() if float(s.get("totalPeerRendezvousWait", 0.0)) != 0.0]
    out["rendezvous_check"] = {"cells_with_rendezvous_wait": len(rz), "e_g": [list(k) for k in rz[:3]], "holds": not rz}
    return out, cells


def deg_gap(cells, arm, topos):
    ex = tot = 0.0
    for t in topos:
        for w in WINDOWS:
            for r in RUNGS:
                b = cells[(BAR, 0, t, w, r)]
                for sd in SEEDS:
                    f = cells[(arm, sd, t, w, r)]
                    ex += (float(b["totalPeerExchangeTime"]) - float(f["totalPeerExchangeTime"])) / len(SEEDS)
                    tot += (float(b["averageElapsedTime"]) * int(b["num_tasks"]) - float(f["averageElapsedTime"]) * int(f["num_tasks"])) / len(SEEDS)
    return {"exchange_seconds_saved": ex, "total_latency_seconds_gained": tot, "share": ex / tot if tot else None}


def read_degree_trend(a):
    """--confirm-degree-trend: primary = the structure check, admission and Page's L of FL gnn vs FL MLP over k = 1 < 2 < 4 < 8 (moderate, Holm 2);
    secondary = the per-level descriptive rows, after the primary read's md5 is filed."""
    global SEEDS, BAR, PROG_ALPHA
    deg_args(a)
    SEEDS, BAR, PROG_ALPHA = (1, 2), a.bar, PROG_ALPHA_FILED
    print(f"filed gate text {FILED['gate_text']}; addenda {', '.join(FILED['addenda'])}; gate pin {a.pin}")
    if hashlib.sha256(a.topos.encode()).hexdigest() != a.topos_sha256:
        raise SystemExit("FAIL LOUD: the topology list hash differs from the filed one")
    topos = [int(x) for x in a.topos.split(",")]
    if sorted(set(os.listdir(a.deg_root)) & set(DEG_LEVELS)) != sorted(DEG_LEVELS):
        raise SystemExit(f"FAIL LOUD: --deg-root must hold every level directory {list(DEG_LEVELS)}")
    structure = deg_structure(a)
    ckpt = ckpt_md5s(a.checkpoint_md5)
    w_pins = w_pin_chain(a.repo, a.pin, a.w_pins.split(","), a.gate_change)
    adm, cells_by = {}, {}
    for lv in DEG_LEVELS:
        adm[lv], cells_by[lv] = deg_load_level(a, lv, topos, ckpt, w_pins)
    warm = {adm[lv]["warmth"] for lv in DEG_LEVELS if "warmth" in adm[lv]}
    if len(warm) > 1:
        raise SystemExit(f"FAIL LOUD: the levels record different HEROSIM_WARMTH_PHYSICS {sorted(warm)}")
    fails = {lv: len(adm[lv]["failing"]) for lv in DEG_LEVELS}
    res = {"filed": FILED, "family": "partner structure", "pin": a.pin, "gate_change": a.gate_change, "topos": topos,
           "topologies_note": "topologies already read in the exchange study (30176-30187)", "primary": CONFIRM_PRIMARY, "mlp": CONFIRM_MLP,
           "bar": BAR, "seeds": list(SEEDS), "servers": FLL_SERVERS,
           "w_pins": {sha: sorted(k for k in LEARNED_W_KINDS + ARM2_W_KINDS if k in w.get("w_ms_per_plan", {})) for sha, w in w_pins.items()},
           "structure": structure, "admission": adm}
    if any(v > XS_DROP_OVER for v in fails.values()):
        res["trend"] = {"label": "VOID", "why": f"a level lost more than {XS_DROP_OVER} ids: {fails}", "failing_per_level": fails}
    elif "moderate" in set().union(*(adm[lv]["rungs_wholly_saturated_excluded"] for lv in DEG_LEVELS)):
        res["trend"] = {"label": "VOID", "why": "the moderate rung is saturated on every topology at a level", "failing_per_level": fails}
    else:
        ids = sorted(set.intersection(*[set(adm[lv]["admitted"]) for lv in DEG_LEVELS]))
        tests = []
        for stat in STATS:
            vals = {lv: topo_pct(cells_by[lv], CONFIRM_PRIMARY, SEEDS, CONFIRM_MLP, SEEDS, "moderate", stat, ids) for lv in DEG_LEVELS}
            pt = page_trend([[vals[lv][t] for lv in DEG_LEVELS] for t in ids])
            pt.update({"rung": "moderate", "stat": stat, "levels": list(DEG_LEVELS), "median_pct_by_level": {lv: st.median(vals[lv].values()) for lv in DEG_LEVELS},
                       "per_topology": {str(t): {lv: round(vals[lv][t], 3) for lv in DEG_LEVELS} for t in ids}})
            tests.append(pt)
        family(tests)
        yes = all(e["p_holm"] < PROG_ALPHA for e in tests)
        res["trend"] = {"label": DEG_YES if yes else DEG_NO, "why": "Holm < .025 on both moderate tests" if yes else "not Holm < .025 on both moderate tests",
                        "failing_per_level": fails, "trend_ids": ids, "n": len(ids), "tests": tests}
    rz = sum(adm[lv].get("rendezvous_check", {}).get("cells_with_rendezvous_wait", 0) for lv in DEG_LEVELS)
    res["confirmation"] = {"label": res["trend"]["label"], "stage": a.stage, "family": "partner structure", "rule": DEG_WHY,
                           "rendezvous": f"rendezvous nonzero in {rz} cells" if rz else "R_rendezvous = 0 in every cell read"}
    if a.stage == "secondary":
        m = md5(a.primary_read)
        node = "docs/lineages/uncut_groups_s0_v1.md"
        if m not in git(a.repo, "show", f"{a.record}:{node}"):
            raise SystemExit(f"FAIL LOUD: the primary read {a.primary_read} (md5 {m}) is not filed in {node} at {a.record}")
        if subprocess.run(["git", "-C", a.repo, "merge-base", "--is-ancestor", a.record, a.filed_ref]).returncode:
            raise SystemExit(f"FAIL LOUD: {a.record} is not on {a.filed_ref}: the primary read is not filed")
        with open(a.primary_read) as fh:
            p = json.load(fh)
        same = lambda x, y: json.dumps(x, sort_keys=True, default=str) == json.dumps(y, sort_keys=True, default=str)
        if p.get("confirmation", {}).get("stage") != "primary" or not all(same(res[k], p[k]) for k in ("trend", "admission", "topos", "structure")) \
                or res["confirmation"]["label"] != p["confirmation"]["label"]:
            raise SystemExit(f"FAIL LOUD: the secondary run's primary parts differ from the filed primary read {a.primary_read}")
        res["confirmation"]["primary_read"] = {"path": os.path.abspath(a.primary_read), "md5": m, "record": a.record}
        note, desc = "descriptive, p unadjusted, never a label", {}
        for lv in DEG_LEVELS:
            cells, ids = cells_by[lv], adm[lv]["admitted"]
            if cells is None:
                desc[lv] = {"note": "no cells read: no admitted id"}
                continue
            keep = [r for r in RUNGS if r not in adm[lv]["rungs_wholly_saturated_excluded"]]
            pick = lambda es: [e for e in es if e["rung"] in keep]
            miss = {}
            for arm, sd in ((CONFIRM_PRIMARY, SEEDS), (CONFIRM_MLP, SEEDS), (BAR, (0,))):
                v = [charge_miss(s, str(k))["excess_pp"] for k, s in cells.items() if k[0] == arm and k[4] in keep]
                miss[arm] = {"cells": len(v), "excess_pp_median": st.median(v), "excess_pp_max": max(v), "flagged_over_0_5pp": sum(x > S128_MISS_FLAG_PP for x in v)}
            desc[lv] = {"k": DEG_K[lv], "n_topologies": len(ids), "note": note, "rungs_excluded": adm[lv]["rungs_wholly_saturated_excluded"],
                        "fl_vs_bar": pick(four(cells, CONFIRM_PRIMARY, SEEDS, BAR, (0,), ids)), "mlp_vs_bar": pick(four(cells, CONFIRM_MLP, SEEDS, BAR, (0,), ids)),
                        "fl_vs_mlp": pick(four(cells, CONFIRM_PRIMARY, SEEDS, CONFIRM_MLP, SEEDS, ids)),
                        "gap_fl": deg_gap(cells, CONFIRM_PRIMARY, ids), "gap_mlp": deg_gap(cells, CONFIRM_MLP, ids), "charge_miss": miss,
                        "structure": structure["levels"][lv]["mean_over_workloads"], "connector_pairs_total": structure["levels"][lv]["connector_pairs_total"]}
        res["descriptive"] = desc
    txt = json.dumps(res, indent=1, default=str)
    if a.out:
        if os.path.exists(a.out):
            raise SystemExit(f"FAIL LOUD: {a.out} exists; a confirmation read is written once")
        with open(a.out + ".tmp", "w") as fh:
            fh.write(txt)
        os.replace(a.out + ".tmp", a.out)
    tr = res["trend"]
    rz = sum(adm[lv].get("rendezvous_check", {}).get("cells_with_rendezvous_wait", 0) for lv in DEG_LEVELS)
    print(f"PARTNER STRUCTURE {tr['label']} ({tr['why']})" + (f"; rendezvous nonzero in {rz} cells" if rz else ""))
    for e in tr.get("tests", []):
        print(f"  Page {e['rung']:8s} {e['stat']:8s} L {e['L']} k {e['k']} n {e['n']} p {e['p']:.3g} holm {e['p_holm']:.3g} ({e['method']}) "
              f"medians {', '.join(f'{v} {m_:+.2f}' for v, m_ in e['median_pct_by_level'].items())}")
    return res


# FL MP-OFF twin (uncut_groups_s0_v1, registered 2026-10-19 at 0a251dd9, merged 8251858f): FL gnn vs its MP-OFF twin, seed-matched, on fl3
TWIN = "ra_twin_eng_combfl_nosplit"
TWIN_W_KINDS = (TWIN,)
TW_P7 = "91269ec0171fd0362d2379c3f3102ad678aedc74"
TW_P7B = "f8a55ff07283c5b6159d3775a94e2ad8d0ee6bbc"
TW_TOPOS_SHA256 = "f6c5af0a8f74a8caa83ba45a97903ec47798ae8898e8e226b3ff22291bb59379"  # fl3 30149-30160, FL + lambda's mint line
TW_64_TOPOS = tuple(range(30129, 30141))
TW_SERVERS, TW_64_SERVERS = 24, 64
TW_ORDER = ("MP-ATTRIBUTED", "MP-DIRECTION", "TIE", "TWIN-FASTER")
TW_WHY = ("FL MP-OFF twin (F2, Holm 4 per charge, combined by the weaker label): TWIN-FASTER if the twin is Holm-faster (< .025) on any test; else "
          "MP-ATTRIBUTED iff FL gnn vs the twin has median <= -2 % and Holm < .025 on all four, FL seed s vs twin seed s <= -1 % on all four for each s, "
          "and FL seed s's collapses <= twin seed s's at each rung; else MP-DIRECTION iff every median < 0 and Holm < .025 on at least two; else TIE. "
          "Only MP-ATTRIBUTED licenses a message-passing claim")
TW_DESC = "descriptive, p unadjusted, never a label"


def tw_args(a):
    if a.confirm or a.confirm2 or a.confirm_size64 or a.confirm_fl_lambda or a.confirm_exchange_trend or a.confirm_size128 or a.confirm_degree_trend:
        raise SystemExit("FAIL LOUD: --confirm-mp-twin is its own family; pass no other --confirm flag")
    if a.primary != CONFIRM_PRIMARY or a.twin != TWIN or a.reference or a.compare or a.after_read or a.xs_root or a.deg_root or a.mlp_gate:
        raise SystemExit(f"FAIL LOUD: --confirm-mp-twin reads --primary {CONFIRM_PRIMARY} --twin {TWIN}; no reference, compare, after-read, xs-root, deg-root or mlp-gate")
    if a.cotenancy != "required" or a.bar != "cd_ib0_inflightfull" or a.stage not in ("primary", "secondary") or a.seeds != "1,2" or a.suffix != "_k100":
        raise SystemExit("FAIL LOUD: --confirm-mp-twin needs --cotenancy required, --bar cd_ib0_inflightfull, --seeds 1,2, --suffix _k100 and --stage primary or secondary")
    if not (a.repo and a.w_pins and a.twin_gate and a.fll_primary_read and a.fll_record):
        raise SystemExit("FAIL LOUD: --confirm-mp-twin needs --repo, --w-pins, --twin-gate, --fll-primary-read and --fll-record (read only after FL + lambda's primary is filed)")
    if a.topos_sha256 != TW_TOPOS_SHA256:
        raise SystemExit("FAIL LOUD: --confirm-mp-twin reads fl3 30149-30160 (sha256 f6c5af0a...) only")
    if a.stage == "primary" and (a.desc_gate or a.desc64_gate or a.s64_gate or a.mlp or a.primary_read or a.record):
        raise SystemExit("FAIL LOUD: --desc-gate / --desc64-gate / --s64-gate / --mlp / --primary-read / --record belong to --stage secondary (seal)")
    if a.stage == "secondary" and not (a.desc_gate and a.desc64_gate and a.s64_gate and a.mlp == CONFIRM_MLP and a.primary_read and a.record):
        raise SystemExit(f"FAIL LOUD: --confirm-mp-twin --stage secondary needs --desc-gate, --desc64-gate, --s64-gate, --mlp {CONFIRM_MLP}, --primary-read and --record")


def tw_filed(repo, read, record, filed_ref, what, node="docs/lineages/uncut_groups_s0_v1.md"):
    """A read's full md5 is in the lineage node at `record`, and `record` is on `filed_ref`."""
    m = md5(read)
    if m not in git(repo, "show", f"{record}:{node}"):
        raise SystemExit(f"FAIL LOUD: {what} {read} (md5 {m}) is not filed in {node} at {record}")
    if subprocess.run(["git", "-C", repo, "merge-base", "--is-ancestor", record, filed_ref]).returncode:
        raise SystemExit(f"FAIL LOUD: {record} is not on {filed_ref}: {what} is not filed")
    return m


def tw_chain(repo, base, w_pins, gate_change):
    """The w chain through P7b: the registered chain up to P7 through w_pin_chain unchanged, then the one link P7 -> P7b, admitted only for
    exactly those two commits, touching only the w file and adding only the twin's keys (w_frozen with TWIN_W_KINDS); every other key frozen."""
    pins = w_pins.split(",")
    rev = lambda p: git(repo, "rev-parse", "--verify", p + "^{commit}").strip()
    if len(pins) < 2 or rev(pins[-1]) != TW_P7B or rev(pins[-2]) != TW_P7:
        raise SystemExit(f"FAIL LOUD: --confirm-mp-twin --w-pins must end {TW_P7[:8]},{TW_P7B[:8]} (P7, then P7b with the twin's w)")
    ws = w_pin_chain(repo, base, pins[:-1], gate_change)
    if git(repo, "rev-parse", TW_P7B + "^").strip() != TW_P7:
        raise SystemExit(f"FAIL LOUD: P7b {TW_P7B} is not a child of P7 {TW_P7}")
    names = git(repo, "diff", "--name-only", TW_P7, TW_P7B).split()
    if names != [W_FILE]:
        raise SystemExit(f"FAIL LOUD: P7b differs from P7 in {names}, not only {W_FILE}")
    w7b = json.loads(git(repo, "show", f"{TW_P7B}:{W_FILE}"))
    w_frozen(ws[TW_P7], w7b, TW_P7B, LEARNED_W_KINDS + TWIN_W_KINDS)
    w_guard(w7b, TW_P7B)
    for top in ("w_ms_per_plan", "w_ms_per_plan_by_bucket"):
        if TWIN not in w7b.get(top, {}):
            raise SystemExit(f"FAIL LOUD: P7b has no {top}[{TWIN!r}]")
    ws[TW_P7B] = w7b
    return ws


def tw_twin_cells(cells, ws, kr):
    """Every twin cell ran at P7b and served k x the twin's frozen w there (asserts() charge-checks only its learned w kinds)."""
    k = float(ws[TW_P7B]["k_cotenancy"]) if kr else 1.0
    for key, s in cells.items():
        if key[0] != TWIN:
            continue
        if pin_match((s.get("code") or {}).get("commit"), [TW_P7B]) is None:
            raise SystemExit(f"FAIL LOUD: twin cell {key} ran at {(s.get('code') or {}).get('commit')}, not P7b {TW_P7B[:8]}")
        charge_check(key, s, ws[TW_P7B], k)


def tw_pick(cells, failed, kinds):
    keep = lambda d: {k: v for k, v in d.items() if k[0] in kinds}
    return keep(cells), keep(failed)


def tw_label(f2, seeds_alone, h_fl, h_tw):
    if any(e["median_pct"] is not None and e["median_pct"] > 0 and e["p_holm"] < PROG_ALPHA for e in f2):
        return "TWIN-FASTER", "the twin is Holm-faster on: " + ", ".join(f"{e['rung']}/{e['stat']}" for e in f2 if e["median_pct"] is not None and e["median_pct"] > 0 and e["p_holm"] < PROG_ALPHA)
    if win_four(f2, WIN_MED, PROG_ALPHA):
        seed_ok = all(e["median_pct"] is not None and e["median_pct"] <= SEED_MED for es in seeds_alone.values() for e in es)
        col_ok = all(h_fl[r]["per_seed"][sd]["collapse_any"] <= h_tw[r]["per_seed"][sd]["collapse_any"] for r in RUNGS for sd in SEEDS)
        if seed_ok and col_ok:
            return "MP-ATTRIBUTED", "every clause holds"
        why = f"WIN line met but seeds_alone_ok={seed_ok} collapse_ok={col_ok} (seed-matched)"
    else:
        why = "the full win rule fails"
    if all(e["median_pct"] is not None and e["median_pct"] < 0 for e in f2) and sum(e["p_holm"] < PROG_ALPHA for e in f2) >= 2:
        return "MP-DIRECTION", why + "; every median < 0 and Holm < .025 on at least two"
    return "TIE", why


def tw_verdict(cells, topos):
    """One charge: F2 (FL gnn vs the twin, Holm 4), the seed-matched seeds alone and collapses (against cd, per seed cell)."""
    res = {"health": {arm: health(cells, {}, arm, sd, topos) for arm, sd in ((CONFIRM_PRIMARY, SEEDS), (TWIN, SEEDS), (BAR, (0,)), (CD, (0,)))}}
    f2 = family(four(cells, CONFIRM_PRIMARY, SEEDS, TWIN, SEEDS, topos))
    seeds_alone = {sd: four(cells, CONFIRM_PRIMARY, (sd,), TWIN, (sd,), topos) for sd in SEEDS}
    lab, why = tw_label(f2, seeds_alone, res["health"][CONFIRM_PRIMARY], res["health"][TWIN])
    res["F2"] = {"contrast": f"{CONFIRM_PRIMARY} vs {TWIN} (seed-matched clauses)", "tests": f2, "seeds_alone": seeds_alone, "label": lab, "why": why}
    return res


def read_mp_twin(a):
    """--confirm-mp-twin: --stage primary = F2 (FL gnn vs the MP-OFF twin) under both charges; --stage secondary = the descriptive rows (the twin vs
    FL MLP and vs the bar at 24 servers, the twin at 64 servers), after this read's primary md5 is filed. Both stages only after FL + lambda's
    primary md5 is filed (--fll-primary-read, --fll-record)."""
    global SEEDS, BAR, PROG_ALPHA
    tw_args(a)
    SEEDS, BAR, PROG_ALPHA = (1, 2), a.bar, PROG_ALPHA_FILED
    print(f"filed gate text {FILED['gate_text']}; addenda {', '.join(FILED['addenda'])}; gate pin {a.pin}")
    if hashlib.sha256(a.topos.encode()).hexdigest() != a.topos_sha256:
        raise SystemExit("FAIL LOUD: the topology list hash differs from the filed one")
    topos = [int(x) for x in a.topos.split(",")]
    fll_md5 = tw_filed(a.repo, a.fll_primary_read, a.fll_record, a.filed_ref, "FL + lambda's primary read")
    with open(a.fll_primary_read) as fh:
        if json.load(fh).get("confirmation", {}).get("family") != "FL + lambda":
            raise SystemExit(f"FAIL LOUD: {a.fll_primary_read} is not FL + lambda's primary read")
    gc, gf = tw_pick(*load(a.gate, a.suffix), (CONFIRM_PRIMARY, CONFIRM_PRIMARY + KR, BAR, CD))
    tc, tf = load(a.twin_gate, a.suffix)
    stray = sorted({k[0] for k in list(tc) + list(tf) if k[0] not in (TWIN, TWIN + KR)})
    if stray:
        raise SystemExit(f"FAIL LOUD: --twin-gate holds cells other than the twin: {stray[:3]}")
    grid_n = grid_or_void(gc, gf, [(CONFIRM_PRIMARY, SEEDS), (CONFIRM_PRIMARY + KR, SEEDS), (BAR, (0,)), (CD, (0,))], topos, "--gate (FL gnn and the bars)")
    twin_n = grid_or_void(tc, tf, [(TWIN, SEEDS), (TWIN + KR, SEEDS)], topos, "--twin-gate")
    servers_eq({**gc, **tc}, TW_SERVERS, "--gate / --twin-gate")
    ckpt = ckpt_md5s(a.checkpoint_md5)
    ws = tw_chain(a.repo, a.pin, a.w_pins, a.gate_change)
    learned = [CONFIRM_PRIMARY, TWIN]
    cells, cells_kr = split_kr({**gc, **tc})
    warmth = asserts(cells, a.pin, learned, ckpt, ws)
    if asserts(cells_kr, a.pin, learned, ckpt, ws, kr=True) != warmth:
        raise SystemExit("FAIL LOUD: the co-tenancy cells record another HEROSIM_WARMTH_PHYSICS")
    tw_twin_cells(cells, ws, False)
    tw_twin_cells(cells_kr, ws, True)
    cells_kr.update({k: v for k, v in cells.items() if k[0] not in learned})
    res = {"filed": FILED, "family": "FL MP-OFF twin", "pin": a.pin, "gate_change": a.gate_change, "topos": topos, "warmth_physics": warmth,
           "primary": CONFIRM_PRIMARY, "twin": TWIN, "seeds": list(SEEDS), "bar": BAR, "servers": TW_SERVERS,
           "fll_primary_read": {"path": os.path.abspath(a.fll_primary_read), "md5": fll_md5, "record": a.fll_record},
           "w_pins": {sha: sorted(k for k in LEARNED_W_KINDS + ARM2_W_KINDS + TWIN_W_KINDS if k in w.get("w_ms_per_plan", {})) for sha, w in ws.items()},
           "hosts": hosts({**cells, **{(k[0] + KR,) + k[1:]: v for k, v in cells_kr.items() if k[0] in learned}})}
    res.update(tw_verdict(cells, topos))
    res["cotenancy"] = tw_verdict(cells_kr, topos)
    m, c = res["F2"]["label"], res["cotenancy"]["F2"]["label"]
    lab = max(m, c, key=TW_ORDER.index)
    res["combined"] = {"F2": lab, "why": f"weaker of frozen {m!r} and co-tenancy {c!r}"}
    res["confirmation"] = {"label": lab, "stage": a.stage, "rule": TW_WHY, "family": "FL MP-OFF twin", "grid_cells": grid_n, "twin_grid_cells": twin_n}
    if a.stage == "secondary":
        mm = tw_filed(a.repo, a.primary_read, a.record, a.filed_ref, "the twin's primary read")
        with open(a.primary_read) as fh:
            p = json.load(fh)
        same = lambda x, y: json.dumps(x, sort_keys=True, default=str) == json.dumps(y, sort_keys=True, default=str)
        if p.get("confirmation", {}).get("stage") != "primary" or not all(same(res[k], p[k]) for k in ("F2", "combined", "topos")) \
                or not same(res["cotenancy"]["F2"], p["cotenancy"]["F2"]) or res["confirmation"]["label"] != p["confirmation"]["label"]:
            raise SystemExit(f"FAIL LOUD: the secondary run's primary parts differ from the filed primary read {a.primary_read}")
        res["confirmation"]["primary_read"] = {"path": os.path.abspath(a.primary_read), "md5": mm, "record": a.record}
        dc, df = tw_pick(*load(a.desc_gate, a.suffix), (CONFIRM_MLP, CONFIRM_MLP + KR))
        res["confirmation"]["desc_grid_cells"] = grid_or_void(dc, df, [(CONFIRM_MLP, SEEDS), (CONFIRM_MLP + KR, SEEDS)], topos, "--desc-gate (FL MLP)")
        servers_eq(dc, TW_SERVERS, "--desc-gate")
        d100, dkr = split_kr(dc)
        if asserts(d100, a.pin, [CONFIRM_MLP], ckpt, ws) != warmth or asserts(dkr, a.pin, [CONFIRM_MLP], ckpt, ws, kr=True) != warmth:
            raise SystemExit("FAIL LOUD: the --desc-gate cells record another HEROSIM_WARMTH_PHYSICS")
        c64, f64 = load(a.desc64_gate, a.suffix)
        stray = sorted({k[0] for k in list(c64) + list(f64) if k[0] != TWIN})
        if stray:
            raise SystemExit(f"FAIL LOUD: --desc64-gate holds cells other than the twin _k100: {stray[:3]}")
        res["confirmation"]["desc64_grid_cells"] = grid_or_void(c64, f64, [(TWIN, SEEDS)], list(TW_64_TOPOS), "--desc64-gate")
        s64, s64f = tw_pick(*load(a.s64_gate, a.suffix), (CONFIRM_PRIMARY, BAR))
        res["confirmation"]["s64_grid_cells"] = grid_or_void(s64, s64f, [(CONFIRM_PRIMARY, SEEDS), (BAR, (0,))], list(TW_64_TOPOS), "--s64-gate (size-64 FL gnn and bar)")
        servers_eq({**c64, **s64}, TW_64_SERVERS, "--desc64-gate / --s64-gate")
        if asserts({**c64, **s64}, a.pin, [CONFIRM_PRIMARY, TWIN], ckpt, ws) != warmth:
            raise SystemExit("FAIL LOUD: the 64-server cells record another HEROSIM_WARMTH_PHYSICS")
        tw_twin_cells(c64, ws, False)
        for v, base, dd in ((res, cells, d100), (res["cotenancy"], cells_kr, dkr)):
            both = {**base, **dd}
            v["descriptive"] = {"twin_vs_mlp": {"note": TW_DESC + "; is the GNN without message passing still ahead of the pointwise model", "tests": four(both, TWIN, SEEDS, CONFIRM_MLP, SEEDS, topos)},
                                "twin_vs_bar": {"note": TW_DESC, "tests": four(both, TWIN, SEEDS, BAR, (0,), topos)}}
        both64 = {**c64, **s64}
        res["descriptive_64"] = {"label": "topologies already read in size-64", "note": TW_DESC + "; _k100, 64 servers, 30129-30140",
                                 "twin_vs_bar": four(both64, TWIN, SEEDS, BAR, (0,), list(TW_64_TOPOS)),
                                 "fl_vs_twin": four(both64, CONFIRM_PRIMARY, SEEDS, TWIN, SEEDS, list(TW_64_TOPOS))}
    txt = json.dumps(res, indent=1, default=str)
    if a.out:
        if os.path.exists(a.out):
            raise SystemExit(f"FAIL LOUD: {a.out} exists; a confirmation read is written once")
        with open(a.out + ".tmp", "w") as fh:
            fh.write(txt)
        os.replace(a.out + ".tmp", a.out)
    print(f"MP-TWIN {lab} ({res['combined']['why']})")
    for tag, x in (("", res), ("_kr ", res["cotenancy"])):
        for e in x["F2"]["tests"]:
            print(f"  F2 {tag}{e['rung']:8s} {e['stat']:8s} median {e['median_pct']} faster {e['faster']}/{e['n_topologies']} p {e['p']} holm {e['p_holm']}")
    return res


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
    ap.add_argument("--confirm", action="store_true",
                    help="the FL confirmation (registered 2026-10-18): FL gnn on fresh topologies, both verdicts, B6 lifted for this family only")
    ap.add_argument("--confirm2", action="store_true",
                    help="FL confirmation 2 (registered 2026-10-18 at 2ea36408): FL gnn seeds 3-6 on fresh topologies, Holm < .0125, both verdicts")
    ap.add_argument("--confirm-size64", action="store_true",
                    help="FL size-64 (registered 2026-10-18): FL gnn seeds 1-2 on 64-server topologies against the in-flight bar, Holm < .025, both verdicts")
    ap.add_argument("--reactive-gate", default="", help="--confirm-size64 --stage secondary: the reactive (admission) cells, read as a descriptive row")
    ap.add_argument("--desc-gate", default="", help="--confirm2 --stage secondary: the descriptive cells (FL gnn s1-2, FL MLP s1-2), read after the primary is filed")
    ap.add_argument("--stage", choices=("full", "primary", "secondary"), default="full",
                    help="primary: F1 under both verdicts only; secondary: F2-MC and the descriptive block, only after the primary read is filed")
    ap.add_argument("--primary-read", default="", help="--stage secondary: the primary-stage read JSON")
    ap.add_argument("--record", default="", help="--stage secondary: the commit whose lineage node files the primary read's md5")
    ap.add_argument("--filed-ref", default="origin/reference-physics", help="--stage secondary: --record must be an ancestor of this ref")
    ap.add_argument("--confirm-fl-lambda", action="store_true",
                    help="FL + lambda (registered 2026-10-19): FL + lambda vs its lambda = 0 control FL, seeds 1-2, Holm < .025, seed-matched clauses, both verdicts")
    ap.add_argument("--confirm-exchange-trend", action="store_true",
                    help="exchange-model sensitivity (registered 2026-10-19): Page's trend of FL gnn vs the bar over payload variants under --xs-root")
    ap.add_argument("--xs-root", default="", help="--confirm-exchange-trend: the directory holding one subdirectory per payload variant")
    ap.add_argument("--confirm-size128", action="store_true",
                    help="FL size-128 (registered 2026-10-19): F1 FL gnn vs the in-flight bar at servers = 128; secondary F2 FL gnn vs FL MLP with the charge clause")
    ap.add_argument("--mlp-gate", default="", help="--confirm-size128 --stage secondary: the FL MLP s1-2 cells, _k100 and _kr")
    ap.add_argument("--confirm-degree-trend", action="store_true",
                    help="partner structure (registered 2026-10-19): Page's trend of FL gnn vs FL MLP over partners drawn per task k = 1 < 2 < 4 < 8 under --deg-root")
    ap.add_argument("--deg-root", default="", help="--confirm-degree-trend: the directory holding k1 k2 k4 k8")
    ap.add_argument("--confirm-mp-twin", action="store_true",
                    help="FL MP-OFF twin (registered 2026-10-19): F2 FL gnn vs ra_twin_eng_combfl_nosplit on fl3, seed-matched, both charges")
    ap.add_argument("--twin-gate", default="", help="--confirm-mp-twin: the twin's own cells (s1-2, _k100 and _kr)")
    ap.add_argument("--fll-primary-read", default="", help="--confirm-mp-twin: FL + lambda's primary read, which must be filed first")
    ap.add_argument("--fll-record", default="", help="--confirm-mp-twin: the commit whose node files FL + lambda's primary md5")
    ap.add_argument("--desc64-gate", default="", help="--confirm-mp-twin --stage secondary: the twin at 64 servers (s1-2 _k100, 30129-30140)")
    ap.add_argument("--s64-gate", default="", help="--confirm-mp-twin --stage secondary: size-64's bar and FL gnn cells (30129-30140)")
    a = ap.parse_args(argv)
    if a.confirm_mp_twin:
        return read_mp_twin(a)
    if a.twin_gate or a.fll_primary_read or a.fll_record or a.desc64_gate or a.s64_gate:
        raise SystemExit("FAIL LOUD: --twin-gate / --fll-primary-read / --fll-record / --desc64-gate / --s64-gate belong to --confirm-mp-twin")
    if a.confirm_size128:
        return read_size128(a)
    if a.confirm_degree_trend:
        return read_degree_trend(a)
    if a.mlp_gate or a.deg_root:
        raise SystemExit("FAIL LOUD: --mlp-gate belongs to --confirm-size128 and --deg-root to --confirm-degree-trend")
    if a.confirm_fl_lambda:
        return read_fl_lambda(a)
    if a.confirm_exchange_trend:
        return read_exchange_trend(a)
    global SEEDS, BAR, PROG_ALPHA
    PROG_ALPHA = PROG_ALPHA_FILED
    if a.confirm or a.confirm2 or a.confirm_size64:
        confirm_args(a)
    elif a.stage != "full":
        raise SystemExit("FAIL LOUD: --stage primary / secondary is the FL confirmation's (--confirm)")
    confirm2, size64, mlp2 = a.confirm2, a.confirm_size64, ""
    staged = confirm2 or size64
    if staged:
        a.confirm = True
        PROG_ALPHA = CONFIRM2_ALPHA if confirm2 else PROG_ALPHA_FILED
        mlp2, a.mlp = a.mlp, ""  # the family's MLP lives in --desc-gate, read descriptively after the verdicts
    if a.stage == "primary":
        a.mlp = ""  # F2-MC is a secondary output: its cells are checked as another learned kind in --gate, nothing of it is computed
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
    if staged:
        mc = sorted({k for k in list(cells) + list(failed) if (k[0][:-len(KR)] if k[0].endswith(KR) else k[0]) == CONFIRM_MLP})
        if mc:
            raise SystemExit(f"FAIL LOUD: --confirm{'-size64' if size64 else '2'} --gate holds FL MLP cells {mc[:3]}; they belong in --desc-gate (C3)")
        grid_n = grid_or_void(cells, failed, [(a.primary, SEEDS), (a.primary + KR, SEEDS), (a.bar, (0,)), (CD, (0,))], topos, "--gate")
        if size64:
            servers_or_fail(cells, "--gate")
    learned = [x for x in (a.primary, a.twin, a.mlp, a.reference) if x]
    stray = sorted({(k[0], k[1]) for k in list(cells) + list(failed) if (k[0][:-len(KR)] if k[0].endswith(KR) else k[0]) in (a.primary, a.twin, a.mlp) and k[1] not in SEEDS})
    if stray:
        raise SystemExit(f"FAIL LOUD: learned cells at seeds outside --seeds {list(SEEDS)}: {stray[:6]}")
    if not a.twin and ckpt_tag(a.primary) != "arm3" and SEEDS == (1, 2) and a.primary not in COMB_W_KINDS:
        raise SystemExit("FAIL LOUD: --twin is required (only an Arm 3 read, or a seed-extension read of an arm with no new-seed twin, may go without one)")
    ckpt = ckpt_md5s(a.checkpoint_md5)
    # B6 for the combined lever (every COMB kind, the FAST arms included): never a program-level label
    prog_allowed, closed_why = (False, COMB_B6_WHY) if a.primary in COMB_W_KINDS else (True, FIXED_SEQ_WHY)
    if a.confirm:
        prog_allowed, closed_why = True, CONFIRM_WHY
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
    # A learned kind in --gate that is not one of this read's arms (e.g. F, L and the MLP beside FL in one cell directory) is checked as a
    # learned cell at the w-pin chain (pin, w charge, checkpoint md5, host block) and then left out of everything the read computes.
    base_kind = lambda k: k[:-len(KR)] if k.endswith(KR) else k
    others = sorted({base_kind(k[0]) for k in list(cells) + list(failed)
                     if base_kind(k[0]) in LEARNED_W_KINDS + ARM2_W_KINDS and base_kind(k[0]) not in learned}) if w_pins else []
    oth_warmth = set()
    if others:
        print(f"other learned kinds in --gate, checked at the w-pin chain and not read: {others}")
        bad_failed = sorted(k for k in failed if base_kind(k[0]) in others)
        if bad_failed:
            raise SystemExit(f"FAIL LOUD: failed cells of the other learned kinds in --gate: {bad_failed[:4]}")
        oth = {k: v for k, v in cells.items() if k[0] in others}
        oth_kr = {(k[0][:-len(KR)],) + k[1:]: v for k, v in cells.items() if k[0].endswith(KR) and base_kind(k[0]) in others}
        oth_warmth.add(asserts(oth, a.pin, others, ckpt, w_pins))
        if oth_kr:
            oth_warmth.add(asserts(oth_kr, a.pin, others, ckpt, w_pins, kr=True))
        cells = {k: v for k, v in cells.items() if base_kind(k[0]) not in others}
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
    if oth_warmth and oth_warmth != {warmth}:
        raise SystemExit(f"FAIL LOUD: the other learned kinds in --gate record HEROSIM_WARMTH_PHYSICS {sorted(oth_warmth)}, the read's {warmth}")
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
    res.update(verdict(cells, failed, a, topos, prog_allowed, closed_why))
    if a.cotenancy == "required":
        res["cotenancy"] = verdict(cells_kr, failed_kr, a, topos, prog_allowed, closed_why)
        res["combined"] = combine(res, res["cotenancy"], a)
        lab, why, f2 = res["combined"]["F1"], res["combined"]["why"], res["combined"]
    else:
        res["cotenancy"] = "OFF (not the gate read)"
        lab, why, f2 = res["F1"]["label"], res["F1"]["why"], res["F2"]
    f1 = res["F1"]["tests"]
    if size64:
        res["confirmation"] = {"label": SIZE64_YES if prog_win(lab) else "NOT-CONFIRMED", "reader_label": lab, "stage": a.stage,
                               "rule": SIZE64_WHY, "family": "FL size-64", "grid_cells": grid_n}
    elif confirm2:
        res["confirmation"] = {"label": CONFIRM2_YES if prog_win(lab) else "NOT-CONFIRMED", "reader_label": lab, "stage": a.stage,
                               "rule": CONFIRM2_WHY, "alpha": CONFIRM2_ALPHA_LINE, "family": "confirmation 2", "grid_cells": grid_n}
    elif a.confirm:
        res["confirmation"] = {"label": "CONFIRMED" if prog_win(lab) else "NOT-CONFIRMED", "reader_label": lab, "stage": a.stage,
                               "rule": CONFIRM_WHY}
    if a.confirm:
        if a.stage == "primary":
            for v in (res, res["cotenancy"]):
                v.pop("descriptive", None)
        else:
            check_against_filed_primary(a, res)
            if confirm2:  # C3: --desc-gate is opened only after the primary read is shown to be filed
                desc_confirm2(a, res, cells, cells_kr, topos, ckpt, w_pins, warmth, mlp2)
            if size64:  # the seal: --desc-gate and --reactive-gate are opened only after the primary read is shown to be filed
                desc_size64(a, res, cells, topos, ckpt, w_pins, warmth, mlp2)
    txt = json.dumps(res, indent=1, default=str)
    if a.out:
        if a.confirm and os.path.exists(a.out):
            raise SystemExit(f"FAIL LOUD: {a.out} exists; a confirmation read is written once")
        with open(a.out + ".tmp", "w") as fh:
            fh.write(txt)
        os.replace(a.out + ".tmp", a.out)
    if a.confirm:
        print(f"CONFIRMATION {res['confirmation']['label']} (reader: {lab}; {why})" + (f"; {f2['MODEL-CLASS']}" if a.mlp else ""))
    else:
        print(f"F1 {lab} ({why}); F2 {f2['MP']}" + (f"; {f2['MODEL-CLASS']}" if a.mlp else ""))
    for e in f1:
        print(f"  F1 {e['rung']:8s} {e['stat']:8s} median {e['median_pct']} faster {e['faster']}/{e['n_topologies']} p {e['p']} holm {e['p_holm']}")
    if a.confirm:
        for e in res["cotenancy"]["F1"]["tests"]:
            print(f"  F1 _kr {e['rung']:8s} {e['stat']:8s} median {e['median_pct']} faster {e['faster']}/{e['n_topologies']} p {e['p']} holm {e['p_holm']}")
    if "descriptive" in res:
        for r in RUNGS:
            g = res["descriptive"]["gap_attribution"][r]
            print(f"  {r}: gap {g['gap_seconds']:.1f} s, share in groups > 10 {g['share_gt10']}, opening {g['opening_share']} -> {g['label']}")
    return res


if __name__ == "__main__":
    main(sys.argv[1:])
