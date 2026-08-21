# Local ↔ datalab parity

**The rule this document exists to enforce:** a `total_rtt` produced on one machine may only
be compared against a `total_rtt` produced on another if everything in the *Tier 1* and
*Tier 2* tables below is identical, and that identity was **verified**, not assumed.

Live-gate verdicts are cross-policy comparisons. If a GNN arm shifts with the machine while
the Knative arm does not, the gate stops measuring the policy and starts measuring the venue.
That is not hypothetical — it happened, cost three sessions, and produced a phantom 11–26%
"GNN is better locally" result that was actually a code bug plus an environment nobody was
looking at. See `LINEAGES.md` → `siv1_full_corpus`.

---

## What actually matters, ranked by measured impact

Every number below is measured, not assumed. Method and date in the last column.

| Tier | Axis | Measured impact | How it is checked | Measurement |
|---|---|---|---|---|
| **1** | **Application code** | up to **23.3%** of decisions | `verify_code_identity.py` (git sha + import-closure sha256) | dims 9–11 estimator bug, 2026-08-20 |
| **1** | **Topology / infrastructure** | catastrophic (train/serve mismatch) | `verify_live_infra_parity.py` | existing preflight |
| **1** | **Contracts** (queue, topology, feature layout, network graph) | up to **12.4×** live RTT | checkpoint `.contract.json` sidecar, enforced in `load_gnn_model` | MP mismatch, 2026-08-16 |
| **1** | **Warmth physics** | ~**100×** live total RTT | `require_explicit_warmth_physics` | pre-existing |
| **2** | **Workload trace + seed** | verdict-flipping | no `--seed` in gate scripts (it overrides config topology) | 2026-08-19 |
| **3** | **Library versions** (torch, numpy, PyG, python) | **exactly 0.0** | `verify_venue_parity.py --mode logits` | 2026-08-21, below |
| **3** | **Thread count** (1 vs 4) | **exactly 0.0** | same | 2026-08-21 |
| **3** | **Device** (CPU vs CUDA) | 1.9e-5 on logits, **0/256 argmax flips**; ≤0.09% on `total_rtt` | same, `--device` | 2026-08-21 |

### The Tier 3 measurement, 2026-08-21

Question: *is keeping the repo in sync enough, or do the two machines need identical Python
environments?* Answered by forwarding one frozen 64-graph batch
(`tests/fixtures/venue_parity/`) through `near-rtt-v2-full-corpus-siv1-dim14-ce-only.pt` under
every available stack, at commit `22e8f27`, with byte-identical weights and import closure
(md5-verified on both sides):

| stack | python | torch | numpy | PyG | max\|Δlogit\| | argmax flips |
|---|---|---|---|---|---|---|
| local pipenv (reference) | 3.12.3 | 2.5.1+cu121 | 2.3.0 | 2.6.1 | — | — |
| cluster `gnn` env | 3.12.12 | 2.5.1+cu121 | 1.26.4 | 2.7.0 | **0.0** | **0 / 256** |
| cluster pipenv venv | 3.12.12 | **2.12.0+cu130** | 1.26.4 | 2.8.0 | **0.0** | **0 / 256** |
| cluster pipenv, 4 threads | " | " | " | " | **0.0** | **0 / 256** |
| local, `--device cuda` | 3.12.3 | 2.5.1+cu121 | 2.3.0 | 2.6.1 | 1.9e-5 | **0 / 256** |

The last row is the **negative control**: it proves the probe can detect a difference, so the
zeros above are a real result and not a broken comparison.

**Conclusion: for GNN inference numerics, keeping the repo in sync is enough.** A torch major
jump across a CUDA major (2.5.1+cu121 → 2.12.0+cu130), a numpy major (2.3.0 → 1.26.4) and a
PyG minor (2.6.1 → 2.8.0) together move the logits by *exactly zero bits*. The only nonzero
axis is the accelerator, and at 1.9e-5 it flips no decisions.

**So why pin the environment at all?** Because Tier 3 being zero *today, for this checkpoint,
on this architecture* is a measurement, not a law. It was zero because this checkpoint is
small (43,393 parameters), CPU-friendly, and uses no fused or nondeterministic kernel. A
larger model, a different aggregation, or a future torch release can break it, and the only
way to know is to keep measuring. Pinning turns "we assume it is fine" into "we check that it
is fine." Pinning is *cheap insurance*, not the fix for the incident that prompted it — the
fix for that was Tier 1.

### The corollary worth internalising

Environment drift perturbs floats **symmetrically**: an argmax flip helps as often as it hurts.
A cross-venue gap that is **one-directional on every cell and never flips sign** is therefore
*not* an environment signature — it is a biased-estimator signature, i.e. a bug in the code
that computes features. Diagnose in that order. In the 2026-08-20 incident, three sessions were
spent on the environment when the sign pattern had already ruled it out on day one.

---

## The sync procedure

### Source code — git, never ad hoc rsync

```bash
# local
git push origin <branch>
# cluster
ssh datalab 'cd ~/gnn-herosim && git pull --ff-only'
# then ALWAYS confirm, do not assume:
git rev-parse HEAD                       # local
ssh datalab 'cd ~/gnn-herosim && git rev-parse HEAD'
```

Rsyncing source is an exception, not the default (datalab-pitfalls #5). It is defensible only
for brand-new files that cannot clobber anything, and only when you say so out loud and commit
promptly. If you rsync source, the two trees are no longer git-synchronized — which is exactly
the setup that makes datalab-pitfalls #3 (independent uncommitted edits on both sides)
possible.

### Binaries — rsync, never git

Models, datasets, graph caches. `models/` is gitignored, so **a checkpoint's
`.contract.json` sidecar is the only lineage record that travels with it.** 21 of 29
checkpoints currently have no sidecar and silently resolve to `legacy_v0`. When you rsync a
`.pt`, rsync its sidecar in the same command or the receiving venue is serving under an
unknown contract.

Always md5 both sides afterwards — a truncated or resumed transfer is not visibly different
from a complete one:

```bash
md5sum models/<ckpt>.pt
ssh datalab 'cd ~/gnn-herosim && md5sum models/<ckpt>.pt'
```

### Environment — one lockfile, both sides

`envs/herosim-lock.txt` is the single canonical spec. The canonical stack is **the local one**
(torch 2.5.1+cu121 / numpy 2.3.0 / PyG 2.6.1), because that is the interpreter the re-graded
`siv1_full_corpus` result was measured on.

There have historically been *three* competing answers — `Pipfile` (leaves torch/numpy/PyG as
`"*"`, which is the root cause of the drift), a `Pipfile.lock` stale since Jan 2026 that omits
torch/PyG/orjson entirely, and a `requirements.txt` with no torch at all. Do not add a fourth.

### Never let `pipenv run` decide the interpreter

This is the failure that started all of this. `micromamba activate gnn` followed by
`pipenv run python3` does **not** run in the `gnn` env — `pipenv run` resolves its own venv and
shells straight past the activation. On the cluster that silently created and used a third,
unmanaged, undeclared environment for every gate that has ever run.

```bash
# in any .sh that may run under sbatch
${HEROSIM_PY:-pipenv run python3} scripts_cosim/run_simulation.py ...

# in the .sbatch, right after activation
micromamba activate gnn
export HEROSIM_PY=python3
```

---

## Verification — the checks, in the order to run them

```bash
# 0. same commit, clean trees, both sides
git status --porcelain && git rev-parse HEAD
ssh datalab 'cd ~/gnn-herosim && git status --porcelain && git rev-parse HEAD'

# 1. same code actually reachable from the entry point (import closure, not a hand-list)
pipenv run python3 scripts_cosim/verify_code_identity.py --policy gnn

# 2. same topology as the training corpus
pipenv run python3 scripts_cosim/verify_live_infra_parity.py --dataset <cell> -v

# 3. same decisions from the same weights  (~6 s, safe on a login node)
pipenv run python3 scripts_cosim/verify_venue_parity.py --mode logits --assert

# 4. same total_rtt end to end  (minutes; keeps a knative control arm on purpose)
pipenv run python3 scripts_cosim/verify_venue_parity.py --mode run --config <cell>.json --assert
```

Step 3 keeps **both** a `knative` and a `gnn` arm for a structural reason: Knative never
touches `build_inference_feature_bundle`, so a Knative-only cross-check is *incapable* of
detecting a divergence on the learned path. "The two venues agreed on `knative/cell01`" was
offered as evidence the venues matched. It never was.

### Re-minting the fixture

The fixture and its reference are committed under `tests/fixtures/venue_parity/` (~300 KiB).
Re-mint only on the blessed interpreter, and only when the checkpoint or the graph layout
changes — a reference minted on a drifted interpreter blesses the drift:

```bash
pipenv run python3 scripts_cosim/verify_venue_parity.py --capture \
  --cache-dir simulation_data/graphs_cache_full_corpus_siv1_dim14
pipenv run python3 scripts_cosim/verify_venue_parity.py --mode logits --write-reference
```

`tests/test_venue_parity.py` guards the fixture against rot (missing env stamp, pickled
objects, checkpoint drift). Note that the cluster `gnn` env has **no `pytest` installed**, so
on datalab the guard runs only through the CLI probe — folding `pytest` into
`envs/herosim-lock.txt` would close that gap. `envs/herosim-lock.txt` now exists (added
2026-08-21; it had been cited as canonical here and in `CLAUDE.md` for a while without being
a real file).

### The cluster `gnn` env had rotted — repaired 2026-08-21

Worth recording because it cost a failed job and is invisible until something imports the
broken package. The env had **two disagreeing installs of the same packages**: `conda-meta`
declared `scikit-learn 1.9.0` built against numpy 2 (`np2py312`), while pip's metadata said
`1.6.1`, over an env whose numpy is `1.26.4`. `scipy 1.17.1` was damaged the same way — its
Python layer imported a `_promote` symbol its own compiled `_rotation` extension did not
export. Nothing surfaced until `training_contract.split_ids_by_canonical_parent` did its
**lazy** `from sklearn.model_selection import train_test_split` at line 128, which is why an
import-closure preflight of the same modules passed cleanly minutes earlier. Job 709234 died
84 s in; 709235 is the resubmit.

Repair, which also moves the cluster *toward* the canonical stack rather than sideways:

```bash
python3 -m pip install --no-deps --force-reinstall "scipy==1.15.3" "scikit-learn==1.7.0"
```

`--no-deps` is load-bearing: it keeps pip from touching `numpy`/`torch` on the way past.
Afterwards, **prove the repair was numerically inert** rather than assuming it — the whole
point of the probe:

```bash
python3 scripts_cosim/verify_venue_parity.py --mode logits --assert
# max|delta| 0.0, argmax flips 0/256 -- confirmed after this repair
```

Two lessons. An import-closure check only covers *eager* imports; a lazy import inside a
function is invisible to it, so a preflight that passes is not proof a job will start. And a
`conda-meta` version is not evidence of what will import — check pip and conda metadata
together when an env misbehaves, since a mismatch between them *is* the bug.

---

## When two numbers may be compared

A `total_rtt` is comparable to another `total_rtt` only if **all** of these hold. If any is
unknown rather than false, the answer is still no — unknown is not a pass.

- [ ] Same git sha, both trees clean (or the dirt is recorded and identical).
- [ ] Same code fingerprint (import closure sha256).
- [ ] Same warmth physics, queue feature contract, topology feature contract, inference
      feature layout, network graph contract.
- [ ] Same checkpoint md5 **and** same `.contract.json`.
- [ ] Same workload trace and same cell config; no `--seed` passed to the live CLI.
- [ ] Same `PYTHONHASHSEED`.
- [ ] Environment recorded on both sides — interpreter path included — even though Tier 3
      currently measures zero. Recorded, so the next incident is diagnosable in one command
      instead of three sessions.

Anything else is a comparison between two experiments, not two policies.
