# archive/

Retired code, kept for reproducibility. **Nothing here is deleted from history and
nothing here is live.**

## Rules

- **Do not import from `archive/` in live code.** The live tree is verified closed
  against it (see the GATE checks in `LINEAGES.md`). A new import from here is a bug.
- **Do not "fix" archived code.** It is a frozen record of what a given experiment
  actually ran. If an archived file looks wrong, that may be exactly the finding.
- **For AI agents:** ignore this directory unless the user names a specific lineage.
  `../LINEAGES.md` is the map of what is current.

## Running archived code

Archived modules were moved with `git mv`, so their `from src.<x> import ...`
statements no longer resolve in place. This is intentional — the archive is a
reading reference, not a runnable tree. To actually **run** a retired experiment,
check out the pre-cleanup restore point, where everything still resolves:

```bash
git checkout pre-cleanup-2026-08
```

`git log --follow archive/<path>` works normally; history followed the moves.

## Contents

| Directory | What it is | Why retired |
|---|---|---|
| `pre_gnn_herosim/` | The original HeROsim proactive-autoscaling stack: XGBoost/GPR demand prediction, Bayesian optimisation over infrastructure, LHS sampling, the `scenario-*.sh` demos, and the HRO/HRC/proactive-Knative policy family. | Superseded by the GNN co-simulation line of work. Import-closed as of the cleanup: no live module referenced it except a 7-constant file (now `src/placement/constants.py`) and one dead import. |

Later phases add one directory per retired GNN-era lineage. Every directory here
must have a row in `../LINEAGES.md` giving its status and outcome.
