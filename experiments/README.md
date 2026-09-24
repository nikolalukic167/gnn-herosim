# experiments/

One YAML per training run. **Do not fork a trainer script per experiment.**

That habit produced 40 near-identical `train_near_rtt_v2_<name>.py` wrappers in
`src/notebooks/`, each 30–60 lines that set some env vars and `runpy`'d the real
trainer. The `contention_v2` and `contention_v3` wrappers differed in exactly four
values: cache dir, wandb run name, wandb tags, wandb project. Nothing in the tree
distinguished the live one from the 38 dead ones. Those wrappers are now in
`archive/model_sweeps/` and `archive/warmth_sparse/`; see `../LINEAGES.md`.

## Running

```bash
pipenv run python run_experiment.py experiments/contention_v2_gnn.yaml

# resolve everything and print env + argv, without training
pipenv run python run_experiment.py experiments/contention_v2_gnn.yaml --dry-run
```

`--dry-run` is what makes a config auditable: you can diff its resolved environment
and argv against whatever it replaced, and against a sibling config, without a GPU.

## Schema

Unknown keys are a hard error — a typo fails the run instead of silently training
with a default (see the project's no-silent-failures rule).

| Key | Required | Meaning |
|---|---|---|
| `trainer` | yes | `gnn` (`src/notebooks/train_near_rtt.py`) or `mlp` (`src/policy/tabular/train_mlp_dim22_from_batch.py`) |
| `lineage` | yes | Must match a row in `../LINEAGES.md` |
| `cache_dir` | yes | Graph cache, repo-relative |
| `description` | no | Free text for humans |
| `env` | no | Env vars exported before the trainer loads |
| `unset_env` | no | Env vars removed before the trainer loads (e.g. `TRAIN_INIT_CHECKPOINT` for from-scratch) |
| `args` | no | `flag: value` → `--flag value` |
| `path_args` | no | Names of `args` whose values are repo-relative paths to resolve absolute |
| `wandb` | no | `project`, `run_name`, `tags` (list) |

Every run is stamped with `HEROSIM_EXPERIMENT_CONFIG`, `HEROSIM_EXPERIMENT_LINEAGE`,
and a `cfg:<config-stem>` wandb tag, so a wandb run always traces back to its YAML.

## Trainer differences

The two trainers expose wandb differently, and `run_experiment.py` encodes this rather
than guessing:

- **gnn** reads `WANDB_RUN_NAME` / `WANDB_TAGS` from the environment and takes
  `--wandb-project`.
- **mlp** takes `--wandb-project` / `--wandb-run-name` and has **no tag support** —
  putting `wandb.tags` in an `mlp` config is a hard error rather than a silent no-op.

## Note on the MLP configs

The archived MLP wrappers never passed `--wandb-project`, and the MLP trainer only
initialises wandb when that flag is set — so **MLP runs were not being logged to wandb
at all**, contrary to the project rule that all training runs must be. The `*_mlp.yaml`
configs here set it. That is a deliberate behaviour change, and the only way these
configs differ from the wrappers they replace; everything else resolves byte-identically.
