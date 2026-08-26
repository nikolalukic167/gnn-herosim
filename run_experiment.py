#!/usr/bin/env python3
"""Run a training experiment from a config file instead of a forked script.

Every experiment used to get its own ``train_near_rtt_v2_<name>.py`` wrapper: a
30-60 line file that set some environment variables, built ``sys.argv`` and
``runpy``'d the real trainer. Forty of those accumulated, differing only in cache
directory and wandb name, and nothing distinguished the current one from the dead
ones. This replaces the fork with a config.

Usage::

    pipenv run python run_experiment.py experiments/contention_v2_gnn.yaml
    pipenv run python run_experiment.py experiments/contention_v2_gnn.yaml --dry-run

``--dry-run`` resolves everything and prints the environment and argv the trainer
would see, without training. That is what makes a config verifiable against the
wrapper it replaced.

Config schema (unknown keys are a hard error -- no silent typos)::

    trainer:     gnn | mlp                  # which trainer to invoke
    description: str                        # free text, for humans
    lineage:     str                        # must match a row in LINEAGES.md
    cache_dir:   path relative to repo root
    env:         {NAME: value}              # exported before the trainer loads
    unset_env:   [NAME, ...]                # removed before the trainer loads
    args:        {cli-flag: value}          # passed as --cli-flag value
    wandb:       {project, run_name, tags}  # tags is a list
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent

# Each trainer exposes wandb differently; encode it rather than guessing.
#   wandb_via_env : run name + tags are read from the environment
#   wandb_via_cli : run name is passed as --wandb-run-name (no tag support)
TRAINERS = {
    "gnn": {
        "path": REPO_ROOT / "src" / "notebooks" / "train_near_rtt.py",
        "wandb": "env",
    },
    "mlp": {
        "path": REPO_ROOT / "src" / "policy" / "tabular" / "train_mlp_dim22_from_batch.py",
        "wandb": "cli",
    },
}

TOP_LEVEL_KEYS = {
    "trainer", "description", "lineage", "cache_dir",
    "env", "unset_env", "args", "path_args", "wandb",
}
WANDB_KEYS = {"project", "run_name", "tags"}


class ConfigError(ValueError):
    """Raised for any malformed config. Never swallowed -- fail loudly."""


def _repo_relative(path: Path) -> str:
    """Path relative to the repo when possible, else absolute.

    Configs normally live in experiments/, but nothing stops one being passed from
    elsewhere; that should not blow up with a pathlib traceback.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"Config not found: {path}")
    with path.open() as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ConfigError(f"Config must be a mapping, got {type(cfg).__name__}: {path}")

    unknown = set(cfg) - TOP_LEVEL_KEYS
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in {path.name}: {sorted(unknown)}. "
            f"Allowed: {sorted(TOP_LEVEL_KEYS)}"
        )
    for required in ("trainer", "lineage", "cache_dir"):
        if required not in cfg:
            raise ConfigError(f"{path.name} is missing required key: {required!r}")
    if cfg["trainer"] not in TRAINERS:
        raise ConfigError(
            f"Unknown trainer {cfg['trainer']!r} in {path.name}. "
            f"Allowed: {sorted(TRAINERS)}"
        )

    wandb_cfg = cfg.get("wandb") or {}
    unknown_wandb = set(wandb_cfg) - WANDB_KEYS
    if unknown_wandb:
        raise ConfigError(
            f"Unknown wandb key(s) in {path.name}: {sorted(unknown_wandb)}. "
            f"Allowed: {sorted(WANDB_KEYS)}"
        )
    tags = wandb_cfg.get("tags")
    if tags is not None and not isinstance(tags, list):
        raise ConfigError(f"wandb.tags must be a list in {path.name}, got {type(tags).__name__}")
    return cfg


def resolve(
    cfg: dict, config_path: Path, *, seed: int | None = None
) -> tuple[Path, dict[str, str], list[str]]:
    """Return (trainer_path, env_to_set, argv) without mutating os.environ.

    ``seed`` implements the "one templated config per arm with per-seed env" option
    (ROUTE_B_STAGE2_PREREGISTRATION.md §3 / build item B6/W1): a draw must vary
    initialisation and batch order ONLY, so each trainer needs a distinct hook and a
    distinct output/run-name so draws never overwrite each other. ``seed=None`` (the
    CLI default) must resolve BIT-FOR-BIT identically to the pre-``--seed`` behavior.
    """
    spec = TRAINERS[cfg["trainer"]]
    trainer = spec["path"]
    if not trainer.is_file():
        raise ConfigError(f"Trainer missing: {trainer}")

    cache_dir = REPO_ROOT / cfg["cache_dir"]

    env: dict[str, str] = {k: str(v) for k, v in (cfg.get("env") or {}).items()}
    env["HEROSIM_EXPERIMENT_CONFIG"] = _repo_relative(config_path)
    env["HEROSIM_EXPERIMENT_LINEAGE"] = str(cfg["lineage"])

    wandb_cfg = cfg.get("wandb") or {}
    # Stamp the config into the run so every wandb entry traces back to its yaml.
    tags = list(wandb_cfg.get("tags") or []) + [f"cfg:{config_path.stem}"]

    argv = [str(trainer), "--cache-dir", str(cache_dir)]

    args = dict(cfg.get("args") or {})
    path_args = set(cfg.get("path_args") or [])
    unknown_paths = path_args - set(args)
    if unknown_paths:
        raise ConfigError(
            f"path_args names arg(s) not present in args: {sorted(unknown_paths)}"
        )
    null_path_args = {flag for flag in path_args if args.get(flag) is None}
    if null_path_args:
        raise ConfigError(
            f"path_args names arg(s) with a null value, which cannot be a path: "
            f"{sorted(null_path_args)}"
        )

    if seed is not None:
        if cfg["trainer"] == "mlp":
            # --random-state seeds init + batch order (and the drawn split, when no
            # --split-artifact pins it); a bare int here overrides whatever the config
            # set, matching argparse's last-flag-wins semantics.
            args["random-state"] = seed
            output = args.get("output")
            if output is None:
                raise ConfigError(
                    f"--seed requires an mlp config to set args.output ({config_path.name})"
                )
            output_path = Path(str(output))
            args["output"] = str(
                output_path.with_name(f"{output_path.stem}_seed{seed}{output_path.suffix}")
            )
        elif cfg["trainer"] == "gnn":
            # NEAR_RTT_TRAIN_SEED seeds torch (train_near_rtt.py:106); the checkpoint
            # path is models/{wandb.run.name}.pt, so distinct run names are what
            # prevents same-seed draws from overwriting each other's weights.
            env["NEAR_RTT_TRAIN_SEED"] = str(seed)

    for flag, value in args.items():
        rendered_flag = f"--{flag}"
        if value is None:
            # Bare store_true flag: emit "--flag" alone, never "--flag None".
            argv.append(rendered_flag)
            continue
        rendered = str(REPO_ROOT / str(value)) if flag in path_args else str(value)
        argv += [rendered_flag, rendered]

    if "project" in wandb_cfg:
        argv += ["--wandb-project", str(wandb_cfg["project"])]

    if spec["wandb"] == "env":
        run_name = wandb_cfg.get("run_name")
        if seed is not None and run_name is not None:
            run_name = f"{run_name}-seed{seed}"
        if run_name is not None:
            env["WANDB_RUN_NAME"] = str(run_name)
        env["WANDB_TAGS"] = ",".join(tags)
    elif spec["wandb"] == "cli":
        if wandb_cfg.get("tags"):
            raise ConfigError(
                f"trainer {cfg['trainer']!r} has no wandb tag support; "
                f"remove wandb.tags from {config_path.name}"
            )
        run_name = wandb_cfg.get("run_name")
        if seed is not None and run_name is not None:
            run_name = f"{run_name}-seed{seed}"
        if run_name is not None:
            argv += ["--wandb-run-name", str(run_name)]
    return trainer, env, argv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", type=Path, help="Path to the experiment yaml.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print resolved env and argv, then exit without training.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Draw seed (ROUTE_B_STAGE2_PREREGISTRATION.md §3): gnn sets "
                         "NEAR_RTT_TRAIN_SEED and suffixes the wandb run name with "
                         "-seed{N}; mlp overrides --random-state and suffixes the "
                         "--output stem with _seed{N}. Omit for today's behavior "
                         "unchanged.")
    ns = ap.parse_args()

    config_path = ns.config.resolve()
    cfg = load_config(config_path)
    trainer, env, argv = resolve(cfg, config_path, seed=ns.seed)

    cache_dir = Path(argv[argv.index("--cache-dir") + 1])
    if not ns.dry_run and not cache_dir.is_dir():
        raise ConfigError(f"Cache dir missing: {cache_dir}")

    if ns.dry_run:
        print(f"config:  {_repo_relative(config_path)}")
        print(f"lineage: {cfg['lineage']}")
        print(f"trainer: {_repo_relative(trainer)}")
        print("env:")
        for k in sorted(env):
            print(f"    {k}={env[k]}")
        for k in cfg.get("unset_env") or []:
            print(f"    (unset) {k}")
        print("argv:")
        print("    " + " ".join(argv))
        return

    for key in cfg.get("unset_env") or []:
        os.environ.pop(key, None)
    os.environ.update(env)
    sys.argv = argv
    runpy.run_path(str(trainer), run_name="__main__")


if __name__ == "__main__":
    main()
