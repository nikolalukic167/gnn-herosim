"""Contract tests for run_experiment.py.

These guard the two properties that make configs safe to replace forked wrappers:
malformed configs fail loudly, and the shipped configs resolve to what the trainer
actually expects.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import run_experiment as rx  # noqa: E402


def write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(body)
    return p


BASE = "trainer: gnn\nlineage: contention_v2_v3\ncache_dir: simulation_data/x\n"


# --- fail loudly -----------------------------------------------------------

def test_unknown_top_level_key_raises(tmp_path):
    cfg = write(tmp_path, BASE + "typo_key: 1\n")
    with pytest.raises(rx.ConfigError, match="Unknown key"):
        rx.load_config(cfg)


def test_unknown_trainer_raises(tmp_path):
    cfg = write(tmp_path, "trainer: nope\nlineage: l\ncache_dir: c\n")
    with pytest.raises(rx.ConfigError, match="Unknown trainer"):
        rx.load_config(cfg)


@pytest.mark.parametrize("missing", ["trainer", "lineage", "cache_dir"])
def test_missing_required_key_raises(tmp_path, missing):
    body = "".join(l + "\n" for l in BASE.strip().split("\n")
                   if not l.startswith(missing + ":"))
    with pytest.raises(rx.ConfigError, match="missing required key"):
        rx.load_config(write(tmp_path, body))


def test_unknown_wandb_key_raises(tmp_path):
    cfg = write(tmp_path, BASE + "wandb:\n  projekt: typo\n")
    with pytest.raises(rx.ConfigError, match="Unknown wandb key"):
        rx.load_config(cfg)


def test_wandb_tags_must_be_list(tmp_path):
    cfg = write(tmp_path, BASE + "wandb:\n  tags: not-a-list\n")
    with pytest.raises(rx.ConfigError, match="tags must be a list"):
        rx.load_config(cfg)


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(rx.ConfigError, match="Config not found"):
        rx.load_config(tmp_path / "nope.yaml")


def test_path_args_naming_absent_arg_raises(tmp_path):
    cfg = write(tmp_path, BASE + "path_args: [output]\nargs:\n  epochs: 1\n")
    with pytest.raises(rx.ConfigError, match="not present in args"):
        rx.resolve(rx.load_config(cfg), cfg)


def test_mlp_rejects_wandb_tags(tmp_path):
    cfg = write(
        tmp_path,
        "trainer: mlp\nlineage: l\ncache_dir: c\nwandb:\n  tags: [a]\n",
    )
    with pytest.raises(rx.ConfigError, match="no wandb tag support"):
        rx.resolve(rx.load_config(cfg), cfg)


# --- shipped configs -------------------------------------------------------

SHIPPED = sorted((REPO_ROOT / "experiments").glob("*.yaml"))


def test_shipped_configs_exist():
    assert SHIPPED, "no experiment configs found"


@pytest.mark.parametrize("cfg_path", SHIPPED, ids=lambda p: p.stem)
def test_shipped_config_resolves(cfg_path):
    cfg = rx.load_config(cfg_path)
    trainer, env, argv = rx.resolve(cfg, cfg_path)
    assert trainer.is_file()
    assert argv[1] == "--cache-dir"
    assert env["HEROSIM_EXPERIMENT_LINEAGE"] == cfg["lineage"]
    # every run must be traceable back to its yaml
    assert env["HEROSIM_EXPERIMENT_CONFIG"] == f"experiments/{cfg_path.name}"


@pytest.mark.parametrize("cfg_path", SHIPPED, ids=lambda p: p.stem)
def test_shipped_config_logs_to_wandb(cfg_path):
    """CLAUDE.md: all training runs must be logged to wandb. No exceptions."""
    cfg = rx.load_config(cfg_path)
    _, env, argv = rx.resolve(cfg, cfg_path)
    assert "--wandb-project" in argv, f"{cfg_path.name} would not log to wandb"
    run_named = "WANDB_RUN_NAME" in env or "--wandb-run-name" in argv
    assert run_named, f"{cfg_path.name} has no wandb run name"


@pytest.mark.parametrize("cfg_path", SHIPPED, ids=lambda p: p.stem)
def test_shipped_config_lineage_is_in_ledger(cfg_path):
    ledger = (REPO_ROOT / "LINEAGES.md").read_text()
    lineage = rx.load_config(cfg_path)["lineage"]
    assert lineage in ledger, (
        f"{cfg_path.name} declares lineage {lineage!r} with no row in LINEAGES.md"
    )
