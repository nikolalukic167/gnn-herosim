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


def test_path_args_naming_null_arg_raises(tmp_path):
    cfg = write(
        tmp_path,
        BASE + "path_args: [partial-state]\nargs:\n  partial-state: null\n",
    )
    with pytest.raises(rx.ConfigError, match="null value"):
        rx.resolve(rx.load_config(cfg), cfg)


def test_mlp_rejects_wandb_tags(tmp_path):
    cfg = write(
        tmp_path,
        "trainer: mlp\nlineage: l\ncache_dir: c\nwandb:\n  tags: [a]\n",
    )
    with pytest.raises(rx.ConfigError, match="no wandb tag support"):
        rx.resolve(rx.load_config(cfg), cfg)


# --- bare flags --------------------------------------------------------------

def test_bare_flag_null_value_emits_flag_alone(tmp_path):
    cfg = write(
        tmp_path,
        BASE + "args:\n  partial-state: null\n  epochs: 5\n",
    )
    _trainer, _env, argv = rx.resolve(rx.load_config(cfg), cfg)
    assert "--partial-state" in argv
    idx = argv.index("--partial-state")
    # bare flag: next token is the NEXT flag, never a rendered "None"
    assert argv[idx + 1] == "--epochs"


def test_non_null_arg_still_renders_value(tmp_path):
    cfg = write(tmp_path, BASE + "args:\n  epochs: 5\n")
    _trainer, _env, argv = rx.resolve(rx.load_config(cfg), cfg)
    i = argv.index("--epochs")
    assert argv[i + 1] == "5"


# --- --seed --------------------------------------------------------------

def test_no_seed_behaves_exactly_as_today(tmp_path):
    cfg = write(
        tmp_path,
        BASE + "args:\n  epochs: 5\nwandb:\n  run_name: my-run\n",
    )
    loaded = rx.load_config(cfg)
    _trainer, env_a, argv_a = rx.resolve(loaded, cfg)
    _trainer, env_b, argv_b = rx.resolve(loaded, cfg, seed=None)
    assert env_a == env_b
    assert argv_a == argv_b
    assert env_a["WANDB_RUN_NAME"] == "my-run"


def test_seed_gnn_sets_env_and_suffixes_run_name(tmp_path):
    cfg = write(
        tmp_path,
        BASE + "args:\n  epochs: 5\nwandb:\n  run_name: my-run\n",
    )
    _trainer, env, _argv = rx.resolve(rx.load_config(cfg), cfg, seed=3)
    assert env["NEAR_RTT_TRAIN_SEED"] == "3"
    assert env["WANDB_RUN_NAME"] == "my-run-seed3"


def test_seed_gnn_no_run_name_configured_no_crash(tmp_path):
    cfg = write(tmp_path, BASE + "args:\n  epochs: 5\n")
    _trainer, env, _argv = rx.resolve(rx.load_config(cfg), cfg, seed=3)
    assert env["NEAR_RTT_TRAIN_SEED"] == "3"
    assert "WANDB_RUN_NAME" not in env


def test_seed_mlp_overrides_random_state_and_output_stem(tmp_path):
    cfg = write(
        tmp_path,
        "trainer: mlp\nlineage: l\ncache_dir: c\n"
        "path_args: [output]\n"
        "args:\n  output: models/tabular/foo.pt\n  random-state: 42\n"
        "wandb:\n  run_name: mlp-run\n",
    )
    _trainer, env, argv = rx.resolve(rx.load_config(cfg), cfg, seed=7)
    assert "NEAR_RTT_TRAIN_SEED" not in env
    i = argv.index("--random-state")
    assert argv[i + 1] == "7"
    j = argv.index("--output")
    assert argv[j + 1].endswith("models/tabular/foo_seed7.pt")
    k = argv.index("--wandb-run-name")
    assert argv[k + 1] == "mlp-run-seed7"


def test_seed_mlp_without_output_raises(tmp_path):
    cfg = write(tmp_path, "trainer: mlp\nlineage: l\ncache_dir: c\n")
    with pytest.raises(rx.ConfigError, match="args.output"):
        rx.resolve(rx.load_config(cfg), cfg, seed=1)


def test_seed_mlp_random_state_argv_has_single_value(tmp_path):
    """--random-state must appear once, not twice (config value then override)."""
    cfg = write(
        tmp_path,
        "trainer: mlp\nlineage: l\ncache_dir: c\n"
        "path_args: [output]\n"
        "args:\n  output: models/tabular/foo.pt\n  random-state: 42\n",
    )
    _trainer, _env, argv = rx.resolve(rx.load_config(cfg), cfg, seed=7)
    assert argv.count("--random-state") == 1


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
