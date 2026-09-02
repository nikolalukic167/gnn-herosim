"""The closed-loop episode must run under the environment the GO verdict was measured in.

`run_sampling_probe.py` is the registered Increment-1 instrument. Its GO verdict —
exploration affordable at every temperature, paired noise an order of magnitude under the
bar — is a statement about episodes run under *that* environment: `node_disk_v2` warmth,
batch size 4, a 2 ms batch timeout, and no exported feature-layout or contract variables
(the loader adopts them from the checkpoint's sidecar and raises on a conflict).

The trainer runs its own episodes through `closed_loop/episode.py`. Sharing one function
would be the obvious fix for the duplication, but it would also mean a later edit for the
trainer silently rewrites the conditions the registered verdict was measured under. So
the probe stays frozen and this test holds the two together instead: if either drifts,
the GO verdict stops covering the pilot and this fails rather than nobody noticing.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.closed_loop.episode import ARM_POLICY, episode_env

PROBE = REPO_ROOT / "scripts_cosim/closed_loop/run_sampling_probe.py"

# Set by the trainer's episodes and by nothing in the probe. Each is either inert for the
# probe's configuration or exists only to carry the replay reservoir out of the episode.
TRAINER_ONLY = {
    "GNN_SAMPLE_RESERVOIR_K",   # k=0 in the probe; the probe never replays
    "HEROSIM_EPISODE_REPLAY_OUT",
    "MLP_MODEL_PATH",           # the probe has no MLP arm
}


def _probe_env_block() -> dict[str, str]:
    """The literal `env[...] = ...` assignments inside the probe's run_episode."""
    tree = ast.parse(PROBE.read_text())
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "run_episode"
    )
    found: dict[str, str] = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if not (isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name)
                and tgt.value.id == "env"):
            continue
        key = tgt.slice.value if isinstance(tgt.slice, ast.Constant) else None
        if key is None:
            continue
        if isinstance(node.value, ast.Constant):
            found[key] = str(node.value.value)
        else:
            found[key] = "<dynamic>"
    return found


def test_probe_still_exists_and_declares_its_env():
    assert PROBE.exists(), "the registered Increment-1 probe was deleted"
    block = _probe_env_block()
    assert block, "could not parse the probe's env block; this test has gone blind"


@pytest.mark.parametrize("temperature,seed", [(None, None), (0.1, 7)])
def test_trainer_episode_env_matches_the_probe(temperature, seed):
    probe_env = _probe_env_block()
    ours = episode_env(
        model=Path("models/gnn-linkmp-lgon-s8.pt"), arm="gnn",
        temperature=temperature, seed=seed, reservoir_k=0, replay_out=None,
    )
    for key, value in probe_env.items():
        if key in ("GNN_DECODE_MODE", "GNN_SAMPLE_TEMPERATURE", "GNN_SAMPLE_SEED"):
            continue  # set per-arm below
        if value == "<dynamic>":
            assert key in ours, f"{key} is set by the probe but not by the trainer's episodes"
            continue
        assert ours.get(key) == value, (
            f"{key}: probe sets {value!r}, trainer episode sets {ours.get(key)!r}. The "
            f"GO verdict was measured under the probe's value."
        )


def test_no_layout_or_contract_variable_is_exported():
    """The loader adopts these from the sidecar; exporting one changes every score."""
    ours = episode_env(
        model=Path("m.pt"), arm="gnn", temperature=0.1, seed=1,
        reservoir_k=8, replay_out=Path("/tmp/r.pt"),
    )
    base = dict(__import__("os").environ)
    for key in ("INFERENCE_FEATURE_LAYOUT", "NETWORK_GRAPH_CONTRACT",
                "QUEUE_FEATURE_CONTRACT", "TOPOLOGY_FEATURE_CONTRACT"):
        assert ours.get(key) == base.get(key), (
            f"{key} is set by the episode environment. A declared-but-different layout "
            f"is a hard error in the loader; a declared-and-matching one is worse, "
            f"because it stops the sidecar from being the source of truth."
        )


def test_greedy_episode_carries_no_sampling_state():
    ours = episode_env(model=Path("m.pt"), arm="gnn", temperature=None, seed=None)
    assert ours["GNN_DECODE_MODE"] == "argmax"
    for key in ("GNN_SAMPLE_TEMPERATURE", "GNN_SAMPLE_SEED",
                "GNN_SAMPLE_RESERVOIR_K", "HEROSIM_EPISODE_REPLAY_OUT"):
        assert key not in ours, (
            f"the self-critical baseline episode carries {key}; a stale value from the "
            f"parent environment would make the baseline stochastic and the advantage "
            f"a difference of two noisy numbers instead of one."
        )


def test_sampled_episode_declares_everything_it_needs():
    ours = episode_env(
        model=Path("m.pt"), arm="gnn", temperature=0.3, seed=11,
        reservoir_k=64, replay_out=Path("/tmp/r.pt"),
    )
    assert ours["GNN_DECODE_MODE"] == "sample"
    assert ours["GNN_SAMPLE_TEMPERATURE"] == "0.3"
    assert ours["GNN_SAMPLE_SEED"] == "11"
    assert ours["GNN_SAMPLE_RESERVOIR_K"] == "64"
    assert ours["HEROSIM_EPISODE_REPLAY_OUT"] == "/tmp/r.pt"


def test_arm_policy_names_are_accepted_by_the_runner():
    """`--policy` takes the runner's registry names, not simulation.py strategy strings.

    The authority is `valid_policies` in executesimulation.py — the list the runner
    actually checks the flag against, and the one that rejects a wrong guess after a 5 s
    startup round-trip. simulation.py's dict is keyed by strategy strings
    (`mlp_batch_mlp_batch`, `kn_network_batch_kn_network_batch`) and is NOT what --policy
    is matched on; checking it instead would pass for the wrong reason.
    """
    source = (REPO_ROOT / "src/executesimulation.py").read_text()
    block = source.split("valid_policies = [", 1)[1].split("]", 1)[0]
    valid = set(re.findall(r"'([a-z_0-9]+)'", block))
    assert valid, "could not parse valid_policies; this test has gone blind"
    for arm, name in ARM_POLICY.items():
        assert name in valid, (
            f"arm {arm!r} maps to policy {name!r}, which executesimulation.py rejects. "
            f"Known: {sorted(valid)}"
        )
