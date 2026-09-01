"""Guards for the Phase 3 (P1 closed-loop) sampled decode.

The sampled decode is the only stochastic decode mode in the tree, and it feeds a
policy-gradient estimator, so three things must hold or the whole loop is wrong:
  1. it is REPRODUCIBLE from its seed (an unreproducible episode cannot be paired
     across arms, which is the entire CRN design),
  2. the recorded log-probs are the log-probs of the ACTIONS ACTUALLY TAKEN
     (a mismatch silently inverts the gradient),
  3. it does not perturb any deterministic mode — every existing gate must be
     bit-identical with this code present.
"""

import math

import pytest
import torch

from src.policy.gnn import seq_decode as sd


def _logits(rows):
    return [torch.tensor(r, dtype=torch.float32) for r in rows]


PLACEMENTS = {0: [(1, 10), (2, 20), (3, 30)], 1: [(1, 10), (4, 40)]}
KEYS = {0: ["n1:10", "n2:20", "n3:30"], 1: ["n1:10", "n4:40"]}
LOGITS = _logits([[0.2, 1.5, -0.3], [0.7, 0.9]])


class TestReproducibility:
    def test_same_seed_same_episode(self):
        out = []
        for _ in range(2):
            sd.reset_episode_trajectory(temperature=1.0, seed=7)
            combos = [
                sd.decode_sampled_placement(LOGITS, PLACEMENTS, 2, {}, KEYS, temperature=1.0)
                for _ in range(25)
            ]
            out.append(combos)
        assert out[0] == out[1]

    def test_different_seed_different_episode(self):
        runs = []
        for seed in (1, 2):
            sd.reset_episode_trajectory(temperature=1.0, seed=seed)
            runs.append([
                sd.decode_sampled_placement(LOGITS, PLACEMENTS, 2, {}, KEYS, temperature=1.0)
                for _ in range(40)
            ])
        assert runs[0] != runs[1]

    def test_sampling_without_open_episode_fails_loud(self):
        sd.clear_episode_trajectory()
        with pytest.raises(RuntimeError, match="no open episode"):
            sd.decode_sampled_placement(LOGITS, PLACEMENTS, 2, {}, KEYS, temperature=1.0)

    def test_nonpositive_temperature_fails_loud(self):
        with pytest.raises(ValueError, match="temperature"):
            sd.reset_episode_trajectory(temperature=0.0, seed=1)


class TestLogProbsMatchActions:
    def test_recorded_logprob_is_the_chosen_actions_logprob(self):
        sd.reset_episode_trajectory(temperature=0.7, seed=3)
        traj = sd.get_episode_trajectory()
        combo = sd.decode_sampled_placement(
            LOGITS, PLACEMENTS, 2, {}, KEYS, temperature=0.7
        )
        assert combo is not None and len(traj.task_choices) == 2
        for t_idx, (chosen, lp) in enumerate(zip(traj.task_choices, traj.logprobs)):
            expected = torch.log_softmax(
                LOGITS[t_idx].to(torch.float64) / 0.7, dim=0
            )[chosen].item()
            assert lp == pytest.approx(expected, abs=1e-12)
            # and the chosen index really is the placement that came out
            assert combo[t_idx] == PLACEMENTS[t_idx][chosen]

    def test_logprobs_are_negative_and_finite(self):
        sd.reset_episode_trajectory(temperature=1.0, seed=11)
        traj = sd.get_episode_trajectory()
        for _ in range(30):
            sd.decode_sampled_placement(LOGITS, PLACEMENTS, 2, {}, KEYS, temperature=1.0)
        assert traj.logprobs and all(lp < 0 and math.isfinite(lp) for lp in traj.logprobs)
        assert len(traj.task_choices) == len(traj.logprobs) == 60

    def test_low_temperature_concentrates_on_argmax(self):
        sd.reset_episode_trajectory(temperature=0.01, seed=5)
        combos = [
            sd.decode_sampled_placement(LOGITS, PLACEMENTS, 2, {}, KEYS, temperature=0.01)
            for _ in range(40)
        ]
        # task 0's argmax is index 1; at T=0.01 sampling should agree essentially always
        assert all(c[0] == PLACEMENTS[0][1] for c in combos)

    def test_summary_shape(self):
        sd.reset_episode_trajectory(temperature=0.3, seed=2)
        traj = sd.get_episode_trajectory()
        for _ in range(5):
            sd.decode_sampled_placement(LOGITS, PLACEMENTS, 2, {}, KEYS, temperature=0.3)
        d = traj.to_dict()
        assert d["n_decisions"] == 10 and d["temperature"] == 0.3
        assert d["sum_logprob"] < 0 and d["mean_n_candidates"] == pytest.approx(2.5)


class TestDeterministicModesUnaffected:
    def test_argmax_decode_ignores_trajectory_state(self):
        sd.reset_episode_trajectory(temperature=1.0, seed=1)
        traj = sd.get_episode_trajectory()
        combo = sd.decode_sequential_placement(LOGITS, PLACEMENTS, 2, {}, KEYS)
        assert combo == (PLACEMENTS[0][1], PLACEMENTS[1][1])
        assert traj.task_choices == []  # the argmax path records nothing

    def test_argmax_is_stable_across_repeats(self):
        sd.clear_episode_trajectory()
        first = sd.decode_sequential_placement(LOGITS, PLACEMENTS, 2, {}, KEYS)
        for _ in range(10):
            assert sd.decode_sequential_placement(LOGITS, PLACEMENTS, 2, {}, KEYS) == first

    def test_sample_is_a_known_mode_and_argmax_still_is(self):
        assert {"sample", "sampled", "argmax"} <= sd.KNOWN_DECODE_MODES
