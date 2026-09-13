"""Guards for the Phase 3 (P1 closed-loop) policy-gradient estimator.

The loop optimises the live metric with no labels anywhere, so nothing downstream will
notice if the estimator is wrong — a broken gradient produces a training curve that
looks exactly like a hard problem. These are the checks that can tell the difference:

  1. the reservoir includes every batch with probability k/N, which is the *only* reason
     the N/k rescale is unbiased;
  2. a decode that failed and fell back leaves no log-probs behind, so the episode's
     return is never credited to actions the simulator did not execute;
  3. pass 2 replays the same distribution pass 1 sampled from — checked here on real
     modules, because the trainer's own runtime check can only fire after an episode has
     already been paid for;
  4. the reservoir draws never touch the action RNG, so two arms paired under common
     random numbers still share a trace at different `--reservoir-k`.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.policy.gnn import seq_decode as sd

PLACEMENTS = {0: [(1, 10), (2, 20), (3, 30)], 1: [(1, 10), (4, 40)]}
KEYS = {0: ["n1:10", "n2:20", "n3:30"], 1: ["n1:10", "n4:40"]}
LOGITS = [
    torch.tensor([0.2, 1.5, -0.3], dtype=torch.float32),
    torch.tensor([0.7, 0.9], dtype=torch.float32),
]


def _decode(temperature=1.0):
    return sd.decode_sampled_placement(
        LOGITS, PLACEMENTS, 2, {}, KEYS, temperature=temperature
    )


class TestReservoirIsUniform:
    def test_every_batch_has_inclusion_probability_k_over_n(self):
        """Algorithm R, measured. This is what makes the N/k rescale unbiased."""
        n_batches, k, trials = 20, 5, 4000
        counts = [0] * n_batches
        for t in range(trials):
            sd.reset_episode_trajectory(temperature=1.0, seed=t, reservoir_k=k)
            traj = sd.get_episode_trajectory()
            for b in range(n_batches):
                _decode()
                traj.offer_replay(lambda b=b: b)
            for item in traj.reservoir:
                counts[item.payload] += 1
        expected = trials * k / n_batches
        for b, c in enumerate(counts):
            # ~3.5 binomial sd; a systematically favoured position (the classic
            # off-by-one in Algorithm R) lands far outside this.
            assert abs(c - expected) < 4.0 * math.sqrt(
                trials * (k / n_batches) * (1 - k / n_batches)
            ), f"batch {b} kept {c} times, expected ~{expected:.0f}"

    def test_reservoir_holds_exactly_k_once_past_k(self):
        sd.reset_episode_trajectory(temperature=1.0, seed=1, reservoir_k=3)
        traj = sd.get_episode_trajectory()
        for b in range(10):
            _decode()
            traj.offer_replay(lambda b=b: b)
        assert len(traj.reservoir) == 3
        assert traj.n_batches == 10

    def test_k_zero_records_batch_count_but_stores_nothing(self):
        """The Increment-1 probe ran this way; it must stay free."""
        sd.reset_episode_trajectory(temperature=1.0, seed=1, reservoir_k=0)
        traj = sd.get_episode_trajectory()
        calls = []
        for _ in range(5):
            _decode()
            traj.offer_replay(lambda: calls.append(1))
        assert traj.reservoir == [] and traj.n_batches == 5
        assert calls == [], "payload factory ran despite the reservoir being disabled"

    def test_payload_factory_runs_at_most_once_per_batch(self):
        """An episode is ~7.5k batches; materialising every payload is the cost this avoids."""
        sd.reset_episode_trajectory(temperature=1.0, seed=2, reservoir_k=4)
        traj = sd.get_episode_trajectory()
        made = 0

        def factory():
            nonlocal made
            made += 1
            return made

        for _ in range(200):
            _decode()
            traj.offer_replay(factory)
        assert made <= 200
        # Past the fill phase acceptance is k/(i+1), so the expected total is
        # k*(1 + ln(N/k)) ~= 4*(1+ln(50)) ~= 20, nowhere near 200.
        assert made < 60, f"materialised {made} payloads for a k=4 reservoir"


class TestAbandonedBatchesLeaveNoTrace:
    def test_abandon_removes_exactly_the_open_batch(self):
        sd.reset_episode_trajectory(temperature=1.0, seed=3, reservoir_k=2)
        traj = sd.get_episode_trajectory()
        _decode()
        traj.offer_replay(lambda: "kept")
        assert len(traj.task_choices) == 2
        _decode()  # a second batch that we then abandon
        assert len(traj.task_choices) == 4
        dropped = traj.abandon_open_batch()
        assert dropped == 2
        assert len(traj.task_choices) == 2
        assert len(traj.logprobs) == 2
        assert len(traj.task_n_candidates) == 2
        assert traj.n_batches == 1

    def test_abandon_then_commit_keeps_the_committed_one(self):
        sd.reset_episode_trajectory(temperature=1.0, seed=4, reservoir_k=2)
        traj = sd.get_episode_trajectory()
        _decode()
        traj.abandon_open_batch()
        _decode()
        traj.offer_replay(lambda: "kept")
        assert traj.n_batches == 1
        assert len(traj.reservoir) == 1 and traj.reservoir[0].payload == "kept"
        assert len(traj.reservoir[0].chosen) == 2

    def test_abandon_on_a_clean_boundary_is_a_noop(self):
        sd.reset_episode_trajectory(temperature=1.0, seed=5, reservoir_k=1)
        traj = sd.get_episode_trajectory()
        _decode()
        traj.offer_replay(lambda: 0)
        assert traj.abandon_open_batch() == 0
        assert len(traj.task_choices) == 2


class TestReservoirDoesNotPerturbActions:
    def test_action_stream_is_identical_at_every_k(self):
        """CRN pairing dies silently if the reservoir consumes action draws."""
        runs = {}
        for k in (0, 1, 8, 64):
            sd.reset_episode_trajectory(temperature=1.0, seed=99, reservoir_k=k)
            traj = sd.get_episode_trajectory()
            for _ in range(60):
                _decode()
                traj.offer_replay(lambda: None)
            runs[k] = list(traj.task_choices)
        assert runs[0] == runs[1] == runs[8] == runs[64]


class TestReplayReproducesPassOne:
    """The check the trainer runs every step, exercised here on real modules."""

    def _mlp_policy(self):
        from scripts_cosim.closed_loop.adapters import LoadedPolicy
        from src.policy.tabular.mlp_model import PointwiseEdgeMLP

        torch.manual_seed(0)
        model = PointwiseEdgeMLP(input_dim=22, hidden_dim=16)
        model.eval()
        return LoadedPolicy(model=model, device=torch.device("cpu"), arm="mlp")

    def test_mlp_replay_matches_recorded_logprobs(self):
        from scripts_cosim.closed_loop.adapters import batch_logprob, replay_logits

        policy = self._mlp_policy()
        torch.manual_seed(1)
        feat = torch.randn(7, 22, dtype=torch.float64)
        boundaries = [(0, 3), (3, 7)]
        payload = (feat, boundaries)

        # Pass 1: forward under no_grad, sample, record — exactly the serving sequence.
        with torch.no_grad():
            logits = replay_logits(policy, payload)
        temperature = 0.3
        sd.reset_episode_trajectory(temperature=temperature, seed=17, reservoir_k=1)
        combo = sd.decode_sampled_placement(
            list(logits), {0: [(1, 1)] * 3, 1: [(2, 2)] * 4}, 2,
            {}, {0: ["a", "b", "c"], 1: ["d", "e", "f", "g"]},
            temperature=temperature,
        )
        assert combo is not None
        traj = sd.get_episode_trajectory()
        traj.offer_replay(lambda: payload)
        item = traj.reservoir[0]

        # Pass 2: replay with grad, on the same weights.
        replayed = batch_logprob(policy, item.payload, item.chosen, temperature)
        recorded = sum(item.logprobs)
        assert abs(float(replayed) - recorded) < 1e-9
        assert replayed.requires_grad, "pass 2 produced no autograd graph to descend"

    def test_gradient_flows_to_every_parameter(self):
        from scripts_cosim.closed_loop.adapters import batch_logprob

        policy = self._mlp_policy()
        torch.manual_seed(2)
        payload = (torch.randn(5, 22, dtype=torch.float64), [(0, 2), (2, 5)])
        lp = batch_logprob(policy, payload, [0, 1], 0.5)
        (-1.0 * lp).backward()
        for name, p in policy.model.named_parameters():
            assert p.grad is not None, f"{name} received no gradient"
            assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"

    def test_out_of_range_action_fails_loud(self):
        from scripts_cosim.closed_loop.adapters import batch_logprob

        policy = self._mlp_policy()
        payload = (torch.randn(4, 22, dtype=torch.float64), [(0, 2), (2, 4)])
        with pytest.raises(RuntimeError, match="outside the"):
            batch_logprob(policy, payload, [5, 0], 1.0)

    def test_more_recorded_decisions_than_replayed_rows_fails_loud(self):
        from scripts_cosim.closed_loop.adapters import batch_logprob

        policy = self._mlp_policy()
        payload = (torch.randn(4, 22, dtype=torch.float64), [(0, 4)])
        with pytest.raises(RuntimeError, match="does not correspond"):
            batch_logprob(policy, payload, [0, 1], 1.0)


class TestRescaleIsUnbiased:
    def test_n_over_k_recovers_the_full_episode_sum_in_expectation(self):
        """The estimator's one statistical claim, measured end to end."""
        n_batches, k, trials = 30, 6, 3000
        per_batch = [float(b) + 1.0 for b in range(n_batches)]
        truth = sum(per_batch)
        estimates = []
        for t in range(trials):
            sd.reset_episode_trajectory(temperature=1.0, seed=t, reservoir_k=k)
            traj = sd.get_episode_trajectory()
            for b in range(n_batches):
                _decode()
                traj.offer_replay(lambda b=b: b)
            got = sum(per_batch[item.payload] for item in traj.reservoir)
            estimates.append(got * n_batches / k)
        mean = sum(estimates) / trials
        assert abs(mean - truth) / truth < 0.02, (
            f"N/k rescale is biased: {mean:.1f} vs {truth:.1f}"
        )


class TestGateStatistic:
    """The exact test that produces the registered verdict.

    A one-sided exact Wilcoxon is the same test this program used for Phase 1,
    `mp_ablation_v1` and `link_mp_v1`, so the Phase 3 number has to be comparable to
    those. An off-by-one in the tail enumeration would move a p-value across alpha with
    nothing else in the pipeline noticing.
    """

    def test_matches_scipy_on_untied_samples(self):
        scipy_stats = pytest.importorskip("scipy.stats")
        from scripts_cosim.closed_loop.analyze_gate import exact_wilcoxon_greater

        cases = [
            [0.01, 0.02, -0.005, 0.03, 0.004, 0.011, -0.002, 0.008],
            [-0.01, -0.02, 0.005, -0.03, -0.004],
            [0.001 * (i + 1) for i in range(10)],
        ]
        for diffs in cases:
            _, ours = exact_wilcoxon_greater(diffs)
            ref = scipy_stats.wilcoxon(diffs, alternative="greater", mode="exact").pvalue
            assert ours == pytest.approx(ref, abs=1e-12), f"{diffs}: {ours} vs {ref}"

    def test_all_positive_hits_the_floor_p_value(self):
        from scripts_cosim.closed_loop.analyze_gate import exact_wilcoxon_greater

        # With every seed improving, p is 2^-n: the smallest the test can report, and the
        # reason n = 5 could reach alpha at all while n = 4 could not.
        for n in (5, 8, 16):
            _, p = exact_wilcoxon_greater([0.001 * (i + 1) for i in range(n)])
            assert p == pytest.approx(2.0 ** -n, rel=1e-9)

    def test_zero_differences_are_dropped_not_counted(self):
        from scripts_cosim.closed_loop.analyze_gate import exact_wilcoxon_greater

        _, with_zeros = exact_wilcoxon_greater([0.01, 0.0, 0.02, 0.0, 0.03])
        _, without = exact_wilcoxon_greater([0.01, 0.02, 0.03])
        assert with_zeros == pytest.approx(without)

    def test_a_null_sample_does_not_fire(self):
        from scripts_cosim.closed_loop.analyze_gate import exact_wilcoxon_greater

        _, p = exact_wilcoxon_greater([0.01, -0.011, 0.009, -0.012, 0.002, -0.003])
        assert p > 0.05


class TestLargeNGateStatistic:
    """The n > 22 path, added because Amendment E registered n = 120 against a tool that
    refused above n = 22. 2^120 sign patterns cannot be enumerated, so the same null is
    sampled instead. These prove it is the SAME test, not a more convenient one."""

    def test_matches_scipy_at_large_n(self):
        scipy_stats = pytest.importorskip("scipy.stats")
        import random

        from scripts_cosim.closed_loop.analyze_gate import exact_wilcoxon_greater

        for n, shift in ((30, 0.0), (60, 0.004), (120, 0.0), (120, 0.002)):
            rng = random.Random(n * 7 + int(shift * 1000))
            diffs = [rng.gauss(shift, 0.05) for _ in range(n)]
            _, ours = exact_wilcoxon_greater(diffs)
            ref = scipy_stats.wilcoxon(diffs, alternative="greater").pvalue
            assert abs(ours - ref) < 0.01, f"n={n} shift={shift}: {ours} vs scipy {ref}"

    def test_the_two_large_n_estimators_agree(self):
        import random

        from scripts_cosim.closed_loop.analyze_gate import (
            _monte_carlo_signflip_p, _normal_approx_p, exact_wilcoxon_greater,
        )

        rng = random.Random(11)
        diffs = [rng.gauss(0.001, 0.05) for _ in range(120)]
        w, _ = exact_wilcoxon_greater(diffs)
        # exact_wilcoxon_greater already raises if they disagree; assert it directly too,
        # so a loosened tolerance cannot silently hide a divergence.
        nz = [d for d in diffs if d != 0.0]
        order = sorted(range(len(nz)), key=lambda i: abs(nz[i]))
        ranks = [0.0] * len(nz)
        for pos, idx in enumerate(order):
            ranks[idx] = pos + 1.0
        assert abs(_monte_carlo_signflip_p(ranks, w) - _normal_approx_p(ranks, w, len(nz))) < 0.01

    def test_exact_and_sampled_paths_agree_at_the_boundary(self):
        """At n = 22 both paths are available; they must give the same answer."""
        import random

        from scripts_cosim.closed_loop import analyze_gate as ag

        rng = random.Random(5)
        diffs = [rng.gauss(0.003, 0.02) for _ in range(22)]
        _, exact_p = ag.exact_wilcoxon_greater(diffs)
        saved = ag.EXACT_MAX_N
        try:
            ag.EXACT_MAX_N = 10          # force the sampled path on the same data
            _, mc_p = ag.exact_wilcoxon_greater(diffs)
        finally:
            ag.EXACT_MAX_N = saved
        assert abs(exact_p - mc_p) < 0.01, f"exact {exact_p} vs sampled {mc_p}"

    def test_p_value_is_reproducible(self):
        import random

        from scripts_cosim.closed_loop.analyze_gate import exact_wilcoxon_greater

        rng = random.Random(3)
        diffs = [rng.gauss(0.0, 0.05) for _ in range(120)]
        assert exact_wilcoxon_greater(diffs)[1] == exact_wilcoxon_greater(diffs)[1]
