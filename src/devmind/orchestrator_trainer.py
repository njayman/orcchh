from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from devmind.agent import AgenticOrchestrator, PPONetwork
from devmind.environment import ScenarioConfig
from devmind.evaluation import EvalMetrics, make_env, ppo_policy, run_episode
from devmind.orchestrator import THRESHOLD_BOUNDS, PolicyDecision, PolicyOrchestrator, ToleranceThresholds
from devmind.trainer import PPOTrainer, randomize_scenario, save_state_stats

_ACTIONS = [PolicyDecision.REUSE, PolicyDecision.FINE_TUNE, PolicyDecision.TRAIN_NEW]


@dataclass
class MetaRewardWeights:
    cost_reuse: float = 0.0
    cost_fine_tune: float = 0.1
    cost_train_new: float = 0.3
    invalid_action_penalty: float = 2.0


def _quality(m: EvalMetrics, t: ToleranceThresholds) -> float:
    """+1.0 if m meets every tolerance; otherwise the negative sum of how far it
    misses each one. Reuses the exact same three comparisons as
    orchestrator.py's _meets_tolerance/dominant_signal, kept local to avoid a
    circular import (orchestrator.py doesn't import this module)."""
    gaps = [
        m.sla_violation_rate - t.max_sla_violation_rate,
        t.min_accuracy - m.accuracy,
        m.escalation_rate - t.max_escalation_rate,
    ]
    if all(g <= 0 for g in gaps):
        return 1.0
    return -sum(max(0.0, g) for g in gaps)


def build_observation(
    m: EvalMetrics, t: ToleranceThresholds, scenario: ScenarioConfig, library_empty: bool
) -> np.ndarray:
    """Shared between training (OrchestrationDecisionEnv) and live inference
    (PolicyOrchestrator._meta_decide) -- must stay the single source of truth
    for this feature layout, or the two silently drift out of sync."""
    gaps = [
        m.sla_violation_rate - t.max_sla_violation_rate,
        t.min_accuracy - m.accuracy,
        m.escalation_rate - t.max_escalation_rate,
    ]
    return np.array(
        [
            m.accuracy, m.sla_violation_rate, m.escalation_rate, m.fallback_rate,
            *gaps,
            scenario.rtt_base / 200.0,
            scenario.base_rate / 6000.0,
            scenario.burst_rate / 25000.0,
            scenario.edge_stress_prob,
            scenario.edge_degrade_prob,
            1.0 if library_empty else 0.0,
        ],
        dtype=np.float32,
    )


class OrchestrationDecisionEnv(gym.Env):
    """One episode = one synthetic client onboarding decision, always done=True
    after a single step -- a contextual bandit dressed as an episode-length-1
    MDP so PPOTrainer (trainer.py) works unmodified. Observation is 13-dim to
    match PPONetwork's default input_dim, deliberately the same width as the
    per-request Gold vector."""

    def __init__(
        self,
        seed_policy_path: str = "ppo_policy.pt",
        base_scenario: ScenarioConfig | None = None,
        meta_fine_tune_steps: int = 500,
        meta_train_new_steps: int = 800,
        max_samples: int = 200,
        empty_library_prob: float = 0.3,
        reward_weights: MetaRewardWeights | None = None,
        edge_model: Any = None,
        cloud_model: Any = None,
        rng_seed: int | None = None,
    ):
        super().__init__()
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(13,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self._edge_model = edge_model
        self._cloud_model = cloud_model
        self._base_scenario = base_scenario or ScenarioConfig.steady()
        self._meta_fine_tune_steps = meta_fine_tune_steps
        self._meta_train_new_steps = meta_train_new_steps
        self._max_samples = max_samples
        self._empty_library_prob = empty_library_prob
        self._weights = reward_weights or MetaRewardWeights()
        self._rng = np.random.default_rng(rng_seed)

        # library_dir is unused -- we call PolicyOrchestrator._train() directly
        # rather than _fine_tune()/_train_new(), so nothing gets saved to disk.
        self._orch = PolicyOrchestrator(
            library_dir=tempfile.mkdtemp(prefix="devmind_meta_env_"),
            fine_tune_steps=meta_fine_tune_steps,
            train_new_steps=meta_train_new_steps,
            edge_model=edge_model,
            cloud_model=cloud_model,
        )

        seed_ppo = PPONetwork()
        seed_ppo.load_state_dict(torch.load(seed_policy_path, map_location="cpu", weights_only=True))
        self._seed_state_dict = seed_ppo.state_dict()

        self._scenario: ScenarioConfig | None = None
        self._thresholds: ToleranceThresholds | None = None
        self._library_empty = False
        self._seed_metrics: EvalMetrics | None = None

    def _evaluate_policy(self, policy: PPONetwork, scenario: ScenarioConfig) -> EvalMetrics:
        agent = AgenticOrchestrator(policy)
        env = make_env(scenario, self._max_samples, self._edge_model, self._cloud_model)
        return run_episode(env, ppo_policy(agent), desc="meta_env/eval")

    def _sample_thresholds(self) -> ToleranceThresholds:
        return ToleranceThresholds(
            max_sla_violation_rate=float(self._rng.uniform(*THRESHOLD_BOUNDS["max_sla_violation_rate"])),
            min_accuracy=float(self._rng.uniform(*THRESHOLD_BOUNDS["min_accuracy"])),
            max_escalation_rate=float(self._rng.uniform(*THRESHOLD_BOUNDS["max_escalation_rate"])),
        )

    def _build_observation(
        self, m: EvalMetrics, t: ToleranceThresholds, scenario: ScenarioConfig, library_empty: bool
    ) -> np.ndarray:
        return build_observation(m, t, scenario, library_empty)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._scenario = randomize_scenario(self._base_scenario, self._rng)
        self._thresholds = self._sample_thresholds()
        self._library_empty = bool(self._rng.uniform() < self._empty_library_prob)

        if self._library_empty:
            self._seed_metrics = None
            worst = EvalMetrics(accuracy=0.0, sla_violation_rate=1.0, escalation_rate=1.0, fallback_rate=1.0)
            obs = self._build_observation(worst, self._thresholds, self._scenario, True)
        else:
            seed_ppo = PPONetwork()
            seed_ppo.load_state_dict(self._seed_state_dict)
            self._seed_metrics = self._evaluate_policy(seed_ppo, self._scenario)
            obs = self._build_observation(self._seed_metrics, self._thresholds, self._scenario, False)
        return obs, {}

    def step(self, action: int):
        decision = _ACTIONS[int(action)]
        w = self._weights

        if self._library_empty and decision in (PolicyDecision.REUSE, PolicyDecision.FINE_TUNE):
            # Nothing exists to reuse/fine-tune -- real onboard() doesn't offer these as
            # valid outcomes when the library is empty either (select_decision falls
            # straight to TRAIN_NEW). No training executed, keeps this branch cheap.
            reward = -w.invalid_action_penalty
        elif decision == PolicyDecision.REUSE:
            reward = _quality(self._seed_metrics, self._thresholds) - w.cost_reuse
        elif decision == PolicyDecision.FINE_TUNE:
            policy = self._orch._train(self._scenario, self._meta_fine_tune_steps, init_state_dict=self._seed_state_dict)
            metrics = self._evaluate_policy(policy, self._scenario)
            reward = _quality(metrics, self._thresholds) - w.cost_fine_tune
        else:
            policy = self._orch._train(self._scenario, self._meta_train_new_steps, init_state_dict=None)
            metrics = self._evaluate_policy(policy, self._scenario)
            reward = _quality(metrics, self._thresholds) - w.cost_train_new

        reward = float(np.clip(reward, -3.0, 1.5))
        obs = np.zeros(13, dtype=np.float32)  # terminal obs, unused since done=True every step
        return obs, reward, True, False, {"decision": decision.value}


def train_meta_policy(
    seed_policy_path: str = "ppo_policy.pt",
    total_steps: int = 320,
    rollout_size: int = 16,
    meta_fine_tune_steps: int = 500,
    meta_train_new_steps: int = 800,
    max_samples: int = 200,
    seed: int | None = None,
    edge_model: Any = None,
    cloud_model: Any = None,
) -> tuple[PPONetwork, np.ndarray]:
    if edge_model is None or cloud_model is None:
        from devmind.model_clients import BERTLargeCloud, DistilBERTEdge

        edge_model = edge_model or DistilBERTEdge()
        cloud_model = cloud_model or BERTLargeCloud()
    env = OrchestrationDecisionEnv(
        seed_policy_path=seed_policy_path,
        meta_fine_tune_steps=meta_fine_tune_steps,
        meta_train_new_steps=meta_train_new_steps,
        max_samples=max_samples,
        edge_model=edge_model,
        cloud_model=cloud_model,
        rng_seed=seed,
    )
    trainer = PPOTrainer(env, batch_size=min(64, rollout_size))
    step = 0
    seen_states: list[np.ndarray] = []
    while step < total_steps:
        trainer.collect_rollout(rollout_size)
        seen_states.extend(trainer.buffer.states)
        metrics = trainer.train()
        step += rollout_size
        print(f"meta step={step}/{total_steps} loss={metrics['loss']:.4f}")
    return trainer.policy, np.stack(seen_states, axis=0)


def main() -> None:
    print("Training orchestrator meta-policy (REUSE / FINE_TUNE / TRAIN_NEW)...")
    policy, states = train_meta_policy()
    torch.save(policy.state_dict(), "meta_policy.pt")
    save_state_stats(states, "meta_state_stats.json")
    print("Saved meta_policy.pt and meta_state_stats.json")


class _FakeModel:
    """No-network stand-in for DistilBERTEdge/BERTLargeCloud, matching their
    .predict(text, true_label) -> InferenceResult interface."""

    def predict(self, text: str, true_label: int | None = None):
        from devmind.model_clients import InferenceResult

        correct = true_label is None or (hash(text) % 2 == true_label % 2)
        return InferenceResult(confidence=0.7, latency_ms=5.0, is_correct=correct)


def demo() -> None:
    good = EvalMetrics(accuracy=0.9, sla_violation_rate=0.05, escalation_rate=0.3)
    t = ToleranceThresholds()
    assert _quality(good, t) == 1.0, "metrics that clear every threshold must score exactly 1.0"

    bad = EvalMetrics(accuracy=0.5, sla_violation_rate=0.5, escalation_rate=0.9)
    q_bad = _quality(bad, t)
    assert q_bad < 0, "metrics missing thresholds must score negative"
    expected = -(
        (bad.sla_violation_rate - t.max_sla_violation_rate)
        + (t.min_accuracy - bad.accuracy)
        + (bad.escalation_rate - t.max_escalation_rate)
    )
    assert abs(q_bad - expected) < 1e-9

    assert _ACTIONS == [PolicyDecision.REUSE, PolicyDecision.FINE_TUNE, PolicyDecision.TRAIN_NEW]

    fake = _FakeModel()
    env = OrchestrationDecisionEnv(
        seed_policy_path="ppo_policy.pt",
        meta_fine_tune_steps=4,
        meta_train_new_steps=4,
        max_samples=8,
        empty_library_prob=1.0,  # force the cheap invalid-action branch deterministically
        edge_model=fake,
        cloud_model=fake,
        rng_seed=0,
    )
    obs, info = env.reset(seed=0)
    assert obs.shape == (13,)
    assert obs[-1] == 1.0, "empty-library flag (last slot) must be set"
    assert env.observation_space.shape == (13,)
    assert env.action_space.n == 3

    _, reward, done, truncated, step_info = env.step(0)  # REUSE with an empty library
    assert reward < 0 and done and not truncated, "REUSE with nothing to reuse must be penalized, one-step episode"
    assert step_info["decision"] == "reuse"

    env2 = OrchestrationDecisionEnv(
        seed_policy_path="ppo_policy.pt",
        meta_fine_tune_steps=4,
        meta_train_new_steps=4,
        max_samples=8,
        empty_library_prob=0.0,  # force the real-seed branch
        edge_model=fake,
        cloud_model=fake,
        rng_seed=0,
    )
    obs2, _ = env2.reset(seed=0)
    assert obs2[-1] == 0.0
    _, reward2, done2, _, info2 = env2.step(0)  # REUSE the (fake-evaluated) seed
    assert done2 and info2["decision"] == "reuse"

    policy, states = train_meta_policy(
        seed_policy_path="ppo_policy.pt",
        total_steps=8,
        rollout_size=4,
        meta_fine_tune_steps=4,
        meta_train_new_steps=4,
        max_samples=8,
        seed=0,
        edge_model=fake,
        cloud_model=fake,
    )
    assert isinstance(policy, PPONetwork)
    assert states.shape == (8, 13)

    print("orchestrator_trainer self-check passed")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":
        demo()
    else:
        main()
