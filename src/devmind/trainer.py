from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from devmind.agent import PPONetwork
from devmind.environment import InferenceGatewayEnv, ScenarioConfig


@dataclass
class PPOBuffer:
    states: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def clear(self) -> None:
        for attr in ["states", "actions", "rewards", "dones", "log_probs", "values"]:
            getattr(self, attr).clear()

    def to_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "states": torch.FloatTensor(np.array(self.states)).to(device),
            "actions": torch.LongTensor(np.array(self.actions)).to(device),
            "rewards": torch.FloatTensor(np.array(self.rewards)).to(device),
            "dones": torch.BoolTensor(np.array(self.dones)).to(device),
            "log_probs": torch.FloatTensor(np.array(self.log_probs)).to(device),
            "values": torch.FloatTensor(np.array(self.values)).to(device),
        }


class PPOTrainer:
    def __init__(
        self,
        env: InferenceGatewayEnv,
        policy_lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        epochs: int = 4,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.env = env
        self.policy = PPONetwork()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.buffer = PPOBuffer()
        self._step = 0

    def collect_rollout(self, steps: int) -> None:
        state, _ = self.env.reset()
        for _ in range(steps):
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits, value = self.policy(s)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            next_state, reward, done, _, _ = self.env.step(int(action.item()))
            self.buffer.states.append(state)
            self.buffer.actions.append(int(action.item()))
            self.buffer.rewards.append(reward)
            self.buffer.dones.append(done)
            self.buffer.log_probs.append(float(log_prob.item()))
            self.buffer.values.append(float(value.item()))
            state = next_state
            self._step += 1
            if done:
                state, _ = self.env.reset()

    def train(self) -> dict[str, float]:
        data = self.buffer.to_tensors(self.device)
        returns = self._compute_returns(data["rewards"], data["dones"])
        advantages = self._compute_gae(data["rewards"], data["values"], data["dones"])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        n = len(self.buffer.states)
        indices = np.arange(n)

        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.batch_size):
                idx = indices[start : start + self.batch_size]
                batch_states = data["states"][idx]
                batch_actions = data["actions"][idx]
                batch_log_probs = data["log_probs"][idx]
                batch_returns = returns[idx]
                batch_adv = advantages[idx]

                logits, values = self.policy(batch_states)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_log_probs)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                total_loss += float(loss.item())

        self.buffer.clear()
        return {"loss": total_loss / max(n // self.batch_size, 1)}

    def _compute_returns(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        returns = []
        R = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - float(d))
            returns.insert(0, R)
        return torch.tensor(returns, device=self.device)

    def _compute_gae(
        self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        advantages = []
        gae = 0.0
        values_next = torch.cat([values[1:], torch.zeros(1, device=self.device)])
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_next[t] * (1 - float(dones[t])) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - float(dones[t])) * gae
            advantages.insert(0, gae)
        return torch.tensor(advantages, device=self.device)


def randomize_scenario(base: ScenarioConfig, rng: np.random.Generator) -> ScenarioConfig:
    return dataclasses.replace(
        base,
        base_rate=float(rng.uniform(2000, 6000)),
        burst_rate=float(rng.uniform(6000, 25000)),
        burst_start_frac=float(rng.uniform(0.1, 0.5)),
        burst_duration_frac=float(rng.uniform(0.2, 0.6)),
        degraded_start_frac=float(rng.uniform(0.0, 0.3)),
        degraded_duration_frac=float(rng.uniform(0.3, 1.0)),
        rtt_base=float(rng.uniform(20, 150)),
        rtt_degraded=float(rng.uniform(200, 1000)),
        edge_stress_prob=float(rng.uniform(0.05, 0.4)),
        edge_degrade_prob=float(rng.uniform(0.01, 0.15)),
        edge_unreachable_prob=float(rng.uniform(0.0, 0.03)),
        cloud_service_time_ms=float(rng.uniform(5, 60)),
    )


def train_agent(
    scenario: ScenarioConfig,
    total_steps: int = 50_000,
    domain_randomization: bool = True,
    seed: int | None = None,
) -> tuple[PPONetwork, np.ndarray]:
    env = InferenceGatewayEnv(scenario)
    trainer = PPOTrainer(env)
    rng = np.random.default_rng(seed)
    step = 0
    seen_states: list[np.ndarray] = []
    pbar = tqdm(total=total_steps, desc="Training PPO")
    while step < total_steps:
        if domain_randomization:
            env.scenario = randomize_scenario(scenario, rng)
        trainer.collect_rollout(2048)
        seen_states.extend(trainer.buffer.states)
        metrics = trainer.train()
        step += 2048
        pbar.update(2048)
        pbar.set_postfix(loss=f"{metrics['loss']:.4f}")
    pbar.close()
    return trainer.policy, np.stack(seen_states, axis=0)


def save_state_stats(states: np.ndarray, path: str) -> None:
    import json

    mean = states.mean(axis=0)
    std = np.maximum(states.std(axis=0), 1e-6)
    with open(path, "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)


def main() -> None:
    print("Training PPO agent with domain randomization (steady base scenario)...")
    policy, states = train_agent(ScenarioConfig.steady(), domain_randomization=True)
    torch.save(policy.state_dict(), "ppo_policy.pt")
    save_state_stats(states, "state_stats.json")
    print("Saved ppo_policy.pt and state_stats.json")


def demo() -> None:
    rng = np.random.default_rng(0)
    base = ScenarioConfig.steady()
    samples = [randomize_scenario(base, rng) for _ in range(20)]
    assert len({s.rtt_base for s in samples}) > 1, "randomize_scenario should vary rtt_base across calls"
    assert all(20 <= s.rtt_base <= 150 for s in samples)
    assert all(200 <= s.rtt_degraded <= 1000 for s in samples)
    assert all(s.name == base.name and s.task == base.task for s in samples)
    print("trainer self-check passed")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":
        demo()
    else:
        main()
