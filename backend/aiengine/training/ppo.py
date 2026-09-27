"""A clean, dependency-light PPO implementation (continuous actions).

This is the "engine" of the platform: it knows nothing about any specific
task, only about the generic Environment interface. It is intentionally
readable (CleanRL-style, single file) rather than hidden behind a
heavyweight framework, so it stays hackable and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from aiengine.envs.base import Environment


@dataclass
class PPOConfig:
    total_steps: int = 200_000
    rollout_steps: int = 2048
    epochs: int = 10
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    hidden_size: int = 128


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden, act_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h = self.backbone(obs)
        mean = torch.tanh(self.actor_mean(h))
        std = self.actor_logstd.exp().expand_as(mean)
        value = self.critic(h).squeeze(-1)
        return Normal(mean, std), value


@dataclass
class TrainingMetrics:
    step: int = 0
    episode: int = 0
    last_episode_return: float = 0.0
    mean_return_100: float = 0.0
    returns: list = field(default_factory=list)


class PPOTrainer:
    """Runs PPO against a single Environment instance, single process."""

    def __init__(
        self,
        env: Environment,
        config: PPOConfig | None = None,
        on_tick: Callable[[dict, TrainingMetrics], None] | None = None,
        device: str = "cpu",
    ) -> None:
        self.env = env
        self.cfg = config or PPOConfig()
        self.on_tick = on_tick
        self.device = torch.device(device)

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.model = ActorCritic(obs_dim, act_dim, self.cfg.hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

        self.metrics = TrainingMetrics()
        self._stop_flag = False
        self._episode_return = 0.0

    def request_stop(self) -> None:
        self._stop_flag = True

    def train(self) -> None:
        cfg = self.cfg
        obs = self.env.reset()
        while self.metrics.step < cfg.total_steps and not self._stop_flag:
            batch = self._collect_rollout(obs, cfg.rollout_steps)
            obs = batch.pop("last_obs")
            self._update(batch)
            if self._stop_flag:
                break

    def _collect_rollout(self, obs: np.ndarray, n_steps: int) -> dict:
        cfg = self.cfg
        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []

        for _ in range(n_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                dist, value = self.model(obs_t)
                action = dist.sample()
                logp = dist.log_prob(action).sum(-1)

            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, done, info = self.env.step(action_np)

            obs_buf.append(obs)
            act_buf.append(action_np)
            logp_buf.append(logp.item())
            rew_buf.append(reward)
            done_buf.append(done)
            val_buf.append(value.item())

            self._episode_return += reward
            self.metrics.step += 1

            if self.on_tick is not None:
                self.on_tick(self.env.render_state(), self.metrics)

            if done:
                self.metrics.episode += 1
                self.metrics.last_episode_return = self._episode_return
                self.metrics.returns.append(self._episode_return)
                self.metrics.returns = self.metrics.returns[-100:]
                self.metrics.mean_return_100 = float(np.mean(self.metrics.returns))
                self._episode_return = 0.0
                next_obs = self.env.reset()

            obs = next_obs
            if self._stop_flag:
                break

        with torch.no_grad():
            last_val = self.model(
                torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            )[1].item()

        advantages, returns = self._compute_gae(rew_buf, val_buf, done_buf, last_val)

        return {
            "obs": np.array(obs_buf, dtype=np.float32),
            "actions": np.array(act_buf, dtype=np.float32),
            "logp": np.array(logp_buf, dtype=np.float32),
            "advantages": advantages,
            "returns": returns,
            "last_obs": obs,
        }

    def _compute_gae(self, rewards, values, dones, last_val):
        cfg = self.cfg
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        next_val = last_val
        for t in reversed(range(n)):
            mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + cfg.gamma * next_val * mask - values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * mask * gae
            advantages[t] = gae
            next_val = values[t]
        returns = advantages + np.array(values, dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def _update(self, batch: dict) -> None:
        cfg = self.cfg
        obs = torch.as_tensor(batch["obs"], device=self.device)
        actions = torch.as_tensor(batch["actions"], device=self.device)
        old_logp = torch.as_tensor(batch["logp"], device=self.device)
        advantages = torch.as_tensor(batch["advantages"], device=self.device)
        returns = torch.as_tensor(batch["returns"], device=self.device)

        n = obs.shape[0]
        idx = np.arange(n)
        for _ in range(cfg.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, cfg.minibatch_size):
                mb = idx[start:start + cfg.minibatch_size]
                dist, value = self.model(obs[mb])
                logp = dist.log_prob(actions[mb]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratio = (logp - old_logp[mb]).exp()
                surr1 = ratio * advantages[mb]
                surr2 = torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * advantages[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((value - returns[mb]) ** 2).mean()
                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
