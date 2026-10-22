"""Original, small PyTorch Double DQN implementation.

Training targets come exclusively from environment rewards and bootstrapped
action values. No demonstrations, baseline labels, or pit-decision overrides.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .env import ENVIRONMENT_VERSION, WEAR_LIMIT, RaceEnv, action_mask


@dataclass(frozen=True)
class DQNConfig:
    observation_n: int = RaceEnv.observation_n
    action_n: int = 4
    hidden_n: int = 128
    learning_rate: float = 0.0003
    gamma: float = 1.0
    batch_size: int = 256
    replay_capacity: int = 100_000
    warmup: int = 4096
    target_interval: int = 250
    gradient_clip: float = 10.0

    def __post_init__(self):
        counts = (
            self.observation_n,
            self.action_n,
            self.hidden_n,
            self.batch_size,
            self.replay_capacity,
            self.warmup,
            self.target_interval,
        )
        if any(not isinstance(n, int) or n < 1 for n in counts):
            raise ValueError("Network/replay sizes and intervals must be positive integers.")
        if self.observation_n != RaceEnv.observation_n or self.action_n != 4:
            raise ValueError("Network dimensions must match the contextual race environment.")
        if not self.batch_size <= self.warmup <= self.replay_capacity:
            raise ValueError("Require batch_size <= warmup <= replay_capacity.")
        if not 0 <= self.gamma <= 1 or self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("Invalid DQN learning settings.")


class QNetwork(nn.Module):
    def __init__(self, config: DQNConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.observation_n, config.hidden_n),
            nn.ReLU(),
            nn.Linear(config.hidden_n, config.hidden_n),
            nn.ReLU(),
            nn.Linear(config.hidden_n, config.action_n),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.layers(observations)


class ReplayBuffer:
    def __init__(self, capacity: int, observation_n: int, seed: int):
        self.states = np.empty((capacity, observation_n), dtype=np.float32)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.done = np.empty(capacity, dtype=np.float32)
        self.capacity, self.size, self.position = capacity, 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, states, actions, rewards, next_states, done) -> None:
        count = len(actions)
        if count > self.capacity:
            raise ValueError("A transition batch cannot exceed replay capacity.")
        indices = (self.position + np.arange(count)) % self.capacity
        self.states[indices] = states
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.next_states[indices] = next_states
        self.done[indices] = done
        self.position = (self.position + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample(self, count: int):
        if count > self.size:
            raise ValueError("Not enough transitions in replay.")
        indices = self.rng.choice(self.size, size=count, replace=False)
        return tuple(
            torch.from_numpy(a[indices])
            for a in (
                self.states,
                self.actions,
                self.rewards,
                self.next_states,
                self.done,
            )
        )


class DQNAgent:
    def __init__(self, config: DQNConfig | None = None, seed: int = 7, training: bool = True):
        self.config = config or DQNConfig()
        self.rng = np.random.default_rng(seed)
        # fork_rng avoids changing unrelated callers' global Torch RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = QNetwork(self.config)
            self.target = QNetwork(self.config)
        self.target.load_state_dict(self.network.state_dict())
        self.target.requires_grad_(False)
        self.target.eval()
        self.optimizer = (
            torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
            if training
            else None
        )
        self.replay = (
            ReplayBuffer(self.config.replay_capacity, self.config.observation_n, seed)
            if training
            else None
        )
        self.updates = 0
        self.metadata: dict = {}

    def act_batch(self, observations: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[1] != self.config.observation_n:
            raise ValueError("Observation dimensions do not match this checkpoint.")
        if not np.isfinite(observations).all() or not 0 <= epsilon <= 1:
            raise ValueError("Observations must be finite and epsilon in [0, 1].")
        with torch.inference_mode():
            values = self.network(torch.from_numpy(observations))
            allowed = action_mask(observations)
            values = values.masked_fill(~torch.from_numpy(allowed), -torch.inf)
            actions = values.argmax(dim=1).numpy()
        if epsilon:
            mask = self.rng.random(len(actions)) < epsilon
            # Independent continuous draws give a uniform choice among legal
            # actions only; exploration must obey the same physics as inference.
            draws = self.rng.random((int(mask.sum()), self.config.action_n))
            draws[~allowed[mask]] = -1
            actions[mask] = draws.argmax(axis=1)
        return actions

    def act(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        return int(self.act_batch(np.asarray(observation)[None, :], epsilon)[0])

    def observe(self, states, actions, rewards, next_states, done) -> None:
        if self.replay is None:
            raise RuntimeError("This agent was loaded for inference only.")
        self.replay.add(states, actions, rewards, next_states, done)

    def learn(self) -> float | None:
        if self.replay is None or self.optimizer is None:
            raise RuntimeError("This agent was loaded for inference only.")
        if self.replay.size < self.config.warmup:
            return None
        states, actions, rewards, next_states, done = self.replay.sample(self.config.batch_size)
        predicted = self.network(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            # Double DQN: online network selects; frozen target network evaluates.
            allowed = torch.from_numpy(action_mask(next_states.numpy()))
            selected = self.network(next_states).masked_fill(~allowed, -torch.inf).argmax(
                dim=1, keepdim=True
            )
            future = self.target(next_states).gather(1, selected).squeeze(1)
            targets = rewards + self.config.gamma * (1.0 - done) * future
        loss = nn.functional.smooth_l1_loss(predicted, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite DQN loss; checkpoint was not saved.")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config.target_interval == 0:
            self.target.load_state_dict(self.network.state_dict())
        return float(loss.item())

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 2,
            "environment_version": ENVIRONMENT_VERSION,
            "tyre_wear_limit": WEAR_LIMIT,
            "algorithm": "double_dqn",
            "observation_names": list(RaceEnv.OBSERVATION_NAMES),
            "config": asdict(self.config),
            "updates": self.updates,
            "state_dict": self.network.state_dict(),
            "metadata": self.metadata | (metadata or {}),
        }
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> DQNAgent:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (payload.get("format_version") != 2 or payload.get("algorithm") != "double_dqn"
                or payload.get("environment_version") != ENVIRONMENT_VERSION
                or payload.get("tyre_wear_limit") != WEAR_LIMIT):
            raise ValueError("Checkpoint does not match the 100% wear-limit environment; retrain it.")
        if payload["observation_names"] != list(RaceEnv.OBSERVATION_NAMES):
            raise ValueError("Checkpoint observation schema does not match this environment.")
        agent = cls(DQNConfig(**payload["config"]), training=False)
        agent.network.load_state_dict(payload["state_dict"])
        agent.target.load_state_dict(payload["state_dict"])
        agent.network.eval()
        agent.updates = int(payload["updates"])
        agent.metadata = payload["metadata"]
        return agent
