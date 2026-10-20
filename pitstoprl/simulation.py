"""Shared frozen-policy race rollout for validation, evaluation and demos."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .env import RaceEnv


def run_episode(env: RaceEnv, policy: Callable[[np.ndarray], int], seed: int) -> dict:
    obs = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    while not done:
        action = policy(obs)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
    return {
        "race_time": float(info["total_time"]),
        "reward": total_reward,
        "pit_stops": env.pit_count,
        "compliant": env.changed_compound,
        "history": list(env.history),
    }
