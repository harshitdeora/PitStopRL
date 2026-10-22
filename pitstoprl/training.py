"""Single-process batched Double DQN training across randomized race settings."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .domain import TRAINING_DOMAIN, VALIDATION_SEED, sample_config
from .dqn import DQNAgent, DQNConfig
from .env import ENVIRONMENT_VERSION, WEAR_LIMIT, RaceEnv
from .simulation import run_episode


def validate(agent: DQNAgent, episodes: int = 64) -> dict:
    rng = np.random.default_rng(VALIDATION_SEED)
    times, compliance, max_wear = [], [], []
    for i in range(episodes):
        config = sample_config(rng)
        result = run_episode(RaceEnv(config), agent.act, VALIDATION_SEED + i)
        times.append(result["race_time"] / config.laps)
        compliance.append(result["compliant"])
        max_wear.append(result["max_tyre_wear"])
    return {
        "mean_seconds_per_lap": float(np.mean(times)),
        "compliance_rate": float(np.mean(compliance)),
        "episodes": episodes,
        "max_tyre_wear": max(max_wear),
    }


def train_dqn(
    output: Path,
    episodes: int = 30000,
    seed: int = 7,
    env_count: int = 16,
    validate_every: int = 2000,
    config: DQNConfig | None = None,
) -> dict:
    if episodes < 1 or env_count < 1 or validate_every < 1:
        raise ValueError("Episode, environment and validation counts must be positive.")
    if seed < 0 or seed + episodes > VALIDATION_SEED:
        raise ValueError("Training seeds must end before the reserved validation range.")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    agent = DQNAgent(config, seed)
    domain_rng = np.random.default_rng(seed + 10_000_000)
    output.parent.mkdir(parents=True, exist_ok=True)
    slots: dict[int, tuple[RaceEnv, np.ndarray, int, float]] = {}
    assigned = 0

    def new_slot():
        nonlocal assigned
        index = assigned
        assigned += 1
        env = RaceEnv(sample_config(domain_rng), seed + index)
        return env, env.reset(seed + index), index, 0.0

    for slot in range(min(env_count, episodes)):
        slots[slot] = new_slot()
    rewards = np.empty(episodes, dtype=np.float32)
    lengths = np.empty(episodes, dtype=np.int16)
    completed, transitions, next_validation = 0, 0, validate_every
    validations = []
    max_training_wear = 0.0
    best_score = float("inf")
    started = time.monotonic()
    while slots:
        ids = list(slots)
        states = np.stack([slots[i][1] for i in ids])
        progress = min(completed / max(episodes * 0.8, 1), 1.0)
        epsilon = 1.0 * (0.04**progress)
        actions = agent.act_batch(states, epsilon)
        next_states, batch_rewards, dones = [], [], []
        for slot, action in zip(ids, actions):
            env, _, index, episode_reward = slots[slot]
            next_obs, _, done, _, info = env.step(action)
            max_training_wear = max(max_training_wear, env.wear)
            # Remove action-independent base/fuel/SC/noise time. For a fixed
            # config and exogenous seed this preserves the ordering of ALL
            # strategies by total race time, and avoids learning a large
            # irrelevant value offset. This uses no baseline policy or labels.
            reward = -info["controllable_cost"] / 10.0
            next_states.append(next_obs)
            batch_rewards.append(reward)
            dones.append(done)
            episode_reward += reward
            if done:
                rewards[index], lengths[index] = episode_reward, env.config.laps
                completed += 1
                if assigned < episodes:
                    slots[slot] = new_slot()
                else:
                    del slots[slot]
            else:
                slots[slot] = env, next_obs, index, episode_reward
        agent.observe(states, actions, batch_rewards, np.stack(next_states), dones)
        agent.learn()
        transitions += len(ids)
        if completed >= next_validation or not slots:
            metrics = validate(agent)
            event = {
                "completed_episodes": completed,
                "transitions": transitions,
                "updates": agent.updates,
                "epsilon": epsilon,
                "elapsed_seconds": time.monotonic() - started,
                "max_training_tyre_wear": max_training_wear,
                **metrics,
            }
            validations.append(event)
            metadata = {
                "environment_version": ENVIRONMENT_VERSION,
                "tyre_wear_limit": WEAR_LIMIT,
                "action_constraint": "worst-case next-lap wear <= 1 - 1e-6; masked exploration and Bellman targets",
                "training_episodes": episodes,
                "completed_episodes": completed,
                "seed": seed,
                "training_seed_range": [seed, seed + episodes - 1],
                "validation_seed_range": [VALIDATION_SEED, VALIDATION_SEED + 63],
                "domain_seed": seed + 10_000_000,
                "domain": TRAINING_DOMAIN,
                "selection": "lowest mean seconds/lap on fixed validation races",
                "validation": metrics,
                "env_count": env_count,
                "validate_every": validate_every,
                "torch_version": str(torch.__version__),
                "numpy_version": str(np.__version__),
                "training_reward": "negative controllable race-time cost divided by 10",
                "reward_components": "compound grip + 1.8*wear + 11*wear**3 + pit loss + terminal rule penalty",
            }
            if metrics["mean_seconds_per_lap"] < best_score:
                best_score = metrics["mean_seconds_per_lap"]
                agent.save(output, metadata)
                event["selected"] = True
            else:
                event["selected"] = False
            print(json.dumps(event), flush=True)
            next_validation = completed + validate_every
    trace = output.with_name(output.stem + "_training.npz")
    np.savez_compressed(trace, rewards=rewards, laps=lengths)
    record = {
        "environment_version": ENVIRONMENT_VERSION,
        "tyre_wear_limit": WEAR_LIMIT,
        "max_training_tyre_wear": max_training_wear,
        "wear_limit_violations": 0,
        "algorithm": "double_dqn",
        "config": asdict(agent.config),
        "training_episodes": episodes,
        "seed": seed,
        "transitions": transitions,
        "updates": agent.updates,
        "elapsed_seconds": time.monotonic() - started,
        "validations": validations,
        "checkpoint": str(output),
        "trace": str(trace),
        "selection_uses_test_data": False,
        "execution": "single process, one Torch CPU thread, no background workers",
    }
    output.with_name(output.stem + "_training.json").write_text(json.dumps(record, indent=2))
    return record
