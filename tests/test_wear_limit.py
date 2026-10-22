"""Hard constraints, policy behaviour, and constrained DQN targets."""

import copy

import numpy as np
import pytest
import torch

from pitstoprl.dqn import DQNAgent, DQNConfig
from pitstoprl.env import Action, RaceConfig, RaceEnv, action_mask
from pitstoprl.policies import FixedStintPolicy, RuleBasedPolicy


class MaximumWearRandom:
    """Adversarial maximum wear on every lap; no safety cars."""

    def normal(self, mean, std):
        return mean

    def uniform(self, low, high):
        return high

    def random(self):
        return 1.0


def tiny_agent(seed=8):
    return DQNAgent(DQNConfig(hidden_n=8, batch_size=4, warmup=4, replay_capacity=16), seed)


def test_unsafe_action_rejected_before_state_or_randomness_changes():
    env = RaceEnv(seed=8)
    env.wear = 0.98
    before = copy.deepcopy(env.rng.bit_generator.state)
    assert env.action_mask().tolist() == [False, True, True, True]
    with pytest.raises(ValueError, match="100% tyre wear"):
        env.step(Action.STAY_OUT)
    assert env.wear == 0.98 and env.lap == 0 and env.pit_count == 0
    assert env.rng.bit_generator.state == before
    assert not env.history
    env.step(Action.PIT_HARD)
    assert 0 < env.wear < 1


@pytest.mark.parametrize("multiplier", [0.7, 1.3, 3.0, 25.0])
@pytest.mark.parametrize("kind", ["fixed", "wear", "exploring_dqn"])
def test_all_strategies_respect_limit_under_maximum_wear(multiplier, kind):
    env = RaceEnv(RaceConfig.for_laps(300, wear_multiplier=multiplier))
    env.rng = MaximumWearRandom()
    agent = tiny_agent()
    policy = {"fixed": FixedStintPolicy(), "wear": RuleBasedPolicy(),
              "exploring_dqn": lambda obs: agent.act(obs, epsilon=1)}[kind]
    obs = env._observation()
    for _ in range(300):
        action = policy(obs)
        assert env.action_mask()[action]
        obs, _, _, _, _ = env.step(action)
        assert 0 <= env.wear <= 1
    assert env.pit_count > 1


def test_fixed_baseline_keeps_planned_switch_and_refreshes_hard_tyres():
    env = RaceEnv(RaceConfig.for_laps(120, wear_multiplier=1.3, safety_car_probability=0))
    env.rng = MaximumWearRandom()
    policy = FixedStintPolicy()
    obs = env._observation()
    for _ in range(120):
        obs, _, _, _, _ = env.step(policy(obs))
    hard_stops = [row["lap"] for row in env.history if row["action"] == "PIT_HARD"]
    medium_stops = [row["lap"] for row in env.history if row["action"] == "PIT_MEDIUM"]
    assert hard_stops[0] == 49  # Decision after completing 40% of 120 laps.
    assert len(hard_stops) > 1
    assert medium_stops and all(lap < hard_stops[0] for lap in medium_stops)
    assert max(row["wear"] for row in env.history) <= 1


def test_safety_car_changes_feasibility_without_clipping_wear():
    env = RaceEnv()
    env.wear = 0.98
    assert not env.action_mask()[Action.STAY_OUT]
    env.safety_car_remaining = 2
    assert env.action_mask()[Action.STAY_OUT]
    env.rng = MaximumWearRandom()
    env.step(Action.STAY_OUT)
    assert env.wear == pytest.approx(0.98 + 0.039 * 0.35 * 1.07)


def test_greedy_and_exploratory_dqn_only_choose_legal_actions():
    agent = tiny_agent()
    for parameter in agent.network.parameters():
        parameter.data.zero_()
    agent.network.layers[-1].bias.data[:] = torch.tensor([100.0, 1.0, 2.0, 3.0])
    env = RaceEnv()
    env.wear = 0.98
    observations = np.repeat(env._observation()[None, :], 512, axis=0)
    assert set(agent.act_batch(observations).tolist()) == {3}
    exploratory = agent.act_batch(observations, epsilon=1)
    assert set(exploratory.tolist()) == {1, 2, 3}


def test_bootstrap_does_not_use_illegal_highest_q_action():
    agent = tiny_agent()
    for network in (agent.network, agent.target):
        for parameter in network.parameters():
            parameter.data.zero_()
    agent.network.layers[-1].bias.data[:] = torch.tensor([100.0, 2.0, 1.0, 0.0])
    agent.target.layers[-1].bias.data[:] = torch.tensor([1000.0, 3.0, 1.0, 0.0])
    env = RaceEnv()
    current = env._observation()
    env.wear = 0.98
    future = env._observation()
    assert not action_mask(future)[0]
    agent.observe(np.repeat(current[None, :], 4, axis=0), np.full(4, 3),
                  np.zeros(4), np.repeat(future[None, :], 4, axis=0), np.zeros(4))
    # Online selects legal action 1, target evaluates it at 3; Huber(0, 3)=2.5.
    assert agent.learn() == pytest.approx(2.5)


def test_old_environment_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "agent.pt"
    tiny_agent().save(path)
    payload = torch.load(path, weights_only=True)
    payload["environment_version"] = "unbounded_wear"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="retrain"):
        DQNAgent.load(path)


def test_impossible_fresh_tyre_settings_are_rejected():
    with pytest.raises(ValueError, match="fresh hard tyres"):
        RaceConfig(wear_multiplier=40)
