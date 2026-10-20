from pathlib import Path

import numpy as np
import pytest
import torch

from pitstoprl.dqn import DQNAgent, DQNConfig, ReplayBuffer
from pitstoprl.env import RaceEnv


def tiny_config(**kwargs):
    return DQNConfig(hidden_n=16, batch_size=4, warmup=4, replay_capacity=8, **kwargs)


def test_checkpoint_preserves_all_predictions_and_schema(tmp_path: Path):
    agent = DQNAgent(tiny_config(), seed=10)
    observations = np.stack([RaceEnv(seed=n).reset(n) for n in range(3)])
    before = agent.network(torch.from_numpy(observations)).detach().numpy()
    path = tmp_path / "agent.pt"
    agent.save(path, {"test": "round trip"})
    loaded = DQNAgent.load(path)
    after = loaded.network(torch.from_numpy(observations)).detach().numpy()
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(agent.act_batch(observations), loaded.act_batch(observations))
    assert loaded.replay is None
    assert loaded.metadata == {"test": "round trip"}
    with pytest.raises(ValueError, match="dimensions"):
        loaded.act(np.zeros(8))


def test_terminal_update_ignores_next_state_and_target_stays_frozen():
    agent = DQNAgent(tiny_config(target_interval=2), seed=2)
    for net in (agent.network, agent.target):
        for p in net.parameters():
            p.data.zero_()
    # Huge bootstrap would dominate if terminal masking were missing.
    agent.target.layers[-1].bias.data.fill_(100.0)
    states = np.zeros((4, RaceEnv.observation_n), dtype=np.float32)
    actions = np.ones(4, dtype=np.int64)
    agent.observe(states, actions, np.ones(4), states, np.ones(4))
    frozen = {k: v.clone() for k, v in agent.target.state_dict().items()}
    assert agent.learn() == pytest.approx(0.5)
    assert all(torch.equal(v, frozen[k]) for k, v in agent.target.state_dict().items())
    assert agent.network.layers[-1].bias[1] > 0
    agent.learn()
    assert all(
        torch.equal(v, agent.network.state_dict()[k]) for k, v in agent.target.state_dict().items()
    )


def test_double_dqn_selects_online_action_but_evaluates_target():
    agent = DQNAgent(tiny_config(), seed=2)
    for net in (agent.network, agent.target):
        for p in net.parameters():
            p.data.zero_()
    agent.network.layers[-1].bias.data[:] = torch.tensor([0.0, 2.0, 0.0, 0.0])
    agent.target.layers[-1].bias.data[:] = torch.tensor([100.0, 3.0, 0.0, 0.0])
    states = np.zeros((4, RaceEnv.observation_n), dtype=np.float32)
    agent.observe(states, np.zeros(4, dtype=int), np.zeros(4), states, np.zeros(4))
    # target = 3, prediction = 0; Huber = 3 - .5, not 100 - .5.
    assert agent.learn() == pytest.approx(2.5)


def test_replay_wraps_without_mixing_transitions():
    replay = ReplayBuffer(5, 2, seed=1)
    for start in (0, 3):
        ids = np.arange(start, start + 3)
        states = np.stack([ids, ids], axis=1).astype(np.float32)
        replay.add(states, ids, ids + 10, states + 1, np.zeros(3))
    states, actions, rewards, next_states, _ = replay.sample(5)
    assert set(actions.tolist()) == {1, 2, 3, 4, 5}
    np.testing.assert_array_equal(states[:, 0], actions)
    np.testing.assert_array_equal(rewards, actions + 10)
    np.testing.assert_array_equal(next_states, states + 1)


def test_seeded_initialization_and_inference_do_not_change_weights():
    a, b = DQNAgent(tiny_config(), seed=9), DQNAgent(tiny_config(), seed=9)
    before = {k: v.clone() for k, v in a.network.state_dict().items()}
    a.act(RaceEnv().reset())
    for key, value in a.network.state_dict().items():
        assert torch.equal(value, b.network.state_dict()[key])
        assert torch.equal(value, before[key])
