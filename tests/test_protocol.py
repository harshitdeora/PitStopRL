from pathlib import Path

import numpy as np
import pytest

from pitstoprl.domain import HELD_OUT_LAPS, TEST_SEED, TRAINING_LAPS
from pitstoprl.dqn import DQNAgent, DQNConfig
from pitstoprl.env import Action, RaceConfig, RaceEnv
from pitstoprl.evaluate import evaluate
from pitstoprl.training import train_dqn


def test_centered_reward_preserves_paired_total_time_objective():
    def rollout(pit_every):
        env = RaceEnv(RaceConfig.for_laps(30), seed=99)
        centered_return = uncontrollable = raw_return = 0.0
        for i in range(30):
            action = Action.PIT_HARD if i % pit_every == 0 else Action.STAY_OUT
            _, reward, _, _, info = env.step(action)
            centered_return -= info["controllable_cost"] / 10
            raw_return += reward
            uncontrollable += info["uncontrollable_time"]
        assert env.total_time == pytest.approx(uncontrollable - 10 * centered_return)
        assert raw_return == pytest.approx(30 - env.total_time / 100)
        return env.total_time, centered_return, uncontrollable

    a, b = rollout(7), rollout(20)
    assert a[2] == pytest.approx(b[2])
    assert a[0] - b[0] == pytest.approx(-10 * (a[1] - b[1]))


def test_held_out_lengths_are_not_training_lengths():
    assert not set(HELD_OUT_LAPS) & set(TRAINING_LAPS)


def test_evaluation_rejects_overlapping_seeds_before_writing(tmp_path: Path):
    with pytest.raises(ValueError, match="fresh test seeds"):
        evaluate(tmp_path / "missing.pt", 2, 10000, tmp_path / "results")
    assert not (tmp_path / "results").exists()


def test_evaluation_exports_paired_metrics_without_learning(tmp_path: Path, monkeypatch):
    config = DQNConfig(hidden_n=8, batch_size=4, warmup=4, replay_capacity=8)
    agent = DQNAgent(config, seed=8)
    checkpoint = tmp_path / "agent.pt"
    agent.save(checkpoint, {"training_seed_range": [7, 20]})
    original = checkpoint.read_bytes()
    # Small integration benchmark exercises real output and plotting paths.
    monkeypatch.setattr(
        "pitstoprl.evaluate.benchmark_configs", lambda: {"reference_57": RaceConfig.for_laps(3)}
    )
    summary = evaluate(checkpoint, 2, TEST_SEED, tmp_path / "results")
    assert set(summary["reference_57"]) == {"Double DQN", "Planned + safety", "Rule based"}
    assert (tmp_path / "results/protocol.json").exists()
    assert (tmp_path / "results/evaluation.csv").exists()
    assert checkpoint.read_bytes() == original


def test_small_training_run_saves_selected_model_and_full_trace(tmp_path: Path):
    config = DQNConfig(hidden_n=8, batch_size=4, warmup=4, replay_capacity=64)
    checkpoint = tmp_path / "tiny.pt"
    result = train_dqn(checkpoint, episodes=3, seed=8, env_count=2, validate_every=2, config=config)
    loaded = DQNAgent.load(checkpoint)
    assert loaded.metadata["training_seed_range"] == [8, 10]
    assert loaded.metadata["validation_seed_range"][0] == 1_000_000
    assert result["selection_uses_test_data"] is False
    assert result["environment_version"] == "wear_limit_v1"
    assert result["max_training_tyre_wear"] <= 1
    assert result["wear_limit_violations"] == 0
    with np.load(tmp_path / "tiny_training.npz") as trace:
        assert len(trace["rewards"]) == 3
        assert np.isfinite(trace["rewards"]).all()
    assert not checkpoint.with_name("tiny.pt.tmp").exists()
