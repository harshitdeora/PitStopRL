import numpy as np
import pytest

from pitstoprl.domain import HELD_OUT_LAPS, sample_config
from pitstoprl.env import Action, RaceConfig, RaceEnv
from pitstoprl.policies import FixedStintPolicy


def test_reset_is_reproducible():
    env = RaceEnv(seed=3)
    first = env.reset(seed=42)
    trajectory_a = [env.step(Action.STAY_OUT)[1] for _ in range(4)]
    second = env.reset(seed=42)
    trajectory_b = [env.step(Action.STAY_OUT)[1] for _ in range(4)]
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(trajectory_a, trajectory_b)


def test_episode_has_configured_horizon():
    env = RaceEnv(RaceConfig(laps=5), seed=0)
    env.reset(seed=0)
    for index in range(5):
        _, _, done, truncated, _ = env.step(Action.PIT_HARD if index == 2 else Action.STAY_OUT)
        assert done is (index == 4)
        assert not truncated
    with pytest.raises(RuntimeError):
        env.step(Action.STAY_OUT)


def test_safety_car_reduces_pit_loss():
    env = RaceEnv(seed=1)
    env.reset(seed=1)
    env.safety_car_remaining = 2
    _, _, _, _, info = env.step(Action.PIT_HARD)
    assert info["pit_loss"] == pytest.approx(
        env.config.pit_loss_seconds * env.config.safety_car_pit_factor
    )


def test_terminal_penalty_enforces_compound_change():
    env = RaceEnv(RaceConfig(laps=1), seed=2)
    env.reset(seed=2)
    _, _, done, _, info = env.step(Action.STAY_OUT)
    assert done
    assert info["rule_penalty"] == env.config.terminal_rule_penalty


@pytest.mark.parametrize("laps", [1, 15, 30, 53, 66, 80, 90, 120])
def test_configurable_horizon_fuel_and_baseline_trigger(laps):
    env = RaceEnv(RaceConfig.for_laps(laps), seed=4)
    obs = env.reset(seed=4)
    rule = FixedStintPolicy()
    reward_sum = 0.0
    for step in range(laps):
        assert obs.shape == (RaceEnv.observation_n,)
        assert round(float(obs[8]) * 100) == laps - step
        assert round(float(obs[9]) * 100) == laps
        obs, reward, done, _, _ = env.step(rule(obs))
        reward_sum += reward
        assert done == (step == laps - 1)
    assert env.pit_count >= 1
    assert env.changed_compound
    assert max(row["wear"] for row in env.history) <= 1
    assert env.fuel == pytest.approx(0, abs=1e-8)
    assert reward_sum == pytest.approx(laps - env.total_time / 100)


def test_context_distinguishes_identical_progress_in_different_races():
    short = RaceEnv(RaceConfig.for_laps(30, pit_loss_seconds=12.0, wear_multiplier=0.7))
    long = RaceEnv(RaceConfig.for_laps(90, pit_loss_seconds=32.0, wear_multiplier=1.3))
    short.lap, long.lap = 15, 45
    a, b = short._observation(), long._observation()
    assert a[0] == b[0]
    assert a[8] != b[8]
    assert a[10] != b[10]
    assert a[11] != b[11]


@pytest.mark.parametrize(
    "settings",
    [
        {"laps": 0},
        {"laps": 2.5},
        {"laps": 90},
        {"wear_multiplier": -1},
        {"safety_car_probability": 1.2},
        {"safety_car_duration_max": 1},
        {"pit_loss_seconds": float("nan")},
    ],
)
def test_invalid_race_settings_fail_early(settings):
    with pytest.raises(ValueError):
        RaceConfig(**settings)


def test_training_domain_is_seeded_and_excludes_held_out_lengths():
    a, b = np.random.default_rng(8), np.random.default_rng(8)
    for _ in range(200):
        config = sample_config(a)
        assert config == sample_config(b)
        assert config.laps not in HELD_OUT_LAPS
        assert config.initial_fuel_kg == pytest.approx(1.8 * config.laps)


@pytest.mark.parametrize("action", [-1, 4, 1.5, True, None])
def test_invalid_actions_are_rejected(action):
    with pytest.raises(ValueError):
        RaceEnv().step(action)
