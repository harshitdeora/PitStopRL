"""A small, reproducible F1-style race strategy environment.

The environment intentionally exposes a minimal Gym-like API without requiring
Gymnasium. One step is one completed lap. The agent observes the race before
choosing whether to stay out or pit for one of three dry compounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Compound(IntEnum):
    SOFT = 0
    MEDIUM = 1
    HARD = 2


class Action(IntEnum):
    STAY_OUT = 0
    PIT_SOFT = 1
    PIT_MEDIUM = 2
    PIT_HARD = 3


@dataclass(frozen=True)
class RaceConfig:
    laps: int = 57
    base_lap_time: float = 88.0
    initial_fuel_kg: float = 102.6
    fuel_burn_kg: float = 1.8
    fuel_penalty_per_kg: float = 0.030
    pit_loss_seconds: float = 22.0
    safety_car_pit_factor: float = 0.55
    safety_car_probability: float = 0.035
    safety_car_duration_min: int = 2
    safety_car_duration_max: int = 4
    safety_car_lap_delta: float = 25.0
    lap_noise_std: float = 0.18
    terminal_rule_penalty: float = 30.0
    reward_reference_seconds: float = 100.0
    reward_scale: float = 100.0
    wear_multiplier: float = 1.0

    def __post_init__(self) -> None:
        values = vars(self)
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("Race settings must be finite.")
        if not isinstance(self.laps, int) or isinstance(self.laps, bool) or self.laps < 1:
            raise ValueError("laps must be a positive integer.")
        if self.initial_fuel_kg <= 0 or self.reward_scale <= 0 or self.base_lap_time <= 0:
            raise ValueError("Initial fuel, base lap time and reward scale must be positive.")
        if not 0 <= self.safety_car_probability <= 1:
            raise ValueError("Safety-car probability must be in [0, 1].")
        if not 0 <= self.safety_car_pit_factor <= 1:
            raise ValueError("Safety-car pit factor must be in [0, 1].")
        durations = (self.safety_car_duration_min, self.safety_car_duration_max)
        if any(not isinstance(v, int) or isinstance(v, bool) for v in durations):
            raise ValueError("Safety-car durations must be integers.")
        if not 1 <= durations[0] <= durations[1]:
            raise ValueError("Invalid safety-car duration range.")
        nonnegative = (
            "fuel_burn_kg",
            "fuel_penalty_per_kg",
            "pit_loss_seconds",
            "safety_car_lap_delta",
            "lap_noise_std",
            "terminal_rule_penalty",
        )
        if any(values[key] < 0 for key in nonnegative) or self.wear_multiplier <= 0:
            raise ValueError("Costs must be nonnegative and wear multiplier positive.")
        if self.initial_fuel_kg + 1e-8 < self.laps * self.fuel_burn_kg:
            raise ValueError("Starting fuel must cover the configured race distance.")

    @classmethod
    def for_laps(cls, laps: int, **settings) -> RaceConfig:
        """Scale starting fuel with race length unless explicitly supplied."""
        burn = settings.get("fuel_burn_kg", 1.8)
        settings.setdefault("initial_fuel_kg", laps * burn)
        return cls(laps=laps, **settings)


class RaceEnv:
    """Finite-horizon stochastic race environment.

    Observation: 27 features, including race length and environment context.
    See OBSERVATION_NAMES for the ordered feature schema and units.
    The action space is discrete with four choices from :class:`Action`.
    """

    wear_rates = np.array([0.058, 0.039, 0.029], dtype=float)
    grip_offsets = np.array([-0.50, 0.00, 0.48], dtype=float)
    OBSERVATION_NAMES = (
        "lap_progress",
        "compound_over_2",
        "tyre_age_over_30",
        "wear",
        "fuel_fraction",
        "safety_car_active",
        "sc_laps_over_4",
        "changed_compound",
        "laps_remaining_over_100",
        "total_laps_over_100",
        "pit_loss_over_40",
        "wear_multiplier",
        "sc_probability_over_0_1",
        "sc_pit_factor",
        "base_lap_time_over_100",
        "wear_squared",
        "wear_cubed",
        "initial_fuel_over_180",
        "fuel_burn_over_3",
        "fuel_penalty_over_0_05",
        "sc_lap_delta_over_40",
        "lap_noise_std",
        "rule_penalty_over_100",
        "reward_scale_over_100",
        "sc_duration_min_over_4",
        "sc_duration_max_over_4",
        "reward_reference_over_100",
    )
    observation_n = len(OBSERVATION_NAMES)

    def __init__(self, config: RaceConfig | None = None, seed: int | None = None):
        self.config = config or RaceConfig()
        self.rng = np.random.default_rng(seed)
        self.action_n = len(Action)
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.lap = 0
        self.compound = Compound.MEDIUM
        self.tyre_age = 0
        self.wear = 0.0
        self.fuel = self.config.initial_fuel_kg
        self.safety_car_remaining = 0
        self.changed_compound = False
        self.pit_count = 0
        self.total_time = 0.0
        self.history: list[dict[str, float | int | bool | str]] = []
        return self._observation()

    @property
    def safety_car(self) -> bool:
        return self.safety_car_remaining > 0

    def _observation(self) -> np.ndarray:
        c = self.config
        return np.array(
            [
                self.lap / c.laps,
                int(self.compound) / 2.0,
                self.tyre_age / 30.0,
                self.wear,
                max(self.fuel / c.initial_fuel_kg, 0.0),
                float(self.safety_car),
                self.safety_car_remaining / 4.0,
                float(self.changed_compound),
                (c.laps - self.lap) / 100.0,
                c.laps / 100.0,
                c.pit_loss_seconds / 40.0,
                c.wear_multiplier,
                c.safety_car_probability / 0.1,
                c.safety_car_pit_factor,
                c.base_lap_time / 100.0,
                self.wear**2,
                self.wear**3,
                c.initial_fuel_kg / 180.0,
                c.fuel_burn_kg / 3.0,
                c.fuel_penalty_per_kg / 0.05,
                c.safety_car_lap_delta / 40.0,
                c.lap_noise_std,
                c.terminal_rule_penalty / 100.0,
                c.reward_scale / 100.0,
                c.safety_car_duration_min / 4.0,
                c.safety_car_duration_max / 4.0,
                c.reward_reference_seconds / 100.0,
            ],
            dtype=np.float32,
        )

    def _maybe_start_safety_car(self) -> None:
        if (
            self.safety_car_remaining == 0
            and self.lap < self.config.laps - 2
            and self.rng.random() < self.config.safety_car_probability
        ):
            self.safety_car_remaining = int(
                self.rng.integers(
                    self.config.safety_car_duration_min,
                    self.config.safety_car_duration_max + 1,
                )
            )

    def step(self, action: int | Action):
        if self.lap >= self.config.laps:
            raise RuntimeError("Episode is complete; call reset().")
        try:
            if isinstance(action, (bool, np.bool_)) or int(action) != action:
                raise ValueError("Action must be an integer.")
            action = Action(int(action))
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"Action must be in [0, {self.action_n - 1}].") from exc

        c = self.config
        was_safety_car = self.safety_car
        pit_loss = 0.0
        pitted = action != Action.STAY_OUT
        if pitted:
            new_compound = Compound(int(action) - 1)
            self.changed_compound = self.changed_compound or new_compound != self.compound
            self.compound = new_compound
            self.tyre_age = 0
            self.wear = 0.0
            self.pit_count += 1
            factor = c.safety_car_pit_factor if was_safety_car else 1.0
            pit_loss = c.pit_loss_seconds * factor

        fuel_delta = c.fuel_penalty_per_kg * self.fuel
        tyre_delta = float(self.grip_offsets[int(self.compound)])
        tyre_delta += 1.8 * self.wear + 11.0 * self.wear**3
        noise = float(self.rng.normal(0.0, c.lap_noise_std))
        sc_delta = c.safety_car_lap_delta if was_safety_car else 0.0
        lap_time = c.base_lap_time + fuel_delta + tyre_delta + noise + sc_delta + pit_loss

        wear_multiplier = 0.35 if was_safety_car else 1.0
        wear_noise = float(self.rng.uniform(0.93, 1.07))
        self.wear += (
            float(self.wear_rates[int(self.compound)])
            * c.wear_multiplier
            * wear_multiplier
            * wear_noise
        )
        self.tyre_age += 1
        self.fuel = max(0.0, self.fuel - c.fuel_burn_kg)
        self.total_time += lap_time
        self.lap += 1

        if self.safety_car_remaining > 0:
            self.safety_car_remaining -= 1
        self._maybe_start_safety_car()

        terminated = self.lap >= c.laps
        rule_penalty = c.terminal_rule_penalty if terminated and not self.changed_compound else 0.0
        if rule_penalty:
            self.total_time += rule_penalty
        # With a fixed race horizon, this constant-shifted time reward has the
        # same optimum as minimizing total race time while learning much faster.
        reward = (c.reward_reference_seconds - lap_time - rule_penalty) / c.reward_scale
        record = {
            "lap": self.lap,
            "action": action.name,
            "compound": self.compound.name,
            "pitted": pitted,
            "safety_car": was_safety_car,
            "wear": self.wear,
            "lap_time": lap_time,
        }
        self.history.append(record)
        info = {
            "lap_time": lap_time,
            "pit_loss": pit_loss,
            "rule_penalty": rule_penalty,
            "total_time": self.total_time,
            # The remainder of lap time is independent of the policy under
            # paired exogenous randomness. Useful for variance-reduced RL reward.
            "controllable_cost": tyre_delta + pit_loss + rule_penalty,
            "uncontrollable_time": c.base_lap_time + fuel_delta + noise + sc_delta,
            "history": self.history if terminated else None,
        }
        return self._observation(), reward, terminated, False, info
