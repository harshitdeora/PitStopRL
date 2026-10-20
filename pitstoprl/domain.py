"""Predeclared training domain and disjoint evaluation protocol."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .env import RaceConfig

HELD_OUT_LAPS = (53, 66, 80)
TRAINING_LAPS = tuple(n for n in range(20, 91) if n not in HELD_OUT_LAPS)
VALIDATION_SEED = 1_000_000
TEST_SEED = 2_000_000
TRAINING_DOMAIN = {
    "laps": list(TRAINING_LAPS),
    "held_out_laps": list(HELD_OUT_LAPS),
    "wear_multiplier": [0.70, 1.30],
    "pit_loss_seconds": [12.0, 32.0],
    "safety_car_probability": [0.0, 0.07],
    "safety_car_pit_factor": [0.45, 0.70],
    "base_lap_time": [75.0, 100.0],
    "lap_noise_std": [0.10, 0.40],
    "fuel": "1.8 kg per configured lap; no refuelling",
}


def sample_config(rng: np.random.Generator) -> RaceConfig:
    return RaceConfig.for_laps(
        int(rng.choice(TRAINING_LAPS)),
        wear_multiplier=float(rng.uniform(0.70, 1.30)),
        pit_loss_seconds=float(rng.uniform(12.0, 32.0)),
        safety_car_probability=float(rng.uniform(0.0, 0.07)),
        safety_car_pit_factor=float(rng.uniform(0.45, 0.70)),
        base_lap_time=float(rng.uniform(75.0, 100.0)),
        lap_noise_std=float(rng.uniform(0.10, 0.40)),
    )


def benchmark_configs() -> dict[str, RaceConfig]:
    """Declare tests before training; never select checkpoints using these."""
    cases = {"reference_57": RaceConfig()}
    cases.update({f"laps_{n}": RaceConfig.for_laps(n) for n in (30, 53, 66, 80, 90)})
    cases.update(
        {
            "wear_low": RaceConfig(wear_multiplier=0.70),
            "wear_high": RaceConfig(wear_multiplier=1.30),
            "pit_cheap": RaceConfig(pit_loss_seconds=12.0),
            "pit_expensive": RaceConfig(pit_loss_seconds=32.0),
            "no_safety_cars": RaceConfig(safety_car_probability=0.0),
            "more_safety_cars": RaceConfig(safety_car_probability=0.07),
            "held_out_combination": RaceConfig.for_laps(
                66,
                wear_multiplier=1.20,
                pit_loss_seconds=29.0,
                safety_car_probability=0.06,
                base_lap_time=82.0,
            ),
        }
    )
    return cases


def config_dict(config: RaceConfig) -> dict:
    return asdict(config)
