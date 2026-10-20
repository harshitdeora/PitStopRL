"""Run one human-readable race using the trained agent."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .dqn import DQNAgent
from .env import RaceConfig, RaceEnv
from .simulation import run_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one PitStopRL race.")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/dqn.pt"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--laps", type=int, default=57)
    parser.add_argument("--wear-multiplier", type=float, default=1.0)
    parser.add_argument("--pit-loss", type=float, default=22.0)
    parser.add_argument("--sc-probability", type=float, default=0.035)
    args = parser.parse_args()
    torch.set_num_threads(1)
    agent = DQNAgent.load(args.checkpoint)
    config = RaceConfig.for_laps(
        args.laps,
        wear_multiplier=args.wear_multiplier,
        pit_loss_seconds=args.pit_loss,
        safety_car_probability=args.sc_probability,
    )
    result = run_episode(RaceEnv(config), agent.act, args.seed)
    print(
        f"Laps: {args.laps} | race time: {result['race_time']:.2f} s | "
        f"stops: {result['pit_stops']} | compound rule: {result['compliant']}"
    )
    for row in result["history"]:
        if row["pitted"] or row["safety_car"]:
            marker = "PIT" if row["pitted"] else "SC"
            print(
                f"Lap {row['lap']:02d}  {marker:3s}  {row['compound']:6s}  wear={row['wear']:.2f}"
            )


if __name__ == "__main__":
    main()
