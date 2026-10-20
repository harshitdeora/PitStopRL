"""Train contextual Double DQN across randomized race settings."""

from __future__ import annotations

import argparse
from pathlib import Path

from .training import train_dqn


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL-only pit strategy.")
    parser.add_argument("--episodes", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--env-count", type=int, default=16)
    parser.add_argument("--validate-every", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("artifacts/dqn.pt"))
    args = parser.parse_args()
    train_dqn(args.output, args.episodes, args.seed, args.env_count, args.validate_every)


if __name__ == "__main__":
    main()
