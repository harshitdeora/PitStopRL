"""Evaluation-only baseline strategies."""

from __future__ import annotations

import numpy as np

from .env import Action, Compound, action_mask


class FixedStintPolicy:
    """Medium-to-hard at 40% distance, with additional tyre-safety stops."""

    def __call__(self, obs: np.ndarray) -> int:
        allowed = action_mask(obs)
        current = Compound(round(float(obs[1]) * 2))
        total = round(float(obs[9]) * 100)
        remaining = round(float(obs[8]) * 100)
        trigger = min(max(int(total * 0.4), 0), total - 1)
        hard_phase = total - remaining >= trigger
        target = Compound.HARD if hard_phase else Compound.MEDIUM
        if (hard_phase and current != Compound.HARD) or not allowed[Action.STAY_OUT]:
            preferred = int(target) + 1
            return preferred if allowed[preferred] else int(Action.PIT_HARD)
        return int(Action.STAY_OUT)


class RuleBasedPolicy:
    """Simple wear-threshold rule; it does not reason about safety cars."""

    def __call__(self, obs: np.ndarray) -> int:
        compound_scaled, age_scaled, wear, changed = obs[1], obs[2], obs[3], obs[7]
        age = float(age_scaled) * 30
        remaining = round(float(obs[8]) * 100)
        current = Compound(round(float(compound_scaled) * 2))
        allowed = action_mask(obs)
        should_pit = wear > np.float32(0.80) or not allowed[Action.STAY_OUT]
        if not should_pit and remaining < 14 and changed < 0.5 and age >= 10:
            should_pit = True
        if not should_pit:
            return int(Action.STAY_OUT)
        if remaining <= 13:
            target = Compound.SOFT
        elif remaining <= 27:
            target = Compound.MEDIUM
        else:
            target = Compound.HARD
        if target == current:
            target = Compound.HARD if current != Compound.HARD else Compound.MEDIUM
        preferred = int(target) + 1
        return preferred if allowed[preferred] else int(Action.PIT_HARD)
