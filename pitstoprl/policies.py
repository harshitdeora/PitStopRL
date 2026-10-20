"""Evaluation-only baseline strategies."""

from __future__ import annotations

import numpy as np

from .env import Action, Compound


class FixedStintPolicy:
    """Evaluation-only medium-to-hard stop after 40% of the race."""

    def __call__(self, obs: np.ndarray) -> int:
        changed = bool(obs[7] >= 0.5)
        total = round(float(obs[9]) * 100)
        remaining = round(float(obs[8]) * 100)
        trigger = min(max(int(total * 0.4), 0), total - 1)
        should_pit = total - remaining >= trigger and not changed
        return int(Action.PIT_HARD if should_pit else Action.STAY_OUT)


class RuleBasedPolicy:
    """Simple wear-threshold rule; it does not reason about safety cars."""

    def __call__(self, obs: np.ndarray) -> int:
        compound_scaled, age_scaled, wear, changed = obs[1], obs[2], obs[3], obs[7]
        age = float(age_scaled) * 30
        remaining = round(float(obs[8]) * 100)
        current = Compound(round(float(compound_scaled) * 2))
        should_pit = wear > 0.80
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
        return int(target) + 1
