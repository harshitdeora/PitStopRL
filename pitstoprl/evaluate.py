"""Frozen-policy generalization benchmark with disjoint, paired test seeds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .domain import TEST_SEED, benchmark_configs, config_dict
from .dqn import DQNAgent
from .env import RaceEnv
from .policies import FixedStintPolicy, RuleBasedPolicy
from .simulation import run_episode


def evaluate(
    checkpoint: Path,
    episodes: int,
    seed: int,
    output_dir: Path,
) -> dict:
    if episodes < 2 or seed < TEST_SEED:
        raise ValueError("Use at least two episodes and fresh test seeds >= 2000000.")
    torch.set_num_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    agent = DQNAgent.load(checkpoint)
    fingerprint = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for lo, hi in [
        agent.metadata.get("training_seed_range", [0, -1]),
        agent.metadata.get("validation_seed_range", [0, -1]),
    ]:
        if seed <= hi and seed + episodes - 1 >= lo:
            raise ValueError("Evaluation overlaps checkpoint training or validation seeds.")
    policies = {
        "Double DQN": agent.act,
        "Fixed stint": FixedStintPolicy(),
        "Rule based": RuleBasedPolicy(),
    }
    rows = []
    sample_histories = {}
    summary: dict = {}
    cases = benchmark_configs()
    for scenario, config in cases.items():
        times_by_name = {}
        summary[scenario] = {}
        signatures = {}
        for name, policy in policies.items():
            subset = []
            for episode in range(episodes):
                result = run_episode(RaceEnv(config), policy, seed + episode)
                flags = tuple(lap["safety_car"] for lap in result["history"])
                if name == "Double DQN":
                    signatures[episode] = flags
                elif flags != signatures[episode]:
                    raise RuntimeError("Safety-car pairing failed.")
                row = {k: v for k, v in result.items() if k != "history"}
                row.update(scenario=scenario, strategy=name, episode=episode, seed=seed + episode)
                subset.append(row)
                rows.append(row)
                if episode == 0 and scenario == "reference_57":
                    sample_histories[name] = result["history"]
            times = np.array([r["race_time"] for r in subset])
            times_by_name[name] = times
            summary[scenario][name] = {
                "mean_seconds": round(float(times.mean()), 3),
                "std_seconds": round(float(times.std(ddof=1)), 3),
                "ci95_seconds": round(float(1.96 * times.std(ddof=1) / np.sqrt(episodes)), 3),
                "mean_pit_stops": round(float(np.mean([r["pit_stops"] for r in subset])), 3),
                "compliance_rate": round(float(np.mean([r["compliant"] for r in subset])), 4),
            }
        for name, times in times_by_name.items():
            paired_gap = times - times_by_name["Double DQN"]
            summary[scenario][name]["delta_vs_agent_seconds"] = round(float(paired_gap.mean()), 3)
            summary[scenario][name]["paired_ci95_seconds"] = round(
                float(1.96 * paired_gap.std(ddof=1) / np.sqrt(episodes)),
                3,
            )
        print(f"Evaluated {scenario}: {episodes} paired races per policy", flush=True)
    with (output_dir / "evaluation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "episodes_per_scenario_per_policy": episodes,
                "strategies": list(policies),
                "test_seed_range": [seed, seed + episodes - 1],
                "checkpoint_sha256": fingerprint,
                "checkpoint_metadata": agent.metadata,
                "scenarios": {name: config_dict(c) for name, c in cases.items()},
                "paired_confidence_intervals": "normal approximation, marginal 95%; not multiplicity corrected",
                "scope": "Synthetic simulator; no real F1 performance or universal transfer claim.",
            },
            indent=2,
        )
    )
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != fingerprint:
        raise RuntimeError("Checkpoint changed during evaluation.")
    _plot_comparison(summary["reference_57"], output_dir / "strategy_comparison.png")
    _plot_generalization(summary, output_dir / "generalization.png")
    _plot_timeline(sample_histories, output_dir / "sample_strategy.png")
    rewards_path = checkpoint.with_name(checkpoint.stem + "_training.npz")
    if rewards_path.exists():
        with np.load(rewards_path) as trace:
            _plot_training(trace["rewards"] / trace["laps"], output_dir / "training_curve.png")
    return summary


def _plot_generalization(summary: dict, path: Path) -> None:
    names = list(summary)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    gaps = [-summary[n]["Rule based"]["delta_vs_agent_seconds"] for n in names]
    ci = [summary[n]["Rule based"]["paired_ci95_seconds"] for n in names]
    ax.barh(
        names, gaps, xerr=ci, capsize=3, color=["#c44945" if gap > 0 else "#21856f" for gap in gaps]
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.invert_yaxis()
    ax.set(
        title="Frozen DQN across race conditions (paired 95% CI)",
        xlabel="DQN time minus wear-rule time (s); negative = DQN faster",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_comparison(summary: dict, path: Path) -> None:
    names = list(summary)
    means = np.array([summary[name]["mean_seconds"] for name in names])
    ci = np.array([summary[name]["paired_ci95_seconds"] for name in names])
    relative = np.array([summary[name]["delta_vs_agent_seconds"] for name in names])
    colors = ["#e10600", "#6c757d", "#1f77b4"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(names, relative, yerr=ci, capsize=5, color=colors)
    ax.set_ylabel("Mean race-time gap vs Double DQN (s)")
    ax.set_title("Reference race: paired stochastic evaluation (95% CI)")
    ax.grid(axis="y", alpha=0.25)
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(ci) + 0.4,
            f"{mean:.1f}s",
            ha="center",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_timeline(histories: dict, path: Path) -> None:
    color = {"SOFT": "#e10600", "MEDIUM": "#f4c430", "HARD": "#e8e8e8"}
    fig, axes = plt.subplots(len(histories), 1, figsize=(8.0, 4.8), sharex=True)
    for ax, (name, history) in zip(axes, histories.items()):
        for item in history:
            edge = "#111" if item["pitted"] else "none"
            ax.barh(0, 1, left=item["lap"] - 1, color=color[item["compound"]], edgecolor=edge)
            if item["safety_car"]:
                ax.axvspan(item["lap"] - 1, item["lap"], color="#2ca02c", alpha=0.28)
        ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center")
    axes[-1].set_xlabel("Lap (green shading = safety car; black edge = pit lap)")
    fig.suptitle("Example strategy decisions on the same random race", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_training(rewards: np.ndarray, path: Path) -> None:
    window = min(len(rewards), min(1000, max(10, len(rewards) // 100)))
    rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
    x = np.arange(window, len(rewards) + 1)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(x, rolling, color="#e10600", linewidth=1.5)
    ax.set(
        title=f"Training reward ({window}-episode moving average)",
        xlabel="Episode",
        ylabel="Mean reward per lap (randomized settings)",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen DQN on fresh seeds and settings.")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/dqn.pt"))
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=TEST_SEED)
    parser.add_argument("--output-dir", type=Path, default=Path("results/dqn"))
    args = parser.parse_args()
    summary = evaluate(args.checkpoint, args.episodes, args.seed, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
