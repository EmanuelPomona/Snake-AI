import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import DEFAULT_DASHBOARD_PATH, DEFAULT_LOG_DIR, EXPERIMENT_NAME


CHARTS = [
    ("score", "Score by Episode"),
    ("average_score_100", "Average Score (Last 100)"),
    ("best_score", "Best Score"),
    ("reward_total", "Total Reward"),
    ("average_loss", "Loss"),
    ("epsilon", "Epsilon"),
    ("memory_size", "Memory Size"),
]


def plot_training_log(log_path, save_path=None):
    csv_path = Path(log_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"No training log found at {csv_path}")

    data = pd.read_csv(csv_path)
    if data.empty:
        raise ValueError(f"Training log is empty: {csv_path}")

    latest = data.iloc[-1]
    recent = data.tail(250)

    fig, axes = plt.subplots(4, 2, figsize=(16, 14), constrained_layout=True)
    axes = axes.flatten()

    for axis, (column, title) in zip(axes, CHARTS):
        if column not in data:
            axis.set_title(f"{title} unavailable")
            axis.axis("off")
            continue

        axis.plot(data["episode"], data[column])
        axis.set_title(title)
        axis.set_xlabel("Episode")
        axis.grid(True, alpha=0.3)

    recent_axis = axes[len(CHARTS)]
    recent_axis.plot(recent["episode"], recent["score"], label="Score", alpha=0.7)
    recent_axis.plot(
        recent["episode"],
        recent["average_score_100"],
        label="Average Score 100",
        linewidth=2,
    )
    recent_axis.set_title("Recent Episodes")
    recent_axis.set_xlabel("Episode")
    recent_axis.grid(True, alpha=0.3)
    recent_axis.legend()

    fig.suptitle(
        "Snake DQN Training\n"
        f"Episode {int(latest['episode'])} | "
        f"Score {int(latest['score'])} | "
        f"Avg100 {latest['average_score_100']:.2f} | "
        f"Best {int(latest['best_score'])} | "
        f"Epsilon {latest['epsilon']:.3f}",
        fontsize=16,
    )

    if save_path:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Saved dashboard to {output_path}")
    else:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Plot Snake DQN training logs for {EXPERIMENT_NAME}."
    )
    parser.add_argument("--log", default=f"{DEFAULT_LOG_DIR}/training.csv")
    parser.add_argument("--save", default=DEFAULT_DASHBOARD_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    plot_training_log(args.log, args.save)


if __name__ == "__main__":
    main()
