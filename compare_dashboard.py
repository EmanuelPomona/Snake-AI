import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_run(label, path):
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {label} training log: {csv_path}")

    data = pd.read_csv(csv_path)
    if data.empty:
        raise ValueError(f"{label} training log is empty: {csv_path}")

    data = data.copy()
    data["run"] = label
    return data


def plot_comparison(v1_log, v2_log, save_path):
    runs = {
        "v1_11_features": load_run("v1_11_features", v1_log),
        "v2_22_features": load_run("v2_22_features", v2_log),
    }

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)
    axes = axes.flatten()

    plot_specs = [
        ("score", "Score by Episode"),
        ("average_score_100", "Average Score 100"),
        ("best_score", "Best Score"),
        ("reward_total", "Total Reward"),
        ("average_loss", "Average Loss"),
        ("memory_size", "Replay Memory Size"),
    ]

    for axis, (column, title) in zip(axes, plot_specs):
        for label, data in runs.items():
            if column not in data:
                continue

            axis.plot(data["episode"], data[column], label=label, alpha=0.85)

        axis.set_title(title)
        axis.set_xlabel("Episode")
        axis.grid(True, alpha=0.3)
        axis.legend()

    v1_latest = runs["v1_11_features"].iloc[-1]
    v2_latest = runs["v2_22_features"].iloc[-1]

    fig.suptitle(
        "Snake DQN Comparison\n"
        f"v1: episode {int(v1_latest['episode'])}, "
        f"avg100 {v1_latest['average_score_100']:.2f}, "
        f"best {int(v1_latest['best_score'])} | "
        f"v2: episode {int(v2_latest['episode'])}, "
        f"avg100 {v2_latest['average_score_100']:.2f}, "
        f"best {int(v2_latest['best_score'])}",
        fontsize=16,
    )

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved comparison dashboard to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Snake DQN v1 and v2 runs.")
    parser.add_argument("--v1-log", default="logs/training.csv")
    parser.add_argument("--v2-log", default="logs/state_v2/training.csv")
    parser.add_argument("--save", default="logs/comparison_dashboard.png")
    return parser.parse_args()


def main():
    args = parse_args()
    plot_comparison(args.v1_log, args.v2_log, args.save)


if __name__ == "__main__":
    main()
