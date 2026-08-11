import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from config import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_EVAL_EPISODES,
    DEFAULT_LOG_DIR,
    SEED,
    TRAINING_SPEED,
)
from train import (
    default_eval_checkpoint,
    evaluate_dqn,
    evaluate_random,
    print_evaluation_report,
    set_seed,
)


BENCHMARK_FIELDS = [
    "timestamp_utc",
    "agent",
    "episodes",
    "checkpoint",
    "average_score",
    "median_score",
    "best_score",
    "average_episode_length",
]


def append_benchmark_row(path, row):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()

    with output_path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=BENCHMARK_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_row(agent_name, result, checkpoint):
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "agent": agent_name,
        "episodes": result["episodes"],
        "checkpoint": str(checkpoint) if checkpoint else "",
        "average_score": f"{result['average_score']:.4f}",
        "median_score": f"{result['median_score']:.4f}",
        "best_score": result["best_score"],
        "average_episode_length": f"{result['average_episode_length']:.4f}",
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Snake agents and optionally save comparable metrics."
    )
    parser.add_argument("--agent", choices=["random", "dqn", "both"], default="both")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output", default=f"{DEFAULT_LOG_DIR}/benchmark.csv")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--speed", type=int, default=TRAINING_SPEED)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print results without appending them to the benchmark CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    args.eval_episodes = args.episodes

    rows = []
    checkpoint_path = None

    if args.agent in ("random", "both"):
        random_result = evaluate_random(args)
        print_evaluation_report("Random Agent", random_result)
        rows.append(build_row("random", random_result, None))

    if args.agent in ("dqn", "both"):
        checkpoint_path = (
            Path(args.checkpoint)
            if args.checkpoint
            else default_eval_checkpoint(args.checkpoint_dir)
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No DQN checkpoint found at {checkpoint_path}. Train first or pass --checkpoint."
            )

        dqn_result = evaluate_dqn(args, checkpoint_path)
        print_evaluation_report("DQN Agent", dqn_result)
        rows.append(build_row("dqn", dqn_result, checkpoint_path))

    if not args.no_save:
        for row in rows:
            append_benchmark_row(args.output, row)
        print(f"Saved benchmark rows to {args.output}")


if __name__ == "__main__":
    main()
