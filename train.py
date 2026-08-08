import argparse
import csv
import random
import statistics
from pathlib import Path

import numpy as np
import torch

from agent import DQNAgent, RandomAgent
from config import (
    CHECKPOINT_FREQUENCY,
    DEFAULT_CHECKPOINT_DIR,
    DECISION_LOG_EVERY_N_EPISODES,
    DEFAULT_EVAL_EPISODES,
    DEFAULT_LOG_DIR,
    DEFAULT_RENDER,
    DEFAULT_TRAIN_EPISODES,
    EXPERIMENT_NAME,
    SEED,
    STATE_SIZE,
    TARGET_UPDATE_FREQUENCY,
    TRAINING_SPEED,
)
from snake_game import SnakeGameAI


TRAINING_FIELDS = [
    "episode",
    "score",
    "reward_total",
    "steps",
    "epsilon",
    "loss",
    "average_loss",
    "memory_size",
    "best_score",
    "average_score_100",
    "straight_actions",
    "right_actions",
    "left_actions",
    "random_actions",
    "model_actions",
    "food_eaten",
    "death_reason",
    "mean_q_value",
    "max_q_value",
]

DECISION_FIELDS = [
    "episode",
    "step",
    "state",
    "q_straight",
    "q_right",
    "q_left",
    "chosen_action",
    "decision_source",
    "reward",
    "done",
    "score",
    "epsilon",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def append_csv_row(path, fieldnames, row):
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def action_index(action):
    return action.index(1)


def action_name(index):
    return ["straight", "right", "left"][index]


def should_log_decision(log_decisions, episode, decision_frequency):
    return log_decisions and episode % decision_frequency == 0


def load_score_history(training_log):
    if not training_log.exists():
        return []

    with training_log.open(newline="") as file:
        reader = csv.DictReader(file)
        return [int(row["score"]) for row in reader if row.get("score") != ""]


def train(args):
    set_seed(args.seed)

    log_dir = Path(args.log_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    training_log = log_dir / "training.csv"
    decisions_log = log_dir / "decisions.csv"

    if args.reset_logs:
        for log_path in (training_log, decisions_log):
            if log_path.exists():
                log_path.unlink()

    game = SnakeGameAI(render=args.render, speed=args.speed)
    agent = DQNAgent()

    checkpoint_path = args.checkpoint
    latest_checkpoint = checkpoint_dir / "latest.pt"

    if checkpoint_path is None and args.resume and latest_checkpoint.exists():
        checkpoint_path = latest_checkpoint

    if checkpoint_path:
        checkpoint = agent.load_checkpoint(checkpoint_path)
        print(
            "Loaded checkpoint:",
            checkpoint_path,
            "episode",
            checkpoint.get("episode", 0),
        )

    target_episode = args.episodes
    score_history = load_score_history(training_log)

    reward_total = 0
    step_losses = []
    action_counts = [0, 0, 0]
    random_actions = 0
    model_actions = 0
    q_value_total = 0.0
    max_q_value = float("-inf")

    print(
        "Training forever. Press Ctrl+C to stop safely after the current step."
        if target_episode == 0
        else f"Training until episode {target_episode}.",
        flush=True,
    )

    try:
        while target_episode == 0 or agent.episode < target_episode:
            state_old = agent.get_state(game)

            action = agent.get_action(state_old, training=True)
            chosen_index = action_index(action)

            reward, done, score = game.play_step(action)
            state_new = agent.get_state(game)

            loss = agent.train_short_memory(state_old, action, reward, state_new, done)
            agent.remember(state_old, action, reward, state_new, done)

            reward_total += reward
            step_losses.append(loss)
            action_counts[chosen_index] += 1

            if agent.last_decision_source == "RANDOM":
                random_actions += 1
            else:
                model_actions += 1

            q_value_total += statistics.mean(agent.last_q_values)
            max_q_value = max(max_q_value, max(agent.last_q_values))

            if should_log_decision(
                args.log_decisions,
                agent.episode + 1,
                args.decision_frequency,
            ):
                append_csv_row(
                    decisions_log,
                    DECISION_FIELDS,
                    {
                        "episode": agent.episode + 1,
                        "step": game.frame_iteration,
                        "state": state_old,
                        "q_straight": agent.last_q_values[0],
                        "q_right": agent.last_q_values[1],
                        "q_left": agent.last_q_values[2],
                        "chosen_action": action_name(chosen_index),
                        "decision_source": agent.last_decision_source,
                        "reward": reward,
                        "done": done,
                        "score": score,
                        "epsilon": agent.epsilon,
                    },
                )

            if done:
                steps = game.frame_iteration
                death_reason = game.death_reason or "unknown"

                game.reset()
                agent.episode += 1

                replay_loss = agent.train_long_memory()
                agent.decay_epsilon()

                if agent.episode % TARGET_UPDATE_FREQUENCY == 0:
                    agent.update_target_network()

                score_history.append(score)
                if score > agent.best_score:
                    agent.best_score = score
                    agent.save_checkpoint(checkpoint_dir / "best.pt")

                agent.save_checkpoint(checkpoint_dir / "latest.pt", include_memory=True)

                if agent.episode % CHECKPOINT_FREQUENCY == 0:
                    agent.save_checkpoint(
                        checkpoint_dir / f"episode_{agent.episode:06d}.pt",
                        include_memory=True,
                    )

                all_losses = step_losses + [replay_loss]
                average_loss = statistics.mean(all_losses) if all_losses else 0.0
                average_score_100 = statistics.mean(score_history[-100:])

                row = {
                    "episode": agent.episode,
                    "score": score,
                    "reward_total": reward_total,
                    "steps": steps,
                    "epsilon": agent.epsilon,
                    "loss": replay_loss,
                    "average_loss": average_loss,
            "memory_size": len(agent.memory),
            "best_score": agent.best_score,
            "average_score_100": average_score_100,
                    "straight_actions": action_counts[0],
                    "right_actions": action_counts[1],
                    "left_actions": action_counts[2],
                    "random_actions": random_actions,
                    "model_actions": model_actions,
                    "food_eaten": score,
                    "death_reason": death_reason,
                    "mean_q_value": q_value_total / max(steps, 1),
                    "max_q_value": max_q_value,
                }
                append_csv_row(training_log, TRAINING_FIELDS, row)

                print(
                    f"Episode {agent.episode:4d} | "
                    f"score {score:3d} | "
                    f"avg100 {average_score_100:.2f} | "
                    f"best {agent.best_score:3d} | "
                    f"epsilon {agent.epsilon:.3f} | "
                    f"loss {average_loss:.4f}",
                    flush=True,
                )

                reward_total = 0
                step_losses = []
                action_counts = [0, 0, 0]
                random_actions = 0
                model_actions = 0
                q_value_total = 0.0
                max_q_value = float("-inf")
    except KeyboardInterrupt:
        agent.save_checkpoint(checkpoint_dir / "latest.pt", include_memory=True)
        print(
            f"\nStopped. Saved latest checkpoint at episode {agent.episode}.",
            flush=True,
        )
        return



def run_episode(game, agent, training=False):
    game.reset()

    while True:
        state = agent.get_state(game)

        if isinstance(agent, DQNAgent):
            action = agent.get_action(state, training=training)
        else:
            action = agent.get_action(state)

        reward, done, score = game.play_step(action)

        if done:
            return {
                "score": score,
                "steps": game.frame_iteration,
                "reward": reward,
                "death_reason": game.death_reason,
            }


def evaluate_dqn(args, checkpoint_path):
    game = SnakeGameAI(render=args.render, speed=args.speed)
    agent = DQNAgent(epsilon_start=0.0)
    agent.load_checkpoint(checkpoint_path)
    agent.epsilon = 0.0
    agent.model.eval()
    agent.target_model.eval()

    return evaluate_agent(game, agent, args.eval_episodes)


def evaluate_random(args):
    game = SnakeGameAI(render=args.render, speed=args.speed)
    agent = RandomAgent()
    return evaluate_agent(game, agent, args.eval_episodes)


def evaluate_agent(game, agent, episodes):
    scores = []
    lengths = []

    for _ in range(episodes):
        result = run_episode(game, agent, training=False)
        scores.append(result["score"])
        lengths.append(result["steps"])

    return {
        "episodes": episodes,
        "average_score": statistics.mean(scores),
        "median_score": statistics.median(scores),
        "best_score": max(scores),
        "average_episode_length": statistics.mean(lengths),
    }


def print_evaluation_report(name, result):
    print(name)
    print(f"Episodes: {result['episodes']}")
    print(f"Average score: {result['average_score']:.2f}")
    print(f"Median score: {result['median_score']:.2f}")
    print(f"Best score: {result['best_score']}")
    print(f"Average episode length: {result['average_episode_length']:.2f}")


def default_eval_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    best = checkpoint_dir / "best.pt"
    latest = checkpoint_dir / "latest.pt"

    if best.exists():
        return best
    return latest


def evaluate(args):
    if args.agent in ("random", "both"):
        random_result = evaluate_random(args)
        print_evaluation_report("Random Agent", random_result)

    if args.agent in ("dqn", "both"):
        checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_eval_checkpoint(
            args.checkpoint_dir
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No DQN checkpoint found at {checkpoint_path}. Train first or pass --checkpoint."
            )

        dqn_result = evaluate_dqn(args, checkpoint_path)
        print_evaluation_report("DQN Agent", dqn_result)


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Train or evaluate Snake DQN ({EXPERIMENT_NAME}, {STATE_SIZE} inputs)."
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_TRAIN_EPISODES)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--agent", choices=["dqn", "random", "both"], default="dqn")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--speed", type=int, default=TRAINING_SPEED)
    parser.add_argument("--render", action="store_true", default=DEFAULT_RENDER)
    parser.add_argument("--log-decisions", action="store_true")
    parser.add_argument("--reset-logs", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--decision-frequency",
        type=int,
        default=DECISION_LOG_EVERY_N_EPISODES,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
