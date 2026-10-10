# Snake AI

Snake AI is a Python reinforcement learning project that turns a Pygame Snake environment into a Deep Q-Network training loop. The project focuses on making the RL pipeline inspectable: state features, actions, rewards, replay memory, checkpoints, CSV logs, and dashboards are all kept explicit.

## What This Project Demonstrates

- A playable Snake environment converted into an RL environment.
- Relative actions: straight, turn right, turn left.
- A random baseline agent.
- A PyTorch DQN agent with replay memory, target network updates, and epsilon-greedy exploration.
- Training and evaluation scripts.
- Repeatable benchmarking against a random baseline.
- Checkpoint saving and resume support.
- CSV logging and Matplotlib dashboards.
- Experiment tracking for a stronger 11-feature v1 run and a later 22-feature v2 experiment.

## Current Status

This is an educational RL project, not a solved Snake bot.

The strongest run so far was the v1 11-feature DQN:

```text
episodes: 2088
best score: 35
final average_score_100: 8.20
```

For comparison, the random baseline agent averages 0.13 points per game (best score 1) over 100 games, measured with `python benchmark.py --agent random --episodes 100 --no-save`.

The v2 22-feature experiment added lookahead and reachable-space features, but early results were worse:

```text
episodes: 902
best score: 3
final average_score_100: 0.21
```

That negative result is intentional to keep in the project history: richer state features do not automatically improve an RL agent without matching training changes.

## Visual Results

#Snake Runs:

Initial Runs:

<img width="1282" height="1008" alt="1537842B-975D-4581-B97F-D29273002C47_1_206_a" src="https://github.com/user-attachments/assets/a08cfac8-c047-49f8-a594-6d3bc0ec0008" />

Latter Runs:
<img width="454" height="360" alt="8EE625FC-34F2-4A41-8CEB-952A552BCECC_4_5005_c" src="https://github.com/user-attachments/assets/1378c337-a772-4df5-94ac-32ab1926270d" />


Comparison dashboard:

![Snake DQN comparison](docs/assets/comparison_dashboard.png)

v1 dashboard:

![v1 dashboard](docs/assets/v1_dashboard.png)

v2 dashboard:

![v2 dashboard](docs/assets/v2_dashboard.png)

## Project Structure

```text
.
├── agent.py              # RandomAgent, DQNAgent, state feature extraction
├── benchmark.py          # random vs DQN benchmark runner
├── compare_dashboard.py  # v1 vs v2 dashboard generator
├── config.py             # hyperparameters and experiment defaults
├── dashboard.py          # single-run dashboard generator
├── model.py              # LinearQNet and checkpoint helpers
├── snake_game.py         # Pygame Snake environment
├── train.py              # training and evaluation CLI
├── trainer.py            # Bellman target / DQN training step
├── PROJECT_HISTORY.md    # iteration log and experiment notes
└── requirements.txt
```

Generated artifacts are intentionally ignored by Git:

```text
logs/
checkpoints/
.venv/
__pycache__/
```

Curated dashboard snapshots are committed under `docs/assets/`.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run The Random-Agent Demo

```bash
python snake_game.py
```

## Train The Current v2 DQN

Headless training:

```bash
python train.py --episodes 500 --reset-logs
```

Visual training:

```bash
python train.py --episodes 0 --resume --render --speed 20 --log-decisions --decision-frequency 1
```

Stop safely with `Ctrl+C`. The latest checkpoint is saved before exit.

## Evaluate

Evaluate the current DQN checkpoint:

```bash
python train.py --eval --agent dqn --eval-episodes 100
```

Compare random baseline and DQN:

```bash
python train.py --eval --agent both --eval-episodes 100
```

## Benchmark

Run a repeatable benchmark and append results to `logs/state_v2/benchmark.csv`:

```bash
python benchmark.py --agent both --episodes 100
```

Print benchmark results without saving:

```bash
python benchmark.py --agent both --episodes 100 --no-save
```

The benchmark records average score, median score, best score, and average episode length. This gives future training runs a cleaner comparison against the random baseline.

## Dashboards

Generate the current experiment dashboard:

```bash
python dashboard.py
```

Generate the comparison dashboard after local v1 and v2 logs exist:

```bash
python compare_dashboard.py
```

## Notes For Reviewers

- Checkpoints and raw logs can become large, so they are excluded from Git.
- Benchmark CSV outputs are also generated locally and excluded from Git.
- `PROJECT_HISTORY.md` records the major implementation iterations and experimental observations.
- The v2 model starts from scratch because its input size changed from 11 to 22.
- The next likely improvement is reward shaping or a revised feature design, not simply longer v2 training.
