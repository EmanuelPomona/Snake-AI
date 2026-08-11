# Snake AI Project History

This document tracks the major iterations of the Snake reinforcement learning project, why each change was made, and what evidence was produced. It is intended as source material for a future paper or write-up.

## Project Goal

Build a transparent reinforcement learning Snake agent from scratch:

- Start with a playable Snake game.
- Convert it into an RL environment.
- Add a random baseline agent.
- Add a DQN agent.
- Track training evidence through logs, checkpoints, and dashboards.
- Improve the agent by improving what it can observe.

## Iteration 1: Local Python Environment and Pygame

Files involved:

- `snake_game.py`
- `.venv/`
- `requirements.txt`

What changed:

- Created a local virtual environment in `.venv`.
- Tried installing `pygame`.
- Regular `pygame` failed on Python 3.14 because it attempted to compile from source and could not find SDL headers.
- Installed `pygame-ce==2.5.7`, which provides the `pygame` import and has a compatible Python 3.14 macOS wheel.

Why:

- Keep dependencies local to the project.
- Get the visual Snake window running without changing system Python.

Result:

- `snake_game.py` launched successfully through:

```bash
.venv/bin/python snake_game.py
```

## Iteration 2: Human-Controlled Snake to RL Environment

Files involved:

- `snake_game.py`

What changed:

- Replaced keyboard direction control with relative actions.
- Renamed the environment class to `SnakeGameAI`.
- Changed `play_step()` to accept an action.
- Returned reward, game-over status, and score:

```python
reward, game_over, score = game.play_step(action)
```

Action format:

```text
[1, 0, 0] = straight
[0, 1, 0] = turn right
[0, 0, 1] = turn left
```

Reward function:

```text
food eaten  = +10
death       = -10
normal step = 0
```

Also added:

- `frame_iteration`
- timeout condition: `frame_iteration > 100 * len(snake)`
- `is_collision(point=None)` so the agent can ask about hypothetical moves.

Why:

- Reinforcement learning needs an environment that accepts agent actions and returns rewards.
- Relative actions are better than absolute arrow-key directions because the agent learns decisions in the snake's frame of reference.

## Iteration 3: Random Agent Baseline

Files involved:

- `agent.py`
- `snake_game.py`

What changed:

- Added `RandomAgent`.
- `RandomAgent.get_action(state)` initially ignored state and returned random actions.
- Connected the game loop to the agent.

Why:

- Establish a baseline.
- Verify that the environment can be driven by an automated agent before adding neural networks.

Result:

- Snake moved by itself and usually died quickly, which was expected.

## Iteration 4: 11-Feature State Representation

Files involved:

- `agent.py`

What changed:

Added `get_state(game)` with 11 features:

```text
0  danger straight
1  danger right
2  danger left
3  moving left
4  moving right
5  moving up
6  moving down
7  food left
8  food right
9  food up
10 food down
```

Why:

- The agent needed an observation vector.
- The first state representation was deliberately simple and inspectable.

Limitations:

- It only captured immediate danger.
- It did not know full body shape.
- It did not know whether a move would trap it later.
- It did not know how much free space was reachable.

## Iteration 5: DQN Agent

Files involved:

- `model.py`
- `trainer.py`
- `agent.py`
- `train.py`
- `config.py`

What changed:

Added a Deep Q-Network:

```text
11 inputs -> 256 hidden units -> 3 Q-values
```

The 3 output Q-values represent:

```text
0 = straight
1 = right
2 = left
```

Added:

- `LinearQNet`
- `QTrainer`
- Bellman target learning
- Smooth L1 loss
- Adam optimizer
- replay memory
- short-term training after each action
- long-term replay training after each episode
- epsilon-greedy exploration
- target network

Why:

- Move from random behavior to value-based reinforcement learning.
- Let the model learn which actions tend to produce better future rewards.

## Iteration 6: Logging, Checkpoints, Dashboard, and Evaluation

Files involved:

- `train.py`
- `dashboard.py`
- `logs/training.csv`
- `logs/decisions.csv`
- `checkpoints/latest.pt`
- `checkpoints/best.pt`

What changed:

Training logs were added:

```text
episode
score
reward_total
steps
epsilon
loss
average_loss
memory_size
best_score
average_score_100
straight_actions
right_actions
left_actions
random_actions
model_actions
food_eaten
death_reason
mean_q_value
max_q_value
```

Decision logs were added:

```text
episode
step
state
q_straight
q_right
q_left
chosen_action
decision_source
reward
done
score
epsilon
```

Checkpoints were added:

- `latest.pt`
- `best.pt`
- milestone checkpoints such as `episode_001500.pt`

Dashboard was added:

- score by episode
- average score over last 100 episodes
- best score
- total reward
- loss
- epsilon
- memory size
- recent episodes

Why:

- Training needed evidence.
- RL is noisy, so single episodes are not enough.
- `average_score_100` became the main evidence of improvement.

## Iteration 7: Continuous Training and Visual Training

Files involved:

- `train.py`

What changed:

Added continuous training:

```bash
.venv/bin/python train.py --episodes 0 --resume
```

Added visual continuous training:

```bash
.venv/bin/python train.py --episodes 0 --resume --render --speed 20 --log-decisions --decision-frequency 1
```

Added:

- safe `Ctrl+C` handling
- immediate output flushing
- automatic save on stop
- resume from `latest.pt`

Why:

- Allow long-running learning.
- Watch the agent play while training.
- Keep logs updating while the visual window is running.

## Iteration 8: Replay Memory Persistence

Files involved:

- `agent.py`
- `train.py`

Problem observed:

- Replay memory grew to `100000`.
- After stopping and resuming, memory dropped because checkpoints saved model weights but not replay memory.
- This made resumed runs lose their experience buffer.

What changed:

- `latest.pt` now saves replay memory.
- Milestone checkpoints also save replay memory.
- `best.pt` remains smaller and model-focused.

Current verified checkpoint state after this change:

```text
episode 2088
best_score 35
memory_size 42436
```

Why:

- Preserve learned weights and recent experiences.
- Improve continuity across training sessions.
- Avoid cold replay memory after restart.

## Iteration 9: Results From 11-Feature DQN

Observed evidence:

```text
latest episode: 2088
latest score: 14
average_score_100: 8.20
best score: 35
replay memory saved: 42436 experiences
```

Interpretation:

- The model improved beyond random behavior.
- Scores became meaningfully higher than early training.
- The agent still looked goofy because the 11-feature state was too limited.

Key limitation:

- The agent could avoid immediate danger but could still trap itself because it had no information about longer-term space.

## Iteration 10: State v2 Feature Upgrade

Files involved:

- `agent.py`
- `model.py`
- `config.py`
- `train.py`
- `dashboard.py`

What changed:

Created a new experiment version:

```text
EXPERIMENT_NAME = state_v2
STATE_SIZE = 22
```

New default output paths:

```text
logs/state_v2/
checkpoints/state_v2/
```

The v2 state keeps the original 11 features and adds 11 more:

```text
11 danger two steps straight
12 danger two steps right
13 danger two steps left
14 food dx normalized
15 food dy normalized
16 food distance if straight
17 food distance if right
18 food distance if left
19 reachable space if straight
20 reachable space if right
21 reachable space if left
```

Why:

- The old agent only knew immediate danger.
- The new features give it a small amount of lookahead.
- Reachable-space features help it prefer moves that leave more room and avoid self-trapping.
- Food-distance-if-action features help it compare action consequences, not only the current food direction.

Important:

- This changes the model input size from 11 to 22.
- Old 11-feature checkpoints are not compatible with the v2 model.
- The old experiment remains preserved in:

```text
logs/
checkpoints/
```

The new v2 experiment starts fresh in:

```text
logs/state_v2/
checkpoints/state_v2/
```

Validation after implementation:

```text
state size: 22
checkpoint input_size: 22
checkpoint feature count: 22
smoke training episodes: 3
v2 checkpoint memory after smoke test: 167 experiences
v2 dashboard: logs/state_v2/dashboard.png
```

## Iteration 11: v1 vs v2 Comparison Dashboard

Files involved:

- `compare_dashboard.py`
- `logs/comparison_dashboard.png`
- `logs/training.csv`
- `logs/state_v2/training.csv`

What changed:

- Added a comparison dashboard that plots v1 and v2 on the same charts.
- Preserved v1 artifacts at the original top-level paths:

```text
logs/training.csv
logs/dashboard.png
checkpoints/latest.pt
checkpoints/best.pt
```

- Preserved v2 artifacts separately:

```text
logs/state_v2/training.csv
logs/state_v2/dashboard.png
checkpoints/state_v2/latest.pt
checkpoints/state_v2/best.pt
```

Comparison dashboard:

```text
logs/comparison_dashboard.png
```

Current comparison snapshot:

```text
v1 episodes: 2088
v1 environment steps: 356789
v1 total score across episodes: 8399
v1 final average_score_100: 8.20
v1 best score: 35

v2 episodes: 3
v2 environment steps: 167
v2 total score across episodes: 0
v2 final average_score_100: 0.00
v2 best score: 0
```

Training-time estimate:

```text
v1 environment time at 20 FPS: about 297.3 minutes
v1 environment time at 30 FPS: about 198.2 minutes
v2 environment time at 20 FPS: about 0.14 minutes
```

Important caveat:

- The original logs did not include per-episode wall-clock timestamps.
- These are estimates based on total environment steps and the configured Pygame speed.
- Actual wall-clock time differs because some runs were visual at `--speed 20`, some were visual at other speeds, and some were headless/fast.

## Iteration 12: v2 Training Observation and Failure Mode

Files involved:

- `logs/state_v2/training.csv`
- `logs/state_v2/dashboard.png`
- `checkpoints/state_v2/latest.pt`
- `logs/comparison_dashboard.png`

What happened:

- v2 was trained visually after the feature upgrade.
- Training was stopped safely at episode 902.
- Logs, checkpoints, and dashboards were refreshed after stopping.

Observed v2 metrics:

```text
episodes: 902
latest score: 0
latest average_score_100: 0.21
best score: 3
replay memory size: 96899
epsilon: 0.05
```

Average score by episode block:

```text
1-100:   0.100, max 1
101-200: 0.120, max 2
201-300: 0.250, max 2
301-400: 0.210, max 3
401-500: 0.170, max 2
501-600: 0.180, max 2
601-700: 0.130, max 1
701-800: 0.120, max 2
801-902: 0.206, max 2
```

Interpretation:

- v2 did not meaningfully improve during this run.
- The richer state representation alone did not produce better behavior.
- Since epsilon reached its floor and the replay buffer nearly filled, the problem is probably not just lack of episodes.
- The agent may need reward shaping, a better exploration schedule, a corrected feature design, or a different way to use space-awareness.

Research note:

- This is useful negative evidence. It shows that adding features can make the learning problem harder if the new representation is not paired with a training strategy that can exploit it.

## Iteration 13: Repeatable Benchmarking Workflow

Files involved:

- `benchmark.py`
- `README.md`
- `PROJECT_HISTORY.md`

What changed:

- Added a dedicated benchmark script that evaluates:

```text
random agent
DQN checkpoint
both agents side by side
```

- The benchmark can append comparable rows to:

```text
logs/state_v2/benchmark.csv
```

- Each benchmark row records:

```text
timestamp_utc
agent
episodes
checkpoint
average_score
median_score
best_score
average_episode_length
```

Why:

- Training logs show how learning changes episode by episode, but they are not the same as a controlled evaluation.
- A benchmark gives the project a repeatable way to compare the trained DQN against a random baseline.
- This makes future improvements easier to defend with evidence instead of visual impressions alone.

Example usage:

```bash
.venv/bin/python benchmark.py --agent both --episodes 100
```

Public repo note:

- Raw benchmark CSVs remain ignored by Git, just like training logs and checkpoints.
- The script itself is committed so anyone can reproduce the benchmark locally after training or downloading a checkpoint.

## Current Research Notes

Main metric to watch:

```text
average_score_100
```

Why:

- Individual Snake episodes are noisy.
- Best score can stay flat for a long time.
- Average score over 100 episodes shows whether behavior is improving consistently.

Current hypothesis:

- The v1 agent learned useful but incomplete behavior.
- The v2 agent was expected to outperform v1 because it can reason about immediate space and short-term consequences.
- Early v2 evidence did not support that expectation. At episode 902 it was still far below v1.
- The next iteration should change the training signal or feature design, not just continue the same v2 run indefinitely.

Future ideas:

- Add flood-fill dead-end penalties.
- Add Double DQN.
- Add explicit tail awareness.
- Reduce decision logging during long headless training to improve speed.
- Compare v1 and v2 over equal episode counts.
