import random
import sys
from collections import deque

import torch

from config import (
    BATCH_SIZE,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    GAMMA,
    HIDDEN_SIZE,
    LEARNING_RATE,
    MAX_MEMORY,
    STATE_SIZE,
)
from model import LinearQNet, load_checkpoint, save_checkpoint
from trainer import QTrainer


STATE_FEATURE_NAMES = [
    "danger_straight",
    "danger_right",
    "danger_left",
    "moving_left",
    "moving_right",
    "moving_up",
    "moving_down",
    "food_left",
    "food_right",
    "food_up",
    "food_down",
    "danger_two_steps_straight",
    "danger_two_steps_right",
    "danger_two_steps_left",
    "food_dx_normalized",
    "food_dy_normalized",
    "food_distance_if_straight",
    "food_distance_if_right",
    "food_distance_if_left",
    "reachable_space_if_straight",
    "reachable_space_if_right",
    "reachable_space_if_left",
]


def get_game_state(game):
    head = game.head
    game_module = sys.modules[game.__class__.__module__]
    block_size = game_module.BLOCK_SIZE
    width = game_module.WIDTH
    height = game_module.HEIGHT
    point_type = type(head)

    point_left = point_type(head.x - block_size, head.y)
    point_right = point_type(head.x + block_size, head.y)
    point_up = point_type(head.x, head.y - block_size)
    point_down = point_type(head.x, head.y + block_size)

    dir_left = game.direction.name == "LEFT"
    dir_right = game.direction.name == "RIGHT"
    dir_up = game.direction.name == "UP"
    dir_down = game.direction.name == "DOWN"

    current_direction = game.direction.name
    straight_direction = current_direction
    right_direction = turn_right(current_direction)
    left_direction = turn_left(current_direction)

    point_straight = move_point(head, straight_direction, block_size, point_type)
    point_turn_right = move_point(head, right_direction, block_size, point_type)
    point_turn_left = move_point(head, left_direction, block_size, point_type)

    point_two_straight = move_point(head, straight_direction, block_size, point_type, 2)
    point_two_right = move_point(head, right_direction, block_size, point_type, 2)
    point_two_left = move_point(head, left_direction, block_size, point_type, 2)

    danger_straight = (
        (dir_right and game.is_collision(point_right))
        or (dir_left and game.is_collision(point_left))
        or (dir_up and game.is_collision(point_up))
        or (dir_down and game.is_collision(point_down))
    )

    danger_right = (
        (dir_right and game.is_collision(point_down))
        or (dir_down and game.is_collision(point_left))
        or (dir_left and game.is_collision(point_up))
        or (dir_up and game.is_collision(point_right))
    )

    danger_left = (
        (dir_right and game.is_collision(point_up))
        or (dir_up and game.is_collision(point_left))
        or (dir_left and game.is_collision(point_down))
        or (dir_down and game.is_collision(point_right))
    )

    max_food_distance = width + height
    max_reachable_cells = (width // block_size) * (height // block_size)

    distance_straight = normalized_food_distance(
        game,
        point_straight,
        max_food_distance,
    )
    distance_right = normalized_food_distance(game, point_turn_right, max_food_distance)
    distance_left = normalized_food_distance(game, point_turn_left, max_food_distance)

    reachable_straight = normalized_reachable_space(
        game,
        point_straight,
        block_size,
        width,
        height,
        point_type,
        max_reachable_cells,
    )
    reachable_right = normalized_reachable_space(
        game,
        point_turn_right,
        block_size,
        width,
        height,
        point_type,
        max_reachable_cells,
    )
    reachable_left = normalized_reachable_space(
        game,
        point_turn_left,
        block_size,
        width,
        height,
        point_type,
        max_reachable_cells,
    )

    state = [
        # Original 11 features.
        danger_straight,
        danger_right,
        danger_left,
        dir_left,
        dir_right,
        dir_up,
        dir_down,
        game.food.x < game.head.x,
        game.food.x > game.head.x,
        game.food.y < game.head.y,
        game.food.y > game.head.y,

        # New v2 features.
        game.is_collision(point_two_straight),
        game.is_collision(point_two_right),
        game.is_collision(point_two_left),
        (game.food.x - game.head.x) / width,
        (game.food.y - game.head.y) / height,
        distance_straight,
        distance_right,
        distance_left,
        reachable_straight,
        reachable_right,
        reachable_left,
    ]

    return [float(value) for value in state]


def turn_right(direction_name):
    clockwise = ["RIGHT", "DOWN", "LEFT", "UP"]
    return clockwise[(clockwise.index(direction_name) + 1) % 4]


def turn_left(direction_name):
    clockwise = ["RIGHT", "DOWN", "LEFT", "UP"]
    return clockwise[(clockwise.index(direction_name) - 1) % 4]


def move_point(point, direction_name, block_size, point_type, steps=1):
    distance = block_size * steps

    if direction_name == "RIGHT":
        return point_type(point.x + distance, point.y)

    if direction_name == "LEFT":
        return point_type(point.x - distance, point.y)

    if direction_name == "UP":
        return point_type(point.x, point.y - distance)

    return point_type(point.x, point.y + distance)


def normalized_food_distance(game, point, max_food_distance):
    distance = abs(game.food.x - point.x) + abs(game.food.y - point.y)
    return distance / max_food_distance


def normalized_reachable_space(
    game,
    start,
    block_size,
    width,
    height,
    point_type,
    max_reachable_cells,
):
    if game.is_collision(start):
        return 0.0

    occupied = set(game.snake[1:])
    queue = deque([start])
    seen = {start}

    while queue:
        point = queue.popleft()

        for direction_name in ("RIGHT", "LEFT", "UP", "DOWN"):
            neighbor = move_point(point, direction_name, block_size, point_type)

            if neighbor in seen or neighbor in occupied:
                continue

            if neighbor.x < 0 or neighbor.x >= width:
                continue

            if neighbor.y < 0 or neighbor.y >= height:
                continue

            seen.add(neighbor)
            queue.append(neighbor)

    return len(seen) / max_reachable_cells


class RandomAgent:
    def get_state(self, game):
        return get_game_state(game)

    def get_action(self, state):
        move = random.randint(0, 2)

        final_move = [0, 0, 0]
        final_move[move] = 1

        return final_move


class DQNAgent:
    def __init__(
        self,
        input_size=STATE_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=3,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        max_memory=MAX_MEMORY,
        batch_size=BATCH_SIZE,
        epsilon_start=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
    ):
        self.episode = 0
        self.best_score = -1
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=max_memory)

        self.model = LinearQNet(input_size, hidden_size, output_size)
        self.target_model = LinearQNet(input_size, hidden_size, output_size)
        self.update_target_network()
        self.target_model.eval()

        self.trainer = QTrainer(
            self.model,
            self.target_model,
            learning_rate,
            gamma,
        )

        self.last_decision_source = None
        self.last_q_values = [0.0, 0.0, 0.0]
        self.input_size = input_size

    def get_state(self, game):
        return get_game_state(game)

    def get_action(self, state, training=True):
        q_values = self.predict_q_values(state)

        # Epsilon keeps early training exploratory while the network is still
        # mostly random. As epsilon decays, the model is trusted more often.
        if training and random.random() < self.epsilon:
            move = random.randint(0, 2)
            self.last_decision_source = "RANDOM"
        else:
            move = max(range(len(q_values)), key=q_values.__getitem__)
            self.last_decision_source = "MODEL"

        final_move = [0, 0, 0]
        final_move[move] = 1

        return final_move

    def predict_q_values(self, state):
        state_tensor = torch.as_tensor(state, dtype=torch.float32)

        with torch.no_grad():
            prediction = self.model(state_tensor)

        self.last_q_values = [float(value) for value in prediction.tolist()]
        return self.last_q_values

    def remember(self, state, action, reward, next_state, done):
        # Replay memory lets each old experience be reused many times, reducing
        # the correlation from learning only from the newest game sequence.
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        return self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        if not self.memory:
            return 0.0

        if len(self.memory) > self.batch_size:
            sample = random.sample(self.memory, self.batch_size)
        else:
            sample = list(self.memory)

        states, actions, rewards, next_states, dones = zip(*sample)
        return self.trainer.train_step(states, actions, rewards, next_states, dones)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def update_target_network(self):
        # The target network changes less often, which makes Bellman targets
        # less jumpy while the online model is being updated every step.
        self.target_model.load_state_dict(self.model.state_dict())

    def save_checkpoint(self, path, extra=None, include_memory=False):
        checkpoint_extra = {
            "input_size": self.input_size,
            "state_feature_names": STATE_FEATURE_NAMES,
            **dict(extra or {}),
        }

        if include_memory:
            checkpoint_extra["memory"] = list(self.memory)
            checkpoint_extra["memory_maxlen"] = self.memory.maxlen
            checkpoint_extra["batch_size"] = self.batch_size

        save_checkpoint(
            path,
            self.model,
            self.trainer.optimizer,
            self.episode,
            self.best_score,
            self.epsilon,
            target_model=self.target_model,
            extra=checkpoint_extra,
        )

    def load_checkpoint(self, path):
        checkpoint = load_checkpoint(
            path,
            self.model,
            self.trainer.optimizer,
            target_model=self.target_model,
        )
        self.episode = int(checkpoint.get("episode", 0))
        self.best_score = int(checkpoint.get("best_score", 0))
        self.epsilon = float(checkpoint.get("epsilon", self.epsilon))
        if "memory" in checkpoint:
            memory_maxlen = int(checkpoint.get("memory_maxlen", self.memory.maxlen))
            self.memory = deque(checkpoint["memory"], maxlen=memory_maxlen)
            self.batch_size = int(checkpoint.get("batch_size", self.batch_size))
        self.target_model.eval()
        return checkpoint
