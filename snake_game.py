
import pygame
import random
from enum import Enum
from collections import namedtuple

from config import REWARD_DEATH, REWARD_FOOD, REWARD_STEP


pygame.init()

FONT = pygame.font.Font(None, 30)

BLOCK_SIZE = 20
SPEED = 10

WIDTH = 640
HEIGHT = 480


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x y")


class SnakeGameAI:
    def __init__(self, render=True, speed=SPEED):
        self.render = render
        self.speed = speed
        self.display = None

        if self.render:
            self.display = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Snake RL")

        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.frame_iteration = 0
        self.death_reason = None
        self.direction = Direction.RIGHT

        center_x = WIDTH // 2
        center_y = HEIGHT // 2

        self.head = Point(center_x, center_y)

        self.snake = [
            self.head,
            Point(center_x - BLOCK_SIZE, center_y),
            Point(center_x - 2 * BLOCK_SIZE, center_y),
        ]

        self.score = 0
        self.food = None

        self._place_food()

    def _place_food(self):
        x = random.randint(0, (WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE

        self.food = Point(x, y)

        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1

        # 1. Handle quit event
        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        # 2. Move snake based on agent action
        self._move(action)

        self.snake.insert(0, self.head)

        # 3. Check collision or timeout
        reward = REWARD_STEP
        game_over = False

        collision = self.is_collision()
        timeout = self.frame_iteration > 100 * len(self.snake)

        if collision or timeout:
            game_over = True
            reward = REWARD_DEATH
            self.death_reason = "collision" if collision else "timeout"
            return reward, game_over, self.score

        # 4. Check if food was eaten
        if self.head == self.food:
            self.score += 1
            reward = REWARD_FOOD
            self._place_food()
        else:
            self.snake.pop()

        # 5. Draw everything
        if self.render:
            self._update_ui()

        if self.render and self.speed > 0:
            self.clock.tick(self.speed)

        return reward, game_over, self.score

    def is_collision(self, point=None):
        if point is None:
            point = self.head

        # Wall collision
        if point.x >= WIDTH or point.x < 0:
            return True

        if point.y >= HEIGHT or point.y < 0:
            return True

        # Snake body collision
        if point in self.snake[1:]:
            return True

        return False

    def _move(self, action):
        # action can be either:
        # 0 = straight, 1 = right turn, 2 = left turn
        # or one-hot: [1, 0, 0], [0, 1, 0], [0, 0, 1]
        if isinstance(action, int):
            action_index = action
        else:
            action_index = action.index(1)

        clockwise = [
            Direction.RIGHT,
            Direction.DOWN,
            Direction.LEFT,
            Direction.UP,
        ]
        current_index = clockwise.index(self.direction)

        if action_index == 0:
            new_direction = clockwise[current_index]
        elif action_index == 1:
            next_index = (current_index + 1) % 4
            new_direction = clockwise[next_index]
        else:
            next_index = (current_index - 1) % 4
            new_direction = clockwise[next_index]

        self.direction = new_direction

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE

        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE

        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE

        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def _update_ui(self):
        if self.display is None:
            return

        self.display.fill((0, 0, 0))

        for point in self.snake:
            pygame.draw.rect(
                self.display,
                (0, 255, 0),
                pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE),
            )

        pygame.draw.rect(
            self.display,
            (255, 0, 0),
            pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE),
        )

        text = FONT.render(f"Score: {self.score}", True, (255, 255, 255))
        self.display.blit(text, (10, 10))

        pygame.display.flip()


if __name__ == "__main__":
    from agent import RandomAgent

    game = SnakeGameAI()
    agent = RandomAgent()

    while True:
        state = agent.get_state(game)
        action = agent.get_action(state)
        reward, game_over, score = game.play_step(action)

        print("State:", state)
        print("Action:", action)
        print("Reward:", reward)

        if game_over:
            print("Final score:", score)
            break

    pygame.quit()
