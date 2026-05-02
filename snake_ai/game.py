import pygame
import random
from collections import deque
from enum import Enum

BLOCK = 20
WHITE  = (255, 255, 255)
GREEN1 = (0, 210, 0)
GREEN2 = (0, 150, 0)
RED    = (200, 30, 30)
BLACK  = (0, 0, 0)
GRAY   = (35, 35, 35)
BLUE   = (50, 120, 220)


class Direction(Enum):
    RIGHT = 1
    LEFT  = 2
    UP    = 3
    DOWN  = 4


class Point:
    __slots__ = ('x', 'y')

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return f'Point({self.x}, {self.y})'

_CW = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]


def _turn_right(d):
    return _CW[(_CW.index(d) + 1) % 4]


def _turn_left(d):
    return _CW[(_CW.index(d) - 1) % 4]


def _move_point(pt, d):
    if d == Direction.RIGHT:
        return Point(pt.x + BLOCK, pt.y)
    if d == Direction.LEFT:
        return Point(pt.x - BLOCK, pt.y)
    if d == Direction.UP:
        return Point(pt.x, pt.y - BLOCK)
    return Point(pt.x, pt.y + BLOCK)


class SnakeGame:
    def __init__(self, w=480, h=480, render=True, speed=15):
        self.w = w
        self.h = h
        self.do_render = render
        self.speed = speed

        if self.do_render:
            pygame.init()
            self.screen = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake AI')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('arial', 18)
        else:
            self.screen = None

        self.reset()


    def reset(self):
        cx = (self.w // (2 * BLOCK)) * BLOCK
        cy = (self.h // (2 * BLOCK)) * BLOCK

        self.direction = Direction.RIGHT
        self.head = Point(cx, cy)
        self.snake = deque([
            self.head,
            Point(cx - BLOCK, cy),
            Point(cx - 2 * BLOCK, cy),
        ])

        self.score = 0
        self.steps = 0
        self.overlay = None
        self._place_food()
        return self.get_state()

    def get_state(self):
        h = self.head
        d = self.direction

        pt_str = _move_point(h, d)
        pt_r   = _move_point(h, _turn_right(d))
        pt_l   = _move_point(h, _turn_left(d))

        state = [
            # Danger in relative directions
            int(self._is_collision(pt_str)),
            int(self._is_collision(pt_r)),
            int(self._is_collision(pt_l)),
            # Current heading (one-hot)
            int(d == Direction.LEFT),
            int(d == Direction.RIGHT),
            int(d == Direction.UP),
            int(d == Direction.DOWN),
            # Food location relative to head
            int(self.food.x < h.x),   # food left
            int(self.food.x > h.x),   # food right
            int(self.food.y < h.y),   # food above  (y-axis: 0 = top)
            int(self.food.y > h.y),   # food below
        ]
        return state  # list of 11 ints (0 or 1)

    def step(self, action):
        """
        action: 0 = go straight, 1 = turn right, 2 = turn left
        returns: (reward, done, score)
          reward: +10 food, -10 death, -0.1 per step
        """
        self.steps += 1

        if self.do_render:
            self._handle_events()

        # Update direction
        if action == 1:
            self.direction = _turn_right(self.direction)
        elif action == 2:
            self.direction = _turn_left(self.direction)

        # Advance head
        self.head = _move_point(self.head, self.direction)
        self.snake.appendleft(self.head)

        # Collision check
        if self._is_collision():
            if self.do_render:
                self._draw(self.overlay)
            return -10, True, self.score

        # Food check
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
            reward = -0.1

        # Step-limit guard: prevents infinite circling
        if self.steps > 200 * len(self.snake):
            if self.do_render:
                self._draw(self.overlay)
            return reward, True, self.score

        if self.do_render:
            self._draw(self.overlay)

        return reward, False, self.score

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x < 0 or pt.x >= self.w or pt.y < 0 or pt.y >= self.h:
            return True
        snake_list = list(self.snake)
        return pt in set(snake_list[1:]) # still O(n) build but O(1) lookup

    def _place_food(self):
        snake_set = set(self.snake)
        while True:
            x = random.randint(0, (self.w - BLOCK) // BLOCK) * BLOCK
            y = random.randint(0, (self.h - BLOCK) // BLOCK) * BLOCK
            food = Point(x, y)
            if food not in snake_set:
                self.food = food
                return

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

    def _draw(self, overlay=None):
        self.screen.fill(BLACK)

        # Subtle grid
        for x in range(0, self.w, BLOCK):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, self.h))
        for y in range(0, self.h, BLOCK):
            pygame.draw.line(self.screen, GRAY, (0, y), (self.w, y))

        # Snake body
        for i, pt in enumerate(self.snake):
            col = GREEN1 if i == 0 else GREEN2
            pygame.draw.rect(self.screen, col,
                             (pt.x + 1, pt.y + 1, BLOCK - 2, BLOCK - 2),
                             border_radius=3)

        # Eyes on head
        head = self.snake[0]
        ex, ey = head.x + BLOCK // 2, head.y + BLOCK // 2
        pygame.draw.circle(self.screen, WHITE, (ex - 3, ey - 3), 3)
        pygame.draw.circle(self.screen, WHITE, (ex + 3, ey - 3), 3)

        # Food
        pygame.draw.rect(self.screen, RED,
                         (self.food.x + 1, self.food.y + 1, BLOCK - 2, BLOCK - 2),
                         border_radius=4)

        # HUD
        score_txt = self.font.render(
            f'Score: {self.score}   Steps: {self.steps}', True, WHITE)
        self.screen.blit(score_txt, (5, 5))

        # Optional AI overlay (dict passed from train loop)
        if overlay:
            lines = [
                f"Gen: {overlay.get('gen', '-')}",
                f"Best fitness: {overlay.get('best_fitness', '-')}",
                f"All-time best: {overlay.get('alltime_best', '-')}",
            ]
            for i, line in enumerate(lines):
                txt = self.font.render(line, True, BLUE)
                self.screen.blit(txt, (self.w - 200, 5 + i * 22))

        pygame.display.flip()
        self.clock.tick(self.speed)



def _direction_to_action(current: Direction, intended: Direction) -> int:
    ci = _CW.index(current)
    ii = _CW.index(intended)
    diff = (ii - ci) % 4
    if diff == 1:
        return 1   # turn right
    if diff == 3:
        return 2   # turn left
    return 0       # straight (or 180° → treat as straight)


def human_play():
    game = SnakeGame(render=True, speed=10)
    intended = Direction.RIGHT

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_LEFT,  pygame.K_a): intended = Direction.LEFT
                if event.key in (pygame.K_RIGHT, pygame.K_d): intended = Direction.RIGHT
                if event.key in (pygame.K_UP,    pygame.K_w): intended = Direction.UP
                if event.key in (pygame.K_DOWN,  pygame.K_s): intended = Direction.DOWN
                if event.key == pygame.K_q:
                    pygame.quit()
                    return

        action = _direction_to_action(game.direction, intended)
        _, done, score = game.step(action)

        if done:
            print(f'Game over! Score: {score}  Steps: {game.steps}')
            game.reset()
            intended = Direction.RIGHT


if __name__ == '__main__':
    human_play()
