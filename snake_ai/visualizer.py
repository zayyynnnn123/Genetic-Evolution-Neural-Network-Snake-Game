import pygame
from game import SnakeGame, BLOCK, BLACK, GREEN1, GREEN2, RED, WHITE, GRAY

COLS, ROWS = 3, 2
CELL   = 240                       # each mini-game is 240×240 px
WIN_W  = COLS * CELL               # 720
WIN_H  = ROWS * CELL + 30          # 480 + status bar

SCALE = CELL / 480                 # 0.5 — game coords → cell coords
SB    = max(1, int(BLOCK * SCALE)) # 10  — scaled block size

GOLD        = (255, 200, 50)
DIM_HEAD    = (30,  80,  30)
DIM_BODY    = (20,  55,  20)
BORDER_BEST = GOLD
BORDER_NORM = (50,  50,  50)
BAR_BG      = (18,  18,  18)


def _draw_cell(surf, font, col, row, game, label, is_best, is_dead):
    ox, oy = col * CELL, row * CELL

    pygame.draw.rect(surf, BLACK, (ox, oy, CELL, CELL))

    for x in range(0, CELL + 1, SB):
        pygame.draw.line(surf, GRAY, (ox + x, oy), (ox + x, oy + CELL))
    for y in range(0, CELL + 1, SB):
        pygame.draw.line(surf, GRAY, (ox, oy + y), (ox + CELL, oy + y))

    hi = DIM_HEAD if is_dead else GREEN1
    lo = DIM_BODY if is_dead else GREEN2
    fc = DIM_BODY if is_dead else RED

    for i, pt in enumerate(game.snake):
        sx = ox + int(pt.x * SCALE)
        sy = oy + int(pt.y * SCALE)
        pygame.draw.rect(surf, hi if i == 0 else lo,
                         (sx + 1, sy + 1, SB - 2, SB - 2))

    fx = ox + int(game.food.x * SCALE)
    fy = oy + int(game.food.y * SCALE)
    pygame.draw.rect(surf, fc, (fx + 1, fy + 1, SB - 2, SB - 2))

    pygame.draw.rect(surf, BORDER_BEST if is_best else BORDER_NORM,
                     (ox, oy, CELL, CELL), 2)

    txt = font.render(f"{label}   score:{game.score}", True,
                      GOLD if is_best else WHITE)
    surf.blit(txt, (ox + 4, oy + 4))


def show_grid(current_five, prev_best_nn, gen, best_fit, alltime_best_fit, speed=40):
    """
    Renders 6 snakes simultaneously in a 3×2 grid.
      slot 0 (gold border) : prev_best_nn  — best agent from the previous generation
      slots 1-5            : current_five  — top 5 agents from the current generation
    Blocks until all 6 snakes are dead, then returns.
    """
    if not pygame.get_init():
        pygame.init()
    if not pygame.font.get_init():
        pygame.font.init()

    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption(f"Snake AI — Generation {gen}")
    clock  = pygame.time.Clock()
    font_s = pygame.font.SysFont("arial", 11)
    font_h = pygame.font.SysFont("arial", 13)

    all_nns = [prev_best_nn] + list(current_five[:5])
    labels  = ["PREV BEST"] + [f"Agent {i + 1}" for i in range(5)]
    bests   = [True] + [False] * 5

    games = []
    for nn in all_nns:
        g = SnakeGame(render=False)
        g.reset()
        games.append(g)

    dead = [False] * 6

    while not all(dead):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        for i in range(6):
            if dead[i]:
                continue
            state  = games[i].get_state()
            action = all_nns[i].predict(state)
            _, gone, _ = games[i].step(action)
            if gone:
                dead[i] = True

        screen.fill((12, 12, 12))

        for i in range(6):
            _draw_cell(screen, font_s,
                       i % COLS, i // COLS,
                       games[i], labels[i], bests[i], dead[i])

        bar_y = ROWS * CELL
        pygame.draw.rect(screen, BAR_BG, (0, bar_y, WIN_W, 30))
        status = (f"Gen {gen}   "
                  f"Best: {best_fit:.0f}   "
                  f"All-time: {alltime_best_fit:.0f}   "
                  f"alive: {6 - sum(dead)}/6")
        screen.blit(font_h.render(status, True, (150, 150, 150)), (8, bar_y + 8))

        pygame.display.flip()
        clock.tick(speed)

    pygame.time.wait(600)
