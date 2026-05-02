import os
import csv
import pickle
import numpy as np
from neural_net import NeuralNetwork
from game import SnakeGame
from visualizer import show_grid


POP_SIZE      = 150
GENERATIONS   = 180
ELITE_FRAC    = 0.10   # top 10% copied unchanged each gen
MUTATION_RATE = 0.15   # per-weight probability of mutation
MUTATION_STR  = 0.20   # std of gausian noise applied
LAYER_SIZES   = [11, 16, 3]
SAVE_PATH     = "best_snake.pkl"
LOG_PATH      = "training_log.csv"
RENDER_EVERY  = 5      # show best agent live every N gens (0 = never)
RENDER_SPEED  = 4000   # fps during live preview!



def evaluate(nn: NeuralNetwork) -> tuple:
    game = SnakeGame(render=False)
    state = game.reset()
    while True:
        action = nn.predict(state)
        _, done, _ = game.step(action)
        if done:
            break
        state = game.get_state()
    return game.score, game.steps


def fitness(score: int, steps: int) -> float:
    return score * 100 + steps * 0.1


def tournament_select(pop: list, fits: list, k: int = 5) -> NeuralNetwork:
    idxs = np.random.choice(len(pop), k, replace=False)
    best = idxs[np.argmax([fits[i] for i in idxs])]
    return pop[best]


def crossover(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    mask = np.random.rand(len(a)) < 0.5
    return np.where(mask, a, b)


def mutate(flat: np.ndarray) -> np.ndarray:
    mask = np.random.rand(len(flat)) < MUTATION_RATE
    return flat + mask * np.random.randn(len(flat)) * MUTATION_STR


def make_child(pop: list, fits: list) -> NeuralNetwork:
    p1 = tournament_select(pop, fits)
    p2 = tournament_select(pop, fits)
    child_flat = mutate(crossover(p1.get_weights_flat(), p2.get_weights_flat()))
    child = NeuralNetwork(LAYER_SIZES)
    child.set_weights_flat(child_flat)
    return child


def run() -> NeuralNetwork:
    pop = [NeuralNetwork(LAYER_SIZES) for _ in range(POP_SIZE)]

    alltime_best_fit = -np.inf
    alltime_best_nn  = None
    prev_best_nn     = None
    n_elite = max(1, int(POP_SIZE * ELITE_FRAC))

    with open(LOG_PATH, 'w', newline='') as log_f:
        log_w = csv.writer(log_f)
        log_w.writerow(['generation', 'best_fitness', 'avg_fitness', 'best_score', 'alltime_best'])

        for gen in range(1, GENERATIONS + 1):
            results = [evaluate(nn) for nn in pop]
            fits    = [fitness(s, t) for s, t in results]

            order      = np.argsort(fits)[::-1]
            best_idx   = int(order[0])
            best_fit   = fits[best_idx]
            best_score = results[best_idx][0]

            if best_fit > alltime_best_fit:
                alltime_best_fit = best_fit
                alltime_best_nn  = pop[best_idx].copy()
                with open(SAVE_PATH, 'wb') as f:
                    pickle.dump(alltime_best_nn, f)
                print(f"  *** saved  fitness={alltime_best_fit:.1f}  score={best_score}")

            avg_fit = np.mean(fits)
            print(f"gen {gen:4d} | best={best_fit:8.1f} | avg={avg_fit:8.1f} | "
                  f"score={best_score:3d} | alltime={alltime_best_fit:.1f}")

            log_w.writerow([gen, round(best_fit, 2), round(avg_fit, 2), best_score,
                            round(alltime_best_fit, 2)])
            log_f.flush()

            if RENDER_EVERY and gen % RENDER_EVERY == 0 and prev_best_nn is not None:
                current_five = [pop[order[i]].copy() for i in range(min(5, len(order)))]
                show_grid(current_five, prev_best_nn,
                          gen, best_fit, alltime_best_fit, RENDER_SPEED)

            prev_best_nn = pop[best_idx].copy()

            next_pop = [pop[i].copy() for i in order[:n_elite]]
            while len(next_pop) < POP_SIZE:
                next_pop.append(make_child(pop, fits))
            pop = next_pop

    print("Done.")
    return alltime_best_nn


if __name__ == '__main__':
    run()
