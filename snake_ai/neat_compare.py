"""
neat_compare.py
Experiment 5 — NEAT comparison baseline for the Snake neuroevolution project.

Trains a NEAT agent using neat-python for 180 generations with pop_size=150,
the same fitness function as the GA, and the same 11-input / 3-output interface.
Unlike the fixed-topology GA, NEAT starts with no hidden nodes and evolves
both weights AND topology (add/remove nodes and connections) via speciation.

Logs results to neat_training_log.csv — same columns as training_log.csv plus
three NEAT-specific columns: num_species, avg_nodes, avg_connections.

Run from snake_ai/:
    python neat_compare.py
"""

import os
import csv
import pickle
import numpy as np

try:
    import neat
    from neat.reporting import BaseReporter as _BaseReporter
except ImportError:
    raise SystemExit("neat-python not installed.  Run:  pip install neat-python")

from game import SnakeGame
from visualizer import show_grid

GENERATIONS  = 180
RENDER_EVERY = 5     # show grid every N gens (0 = never)
RENDER_SPEED = 4000  # fps during live preview
_DIR         = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH  = os.path.join(_DIR, "config-neat.txt")
SAVE_PATH    = os.path.join(_DIR, "best_neat.pkl")
LOG_PATH     = os.path.join(_DIR, "neat_training_log.csv")


class _NetWrapper:
    """Wraps a neat FeedForwardNetwork to match the .predict() interface show_grid expects."""
    def __init__(self, net):
        self._net = net

    def predict(self, state):
        return int(np.argmax(self._net.activate(list(state))))


def _fitness(score: int, steps: int) -> float:
    """Same formula as train.py so results are directly comparable."""
    return score * 100 + steps * 0.1


# ── Genome evaluation ─────────────────────────────────────────────────────────
def eval_genomes(genomes, config):
    """
    Evaluate every genome in the population for one generation.
    Sets genome.fitness (required by neat-python) and genome.score
    (food eaten — used by the reporter to log the raw game metric).
    """
    for _genome_id, genome in genomes:
        net   = neat.nn.FeedForwardNetwork.create(genome, config)
        game  = SnakeGame(render=False)
        state = game.reset()
        while True:
            action = int(np.argmax(net.activate(list(state))))
            _, done, _ = game.step(action)
            if done:
                break
            state = game.get_state()
        genome.score   = game.score
        genome.fitness = _fitness(game.score, game.steps)


# ── Per-generation reporter ───────────────────────────────────────────────────
class _NeatLogger(_BaseReporter):
    """
    Logs one CSV row per generation and prints a summary line that mirrors
    the format used by train.py so the two outputs can be compared directly.
    Also saves best_neat.pkl whenever a new all-time best is found.
    """

    def __init__(self, config):
        self._gen          = 0
        self._config       = config
        self.alltime_best  = -float("inf")
        self._prev_best_nn = None
        self._top5_nns     = []
        self._f = open(LOG_PATH, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow([
            "generation", "best_fitness", "avg_fitness", "best_score",
            "alltime_best", "num_species", "avg_nodes", "avg_connections",
        ])

    # ── neat-python reporter hooks ────────────────────────────────────────────

    def start_generation(self, generation):
        self._gen = generation

    def post_evaluate(self, config, population, species, best_genome):
        fits      = [g.fitness for g in population.values()]
        best_fit  = float(best_genome.fitness)
        avg_fit   = float(np.mean(fits))
        best_score = int(getattr(best_genome, "score", 0))

        # NEAT-specific topology stats
        n_species = len(species.species)
        avg_nodes = float(np.mean([len(g.nodes) for g in population.values()]))
        avg_conns = float(np.mean([
            sum(1 for c in g.connections.values() if c.enabled)
            for g in population.values()
        ]))

        # build wrappers for top-5 genomes this generation
        sorted_genomes = sorted(population.values(), key=lambda g: g.fitness, reverse=True)
        self._top5_nns = [
            _NetWrapper(neat.nn.FeedForwardNetwork.create(g, self._config))
            for g in sorted_genomes[:5]
        ]

        if best_fit > self.alltime_best:
            self.alltime_best = best_fit
            with open(SAVE_PATH, "wb") as f:
                pickle.dump(best_genome, f)
            print(f"  *** saved  fitness={best_fit:.1f}  score={best_score}")

        print(
            f"gen {self._gen:4d} | best={best_fit:8.1f} | avg={avg_fit:8.1f} | "
            f"score={best_score:3d} | alltime={self.alltime_best:.1f} | "
            f"species={n_species:2d} | nodes={avg_nodes:4.1f} | conns={avg_conns:5.1f}"
        )

        self._w.writerow([
            self._gen,
            round(best_fit,  2), round(avg_fit,  2),
            best_score,          round(self.alltime_best, 2),
            n_species,           round(avg_nodes, 2), round(avg_conns, 2),
        ])
        self._f.flush()

        if RENDER_EVERY and self._gen % RENDER_EVERY == 0 and self._prev_best_nn is not None:
            show_grid(self._top5_nns, self._prev_best_nn,
                      self._gen, best_fit, self.alltime_best, RENDER_SPEED)

        self._prev_best_nn = _NetWrapper(
            neat.nn.FeedForwardNetwork.create(best_genome, self._config)
        )

    def complete_extinction(self):
        print("  [!] All species extinct — neat-python will reset.")

    def close(self):
        self._f.close()


# ── Entry point ───────────────────────────────────────────────────────────────
def run():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            f"NEAT config not found: {CONFIG_PATH}\n"
            "Make sure config-neat.txt is in the same directory."
        )

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH,
    )

    pop    = neat.Population(config)
    logger = _NeatLogger(config)
    pop.add_reporter(logger)

    print(f"NEAT training — {GENERATIONS} generations, pop={config.pop_size}")
    print(f"Topology: starts minimal (11 inputs, 0 hidden, 3 outputs)")
    print(f"Fitness : score * 100 + steps * 0.1  (same as GA baseline)\n")

    winner = pop.run(eval_genomes, GENERATIONS)
    logger.close()

    final_score = int(getattr(winner, "score", 0))
    print(f"\nDone.")
    print(f"  All-time best fitness : {logger.alltime_best:.1f}")
    print(f"  Winner score          : {final_score}")
    print(f"  Winner nodes          : {len(winner.nodes)}")
    print(f"  Winner connections    : {sum(1 for c in winner.connections.values() if c.enabled)}")
    print(f"  Results saved to      : {LOG_PATH}")
    return winner


if __name__ == "__main__":
    run()
