#VIDEO:
https://youtu.be/W2reMDvzU_A


# Genetic Evolution Neural Network — Snake Game

A neuroevolution project that trains feedforward neural networks to play Snake using a custom **Genetic Algorithm (GA)** and compares it against **NEAT** (NeuroEvolution of Augmenting Topologies). The neural network is built entirely from scratch in Python — no PyTorch, no TensorFlow.

After 180 generations the GA agent reaches scores of **250+**, a ~260× improvement over a random baseline.

---

## Demo

The visualizer runs 6 agents simultaneously during training, showing the previous generation's best (gold border) alongside the current top 5:

| Generation 1 (random) | Generation 180 (evolved) |
|---|---|
| Avg score ≈ 0–1 | Best score ≈ 257 |

Key breakthroughs during training:

| Generation | Score milestone |
|---|---|
| 48 | 13 |
| 73 | 40 |
| 91 | 101 |
| 132 | 204 |
| 165 | 257 |

---

## How It Works

### Neural Network

A fully custom feedforward network (`neural_net.py`) with no external ML frameworks:

- **Architecture**: 11 inputs → 16 hidden (ReLU) → 3 outputs
- **Inputs (11 features)**:
  - 3 danger sensors (collision ahead / left / right)
  - 4 direction flags (one-hot current heading)
  - 4 food direction flags (relative position)
- **Outputs**: straight / turn right / turn left

### Genetic Algorithm

Each generation, 150 agents play Snake independently. The fittest weights are selected and used to breed the next generation:

| Parameter | Value |
|---|---|
| Population | 150 |
| Generations | 180 |
| Elite fraction | 10% (copied unchanged) |
| Selection | Tournament (k=5) |
| Crossover | Uniform weight crossover |
| Mutation rate | 15% of weights |
| Mutation strength | Gaussian noise σ=0.2 |
| Fitness | `score × 100 + steps × 0.1` |

### NEAT Comparison

`neat_compare.py` trains a NEAT population with the identical fitness function. Unlike the fixed-topology GA, NEAT evolves both weights and network structure (nodes and connections are added/removed over time).

**Result**: The fixed-topology GA outperforms NEAT (score 259 vs lower), likely because the hand-crafted 11-feature state is already sufficient — NEAT's topology search adds complexity without benefit on this problem.

---

## Project Structure

```
snake_ai/
├── game.py             # Snake game engine (Pygame, 480×480, human-playable)
├── neural_net.py       # Custom feedforward NN from scratch
├── train.py            # GA training loop
├── neat_compare.py     # NEAT training for comparison
├── visualizer.py       # 6-agent real-time grid visualization
├── plot_training.py    # 6 analysis/research figures
└── config-neat.txt     # NEAT-python hyperparameter config
best_snake.pkl          # Pre-trained GA model
plots/                  # Generated analysis figures
```

---

## Getting Started

**Install dependencies:**

```bash
pip install numpy pygame matplotlib neat-python
```

**Train the GA agent:**

```bash
cd snake_ai
python train.py
```

Runs 180 generations, saves the best model to `best_snake.pkl`, logs metrics to `training_log.csv`, and optionally renders a 6-agent visualization every 5 generations (set `RENDER_EVERY = 5` in `train.py`).

**Train NEAT for comparison:**

```bash
cd snake_ai
python neat_compare.py
```

**Generate analysis figures:**

```bash
cd snake_ai
python plot_training.py
```

Reads the training CSV logs and writes 6 charts to `plots/`. Falls back to embedded baseline data if no CSV is present.

**Play manually:**

```bash
cd snake_ai
python game.py
```

Arrow keys or WASD to move. Q to quit.

---

## Configuration

Key hyperparameters in `train.py`:

```python
POP_SIZE      = 150       # Agents per generation
GENERATIONS   = 180       # Training generations
ELITE_FRAC    = 0.10      # Fraction copied unchanged
MUTATION_RATE = 0.15      # Fraction of weights mutated per agent
MUTATION_STR  = 0.20      # Gaussian noise std dev
LAYER_SIZES   = [11, 16, 3]
RENDER_EVERY  = 5         # 0 = no visualization during training
```

---

## Analysis Plots

`plot_training.py` generates 6 figures:

1. Fitness learning curves (GA vs NEAT, best + mean per generation)
2. Score progression with smoothed trend line
3. All-time best score staircase (breakthrough moments)
4. Final performance bar chart (GA vs Random vs NEAT)
5. Convergence speed (generations to reach score thresholds)
6. Combined 2×3 overview panel

---

## Requirements

- Python 3.8+
- numpy
- pygame
- matplotlib
- neat-python (only needed for `neat_compare.py`)

---

## Author

Made by **Zain Aleem**
