"""
generate_report.py
Generates snake_ai_report.pdf — complete academic report with embedded figures.
Run:  python generate_report.py
"""

import os, sys, subprocess

_DIR   = os.path.dirname(os.path.abspath(__file__))
PLOTS  = os.path.join(_DIR, "plots")
OUTPUT = os.path.join(_DIR, "snake_ai_report.pdf")

try:
    import reportlab
except ImportError:
    print("Installing reportlab …")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak, KeepTogether, Preformatted,
    HRFlowable,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

# ── Page geometry ─────────────────────────────────────────────────────────────
W, H   = A4
ML = MR = 2.5 * cm
MT = MB = 2.0 * cm
BW      = W - ML - MR   # usable body width ≈ 453 pts

# ── Colour palette ────────────────────────────────────────────────────────────
DARK_BLUE  = colors.HexColor("#1565C0")
MID_BLUE   = colors.HexColor("#1976D2")
LIGHT_BLUE = colors.HexColor("#E3F2FD")
AMBER      = colors.HexColor("#F57C00")
GREEN_D    = colors.HexColor("#2E7D32")
DARK_GRAY  = colors.HexColor("#212121")
MID_GRAY   = colors.HexColor("#757575")
LIGHT_GRAY = colors.HexColor("#EEEEEE")
CODE_BG    = colors.HexColor("#F5F5F5")
CODE_BORDER= colors.HexColor("#BDBDBD")
TABLE_HEAD = colors.HexColor("#1565C0")
TABLE_ALT  = colors.HexColor("#F5F7FF")
RED_D      = colors.HexColor("#C62828")

# ── Style definitions ─────────────────────────────────────────────────────────
SS = getSampleStyleSheet()

def ps(name, parent="Normal", **kw):
    return ParagraphStyle(name, parent=SS[parent], **kw)

TITLE_S    = ps("RT", fontSize=24, leading=30, textColor=DARK_BLUE,
                alignment=TA_CENTER, spaceAfter=4, fontName="Helvetica-Bold")
SUBTITLE_S = ps("RS", fontSize=13, leading=18, textColor=MID_GRAY,
                alignment=TA_CENTER, spaceAfter=4)
META_S     = ps("RM", fontSize=10, leading=14, textColor=MID_GRAY,
                alignment=TA_CENTER, spaceAfter=2)
H1_S       = ps("H1", fontSize=15, leading=20, textColor=DARK_BLUE,
                spaceBefore=16, spaceAfter=6, fontName="Helvetica-Bold")
H2_S       = ps("H2", fontSize=12, leading=16, textColor=MID_BLUE,
                spaceBefore=10, spaceAfter=4, fontName="Helvetica-Bold")
H3_S       = ps("H3", fontSize=10, leading=14, textColor=DARK_GRAY,
                spaceBefore=7, spaceAfter=3, fontName="Helvetica-Bold")
BODY_S     = ps("BD", fontSize=10, leading=15, textColor=DARK_GRAY,
                alignment=TA_JUSTIFY, spaceAfter=5)
CAPTION_S  = ps("CA", fontSize=8.5, leading=12, textColor=MID_GRAY,
                alignment=TA_CENTER, spaceAfter=10,
                fontName="Helvetica-Oblique")
BULLET_S   = ps("BU", fontSize=10, leading=14, textColor=DARK_GRAY,
                leftIndent=14, spaceAfter=3)
CODE_S     = ParagraphStyle("CO", fontName="Courier", fontSize=7.8,
                leading=10.5, textColor=DARK_GRAY)
ABSTRACT_S = ps("AB", fontSize=10, leading=15, textColor=DARK_GRAY,
                alignment=TA_JUSTIFY,
                leftIndent=1*cm, rightIndent=1*cm, spaceAfter=6)

# ── Flowable helpers ──────────────────────────────────────────────────────────
def P(txt, style=BODY_S):
    return Paragraph(txt, style)

def SP(n=0.3):
    return Spacer(1, n * cm)

def HR(color=MID_BLUE, thickness=0.5):
    return HRFlowable(width="100%", thickness=thickness, color=color,
                      spaceAfter=4, spaceBefore=4)

def code_block(text):
    """Monospaced code block with gray background."""
    pre = Preformatted(text, CODE_S)
    t = Table([[pre]], colWidths=[BW * 0.97])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), CODE_BG),
        ("BOX",           (0,0), (-1,-1), 0.6, CODE_BORDER),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 4),
    ]))
    return t

def data_table(headers, rows, col_widths=None):
    """Styled two-tone data table."""
    data = [headers] + rows
    cw   = col_widths or [BW / len(headers)] * len(headers)
    t    = Table(data, colWidths=cw)
    style = [
        ("BACKGROUND",  (0,0), (-1,0), TABLE_HEAD),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("LEADING",     (0,0), (-1,-1), 13),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("ALIGN",       (0,1), (0,-1), "LEFT"),
        ("BOX",         (0,0), (-1,-1), 0.5, MID_BLUE),
        ("INNERGRID",   (0,0), (-1,-1), 0.3, colors.HexColor("#CFD8DC")),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING",(0,0), (-1,-1), 6),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0,i), (-1,i), TABLE_ALT))
    t.setStyle(TableStyle(style))
    return t

def fig_block(fname, caption, aspect, w_frac=0.94):
    """Image + italic caption. aspect = height/width of the figure."""
    path = os.path.join(PLOTS, fname)
    fw   = BW * w_frac
    fh   = fw * aspect
    items = []
    if os.path.exists(path):
        items.append(Image(path, width=fw, height=fh))
    else:
        items.append(P(f"[figure not found: {fname}]", CAPTION_S))
    items.append(P(caption, CAPTION_S))
    return items

# ── Banner helper (coloured box) ──────────────────────────────────────────────
def banner(text, bg=DARK_BLUE, fg=colors.white, font_size=11):
    cell = Paragraph(f"<b>{text}</b>",
                     ParagraphStyle("bn", fontName="Helvetica-Bold",
                                    fontSize=font_size, textColor=fg,
                                    alignment=TA_CENTER))
    t = Table([[cell]], colWidths=[BW])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), bg),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
    ]))
    return t

# ─────────────────────────────────────────────────────────────────────────────
# BUILD CONTENT
# ─────────────────────────────────────────────────────────────────────────────
story = []

# ═══════════════════════════════════════════════════════════════
# TITLE PAGE
# ═══════════════════════════════════════════════════════════════
story += [
    SP(2.5),
    banner("SNAKE AI", bg=DARK_BLUE, font_size=28),
    SP(0.4),
    banner("Neuroevolutionary Agent Training via Genetic Algorithms",
           bg=MID_BLUE, font_size=13),
    SP(1.5),
    P("Experiment 1 — Baseline GA  |  Experiment 5 — NEAT Comparison", SUBTITLE_S),
    SP(0.6),
    HR(DARK_BLUE, 1.2),
    SP(0.6),
    P("Institute of Business Administration (IBA), Karachi", META_S),
    P("Supervisor: Dr. Syed Ali Raza", META_S),
    P("Submitted: May 2026", META_S),
    SP(2),
    HR(MID_BLUE, 0.5),
    SP(0.5),
]

# Summary box on title page
summary_rows = [
    ["All-time best score",  "259  (gen 166)"],
    ["Peak best fitness",    "29,571  (gen 165)"],
    ["Mean fitness (gen 1)", "30.4  (random baseline)"],
    ["Mean fitness (gen 180)","7,910  (263× improvement)"],
    ["Score ≥ 100 reached",  "Generation 91"],
    ["Score ≥ 200 reached",  "Generation 132"],
    ["Network architecture", "[11 → 16 → 3]  (243 parameters)"],
    ["Population / Gens",    "150 agents / 180 generations"],
]
story.append(data_table(
    ["Key Result", "Value"],
    summary_rows,
    col_widths=[BW * 0.52, BW * 0.48]
))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════
# ABSTRACT
# ═══════════════════════════════════════════════════════════════
story += [
    banner("Abstract", bg=MID_BLUE, font_size=11),
    SP(0.3),
    P(
        "We present a neuroevolutionary approach to training a Snake game agent using a "
        "fixed-topology Genetic Algorithm (GA). A feedforward neural network with architecture "
        "[11 → 16 → 3] serves as the agent policy, receiving 11 binary inputs encoding "
        "proximity danger, current heading, and food direction. The GA runs 180 generations "
        "over a population of 150 agents, achieving an all-time best score of <b>259 food "
        "items</b> (fitness 29,571) by generation 166. Population mean fitness grew from 30.4 "
        "at generation 1 to 7,910 by generation 180 — a <b>263× improvement</b>, demonstrating "
        "genuine population-wide learning rather than just a single lucky individual. "
        "Score milestones of 50, 100, and 200 were reached at generations 77, 91, and 132 "
        "respectively, with learning exhibiting a clear punctuated-equilibrium pattern: long "
        "plateaus interrupted by rapid capability jumps. "
        "We additionally implement a NEAT (NeuroEvolution of Augmenting Topologies) comparison "
        "agent using the same fitness formula and generation budget. Six configuration-level "
        "reasons are identified for why the fixed-topology GA is expected to outperform NEAT "
        "on this well-specified task.", ABSTRACT_S),
    SP(0.4),
]

# ═══════════════════════════════════════════════════════════════
# 1. INTRODUCTION
# ═══════════════════════════════════════════════════════════════
story += [
    P("1.  Introduction", H1_S), HR(),
    P(
        "The game of Snake presents a deceptively challenging control problem: an agent must "
        "navigate a grid, consuming food to grow its body, while avoiding collisions with walls "
        "and its own tail. As the body grows, the state space expands exponentially, and naive "
        "search-based strategies quickly become intractable. Classical reinforcement learning "
        "(RL) approaches require careful reward shaping and suffer from sparse reward signals — "
        "the agent receives positive feedback only when food is eaten, with all intermediate "
        "steps yielding near-zero gradients.", BODY_S),
    P(
        "Neuroevolution sidesteps the credit-assignment problem entirely by evolving network "
        "weights directly. Rather than computing gradients, a population of agents is evaluated "
        "over complete episodes; the fittest individuals reproduce and their traits propagate "
        "to the next generation via crossover and mutation. This approach is robust to "
        "non-differentiable reward signals, does not require a replay buffer, and naturally "
        "explores diverse behavioural strategies in parallel.", BODY_S),
    P(
        "This report documents <b>Experiment 1</b> — the baseline GA run — and the "
        "<b>Experiment 5</b> NEAT comparison implementation. The fitness function used "
        "throughout is:", BODY_S),
    SP(0.1),
    code_block("fitness(score, steps) = score × 100 + steps × 0.1"),
    SP(0.2),
    P(
        "The <i>steps × 0.1</i> term provides a weak survival incentive, preventing the agent "
        "from dying immediately while keeping score as the dominant signal.", BODY_S),
    SP(0.3),
]

# ═══════════════════════════════════════════════════════════════
# 2. SYSTEM ARCHITECTURE
# ═══════════════════════════════════════════════════════════════
story += [
    P("2.  System Architecture", H1_S), HR(),
    P("2.1  Game Environment  (game.py)", H2_S),
    P(
        "The Snake environment is implemented in Pygame on a 480 × 480 pixel grid with "
        "20 px blocks, yielding a 24 × 24 cell playing field. The agent selects from three "
        "relative actions each step: <b>0</b> = go straight, <b>1</b> = turn right, "
        "<b>2</b> = turn left. An episode terminates on wall or self-collision, or when "
        "steps exceed 200 × snake length (an anti-loop safeguard).", BODY_S),
    P("<b>State representation — 11 binary inputs:</b>", H3_S),
]

state_rows = [
    ["0",   "Danger straight",  "1 if the cell directly ahead is lethal"],
    ["1",   "Danger right",     "1 if the cell to the right is lethal"],
    ["2",   "Danger left",      "1 if the cell to the left is lethal"],
    ["3–6", "Current heading",  "One-hot: [LEFT, RIGHT, UP, DOWN]"],
    ["7–10","Food direction",   "One-hot: [food_left, food_right, food_above, food_below]"],
]
story.append(data_table(
    ["Index", "Feature", "Description"],
    state_rows,
    col_widths=[BW*0.12, BW*0.24, BW*0.64]
))
story.append(SP(0.3))

story += [
    P("The <tt>get_state()</tt> method that produces these 11 values:", BODY_S),
    code_block(
"""\
def get_state(self):
    h, d = self.head, self.direction

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
        int(self.food.y < h.y),   # food above
        int(self.food.y > h.y),   # food below
    ]
    return state   # 11 ints, all 0 or 1"""
    ),
    SP(0.4),
    P("Step rewards: <b>+10</b> food eaten, <b>−10</b> death, <b>−0.1</b> per step survived.", BODY_S),
    SP(0.3),
    P("2.2  Neural Network  (neural_net.py)", H2_S),
    P(
        "The agent policy is a minimal feedforward network: "
        "<b>11 inputs → 16 ReLU hidden units → 3 linear outputs</b>. "
        "The output with the highest logit is selected as the action (argmax). "
        "Total trainable parameters: 11×16 + 16 + 16×3 + 3 = <b>243</b>. "
        "Weights are initialised from N(0, 0.5). "
        "A flat vector representation enables direct GA weight manipulation.", BODY_S),
    code_block(
"""\
class NeuralNetwork:
    def __init__(self, layer_sizes=[11, 16, 3]):
        self.layer_sizes = layer_sizes
        self.weights, self.biases = [], []
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.5
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = self.relu(x)
        return x

    def predict(self, state):
        x = np.array([state], dtype=float)
        return int(np.argmax(self.forward(x)))   # 0, 1, or 2

    def get_weights_flat(self):   # flatten all weights+biases to 1-D array
        parts = [w.flatten() for w in self.weights] + [b.flatten() for b in self.biases]
        return np.concatenate(parts)

    def set_weights_flat(self, flat):   # inverse of get_weights_flat
        idx = 0
        for i, w in enumerate(self.weights):
            self.weights[i] = flat[idx:idx+w.size].reshape(w.shape); idx += w.size
        for i, b in enumerate(self.biases):
            self.biases[i]  = flat[idx:idx+b.size].reshape(b.shape); idx += b.size"""
    ),
    SP(0.3),
    PageBreak(),
]

# ═══════════════════════════════════════════════════════════════
# 3. GENETIC ALGORITHM
# ═══════════════════════════════════════════════════════════════
story += [
    P("3.  Genetic Algorithm  (train.py)", H1_S), HR(),
    P("3.1  Hyperparameters", H2_S),
]

hp_rows = [
    ["Population size",    "150",    "Agents evaluated per generation"],
    ["Generations",        "180",    "Total training iterations"],
    ["Elite fraction",     "10 %",   "Top 15 agents copied unchanged each gen"],
    ["Mutation rate",      "0.15",   "Per-weight probability of mutation"],
    ["Mutation strength",  "0.20",   "Gaussian noise std applied to mutated weights"],
    ["Architecture",       "[11,16,3]","Input → Hidden → Output layer sizes"],
    ["Parameters",         "243",    "Total trainable weights + biases"],
    ["Fitness formula",    "score×100 + steps×0.1", "Reward function"],
    ["Tournament size (k)","5",      "Candidates sampled per parent selection"],
]
story.append(data_table(
    ["Parameter", "Value", "Notes"],
    hp_rows,
    col_widths=[BW*0.28, BW*0.24, BW*0.48]
))
story.append(SP(0.4))

story += [
    P("3.2  Selection, Crossover and Mutation", H2_S),
    P(
        "Each new agent is produced by <b>tournament selection</b> of two parents, "
        "<b>uniform crossover</b> of their flat weight vectors, followed by "
        "<b>per-weight Gaussian mutation</b>. The top 15 elite agents bypass this process "
        "and are copied directly into the next generation.", BODY_S),
    code_block(
"""\
def fitness(score, steps):
    return score * 100 + steps * 0.1

def tournament_select(pop, fits, k=5):
    idxs = np.random.choice(len(pop), k, replace=False)
    best = idxs[np.argmax([fits[i] for i in idxs])]
    return pop[best]

def crossover(a, b):
    mask = np.random.rand(len(a)) < 0.5   # uniform crossover
    return np.where(mask, a, b)

def mutate(flat):
    mask = np.random.rand(len(flat)) < MUTATION_RATE   # 0.15
    return flat + mask * np.random.randn(len(flat)) * MUTATION_STR  # noise std 0.20

def make_child(pop, fits):
    p1 = tournament_select(pop, fits)
    p2 = tournament_select(pop, fits)
    child_flat = mutate(crossover(p1.get_weights_flat(), p2.get_weights_flat()))
    child = NeuralNetwork(LAYER_SIZES)
    child.set_weights_flat(child_flat)
    return child"""
    ),
    SP(0.4),
    P("3.3  Main Training Loop", H2_S),
    code_block(
"""\
for gen in range(1, GENERATIONS + 1):
    results = [evaluate(nn) for nn in pop]          # run each agent one episode
    fits    = [fitness(s, t) for s, t in results]

    order    = np.argsort(fits)[::-1]               # rank best → worst
    best_fit = fits[order[0]]

    if best_fit > alltime_best_fit:                 # new record → save checkpoint
        alltime_best_fit = best_fit
        pickle.dump(pop[order[0]], open(SAVE_PATH, 'wb'))

    # Elitism: top 10 % copied unchanged
    next_pop = [pop[i].copy() for i in order[:n_elite]]
    # Fill remaining slots with offspring
    while len(next_pop) < POP_SIZE:
        next_pop.append(make_child(pop, fits))
    pop = next_pop"""
    ),
    SP(0.3),
]

# ═══════════════════════════════════════════════════════════════
# 4. TRAINING RESULTS
# ═══════════════════════════════════════════════════════════════
story += [
    PageBreak(),
    P("4.  Training Results", H1_S), HR(),
    P(
        "The baseline GA run completed 180 generations, evaluating 150 agents per generation "
        "(27,000 total episode evaluations). All results below are from a single run logged in "
        "<tt>training_log.csv</tt>.", BODY_S),
    SP(0.2),
]

res_rows = [
    ["All-time best score",         "259",        "Generation 166"],
    ["All-time best fitness",        "29,571",     "Generation 165  (score 257)"],
    ["Generation 1 avg fitness",     "30.4",       "Random-policy baseline"],
    ["Generation 180 avg fitness",   "7,910",      "263× improvement over gen 1"],
    ["Score ≥ 50 first reached",     "Gen 77",     "Best score that gen: 57"],
    ["Score ≥ 100 first reached",    "Gen 91",     "Best score that gen: 101"],
    ["Score ≥ 200 first reached",    "Gen 132",    "Jump from plateau at ~147"],
    ["Avg score (last 10 gens)",     "209.6",      "Gens 171–180"],
]
story.append(data_table(
    ["Metric", "Value", "Notes"],
    res_rows,
    col_widths=[BW*0.42, BW*0.18, BW*0.40]
))
story.append(SP(0.4))

story += [
    P("4.1  Fitness Curves", H2_S),
    P(
        "Figure 1 shows two panels: the <b>best fitness per generation</b> (top) and the "
        "<b>mean population fitness</b> (bottom). Both use a 7-generation moving average "
        "(solid line) plotted over the raw noisy signal (light fill). "
        "The green dashed staircase tracks the all-time best fitness, rising monotonically. "
        "Five breakthrough annotations mark the major jumps. "
        "The red dotted baseline in the lower panel marks the generation-1 mean (≈ 30.4), "
        "representing the random-policy reference level.", BODY_S),
]
story += fig_block("fig1_fitness_curves.png",
    "Figure 1 — Best and mean fitness per generation (GA, pop=150, 180 gens). "
    "Upper panel: best fitness + all-time record staircase. "
    "Lower panel: mean population fitness vs random baseline.",
    aspect=7/11, w_frac=0.96)

story += [
    P("4.2  Score Progression", H2_S),
    P(
        "Figure 2 plots the best score achieved by any agent in each generation. "
        "Scatter points are the raw per-generation values; the purple curve is the "
        "7-generation moving average. Three annotated milestones show when score thresholds "
        "of 50, 100, and 200 were first surpassed. Horizontal reference lines at every 25 "
        "points provide a reading scale. The random baseline sits near zero (≈ 0.3).", BODY_S),
]
story += fig_block("fig2_score_progression.png",
    "Figure 2 — Best score per generation. Scatter = raw values; curve = 7-gen moving average. "
    "Milestone annotations show generation at which score ≥ 50 / 100 / 200 was first reached.",
    aspect=5/11, w_frac=0.96)

story.append(PageBreak())

story += [
    P("4.3  All-Time Best Score Staircase", H2_S),
    P(
        "Figure 3 visualises when new all-time score records were set, using a step-plot "
        "(each horizontal step = a period with no new record; each vertical rise = a new record). "
        "Notable records are labelled. The pattern clearly shows <b>punctuated equilibrium</b>: "
        "the score stays flat for many generations, then leaps suddenly as a new behavioural "
        "strategy emerges in the population. After generation 166 (score 259), no new record "
        "was set in the final 14 generations, indicating the baseline config has reached near-ceiling performance.", BODY_S),
]
story += fig_block("fig3_alltime_records.png",
    "Figure 3 — All-time best score staircase. Each vertical step is a new record. "
    "Notable scores 13 → 40 → 57 → 101 → 147 → 204 → 224 → 259 are annotated.",
    aspect=5/11, w_frac=0.96)

story += [
    P("4.4  Final Performance Comparison", H2_S),
    P(
        "Figure 4 is a grouped bar chart comparing three agents: a random baseline, the "
        "GA (Experiment 1), and NEAT (Experiment 5). Each group shows two bars: "
        "dark = all-time best score, light = average score over the last 10 generations. "
        "The random agent achieves a best score of 2 and an average near 0.3. "
        "The GA achieves best 259 and avg 209.6. "
        "NEAT results are pending (shown as TBD) — the <tt>neat_training_log.csv</tt> "
        "will populate this chart automatically when Experiment 5 is complete.", BODY_S),
]
story += fig_block("fig4_final_comparison.png",
    "Figure 4 — Final performance comparison: Random agent vs GA (Exp 1) vs NEAT (Exp 5). "
    "Dark bars = all-time best score; light bars = avg over last 10 gens. NEAT result pending.",
    aspect=5.5/9, w_frac=0.80)

story.append(PageBreak())

story += [
    P("4.5  Convergence Speed", H2_S),
    P(
        "Figure 5 is a horizontal bar chart showing the generation at which each score "
        "threshold was first exceeded. Bars are colour-coded green-to-red (low threshold = "
        "reached early = green). Key observations: the snake reaches score 50 by generation 77 "
        "but the jump from 150 to 200 happens in a single generation (132), reflecting the "
        "punctuated-equilibrium dynamic seen in Figure 3. Thresholds 225 and 250 are both "
        "first exceeded at generation 165 in the same leap to score 257.", BODY_S),
]
story += fig_block("fig5_convergence_milestones.png",
    "Figure 5 — Generations required to first reach each score threshold. "
    "Colour: green = easy/early, red = difficult/late.",
    aspect=6/9, w_frac=0.80)

story += [
    P("4.6  Combined Overview", H2_S),
    P(
        "Figure 6 brings all four sub-analyses onto a single page. "
        "Panel A reproduces the best fitness curve; "
        "Panel B the mean population fitness; "
        "Panel C the score scatter; "
        "Panel D the convergence milestones (8-threshold condensed version). "
        "This figure is designed for at-a-glance overview in presentations.", BODY_S),
]
story += fig_block("fig6_combined_overview.png",
    "Figure 6 — Combined 2×3 overview panel. A: best fitness. B: mean fitness. "
    "C: best score scatter. D: convergence milestones.",
    aspect=9/15, w_frac=0.99)

story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════
# 5. NEAT COMPARISON
# ═══════════════════════════════════════════════════════════════
story += [
    P("5.  NEAT Comparison  (Experiment 5)", H1_S), HR(),
    P(
        "Experiment 5 implements <b>NEAT (NeuroEvolution of Augmenting Topologies)</b> as a "
        "direct comparison against the baseline GA. Unlike the GA, which evolves weights "
        "within a fixed [11→16→3] architecture, NEAT simultaneously evolves both the weights "
        "<i>and the network topology</i>, starting from a minimal 11-input / 0-hidden / "
        "3-output genome (33 direct connections) and adding nodes and connections via "
        "structural mutations.", BODY_S),
    P("5.1  Implementation", H2_S),
    P(
        "NEAT is implemented using the <tt>neat-python</tt> library (v2.0.0). "
        "The genome evaluation and custom reporter are the key components:", BODY_S),
    code_block(
"""\
# Genome evaluation — identical game loop to GA
def eval_genomes(genomes, config):
    for _genome_id, genome in genomes:
        net   = neat.nn.FeedForwardNetwork.create(genome, config)
        game  = SnakeGame(render=False)
        state = game.reset()
        while True:
            action = int(np.argmax(net.activate(list(state))))
            _, done, _ = game.step(action)
            if done: break
            state = game.get_state()
        genome.score   = game.score
        genome.fitness = score * 100 + game.steps * 0.1

# Per-generation reporter — logs topology stats absent from GA
def post_evaluate(self, config, population, species, best_genome):
    n_species = len(species.species)
    avg_nodes = np.mean([len(g.nodes) for g in population.values()])
    avg_conns = np.mean([
        sum(1 for c in g.connections.values() if c.enabled)
        for g in population.values()
    ])
    # logs: generation, best_fitness, avg_fitness, best_score,
    #       alltime_best, num_species, avg_nodes, avg_connections"""
    ),
    SP(0.4),
    P("5.2  Configuration Comparison", H2_S),
]

config_rows = [
    ["Population size",      "150",               "150",                  "Same"],
    ["Generations",          "180",               "180",                  "Same"],
    ["Fitness formula",      "score×100+steps×0.1","score×100+steps×0.1","Same"],
    ["Architecture",         "Fixed [11,16,3]",   "Evolves from 11→0→3", "Different"],
    ["Elitism",              "Top 15 (10 %)",     "elitism=2 per species","GA stronger"],
    ["Mutation strength",    "0.20",              "weight_mutate_power=0.5","GA lower"],
    ["Structural mutations", "None",              "conn_add/del prob=0.5","NEAT aggressive"],
    ["Selection pool",       "All 150",           "~18 per species",      "GA broader"],
]
story.append(data_table(
    ["Parameter", "GA (Exp 1)", "NEAT (Exp 5)", "Effect"],
    config_rows,
    col_widths=[BW*0.26, BW*0.22, BW*0.26, BW*0.26]
))
story.append(SP(0.4))

story += [
    P("5.3  Why GA is Expected to Outperform NEAT on This Task", H2_S),
    P(
        "NEAT's structural search advantage is most valuable when the optimal architecture "
        "is unknown. For the Snake task with 11 well-defined binary inputs, [11→16→3] is "
        "already a near-optimal architecture. Six specific configuration-level reasons "
        "favour the GA:", BODY_S),
]

neat_reasons = [
    ("<b>Architecture already known.</b>  GA hand-picks [11→16→3] which is appropriate "
     "from generation 1. NEAT spends early generations rediscovering this topology via "
     "structural mutations."),
    ("<b>NEAT starts with 0 hidden nodes</b> (<tt>num_hidden=0</tt>). Early genomes are "
     "linear models (11→3), incapable of the non-linear decision boundaries needed to "
     "navigate tight corridors."),
    ("<b>Structural mutation too aggressive.</b>  <tt>conn_add_prob=0.5</tt> and "
     "<tt>conn_delete_prob=0.5</tt> cause NEAT to continuously thrash topology rather "
     "than refining weights. Recommended values for this task: 0.10–0.20."),
    ("<b>Elitism too weak.</b>  <tt>elitism=2</tt> per species vs GA's top 15. "
     "Good solutions are more frequently lost between generations in NEAT."),
    ("<b>Mutation strength too high.</b>  NEAT <tt>weight_mutate_power=0.5</tt> "
     "vs GA <tt>MUTATION_STR=0.20</tt>. Coarser weight updates impede fine-tuning."),
    ("<b>Speciation dilutes selection pressure.</b>  With 150 agents split into 5–10 "
     "species, each species effectively runs a tournament over only ~18 agents. "
     "GA's selection operates over the full population of 150."),
]
for r in neat_reasons:
    story.append(P(f"• {r}", BULLET_S))

story += [SP(0.3),
    P(
        "In academic framing: GA wins here not because NEAT is inferior in general, but "
        "because the task is well-specified enough that fixed-topology weight optimisation "
        "is the right tool. NEAT's advantage emerges on tasks where the optimal architecture "
        "is unknown and run budgets are much larger.", BODY_S),
    SP(0.3),
    PageBreak(),
]

# ═══════════════════════════════════════════════════════════════
# 6. DISCUSSION
# ═══════════════════════════════════════════════════════════════
story += [
    P("6.  Discussion", H1_S), HR(),
    P("6.1  Four Phases of Learning", H2_S),
    P(
        "Examining the training log reveals four qualitatively distinct phases:", BODY_S),
]

phase_rows = [
    ["Phase 1", "Gens 1–47",    "30 → 64",   "~1–5",   "Exploration. Agents learn to survive but rarely find food. Mean fitness grows slowly."],
    ["Phase 2", "Gens 48–90",   "74 → 1018", "5 → 90",  "First breakthrough at gen 48 (score 13). Rapid mean fitness rise as food-seeking spreads through the population."],
    ["Phase 3", "Gens 91–131",  "1134→3598", "90→147",  "Consolidation. Population-wide learning (mean rises steadily). Best score plateaus at ~100–147."],
    ["Phase 4", "Gens 132–180", "3398→7910", "147→259", "Elite play. Score 200 cleared at gen 132 in a single leap; best reaches 259 at gen 166."],
]
story.append(data_table(
    ["Phase", "Gens", "Avg Fitness", "Best Score", "Description"],
    phase_rows,
    col_widths=[BW*0.09, BW*0.12, BW*0.13, BW*0.13, BW*0.53]
))
story.append(SP(0.4))

story += [
    P("6.2  Punctuated Equilibrium", H2_S),
    P(
        "The staircase plot (Figure 3) and score scatter (Figure 2) both exhibit a "
        "<b>punctuated equilibrium</b> pattern — the hallmark of evolutionary systems "
        "discovering new behavioural strategies. The score plateaus at 5 (gens 30–47), "
        "then at ~100–147 (gens 91–131), before making sudden leaps. "
        "These plateaus likely correspond to the population converging on a local "
        "behavioural optimum; the leap occurs when a mutation in an elite agent "
        "discovers a qualitatively different navigation strategy that propagates rapidly "
        "through the gene pool via elitism and tournament selection.", BODY_S),
    P("6.3  Population vs Elite Learning", H2_S),
    P(
        "The large and persistent gap between best fitness and mean fitness (visible in "
        "Figure 1a vs 1b) indicates that a small elite group consistently outperforms the "
        "rest of the population. However, mean fitness does grow substantially — from 30 "
        "to 7,910 over 180 generations — confirming that improvements do propagate "
        "population-wide through crossover. The ratio of best to mean fitness narrows "
        "slightly in later generations as the population converges.", BODY_S),
    P("6.4  Known Limitations and Future Improvements", H2_S),
]

lim_items = [
    "<b>Weight initialisation.</b>  Current init uses <tt>× 0.5</tt> scaling. "
    "He initialisation (<tt>sqrt(2 / fan_in)</tt>) is better suited to ReLU activations "
    "and would provide better signal magnitude from generation 1.",
    "<b>Collision detection.</b>  <tt>_is_collision()</tt> rebuilds a Python <tt>set</tt> "
    "from the body deque on every call — a minor O(n) overhead that compounds for long snakes.",
    "<b>Single run.</b>  Results are from one run only. Multiple seeds are needed to "
    "establish variance and confirm reproducibility.",
    "<b>Ceiling at 259.</b>  The final 14 generations produced no new record, suggesting "
    "the [11→16→3] architecture and current hyperparameters have approached their "
    "performance ceiling for 180 generations.",
]
for li in lim_items:
    story.append(P(f"• {li}", BULLET_S))
story.append(SP(0.3))

# ═══════════════════════════════════════════════════════════════
# 7. CONCLUSION
# ═══════════════════════════════════════════════════════════════
story += [
    P("7.  Conclusion", H1_S), HR(),
    P(
        "This report presents a complete neuroevolutionary Snake agent trained via a "
        "custom Genetic Algorithm. The key outcomes are:", BODY_S),
]
conc_bullets = [
    "The GA successfully trained a 243-parameter network to achieve a score of <b>259</b> "
    "(food items eaten in a single episode) over 180 generations.",
    "Population mean fitness improved <b>263×</b>, demonstrating genuine evolutionary "
    "learning rather than a single lucky individual.",
    "Learning proceeded in four identifiable phases with characteristic punctuated-"
    "equilibrium jumps at generations 48, 91, and 132.",
    "A complete NEAT comparison implementation is ready; six config-level reasons "
    "predict that the fixed-topology GA will outperform NEAT on this well-specified task.",
    "Six publication-quality figures are generated by <tt>plot_training.py</tt> and "
    "update automatically when <tt>neat_training_log.csv</tt> is written.",
]
for cb in conc_bullets:
    story.append(P(f"• {cb}", BULLET_S))

story += [
    SP(0.5),
    HR(DARK_BLUE, 1.0),
    SP(0.3),
    P("File Overview", H2_S),
]

file_rows = [
    ["game.py",              "Pygame Snake environment — state, step, reward, renderer"],
    ["neural_net.py",        "Feedforward network — forward pass, flat weight I/O"],
    ["train.py",             "GA training loop — selection, crossover, mutation, logging"],
    ["visualizer.py",        "3×2 grid visualiser shown every RENDER_EVERY generations"],
    ["neat_compare.py",      "NEAT Experiment 5 — eval_genomes + custom reporter"],
    ["config-neat.txt",      "NEAT hyperparameter config (pop=150, relu, 0 hidden start)"],
    ["plot_training.py",     "Generates 6 publication figures to plots/"],
    ["training_log.csv",     "GA run data — 180 gens, 5 columns"],
    ["neat_training_log.csv","Written by neat_compare.py after NEAT run (Exp 5)"],
    ["best_snake.pkl",       "Pickle of the best GA agent (saved on each new record)"],
]
story.append(data_table(
    ["File", "Purpose"],
    file_rows,
    col_widths=[BW*0.32, BW*0.68]
))

story += [
    SP(0.5),
    P(
        "<i>To regenerate all figures: </i><tt>python plot_training.py</tt><br/>"
        "<i>To run NEAT experiment: </i><tt>python neat_compare.py</tt><br/>"
        "<i>To regenerate this report: </i><tt>python generate_report.py</tt>",
        ps("foot", fontSize=9, leading=14, textColor=MID_GRAY,
           leftIndent=0.5*cm, spaceAfter=4))
]

# ─────────────────────────────────────────────────────────────────────────────
# BUILD PDF
# ─────────────────────────────────────────────────────────────────────────────
def _header_footer(canvas, doc):
    canvas.saveState()
    # Header rule
    canvas.setStrokeColor(MID_BLUE)
    canvas.setLineWidth(0.5)
    canvas.line(ML, H - MT + 4, W - MR, H - MT + 4)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MID_GRAY)
    canvas.drawString(ML, H - MT + 7, "Snake AI — Neuroevolution with Genetic Algorithms")
    canvas.drawRightString(W - MR, H - MT + 7, "IBA Karachi  |  Dr. Syed Ali Raza")
    # Footer
    canvas.line(ML, MB - 6, W - MR, MB - 6)
    canvas.drawCentredString(W / 2, MB - 14, f"Page {doc.page}")
    canvas.restoreState()

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    leftMargin=ML, rightMargin=MR,
    topMargin=MT + 0.6*cm, bottomMargin=MB + 0.4*cm,
    title="Snake AI — Neuroevolution with Genetic Algorithms",
    author="IBA Karachi",
)
doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
print(f"\nReport saved: {OUTPUT}")
