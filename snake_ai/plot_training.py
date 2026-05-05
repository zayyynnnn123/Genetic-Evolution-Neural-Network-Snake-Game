

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "lines.linewidth": 1.8,
    "legend.framealpha": 0.75,
})

C_BEST  = "#1976D2"   # blue   best fitness per gen
C_AVG   = "#F57C00"   # amber mean fitness per gen
C_ALLTIME = "#388E3C" # green all-time best
C_SCORE = "#6A1B9A"   # purple  score
C_RAND  = "#D32F2F"   # red    random baseline
C_NEAT  = "#00838F"   # teal   NEAT placeholder

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(_SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

_EMBEDDED = [
    (1,113.2,30.4,1,113.2),(2,202.6,40.4,2,202.6),(3,105.4,45.4,1,202.6),
    (4,114.3,40.8,1,202.6),(5,203.9,43.3,2,203.9),(6,109.1,39.9,1,203.9),
    (7,104.1,39.3,1,203.9),(8,204.8,47.6,2,204.8),(9,104.7,44.8,1,204.8),
    (10,102.6,42.7,1,204.8),(11,137.6,43.1,1,204.8),(12,309.5,49.1,3,309.5),
    (13,206.1,50.0,2,309.5),(14,114.2,47.8,1,309.5),(15,202.5,49.3,2,309.5),
    (16,208.5,52.6,2,309.5),(17,205.9,49.8,2,309.5),(18,136.0,48.2,1,309.5),
    (19,180.1,48.3,1,309.5),(20,203.5,50.1,2,309.5),(21,105.4,50.3,1,309.5),
    (22,106.8,52.7,1,309.5),(23,119.0,44.0,1,309.5),(24,540.1,55.0,4,540.1),
    (25,540.1,54.0,4,540.1),(26,208.9,50.8,2,540.1),(27,448.1,59.9,4,540.1),
    (28,201.4,53.9,2,540.1),(29,213.7,52.2,2,540.1),(30,566.9,54.6,5,566.9),
    (31,209.6,53.5,2,566.9),(32,213.1,55.7,2,566.9),(33,328.6,52.4,3,566.9),
    (34,420.1,55.8,3,566.9),(35,421.1,52.8,4,566.9),(36,311.2,51.6,3,566.9),
    (37,215.6,58.6,2,566.9),(38,524.9,55.3,5,566.9),(39,311.4,52.8,3,566.9),
    (40,206.9,54.2,2,566.9),(41,300.1,56.3,2,566.9),(42,300.1,54.8,2,566.9),
    (43,511.8,58.6,5,566.9),(44,428.9,63.2,4,566.9),(45,516.5,64.0,5,566.9),
    (46,412.4,58.7,4,566.9),(47,204.4,56.3,2,566.9),(48,1352.0,74.6,13,1352.0),
    (49,660.1,75.1,5,1352.0),(50,300.1,60.8,2,1352.0),(51,207.4,70.3,2,1352.0),
    (52,314.4,68.9,3,1352.0),(53,540.1,69.0,4,1352.0),(54,414.2,72.5,4,1352.0),
    (55,300.1,71.3,2,1352.0),(56,660.1,80.4,5,1352.0),(57,540.1,74.4,4,1352.0),
    (58,420.1,79.7,3,1352.0),(59,520.5,73.1,5,1352.0),(60,540.1,76.8,4,1352.0),
    (61,1030.3,80.0,10,1352.0),(62,1140.1,94.6,9,1352.0),(63,660.1,88.8,5,1352.0),
    (64,1240.9,97.8,12,1352.0),(65,625.4,94.0,6,1352.0),(66,540.1,100.7,4,1352.0),
    (67,1338.1,124.1,13,1352.0),(68,1248.4,146.7,12,1352.0),(69,2071.0,165.7,20,2071.0),
    (70,2724.0,190.8,26,2724.0),(71,2415.1,220.5,23,2724.0),(72,2185.0,285.9,21,2724.0),
    (73,4169.3,388.9,40,4169.3),(74,2587.5,372.4,25,4169.3),(75,4045.3,433.7,39,4169.3),
    (76,4347.7,476.0,42,4347.7),(77,5980.8,516.4,57,5980.8),(78,3900.1,444.7,32,5980.8),
    (79,3683.5,490.4,34,5980.8),(80,3821.3,605.3,37,5980.8),(81,6186.3,518.4,59,6186.3),
    (82,5064.0,515.1,48,6186.3),(83,3908.8,523.1,38,6186.3),(84,4815.9,608.9,45,6186.3),
    (85,5221.5,604.8,49,6186.3),(86,5772.5,572.2,54,6186.3),(87,5949.3,705.4,56,6186.3),
    (88,5948.5,817.8,56,6186.3),(89,6202.7,878.5,58,6202.7),(90,9530.5,1018.8,90,9530.5),
    (91,10661.7,1134.7,101,10661.7),(92,6780.1,1358.0,56,10661.7),(93,7539.0,1513.2,72,10661.7),
    (94,6879.5,1615.3,65,10661.7),(95,7246.0,1997.4,69,10661.7),(96,8922.9,1780.8,85,10661.7),
    (97,8896.8,1488.4,85,10661.7),(98,9024.5,1590.7,86,10661.7),(99,7570.2,1521.0,70,10661.7),
    (100,8502.7,1774.4,80,10661.7),(101,10123.4,2014.4,93,10661.7),(102,10846.8,2082.0,103,10846.8),
    (103,7695.3,1886.1,73,10846.8),(104,9671.0,2554.6,91,10846.8),(105,8830.0,2014.1,84,10846.8),
    (106,9660.1,2168.2,80,10846.8),(107,10314.9,2283.3,97,10846.8),(108,12334.6,2152.4,116,12334.6),
    (109,10012.2,2265.3,95,12334.6),(110,11847.5,2552.7,113,12334.6),(111,10322.7,2824.2,98,12334.6),
    (112,10081.9,2353.8,95,12334.6),(113,12625.2,2854.9,119,12625.2),(114,10512.9,2319.4,99,12625.2),
    (115,11051.0,2490.0,105,12625.2),(116,8999.9,2242.9,85,12625.2),(117,10213.6,2353.4,97,12625.2),
    (118,11101.1,2770.6,105,12625.2),(119,12687.1,3063.2,118,12687.1),(120,10650.0,2665.8,101,12687.1),
    (121,10191.0,2727.8,96,12687.1),(122,10435.5,2937.1,99,12687.1),(123,11318.8,3041.3,106,12687.1),
    (124,10420.6,2670.1,93,12687.1),(125,11585.0,2710.5,104,12687.1),(126,10773.5,2934.4,102,12687.1),
    (127,13127.1,2833.5,123,13127.1),(128,15753.0,2582.7,147,15753.0),(129,15541.8,2797.3,145,15753.0),
    (130,14759.1,2934.3,138,15753.0),(131,15531.7,3598.1,141,15753.0),(132,23271.9,3398.8,204,23271.9),
    (133,17735.9,3168.4,162,23271.9),(134,16815.3,3584.9,154,23271.9),(135,17747.2,3296.2,157,23271.9),
    (136,21389.1,3175.4,186,23271.9),(137,14593.2,3708.0,134,23271.9),(138,13595.0,4272.6,124,23271.9),
    (139,18485.3,4760.3,170,23271.9),(140,25533.6,4959.8,224,25533.6),(141,17969.0,3478.9,163,25533.6),
    (142,16949.8,4208.1,153,25533.6),(143,17777.9,3980.4,162,25533.6),(144,17183.5,4292.4,155,25533.6),
    (145,16957.6,4422.4,159,25533.6),(146,17944.7,5372.5,164,25533.6),(147,16719.4,5291.6,156,25533.6),
    (148,16214.4,4957.2,149,25533.6),(149,18769.6,5703.4,171,25533.6),(150,18745.4,4958.2,170,25533.6),
    (151,18233.9,4360.0,170,25533.6),(152,17810.8,6493.2,159,25533.6),(153,22673.9,5553.2,201,25533.6),
    (154,23189.2,5522.0,207,25533.6),(155,23449.3,6258.4,209,25533.6),(156,19632.2,6079.7,174,25533.6),
    (157,18163.8,5292.4,159,25533.6),(158,18412.9,5572.6,167,25533.6),(159,17366.3,5285.1,151,25533.6),
    (160,22740.1,5211.4,189,25533.6),(161,18088.1,4819.7,165,25533.6),(162,19887.9,5692.2,175,25533.6),
    (163,23532.0,6561.4,209,25533.6),(164,18654.6,6020.6,163,25533.6),(165,29571.2,6976.6,257,29571.2),
    (166,29461.0,6088.3,259,29571.2),(167,26014.9,5491.0,229,29571.2),(168,22384.8,6448.1,196,29571.2),
    (169,24524.6,7074.9,217,29571.2),(170,21530.1,7355.9,188,29571.2),(171,27470.4,7507.6,242,29571.2),
    (172,23113.3,7028.9,200,29571.2),(173,18913.3,6657.1,167,29571.2),(174,20317.0,7785.4,177,29571.2),
    (175,22786.2,7357.3,199,29571.2),(176,23410.4,7132.5,206,29571.2),(177,25511.4,8217.5,225,29571.2),
    (178,24909.1,7494.0,217,29571.2),(179,26355.8,7349.2,230,29571.2),(180,26669.6,7910.2,233,29571.2),
]

def _load_csv(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["generation"]), float(r["best_fitness"]),
                         float(r["avg_fitness"]), int(r["best_score"]),
                         float(r["alltime_best"])))
    return rows

_GA_CSV   = os.path.join(_SCRIPT_DIR, "training_log.csv")
_NEAT_CSV = os.path.join(_SCRIPT_DIR, "neat_training_log.csv")

_raw      = _load_csv(_GA_CSV) if os.path.exists(_GA_CSV) else _EMBEDDED
_neat_raw = _load_csv(_NEAT_CSV) if os.path.exists(_NEAT_CSV) else None

gens        = np.array([r[0] for r in _raw])
best_fit    = np.array([r[1] for r in _raw])
avg_fit     = np.array([r[2] for r in _raw])
best_score  = np.array([r[3] for r in _raw])
alltime_fit = np.array([r[4] for r in _raw])

W = 7 

def _mavg(arr, w=W):
    return np.convolve(arr, np.ones(w) / w, mode="valid")

ma_gens  = gens[W - 1:]
ma_best  = _mavg(best_fit)
ma_avg   = _mavg(avg_fit)
ma_score = _mavg(best_score.astype(float))

# NEAT arrays populated only when neat_training_log.csv is present
if _neat_raw is not None:
    neat_gens       = np.array([r[0] for r in _neat_raw])
    neat_best_fit   = np.array([r[1] for r in _neat_raw])
    neat_avg_fit    = np.array([r[2] for r in _neat_raw])
    neat_best_score = np.array([r[3] for r in _neat_raw])
    neat_alltime    = np.array([r[4] for r in _neat_raw])
    neat_ma_gens    = neat_gens[W - 1:]
    neat_ma_best    = _mavg(neat_best_fit)
    neat_ma_avg     = _mavg(neat_avg_fit)
    neat_ma_score   = _mavg(neat_best_score.astype(float))
else:
    neat_gens = neat_best_fit = neat_avg_fit = neat_best_score = neat_alltime = None
    neat_ma_gens = neat_ma_best = neat_ma_avg = neat_ma_score = None

BREAKTHROUGHS = [
    (48,  1352.0,  "Score 13",  3),
    (73,  4169.3,  "Score 40",  3),
    (91,  10661.7, "Score 101", 3),
    (132, 23271.9, "Score 204", 3),
    (165, 29571.2, "Score 257", 3),
]


def fig1_fitness_curves():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    #  Panel A best fitness per generation 
    ax1.plot(gens, best_fit, color=C_BEST, alpha=0.22, lw=1)
    ax1.plot(ma_gens, ma_best, color=C_BEST, lw=2.2, label=f"GA best fitness ({W}-gen avg)")
    ax1.plot(gens, alltime_fit, color=C_ALLTIME, lw=1.8, ls="--", label="GA all-time best")
    ax1.fill_between(gens, best_fit, alpha=0.06, color=C_BEST)
    if neat_gens is not None:
        ax1.plot(neat_gens, neat_best_fit, color=C_NEAT, alpha=0.22, lw=1)
        ax1.plot(neat_ma_gens, neat_ma_best, color=C_NEAT, lw=2.2, ls="--",
                 label=f"NEAT best fitness ({W}-gen avg)")
        ax1.plot(neat_gens, neat_alltime, color=C_NEAT, lw=1.5, ls=":",
                 alpha=0.7, label="NEAT all-time best")
    else:
        ax1.text(0.98, 0.04, "NEAT pending", transform=ax1.transAxes,
                 ha="right", fontsize=8, color=C_NEAT, alpha=0.65)
    ax1.set_ylabel("Fitness")
    ax1.set_title("Fig 1a — Best Fitness per Generation  (GA pop=150, 180 gens)")
    ax1.legend(loc="upper left")

    for g, fval, lbl, dx in BREAKTHROUGHS:
        ax1.annotate(lbl, xy=(g, fval),
                     xytext=(g + dx, fval * 1.07),
                     fontsize=7.5, color="#333",
                     arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.8))

    # Panel B mean fitness per generation 
    ax2.plot(gens, avg_fit, color=C_AVG, alpha=0.22, lw=1)
    ax2.plot(ma_gens, ma_avg, color=C_AVG, lw=2.2, label=f"GA mean fitness ({W}-gen avg)")
    ax2.fill_between(gens, avg_fit, alpha=0.06, color=C_AVG)
    ax2.axhline(30.4, color=C_RAND, ls=":", lw=1.5, label="Random baseline (gen-1 avg ≈ 30)")
    if neat_gens is not None:
        ax2.plot(neat_gens, neat_avg_fit, color=C_NEAT, alpha=0.22, lw=1)
        ax2.plot(neat_ma_gens, neat_ma_avg, color=C_NEAT, lw=2.2, ls="--",
                 label=f"NEAT mean fitness ({W}-gen avg)")
    else:
        ax2.text(0.98, 0.04, "NEAT pending", transform=ax2.transAxes,
                 ha="right", fontsize=8, color=C_NEAT, alpha=0.65)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Mean Population Fitness")
    ax2.set_title("Fig 1b — Mean Population Fitness per Generation  (population-wide learning)")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig1_fitness_curves.png")
    plt.savefig(path)
    plt.close()
    print(f"  [ok] {path}")


def fig2_score_progression():
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.scatter(gens, best_score, s=10, color=C_SCORE, alpha=0.35, zorder=3)
    ax.plot(ma_gens, ma_score, color=C_SCORE, lw=2.3, label=f"GA best score ({W}-gen avg)")
    ax.axhline(0.3, color=C_RAND, ls=":", lw=1.5, label="Random baseline (~0 food/episode)")

    if neat_gens is not None:
        ax.scatter(neat_gens, neat_best_score, s=10, color=C_NEAT, alpha=0.35, zorder=3,
                   marker="^")
        ax.plot(neat_ma_gens, neat_ma_score, color=C_NEAT, lw=2.3, ls="--",
                label=f"NEAT best score ({W}-gen avg)")
    else:
        ax.text(0.98, 0.04, "NEAT pending", transform=ax.transAxes,
                ha="right", fontsize=8, color=C_NEAT, alpha=0.65)

    for threshold in [10, 25, 50, 100, 150, 200, 250]:
        ax.axhline(threshold, color="#ccc", ls="--", lw=0.8)
        ax.text(182, threshold + 1.5, str(threshold), fontsize=7, color="#888", va="bottom")

    for t, lbl in [(50, "50"), (100, "100"), (200, "200")]:
        idx = int(np.argmax(best_score >= t))
        if best_score[idx] >= t:
            ax.annotate(f"GA ≥ {lbl}\n(gen {gens[idx]})",
                        xy=(gens[idx], best_score[idx]),
                        xytext=(gens[idx] + 4, best_score[idx] + 12),
                        fontsize=7.5, color="#333",
                        arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.8))

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Score  (food items eaten)")
    ax.set_title("Fig 2 — Best Score per Generation  (GA vs NEAT, top agent per generation)")
    ax.set_xlim(0, 185)
    ax.legend(loc="upper left")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig2_score_progression.png")
    plt.savefig(path)
    plt.close()
    print(f"  [ok] {path}")

def fig3_alltime_records():
    # Build the staircase of when new records were set
    rec_gens, rec_scores = [], []
    curr = -1
    for g, s in zip(gens, best_score):
        if s > curr:
            curr = s
            rec_gens.append(g)
            rec_scores.append(s)

    plot_g = rec_gens + [gens[-1] + 1]
    plot_s = rec_scores + [rec_scores[-1]]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.step(plot_g, plot_s, where="post", color=C_ALLTIME, lw=2.5, label="GA all-time best")
    ax.fill_between(plot_g, plot_s, step="post", alpha=0.12, color=C_ALLTIME)

    # Label key GA records
    notable = {13, 40, 57, 101, 147, 204, 224, 259}
    for g, s in zip(rec_gens, rec_scores):
        if s in notable:
            ax.annotate(str(s), xy=(g, s), xytext=(g + 2, s + 6),
                        fontsize=8, color="#333",
                        arrowprops=dict(arrowstyle="->", color="#bbb", lw=0.7))

    if neat_gens is not None:
        neat_rec_gens, neat_rec_scores = [], []
        curr = -1
        for g, s in zip(neat_gens, neat_best_score):
            if s > curr:
                curr = s
                neat_rec_gens.append(g)
                neat_rec_scores.append(s)
        neat_plot_g = neat_rec_gens + [int(neat_gens[-1]) + 1]
        neat_plot_s = neat_rec_scores + [neat_rec_scores[-1]]
        ax.step(neat_plot_g, neat_plot_s, where="post", color=C_NEAT, lw=2.5,
                ls="--", label="NEAT all-time best")
        ax.fill_between(neat_plot_g, neat_plot_s, step="post", alpha=0.08, color=C_NEAT)
    else:
        ax.text(0.98, 0.04, "NEAT pending", transform=ax.transAxes,
                ha="right", fontsize=8, color=C_NEAT, alpha=0.65)

    ax.set_xlabel("Generation")
    ax.set_ylabel("All-Time Best Score")
    ax.set_title("Fig 3 — All-Time Best Score Progression  (GA vs NEAT, each step = new record)")
    ax.legend(loc="upper left")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig3_alltime_records.png")
    plt.savefig(path)
    plt.close()
    print(f"  [ok] {path}")


def fig4_final_comparison():
    ga_best      = int(np.max(best_score))
    ga_avg_final = float(np.mean(best_score[-10:]))

    rand_best = 2
    rand_avg  = 0.3

    if _neat_raw is not None:
        neat_scores  = np.array([r[3] for r in _neat_raw])
        neat_best    = int(np.max(neat_scores))
        neat_avg     = float(np.mean(neat_scores[-10:]))
        neat_pending = False
    else:
        neat_best    = None
        neat_avg     = None
        neat_pending = True

    methods   = ["Random\nAgent", "GA\n(Ours, pop=150)", "NEAT\n(Exp 5)"]
    best_vals = [rand_best,  ga_best,  neat_best]
    avg_vals  = [rand_avg,   ga_avg_final, neat_avg]

    x   = np.arange(len(methods))
    w   = 0.35
    clr = [C_RAND, C_BEST, C_NEAT]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Best score bars
    for i, (bv, c) in enumerate(zip(best_vals, clr)):
        if bv is not None:
            ax.bar(x[i] - w / 2, bv, w, color=c, alpha=0.88,
                   edgecolor="white", label="Best score" if i == 0 else "_")
            ax.text(x[i] - w / 2, bv + 3, str(int(bv)),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Avg score bars
    for i, (av, c) in enumerate(zip(avg_vals, clr)):
        if av is not None:
            ax.bar(x[i] + w / 2, av, w, color=c, alpha=0.42,
                   edgecolor="white", label="Avg score (last 10 gens)" if i == 0 else "_")
            label = f"{av:.1f}" if av >= 1 else f"{av:.2f}"
            ax.text(x[i] + w / 2, av + 2, label,
                    ha="center", va="bottom", fontsize=8.5)

    if neat_pending:
        ax.text(x[2], 18, "TBD", ha="center", va="bottom",
                fontsize=12, color=C_NEAT, fontweight="bold", alpha=0.7)
        ax.text(x[2], 6, "(run Exp 5)", ha="center", va="bottom",
                fontsize=8, color=C_NEAT, alpha=0.6)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="grey", alpha=0.88, label="Best score (all-time)"),
        Patch(color="grey", alpha=0.42, label="Avg score (last 10 gens)"),
    ], loc="upper left")

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Score  (food items eaten)")
    y_top = max(v for v in [ga_best, neat_best] if v is not None) * 1.25
    ax.set_ylim(0, y_top)

    neat_note = ("neat_training_log.csv loaded"
                 if not neat_pending else "NEAT results pending (Experiment 5)")
    ax.set_title("Fig 4 — Final Performance Comparison: Random Agent vs GA vs NEAT\n"
                 f"(GA pop=150, 180 gens  —  {neat_note})")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig4_final_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"  [ok] {path}")

def fig5_convergence_milestones():
    thresholds = [5, 10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]

    ga_first = []
    for t in thresholds:
        idx = int(np.argmax(best_score >= t))
        ga_first.append(int(gens[idx]) if best_score[idx] >= t else None)

    valid_ga = [(t, g) for t, g in zip(thresholds, ga_first) if g is not None]
    tv, gv_ga = zip(*valid_ga)

    neat_first = []
    if neat_gens is not None:
        for t in tv:
            idx = int(np.argmax(neat_best_score >= t))
            neat_first.append(int(neat_gens[idx]) if neat_best_score[idx] >= t else None)
    has_neat = neat_gens is not None and any(g is not None for g in neat_first)

    colors_ga = plt.cm.RdYlGn(np.linspace(0.15, 0.9, len(tv)))

    fig, ax = plt.subplots(figsize=(9, 6 if not has_neat else 7))
    y  = np.arange(len(tv))
    h  = 0.32 if has_neat else 0.65

    # GA bars
    offset = -h / 2 if has_neat else 0
    bars_ga = ax.barh(y + offset, gv_ga, h, color=colors_ga, edgecolor="white",
                      label="GA (pop=150)")
    for bar, g in zip(bars_ga, gv_ga):
        ax.text(g + 1, bar.get_y() + bar.get_height() / 2,
                f"Gen {g}", va="center", fontsize=8, color="#222")

    # NEAT bars
    if has_neat:
        for i, g in enumerate(neat_first):
            if g is not None:
                b = ax.barh(y[i] + h / 2, g, h, color=C_NEAT, edgecolor="white",
                            alpha=0.85, label="NEAT" if i == 0 else "_")
                ax.text(g + 1, b[0].get_y() + b[0].get_height() / 2,
                        f"Gen {g}", va="center", fontsize=8, color=C_NEAT)
            else:
                ax.text(2, y[i] + h / 2, "N/A", va="center", fontsize=7.5,
                        color=C_NEAT, alpha=0.55)
        ax.legend(loc="lower right", fontsize=9)
    else:
        ax.text(0.98, 0.02, "NEAT pending", transform=ax.transAxes,
                ha="right", fontsize=8, color=C_NEAT, alpha=0.65)

    ax.set_yticks(y)
    ax.set_yticklabels([f"Score ≥ {t}" for t in tv])
    ax.set_xlabel("Generation (when threshold was first exceeded)")
    ax.set_title("Fig 5 — Convergence Speed: Generations to Reach Score Thresholds\n"
                 "(GA vs NEAT comparison)")
    ax.set_xlim(0, 170)
    ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig5_convergence_milestones.png")
    plt.savefig(path)
    plt.close()
    print(f"  [ok] {path}")


def fig6_combined_overview():
    fig = plt.figure(figsize=(15, 9))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # A — best fitness
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(gens, best_fit, color=C_BEST, alpha=0.2, lw=1)
    ax1.plot(ma_gens, ma_best, color=C_BEST, lw=2.2, label="GA best fitness")
    ax1.plot(gens, alltime_fit, color=C_ALLTIME, lw=1.7, ls="--", label="GA all-time best")
    ax1.fill_between(gens, best_fit, alpha=0.06, color=C_BEST)
    if neat_gens is not None:
        ax1.plot(neat_gens, neat_best_fit, color=C_NEAT, alpha=0.2, lw=1)
        ax1.plot(neat_ma_gens, neat_ma_best, color=C_NEAT, lw=2.0, ls="--",
                 label="NEAT best fitness")
    for g, fval, lbl, _ in BREAKTHROUGHS[::2]:
        ax1.annotate(lbl, xy=(g, fval), xytext=(g + 3, fval * 1.1),
                     fontsize=7, color="#555",
                     arrowprops=dict(arrowstyle="->", color="#bbb", lw=0.7))
    ax1.set_title("A — Best Fitness per Generation  (GA vs NEAT)")
    ax1.set_ylabel("Fitness")
    ax1.legend(fontsize=8, loc="upper left")

    # B — mean fitness
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.plot(gens, avg_fit, color=C_AVG, alpha=0.2, lw=1)
    ax2.plot(ma_gens, ma_avg, color=C_AVG, lw=2.2, label="GA mean fitness")
    ax2.fill_between(gens, avg_fit, alpha=0.06, color=C_AVG)
    ax2.axhline(30.4, color=C_RAND, ls=":", lw=1.5, label="Random baseline")
    if neat_gens is not None:
        ax2.plot(neat_gens, neat_avg_fit, color=C_NEAT, alpha=0.2, lw=1)
        ax2.plot(neat_ma_gens, neat_ma_avg, color=C_NEAT, lw=2.0, ls="--",
                 label="NEAT mean fitness")
    ax2.set_title("B — Mean Population Fitness per Generation  (GA vs NEAT)")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Mean Fitness")
    ax2.legend(fontsize=8, loc="upper left")

    # C — score progression
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(gens, best_score, s=7, color=C_SCORE, alpha=0.35, label="GA")
    ax3.plot(ma_gens, ma_score, color=C_SCORE, lw=2.0)
    ax3.axhline(0.3, color=C_RAND, ls=":", lw=1.2)
    if neat_gens is not None:
        ax3.scatter(neat_gens, neat_best_score, s=7, color=C_NEAT, alpha=0.35,
                    marker="^", label="NEAT")
        ax3.plot(neat_ma_gens, neat_ma_score, color=C_NEAT, lw=1.8, ls="--")
        ax3.legend(fontsize=7, loc="upper left")
    ax3.set_title("C — Best Score per Gen")
    ax3.set_ylabel("Food Eaten")
    ax3.set_xlabel("Generation")

    ax4 = fig.add_subplot(gs[1, 2])
    thr = [5, 10, 25, 50, 100, 150, 200, 250]
    fg_ga = []
    for t in thr:
        idx = int(np.argmax(best_score >= t))
        fg_ga.append(int(gens[idx]) if best_score[idx] >= t else 185)
    clrs = plt.cm.RdYlGn(np.linspace(0.15, 0.9, len(thr)))
    y4 = np.arange(len(thr))
    if neat_gens is not None:
        h4 = 0.35
        ax4.barh(y4 - h4 / 2, fg_ga, h4, color=clrs, edgecolor="white", label="GA")
        fg_neat = []
        for t in thr:
            idx = int(np.argmax(neat_best_score >= t))
            fg_neat.append(int(neat_gens[idx]) if neat_best_score[idx] >= t else
                           int(neat_gens[-1]))
        ax4.barh(y4 + h4 / 2, fg_neat, h4, color=C_NEAT, edgecolor="white",
                 alpha=0.75, label="NEAT")
        ax4.legend(fontsize=7, loc="lower right")
    else:
        ax4.barh(y4, fg_ga, 0.6, color=clrs, edgecolor="white")
    ax4.set_yticks(y4)
    ax4.set_yticklabels([str(t) for t in thr])
    ax4.set_title("D — Gens to Score Threshold")
    ax4.set_xlabel("Generation")
    ax4.set_ylabel("Score Threshold")
    ax4.invert_yaxis()

    fig.suptitle(
        "Snake GA — Neuroevolution Training Overview\n"
        "Exp 1 Baseline: pop=150, mut_rate=0.15, arch=[11,16,3], 180 generations",
        fontsize=12, fontweight="bold", y=1.01,
    )

    path = os.path.join(PLOTS_DIR, "fig6_combined_overview.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [ok] {path}")


if __name__ == "__main__":
    print(f"Loaded {len(_raw)} generations of data.")
    print("Generating figures...\n")
    fig1_fitness_curves()
    fig2_score_progression()
    fig3_alltime_records()
    fig4_final_comparison()
    fig5_convergence_milestones()
    fig6_combined_overview()
    print(f"\nAll 6 figures saved to {PLOTS_DIR}")
    print("\nSummary stats:")
    print(f"  All-time best score  : {int(np.max(best_score))}")
    print(f"  All-time best fitness: {float(np.max(alltime_fit)):.1f}")
    print(f"  Final gen avg fitness: {avg_fit[-1]:.1f}")
    print(f"  Avg score (last 10 gens): {np.mean(best_score[-10:]):.1f}")
    print(f"  Gen 1 avg fitness (~random): {avg_fit[0]:.1f}")
    print(f"  Improvement ratio (final/gen1 mean): {avg_fit[-1]/avg_fit[0]:.0f}×")
    if _neat_raw is not None:
        neat_scores = np.array([r[3] for r in _neat_raw])
        neat_fits   = np.array([r[1] for r in _neat_raw])
        print(f"\nNEAT (neat_training_log.csv):")
        print(f"  All-time best score  : {int(np.max(neat_scores))}")
        print(f"  All-time best fitness: {float(np.max([r[4] for r in _neat_raw])):.1f}")
        print(f"  Avg score (last 10 gens): {np.mean(neat_scores[-10:]):.1f}")
