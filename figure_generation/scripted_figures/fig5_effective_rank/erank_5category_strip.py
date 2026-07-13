#!/usr/bin/env python3
"""
erank/M 5-category strip/jitter plot for VG-iT presentation.

5 categories:
  1. Extreme Cohesion (finch_like)
  2. Moderate Cohesion (coarsening, mi_based)
  3. No Strategy (ordered)
  4. Control (random)
  5. Diversity (anti_clustering, score_stratified, maximin_dispersion)

Data: results_canonical/representation_diagnostics/erank/erank_allpl_analysis.json
Output: figures/erank_5category_strip.png (600 DPI)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

# ── paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
DEPOSIT_ROOT = SCRIPT_DIR.parents[2]
DATA_PATH = DEPOSIT_ROOT / "results_canonical" / "representation_diagnostics" / "erank" / "erank_allpl_analysis.json"
OUT_PATH = SCRIPT_DIR / "figures" / "erank_5category_strip.png"

# ── 5-category definition ──
CATEGORIES = {
    "finch_like":          ("Extreme Cohesion",  0),
    "coarsening":          ("Moderate Cohesion", 1),
    "mi_based":            ("Moderate Cohesion", 1),
    "ordered":             ("No Strategy",       2),
    "random":              ("Control",           3),
    "anti_clustering":     ("Diversity",         4),
    "score_stratified":    ("Diversity",         4),
    "maximin_dispersion":  ("Diversity",         4),
}

# Category display order and colors
CAT_ORDER = [
    "Extreme Cohesion",
    "Moderate Cohesion",
    "No Strategy",
    "Control",
    "Diversity",
]
CAT_COLORS = {
    "Extreme Cohesion":  "#d62728",  # red
    "Moderate Cohesion": "#ff7f0e",  # orange
    "No Strategy":       "#2ca02c",  # green
    "Control":           "#7f7f7f",  # gray
    "Diversity":         "#1f77b4",  # blue
}
# Labels for legend
CAT_LABELS_KR = {
    "Extreme Cohesion":  "Extreme Cohesion (agglomerative)",
    "Moderate Cohesion": "Moderate Cohesion (coarsening, MI-based)",
    "No Strategy":       "No Strategy (ordered)",
    "Control":           "Control (random)",
    "Diversity":         "Diversity (anti-clust., score-strat., maximin)",
}

# Method display order (left to right, grouped by category)
METHOD_ORDER = [
    "finch_like",
    "coarsening", "mi_based",
    "ordered",
    "random",
    "anti_clustering", "score_stratified", "maximin_dispersion",
]
METHOD_LABELS = {
    "finch_like": "agglom.",
    "coarsening": "coarsen.",
    "mi_based": "MI-based",
    "ordered": "ordered",
    "random": "random",
    "anti_clustering": "anti-clust.",
    "score_stratified": "score-strat.",
    "maximin_dispersion": "maximin",
}

# Dataset display names
DS_NAMES = {
    "electricity": "Electricity (N=321, G=16, M=21)",
    "solar_AL":    "Solar-AL (N=137, G=16, M=9)",
    "traffic":     "Traffic (N=862, G=32, M=27)",
}
# Panel order: Solar-AL first, Electricity second, Traffic third
DS_ORDER = ["solar_AL", "electricity", "traffic"]


def load_bad_blocks():
    """Load bad blocks (MSE > 0.6) for block-level cleaning."""
    import pandas as pd
    csv_path = DEPOSIT_ROOT / "results_canonical" / "rq2a_grouping" / "grouping_invariance_results.csv"
    df = pd.read_csv(csv_path)
    df['block'] = df['dataset'] + '_' + df['pred_len'].astype(str) + '_' + df['seed'].astype(str)
    bad = set(df[df['MSE'] > 0.6]['block'].unique())
    return bad


def load_data():
    """Load JSON and extract overall erank_ratio per method/seed/pl (2,768 clean runs only)."""
    with open(DATA_PATH) as f:
        raw = json.load(f)

    bad_blocks = load_bad_blocks()

    # result[dataset][method] = list of erank_ratio values (clean runs only)
    result = {}
    n_total = 0
    n_removed = 0
    for ds in DS_ORDER:
        ds_name_csv = ds  # matches CSV dataset column
        result[ds] = {}
        for method in METHOD_ORDER:
            vals = []
            for key, layers in raw[ds][method].items():
                # key = "s2021_pl96" etc.
                parts = key.split('_')
                seed = parts[0][1:]   # "s2021" -> "2021"
                pl = parts[1][2:]     # "pl96" -> "96"
                block_id = f"{ds_name_csv}_{pl}_{seed}"
                n_total += 1
                if block_id in bad_blocks:
                    n_removed += 1
                    continue
                if "overall" in layers:
                    vals.append(layers["overall"]["effective_rank_ratio"])
            result[ds][method] = np.array(vals)

    print(f"Loaded: {n_total} total, {n_removed} removed (bad blocks), {n_total - n_removed} clean")
    return result


def make_figure(data):
    """Create 3-panel strip plot."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2), sharey=True)
    fig.subplots_adjust(
        wspace=0.05, left=0.055, right=0.98,
        top=0.90, bottom=0.25,
    )

    rng = np.random.default_rng(42)

    for ax_idx, ds in enumerate(DS_ORDER):
        ax = axes[ax_idx]
        ax.set_title(DS_NAMES[ds], fontsize=16, fontweight="bold", pad=10)

        for m_idx, method in enumerate(METHOD_ORDER):
            vals = data[ds][method]
            cat_name = CATEGORIES[method][0]
            color = CAT_COLORS[cat_name]

            # Jitter x positions
            jitter = rng.uniform(-0.28, 0.28, size=len(vals))
            x_pos = m_idx + jitter

            ax.scatter(
                x_pos, vals,
                c=color, alpha=0.45, s=25, edgecolors="none", zorder=2,
            )
            # Median marker (white fill + black edge = stands out against any color)
            med = np.median(vals)
            ax.scatter(
                [m_idx], [med],
                c="white", s=120, edgecolors="black", linewidths=2.0,
                marker="D", zorder=4,
            )
            # Small colored dot inside the diamond
            ax.scatter(
                [m_idx], [med],
                c=color, s=30, edgecolors="none",
                marker="o", zorder=5,
            )

        # Category separator lines
        for sep_x in [0.5, 2.5, 3.5, 4.5]:
            ax.axvline(sep_x, color="#cccccc", linewidth=0.8, linestyle="--", zorder=0)

        # X-axis
        ax.set_xticks(range(len(METHOD_ORDER)))
        ax.set_xticklabels(
            [METHOD_LABELS[m] for m in METHOD_ORDER],
            rotation=45, ha="right", fontsize=13,
        )
        ax.set_xlim(-0.5, len(METHOD_ORDER) - 0.5)

        # Grid: horizontal only, subtle
        ax.yaxis.grid(True, alpha=0.25, linewidth=0.5)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)

        # Spine cleanup
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Y-axis label on first panel only
    axes[0].set_ylabel("erank / M", fontsize=15, fontweight="bold")
    axes[0].set_ylim(-0.02, 0.58)

    # Legend (shared, below the figure)
    legend_handles = [
        mpatches.Patch(facecolor=CAT_COLORS[c], label=CAT_LABELS_KR[c])
        for c in CAT_ORDER
    ]
    legend_handles.append(
        Line2D([0], [0], marker='D', color='w', markerfacecolor='white',
               markersize=8, markeredgecolor='black', markeredgewidth=1.5,
               label='Median', linestyle='None')
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=12,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=600, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    data = load_data()
    # Print summary stats
    for ds in DS_ORDER:
        print(f"\n=== {ds} ===")
        for method in METHOD_ORDER:
            vals = data[ds][method]
            cat = CATEGORIES[method][0]
            print(f"  {method:20s} [{cat:20s}]  "
                  f"mean={vals.mean():.3f}  std={vals.std():.3f}  "
                  f"min={vals.min():.3f}  max={vals.max():.3f}  n={len(vals)}")
    print()
    make_figure(data)
