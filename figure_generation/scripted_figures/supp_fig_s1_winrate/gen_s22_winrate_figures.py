"""
Generate S22 win rate figures: heatmap and grouped bar.
Data from perds_sign_homogeneity_report.md Section 1.
Clean dataset: 346 blocks, 2768 runs (block-level max MSE > 0.6 removed).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import os

# ── Data from report Section 1-1 ──────────────────────────────────────────────
# ordered win rate against each opponent, per dataset
# Format: {opponent: {dataset: win_rate}}

data = {
    "anti_clustering":   {"ECL": 0.8750, "Solar": 0.7917, "Traffic": 0.6509},
    "coarsening":        {"ECL": 0.7500, "Solar": 0.8750, "Traffic": 0.7264},
    "finch_like":        {"ECL": 0.8750, "Solar": 0.8500, "Traffic": 0.6792},
    "maximin_dispersion":{"ECL": 0.8583, "Solar": 0.7750, "Traffic": 0.4340},
    "mi_based":          {"ECL": 0.8667, "Solar": 0.8833, "Traffic": 0.5377},
    "random":            {"ECL": 0.8583, "Solar": 0.7833, "Traffic": 0.3019},
    "score_stratified":  {"ECL": 0.8167, "Solar": 0.7583, "Traffic": 0.1981},
}

opponents = list(data.keys())
datasets = ["Solar", "ECL", "Traffic"]

# Display labels (shorter)
opp_labels = {
    "anti_clustering":    "anti-\nclustering",
    "coarsening":         "coarsening",
    "finch_like":         "agglomerative",
    "maximin_dispersion": "maximin-\ndispersion",
    "mi_based":           "MI-based",
    "random":             "random",
    "score_stratified":   "score-\nstratified",
}

ds_colors = {"ECL": "#7570b3", "Solar": "#1b9e77", "Traffic": "#e7298a"}

out_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(out_dir, exist_ok=True)

# ── Option A: Heatmap ─────────────────────────────────────────────────────────
fig_h, ax_h = plt.subplots(figsize=(12, 4.0))

# Build matrix: rows=datasets (ECL, Solar, Traffic), cols=opponents
mat = np.array([[data[opp][ds] for opp in opponents] for ds in datasets])

norm = TwoSlopeNorm(vmin=0.10, vcenter=0.50, vmax=1.00)
cmap = plt.cm.RdBu  # red=low (opponent wins), blue=high (ordered wins)

im = ax_h.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

# Colorbar
cbar = fig_h.colorbar(im, ax=ax_h, fraction=0.03, pad=0.02)
cbar.set_label("Ordered Win Rate", fontsize=11)
cbar.set_ticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# Axes ticks
ax_h.set_xticks(range(len(opponents)))
ax_h.set_xticklabels([opp_labels[o] for o in opponents], fontsize=10)
ax_h.set_yticks(range(len(datasets)))
ax_h.set_yticklabels(datasets, fontsize=12, fontweight="bold")
for tick, ds in zip(ax_h.get_yticklabels(), datasets):
    tick.set_color(ds_colors[ds])

# Annotate each cell
for i, ds in enumerate(datasets):
    for j, opp in enumerate(opponents):
        val = mat[i, j]
        # White text on dark cells, dark text on light cells
        bg = cmap(norm(val))
        # luminance check
        lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
        txt_color = "white" if lum < 0.45 else "black"
        sig = ""
        ax_h.text(j, i, f"{val:.3f}{sig}", ha="center", va="center",
                  fontsize=11, fontweight="bold", color=txt_color)

# Reference line annotation
ax_h.axhline(y=-0.5, color="none")  # padding

ax_h.set_title("Ordered Win Rate vs. Each Opponent by Dataset  (346 blocks, 2768 runs)",
               fontsize=13, fontweight="bold", pad=10)

# 0.5 line note in subtitle
fig_h.text(0.5, 0.01,
           "Color scale: red = ordered loses (< 0.5)  |  white = even (0.5)  |  blue = ordered wins (> 0.5)",
           ha="center", fontsize=9, color="gray")

plt.tight_layout(rect=[0, 0.04, 1, 1])
heatmap_path = os.path.join(out_dir, "s22_winrate_heatmap.png")
fig_h.savefig(heatmap_path, dpi=600, bbox_inches="tight")
plt.close(fig_h)
print(f"Saved heatmap: {heatmap_path}")


# ── Option B: Grouped bar ─────────────────────────────────────────────────────

# Reorder: 응집(finch_like, coarsening, mi_based), 대조군(random), 다양성(score_strat, anti_clust, maximin)
opp_order = ["finch_like", "coarsening", "mi_based",
             "random",
             "score_stratified", "anti_clustering", "maximin_dispersion"]

fig_b, ax_b = plt.subplots(figsize=(12, 5.0))

n_opp = len(opp_order)
n_ds = len(datasets)
x = np.arange(n_opp)
width = 0.24
offsets = [-width, 0, width]

bars_all = []
for di, ds in enumerate(datasets):
    vals = [data[opp][ds] for opp in opp_order]
    bars = ax_b.bar(x + offsets[di], vals, width,
                    label=ds, color=ds_colors[ds], alpha=0.85,
                    edgecolor="white", linewidth=0.6)
    bars_all.append((bars, vals, ds))

# Annotate bar values
for bars, vals, ds in bars_all:
    for bar, val in zip(bars, vals):
        ypos = bar.get_height() + 0.012
        if ypos > 1.02:
            ypos = bar.get_height() - 0.045
        ax_b.text(bar.get_x() + bar.get_width() / 2, ypos,
                  f"{val:.2f}", ha="center", va="bottom",
                  fontsize=7.5, fontweight="bold",
                  color=ds_colors[ds])

# 0.5 reference line
ax_b.axhline(0.5, color="black", linewidth=1.4, linestyle="--",
             label="0.5 (equal win rate)", zorder=5)

# Category color strip below x-axis (thin bar, like S21 sidebar)
cat_strip_colors = {
    "finch_like": "#d62728", "coarsening": "#ff7f0e", "mi_based": "#ff7f0e",
    "random": "#7f7f7f",
    "score_stratified": "#1f77b4", "anti_clustering": "#1f77b4", "maximin_dispersion": "#1f77b4",
}
for i, opp in enumerate(opp_order):
    ax_b.bar(i, -0.018, width=0.85, bottom=-0.009,
             color=cat_strip_colors[opp], clip_on=False, zorder=10)

ax_b.set_xticks(x)
ax_b.set_xticklabels([opp_labels[o] for o in opp_order], fontsize=10)
ax_b.tick_params(axis='x', pad=12)
ax_b.set_ylabel("Ordered Win Rate", fontsize=12)
ax_b.set_ylim(0, 1.13)
ax_b.set_yticks([0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax_b.yaxis.grid(True, linestyle=":", alpha=0.5)
ax_b.set_axisbelow(True)

ax_b.set_title("Ordered Win Rate vs. Each Opponent by Dataset  (346 blocks, 2768 runs)",
               fontsize=13, fontweight="bold")

# Legend — 2x2 layout, formal dataset names
ds_formal = {"Solar": "Solar-AL", "ECL": "Electricity", "Traffic": "Traffic"}
legend_patches = [
    mpatches.Patch(color=ds_colors[ds], label=ds_formal[ds])
    for ds in datasets
]
legend_patches.append(
    plt.Line2D([0], [0], color="black", linewidth=1.4, linestyle="--",
               label="0.5 (equal win rate)")
)
legend = ax_b.legend(handles=legend_patches, fontsize=10,
                     loc="upper right", framealpha=0.9, ncol=2)
for text, ds in zip(legend.get_texts(), datasets):
    text.set_color(ds_colors[ds])
    text.set_fontweight("bold")

plt.tight_layout()
bar_path = os.path.join(out_dir, "s22_winrate_bar.png")
fig_b.savefig(bar_path, dpi=600, bbox_inches="tight")
plt.close(fig_b)
print(f"Saved bar chart: {bar_path}")
