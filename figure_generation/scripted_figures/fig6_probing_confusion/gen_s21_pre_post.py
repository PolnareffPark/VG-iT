"""
S21 Pre/Post-pooling probing heatmaps.
Based on probing_hm_h1.py (same frame/style).

Figure 1: Pre-pooling (1 panel) | Post-pooling (1 panel) — aggregated over 12 conditions
Figure 2: Pre (3 DS panels, top row) | Post (3 DS panels, bottom row) — 2×3 grid
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
DEPOSIT_ROOT = SCRIPT_DIR.parents[2]
PRE_POST_JSON = DEPOSIT_ROOT / "results_canonical" / "representation_diagnostics" / "pre_pooling_probing" / "pre_pooling_probing_results.json"
OUT_FIG1 = SCRIPT_DIR / "figures" / "s21_pre_post_agg.png"
OUT_FIG2 = SCRIPT_DIR / "figures" / "s21_pre_post_per_ds.png"
OUT_FIG2_PDF = SCRIPT_DIR / "figures" / "s21_pre_post_per_ds.pdf"

MATRIX_ORDER = ['ordered', 'random', 'finch_like', 'coarsening', 'mi_based',
                'anti_clustering', 'score_stratified', 'maximin_dispersion']

DISPLAY_ORDER = ['finch_like', 'coarsening', 'mi_based', 'ordered', 'random',
                 'score_stratified', 'maximin_dispersion', 'anti_clustering']
DISPLAY_LABELS = ['F', 'C', 'M', 'O', 'R', 'S', 'X', 'A']

CATEGORY_COLORS = [
    '#d62728',  # F
    '#ff7f0e',  # C
    '#ff7f0e',  # M
    '#2ca02c',  # O
    '#7f7f7f',  # R
    '#1f77b4',  # S
    '#1f77b4',  # X
    '#1f77b4',  # A
]

DS_CONDITIONS = {
    'solar_AL':    [f'solar_AL_pl{pl}' for pl in [96, 192, 336, 720]],
    'electricity': [f'electricity_pl{pl}' for pl in [96, 192, 336, 720]],
    'traffic':     [f'traffic_pl{pl}' for pl in [96, 192, 336, 720]],
}

DS_TITLES = {
    'solar_AL': 'Solar-AL (N=137)',
    'electricity': 'Electricity (N=321)',
    'traffic': 'Traffic (N=862)',
}


def reindex_cm(cm, matrix_order, display_order):
    cm = np.array(cm, dtype=float)
    m_idx = {m: i for i, m in enumerate(matrix_order)}
    idx = [m_idx[m] for m in display_order]
    return cm[np.ix_(idx, idx)]


def row_normalise(cm):
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.where(row_sums > 0, cm / row_sums, 0.0)


def draw_panel(fig, ax, cm_raw, title, show_ylabel=True, show_cbar=True, fontsize_cell=13, bold_diagonal=True):
    """Draw a single heatmap panel. Color = row proportion (0-100%), text = raw counts."""
    n = len(DISPLAY_ORDER)
    STRIP_W = 0.15

    # Reindex
    cm = reindex_cm(cm_raw, MATRIX_ORDER, DISPLAY_ORDER)
    cm_norm = row_normalise(cm)

    # Color by row proportion (same as probing_hm_h1.py)
    im = ax.imshow(cm_norm, vmin=0.0, vmax=1.0,
                   cmap='Blues', aspect='equal',
                   extent=[-0.5, n - 0.5, n - 0.5, -0.5])

    # Annotate cells with RAW COUNTS
    for i in range(n):
        for j in range(n):
            val = int(cm[i, j])
            weight = 'bold' if bold_diagonal and i == j else 'normal'
            ax.text(j, i, str(val),
                    ha='center', va='center',
                    fontsize=fontsize_cell, color='#222222', fontweight=weight,
                    fontfamily='DejaVu Sans')

    # Diagonal red dashed borders
    for d in range(n):
        rect = plt.Rectangle((d - 0.5, d - 0.5), 1, 1,
                              fill=False, edgecolor='#cc0000',
                              linewidth=2.0, linestyle='--', zorder=4)
        ax.add_patch(rect)

    # Left color strips
    for i, color in enumerate(CATEGORY_COLORS):
        rect = mpatches.FancyBboxPatch(
            (-(STRIP_W + 0.5), i - 0.48), STRIP_W, 0.96,
            boxstyle="square,pad=0",
            facecolor=color, edgecolor='white', linewidth=0.5,
            transform=ax.transData, clip_on=False)
        ax.add_patch(rect)

    # Top color strips
    for j, color in enumerate(CATEGORY_COLORS):
        rect = mpatches.FancyBboxPatch(
            (j - 0.48, -(STRIP_W + 0.5)), 0.96, STRIP_W,
            boxstyle="square,pad=0",
            facecolor=color, edgecolor='white', linewidth=0.5,
            transform=ax.transData, clip_on=False)
        ax.add_patch(rect)

    ax.set_xlim(-0.5 - STRIP_W - 0.05, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5 - STRIP_W - 0.05)

    ax.set_xticks(range(n))
    ax.set_xticklabels(DISPLAY_LABELS, fontsize=16, fontfamily='DejaVu Sans')
    ax.set_yticks(range(n))
    ax.set_yticklabels(DISPLAY_LABELS if show_ylabel else [''] * n,
                       fontsize=16, fontfamily='DejaVu Sans')
    ax.set_xlabel('Predicted', fontsize=15, labelpad=6)
    if show_ylabel:
        ax.set_ylabel('True', fontsize=15, labelpad=18)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)

    if show_cbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Row proportion (%)', fontsize=13)
        cbar.ax.tick_params(labelsize=11)
        cbar.set_ticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        cbar.set_ticklabels(['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100'])
        cbar.ax.axhline(0.125, color='#cc0000', linewidth=1.2, linestyle='--')
        cbar.ax.text(1.8, 0.125, 'chance\n12.5%', va='center', fontsize=9,
                     color='#cc0000', fontweight='bold', transform=cbar.ax.transData)

    return im


def trim_png(path, pad=90, threshold=248):
    im = Image.open(path).convert('RGBA')
    arr = np.asarray(im)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    mask = (alpha > 0) & np.any(rgb < threshold, axis=2)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        im.save(path)
        return
    box = (
        max(int(cols[0]) - pad, 0),
        max(int(rows[0]) - pad, 0),
        min(int(cols[-1]) + pad + 1, im.width),
        min(int(rows[-1]) + pad + 1, im.height),
    )
    im.crop(box).save(path)


def make_legend(fig, ncol=3, fontsize=14, y=-0.01, diversity_last_row=False, compact=False):
    if compact:
        legend_items = [
            mpatches.Patch(color='#d62728', label='Extreme Cohesion (F, agglomerative)'),
            mpatches.Patch(color='#ff7f0e', label='Moderate Cohesion (C, coarsening; M, MI-based)'),
            mpatches.Patch(color='#2ca02c', label='No Strategy (O, ordered)'),
            mpatches.Patch(color='#7f7f7f', label='Control (R, random)'),
            mpatches.Patch(color='#1f77b4', label='Diversity (S, score-strat.; X, maximin; A, anti-clust.)'),
        ]
    else:
        legend_items = [
            mpatches.Patch(color='#d62728', label='Extreme Cohesion (F, agglomerative)'),
            mpatches.Patch(color='#ff7f0e', label='Moderate Cohesion (C, coarsening | M, MI-based)'),
            mpatches.Patch(color='#2ca02c', label='No Strategy (O, ordered)'),
            mpatches.Patch(color='#7f7f7f', label='Control (R, random)'),
            mpatches.Patch(color='#1f77b4', label='Diversity (S, score-stratified | X, maximin-dispersion | A, anti-clustering)'),
        ]
    if diversity_last_row and ncol == 2:
        # Matplotlib fills legend columns top-to-bottom; this order makes Diversity occupy row 3 alone.
        legend_items = [legend_items[i] for i in [0, 2, 4, 1, 3]]
    legend_kwargs = {}
    if compact:
        legend_kwargs = dict(columnspacing=1.0, handlelength=1.1,
                             handletextpad=0.4, labelspacing=0.55, borderaxespad=0.0)
    fig.legend(handles=legend_items, loc='lower center', ncol=ncol,
               fontsize=fontsize, frameon=False, bbox_to_anchor=(0.5, y),
               **legend_kwargs)

def aggregate_cm(data, condition_keys, level):
    """Sum confusion matrices across conditions."""
    cm = None
    for key in condition_keys:
        if key in data:
            arr = np.array(data[key][level]['confusion_matrix'], dtype=float)
            cm = arr if cm is None else cm + arr
    return cm


def main():
    for out_path in (OUT_FIG1, OUT_FIG2, OUT_FIG2_PDF):
        out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(PRE_POST_JSON) as f:
        data = json.load(f)

    all_keys = sorted(data.keys())

    # -- Figure 1: Aggregated pre vs post (2 panels) --
    pre_agg = aggregate_cm(data, all_keys, 'pre_pooling')
    post_agg = aggregate_cm(data, all_keys, 'post_pooling')

    n_total = int(pre_agg.sum())
    n_per_method = int(pre_agg.sum() / 8)
    chance_count = n_per_method // 8

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig1.subplots_adjust(wspace=0.30, left=0.06, right=0.94, top=0.82, bottom=0.13)

    pre_acc = np.trace(reindex_cm(pre_agg, MATRIX_ORDER, DISPLAY_ORDER)) / n_total * 100
    post_acc = np.trace(reindex_cm(post_agg, MATRIX_ORDER, DISPLAY_ORDER)) / n_total * 100

    draw_panel(fig1, ax1, pre_agg,
               f'Pre-Pooling (before M→1)\nAcc: {pre_acc:.1f}% | chance: 12.5% ({chance_count}/cell)')
    draw_panel(fig1, ax2, post_agg,
               f'Post-Pooling (after M→1)\nAcc: {post_acc:.1f}% | chance: 12.5% ({chance_count}/cell)',
               show_ylabel=False)

    # Title removed per user request — figure is self-explanatory
    make_legend(fig1)
    fig1.savefig(OUT_FIG1, dpi=600, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved: {OUT_FIG1}")

    # -- Figure 2: Per-DS, 2×3 grid (top=pre, bottom=post) --
    fig2, axes = plt.subplots(2, 3, figsize=(18, 16))
    fig2.subplots_adjust(wspace=0.10, hspace=0.35, left=0.05, right=0.95, top=0.88, bottom=0.18)

    ds_order = ['solar_AL', 'electricity', 'traffic']

    for col, ds in enumerate(ds_order):
        keys = DS_CONDITIONS[ds]
        pre_cm = aggregate_cm(data, keys, 'pre_pooling')
        post_cm = aggregate_cm(data, keys, 'post_pooling')

        n_ds = int(pre_cm.sum())
        n_method = n_ds // 8
        chance_ds = n_method // 8

        pre_ds_acc = np.trace(reindex_cm(pre_cm, MATRIX_ORDER, DISPLAY_ORDER)) / n_ds * 100
        post_ds_acc = np.trace(reindex_cm(post_cm, MATRIX_ORDER, DISPLAY_ORDER)) / n_ds * 100

        draw_panel(fig2, axes[0, col], pre_cm,
                   f'Pre - {DS_TITLES[ds]}\nAcc: {pre_ds_acc:.1f}% (chance: {chance_ds}/cell)',
                   show_ylabel=(col == 0), show_cbar=(col == 2), fontsize_cell=24, bold_diagonal=False)
        draw_panel(fig2, axes[1, col], post_cm,
                   f'Post - {DS_TITLES[ds]}\nAcc: {post_ds_acc:.1f}% (chance: {chance_ds}/cell)',
                   show_ylabel=(col == 0), show_cbar=(col == 2), fontsize_cell=24, bold_diagonal=False)

    make_legend(fig2, ncol=2, fontsize=20, y=0.025, diversity_last_row=True, compact=True)
    fig2.savefig(OUT_FIG2, dpi=600, bbox_inches='tight', pad_inches=0)
    fig2.savefig(OUT_FIG2_PDF, dpi=600, bbox_inches='tight', pad_inches=0)
    trim_png(OUT_FIG2)
    plt.close(fig2)
    print(f"Saved: {OUT_FIG2}")
    print(f"Saved: {OUT_FIG2_PDF}")


if __name__ == '__main__':
    main()
