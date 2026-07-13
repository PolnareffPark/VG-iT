"""Reproduce Figure 3 from the canonical VG-iT result files.

Panels (a-c) aggregate the shifted-grouping + FiLM G-sensitivity runs over four
prediction horizons and seeds 2021-2023. Panel (d) uses the dataset-specific
encoder depths and rule-derived group counts reported in the main manuscript.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as ticker
import numpy as np
import os
from pathlib import Path
import pandas as pd

# ============================================================
# Canonical inputs and (a-c) per-dataset performance sweep
# ============================================================
HERE = Path(__file__).resolve().parent


def locate(*relative_paths):
    for base in (HERE, *HERE.parents):
        for relative in relative_paths:
            candidate = base / relative
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Could not locate any canonical input: {relative_paths}")


GSENS_PATH = locate(
    "results_canonical/rq2a_grouping/g_sensitivity_results.csv",
    "test_results/published/rq2a_grouping/g_sensitivity_results.csv",
)
GSENS_FLOPS_PATH = locate(
    "results_canonical/rq2a_grouping/g_sensitivity_flops.csv",
    "test_results/published/rq2a_grouping/g_sensitivity_flops.csv",
)
ACADEMIC_BASELINE_PATH = locate(
    "results_canonical/shared_consolidated_5seed/academic_baselines.csv",
    "test_results/published/shared/_consolidated_5seed/academic_baselines.csv",
)
INDUSTRIAL_BASELINE_PATH = locate(
    "results_canonical/shared_consolidated_5seed/industrial_baselines.csv",
    "test_results/published/shared/_consolidated_5seed/industrial_baselines.csv",
)
EFFICIENCY_PATH = locate(
    "results_canonical/shared_consolidated_5seed/efficiency_profile.csv",
    "test_results/published/shared/_consolidated_5seed/efficiency_profile.csv",
)

gsens = pd.read_csv(GSENS_PATH)
gsens_flops = pd.read_csv(GSENS_FLOPS_PATH)
academic_baselines = pd.read_csv(ACADEMIC_BASELINE_PATH)
industrial_baselines = pd.read_csv(INDUSTRIAL_BASELINE_PATH)
efficiency = pd.read_csv(EFFICIENCY_PATH)
gsens["MSE"] = pd.to_numeric(gsens["MSE"], errors="raise")
gsens_flops["FLOPs_G"] = pd.to_numeric(gsens_flops["FLOPs_G"], errors="raise")
for frame in (academic_baselines, industrial_baselines):
    frame["MSE"] = pd.to_numeric(frame["MSE"], errors="coerce")
efficiency["FLOPs_G"] = pd.to_numeric(efficiency["FLOPs_G"], errors="coerce")


def practical_g(n):
    """Equation (13): nearest power of two to sqrt(N)."""
    return int(2 ** np.floor(np.log2(np.sqrt(n)) + 0.5))


def panel_data(dataset, n, g_values, baseline_frame, y_extent=None):
    sweep = gsens.loc[gsens["dataset"].str.lower() == dataset.lower()].copy()
    assert len(sweep) == 60, f"Expected 60 G-sensitivity rows for {dataset}, got {len(sweep)}"
    assert set(sweep["seed"].astype(int)) == {2021, 2022, 2023}
    assert set(sweep["pred_len"].astype(int)) == {96, 192, 336, 720}

    excluded = sweep["MSE"] > 0.6
    if dataset.lower() == "traffic":
        assert int(excluded.sum()) == 3, "Traffic clean rule must exclude exactly three runs"
        sweep = sweep.loc[~excluded]

    grouped = sweep.groupby("num_groups", sort=True).agg(mse=("MSE", "mean"), count=("MSE", "size"))
    assert list(grouped.index.astype(int)) == g_values
    if dataset.lower() == "traffic":
        assert list(grouped["count"].astype(int)) == [12, 12, 12, 10, 11]
    else:
        assert (grouped["count"] == 12).all()

    measured = gsens_flops.loc[gsens_flops["dataset"].str.lower() == dataset.lower()].copy()
    measured = measured.set_index("num_groups").reindex(g_values)
    assert measured["FLOPs_G"].notna().all()
    assert (measured["use_shifted_grouping"] == 1).all()
    assert (measured["use_film_broadcast"] == 1).all()

    baseline = baseline_frame.loc[
        (baseline_frame["model"] == "iTransformer")
        & (baseline_frame["dataset"].str.lower() == dataset.lower())
        & (baseline_frame["seed"].astype(int).isin([2021, 2022, 2023]))
    ]
    assert len(baseline) == 12, f"Expected 12 iTransformer rows for {dataset}, got {len(baseline)}"

    it_eff = efficiency.loc[
        (efficiency["model"] == "iTransformer")
        & (efficiency["dataset"].str.lower() == dataset.lower())
        & (efficiency["pred_len"].astype(int) == 96)
    ]
    assert len(it_eff) == 1, f"Expected one PL=96 efficiency row for {dataset}"

    return {
        "it_flops": float(it_eff.iloc[0]["FLOPs_G"]),
        "it_mse": float(baseline["MSE"].mean()),
        "g_vals": g_values,
        "flops": measured["FLOPs_G"].astype(float).tolist(),
        "mse": grouped["mse"].astype(float).tolist(),
        "star_g": practical_g(n),
        "y_extent": y_extent,
    }


datasets = {
    "Solar-AL (N=137)": panel_data(
        "solar_AL", 137, [4, 8, 16, 32, 64], academic_baselines
    ),
    "Traffic (N=862)": panel_data(
        "traffic", 862, [8, 16, 32, 64, 128], academic_baselines
    ),
    "BDG2 (N=2817)": panel_data(
        "bdg2", 2817, [16, 32, 64, 128, 256], industrial_baselines, y_extent=0.60
    ),
}
datasets["BDG2 (N=2817)"].update(y_major_step=0.2, y_minor_step=0.1)
g_palette = {
    4: '#E53935', 8: '#FB8C00', 16: '#FDD835', 32: '#43A047',
    64: '#1E88E5', 128: '#8E24AA', 256: '#6D4C41',
}
# label offsets (dx, dy in points, ha) -- scaled down for small cells
SC = 0.6
lbl = {
    'Solar-AL (N=137)': {4:(10,5,'left'),64:(10,-5,'left'),8:(10,-11,'left'),32:(10,5,'left'),16:(10,0,'left')},
    'Traffic (N=862)':  {8:(10,0,'left'),16:(10,6,'left'),32:(-10,-5,'right'),64:(10,-5,'left'),128:(10,0,'left')},
    'BDG2 (N=2817)':    {16:(-10,0,'right'),32:(10,0,'left'),64:(10,0,'left'),128:(10,0,'left'),256:(10,0,'left')},
}

# ============================================================
# (d) Main-setting group count vs attention FLOPs landscape
# ============================================================
D_MODEL = 512
def attn_flops_vg(N, G, L, D=D_MODEL):
    return 4 * L * D * (N**2 / G + G**2)
def attn_flops_it(N, L, D=D_MODEL):
    return 4 * L * D * N**2
def g_star(N):
    return (N**2 / 2) ** (1 / 3)
def to_gflops(x):
    return x / 1e9
G_range = np.logspace(np.log10(2), np.log10(256), 800)
G_ticks = [2, 4, 8, 16, 32, 64, 128, 256]
N_specs = [
    (137,  'Solar-AL',    '#4292c6', practical_g(137),  2),
    (228,  'KAMP',        '#2171b5', practical_g(228),  3),
    (238,  'CARE',        '#08519c', practical_g(238),  3),
    (244,  'BASF',        '#08306b', practical_g(244),  3),
    (321,  'Electricity', '#74c476', practical_g(321),  3),
    (862,  'Traffic',     '#005a23', practical_g(862),  4),
    (2144, 'SDWPF',       '#fc9272', practical_g(2144), 3),
    (2362, 'ASHRAE',      '#cb181d', practical_g(2362), 3),
    (2817, 'BDG2',        '#67000d', practical_g(2817), 3),
]

# ============================================================
# Figure
# ============================================================
fig = plt.figure(figsize=(7.6, 6.1), dpi=300, facecolor='white')
outer = gridspec.GridSpec(2, 2, hspace=0.34, wspace=0.24, figure=fig,
                          left=0.075, right=0.985, top=0.93, bottom=0.11)
positions = [(0, 0), (0, 1), (1, 0)]
panel_tag = ['(a)', '(b)', '(c)']

for idx, (title, d) in enumerate(datasets.items()):
    r, c = positions[idx]
    flops = np.array(d['flops']); mse_raw = np.array(d['mse'])
    g_vals = d['g_vals']; star_g = d['star_g']
    it_f, it_m = d['it_flops'], d['it_mse']
    forced_extent = d.get('y_extent')
    x_pct = flops / it_f * 100
    y_pct = (mse_raw - it_m) / it_m * 100

    inner = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[r, c], width_ratios=[8, 0.9], wspace=0.06)
    ax_l = fig.add_subplot(inner[0]); ax_r = fig.add_subplot(inner[1])

    if forced_extent is not None:
        ylim = (-forced_extent - 0.02, 0.02)
    else:
        y_lo = min(y_pct.min(), 0); y_hi = max(y_pct.max(), 0)
        span = y_hi - y_lo; pad = span * 0.18
        ylim_lo = y_lo - pad; ylim_hi = y_hi + pad
        if y_pct.max() <= 0.1: ylim_hi = min(ylim_hi, 0.1)
        if y_pct.min() >= -0.1: ylim_lo = max(ylim_lo, -0.1)
        ylim = (ylim_lo, ylim_hi)

    x_overrides = {'Solar-AL (N=137)': (89.0, None), 'BDG2 (N=2817)': (22.5, 29.0)}
    x_range = x_pct.max() - x_pct.min(); x_pad = x_range * 0.25 + 0.5
    xlim_l = (x_pct.min() - x_pad, x_pct.max() + x_pad)
    if title in x_overrides:
        lo_ov, hi_ov = x_overrides[title]
        if lo_ov is not None: xlim_l = (lo_ov, xlim_l[1])
        if hi_ov is not None: xlim_l = (xlim_l[0], hi_ov)
    ax_l.set_xlim(xlim_l); ax_l.set_ylim(ylim)
    ax_l.axhline(0, color='#BBB', ls='-', lw=0.7, zorder=1)

    ofs = lbl[title]
    for gv, xp, yp, f_abs in zip(g_vals, x_pct, y_pct, flops):
        cc = g_palette[gv]
        ax_l.scatter(xp, yp, s=70, color=cc, edgecolors='#222', linewidths=0.8, zorder=6, alpha=0.92)
        if gv == star_g:
            ax_l.scatter(xp, yp, s=150, marker='*', color='#FFD700',
                         edgecolors='#AA7700', linewidths=0.8, zorder=7)
        xo, yo, ha = ofs[gv]
        ax_l.annotate(f'G={gv}', (xp, yp), fontsize=5.6, fontweight='bold', color='#333',
                      ha=ha, va='center', xytext=(xo * SC, yo * SC),
                      textcoords='offset points',
                      arrowprops=dict(arrowstyle='-', color='#CCC', lw=0.4, shrinkA=0, shrinkB=2))

    cx = xlim_l[0] + (xlim_l[1]-xlim_l[0]) * 0.03
    cy = ylim[0] + (ylim[1]-ylim[0]) * 0.04
    sx = xlim_l[0] + (xlim_l[1]-xlim_l[0]) * 0.20
    sy = ylim[0] + (ylim[1]-ylim[0]) * 0.20
    ax_l.annotate('', xy=(cx, cy), xytext=(sx, sy),
                  arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.6))
    ax_l.text(sx, sy, 'Better', fontsize=6.5, fontweight='bold', color='#2E7D32', ha='left', va='bottom')

    ax_l.set_xlabel('FLOPs (% of iT)', fontsize=7.5, labelpad=2)
    ax_l.set_ylabel('ΔMSE vs iT (%)', fontsize=7.5, labelpad=2)
    ax_l.set_title(f'{panel_tag[idx]} {title}', fontsize=8.5, fontweight='bold', pad=3)
    ax_l.grid(True, ls=':', lw=0.3, alpha=0.3)
    ax_l.tick_params(labelsize=6.5)
    ax_l.set_facecolor('white')
    if d.get('y_major_step'): ax_l.yaxis.set_major_locator(MultipleLocator(d['y_major_step']))
    if d.get('y_minor_step'): ax_l.yaxis.set_minor_locator(MultipleLocator(d['y_minor_step']))

    ax_r.set_xlim(96, 104); ax_r.set_ylim(ylim)
    ax_r.scatter(100, 0, s=90, color='#CC3311', marker='D', edgecolors='#881100', linewidths=1.0, zorder=8)
    space_below = 0 - ylim[0]; space_above = ylim[1] - 0
    ly, lva = (-12, 'top') if space_below > space_above else (10, 'bottom')
    ax_r.annotate('iT', (100, 0), fontsize=6.0, fontweight='bold', color='#CC3311',
                  xytext=(0, ly), textcoords='offset points', ha='center', va=lva, annotation_clip=False)
    ax_r.axhline(0, color='#BBB', ls='-', lw=0.7, zorder=1)
    ax_r.set_facecolor('white')
    ax_r.yaxis.set_ticklabels([]); ax_r.tick_params(axis='y', length=0)
    ax_r.set_xticks([100]); ax_r.set_xticklabels(['100%'], fontsize=5.5)
    ax_r.grid(False)
    d_mk = 0.025; kw = dict(color='#888', clip_on=False, lw=1.0)
    for yy in [0, 1]:
        ax_l.plot([1-d_mk, 1+d_mk], [yy-d_mk, yy+d_mk], transform=ax_l.transAxes, **kw)
        ax_r.plot([-d_mk, d_mk], [yy-d_mk, yy+d_mk], transform=ax_r.transAxes, **kw)
    ax_l.spines['right'].set_visible(False); ax_r.spines['left'].set_visible(False)
    for sp in list(ax_l.spines.values()) + list(ax_r.spines.values()):
        sp.set_linewidth(0.6); sp.set_color('#AAA')

# ---------- (d) landscape ----------
ax_d = fig.add_subplot(outer[1, 1])
for N, label, color, G_d, n_layers in N_specs:
    g_curve = G_range[G_range <= min(256, N)]
    F_vg = to_gflops(attn_flops_vg(N, g_curve, n_layers))
    ax_d.plot(F_vg, g_curve, color=color, lw=1.6, alpha=0.95, zorder=3)
    F_it = to_gflops(attn_flops_it(N, n_layers))
    ax_d.axvline(F_it, color=color, lw=1.0, alpha=0.55, linestyle=(0, (4, 3)), zorder=1)
    G_s = np.clip(g_star(N), 2, 256)
    ax_d.scatter(to_gflops(attn_flops_vg(N, G_s, n_layers)), G_s, s=70, marker='*',
                 facecolors=color, edgecolors='black', linewidths=0.7, zorder=7)
    ax_d.scatter(to_gflops(attn_flops_vg(N, G_d, n_layers)), G_d, s=34, marker='o',
                 facecolors=color, edgecolors='black', linewidths=0.7, zorder=6)
ax_d.set_xscale('log'); ax_d.set_yscale('log')
ax_d.set_yticks(G_ticks)
ax_d.yaxis.set_major_formatter(ticker.ScalarFormatter()); ax_d.yaxis.set_minor_formatter(ticker.NullFormatter())
ax_d.tick_params(axis='y', which='minor', left=False)
ax_d.set_ylim(1.7, 360)
ax_d.set_xlabel('Attention GFLOPs (main depth; log)', fontsize=7.5, labelpad=2)
ax_d.set_ylabel('Group count $G$ (log)', fontsize=7.5, labelpad=2)
ax_d.set_title('(d) Group count vs attention FLOPs', fontsize=8.5, fontweight='bold', pad=3)
ax_d.tick_params(labelsize=6.5)
for sp in ax_d.spines.values(): sp.set_linewidth(0.7)
# dataset legend (compact, inside panel d)
ds_handles = [Line2D([0], [0], color=c, lw=1.8, label=f'{lbl_} (depth={layers})')
              for _, lbl_, c, _, layers in N_specs]
leg_d = ax_d.legend(handles=ds_handles, loc='upper right', fontsize=4.8, framealpha=0.95,
                    edgecolor='#AAA', borderpad=0.4, handletextpad=0.4, labelspacing=0.22,
                    ncol=1, title='Dataset', title_fontsize=5.4)
leg_d.get_title().set_fontweight('bold')
ax_d.add_artist(leg_d)
mk_handles = [
    Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markeredgecolor='black',
           markersize=8, linewidth=0, label=r'$G^{*}$ (optimal)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markeredgecolor='black',
           markersize=5, linewidth=0, label=r'$G$ (used)'),
    Line2D([0], [0], color='black', lw=1.0, linestyle=(0, (4, 3)), label='iT baseline'),
]
ax_d.legend(handles=mk_handles, loc='lower right', fontsize=5.4, framealpha=0.95,
            edgecolor='#AAA', borderpad=0.4, handletextpad=0.4, labelspacing=0.3, ncol=1)

# shared G-value legend for (a-c), at the very bottom
g_handles = [Line2D([0], [0], marker='D', color='w', markerfacecolor='#CC3311',
                    markeredgecolor='#881100', markersize=6, label='iTransformer'),
             Line2D([0], [0], marker='*', color='w', markerfacecolor='#FFD700',
                    markeredgecolor='#AA7700', markersize=9, label='Rule-derived $G$')]
for gv in sorted(g_palette.keys()):
    g_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=g_palette[gv],
                            markeredgecolor='#222', markersize=5, label=f'G={gv}'))
fig.legend(handles=g_handles, loc='lower center', ncol=9, fontsize=6.2, framealpha=0.95,
           edgecolor='#AAA', bbox_to_anchor=(0.5, 0.005), columnspacing=0.7, handletextpad=0.3)

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(out_dir, exist_ok=True)
for ext in ['png', 'pdf']:
    fig.savefig(os.path.join(out_dir, f'g_selection_2x2.{ext}'), dpi=600,
                bbox_inches='tight', facecolor='white')
plt.close()
print('Done: g_selection_2x2.{png,pdf}')
