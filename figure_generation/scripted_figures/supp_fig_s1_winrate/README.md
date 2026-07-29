# Supplementary Figure S1: Ordered Win-Rate Diagnostics

Manuscript supplementary figure:
- `rendered_Supplementary_Figure_S1_winrate_bar.png`

Run from the repository deposit root:

```bash
python figure_generation/scripted_figures/supp_fig_s1_winrate/generate_supplementary_figure_s1.py
```

Script inputs:
- `results_canonical/rq2a_grouping/grouping_invariance_results.csv`. The script removes blocks whose maximum within-block MSE exceeds 0.6, validates 346 retained blocks and 2,768 rows, and computes all plotted win rates directly from the retained rows.

Generated outputs:
- `figure_generation/scripted_figures/supp_fig_s1_winrate/figures/supplementary_figure_s1_winrate_heatmap.png`
- `figure_generation/scripted_figures/supp_fig_s1_winrate/figures/supplementary_figure_s1_winrate_bar.png`

Notes:
- The supplementary manuscript uses the grouped bar output.
- The rendered manuscript copy is included as `rendered_Supplementary_Figure_S1_winrate_bar.png`.
