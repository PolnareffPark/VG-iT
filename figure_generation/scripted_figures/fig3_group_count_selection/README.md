# Figure 3: Group-count sensitivity

Manuscript figure:
- `rendered_Figure_3_group_count_selection.pdf`

Run from the repository deposit root:

```bash
python figure_generation/scripted_figures/fig3_group_count_selection/plot_g_selection_combined_2x2.py
```

Script inputs:
- `results_canonical/rq2a_grouping/g_sensitivity_results.csv`: 180 shifted-grouping +
  FiLM runs (five values of `G`, four horizons, and seeds 2021--2023 for each of
  Solar-AL, Traffic, and BDG2).
- `results_canonical/rq2a_grouping/g_sensitivity_flops.csv`: corrected all-`G` FLOPs,
  including the FiLM linear projections.
- `results_canonical/shared_consolidated_5seed/{academic,industrial}_baselines.csv`:
  iTransformer MSE baselines, filtered to seeds 2021--2023.
- `results_canonical/shared_consolidated_5seed/efficiency_profile.csv`: iTransformer
  `T=96` FLOPs baselines.

Generated outputs:
- `figure_generation/scripted_figures/fig3_group_count_selection/figures/g_selection_2x2.png`
- `figure_generation/scripted_figures/fig3_group_count_selection/figures/g_selection_2x2.pdf`

Notes:
- The three Traffic rows with final test MSE greater than 0.6 are excluded by an
  explicit assertion-backed rule. No other row is filtered.
- Stars in panels (a)--(c) are the group counts from Equation (13), not values selected
  by the sweep. Panel (d) uses the dataset-specific encoder depths and group counts from
  the main manuscript; Electricity therefore uses `G=16`, not the legacy `G=32`.
- The rendered manuscript copy is included as `rendered_Figure_3_group_count_selection.pdf`.
- Rerunning the script does not overwrite the rendered manuscript copy.
