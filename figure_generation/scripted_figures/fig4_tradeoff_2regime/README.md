# Figure 4: Accuracy-Efficiency Tradeoff

Manuscript figure:
- `rendered_Figure_4_accuracy_efficiency_tradeoff.png`

Run from the repository deposit root:

```bash
python figure_generation/scripted_figures/fig4_tradeoff_2regime/plot_tradeoff_2regime.py
```

Script inputs:
- `results_canonical/shared_consolidated_5seed/academic_baselines.csv`
- `results_canonical/shared_consolidated_5seed/academic_grouping.csv`
- `results_canonical/shared_consolidated_5seed/industrial_baselines.csv`
- `results_canonical/shared_consolidated_5seed/industrial_grouping.csv`
- `results_canonical/shared_consolidated_5seed/vgflash_generality.csv`
- `results_canonical/shared_consolidated_5seed/efficiency_profile.csv`

Generated outputs:
- `figure_generation/scripted_figures/fig4_tradeoff_2regime/figures/tradeoff_2regime.png`
- `figure_generation/scripted_figures/fig4_tradeoff_2regime/figures/tradeoff_2regime.pdf`

Notes:
- The script now resolves inputs relative to the repository deposit root.
- The rendered manuscript copy is included as `rendered_Figure_4_accuracy_efficiency_tradeoff.png`.
