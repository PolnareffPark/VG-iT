# Figure Generation Materials

This directory separates manually drawn artwork from script-generated figures.
Run commands below assume the current directory is the repository deposit root.

Dependencies for scripted figures:

```bash
python -m pip install -r figure_generation/requirements.txt
```

Manual figures:
- Figure 1: `figure_generation/manual_figures_figma/rendered_artwork/Figure_1_architecture_contrast.png`
- Figure 2: `figure_generation/manual_figures_figma/rendered_artwork/Figure_2_vgit_pipeline.png`

Scripted figures:
- Figure 3: `python figure_generation/scripted_figures/fig3_group_count_selection/plot_g_selection_combined_2x2.py`
- Figure 4: `python figure_generation/scripted_figures/fig4_tradeoff_2regime/plot_tradeoff_2regime.py`
- Figure 5: `python figure_generation/scripted_figures/fig5_effective_rank/erank_5category_strip.py`
- Figure 6: `python figure_generation/scripted_figures/fig6_probing_confusion/gen_s21_pre_post.py`
- Supplementary Figure S1: `python figure_generation/scripted_figures/supp_fig_s1_winrate/gen_s22_winrate_figures.py`
- Supplementary Figure S2: `python figure_generation/scripted_figures/supp_fig_s2_epoch1/gen_s24_figures.py`

Each subdirectory contains a README with the figure-specific input files, command,
and generated output names. The `rendered_*` files are the copies included from the
manuscript package; rerunning scripts writes fresh files under each script folder's
`figures/` subdirectory.
