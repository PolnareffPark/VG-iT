# Figure Code Manifest

Last updated: 2026-07-11

Figure generation materials are available under `figure_generation/`.
Scripted figures can be run from the repository deposit root after installing
`figure_generation/requirements.txt`.

| Figure | Source type | Inputs / source material | Run command or source note | Repository rendered output |
|---|---|---|---|---|
| Figure 1 | Manual Figma artwork | Rendered PNG and editable SVG source `figure_generation/manual_figures_figma/source/Figure_1.svg` are included. | See `figure_generation/manual_figures_figma/README.md`. | `figure_generation/manual_figures_figma/rendered_artwork/Figure_1_architecture_contrast.png` |
| Figure 2 | Manual Figma artwork | Rendered PNG and editable SVG source `figure_generation/manual_figures_figma/source/Figure_2.svg` are included. | See `figure_generation/manual_figures_figma/README.md`. | `figure_generation/manual_figures_figma/rendered_artwork/Figure_2_vgit_pipeline.png` |
| Figure 3 | Scripted figure | `g_sensitivity_results.csv`, `g_sensitivity_flops.csv`, the iTransformer baseline CSVs, and `efficiency_profile.csv`; values are loaded and asserted by the script. | `python figure_generation/scripted_figures/fig3_group_count_selection/plot_g_selection_combined_2x2.py` | `figure_generation/scripted_figures/fig3_group_count_selection/rendered_Figure_3_group_count_selection.pdf` |
| Figure 4 | Scripted figure | Consolidated accuracy/resource CSVs in `results_canonical/shared_consolidated_5seed/`. | `python figure_generation/scripted_figures/fig4_tradeoff_2regime/plot_tradeoff_2regime.py` | `figure_generation/scripted_figures/fig4_tradeoff_2regime/rendered_Figure_4_accuracy_efficiency_tradeoff.png` |
| Figure 5 | Scripted figure | `results_canonical/representation_diagnostics/erank/erank_allpl_analysis.json`; `results_canonical/rq2a_grouping/grouping_invariance_results.csv`. | `python figure_generation/scripted_figures/fig5_effective_rank/erank_5category_strip.py` | `figure_generation/scripted_figures/fig5_effective_rank/rendered_Figure_5_effective_rank_distribution.png` |
| Figure 6 | Scripted figure | `results_canonical/representation_diagnostics/pre_pooling_probing/pre_pooling_probing_results.json`. | `python figure_generation/scripted_figures/fig6_probing_confusion/gen_s21_pre_post.py` | `figure_generation/scripted_figures/fig6_probing_confusion/rendered_Figure_6_probing_confusion_matrices.pdf` |
| Supplementary Figure S1 | Scripted figure | Values embedded from `results_canonical/rq2a_grouping/perds_sign_homogeneity_report.md`. | `python figure_generation/scripted_figures/supp_fig_s1_winrate/gen_s22_winrate_figures.py` | `figure_generation/scripted_figures/supp_fig_s1_winrate/rendered_Supplementary_Figure_S1_winrate_bar.png` |
| Supplementary Figure S2 | Scripted figure | All 2,880 recorded values in `results_canonical/rq2a_grouping/epoch1_losses.csv`; the script validates anomaly labels against final MSE and does not synthesize samples. | `python figure_generation/scripted_figures/supp_fig_s2_epoch1/gen_s24_figures.py` | `figure_generation/scripted_figures/supp_fig_s2_epoch1/rendered_Supplementary_Figure_S2_epoch1_distribution.png` |

Validation performed for this package:
- Python syntax compilation passed for all six scripted figure files.
- Figure 3, Figure 4, Figure 5, Figure 6, Supplementary Figure S1, and Supplementary Figure S2 scripts were executed from the repository deposit root.
- Figure 1 and Figure 2 editable SVG exports were parsed as SVG XML and checked for local-path or external-image references.
- Generated test outputs were removed after validation to keep the deposit package centered on scripts, inputs, README files, and rendered artwork copies.
