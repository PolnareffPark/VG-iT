# Figure 6: Pre/Post-Pooling Probing Confusion Matrices

Manuscript figure:
- `rendered_Figure_6_probing_confusion_matrices.pdf`

Run from the repository deposit root:

```bash
python figure_generation/scripted_figures/fig6_probing_confusion/gen_s21_pre_post.py
```

Script input:
- `results_canonical/representation_diagnostics/pre_pooling_probing/pre_pooling_probing_results.json`

Generated outputs:
- `figure_generation/scripted_figures/fig6_probing_confusion/figures/s21_pre_post_agg.png`
- `figure_generation/scripted_figures/fig6_probing_confusion/figures/s21_pre_post_per_ds.png`
- `figure_generation/scripted_figures/fig6_probing_confusion/figures/s21_pre_post_per_ds.pdf`

Notes:
- The manuscript Figure 6 corresponds to the per-dataset output.
- The script now resolves inputs relative to the repository deposit root.
- The rendered manuscript copy is included as `rendered_Figure_6_probing_confusion_matrices.pdf`.
