# Figure 6: Pre/Post-Pooling Probing Confusion Matrices

Manuscript figure:
- `rendered_Figure_6_probing_confusion_matrices.pdf`

Run from the repository deposit root:

```bash
python figure_generation/scripted_figures/fig6_probing_confusion/generate_figure_6.py
python figure_generation/scripted_figures/fig6_probing_confusion/summarize_pre_pooling_probing.py
```

Script input:
- `results_canonical/representation_diagnostics/pre_pooling_probing/pre_pooling_probing_results.json`

Generated outputs:
- `figure_generation/scripted_figures/fig6_probing_confusion/figures/figure_6_pre_post_aggregate.png`
- `figure_generation/scripted_figures/fig6_probing_confusion/figures/figure_6_pre_post_by_dataset.png`
- `figure_generation/scripted_figures/fig6_probing_confusion/figures/figure_6_pre_post_by_dataset.pdf`

Notes:
- The manuscript Figure 6 corresponds to the per-dataset output.
- `summarize_pre_pooling_probing.py` validates every confusion matrix and recomputes the 12 condition-level accuracies plus pooled counts. Its `accuracy > chance + 0.03` flag is a descriptive threshold, not a significance test.
- The script now resolves inputs relative to the repository deposit root.
- The rendered manuscript copy is included as `rendered_Figure_6_probing_confusion_matrices.pdf`.
