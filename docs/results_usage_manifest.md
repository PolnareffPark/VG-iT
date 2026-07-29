# Results Usage Manifest

Last updated: 2026-07-29

Only files listed here are included in the cleaned repository deposit.

## Shared Consolidated Results

Directory: `results_canonical/shared_consolidated_5seed/`

The consolidated CSV files use `provenance_id` fields rather than local run paths. `_source_mapping.json` lists the unique public provenance identifiers by CSV file; local machine paths and local run-directory prefixes are not included in the public deposit.

| File | Rows/role | Used for |
|---|---:|---|
| `academic_baselines.csv` | 840 rows | Main academic accuracy/efficiency table (`tbl:academic-full`), Figure 4 academic panel, Supplementary Table S1. |
| `academic_grouping.csv` | 2880 rows | 30-seed grouping source. Main five-seed academic rows in `tbl:academic-full` use the seed 2021-2025 subset for VG-iT ordered rows; supplementary grouping diagnostics use the full 30-seed file where stated. |
| `industrial_baselines.csv` | 600 rows | Main industrial operating-regime table (`tbl:4`), Figure 4 industrial panel, Supplementary Table S2, and industrial comparator context. |
| `industrial_grouping.csv` | 960 rows | VG-iT ordered industrial rows for the main industrial operating-regime table (`tbl:4`) and Supplementary Table S2. |
| `vgflash_generality.csv` | 180 rows | VGFlash rows in Figure 4, `tbl:4`, and industrial supplementary tables. |
| `efficiency_profile.csv` | 376 rows | FLOPs, VRAM, latency, and parameter context for `tbl:academic-full`, `tbl:4`, Figure 4, Supplementary Table S2, and Supplementary Table S6. Two Crossformer PL=720 rows retain FLOPs/parameter values with the four unavailable runtime/VRAM fields left blank; no values are imputed. |
| `_source_mapping.json` | provenance JSON | Unique public `provenance_id` values grouped by consolidated CSV file. |

Excluded from the cleaned deposit: backup and intermediate efficiency-summary files that are not used by the manuscript package.

## Grouping Sensitivity and Diagnostics

Directory: `results_canonical/rq2a_grouping/`

| File | Rows/role | Used for |
|---|---:|---|
| `g_sensitivity_results.csv` | 180 rows | Supporting data for group-count sensitivity and Figure 3. |
| `g_sensitivity_flops.csv` | 15 rows | Corrected shifted-grouping + FiLM FLOPs for every `G` plotted in Figure 3. |
| `grouping_invariance_results.csv` | 2880 rows | Main Table 6, Supplementary Tables S3-S4, Figure 5 clean-run filtering, and Supplementary Figure S1. `code/scripts/analyze_grouping_statistics.py` regenerates the Bayesian correlated-t ROPE, Wilcoxon-Holm, and Friedman summaries from this CSV. |
| `epoch1_losses.csv` | 2880 rows | Direct source for Supplementary Figure S2 and the epoch-1 diagnostic statistics, including all 960 Traffic rows. |

Excluded from cleaned deposit: generated statistical-analysis Markdown reports, grouping timing files, gradient side-analysis files, granger/acf files, and rank-collapse side analyses. The retained CSV/JSON files are the canonical public evidence inputs.

## Component and Pooling Ablations

Directory: `results_canonical/rq2b_components/`

| File | Rows | Used for |
|---|---:|---|
| `component_ablation_results.csv` | 108 rows | Main Table 7 component ablation. |
| `pooling_sf_results.csv` | 72 rows | Supplementary Table S5 pooling comparison. |

Excluded from cleaned deposit: pooling summary intermediates, raw pooling-result intermediates, and checkpoint-analysis side files.

## Representation Diagnostics

Directories:
- `results_canonical/representation_diagnostics/erank/`
- `results_canonical/representation_diagnostics/pre_pooling_probing/`

| File | Role | Used for |
|---|---|---|
| `erank_allpl_analysis.json` | effective-rank diagnostic source | Main Figure 5. |
| `pre_pooling_probing_results.json` | pre/post pooling confusion matrices | Main Figure 6. `figure_generation/scripted_figures/fig6_probing_confusion/summarize_pre_pooling_probing.py` validates matrix counts and recomputes the descriptive `accuracy > chance + 0.03` condition flag. |

Excluded from cleaned deposit: `erank_30seed_analysis.json`, `erank_deep_analysis.md`, and uncited probing summaries not used by the manuscript package.
