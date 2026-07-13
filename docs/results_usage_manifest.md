# Results Usage Manifest

Last updated: 2026-07-11

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
| `efficiency_profile.csv` | 376 rows | FLOPs, VRAM, latency, and parameter context for `tbl:academic-full`, `tbl:4`, Figure 4, Supplementary Table S2, and Supplementary Table S6. |
| `_source_mapping.json` | provenance JSON | Unique public `provenance_id` values grouped by consolidated CSV file. |

Excluded from cleaned deposit: backup files, `efficiency_v6_long.csv`, `efficiency_v6_wide.csv`, `summary.csv`, and speed-summary intermediate files not used by the manuscript package.

## Grouping Sensitivity and Diagnostics

Directory: `results_canonical/rq2a_grouping/`

| File | Rows/role | Used for |
|---|---:|---|
| `g_sensitivity_results.csv` | 180 rows | Supporting data for group-count sensitivity and Figure 3. |
| `g_sensitivity_flops.csv` | 15 rows | Corrected shifted-grouping + FiLM FLOPs for every `G` plotted in Figure 3. |
| `grouping_invariance_results.csv` | 2880 rows | Main Table 6, Supplementary Tables S3-S4, Figure 5 clean-run filtering, Supplementary Figure S1 diagnostics. |
| `category_rope_analysis.md` | analysis report | Supplementary ROPE matrix and practical-equivalence statements. |
| `perds_sign_homogeneity_report.md` | analysis report | Supplementary Figure S1 win-rate values. |
| `epoch1_loss_analysis.md` | analysis report | Human-readable epoch-1 diagnostic summary. Exploratory gradient/component side-figure outputs are not included. |
| `epoch1_losses.csv` | 2880 rows | Direct source for Supplementary Figure S2 and the epoch-1 diagnostic statistics. |
| `traffic_epoch1_losses.csv` | 960 rows | Traffic-specific convergence-anomaly diagnostic source. |

Excluded from cleaned deposit: grouping timing files, gradient side-analysis files, granger/acf files, and rank-collapse side analyses not cited by the manuscript package.

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
| `pre_pooling_probing_results.json` | pre/post pooling confusion matrices | Main Figure 6. |
| `pre_pooling_probing_results.md` | human-readable probing summary | Main Figure 6 context. |

Excluded from cleaned deposit: `erank_30seed_analysis.json`, `erank_deep_analysis.md`, and uncited probing summaries not used by the manuscript package.
