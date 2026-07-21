# VG-iT Reproducibility Package

This repository is the VG-iT v1.0.0 release of public code and curated
reproducibility materials for:

**VG-iT: Hierarchical Variable-Group Attention for Multivariate Time Series Forecasting**

Authors: Himchan Park, Jongkwan Choi, and Chang Ouk Kim.

This release accompanies the manuscript prepared for submission to Neurocomputing. It is intended to make the
reported experiments auditable without redistributing third-party raw datasets or large
training checkpoints.

## Contents

| Path | Purpose |
|---|---|
| `code/` | VG-iT implementation, iTransformer-family baselines, training scripts, and dataset preprocessing scripts. |
| `results_canonical/` | Curated CSV/JSON/Markdown result files used by the manuscript tables, figures, and supplementary analyses. |
| `figure_generation/` | Figure source materials: Figma exports for Figures 1-2 and reproducible scripts for Figures 3-6 and Supplementary Figures S1-S2. |
| `data_provenance/` | Dataset access, provenance, and preprocessing notes. Raw datasets are not bundled. |
| `docs/results_usage_manifest.md` | Mapping from included result files to manuscript tables, figures, and claims. |
| `docs/figure_code_manifest.md` | Mapping from manuscript figures to source files, inputs, commands, and rendered outputs. |
| `CITATION.cff` | Citation metadata for GitHub and Zenodo. |
| `LICENSE` | MIT license for the authors' original software contributions. |
| `THIRD_PARTY_NOTICES.md` | Third-party license and attribution notes for included components. |
| `LICENSES/` | Third-party license files and the fixed v1.0.0 non-software rights statement. |
| `LICENSE_SCOPE.md` | File-level software, third-party data, and non-software rights scope. |

## What Is Not Included

- Raw third-party datasets are not redistributed. The datasets used in the experiments are
  publicly accessible from their original providers or community-standard mirrors; see
  `data_provenance/dataset_provenance.md`. Solar uses the `Dataset_Solar` text-file loader
  and expects `code/dataset/Solar/solar_AL.txt`; the other listed benchmarks use CSV files.
- Full training checkpoints are not included because the complete training checkpoint set is too large for this public package.
- Intermediate scratch outputs, backup CSV files, local absolute paths, and auxiliary
  datasets not used by the manuscript are excluded from this cleaned package.

## Code

The code package lives under `code/`. See `code/README.md` for installation, dataset
layout, command-line arguments, and reproduction scripts.

Minimal setup:

```bash
cd code
python -m pip install -r requirements.txt
```

PyTorch is intentionally not pinned in `code/requirements.txt`; install the PyTorch build
that matches your CUDA environment before installing the remaining dependencies.
The measured environment is recorded separately in `code/ENVIRONMENT_OBSERVED.md`.

## Data Access

The repository expects CSV files under `code/dataset/<dataset_name>/` when running the
training scripts. Academic benchmark datasets and industrial/high-variate datasets are
listed in `data_provenance/dataset_provenance.md`.

The included result files in `results_canonical/` are derived from those datasets and are
included so that manuscript tables and figures can be audited without rerunning every
training job.

## Figure Reproduction

Install the plotting dependencies from the repository root:

```bash
python -m pip install -r figure_generation/requirements.txt
```

Then run the figure-specific commands listed in `docs/figure_code_manifest.md` or the README
inside each `figure_generation/scripted_figures/*/` directory.

Figures 1 and 2 are manually drawn Figma figures. Editable SVG exports and rendered PNG
copies are included under `figure_generation/manual_figures_figma/`.

## Results

Only curated result files used by the manuscript package are included. The manifest
`docs/results_usage_manifest.md` documents which files support each table, figure, or claim.
The consolidated CSV/JSON files use public provenance identifiers and do not include
local machine paths. See the result-specific `_source_mapping.json` files for the public
provenance identifier inventory.

## Checkpoints

The manuscript results were produced from full training runs, but the checkpoint directory
is too large for this public package. Reproduction should start from the scripts and public
datasets rather than from archived local checkpoints.

## Citation

Public repository: <https://github.com/PolnareffPark/VG-iT>

This release is version `1.0.0`, published on `2026-07-21`.

Version-specific Zenodo DOI: <https://doi.org/10.5281/zenodo.21367491>
(reserved for this release; the link resolves after the Zenodo record is published).

Tagged source: <https://github.com/PolnareffPark/VG-iT/tree/v1.0.0>

## License and Third-Party Notices

VG-iT v1.0.0 is a mixed-rights archive. The original VG-iT software and plotting scripts are released under the MIT License, and identified upstream components remain under their MIT or Apache-2.0 terms. The MIT copyright holder is intentionally listed as `Himchan Park` in `LICENSE` and `code/LICENSE`; manuscript authorship and citation metadata are listed separately in `CITATION.cff`. Raw third-party datasets are not included. Author-generated results, figures, artwork, and prose carry no additional reuse license unless an individual file expressly states otherwise. See `LICENSE_SCOPE.md`, `LICENSES/NONSOFTWARE-RIGHTS.txt`, and `THIRD_PARTY_NOTICES.md` for the fixed file-level rights of this release.
