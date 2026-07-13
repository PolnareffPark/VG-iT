# VG-iT: Hierarchical Variable-Group Attention for Multivariate Time Series Forecasting

Official implementation of **VG-iT (Variable Grouping iTransformer)**, a hierarchical
attention architecture for multivariate time series forecasting in high-variate settings.

> Himchan Park, Jongkwan Choi, Chang Ouk Kim.
> *VG-iT: Hierarchical Variable-Group Attention for Multivariate Time Series Forecasting.*
> Submitted to **Neurocomputing**.

---

## Overview

The iTransformer represents each variate as a token and provides a strong variate-axis
attention baseline, but the quadratic growth of full self-attention becomes costly as the
number of variates increases. **VG-iT** partitions variates into groups, applies local
attention inside each group, and exchanges information across pooled group representatives.
Two lightweight mechanisms bridge the hierarchy:

- **Shifted grouping** — Swin-inspired per-layer group-boundary shifts that exchange
  information across group boundaries.
- **Zero-initialized FiLM** — feature-wise linear modulation that re-injects global context
  into local tokens while preserving the residual identity at initialization.

The hierarchy preserves an explicit variate-interaction pathway while reducing measured
floating-point operations, memory use, and latency relative to full variate-axis attention.
A grouping-strategy study further identifies a **data-free ordered partition** as a
reproducible default.

```
Variates ──► group partition ──► [ local attention | per group ] ──► pooled group reps
                                          ▲                                  │
                                          │  FiLM / gated broadcast          ▼
                                          └────────── global (inter-group) attention
```

---

## Repository structure

```
.
├── run.py                     # entry point (training / evaluation)
├── model/                     # VG-iT and iTransformer-family models
│   ├── VG_iTransformer.py     #   VG-iT (main model)
│   ├── VG_iFlashformer.py     #   VG-iFlashformer (SDPA/FlashAttention local kernel)
│   ├── iTransformer.py        #   iTransformer baseline
│   ├── iFlashformer.py        #   iFlashformer baseline
│   └── iNystromformer.py      #   iNyströmformer baseline
├── layers/                    # attention, embedding, grouping layers
├── data_provider/             # dataset loaders and factory
├── experiments/               # training / evaluation loop
├── utils/                     # grouping strategies, metrics, tools
├── scripts/                   # reproduction scripts (see "Reproducing the paper")
├── requirements.txt
└── LICENSE                    # MIT
```

---

## Installation

The reported runs used **Python 3.11.13**, **PyTorch 2.7.1+cu128**, CUDA 12.8,
cuDNN 90701, and an NVIDIA RTX 3090. See `ENVIRONMENT_OBSERVED.md` for the
recorded package versions and measurement context.

```bash
# 1) install PyTorch for your CUDA version first (see https://pytorch.org)
#    e.g. pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2) install the remaining dependencies
pip install -r requirements.txt
#    or, to skip packages already present:
bash scripts/install_deps.sh
```

`requirements.txt` excludes `torch`/`torchvision`/`torchaudio` so you can match your own
CUDA build.

---

## Command context

All reproduction commands below assume that the current working directory is `code/`:

```bash
cd code
```

The shell scripts target Linux/WSL Bash with standard GNU command-line utilities. They
reference paths such as `./dataset/...`, `./test_results/...`, and
`scripts/utils_progress.sh` relative to `code/`.

---

## Data preparation

Unless stated otherwise, the commands and paths in this file are relative to the
`code/` directory in the repository deposit.

Datasets are **not** bundled with this repository. Download the public datasets from the
sources below and place each under `./dataset/<name>/`, matching the `--root_path` /
`--data_path` arguments used by the scripts (e.g. `./dataset/electricity/electricity.csv`).
Solar is the exception: the supplied reproduction scripts use the special `Dataset_Solar`
loader with `--data Solar`, `--root_path ./dataset/Solar/`, and `--data_path solar_AL.txt`.

### Academic benchmarks

| Dataset | Variates | Source |
|---|---:|---|
| Electricity (ECL) | 321 | [UCI ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) · [laiguokun mirror](https://github.com/laiguokun/multivariate-time-series-data) |
| Traffic | 862 | Caltrans PeMS · [laiguokun mirror](https://github.com/laiguokun/multivariate-time-series-data/tree/master/traffic) |
| Solar-Energy (solar_AL) | 137 | [NREL](https://www.nrel.gov/grid/solar-power-data.html) · [laiguokun mirror](https://github.com/laiguokun/multivariate-time-series-data) |

The community-standard preprocessed versions (used by Informer / Autoformer / iTransformer)
are recommended; they follow the data-distribution protocol of Lai et al. (2018).

### Industrial / high-variate datasets

| Dataset | Variates | Source |
|---|---:|---|
| KAMP (battery pack) | 228 | [KAMP](https://www.kamp-ai.kr/main), *Electronic Components (Battery Pack) Quality Assurance AI Dataset*, historical record `DATASET_SEQ=57`; raw files are not redistributed |
| CARE Wind (wind turbines) | 238 | [Zenodo v6, 10.5281/zenodo.15846963](https://doi.org/10.5281/zenodo.15846963), CC BY-SA 4.0; the retained archive hash matches v6 |
| BASF (chemical plant) | 244 | [NoBOOM Kaggle dataset](https://www.kaggle.com/datasets/faebs94/noboom-anomaly-detection-in-chemical-processes), CC BY 4.0 |
| SDWPF (wind power) | 2,144 | [Figshare item 24798654, version 2](https://figshare.com/articles/dataset/SDWPF_dataset/24798654), CC BY 4.0 |
| ASHRAE (GEPIII) | 2,362 | [Kaggle ASHRAE Energy Prediction](https://www.kaggle.com/competitions/ashrae-energy-prediction); competition/data terms apply |
| BDG2 (Building Data Genome 2) | 2,817 | [buds-lab/building-data-genome-project-2](https://github.com/buds-lab/building-data-genome-project-2) (upstream checkout includes an Attribution-ShareAlike 4.0 license) |

Industrial datasets require conversion from their raw form into the expected CSV layout
(`date` column + numeric variate columns). Preprocessing scripts are provided under
`scripts/preprocess_industrial/`. Some sources (e.g. KAMP) require registration on the
provider's platform; please obtain the data under each provider's license.
Exact access/version notes and SHA-256 values for the nine processed experiment inputs are
recorded in the repository-level `data_provenance/dataset_provenance.md`.

**Data format.** Each loader expects a CSV with a leading `date` column followed by numeric
variate columns, using a 70/10/20 train/validation/test split (the standard long-term
forecasting protocol). See `data_provider/data_loader.py`.

---

## Quick start

```bash
python run.py \
  --is_training 1 --model_id demo --model VG_iTransformer \
  --data custom --root_path ./dataset/electricity/ --data_path electricity.csv \
  --features M --seq_len 96 --pred_len 96 \
  --enc_in 321 --dec_in 321 --c_out 321 \
  --d_model 512 --d_ff 512 --e_layers 3 --n_heads 8 \
  --num_groups 16 --pooling mean \
  --use_shifted_grouping 1 --use_film_broadcast 1 \
  --train_epochs 10 --des demo
```

Key VG-iT arguments: `--num_groups` (number of variate groups), `--use_shifted_grouping`,
`--use_film_broadcast`, `--use_sdpa` (FlashAttention local kernel, used by VG-iFlashformer),
`--grouping_method` (default `ordered`).

---

## Reproducing the paper

| Script | Reproduces |
|---|---|
| `scripts/full_benchmark_vgit.sh` | The 60 ordered VG-iT academic rows used in the main manuscript: three datasets, four horizons, and seeds 2021--2025 |
| `scripts/run_industrial_benchmark.sh` | Five-seed industrial benchmark: iTransformer / iFlashformer / VG-iT on the six high-variate datasets |
| `scripts/run_vg_generality.sh` | Five-seed VG-iFlashformer study (variable grouping is backbone-agnostic) |
| `scripts/run_grouping_invariance.sh` | Full 2,880-run grouping study (eight methods, 30 seeds); accepts optional `METHODS` and `SEEDS` filters |
| `scripts/precompute_grouping_plans.py` | Pre-compute and cache grouping plans (optional acceleration) |

Minimal command set from the repository root:

```bash
cd code
bash scripts/full_benchmark_vgit.sh
bash scripts/run_industrial_benchmark.sh
bash scripts/run_vg_generality.sh
SHARD=0 TOTAL_SHARDS=1 bash scripts/run_grouping_invariance.sh
```

Run scripts from the `code/` directory (they reference `./dataset/...` and
`scripts/utils_progress.sh`). Results are written to `./test_results/` and
`result_long_term_forecast.txt`.

The manuscript's academic ordered VG-iT values were taken from the `ordered`,
seed-2021--2025 subset of the full 30-seed grouping-invariance output. They were not
generated by the removed legacy script that used three seeds and `G=32` for every
dataset. `full_benchmark_vgit.sh` now applies those filters to the same provenance-
generating entry point and preserves the dataset-specific `G`, depth, shifted-grouping,
FiLM, precision, and optimization settings.

`run.py` retains upstream default values such as `./data/electricity/` for
compatibility, but the manuscript reproduction commands pass explicit
`--root_path` and `--data_path` arguments under `./dataset/...`.

---

## Baselines

This repository contains the VG-iT models and the iTransformer-family baselines
(iTransformer, iFlashformer, iNyströmformer). The additional comparison models reported in
the paper were evaluated using their **official public implementations**, including:

- iTransformer — https://github.com/thuml/iTransformer
- Time-Series-Library (PatchTST, Crossformer, DLinear, FEDformer, Non-stationary Transformer, …) — https://github.com/thuml/Time-Series-Library
- S-Mamba — https://github.com/wzhwzhwzh0921/S-D-Mamba
- TimeMixer — https://github.com/kwuking/TimeMixer
- SOFTS — https://github.com/Secilia-Cxy/SOFTS

Please refer to each project's repository and license for the corresponding baseline.

---

## Citation

```bibtex
@article{park2026vgit,
  title   = {VG-iT: Hierarchical Variable-Group Attention for Multivariate Time Series Forecasting},
  author  = {Park, Himchan and Choi, Jongkwan and Kim, Chang Ouk},
  journal = {Neurocomputing},
  year    = {2026},
  note    = {Submitted}
}
```

This implementation builds on the inverted-Transformer formulation of
iTransformer (Liu et al., ICLR 2024).

---

## License

Original VG-iT software authored for this project is released under the MIT
License. Identified third-party or adapted components remain subject to their
upstream MIT or Apache-2.0 licenses. See
[LICENSE_SCOPE.md](../LICENSE_SCOPE.md) and
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for the file-level scope
and attribution details.
