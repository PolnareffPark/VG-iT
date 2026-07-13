# Dataset Provenance and Access Notes

This document records the dataset access and preprocessing assumptions for the VG-iT
Neurocomputing reproducibility package.

Raw third-party datasets are not redistributed in this repository. The repository provides
source code, preprocessing scripts, curated result tables, and figure-generation materials.
Users should obtain the raw datasets from the original providers or community-standard
mirrors and follow the license or access terms of each provider.

## Dataset Scope

The manuscript experiments use the following datasets:

| Group | Dataset | Local expected CSV | Variates | Time steps | Frequency |
|---|---|---|---:|---:|---|
| Academic | Solar-Energy / Solar_AL | `code/dataset/Solar/solar_AL.txt` | 137 | 52,560 | 10 min |
| Academic | Electricity / ECL | `code/dataset/electricity/electricity.csv` | 321 | 26,304 | 1 hour |
| Academic | Traffic | `code/dataset/traffic/traffic.csv` | 862 | 17,544 | 1 hour |
| Industrial | KAMP battery-pack data | `code/dataset/kamp/kamp.csv` | 228 | 32,415 | minutely synthetic index |
| Industrial | CARE Wind Farm C | `code/dataset/care_wind_energy/care_wind.csv` | 238 | 47,264 | 10 min synthetic index |
| Industrial | BASF NoBOOM process data | `code/dataset/basf/basf.csv` | 244 | 215,841 | hourly synthetic index |
| Industrial | SDWPF | `code/dataset/sdwpf_energy/sdwpf.csv` | 2,144 | 84,785 | 10 min |
| Industrial | ASHRAE Great Energy Predictor III | `code/dataset/ashrae_energy/ashrae.csv` | 2,362 | 8,784 | 1 hour |
| Industrial | Building Data Genome Project 2 | `code/dataset/bdg2/bdg2.csv` | 2,817 | 17,544 | 1 hour |

Datasets that were explored but not used in the manuscript are intentionally excluded from
this public deposit.

## Access, Version, and License Status

The table records what can be established from the retained local sources and provider
pages as of 2026-07-11. A provider page being publicly reachable is not itself a license.
Historical revision gaps are recorded as provenance notes, not treated as missing access
sources or as inferred redistribution permissions.

| Dataset | Access/version record | License or terms status |
|---|---|---|
| Solar-AL | LSTNet mirror: `https://github.com/laiguokun/multivariate-time-series-data`; original provider context: NREL. The historical mirror commit was not recorded. | Provider and mirror terms apply; this package makes no blanket license assertion. |
| Electricity | UCI dataset 321, DOI `10.24432/C58C86`; the study uses the community hourly matrix derived from the benchmark distribution. | UCI provider page states CC BY 4.0. The processed mirror revision was not recorded. |
| Traffic | Caltrans PeMS data distributed through the LSTNet benchmark mirror; historical mirror commit not recorded. | Provider and mirror terms apply; raw data are not redistributed and no blanket license is asserted. |
| KAMP | KAMP, operated by KAIST and presented under the Ministry of SMEs and Startups: `https://www.kamp-ai.kr/main`; *Electronic Components (Battery Pack) Quality Assurance AI Dataset*, KOSMO (InterX Co., Ltd./Nestfield Co., Ltd.), 2022-12-23. Historical dataset record `DATASET_SEQ=57` was recorded on 2026-05-28. | Raw provider files are not redistributed. No dataset-specific redistribution license is asserted. |
| CARE Wind Farm C | Zenodo v6 record `10.5281/zenodo.15846963`. The retained `CARE_To_Compare.zip` is 5,503,439,673 bytes with MD5 `2547b58c21ac8c242d13232860cf500c`, matching the official v6 file. | CC BY-SA 4.0. |
| BASF NoBOOM | Kaggle dataset `faebs94/noboom-anomaly-detection-in-chemical-processes`; eight normal-operation BASF process segments were used. | Kaggle dataset page states CC BY 4.0. |
| SDWPF | Figshare dataset `24798654`, version 2 (2024-04-30); the `sdwpf_full` parquet representation was used. | CC BY 4.0. |
| ASHRAE GEPIII | Kaggle competition `ashrae-energy-prediction`. | Competition/data terms apply; no generic open-data license is asserted here. |
| BDG2 | Local upstream checkout commit `9b97ccbe90096aff42ed4fd6493bf7ae692d7118`; repository `buds-lab/building-data-genome-project-2`. | The retained upstream `LICENSE` is Attribution-ShareAlike 4.0; Kaggle identifies CC BY-SA 4.0. |

## Academic Benchmarks

### Solar-Energy / Solar_AL

- Original provider: National Renewable Energy Laboratory (NREL) solar power data.
- Community benchmark source: Lai et al. (2018), LSTNet multivariate time-series data.
- Common mirror: `https://github.com/laiguokun/multivariate-time-series-data`
- Expected processed file: `solar_AL.txt` under `code/dataset/Solar/`. This file is a
  headerless comma-separated numeric matrix loaded by `Dataset_Solar`; it does not use a
  leading `date` column.
- Manuscript use: academic benchmark comparisons and grouping diagnostics.

### Electricity / ECL

- Original provider: UCI Machine Learning Repository, Electricity Load Diagrams 2011-2014.
- Community benchmark source: LSTNet / Informer / Autoformer-style processed versions.
- Provider page: `https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014`
- DOI: `10.24432/C58C86`.
- Provider-page license: CC BY 4.0.
- Common mirror: `https://github.com/laiguokun/multivariate-time-series-data`
- Expected processed file: `electricity.csv` with a leading `date` column followed by numeric
  variate columns.
- Manuscript use: academic benchmark comparisons and grouping diagnostics.

### Traffic

- Original provider: Caltrans Performance Measurement System (PeMS), San Francisco Bay Area
  traffic occupancy data.
- Community benchmark source: Lai et al. (2018), LSTNet multivariate time-series data.
- Common mirror: `https://github.com/laiguokun/multivariate-time-series-data/tree/master/traffic`
- Expected processed file: `traffic.csv` with a leading `date` column followed by numeric
  variate columns.
- Manuscript use: academic benchmark comparisons and grouping diagnostics.

## Industrial and High-Variate Datasets

The industrial preprocessing scripts are located in
`code/scripts/preprocess_industrial/`. They convert provider-format raw files into the
`Dataset_Custom` CSV layout used by the training code:

```text
date,var1,var2,...,varN,OT
```

### KAMP Battery-Pack Data

- Provider: KAMP (Korea AI Manufacturing Platform), operated by KAIST and presented under
  the Ministry of SMEs and Startups, `https://www.kamp-ai.kr/main`.
- Dataset: *Electronic Components (Battery Pack) Quality Assurance AI Dataset*, KOSMO
  (InterX Co., Ltd./Nestfield Co., Ltd.), 2022-12-23; historical record identifier
  `DATASET_SEQ=57` (recorded on 2026-05-28).
- Retained provider package: 51 battery charge and 51 battery discharge CSV files. The
  preprocessing script uses the 51 charge files matching `*_chg.csv`.
- Script: `code/scripts/preprocess_industrial/preprocess_kamp.py`.
- Main preprocessing operations: remove all-NaN padding rows, group-wise forward/backward
  fill within `SerialNumber`, subsample the 10-second records, remove `SerialNumber`, and
  rename `Voltage` to `OT`.

### CARE Wind Farm C

- Provider/source: CARE to Compare, Zenodo v6 `10.5281/zenodo.15846963`, CC BY-SA 4.0.
- Retained archive verification: `CARE_To_Compare.zip`, 5,503,439,673 bytes, MD5
  `2547b58c21ac8c242d13232860cf500c`, matching the official v6 file.
- Raw layout used by the preprocessing script: semicolon-delimited Wind Farm C CSV.
- Script: `code/scripts/preprocess_industrial/preprocess_care_wind.py`.
- Main preprocessing operations: keep `status_type_id == 0`, retain average sensor columns,
  remove max/min/std columns, assign a sequential 10-minute timestamp index, and rename
  `power_6_avg` to `OT`.

### BASF NoBOOM Process Data

- Provider/source: NoBOOM: Anomaly Detection in Chemical Processes,
  `https://www.kaggle.com/datasets/faebs94/noboom-anomaly-detection-in-chemical-processes`.
- Raw layout used by the preprocessing script: eight normal-operation process segments,
  each with a `train_normal_experiment_00*.csv` file.
- Script: `code/scripts/preprocess_industrial/preprocess_basf.py`.
- Main preprocessing operations: concatenate the eight process segments, remove the
  `Anomaly` label column, assign a sequential hourly timestamp index, and rename `T1` to
  `OT`.

### SDWPF

- Provider/source: Figshare SDWPF dataset, item `24798654`, version 2; the work originated
  in the Baidu KDD Cup 2022 Spatial Dynamic Wind Power Forecasting challenge.
- Access page: `https://figshare.com/articles/dataset/SDWPF_dataset/24798654`.
- Related dataset paper: Zhou et al., "SDWPF: A Dataset for Spatial Dynamic Wind Power
  Forecasting over a Large Turbine Array," Scientific Data, 2024.
- Script: `code/scripts/preprocess_industrial/preprocess_sdwpf.py`.
- Main preprocessing operations: pivot 134 turbines x 16 features into a wide multivariate
  matrix, forward/backward fill missing sensor values, and rename `T001_Patv` to `OT`.

### ASHRAE Great Energy Predictor III

- Provider/source: Kaggle ASHRAE Great Energy Predictor III competition.
- Access page: `https://www.kaggle.com/competitions/ashrae-energy-prediction`.
- Script: `code/scripts/preprocess_industrial/preprocess_ashrae.py`.
- Main preprocessing operations: pivot the long-format meter readings into a wide matrix,
  drop columns with more than 50% missing values, forward/backward fill, and rename `b0_m0`
  to `OT`.

### Building Data Genome Project 2

- Provider/source: Building Data Genome Project 2.
- Repository: `https://github.com/buds-lab/building-data-genome-project-2`.
- Dataset paper: Miller et al., "The Building Data Genome Project 2, energy meter data from
  the ASHRAE Great Energy Predictor III competition," Scientific Data, 2020.
- DOI: `10.1038/s41597-020-00712-x`.
- Retained upstream revision: `9b97ccbe90096aff42ed4fd6493bf7ae692d7118`.
- Retained upstream license: Attribution-ShareAlike 4.0 (CC BY-SA family), not CC BY 4.0.
- Script: `code/scripts/preprocess_industrial/preprocess_bdg2.py`.
- Main preprocessing operations: merge eight meter-type CSVs by timestamp, drop columns with
  more than 50% missing values, forward/backward fill, remove all-zero columns, and select
  the first electricity meter as `OT`.

## Integrity Notes

For the academic Solar_AL, Traffic, and Electricity datasets, local processed values were
compared against the LSTNet benchmark data distributed through
`laiguokun/multivariate-time-series-data`. The value matrices matched under numerical
comparison. Formatting differences such as adding a `date` column or using numeric column
names do not change the model inputs because the loaders convert the selected columns to
numeric arrays before training.

For industrial datasets, the preprocessing scripts document the transformations from raw
provider files to the wide CSV layout used by the experiments. The public deposit includes
those scripts and the curated result files, but not the raw provider files.

### Processed-file checksums

The following SHA-256 values identify the exact processed inputs retained for the reported
experiments. These large files are not included in the public deposit; the checksums allow
authors and authorized users to verify a locally reconstructed copy.

| Dataset | Processed path | Bytes | SHA-256 |
|---|---|---:|---|
| Solar-AL | `dataset/Solar/solar_AL.txt` | 180,018,000 | `230327EF72D2ABB387939D4A35D6FD34F1066071BC7C40CE7ECF5531A0122AC2` |
| Electricity | `dataset/electricity/electricity.csv` | 95,581,762 | `7E45845D54C5219BAD0AE6BC1B5316CF8FF9CEAD5D33FA998A5A51C2E4A497AD` |
| Traffic | `dataset/traffic/traffic.csv` | 136,478,119 | `CB06463D56FA17D87F47027CD9389CEAE82A69EDDEE51CDB61480E120DAB0B16` |
| KAMP | `dataset/kamp/kamp.csv` | 43,144,364 | `EF8A7F60EE9EFF85B01396264C9B4ABD104D9989F564E2AF781745B23561D0D2` |
| CARE Wind | `dataset/care_wind_energy/care_wind.csv` | 74,274,883 | `D40DEB1FB79F466B6920489FAE0A34921840AC9F5ED5C0E81F9948E61ACF29D6` |
| BASF | `dataset/basf/basf.csv` | 832,792,194 | `0CE40E61A37DE0073DB931C5A993E2FD9787BB41F62ED731C64AE06BAD149AC0` |
| SDWPF | `dataset/sdwpf_energy/sdwpf.csv` | 1,466,570,414 | `0C4E7DB0231270A234F1B965AE5429977A5148472AE00B77E955938D75AFD2E9` |
| ASHRAE | `dataset/ashrae_energy/ashrae.csv` | 142,373,617 | `043E99217C5F7AC1029770AA198D405185071D0D9CDB984706E02AD6B961F835` |
| BDG2 | `dataset/bdg2/bdg2.csv` | 354,881,379 | `A05C8D1CF0587383CB887F742E32EC2390070A08419943FE3A84EED63A349981` |

## Licensing and Access

- External raw datasets remain under their original provider licenses and access terms.
- This repository does not grant redistribution rights for third-party raw datasets.
- The authors' original VG-iT code contributions are distributed under the MIT License.
  Some included components retain third-party license headers or attribution comments; see
  `THIRD_PARTY_NOTICES.md` and `LICENSES/Apache-2.0.txt` and `LICENSES/MIT-THIRD-PARTY.txt`.
- Curated result tables are derived experiment outputs included for auditability of the
  submitted manuscript.
