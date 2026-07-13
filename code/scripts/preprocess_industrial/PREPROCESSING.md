# Industrial Dataset Preprocessing

This document describes the preprocessing scripts used for the six industrial/high-variate
datasets reported in the VG-iT manuscript. Scripts are located in this directory and should
be run from the `code/` directory.

The scripts convert provider-format raw files into the `Dataset_Custom` CSV format consumed
by `data_provider/data_loader.py`:

```text
date,var1,var2,...,varN,OT
2020-01-01 00:00:00,1.23,4.56,...,7.89,0.12
```

The first column is `date`, intermediate columns are numeric variates, and the final target
column is `OT`. Raw datasets are not bundled with the repository.

## Run All Manuscript-Used Industrial Preprocessing Scripts

```bash
cd code
bash scripts/preprocess_industrial/preprocess_all.sh
```

`preprocess_all.sh` runs exactly the six industrial datasets used in the manuscript:

```text
basf, kamp, bdg2, ashrae, sdwpf, care_wind
```

Exploratory datasets that were not used in the manuscript are not included in the
public deposit.

## Missing-Value Policy

The long-term forecasting benchmarks used by TSLib, PatchTST, Crossformer, and related
work usually assume pre-cleaned CSV files. The industrial sources require explicit cleaning,
so the following conservative rules were used.

### Forward Fill Followed by Backward Fill

Applied to BDG2, ASHRAE, SDWPF, and KAMP after dataset-specific filtering. Forward fill is
used first, and backward fill only handles missing values at the beginning of a series.
This follows the explicit missing-value handling pattern used for PEMS in the iTransformer
codebase.

### Drop Columns With More Than 50% Missing Values

Applied to BDG2 and ASHRAE. This avoids retaining variates dominated by missing readings and
follows the BDG2 paper's documented use of substantial-missingness filters.

### Drop All-NaN Rows

Applied to KAMP because one raw charge file contained all-NaN padding rows. These rows are
file padding, not measurements.

### Group-Wise Fill for KAMP

KAMP concatenates charge records from multiple batteries. Filling across battery boundaries
would leak values from one battery into another, so fill operations are performed within
`SerialNumber` groups before `SerialNumber` is removed.

### No Time Reindexing for CARE Wind

CARE Wind contains gaps after filtering to normal operating status. Reindexing would create
synthetic rows and require interpolation across periods with no valid measurement. The
script therefore preserves the observed normal-operation rows and assigns a sequential
10-minute timestamp index for compatibility with the forecasting loader.

## Dataset-Specific Steps

### BASF NoBOOM Process Data

- Script: `preprocess_basf.py`
- Raw input expected by script: `dataset/industrial_datasets/noboom_basf/industry_process/{1..8}/train_normal_experiment_00{1..8}.csv`
- Output: `dataset/basf/basf.csv`
- Rows: 215,841
- Variates: 244
- Frequency argument: `h`
- Target: `T1` renamed to `OT`
- Missing values after preprocessing: 0

Processing steps:
1. Concatenate eight normal-operation process segments.
2. Remove the `Anomaly` label column.
3. Rename `T1` to `OT`.
4. Generate sequential hourly timestamps because the raw segments do not provide a shared timestamp column.

### KAMP Battery-Pack Data

- Script: `preprocess_kamp.py`
- Raw input expected by script: `dataset/industrial_datasets/Dataset_.../raw_data/train/*_chg.csv`
- Output: `dataset/kamp/kamp.csv`
- Rows: 32,415
- Variates: 228
- Frequency argument: `t`
- Target: `Voltage` renamed to `OT`
- Raw missing values: 475
- Interpolated fraction: 0.0006%

Processing steps:
1. Load 51 charge files and remove all-NaN padding rows.
2. Remove `Date` and `Time`; keep `SerialNumber` only for grouped filling.
3. Apply forward/backward fill within each `SerialNumber` group.
4. Drop any columns that remain missing after grouped filling.
5. Remove `SerialNumber`.
6. Apply 10x subsampling from the 10-second raw interval.
7. Rename `Voltage` to `OT` and generate sequential minutely timestamps.

### BDG2

- Script: `preprocess_bdg2.py`
- Raw input expected by script: `dataset/industrial_datasets/bdg2/.../data/meters/cleaned/{electricity,...}_cleaned.csv`
- Output: `dataset/bdg2/bdg2.csv`
- Rows: 17,544
- Variates: 2,817
- Frequency argument: `h`
- Target: first retained electricity meter renamed to `OT`
- Raw missing values after merge: 2,284,836
- Interpolated fraction: 4.6%

Processing steps:
1. Merge eight meter-type CSVs by timestamp after adding meter-type prefixes.
2. Drop columns with more than 50% missing values.
3. Apply forward/backward fill.
4. Drop all-zero columns.
5. Rename the selected electricity meter target to `OT`.

### ASHRAE Great Energy Predictor III

- Script: `preprocess_ashrae.py`
- Raw input expected by script: `dataset/industrial_datasets/ashrae/train.csv`
- Output: `dataset/ashrae_energy/ashrae.csv`
- Rows: 8,784
- Variates: 2,362
- Frequency argument: `h`
- Target: `b0_m0` renamed to `OT`
- Raw missing values after pivot and filtering: 575,274
- Interpolated fraction: 2.8%

Processing steps:
1. Pivot long-format records (`building_id`, `meter`, `timestamp`, `meter_reading`) into a wide matrix.
2. Drop columns with more than 50% missing values.
3. Apply forward/backward fill.
4. Rename `b0_m0` to `OT`.

### SDWPF

- Script: `preprocess_sdwpf.py`
- Raw input expected by script: `dataset/industrial_datasets/sdwpf/sdwpf_2001_2112_full.parquet`
- Parquet engine required: `pyarrow` (listed in `requirements.txt`).
- Output: `dataset/sdwpf_energy/sdwpf.csv`
- Rows: 84,785
- Variates: 2,144
- Frequency argument: `t`
- Target: `T001_Patv` renamed to `OT`
- Raw missing values: 4,995,708
- Interpolated fraction: 2.7%

Processing steps:
1. Pivot 134 turbines x 16 features into a wide matrix.
2. Apply forward/backward fill to SCADA and weather-feature gaps.
3. Rename `T001_Patv` to `OT`.

### CARE Wind Farm C

- Script: `preprocess_care_wind.py`
- Raw input expected by script: `dataset/industrial_datasets/care_wind/CARE_To_Compare/Wind Farm C/datasets/1.csv`
- Output: `dataset/care_wind_energy/care_wind.csv`
- Rows: 47,264
- Variates: 238
- Frequency argument: `t`
- Target: `power_6_avg` renamed to `OT`
- Raw missing values: 0
- Interpolated fraction: 0%

Processing steps:
1. Load the semicolon-delimited CSV.
2. Keep normal-operation rows with `status_type_id == 0`.
3. Retain `_avg` sensor columns and remove `_max`, `_min`, and `_std` columns.
4. Preserve the observed rows without reindexing.
5. Assign sequential 10-minute timestamps.
6. Rename `power_6_avg` to `OT`.

## Summary Table

| Dataset | Rows | Variates | Frequency argument | Target | Missing-value handling | Interpolated fraction |
|---|---:|---:|---|---|---|---:|
| BASF | 215,841 | 244 | `h` | `T1` | none required | 0% |
| KAMP | 32,415 | 228 | `t` | `Voltage` | group-wise ffill/bfill | 0.0006% |
| BDG2 | 17,544 | 2,817 | `h` | electricity meter | >50% drop, ffill/bfill | 4.6% |
| ASHRAE | 8,784 | 2,362 | `h` | `b0_m0` | >50% drop, ffill/bfill | 2.8% |
| SDWPF | 84,785 | 2,144 | `t` | `T001_Patv` | ffill/bfill | 2.7% |
| CARE Wind | 47,264 | 238 | `t` | `power_6_avg` | none required | 0% |

## References

1. Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting," ICLR 2024.
2. Miller et al., "The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition," Scientific Data 2020, DOI: 10.1038/s41597-020-00712-x.
3. Zhou et al., "SDWPF: A Dataset for Spatial Dynamic Wind Power Forecasting over a Large Turbine Array," Scientific Data 2024, DOI: 10.1038/s41597-024-03427-5.
