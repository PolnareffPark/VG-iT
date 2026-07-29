"""Preprocess Spatial Dynamic Wind Power Forecasting (SDWPF) dataset.

Source: Parquet file with 134 turbines × 16 features in long format.
Output: Wide-format CSV with T{id:03d}_{feature} columns, T001_Patv → OT.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import print_summary, save_output, validate_output

# Paths
SRC_FILE = Path("dataset/industrial_datasets/sdwpf/sdwpf_2001_2112_full.parquet")
OUT_DIR = Path("dataset/sdwpf_energy")

# Features to pivot (all except TurbID and Tmstamp)
FEATURES = [
    "Wspd", "Wdir", "Etmp", "Itmp", "Ndir",
    "Pab1", "Pab2", "Pab3", "Prtv", "T2m",
    "Sp", "RelH", "Wspd_w", "Wdir_w", "Tp", "Patv",
]


def main():
    print(f"[SDWPF] Loading {SRC_FILE} ...")
    df = pd.read_parquet(SRC_FILE)
    print(f"[SDWPF] Raw: {len(df):,} rows, {df['TurbID'].nunique()} turbines")

    # Parse timestamps
    df["Tmstamp"] = pd.to_datetime(df["Tmstamp"])

    # Sort by timestamp and turbine
    df = df.sort_values(["Tmstamp", "TurbID"]).reset_index(drop=True)

    # Get unique timestamps
    timestamps = sorted(df["Tmstamp"].unique())
    print(f"[SDWPF] Timestamps: {len(timestamps):,}")

    # Pivot each feature separately then merge (memory-efficient)
    print("[SDWPF] Pivoting to wide format ...")
    result = pd.DataFrame({"date": timestamps})

    for feat in FEATURES:
        print(f"  Pivoting {feat} ...", end=" ", flush=True)
        pivot = df.pivot_table(
            index="Tmstamp",
            columns="TurbID",
            values=feat,
            aggfunc="first",
        )
        # Rename columns: TurbID → T{id:03d}_{feature}
        pivot.columns = [f"T{int(tid):03d}_{feat}" for tid in pivot.columns]
        pivot = pivot.reset_index().rename(columns={"Tmstamp": "date"})
        result = result.merge(pivot, on="date", how="left")
        print(f"{len(pivot.columns)-1} cols")

    print(f"[SDWPF] Wide: {len(result):,} rows, {len(result.columns)-1} vars")

    # Impute NaN: ffill → bfill (iTransformer PEMS precedent)
    numeric_cols = result.select_dtypes(include="number").columns
    nan_before = result[numeric_cols].isna().sum().sum()
    result[numeric_cols] = result[numeric_cols].ffill().bfill()
    nan_after = result[numeric_cols].isna().sum().sum()
    print(f"[SDWPF] NaN after ffill/bfill: {nan_before:,} → {nan_after:,}")

    # Drop columns that still have NaN (turbine-feature with no data at all)
    still_nan = result[numeric_cols].columns[result[numeric_cols].isna().any()].tolist()
    if still_nan:
        result = result.drop(columns=still_nan)
        print(f"[SDWPF] Dropped {len(still_nan)} cols still with NaN")

    # Target: T001_Patv → OT
    target_col = "T001_Patv"
    if target_col not in result.columns:
        raise KeyError(f"Target column '{target_col}' not found")
    result = result.rename(columns={target_col: "OT"})

    # Reorder: date + features + OT
    cols = ["date"] + [c for c in result.columns if c not in ("date", "OT")] + ["OT"]
    result = result[cols]

    # Format date
    result["date"] = pd.to_datetime(result["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Validate and save
    validate_output(result, "SDWPF")
    print_summary(result, "SDWPF", freq="t")
    save_output(result, OUT_DIR, "sdwpf", meta={
        "freq": "t",
        "source": "SDWPF (Baidu KDD Cup 2022)",
        "original_target": target_col,
    })


if __name__ == "__main__":
    main()
