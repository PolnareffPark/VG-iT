"""Preprocess ASHRAE Great Energy Predictor III dataset.

Source: train.csv in long format (building_id, meter, timestamp, meter_reading).
Output: Wide-format CSV with b{id}_m{meter} columns, b0_m0 → OT.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import print_summary, save_output, validate_output

# Paths
SRC_FILE = Path("dataset/industrial_datasets/ashrae/train.csv")
OUT_DIR = Path("dataset/ashrae_energy")


def main():
    print(f"[ASHRAE] Loading {SRC_FILE} ...")
    df = pd.read_csv(SRC_FILE)
    print(f"[ASHRAE] Raw: {len(df):,} rows")

    # Create column name from building_id and meter
    df["col_name"] = "b" + df["building_id"].astype(str) + "_m" + df["meter"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Pivot to wide format
    print("[ASHRAE] Pivoting to wide format ...")
    wide = df.pivot_table(
        index="timestamp",
        columns="col_name",
        values="meter_reading",
        aggfunc="first",
    )
    wide = wide.reset_index()
    wide = wide.rename(columns={"timestamp": "date"})
    wide = wide.sort_values("date").reset_index(drop=True)
    print(f"[ASHRAE] Pivoted: {len(wide):,} rows, {len(wide.columns)-1} vars")

    # Drop columns with >50% NaN (BDG2 paper threshold — Miller et al. 2020)
    numeric_cols = wide.select_dtypes(include="number").columns
    thresh = len(wide) * 0.5
    before_cols = len(wide.columns)
    wide = wide.dropna(axis=1, thresh=thresh)
    dropped_50 = before_cols - len(wide.columns)
    print(f"[ASHRAE] Dropped {dropped_50} cols (>50% NaN)")

    # Impute remaining NaN: ffill → bfill (iTransformer PEMS precedent)
    numeric_cols = wide.select_dtypes(include="number").columns
    nan_before = wide[numeric_cols].isna().sum().sum()
    wide[numeric_cols] = wide[numeric_cols].ffill().bfill()
    nan_after = wide[numeric_cols].isna().sum().sum()
    print(f"[ASHRAE] NaN after ffill/bfill: {nan_before:,} → {nan_after:,}")

    # Drop columns that still have NaN (entirely empty series)
    still_nan = wide[numeric_cols].columns[wide[numeric_cols].isna().any()].tolist()
    if still_nan:
        wide = wide.drop(columns=still_nan)
        print(f"[ASHRAE] Dropped {len(still_nan)} cols still with NaN")

    # Drop all-zero columns (meter never recorded any nonzero reading)
    numeric_cols = wide.select_dtypes(include="number").columns
    nonzero_mask = (wide[numeric_cols] != 0).any()
    zero_cols = nonzero_mask[~nonzero_mask].index.tolist()
    if zero_cols:
        wide = wide.drop(columns=zero_cols)
        print(f"[ASHRAE] Dropped {len(zero_cols)} all-zero columns")

    # Target: b0_m0 → OT
    target_col = "b0_m0"
    if target_col not in wide.columns:
        # Fall back to first available column
        avail = [c for c in wide.columns if c != "date"]
        target_col = avail[0]
        print(f"[ASHRAE] b0_m0 not found, using {target_col} as target")
    wide = wide.rename(columns={target_col: "OT"})

    # Reorder: date + features + OT
    cols = ["date"] + [c for c in wide.columns if c not in ("date", "OT")] + ["OT"]
    wide = wide[cols]

    # Format date
    wide["date"] = pd.to_datetime(wide["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Validate and save
    validate_output(wide, "ASHRAE")
    print_summary(wide, "ASHRAE", freq="h")
    save_output(wide, OUT_DIR, "ashrae", meta={
        "freq": "h",
        "source": "ASHRAE Great Energy Predictor III",
        "original_target": target_col,
    })


if __name__ == "__main__":
    main()
