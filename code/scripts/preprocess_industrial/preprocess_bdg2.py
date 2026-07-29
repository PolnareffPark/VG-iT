"""Preprocess Building Data Genome Project 2 (BDG2) dataset.

Source: 8 meter-type CSVs (electricity, steam, etc.) with building-level columns.
Output: Single merged CSV with meter-type prefixed columns, first electricity building → OT.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import print_summary, save_output, validate_output

# Paths
SRC_DIR = Path(
    "dataset/industrial_datasets/bdg2/"
    "building-data-genome-project-2-master/data/meters/cleaned"
)
OUT_DIR = Path("dataset/bdg2")

METER_TYPES = [
    "electricity", "chilledwater", "steam", "hotwater",
    "gas", "water", "irrigation", "solar",
]


def main():
    dfs = {}
    for meter in METER_TYPES:
        csv_path = SRC_DIR / f"{meter}_cleaned.csv"
        if not csv_path.exists():
            print(f"[BDG2] WARNING: {csv_path} not found, skipping")
            continue
        df = pd.read_csv(csv_path)
        df = df.rename(columns={"timestamp": "date"})
        # Add meter-type prefix to building columns
        rename_map = {
            col: f"{meter[:4]}_{col}"
            for col in df.columns if col != "date"
        }
        df = df.rename(columns=rename_map)
        dfs[meter] = df
        print(f"[BDG2] {meter}: {len(df):,} rows, {len(df.columns)-1} buildings")

    if not dfs:
        raise FileNotFoundError(f"No BDG2 data found in {SRC_DIR}")

    # Merge all on date
    merged = dfs[list(dfs.keys())[0]]
    for meter in list(dfs.keys())[1:]:
        merged = merged.merge(dfs[meter], on="date", how="outer")

    print(f"[BDG2] Merged: {len(merged):,} rows, {len(merged.columns)-1} cols")

    # Drop columns with >50% NaN
    thresh = len(merged) * 0.5
    before_cols = len(merged.columns)
    merged = merged.dropna(axis=1, thresh=thresh)
    print(f"[BDG2] Dropped {before_cols - len(merged.columns)} cols (>50% NaN)")

    # Impute remaining NaN: ffill → bfill (iTransformer PEMS precedent)
    numeric_cols = merged.select_dtypes(include="number").columns
    nan_before = merged[numeric_cols].isna().sum().sum()
    merged[numeric_cols] = merged[numeric_cols].ffill().bfill()
    nan_after = merged[numeric_cols].isna().sum().sum()
    print(f"[BDG2] NaN after ffill/bfill: {nan_before:,} → {nan_after:,}")

    # Drop columns that still have NaN (entirely empty series)
    still_nan = merged[numeric_cols].columns[merged[numeric_cols].isna().any()].tolist()
    if still_nan:
        merged = merged.drop(columns=still_nan)
        print(f"[BDG2] Dropped {len(still_nan)} cols still with NaN")

    # Drop all-zero columns (meter never recorded any reading)
    numeric_cols = merged.select_dtypes(include="number").columns
    nonzero_mask = (merged[numeric_cols] != 0).any()
    zero_cols = nonzero_mask[~nonzero_mask].index.tolist()
    if zero_cols:
        merged = merged.drop(columns=zero_cols)
        print(f"[BDG2] Dropped {len(zero_cols)} all-zero columns")

    # Select target: first electricity building column → OT
    elec_cols = [c for c in merged.columns if c.startswith("elec_") and c != "date"]
    if not elec_cols:
        raise KeyError("No electricity columns found after filtering")
    target_col = elec_cols[0]
    print(f"[BDG2] Target: {target_col} → OT")
    merged = merged.rename(columns={target_col: "OT"})

    # Reorder: date + features + OT
    cols = ["date"] + [c for c in merged.columns if c not in ("date", "OT")] + ["OT"]
    merged = merged[cols]

    # Validate and save
    validate_output(merged, "BDG2")
    print_summary(merged, "BDG2", freq="h")
    save_output(merged, OUT_DIR, "bdg2", meta={
        "freq": "h",
        "source": "Building Data Genome Project 2",
        "original_target": target_col,
    })


if __name__ == "__main__":
    main()
