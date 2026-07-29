"""Preprocess CARE Wind Farm C dataset.

Source: Semicolon-delimited CSV for single turbine.
Output: CSV with only _avg columns, power_6_avg → OT.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import print_summary, save_output, validate_output

# Paths
SRC_FILE = Path(
    "dataset/industrial_datasets/care_wind/"
    "CARE_To_Compare/Wind Farm C/datasets/1.csv"
)
OUT_DIR = Path("dataset/care_wind_energy")


def main():
    print(f"[CARE Wind] Loading {SRC_FILE} ...")
    df = pd.read_csv(SRC_FILE, sep=";")
    print(f"[CARE Wind] Raw: {len(df):,} rows, {len(df.columns)} cols")

    # Filter: only normal operation (status_type_id == 0)
    if "status_type_id" in df.columns:
        before = len(df)
        df = df[df["status_type_id"] == 0].reset_index(drop=True)
        print(f"[CARE Wind] Filtered status_type_id==0: {before:,} → {len(df):,}")

    # Keep only _avg columns (remove _max, _min, _std for parsimony)
    avg_cols = [c for c in df.columns if c.endswith("_avg")]
    meta_cols = ["time_stamp"]
    keep_cols = meta_cols + avg_cols
    df = df[[c for c in keep_cols if c in df.columns]]
    print(f"[CARE Wind] Kept _avg columns: {len(avg_cols)}")

    # Rename time_stamp → date
    df = df.rename(columns={"time_stamp": "date"})
    df["date"] = pd.to_datetime(df["date"])

    # Sort by original timestamps, then drop gaps
    # Gaps come from non-normal operation periods (status_type_id != 0).
    # Reindexing would fabricate data for periods the turbine was offline.
    # Instead: use original rows as-is with synthetic sequential timestamps (like KAMP).
    df = df.sort_values("date").reset_index(drop=True)
    print(f"[CARE Wind] Original NaN: {df.select_dtypes(include='number').isna().sum().sum()}")

    # Assign synthetic sequential 10-minute timestamps (gaps discarded)
    start = pd.Timestamp("2022-10-01")
    df["date"] = pd.date_range(start, periods=len(df), freq="10min")

    # Target: power_6_avg → OT
    target_col = "power_6_avg"
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found")
    df = df.rename(columns={target_col: "OT"})

    # Reorder: date + features + OT
    cols = ["date"] + [c for c in df.columns if c not in ("date", "OT")] + ["OT"]
    df = df[cols]

    # Format date
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Validate and save
    validate_output(df, "CARE Wind")
    print_summary(df, "CARE Wind", freq="t")
    save_output(df, OUT_DIR, "care_wind", meta={
        "freq": "t",
        "source": "CARE Wind Farm C, Turbine 1",
        "original_target": target_col,
    })


if __name__ == "__main__":
    main()
