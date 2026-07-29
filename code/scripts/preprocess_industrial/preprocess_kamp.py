"""Preprocess KAMP Battery Pack dataset.

Source: 51 battery charge files (*_chg.csv).
Output: Single CSV with synthetic sequential timestamps, Voltage → OT.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import print_summary, save_output, validate_output

# Paths
SRC_DIR = Path(
    "dataset/industrial_datasets/"
    "Dataset_전자부품(배터리팩) 품질보증 AI 데이터셋 (1)/"
    "data/raw_data/train"
)
OUT_DIR = Path("dataset/kamp")


def main():
    # Find all charge files
    chg_files = sorted(SRC_DIR.glob("*_chg.csv"))
    if not chg_files:
        raise FileNotFoundError(f"No *_chg.csv files found in {SRC_DIR}")
    print(f"[KAMP] Found {len(chg_files)} charge files")

    segments = []
    for f in chg_files:
        df_seg = pd.read_csv(f)
        # Drop entirely-NaN rows (padding found in some files, e.g. 1050_chg)
        before = len(df_seg)
        df_seg = df_seg.dropna(how="all").reset_index(drop=True)
        if len(df_seg) < before:
            print(f"  {f.name}: dropped {before - len(df_seg)} all-NaN rows")
        segments.append(df_seg)

    df = pd.concat(segments, ignore_index=True)
    print(f"[KAMP] Combined: {len(df):,} rows, {len(df.columns)} cols")

    # Drop non-sensor metadata columns (keep SerialNumber for grouped imputation)
    drop_cols = [c for c in ["Date", "Time"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Impute NaN within each battery (ffill/bfill per group, not across batteries)
    numeric_cols = [c for c in df.select_dtypes(include="number").columns
                    if c != "SerialNumber"]
    nan_before = df[numeric_cols].isna().sum().sum()
    df[numeric_cols] = df.groupby("SerialNumber")[numeric_cols].transform(
        lambda s: s.ffill().bfill()
    )
    nan_after = df[numeric_cols].isna().sum().sum()
    print(f"[KAMP] NaN after group-wise ffill/bfill: {nan_before:,} → {nan_after:,}")

    # Drop columns that still have NaN (sensor with no data in some battery)
    still_nan_cols = [c for c in numeric_cols
                      if c in df.columns and df[c].isna().any()]
    if still_nan_cols:
        df = df.drop(columns=still_nan_cols)
        print(f"[KAMP] Dropped {len(still_nan_cols)} cols still with NaN")

    # Now drop SerialNumber
    df = df.drop(columns=["SerialNumber"])

    # Subsample every 10th row (10-second data → ~100s intervals for manageability)
    df = df.iloc[::10].reset_index(drop=True)
    print(f"[KAMP] After 10x subsampling: {len(df):,} rows")

    # Rename Voltage → OT (target)
    if "Voltage" not in df.columns:
        raise KeyError("Target column 'Voltage' not found")
    df = df.rename(columns={"Voltage": "OT"})

    # Move OT to last position
    cols = [c for c in df.columns if c != "OT"] + ["OT"]
    df = df[cols]

    # Create synthetic sequential timestamps (battery cycles are not contiguous)
    start = pd.Timestamp("2020-01-01")
    df.insert(0, "date", pd.date_range(start, periods=len(df), freq="min"))
    df["date"] = df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Validate and save
    validate_output(df, "KAMP")
    print_summary(df, "KAMP", freq="t")
    save_output(df, OUT_DIR, "kamp", meta={"freq": "t", "source": "KAMP Battery"})


if __name__ == "__main__":
    main()
