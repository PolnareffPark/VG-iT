"""Preprocess NoBOOM BASF industrial process dataset.

Source: 8 industrial process segments, each with train_normal CSV.
Output: Single CSV with synthetic hourly timestamps, T1 → OT.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import print_summary, save_output, validate_output

# Paths
SRC_DIR = Path("dataset/industrial_datasets/noboom_basf/industry_process")
OUT_DIR = Path("dataset/basf")


def main():
    segments = []
    for i in range(1, 9):
        csv_path = SRC_DIR / str(i) / f"train_normal_experiment_{i:03d}.csv"
        if not csv_path.exists():
            print(f"[BASF] WARNING: {csv_path} not found, skipping")
            continue
        df = pd.read_csv(csv_path)
        print(f"[BASF] Segment {i}: {len(df):,} rows, {len(df.columns)} cols")
        segments.append(df)

    if not segments:
        raise FileNotFoundError(f"No BASF data found in {SRC_DIR}")

    # Concatenate all segments
    df = pd.concat(segments, ignore_index=True)
    print(f"[BASF] Combined: {len(df):,} rows")

    # Drop Anomaly column (label, not a sensor)
    if "Anomaly" in df.columns:
        df = df.drop(columns=["Anomaly"])

    # Rename T1 → OT (target)
    if "T1" not in df.columns:
        raise KeyError("Target column 'T1' not found")
    df = df.rename(columns={"T1": "OT"})

    # Move OT to last position
    cols = [c for c in df.columns if c != "OT"] + ["OT"]
    df = df[cols]

    # Create synthetic hourly timestamps (no real timestamps in BASF)
    start = pd.Timestamp("2020-01-01")
    df.insert(0, "date", pd.date_range(start, periods=len(df), freq="h"))
    df["date"] = df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Validate and save
    validate_output(df, "BASF")
    print_summary(df, "BASF", freq="h")
    save_output(df, OUT_DIR, "basf", meta={"freq": "h", "source": "NoBOOM BASF"})


if __name__ == "__main__":
    main()
