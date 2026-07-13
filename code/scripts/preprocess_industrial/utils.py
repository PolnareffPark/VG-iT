"""Shared utilities for industrial dataset preprocessing."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def validate_output(df: pd.DataFrame, name: str) -> None:
    """Validate that the output DataFrame is Dataset_Custom-compatible.

    Checks:
    - 'date' column exists and is first
    - 'OT' column exists and is last
    - No NaN values
    - All non-date columns are numeric
    """
    errors = []

    # Check 'date' column
    if "date" not in df.columns:
        errors.append("Missing 'date' column")
    elif df.columns[0] != "date":
        errors.append(f"'date' must be first column, got '{df.columns[0]}'")

    # Check 'OT' column
    if "OT" not in df.columns:
        errors.append("Missing 'OT' column")
    elif df.columns[-1] != "OT":
        errors.append(f"'OT' must be last column, got '{df.columns[-1]}'")

    # Check NaN
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        nan_cols = df.columns[df.isna().any()].tolist()
        errors.append(f"{nan_count} NaN values in columns: {nan_cols[:5]}")

    # Check numeric (all columns except 'date')
    for col in df.columns:
        if col == "date":
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            errors.append(f"Non-numeric column: '{col}' (dtype={df[col].dtype})")

    if errors:
        raise ValueError(f"[{name}] Validation failed:\n  " + "\n  ".join(errors))

    print(f"[{name}] Validation passed: {len(df)} rows, {len(df.columns)-1} vars (incl. OT)")


def print_summary(df: pd.DataFrame, name: str, freq: str) -> None:
    """Print dataset summary including split sizes and usable samples per PL."""
    n = len(df)
    n_vars = len(df.columns) - 1  # exclude 'date'
    n_train = int(n * 0.7)
    n_test = int(n * 0.2)
    n_val = n - n_train - n_test

    seq_len = 96
    pred_lens = [96, 192, 336, 720]

    print(f"\n{'='*60}")
    print(f"  Dataset: {name}")
    print(f"  Total rows: {n:,}")
    print(f"  Variables: {n_vars} (+ date)")
    print(f"  Freq: {freq}")
    print(f"  Train/Val/Test: {n_train:,} / {n_val:,} / {n_test:,}")
    print(f"  Date range: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
    print(f"  Usable test samples per PL:")
    for pl in pred_lens:
        usable = n_test - seq_len - pl + 1
        status = "OK" if usable > 100 else "LOW" if usable > 0 else "FAIL"
        print(f"    PL={pl:>3d}: {usable:>6,} [{status}]")
    print(f"{'='*60}\n")


def save_output(df: pd.DataFrame, output_dir: str, name: str,
                meta: dict | None = None) -> Path:
    """Save preprocessed DataFrame as CSV + metadata JSON.

    Args:
        df: Validated DataFrame with date + features + OT
        output_dir: Directory to save into (created if needed)
        name: Dataset name (used for filenames)
        meta: Optional metadata dict

    Returns:
        Path to saved CSV file
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_path = out_path / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[{name}] Saved: {csv_path} ({os.path.getsize(csv_path) / 1e6:.1f} MB)")

    # Save metadata
    if meta is None:
        meta = {}
    meta.update({
        "name": name,
        "rows": len(df),
        "n_vars": len(df.columns) - 1,
        "columns": list(df.columns),
        "target": "OT",
    })
    meta_path = out_path / f"{name}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return csv_path
