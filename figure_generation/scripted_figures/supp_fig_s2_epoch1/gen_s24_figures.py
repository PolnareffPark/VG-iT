"""Reproduce Supplementary Figure S2 from recorded epoch-1 losses."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
TEAL = "#2A9D8F"
CORAL = "#E76F51"
BG = "#FAFAFA"
SCREEN_THRESHOLD = 0.55
ANOMALY_THRESHOLD = 0.60

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": BG,
    "figure.facecolor": "white",
    "grid.color": "#E0E0E0",
    "grid.linewidth": 0.8,
})


def locate_input(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    relative_paths = (
        Path("results_canonical/rq2a_grouping/epoch1_losses.csv"),
        Path("test_results/published/rq2a_grouping/epoch1_losses.csv"),
    )
    for base in (HERE, *HERE.parents):
        for relative in relative_paths:
            candidate = base / relative
            if candidate.is_file():
                return candidate
    raise FileNotFoundError("Could not locate canonical epoch1_losses.csv")


def parse_bool(series: pd.Series) -> pd.Series:
    parsed = series.astype(str).str.strip().str.lower().map({
        "true": True, "false": False, "1": True, "0": False,
    })
    if parsed.isna().any():
        raise ValueError("is_anomaly contains an unrecognized Boolean value")
    return parsed.astype(bool)


def load_and_validate(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path)
    required = {
        "method", "dataset", "pred_len", "seed", "epoch1_train_loss",
        "final_mse", "is_anomaly", "is_cohesion",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if len(data) != 2880:
        raise ValueError(f"Expected 2,880 rows, got {len(data)}")
    if data.duplicated(["method", "dataset", "pred_len", "seed"]).any():
        raise ValueError("Duplicate run keys in epoch1_losses.csv")

    data["epoch1_train_loss"] = pd.to_numeric(data["epoch1_train_loss"], errors="raise")
    data["final_mse"] = pd.to_numeric(data["final_mse"], errors="raise")
    recorded = parse_bool(data["is_anomaly"])
    derived = data["final_mse"] > ANOMALY_THRESHOLD
    if not recorded.equals(derived):
        raise ValueError("is_anomaly does not match final_mse > 0.6")
    if int(recorded.sum()) != 25 or int((~recorded).sum()) != 2855:
        raise ValueError("Expected 25 anomaly and 2,855 normal runs")
    if set(data.loc[recorded, "dataset"].str.lower()) != {"traffic"}:
        raise ValueError("All recorded anomalies must be Traffic runs")
    return data, recorded


def render(data: pd.DataFrame, anomaly: pd.Series, output: Path) -> None:
    normal_loss = data.loc[~anomaly, "epoch1_train_loss"].to_numpy()
    anomaly_loss = data.loc[anomaly, "epoch1_train_loss"].to_numpy()
    screen = data["epoch1_train_loss"] > SCREEN_THRESHOLD
    tp = int((screen & anomaly).sum())
    fp = int((screen & ~anomaly).sum())
    traffic_normal = (~anomaly) & data["dataset"].str.lower().eq("traffic")
    if (tp, fp, int(traffic_normal.sum())) != (25, 12, 935):
        raise ValueError("Unexpected 0.55-screen contingency counts")

    overall_fpr = fp / int((~anomaly).sum())
    traffic_fpr = int((screen & traffic_normal).sum()) / int(traffic_normal.sum())
    lower = np.floor(min(normal_loss.min(), anomaly_loss.min()) * 100) / 100
    upper = np.ceil(max(normal_loss.max(), anomaly_loss.max()) * 100) / 100
    bins = np.arange(lower, upper + 0.011, 0.01)

    fig, (ax_n, ax_a) = plt.subplots(
        2, 1, figsize=(8, 5.2), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08},
    )
    ax_n.hist(normal_loss, bins=bins, color=TEAL, alpha=0.82, edgecolor="white", linewidth=0.35)
    anomaly_counts, _, _ = ax_a.hist(
        anomaly_loss, bins=bins, color=CORAL, alpha=0.88, edgecolor="white", linewidth=0.45
    )
    for axis in (ax_n, ax_a):
        axis.axvline(SCREEN_THRESHOLD, color="#333333", linewidth=1.8, linestyle="--")
        axis.grid(axis="x")
        axis.set_axisbelow(True)
    ax_n.set_ylabel("Normal-run count")
    ax_a.set_ylabel("Anomaly count")
    ax_a.set_xlabel("Recorded epoch-1 training loss")
    ax_a.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_a.set_ylim(0, max(anomaly_counts.max() + 1, 3))

    ax_n.text(
        0.02, 0.92,
        f"Normal runs: n = 2,855\nmean = {normal_loss.mean():.3f}; sample SD = {normal_loss.std(ddof=1):.3f}",
        transform=ax_n.transAxes, va="top", ha="left", fontsize=9, color="#1A6B63",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor=TEAL, alpha=0.92),
    )
    ax_n.text(
        0.98, 0.92,
        "Recall-first screen: $L_1 > 0.55$\n"
        f"TP {tp}/25; FP {fp}/2,855\n"
        f"Recall 100%; overall FPR {overall_fpr:.2%}",
        transform=ax_n.transAxes, va="top", ha="right", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor="#999999", alpha=0.94),
    )
    ax_n.legend(handles=[
        mpatches.Patch(color=TEAL, alpha=0.82, label="Normal (n = 2,855)"),
        mpatches.Patch(color=CORAL, alpha=0.88, label="Anomaly (n = 25; all Traffic)"),
        plt.Line2D([0], [0], color="#333333", linewidth=1.8, linestyle="--", label="Recall-first screen = 0.55"),
    ], loc="lower right", ncol=1, fontsize=8.3, framealpha=0.95)

    fig.suptitle("Epoch-1 training-loss diagnostic (2,880 runs; three datasets)", fontsize=12, y=0.985)
    fig.text(
        0.5, 0.94,
        f"Anomaly label: final test MSE > 0.6 (25 runs, all Traffic); Traffic-normal FPR at 0.55 = {traffic_fpr:.2%}",
        ha="center", va="top", fontsize=9.2, color="#444444",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.88, hspace=0.08)
    fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=HERE / "figures" / "s24_epoch1_distribution.png")
    args = parser.parse_args()
    path = locate_input(args.input)
    data, anomaly = load_and_validate(path)
    render(data, anomaly, args.output.resolve())


if __name__ == "__main__":
    main()
