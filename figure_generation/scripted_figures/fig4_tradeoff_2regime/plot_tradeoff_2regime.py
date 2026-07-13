"""
Two-regime accuracy-efficiency summary of VG-iT against all baselines.

Left  (Academic):   multi-model accuracy-cost Pareto. x = FLOPs at Traffic (log),
                    y = mean MSE rank over all academic datasets x horizons
                    (1 = best, axis inverted so up = better), bubble = train VRAM.
                    VG-iT sits on the knee and dominates the iTransformer family,
                    SOFTS, S-Mamba and TimeMixer; only Gateformer is more accurate,
                    at ~15x the FLOPs.

Right (Industrial): multi-model resource scaling. x = variate count N (log),
                    y = FLOPs (GFLOPs, log), marker area = peak train VRAM. Among the
                    variate-attention family, iTransformer explodes on both FLOPs and
                    memory and iFlashformer escapes only memory (its FLOPs track
                    iTransformer); VG-iT is the only one that curbs both. VG-iT matches
                    iTransformer MSE within +-2.5% across the six datasets (Table 4).

Each panel plays the regime's honest strength: academic accuracy-per-FLOP, and
industrial resource scaling. Per-condition values for all eight metrics are in the
main and supplementary tables. Reuses the data conventions of
plot_pareto_frontier_ratios.py.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEPOSIT_ROOT = SCRIPT_DIR.parents[2]
DATA_DIR = DEPOSIT_ROOT / "results_canonical" / "shared_consolidated_5seed"
OUT = SCRIPT_DIR / "figures" / "tradeoff_2regime.png"

DISPLAY = {
    "VG_iTransformer_SF": "VG-iT",
    "VG_iFlashformer_SF": "VGFlash",
    "iNystromformer": "iNystromformer",
    "Nonstationary_Transformer": "Non-stationary Transformer",
    "PatchTST_sl96": "PatchTST",
}

ACAD_DATASETS = ["solar_AL", "electricity", "traffic"]
IND_DATASETS = ["kamp", "care_wind", "basf", "sdwpf", "ashrae", "bdg2"]
IND_MODELS = ["VG_iTransformer_SF", "VG_iFlashformer_SF", "iTransformer",
              "iFlashformer", "S-Mamba", "SOFTS", "TimeMixer"]

FAMILY = {
    "iTransformer": "variate", "iFlashformer": "variate",
    "iNystromformer": "variate", "VarDrop": "variate", "Gateformer": "variate",
    "Transformer": "temporal", "PatchTST": "temporal",
    "Non-stationary Transformer": "temporal", "Crossformer": "temporal",
    "TimeMixer": "channel", "S-Mamba": "ssm",
    "DLinear": "non-attention", "TSMixer": "non-attention", "SOFTS": "non-attention",
}
FAMILY_COLOR = {
    "variate": "#2f6fa3", "temporal": "#9aa1aa", "channel": "#8a7f2a",
    "ssm": "#7b5aa6", "non-attention": "#2e8b67",
}
VG_COLOR, VGF_COLOR = "#c9342c", "#df6d2f"
# Per-model colours for the industrial line panel.
IND_COLOR = {
    "VG-iT": VG_COLOR, "VGFlash": VGF_COLOR,
    "iTransformer": "#1f2328", "iFlashformer": "#2f6fa3",
    "S-Mamba": "#7b5aa6", "SOFTS": "#2e8b67", "TimeMixer": "#8a7f2a",
}


# ---------- accuracy (academic ranks) ----------
def load_accuracy() -> pd.DataFrame:
    frames = [pd.read_csv(DATA_DIR / n) for n in
              ("academic_baselines.csv", "industrial_baselines.csv",
               "vgflash_generality.csv")]
    for name in ("academic_grouping.csv", "industrial_grouping.csv"):
        frame = pd.read_csv(DATA_DIR / name)
        frame = frame[frame["model"] == "ordered"].copy()
        frame["model"] = "VG_iTransformer_SF"
        if name.startswith("academic"):
            frame["seed"] = pd.to_numeric(frame["seed"], errors="coerce")
            frame = frame[frame["seed"].between(2021, 2025)]
        frames.append(frame)
    acc = pd.concat(frames, ignore_index=True)
    acc["model"] = acc["model"].replace(DISPLAY)
    acc["MSE"] = pd.to_numeric(acc["MSE"], errors="coerce")
    acc["pred_len"] = pd.to_numeric(acc["pred_len"], errors="coerce")
    acc = acc.dropna(subset=["MSE", "pred_len"])
    return acc.groupby(["dataset", "pred_len", "model"], as_index=False).agg(
        MSE=("MSE", "mean"))


def academic_points() -> pd.DataFrame:
    acc = load_accuracy()
    sub = acc[acc["dataset"].isin(ACAD_DATASETS)].copy()
    sub["rank"] = sub.groupby(["dataset", "pred_len"])["MSE"].rank(method="average")
    ranks = sub.groupby("model", as_index=False).agg(mse_rank=("rank", "mean"))

    eff = pd.read_csv(DATA_DIR / "efficiency_profile.csv")
    eff = eff[(eff["dataset"] == "traffic") & (eff["pred_len"] == 96)].copy()
    eff["model"] = eff["model"].replace(DISPLAY)
    for col in ("FLOPs_G", "train_vram_GB"):
        eff[col] = pd.to_numeric(eff[col], errors="coerce")
    eff = eff.groupby("model", as_index=False).agg(
        flops=("FLOPs_G", "mean"), train_vram=("train_vram_GB", "mean"))

    df = ranks.merge(eff, on="model", how="inner")
    flops, rank = df["flops"].to_numpy(float), df["mse_rank"].to_numpy(float)
    keep = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        for j in range(len(df)):
            if i != j and flops[j] <= flops[i] and rank[j] <= rank[i] \
                    and (flops[j] < flops[i] or rank[j] < rank[i]):
                keep[i] = False
                break
    df["pareto"] = keep
    return df.sort_values("flops").reset_index(drop=True)


def industrial_series() -> pd.DataFrame:
    eff = pd.read_csv(DATA_DIR / "efficiency_profile.csv")
    eff = eff[(eff["pred_len"] == 96) & eff["model"].isin(IND_MODELS)
              & eff["dataset"].isin(IND_DATASETS)].copy()
    eff["model"] = eff["model"].replace(DISPLAY)
    for col in ("FLOPs_G", "train_vram_GB", "N"):
        eff[col] = pd.to_numeric(eff[col], errors="coerce")
    return eff.groupby(["model", "dataset"], as_index=False).agg(
        N=("N", "mean"), flops=("FLOPs_G", "mean"),
        train_vram=("train_vram_GB", "mean")).sort_values("N")


def vram_size(v: np.ndarray) -> np.ndarray:
    return 14.0 + 16.0 * np.asarray(v)


# ---------- drawing ----------
def draw_academic(ax, log: bool = True) -> None:
    df = academic_points()
    front = df[df["pareto"]].sort_values("flops")
    ax.plot(front["flops"], front["mse_rank"], color="#171a1f", lw=1.1,
            ls=(0, (4, 2)), alpha=0.6, zorder=2)

    for _, r in df.iterrows():
        m = r["model"]
        if m == "VG-iT":
            # circle (not star) so marker area stays comparable to the baselines
            color, marker, edge, lw, z = VG_COLOR, "o", "white", 0.7, 8
            s = vram_size(r["train_vram"])
        elif m == "VGFlash":
            color, marker, edge, lw, z = VGF_COLOR, "o", "white", 0.7, 7
            s = vram_size(r["train_vram"])
        else:
            color, marker, edge, lw, z = FAMILY_COLOR[FAMILY[m]], "o", "white", 0.7, 4
            s = vram_size(r["train_vram"])
        ax.scatter(r["flops"], r["mse_rank"], s=s, marker=marker, color=color,
                   edgecolors=edge, linewidths=lw,
                   alpha=0.96 if m in ("VG-iT", "VGFlash") else 0.82, zorder=z)

    no_leader = set()  # every label gets a connector line to its marker
    labelled = set(df["model"])  # label every model
    offsets = {"VG-iT": (-48, 22), "VGFlash": (-22, 16),
               "SOFTS": (16, -14), "iNystromformer": (18, -26),
               "TimeMixer": (16, -16), "iFlashformer": (0, 32),
               "iTransformer": (40, -10), "VarDrop": (8, -14),
               "S-Mamba": (14, 13), "Gateformer": (-22, 16),
               "PatchTST": (-14, 12), "DLinear": (20, -6),
               "TSMixer": (-14, 14), "Transformer": (-12, -10),
               "Non-stationary Transformer": (10, -12), "Crossformer": (10, -12)}
    for _, r in df.iterrows():
        m = r["model"]
        if m not in labelled:
            continue
        bold = m in ("VG-iT", "VGFlash")
        col = VG_COLOR if m == "VG-iT" else VGF_COLOR if m == "VGFlash" else "#2a2e34"
        arrow = (None if m in no_leader else
                 dict(arrowstyle="-", color="#b5bcc6", lw=0.8, shrinkA=0, shrinkB=2))
        dx, dy = offsets.get(m, (10, 8))
        # align text away from the marker so labels extend into open space,
        # not back over the dense cluster (critical on the log x-axis)
        ha = "right" if dx <= -8 else "left" if dx >= 8 else "center"
        va = "bottom" if dy >= 6 else "top" if dy <= -6 else "center"
        ax.annotate(m, (r["flops"], r["mse_rank"]), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, va=va,
                    fontsize=12 if bold else 11, color=col,
                    weight="bold" if bold else "normal", zorder=9, arrowprops=arrow)

    ax.set_xscale("log" if log else "linear")
    if log:
        ax.set_xlim(0.007, 300)
    ax.set_ylim(15.6, 2.7)  # inverted (up = better); head/foot room so end labels don't clip
    ax.grid(True, color="#eef1f5", lw=0.8, zorder=0)
    for sp in ax.spines.values():
        sp.set_color("#ccd2da")
    ax.tick_params(axis="both", labelsize=13)
    ax.set_xlabel("FLOPs at Traffic, T=96  (GFLOPs, log)" if log
                  else "FLOPs at Traffic, T=96  (GFLOPs, linear)", fontsize=15)
    ax.set_ylabel("Mean MSE rank  (1 = best)", fontsize=15)
    ax.set_title("(a) Academic (16 models)",
                 fontsize=16, weight="bold", loc="left", pad=6)
    ax.text(0.985, 0.03, "upper-left = more accurate & cheaper", transform=ax.transAxes,
            fontsize=11, color="#586271", style="italic", ha="right", va="bottom")
    # VRAM marker-size key (matches the industrial panel; bubble area = train VRAM)
    for gb in (1, 5, 15):
        ax.scatter([], [], s=vram_size(gb), color="#9aa1aa", edgecolors="white",
                   linewidths=0.8, label=f"{gb} GB")
    ax.legend(title="Marker area = train VRAM", loc="upper left", fontsize=11.5,
              title_fontsize=11.5, frameon=False, labelspacing=1.1, borderpad=0.6)


def draw_industrial(ax, log: bool = True) -> None:
    series = industrial_series()
    order = ["iTransformer", "iFlashformer", "S-Mamba", "SOFTS", "TimeMixer",
             "VGFlash", "VG-iT"]
    for m in order:
        s = series[series["model"] == m]
        if s.empty:
            continue
        is_vg = m in ("VG-iT", "VGFlash")
        ls = "-" if m in ("VG-iT", "iTransformer", "S-Mamba", "SOFTS", "TimeMixer") else (0, (5, 2))
        ax.plot(s["N"], s["flops"], color=IND_COLOR[m], lw=2.2 if m == "VG-iT" else 1.3,
                ls=ls, alpha=0.9, zorder=6 if is_vg else 4)
        ax.scatter(s["N"], s["flops"], s=vram_size(s["train_vram"].to_numpy()),
                   color=IND_COLOR[m], edgecolors="white", linewidths=0.8,
                   zorder=7 if is_vg else 5, marker="o")
        # right-edge label at the largest N; offset the FLOPs-identical pairs
        last = s.iloc[-1]
        edge = {"iFlashformer": (12, 13), "iTransformer": (12, -14),
                "S-Mamba": (14, 0), "SOFTS": (14, 2),
                "VGFlash": (14, 13), "VG-iT": (14, -2),
                "TimeMixer": (14, -15)}.get(m, (13, 0))
        ax.annotate(m, (last["N"], last["flops"]), textcoords="offset points",
                    xytext=edge, fontsize=12 if is_vg else 11,
                    color=IND_COLOR[m], weight="bold" if is_vg else "normal",
                    va="center", zorder=9)

    ax.set_xscale("log" if log else "linear")
    ax.set_yscale("log" if log else "linear")
    ax.grid(True, which="both", color="#eef1f5", lw=0.8, zorder=0)
    for sp in ax.spines.values():
        sp.set_color("#ccd2da")
    ax.tick_params(axis="both", labelsize=13)
    if log:
        ax.set_xlim(200, 4200)
    else:
        ax.set_xlim(40, 3700)
        ax.set_ylim(-3, 76)
    ax.set_xlabel("Number of variates  N", fontsize=15)
    ax.set_ylabel("FLOPs  (GFLOPs)", fontsize=15)
    ax.set_title("(b) Industrial (7 models)",
                 fontsize=16, weight="bold", loc="left", pad=6)
    # VRAM marker-size key
    for i, gb in enumerate((1, 5, 15)):
        ax.scatter([], [], s=vram_size(gb), color="#9aa1aa", edgecolors="white",
                   linewidths=0.8, label=f"{gb} GB")
    ax.legend(title="Marker area = train VRAM", loc="upper left", fontsize=11.5,
              title_fontsize=11.5, frameon=False, labelspacing=1.1, borderpad=0.6)


def render(out_path: Path, academic_log: bool = True,
           industrial_log: bool = False) -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.0, 5.9))
    draw_academic(axL, log=academic_log)
    draw_industrial(axR, log=industrial_log)

    fam_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=VG_COLOR,
                   markeredgecolor="white", markersize=12, linestyle="None"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=VGF_COLOR,
                   markeredgecolor="white", markersize=11, linestyle="None"),
    ] + [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=c,
                    markeredgecolor="white", markersize=10, linestyle="None")
         for c in FAMILY_COLOR.values()]
    fam_labels = ["VG-iT (ours)", "VGFlash (ours)", "Variate-axis attn.",
                  "Temporal / cross-axis", "Channel-indep.", "State-space",
                  "Non-attention"]
    fig.legend(fam_handles, fam_labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.005), frameon=False, fontsize=11,
               handletextpad=0.4, columnspacing=1.8)

    fig.subplots_adjust(left=0.075, right=0.97, bottom=0.22, top=0.91, wspace=0.20)

    fig.savefig(out_path, dpi=600)  # >=500 dpi: Elsevier combination (line+halftone) artwork spec
    fig.savefig(out_path.with_suffix(".pdf"))  # vector copy for final submission
    plt.close(fig)
    print(f"saved {out_path} (+ .pdf)")


def main() -> None:
    print(academic_points()[["model", "mse_rank", "flops", "train_vram", "pareto"]]
          .to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    render(OUT, academic_log=True, industrial_log=False)


if __name__ == "__main__":
    main()
