#!/usr/bin/env python3
"""Reproduce manuscript-linked grouping statistics from the archived CSV.

The ROPE calculation intentionally uses ``baycomp.two_on_single`` from
baycomp 1.0.3. That API implements a Bayesian correlated t-test, not a
Bayesian signed-rank test. Generated output is structured JSON; no narrative
analysis report is stored in the release.
"""

import argparse
import importlib.metadata
import json
from itertools import combinations
from pathlib import Path

import baycomp
import numpy as np
import pandas as pd
from scipy import stats


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    REPOSITORY_ROOT
    / "results_canonical"
    / "rq2a_grouping"
    / "grouping_invariance_results.csv"
)
BLOCK_COLUMNS = ["dataset", "pred_len", "seed"]
KEY_COLUMNS = ["method", *BLOCK_COLUMNS]
METHODS = [
    "finch_like",
    "coarsening",
    "mi_based",
    "anti_clustering",
    "score_stratified",
    "maximin_dispersion",
    "ordered",
    "random",
]
COHESION = ["finch_like", "coarsening", "mi_based"]
DIVERSITY = ["anti_clustering", "score_stratified", "maximin_dispersion"]
THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8, 1.0, None]


def validate_input(frame):
    required = {"method", "dataset", "N", "pred_len", "seed", "MSE", "MAE", "model_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(frame) != 2880:
        raise ValueError(f"Expected 2,880 rows, found {len(frame)}")
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError("Duplicate method/dataset/pred_len/seed rows found")
    if set(frame["method"]) != set(METHODS):
        raise ValueError("Unexpected method inventory")
    if set(frame["dataset"]) != {"electricity", "solar_AL", "traffic"}:
        raise ValueError("Unexpected dataset inventory")
    if set(frame["pred_len"]) != {96, 192, 336, 720}:
        raise ValueError("Unexpected prediction-length inventory")
    if set(frame["seed"]) != set(range(2021, 2051)):
        raise ValueError("Unexpected random-seed inventory")
    block_sizes = frame.groupby(BLOCK_COLUMNS)["method"].nunique()
    if len(block_sizes) != 360 or not (block_sizes == 8).all():
        raise ValueError("Expected 360 complete eight-method blocks")


def clean_blocks(frame, threshold):
    if threshold is None:
        return frame.copy(), 0
    block_max = frame.groupby(BLOCK_COLUMNS)["MSE"].max()
    bad_blocks = block_max[block_max > threshold].index
    clean = frame.set_index(BLOCK_COLUMNS).drop(index=bad_blocks).reset_index()
    return clean, len(bad_blocks)


def pivot(frame):
    wide = frame.pivot(index=BLOCK_COLUMNS, columns="method", values="MSE")
    if wide[METHODS].isna().any().any():
        raise ValueError("Incomplete method block after pivot")
    return wide[METHODS]


def paired_cohens_d(x, y):
    difference = np.asarray(x) - np.asarray(y)
    standard_deviation = difference.std(ddof=1)
    return 0.0 if standard_deviation == 0 else float(difference.mean() / standard_deviation)


def holm_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, values[index] * (total - rank))
        adjusted[index] = min(running, 1.0)
    return adjusted


def correlated_t_rope(x, y, rope):
    # baycomp uses delta = y - x; swap tails to report the x - y convention.
    bayes_left, probability_rope, bayes_right = baycomp.two_on_single(x, y, rope=rope)
    probability_left = float(bayes_right)
    probability_right = float(bayes_left)
    probability_rope = float(probability_rope)
    if probability_rope > 0.95:
        decision = "EQUIVALENT"
    elif probability_left > 0.95:
        decision = "LEFT_BETTER"
    elif probability_right > 0.95:
        decision = "RIGHT_BETTER"
    else:
        decision = "INCONCLUSIVE"
    return probability_left, probability_rope, probability_right, decision


def pairwise_rope(frame):
    wide = pivot(frame)
    rope = 0.01 * float(frame["MSE"].mean())
    records = []
    for left, right in combinations(METHODS, 2):
        x = wide[left].to_numpy()
        y = wide[right].to_numpy()
        p_left, p_rope, p_right, decision = correlated_t_rope(x, y, rope)
        wilcoxon = stats.wilcoxon(x, y, alternative="two-sided")
        records.append(
            {
                "left": left,
                "right": right,
                "n_blocks": len(wide),
                "mean_difference_left_minus_right": float((x - y).mean()),
                "paired_cohens_d": paired_cohens_d(x, y),
                "probability_left_better": p_left,
                "probability_rope": p_rope,
                "probability_right_better": p_right,
                "decision": decision,
                "wilcoxon_statistic": float(wilcoxon.statistic),
                "wilcoxon_p": float(wilcoxon.pvalue),
            }
        )
    return rope, records


def friedman_wilcoxon(frame):
    wide = pivot(frame)
    arrays = [wide[method].to_numpy() for method in METHODS]
    friedman = stats.friedmanchisquare(*arrays)
    pairs = list(combinations(METHODS, 2))
    raw_p = []
    statistics = []
    for left, right in pairs:
        result = stats.wilcoxon(wide[left], wide[right], alternative="two-sided")
        statistics.append(float(result.statistic))
        raw_p.append(float(result.pvalue))
    adjusted = holm_adjust(raw_p)
    posthoc = [
        {
            "left": left,
            "right": right,
            "statistic": statistics[index],
            "p_raw": raw_p[index],
            "p_holm": float(adjusted[index]),
            "significant_holm_0_05": bool(adjusted[index] < 0.05),
        }
        for index, (left, right) in enumerate(pairs)
    ]
    mean_ranks = wide.rank(axis=1, method="average").mean()
    return {
        "n_blocks": len(wide),
        "chi_square": float(friedman.statistic),
        "p_value": float(friedman.pvalue),
        "kendalls_w": float(friedman.statistic / (len(wide) * (len(METHODS) - 1))),
        "mean_ranks": {method: float(mean_ranks[method]) for method in METHODS},
        "wilcoxon_holm": posthoc,
    }


def category_dataset_comparisons(clean_frame):
    wide = pivot(clean_frame).copy()
    categories = pd.DataFrame(index=wide.index)
    categories["ordered_control_aggregate"] = wide[["ordered", "random"]].mean(axis=1)
    categories["diversity"] = wide[DIVERSITY].mean(axis=1)
    categories["cohesion"] = wide[COHESION].mean(axis=1)
    category_pairs = list(combinations(categories.columns, 2))
    output = []
    for dataset in sorted(clean_frame["dataset"].unique()):
        subset = categories.xs(dataset, level="dataset")
        raw = []
        temporary = []
        for left, right in category_pairs:
            x = subset[left].to_numpy()
            y = subset[right].to_numpy()
            test = stats.wilcoxon(x, y, alternative="two-sided")
            raw.append(float(test.pvalue))
            temporary.append(
                {
                    "dataset": dataset,
                    "left": left,
                    "right": right,
                    "n_blocks": len(subset),
                    "mean_difference_left_minus_right": float((x - y).mean()),
                    "paired_cohens_d": paired_cohens_d(x, y),
                    "wilcoxon_statistic": float(test.statistic),
                    "wilcoxon_p_raw": float(test.pvalue),
                }
            )
        adjusted = holm_adjust(raw)
        for index, record in enumerate(temporary):
            record["wilcoxon_p_holm_within_dataset"] = float(adjusted[index])
            record["significant_holm_0_05"] = bool(adjusted[index] < 0.05)
            output.append(record)
    return output


def ordered_win_rates(clean_frame):
    wide = pivot(clean_frame)
    output = []
    for opponent in sorted(set(METHODS) - {"ordered"}):
        for dataset in sorted(clean_frame["dataset"].unique()):
            subset = wide.xs(dataset, level="dataset")
            ordered_wins = int((subset["ordered"] < subset[opponent]).sum())
            opponent_wins = int((subset[opponent] < subset["ordered"]).sum())
            ties = int(len(subset) - ordered_wins - opponent_wins)
            contested = ordered_wins + opponent_wins
            test = stats.binomtest(ordered_wins, contested, p=0.5, alternative="two-sided")
            output.append(
                {
                    "dataset": dataset,
                    "opponent": opponent,
                    "ordered_wins": ordered_wins,
                    "opponent_wins": opponent_wins,
                    "ties": ties,
                    "ordered_win_rate_excluding_ties": ordered_wins / contested,
                    "binomial_p_two_sided": float(test.pvalue),
                }
            )
    return output


def win_rate_homogeneity(clean_frame):
    wide = pivot(clean_frame)
    pairs = list(combinations(METHODS, 2))
    datasets = sorted(clean_frame["dataset"].unique())
    prediction_lengths = sorted(clean_frame["pred_len"].unique())

    def one_resolution(conditions, label):
        records = []
        raw_p = []
        for left, right in pairs:
            rows = []
            for condition in conditions:
                if label == "dataset":
                    subset = wide.xs(condition, level="dataset")
                else:
                    dataset, prediction_length = condition
                    subset = wide.xs((dataset, prediction_length), level=("dataset", "pred_len"))
                rows.append(
                    [
                        int((subset[left] < subset[right]).sum()),
                        int((subset[right] < subset[left]).sum()),
                    ]
                )
            contingency = np.asarray(rows)
            result = stats.chi2_contingency(contingency, correction=False)
            raw_p.append(float(result.pvalue))
            records.append(
                {
                    "left": left,
                    "right": right,
                    "chi_square": float(result.statistic),
                    "degrees_of_freedom": int(result.dof),
                    "p_raw": float(result.pvalue),
                    "win_counts": contingency.tolist(),
                }
            )
        adjusted = stats.false_discovery_control(np.asarray(raw_p), method="bh")
        for index, record in enumerate(records):
            record["p_bh"] = float(adjusted[index])
            record["significant_bh_0_05"] = bool(adjusted[index] < 0.05)
        return records

    return {
        "by_dataset": one_resolution(datasets, "dataset"),
        "by_dataset_prediction_length": one_resolution(
            [(dataset, length) for dataset in datasets for length in prediction_lengths],
            "dataset_prediction_length",
        ),
    }


def threshold_sensitivity(frame):
    output = []
    for threshold in THRESHOLDS:
        retained, removed_blocks = clean_blocks(frame, threshold)
        method_means = retained.groupby("method")["MSE"].mean()
        non_cohesion = ["ordered", "random", *DIVERSITY]
        non_cohesion_means = method_means[non_cohesion]
        output.append(
            {
                "threshold": threshold,
                "removed_blocks": removed_blocks,
                "retained_blocks": retained.groupby(BLOCK_COLUMNS).ngroups,
                "retained_rows": len(retained),
                "method_mean_mse": {method: float(method_means[method]) for method in METHODS},
                "non_cohesion_mean_mse": float(non_cohesion_means.mean()),
                "non_cohesion_spread_mse": float(non_cohesion_means.max() - non_cohesion_means.min()),
                "cohesion_mean_mse": float(method_means[COHESION].mean()),
            }
        )
    return output


def assert_manuscript_invariants(result):
    if result["input"]["raw_rows"] != 2880 or result["input"]["raw_blocks"] != 360:
        raise ValueError("Raw archive shape no longer matches the manuscript")
    if result["input"]["anomaly_rows_above_0_6"] != 25:
        raise ValueError("Anomaly-row count no longer matches the manuscript")
    if result["input"]["removed_blocks_at_0_6"] != 14:
        raise ValueError("Removed-block count no longer matches the manuscript")
    if result["input"]["clean_rows"] != 2768 or result["input"]["clean_blocks"] != 346:
        raise ValueError("Clean archive shape no longer matches the manuscript")
    if not np.isclose(result["raw"]["mean_mse"], 0.2831003110545377, rtol=0, atol=1e-15):
        raise ValueError("Raw mean MSE changed")
    if not np.isclose(result["clean"]["mean_mse"], 0.2718641095057216, rtol=0, atol=1e-15):
        raise ValueError("Clean mean MSE changed")
    raw_equivalent = sum(row["decision"] == "EQUIVALENT" for row in result["raw"]["rope_pairwise"])
    clean_equivalent = sum(row["decision"] == "EQUIVALENT" for row in result["clean"]["rope_pairwise"])
    if (raw_equivalent, clean_equivalent) != (3, 28):
        raise ValueError("Correlated-t ROPE decision counts changed")
    friedman = result["clean"]["friedman"]
    if not np.isclose(friedman["chi_square"], 235.5375722543, rtol=0, atol=1e-9):
        raise ValueError("Clean Friedman chi-square changed")
    if not np.isclose(friedman["p_value"], 3.3029615288e-47, rtol=1e-9, atol=0):
        raise ValueError("Clean Friedman p-value changed")
    significant_pairs = sum(row["significant_holm_0_05"] for row in friedman["wilcoxon_holm"])
    if significant_pairs != 16:
        raise ValueError("Clean Wilcoxon-Holm significant-pair count changed")
    homogeneity = result["clean"]["win_rate_homogeneity"]
    significant_dataset = sum(row["significant_bh_0_05"] for row in homogeneity["by_dataset"])
    significant_dataset_length = sum(
        row["significant_bh_0_05"] for row in homogeneity["by_dataset_prediction_length"]
    )
    if (significant_dataset, significant_dataset_length) != (26, 25):
        raise ValueError("Win-rate homogeneity counts changed")
    win_frame = pd.DataFrame(result["clean"]["ordered_win_rates"])
    averages = win_frame.groupby("dataset")["ordered_win_rate_excluding_ties"].mean()
    expected = {"electricity": 0.8428571428571429, "solar_AL": 0.8166666666666667, "traffic": 0.5040431266846361}
    for dataset, value in expected.items():
        if not np.isclose(averages[dataset], value, rtol=0, atol=1e-12):
            raise ValueError(f"Ordered mean win rate changed for {dataset}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, help="Optional JSON output path; stdout if omitted.")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    validate_input(frame)
    clean, removed_blocks = clean_blocks(frame, 0.6)
    raw_rope, raw_pairwise = pairwise_rope(frame)
    clean_rope, clean_pairwise = pairwise_rope(clean)
    result = {
        "methodology": {
            "bayesian_rope": "baycomp 1.0.3 two_on_single (Bayesian correlated t-test)",
            "rope_fraction_of_analysis_mean_mse": 0.01,
            "clean_block_rule": "remove a complete dataset/pred_len/seed block if max MSE > 0.6",
            "software": {
                package: importlib.metadata.version(package)
                for package in ("numpy", "pandas", "scipy", "baycomp")
            },
        },
        "input": {
            "path": args.input.name,
            "raw_rows": len(frame),
            "raw_blocks": frame.groupby(BLOCK_COLUMNS).ngroups,
            "anomaly_rows_above_0_6": int((frame["MSE"] > 0.6).sum()),
            "removed_blocks_at_0_6": removed_blocks,
            "clean_rows": len(clean),
            "clean_blocks": clean.groupby(BLOCK_COLUMNS).ngroups,
        },
        "raw": {
            "mean_mse": float(frame["MSE"].mean()),
            "rope_width": raw_rope,
            "rope_pairwise": raw_pairwise,
            "friedman": friedman_wilcoxon(frame),
        },
        "clean": {
            "mean_mse": float(clean["MSE"].mean()),
            "rope_width": clean_rope,
            "rope_pairwise": clean_pairwise,
            "friedman": friedman_wilcoxon(clean),
            "category_dataset_comparisons": category_dataset_comparisons(clean),
            "ordered_win_rates": ordered_win_rates(clean),
            "win_rate_homogeneity": win_rate_homogeneity(clean),
        },
        "threshold_sensitivity": threshold_sensitivity(frame),
    }
    assert_manuscript_invariants(result)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
