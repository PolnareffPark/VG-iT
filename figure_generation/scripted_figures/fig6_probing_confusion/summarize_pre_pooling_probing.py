#!/usr/bin/env python3
"""Validate and summarize the archived pre/post-pooling probing results.

The condition flag is a descriptive threshold, accuracy > chance + 0.03. It is
not a statistical significance test.
"""

import argparse
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    REPOSITORY_ROOT
    / "results_canonical"
    / "representation_diagnostics"
    / "pre_pooling_probing"
    / "pre_pooling_probing_results.json"
)
EXPECTED_CONDITIONS = {
    f"{dataset}_pl{prediction_length}"
    for dataset in ("electricity", "solar_AL", "traffic")
    for prediction_length in (96, 192, 336, 720)
}
DESCRIPTIVE_MARGIN = 0.03


def matrix_counts(matrix):
    if len(matrix) != 8 or any(len(row) != 8 for row in matrix):
        raise ValueError("Each confusion matrix must be 8 by 8")
    if any(not isinstance(value, int) or value < 0 for row in matrix for value in row):
        raise ValueError("Confusion-matrix entries must be nonnegative integers")
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[index][index] for index in range(8))
    return correct, total


def validate_level(condition, level_name, level):
    correct, total = matrix_counts(level["confusion_matrix"])
    accuracy = correct / total
    if total != level["n_samples"]:
        raise ValueError(f"{condition}/{level_name}: n_samples does not match the matrix")
    if abs(accuracy - level["accuracy"]) > 1e-12:
        raise ValueError(f"{condition}/{level_name}: accuracy does not match the matrix")
    if abs(level["chance"] - 0.125) > 1e-12:
        raise ValueError(f"{condition}/{level_name}: unexpected chance level")
    return correct, total, accuracy


def build_summary(data):
    if set(data) != EXPECTED_CONDITIONS:
        missing = sorted(EXPECTED_CONDITIONS - set(data))
        extra = sorted(set(data) - EXPECTED_CONDITIONS)
        raise ValueError(f"Unexpected condition inventory; missing={missing}, extra={extra}")

    rows = []
    pooled = {"pre_pooling": [0, 0], "post_pooling": [0, 0]}
    for condition in sorted(data):
        levels = data[condition]
        validated = {}
        for level_name in ("pre_pooling", "post_pooling"):
            correct, total, accuracy = validate_level(condition, level_name, levels[level_name])
            pooled[level_name][0] += correct
            pooled[level_name][1] += total
            validated[level_name] = (correct, total, accuracy)
        pre = validated["pre_pooling"]
        post = validated["post_pooling"]
        rows.append(
            {
                "condition": condition,
                "pre_correct": pre[0],
                "pre_total": pre[1],
                "pre_accuracy": pre[2],
                "post_correct": post[0],
                "post_total": post[1],
                "post_accuracy": post[2],
                "exceeds_descriptive_threshold": pre[2] > 0.125 + DESCRIPTIVE_MARGIN,
            }
        )

    flagged = [row["condition"] for row in rows if row["exceeds_descriptive_threshold"]]
    if flagged != ["electricity_pl96"]:
        raise ValueError(f"Unexpected descriptive-threshold conditions: {flagged}")
    if pooled != {"pre_pooling": [356, 2768], "post_pooling": [344, 2768]}:
        raise ValueError(f"Unexpected pooled counts: {pooled}")
    return rows, pooled


def render_markdown(rows, pooled):
    lines = [
        "# Pre/Post-Pooling Probing Validation",
        "",
        "The condition flag uses the descriptive rule `accuracy > chance + 0.03` ",
        "with chance = 0.125. It is not a statistical significance test.",
        "",
        "| Condition | Pre accuracy | Post accuracy | Exceeds chance by >3 percentage points? |",
        "|---|---:|---:|:---:|",
    ]
    for row in rows:
        flag = "yes" if row["exceeds_descriptive_threshold"] else "no"
        lines.append(
            f"| {row['condition']} | {row['pre_accuracy']:.6f} "
            f"| {row['post_accuracy']:.6f} | {flag} |"
        )
    pre_correct, pre_total = pooled["pre_pooling"]
    post_correct, post_total = pooled["post_pooling"]
    lines.extend(
        [
            "",
            f"Pooled pre-pooling accuracy: {pre_correct}/{pre_total} = {pre_correct / pre_total:.6f}.",
            f"Pooled post-pooling accuracy: {post_correct}/{post_total} = {post_correct / post_total:.6f}.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, help="Optional generated Markdown output path.")
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as handle:
        data = json.load(handle)
    rows, pooled = build_summary(data)
    rendered = render_markdown(rows, pooled)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
