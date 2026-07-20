import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


STAGES = [
    ("early_0_9", "0-9"),
    ("mid_10_29", "10-29"),
    ("late_30_49", "30-49"),
]

METRICS = [
    ("P", r"$P_t$"),
    ("R", r"$R_t$"),
]

LABEL_ORDER = ["best", "worst", "seed_sensitive"]
LABEL_DISPLAY = {
    "best": "best",
    "worst": "worst",
    "seed_sensitive": "sensitive",
}
LABEL_COLORS = {
    "best": "tab:green",
    "worst": "tab:red",
    "seed_sensitive": "tab:purple",
}


def read_rows(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["sample_id"] = int(row["sample_id"])
            row["seed"] = int(row["seed"])
            for prefix, _ in METRICS:
                row[f"{prefix}_raw_sum"] = float(row[f"{prefix}_raw_sum"])
                for stage_key, _ in STAGES:
                    row[f"{prefix}_norm_{stage_key}"] = float(row[f"{prefix}_norm_{stage_key}"])
                    row[f"{prefix}_{stage_key}_raw_sum"] = (
                        row[f"{prefix}_raw_sum"] * row[f"{prefix}_norm_{stage_key}"]
                    )
            rows.append(row)
    return rows


def mean(values):
    return sum(values) / len(values) if values else 0.0


def stdev(values):
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return (sum((v - mu) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)

    summary = []
    for label in LABEL_ORDER:
        label_rows = grouped.get(label, [])
        if not label_rows:
            continue
        for prefix, metric_label in METRICS:
            for stage_key, stage_label in STAGES:
                values = [row[f"{prefix}_{stage_key}_raw_sum"] for row in label_rows]
                summary.append(
                    {
                        "label": label,
                        "label_display": LABEL_DISPLAY.get(label, label),
                        "metric": prefix,
                        "metric_display": metric_label,
                        "stage": stage_key,
                        "stage_display": stage_label,
                        "n": len(values),
                        "mean": mean(values),
                        "std": stdev(values),
                    }
                )
    return summary


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def get_value(summary, label, metric, stage, key):
    for row in summary:
        if row["label"] == label and row["metric"] == metric and row["stage"] == stage:
            return row[key]
    return 0.0


def plot_summary(summary, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=False)
    x = np.arange(len(STAGES))
    width = 0.24
    offsets = np.linspace(-width, width, len(LABEL_ORDER))

    for ax, (metric, metric_label) in zip(axes, METRICS):
        for offset, label in zip(offsets, LABEL_ORDER):
            means = [get_value(summary, label, metric, stage_key, "mean") for stage_key, _ in STAGES]
            stds = [get_value(summary, label, metric, stage_key, "std") for stage_key, _ in STAGES]
            ax.bar(
                x + offset,
                means,
                width=width,
                yerr=stds,
                capsize=3,
                label=LABEL_DISPLAY.get(label, label),
                color=LABEL_COLORS.get(label),
                alpha=0.85,
                linewidth=0.5,
                edgecolor="black",
            )
        ax.set_title(f"{metric_label} stage raw-sum mean")
        ax.set_xticks(x)
        ax.set_xticklabels([stage_label for _, stage_label in STAGES])
        ax.set_xlabel("DDIM step index range")
        ax.set_ylabel("Raw sum")
        ax.grid(axis="y", alpha=0.25)

    axes[1].legend(title="label", loc="best")
    fig.suptitle(r"Stage-wise $P_t$ and $R_t$ raw sums by label", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        default="results/prompt_pressure_saved_latent_top10_merged_analysis/distribution_metrics/per_seed_distribution_metrics.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/prompt_pressure_saved_latent_top10_merged_analysis/stage_pressure_by_label",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rows = read_rows(Path(args.input_csv))
    summary = summarize(rows)
    write_csv(output_dir / "pressure_stage_raw_sums_by_label.csv", summary)
    plot_summary(summary, output_dir / "pressure_stage_raw_sums_by_label.png")
    print(f"saved CSV: {output_dir / 'pressure_stage_raw_sums_by_label.csv'}")
    print(f"saved plot: {output_dir / 'pressure_stage_raw_sums_by_label.png'}")


if __name__ == "__main__":
    main()
