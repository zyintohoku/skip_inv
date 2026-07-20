import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


LABEL_ORDER = ["best", "worst", "seed_sensitive"]
DISPLAY_LABEL = {
    "best": "best",
    "worst": "worst",
    "seed_sensitive": "most_sensitive",
}


def read_csv_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return sum(values) / len(values)


def stdev(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def fmt(value, digits=4):
    if value == "":
        return ""
    return f"{float(value):.{digits}f}"


def mean_std_text(mean_value, std_value):
    return f"{fmt(mean_value)} +- {fmt(std_value)}"


def psnr_stats(joined_rows):
    grouped = defaultdict(list)
    for row in joined_rows:
        grouped[(row["label"], int(row["sample_id"]))].append(float(row["PSNR"]))

    by_sample = {
        key: {"mean": mean(values), "std": stdev(values), "n": len(values)}
        for key, values in grouped.items()
    }

    label_grouped = defaultdict(list)
    for row in joined_rows:
        label_grouped[row["label"]].append(float(row["PSNR"]))
    by_label = {
        label: {"mean": mean(values), "std": stdev(values), "n": len(values)}
        for label, values in label_grouped.items()
    }
    return by_sample, by_label


def make_prompt_rows(per_sample_rows, psnr_by_sample):
    rows = []
    sorted_rows = sorted(
        per_sample_rows,
        key=lambda row: (
            LABEL_ORDER.index(row["label"]) if row["label"] in LABEL_ORDER else 99,
            int(row["sample_id"]),
        ),
    )
    for row in sorted_rows:
        key = (row["label"], int(row["sample_id"]))
        psnr = psnr_by_sample[key]
        rows.append(
            {
                "scope": "prompt",
                "label": DISPLAY_LABEL.get(row["label"], row["label"]),
                "sample_id": row["sample_id"],
                "prompt": row["prompt"],
                "n": row["n"],
                "P_t_sum": mean_std_text(row["P_raw_sum_mean"], row["P_raw_sum_std"]),
                "P_t_sum_mean": fmt(row["P_raw_sum_mean"]),
                "P_t_sum_std": fmt(row["P_raw_sum_std"]),
                "P_t_entropy": fmt(row["P_entropy_mean"]),
                "P_t_gini": fmt(row["P_gini_mean"]),
                "R_t_sum": mean_std_text(row["R_raw_sum_mean"], row["R_raw_sum_std"]),
                "R_t_sum_mean": fmt(row["R_raw_sum_mean"]),
                "R_t_sum_std": fmt(row["R_raw_sum_std"]),
                "R_t_entropy": fmt(row["R_entropy_mean"]),
                "R_t_gini": fmt(row["R_gini_mean"]),
                "PSNR": mean_std_text(psnr["mean"], psnr["std"]),
                "PSNR_mean": fmt(psnr["mean"]),
                "PSNR_std": fmt(psnr["std"]),
            }
        )
    return rows


def make_label_rows(per_label_rows, psnr_by_label):
    rows = []
    sorted_rows = sorted(
        per_label_rows,
        key=lambda row: LABEL_ORDER.index(row["label"]) if row["label"] in LABEL_ORDER else 99,
    )
    for row in sorted_rows:
        psnr = psnr_by_label[row["label"]]
        rows.append(
            {
                "scope": "label",
                "label": DISPLAY_LABEL.get(row["label"], row["label"]),
                "sample_id": "",
                "prompt": f"{DISPLAY_LABEL.get(row['label'], row['label'])} label aggregate",
                "n": row["n"],
                "P_t_sum": mean_std_text(row["P_raw_sum_mean"], row["P_raw_sum_std"]),
                "P_t_sum_mean": fmt(row["P_raw_sum_mean"]),
                "P_t_sum_std": fmt(row["P_raw_sum_std"]),
                "P_t_entropy": fmt(row["P_entropy_mean"]),
                "P_t_gini": fmt(row["P_gini_mean"]),
                "R_t_sum": mean_std_text(row["R_raw_sum_mean"], row["R_raw_sum_std"]),
                "R_t_sum_mean": fmt(row["R_raw_sum_mean"]),
                "R_t_sum_std": fmt(row["R_raw_sum_std"]),
                "R_t_entropy": fmt(row["R_entropy_mean"]),
                "R_t_gini": fmt(row["R_gini_mean"]),
                "PSNR": mean_std_text(psnr["mean"], psnr["std"]),
                "PSNR_mean": fmt(psnr["mean"]),
                "PSNR_std": fmt(psnr["std"]),
            }
        )
    return rows


def write_markdown(path: Path, prompt_rows, label_rows):
    columns = [
        "label",
        "sample_id",
        "prompt",
        "n",
        "P_t_sum",
        "P_t_entropy",
        "P_t_gini",
        "R_t_sum",
        "R_t_entropy",
        "R_t_gini",
        "PSNR",
    ]

    def table(rows):
        lines = []
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            values = [str(row[col]).replace("|", "\\|") for col in columns]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Saved-Latent Prompt-Pressure Summary\n\n")
        f.write(
            "Rows summarize 10 saved seed latents per prompt. Label rows summarize all 100 seed-level points per label.\n\n"
        )
        f.write("## Prompt Rows\n\n")
        f.write(table(prompt_rows))
        f.write("\n\n## Label Rows\n\n")
        f.write(table(label_rows))
        f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--merged_root",
        type=str,
        default="results/prompt_pressure_saved_latent_top10_merged_analysis",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/prompt_pressure_saved_latent_top10_merged_analysis/summary_tables",
    )
    args = parser.parse_args()

    merged_root = Path(args.merged_root)
    metrics_dir = merged_root / "distribution_metrics"
    pressure_csv = merged_root / "pressure_vs_psnr" / "pressure_vs_psnr_points.csv"

    per_sample_rows = read_csv_rows(metrics_dir / "per_sample_distribution_metrics.csv")
    per_label_rows = read_csv_rows(metrics_dir / "per_label_distribution_metrics.csv")
    joined_rows = read_csv_rows(pressure_csv)
    psnr_by_sample, psnr_by_label = psnr_stats(joined_rows)

    prompt_rows = make_prompt_rows(per_sample_rows, psnr_by_sample)
    label_rows = make_label_rows(per_label_rows, psnr_by_label)
    all_rows = prompt_rows + label_rows

    fieldnames = [
        "scope",
        "label",
        "sample_id",
        "prompt",
        "n",
        "P_t_sum",
        "P_t_sum_mean",
        "P_t_sum_std",
        "P_t_entropy",
        "P_t_gini",
        "R_t_sum",
        "R_t_sum_mean",
        "R_t_sum_std",
        "R_t_entropy",
        "R_t_gini",
        "PSNR",
        "PSNR_mean",
        "PSNR_std",
    ]
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "prompt_label_pressure_psnr_summary.csv", all_rows, fieldnames)
    write_markdown(output_dir / "prompt_label_pressure_psnr_summary.md", prompt_rows, label_rows)

    print(f"saved CSV: {output_dir / 'prompt_label_pressure_psnr_summary.csv'}")
    print(f"saved Markdown: {output_dir / 'prompt_label_pressure_psnr_summary.md'}")
    print(f"rows: {len(prompt_rows)} prompt + {len(label_rows)} label")


if __name__ == "__main__":
    main()
