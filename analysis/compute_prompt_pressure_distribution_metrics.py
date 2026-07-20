import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


METRICS = [
    ("prompt_pressure_P_t", "P"),
    ("relative_pressure_R_t", "R"),
    ("guidance_delta_l2", "Delta"),
]


def load_traces(root: Path):
    traces = []
    for path in sorted(root.glob("*/*/seed_*/prompt_pressure_trace.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        traces.append(
            {
                "label": path.parts[-4],
                "sample_id": int(data["sample_id"]),
                "mapping_key": data["mapping_key"],
                "prompt": data["prompt"],
                "seed": int(data["seed"]),
                "records": data["records"],
            }
        )
    if not traces:
        raise FileNotFoundError(f"No prompt_pressure_trace.json files found under {root}")
    return traces


def normalize_sum(values):
    total = sum(values)
    if total <= 0:
        return [0.0 for _ in values]
    return [v / total for v in values]


def entropy(probabilities, normalize=True):
    h = -sum(p * math.log(p) for p in probabilities if p > 0)
    if normalize and len(probabilities) > 1:
        return h / math.log(len(probabilities))
    return h


def gini(values):
    values = sorted(v for v in values if v >= 0)
    n = len(values)
    if n == 0:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0
    weighted_sum = sum((i + 1) * v for i, v in enumerate(values))
    return (2 * weighted_sum) / (n * total) - (n + 1) / n


def curve_metrics(values):
    norm_values = normalize_sum(values)
    peak_step = max(range(len(values)), key=lambda i: values[i])
    norm_peak_step = max(range(len(norm_values)), key=lambda i: norm_values[i])
    early = sum(norm_values[:10])
    mid = sum(norm_values[10:30])
    late = sum(norm_values[30:])
    return {
        "raw_sum": sum(values),
        "raw_mean": sum(values) / len(values),
        "raw_peak": values[peak_step],
        "raw_peak_step": peak_step,
        "raw_peak_timestep": None,
        "norm_peak": norm_values[norm_peak_step],
        "norm_peak_step": norm_peak_step,
        "entropy": entropy(norm_values, normalize=True),
        "entropy_raw_nats": entropy(norm_values, normalize=False),
        "gini": gini(norm_values),
        "norm_early_0_9": early,
        "norm_mid_10_29": mid,
        "norm_late_30_49": late,
    }


def mean(values):
    return sum(values) / len(values)


def stdev(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, default="outputs/prompt_pressure_top10")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/prompt_pressure_top10_analysis/distribution_metrics",
    )
    args = parser.parse_args()

    traces = load_traces(Path(args.input_root))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows = []
    for trace in traces:
        row = {
            "label": trace["label"],
            "sample_id": trace["sample_id"],
            "mapping_key": trace["mapping_key"],
            "prompt": trace["prompt"],
            "seed": trace["seed"],
            "num_steps": len(trace["records"]),
        }
        for source_key, short_name in METRICS:
            values = [r[source_key] for r in trace["records"]]
            metrics = curve_metrics(values)
            peak_step = metrics["raw_peak_step"]
            metrics["raw_peak_timestep"] = trace["records"][peak_step]["timestep"]
            for metric_name, metric_value in metrics.items():
                row[f"{short_name}_{metric_name}"] = metric_value
        per_seed_rows.append(row)

    write_csv(output_dir / "per_seed_distribution_metrics.csv", per_seed_rows)

    sample_groups = defaultdict(list)
    label_groups = defaultdict(list)
    for row in per_seed_rows:
        sample_groups[(row["label"], row["sample_id"], row["prompt"])].append(row)
        label_groups[row["label"]].append(row)

    metric_cols = [
        col
        for col in per_seed_rows[0]
        if col.startswith("P_") or col.startswith("R_") or col.startswith("Delta_")
    ]

    per_sample_rows = []
    for (label, sample_id, prompt), rows in sorted(sample_groups.items()):
        out = {
            "label": label,
            "sample_id": sample_id,
            "prompt": prompt,
            "n": len(rows),
        }
        for col in metric_cols:
            values = [float(r[col]) for r in rows]
            out[f"{col}_mean"] = mean(values)
            out[f"{col}_std"] = stdev(values)
        per_sample_rows.append(out)
    write_csv(output_dir / "per_sample_distribution_metrics.csv", per_sample_rows)

    per_label_rows = []
    for label, rows in sorted(label_groups.items()):
        out = {"label": label, "n": len(rows)}
        for col in metric_cols:
            values = [float(r[col]) for r in rows]
            out[f"{col}_mean"] = mean(values)
            out[f"{col}_std"] = stdev(values)
            out[f"{col}_min"] = min(values)
            out[f"{col}_max"] = max(values)
        per_label_rows.append(out)
    write_csv(output_dir / "per_label_distribution_metrics.csv", per_label_rows)

    with (output_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# Prompt Pressure Distribution Metrics\n\n")
        f.write("Metrics are computed for `P_t` and `R_t` over each 50-step trace.\n\n")
        f.write("- `raw_peak`: max value on the original curve.\n")
        f.write("- `norm_peak`: max value after sum-normalization, i.e. `x_t / sum_t x_t`.\n")
        f.write("- `entropy`: normalized Shannon entropy of the sum-normalized curve, in `[0,1]`; lower means more concentrated.\n")
        f.write("- `gini`: Gini coefficient of the curve values; higher means more concentrated/uneven.\n\n")
        f.write("## Label-Level Means\n\n")
        f.write("| label | n | Delta raw mean | Delta raw peak | Delta norm peak | Delta entropy | Delta gini | P raw mean | P raw peak | P norm peak | P entropy | P gini | R raw mean | R raw peak | R norm peak | R entropy | R gini |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in per_label_rows:
            f.write(
                f"| {row['label']} | {int(row['n'])} | "
                f"{row['Delta_raw_mean_mean']:.4f} | {row['Delta_raw_peak_mean']:.4f} | "
                f"{row['Delta_norm_peak_mean']:.4f} | {row['Delta_entropy_mean']:.4f} | "
                f"{row['Delta_gini_mean']:.4f} | "
                f"{row['P_raw_mean_mean']:.4f} | {row['P_raw_peak_mean']:.4f} | {row['P_norm_peak_mean']:.4f} | "
                f"{row['P_entropy_mean']:.4f} | {row['P_gini_mean']:.4f} | "
                f"{row['R_raw_mean_mean']:.4f} | {row['R_raw_peak_mean']:.4f} | {row['R_norm_peak_mean']:.4f} | "
                f"{row['R_entropy_mean']:.4f} | {row['R_gini_mean']:.4f} |\n"
            )

    print(f"saved distribution metrics to {output_dir}")


if __name__ == "__main__":
    main()
