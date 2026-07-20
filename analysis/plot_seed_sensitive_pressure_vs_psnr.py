import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


def read_csv_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_labels(label_text: str):
    if label_text == "all":
        return None
    return {item.strip() for item in label_text.split(",") if item.strip()}


def merge_pressure_psnr(metrics_rows, psnr_rows, labels):
    psnr_by_key = {
        (int(row["sample_id"]), int(row["seed"])): row
        for row in psnr_rows
    }

    merged = []
    for row in metrics_rows:
        if labels is not None and row["label"] not in labels:
            continue
        sample_id = int(row["sample_id"])
        seed = int(row["seed"])
        psnr_row = psnr_by_key.get((sample_id, seed))
        if psnr_row is None:
            continue
        merged.append(
            {
                "label": row["label"],
                "sample_id": sample_id,
                "seed": seed,
                "mapping_key": row["mapping_key"],
                "prompt": row["prompt"],
                "P_t_sum": float(row["P_raw_sum"]),
                "R_t_sum": float(row["R_raw_sum"]),
                "Delta_sum": float(row["Delta_raw_sum"]),
                "PSNR": float(psnr_row["psnr"]),
                "MSE": float(psnr_row["mse"]),
                "gen_path": psnr_row["gen_path"],
                "rec_path": psnr_row["rec_path"],
            }
        )
    return sorted(merged, key=lambda r: (r["sample_id"], r["seed"]))


def label_color(label: str):
    if label == "best":
        return "tab:green"
    if label == "worst":
        return "tab:red"
    if label == "seed_sensitive":
        return "tab:purple"
    return "tab:blue"


def plot_scatter(rows, x_key: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = sorted({row["label"] for row in rows})
    label_order = [label for label in ["best", "worst", "seed_sensitive"] if label in labels]
    label_order.extend(label for label in labels if label not in label_order)

    for label in label_order:
        sample_rows = [row for row in rows if row["label"] == label]
        xs = [row[x_key] for row in sample_rows]
        ys = [row["PSNR"] for row in sample_rows]
        ax.scatter(
            xs,
            ys,
            s=42,
            alpha=0.82,
            color=label_color(label),
            label=label,
            edgecolors="white",
            linewidths=0.5,
        )

    ax.set_xlabel(x_key)
    ax.set_ylabel("PSNR")
    ax.grid(alpha=0.25)
    ax.legend(
        title="label",
        fontsize=8,
        title_fontsize=9,
        frameon=False,
        loc="best",
    )
    fig.suptitle(f"{x_key} vs PSNR", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics_csv",
        nargs="+",
        default=[
            "results/prompt_pressure_seed_sensitive_top10_analysis/distribution_metrics/per_seed_distribution_metrics.csv"
        ],
        help="One or more per_seed_distribution_metrics.csv files.",
    )
    parser.add_argument(
        "--psnr_csv",
        type=str,
        default="results/aidi_gs7_seed_psnr/aidi_gs7_seed_psnr_detail.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/prompt_pressure_seed_sensitive_top10_analysis/pressure_vs_psnr",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="seed_sensitive",
        help="Comma-separated labels to include, or all.",
    )
    args = parser.parse_args()

    metric_rows = []
    for metrics_csv in args.metrics_csv:
        metric_rows.extend(read_csv_rows(Path(metrics_csv)))

    rows = merge_pressure_psnr(
        metric_rows,
        read_csv_rows(Path(args.psnr_csv)),
        parse_labels(args.label),
    )
    if not rows:
        raise FileNotFoundError("No joined pressure/PSNR rows found.")

    output_dir = Path(args.output_dir)
    output_csv = output_dir / "pressure_vs_psnr_points.csv"
    write_csv(output_csv, rows)
    plot_scatter(rows, "P_t_sum", output_dir / "P_t_sum_vs_PSNR.png")
    plot_scatter(rows, "R_t_sum", output_dir / "R_t_sum_vs_PSNR.png")

    print(f"saved {len(rows)} joined rows to {output_csv}")
    print(f"saved plot: {output_dir / 'P_t_sum_vs_PSNR.png'}")
    print(f"saved plot: {output_dir / 'R_t_sum_vs_PSNR.png'}")


if __name__ == "__main__":
    main()
