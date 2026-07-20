import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def stdev(values):
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return (sum((v - mu) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def top_sample_ids(path: Path, top_k: int):
    rows = read_csv(path)[:top_k]
    return [int(row["sample_id"]) for row in rows]


def build_rows(detail_csv: Path, top_ids):
    top_set = set(top_ids)
    rows = []
    for row in read_csv(detail_csv):
        sample_id = int(row["sample_id"])
        if sample_id not in top_set:
            continue
        rows.append(
            {
                "seed": int(row["seed"]),
                "sample_id": sample_id,
                "prompt": row["original_prompt"],
                "psnr": float(row["psnr"]),
            }
        )
    return rows


def seed_summary(rows, high_threshold: float, low_threshold: float):
    by_seed = {}
    for row in rows:
        by_seed.setdefault(row["seed"], []).append(row)

    summary = []
    for seed in sorted(by_seed):
        values = [row["psnr"] for row in by_seed[seed]]
        ranks = []
        for row in by_seed[seed]:
            sample_values = sorted(
                [r for r in rows if r["sample_id"] == row["sample_id"]],
                key=lambda item: item["psnr"],
                reverse=True,
            )
            rank = next(i + 1 for i, item in enumerate(sample_values) if item["seed"] == seed)
            ranks.append(rank)
        summary.append(
            {
                "seed": seed,
                "n": len(values),
                "psnr_mean": mean(values),
                "psnr_std": stdev(values),
                "psnr_min": min(values),
                "psnr_max": max(values),
                "num_high_psnr": sum(v >= high_threshold for v in values),
                "num_low_psnr": sum(v <= low_threshold for v in values),
                "mean_within_prompt_rank": mean(ranks),
                "num_prompt_best_seed": sum(rank == 1 for rank in ranks),
                "num_prompt_worst_seed": sum(rank == 10 for rank in ranks),
            }
        )
    return summary


def prompt_seed_matrix(rows, top_ids):
    seeds = sorted({row["seed"] for row in rows})
    row_by_key = {(row["sample_id"], row["seed"]): row for row in rows}
    matrix_rows = []
    for sample_id in top_ids:
        row = {"sample_id": sample_id}
        prompt = next(item["prompt"] for item in rows if item["sample_id"] == sample_id)
        row["prompt"] = prompt
        for seed in seeds:
            row[f"seed_{seed}"] = row_by_key[(sample_id, seed)]["psnr"]
        matrix_rows.append(row)
    return matrix_rows


def plot_seed_bar(summary, output_path: Path):
    seeds = [row["seed"] for row in summary]
    means = [row["psnr_mean"] for row in summary]
    stds = [row["psnr_std"] for row in summary]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(seeds, means, yerr=stds, capsize=3, color="tab:blue", alpha=0.82, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("seed")
    ax.set_ylabel("PSNR over sensitive top10 prompts")
    ax.set_title("FPI GS=7: seed-wise PSNR on most sensitive top10 prompts", fontsize=11)
    ax.set_xticks(seeds)
    ax.grid(axis="y", alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(matrix_rows, output_path: Path):
    seeds = [int(key.replace("seed_", "")) for key in matrix_rows[0] if key.startswith("seed_")]
    data = np.array([[float(row[f"seed_{seed}"]) for seed in seeds] for row in matrix_rows])
    sample_ids = [row["sample_id"] for row in matrix_rows]

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    image = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(seeds)))
    ax.set_xticklabels(seeds)
    ax.set_yticks(np.arange(len(sample_ids)))
    ax.set_yticklabels(sample_ids)
    ax.set_xlabel("seed")
    ax.set_ylabel("sample_id")
    ax.set_title("FPI GS=7: PSNR heatmap for most sensitive top10 prompts", fontsize=11)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=7, color="white" if data[i, j] < data.mean() else "black")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("PSNR")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detail_csv",
        type=str,
        default="results/fpi_gs7_seed_psnr/fpi_gs7_seed_psnr_detail.csv",
    )
    parser.add_argument(
        "--sensitive_csv",
        type=str,
        default="results/fpi_gs7_seed_psnr/prompt_psnr_most_seed_sensitive30.csv",
    )
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--high_threshold", type=float, default=50.0)
    parser.add_argument("--low_threshold", type=float, default=20.0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/fpi_gs7_seed_psnr/sensitive_top10_seed_effect",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    top_ids = top_sample_ids(Path(args.sensitive_csv), args.top_k)
    rows = build_rows(Path(args.detail_csv), top_ids)
    summary = seed_summary(rows, args.high_threshold, args.low_threshold)
    matrix = prompt_seed_matrix(rows, top_ids)

    write_csv(output_dir / "sensitive_top10_seed_psnr_long.csv", rows)
    write_csv(output_dir / "sensitive_top10_seed_summary.csv", summary)
    write_csv(output_dir / "sensitive_top10_prompt_seed_matrix.csv", matrix)
    plot_seed_bar(summary, output_dir / "sensitive_top10_seed_mean_psnr.png")
    plot_heatmap(matrix, output_dir / "sensitive_top10_prompt_seed_psnr_heatmap.png")

    best_seed = max(summary, key=lambda row: row["psnr_mean"])
    worst_seed = min(summary, key=lambda row: row["psnr_mean"])
    print(f"top ids: {top_ids}")
    print(f"best mean seed: {best_seed['seed']} mean={best_seed['psnr_mean']:.4f}")
    print(f"worst mean seed: {worst_seed['seed']} mean={worst_seed['psnr_mean']:.4f}")
    print(f"saved results to {output_dir}")


if __name__ == "__main__":
    main()
