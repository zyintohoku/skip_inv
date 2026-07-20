#!/usr/bin/env python3
"""Plot prompt mean PSNR vs seed-sensitivity std for AIDI-GS7 seeds."""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "aidi_gs7_seed_psnr")
INPUT_CSV = os.path.join(RESULTS_DIR, "prompt_psnr_enriched_by_sample.csv")
OUTPUT_PNG = os.path.join(RESULTS_DIR, "prompt_psnr_mean_vs_std.png")
OUTPUT_PDF = os.path.join(RESULTS_DIR, "prompt_psnr_mean_vs_std.pdf")


def load_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["sample_id"] = int(row["sample_id"])
        row["psnr_mean"] = float(row["psnr_mean"])
        row["psnr_std"] = float(row["psnr_std"])
        row["psnr_min"] = float(row["psnr_min"])
        row["psnr_max"] = float(row["psnr_max"])
    return rows


def main():
    rows = load_rows(INPUT_CSV)
    means = np.array([row["psnr_mean"] for row in rows])
    stds = np.array([row["psnr_std"] for row in rows])
    median_mean = float(np.median(means))
    median_std = float(np.median(stds))

    worst_mean = sorted(rows, key=lambda row: row["psnr_mean"])[:8]
    highest_std = sorted(rows, key=lambda row: row["psnr_std"], reverse=True)[:8]
    label_rows = {row["sample_id"]: row for row in worst_mean + highest_std}

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 7))

    scatter = ax.scatter(
        means,
        stds,
        c=means,
        cmap="viridis",
        s=42,
        alpha=0.78,
        edgecolors="white",
        linewidths=0.45,
    )
    ax.axvline(median_mean, color="#555555", linestyle="--", linewidth=1.1, label=f"median mean = {median_mean:.2f}")
    ax.axhline(median_std, color="#777777", linestyle=":", linewidth=1.2, label=f"median std = {median_std:.2f}")

    for row in label_rows.values():
        ax.annotate(
            str(row["sample_id"]),
            xy=(row["psnr_mean"], row["psnr_std"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="#222222",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.82},
        )

    ax.set_title("Prompt Reconstruction Quality vs Seed Sensitivity", fontsize=15, pad=12)
    ax.set_xlabel("Mean PSNR across 10 seeds (dB)")
    ax.set_ylabel("PSNR std across 10 seeds (dB)")
    ax.legend(loc="upper right", frameon=True)
    ax.text(
        0.01,
        0.98,
        "Each point is one prompt / sample_id\nLabels mark worst mean PSNR and highest seed sensitivity",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9},
    )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Mean PSNR (dB)")

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=220)
    fig.savefig(OUTPUT_PDF)
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
