#!/usr/bin/env python3
"""
Plot per-sample distribution for diagonal AIDI settings:
  aidi_gs1/gs1, aidi_gs3/gs3, aidi_gs5/gs5, aidi_gs7/gs7

X-axis: Init↔Inv -log(MSE) per sample
Y-axis: Gen↔Rec -log(MSE) per sample
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "reconstruction")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "distribution")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "aidi_gs_diagonal_distribution.png")

COLORS = {
    1: "#3498DB",
    3: "#2ECC71",
    5: "#F39C12",
    7: "#9B59B6",
}


def compute_per_sample_metrics(base_dir, rec_dir):
    init_latents = torch.load(os.path.join(base_dir, "init_latents.pt"), map_location="cpu")
    inv_latents = torch.load(os.path.join(base_dir, "inv_latents.pt"), map_location="cpu")
    gen_latents = torch.load(os.path.join(base_dir, "gen_latents.pt"), map_location="cpu")
    rec_latents = torch.load(os.path.join(rec_dir, "rec_latents.pt"), map_location="cpu")

    n = min(len(init_latents), len(inv_latents), len(gen_latents), len(rec_latents))
    init_inv_nlm = []
    gen_rec_nlm = []
    for i in range(n):
        init_inv_mse = F.mse_loss(init_latents[i], inv_latents[i]).item()
        gen_rec_mse = F.mse_loss(gen_latents[i], rec_latents[i]).item()
        init_inv_nlm.append(-np.log(init_inv_mse))
        gen_rec_nlm.append(-np.log(gen_rec_mse))
    return np.array(init_inv_nlm), np.array(gen_rec_nlm)


def main():
    print("=" * 60)
    print("AIDI Diagonal Per-Sample Distribution")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 8), dpi=150)

    plotted = 0
    for gs in [1, 3, 5, 7]:
        base_dir = os.path.join(OUTPUTS_DIR, f"aidi_gs{gs}")
        rec_dir = os.path.join(base_dir, f"gs{gs}")
        if not os.path.isdir(base_dir) or not os.path.isdir(rec_dir):
            print(f"Warning: missing directory for aidi_gs{gs}/gs{gs}")
            continue
        if not os.path.isfile(os.path.join(rec_dir, "rec_latents.pt")):
            print(f"Warning: missing rec_latents.pt for aidi_gs{gs}/gs{gs}")
            continue

        init_inv, gen_rec = compute_per_sample_metrics(base_dir, rec_dir)
        color = COLORS.get(gs, "#666666")
        ax.scatter(
            init_inv,
            gen_rec,
            s=14,
            alpha=0.32,
            c=color,
            edgecolors="none",
            label=f"AIDI-GS{gs} (n={len(init_inv)})",
        )
        ax.scatter(
            float(np.mean(init_inv)),
            float(np.mean(gen_rec)),
            s=180,
            c=color,
            marker="*",
            edgecolors="black",
            linewidth=1.2,
            zorder=5,
        )

        print(
            f"AIDI-GS{gs}: Init↔Inv={np.mean(init_inv):.4f}±{np.std(init_inv):.4f}, "
            f"Gen↔Rec={np.mean(gen_rec):.4f}±{np.std(gen_rec):.4f}, n={len(init_inv)}"
        )
        plotted += 1

    if plotted == 0:
        raise RuntimeError("No valid diagonal AIDI results found.")

    ax.set_xlabel("Init↔Inv -log(MSE)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Gen↔Rec -log(MSE)", fontsize=13, fontweight="bold")
    ax.set_title("AIDI Diagonal Settings Distribution (GS1/3/5/7)", fontsize=14, fontweight="bold", pad=14)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=9, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"✓ Figure saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
