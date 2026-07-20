#!/usr/bin/env python3
"""
Plot per-sample distribution of Gen↔Rec -log(MSE) for AIDI-GS1/3/5/7.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "reconstruction")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "distribution")

COLORS = {
    1: "#3498DB",
    3: "#2ECC71",
    5: "#F39C12",
    7: "#9B59B6",
}


def compute_gen_rec_nlm(result_dir):
    gen_latents = torch.load(os.path.join(result_dir, "gen_latents.pt"), map_location="cpu")
    rec_latents = torch.load(os.path.join(result_dir, "rec_latents.pt"), map_location="cpu")
    values = []
    for gen, rec in zip(gen_latents, rec_latents):
        gen_rec_mse = F.mse_loss(gen, rec).item()
        values.append(-np.log(gen_rec_mse))
    return np.array(values)


def main():
    print("=" * 60)
    print("AIDI GS1/3/5/7 Gen↔Rec Distribution")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)

    plotted = 0
    for gs in [1, 3, 5, 7]:
        result_dir = os.path.join(OUTPUTS_DIR, f"aidi_gs{gs}")
        if not os.path.exists(result_dir):
            print(f"Warning: {result_dir} not found")
            continue

        gen_rec = compute_gen_rec_nlm(result_dir)
        color = COLORS.get(gs, "#666666")
        ax.hist(
            gen_rec,
            bins=50,
            density=True,
            alpha=0.35,
            color=color,
            edgecolor="none",
            label=f"AIDI-GS{gs} (mean={np.mean(gen_rec):.2f}, std={np.std(gen_rec):.2f}, n={len(gen_rec)})",
        )
        ax.axvline(np.mean(gen_rec), color=color, linestyle="--", linewidth=1.6, alpha=0.9)
        plotted += 1

        print(f"GS{gs}: mean={np.mean(gen_rec):.4f}, std={np.std(gen_rec):.4f}, n={len(gen_rec)}")

    if plotted == 0:
        raise RuntimeError("No valid aidi_gs* result directories found under outputs/reconstruction.")

    ax.set_xlabel("Gen↔Rec -log(MSE)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax.set_title("AIDI GS1/3/5/7 Distribution on Gen↔Rec -log(MSE)", fontsize=14, fontweight="bold", pad=14)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

    output_path = os.path.join(RESULTS_DIR, "aidi_gs1357_gen_rec_nlm_distribution.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"✓ Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
