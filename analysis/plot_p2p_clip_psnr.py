#!/usr/bin/env python3
"""
Plot P2P comparison using CLIP score (x) and PSNR (y) from original
p2p evaluation, and add one skip-inv point (skip_inv_dt5e9_fc20).
"""

import json
import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "p2p_evaluation")
INPUT_JSON = os.path.join(RESULTS_DIR, "p2p_evaluation.json")
PER_SAMPLE_JSON = os.path.join(RESULTS_DIR, "p2p_per_sample.json")
SKIP_INV_JSON = os.path.join(PROJECT_ROOT, "results", "p2p_evaluation_skip_inv", "p2p_evaluation.json")
OUTPUT_PNG = os.path.join(RESULTS_DIR, "p2p_clip_psnr_scatter.png")
OUTPUT_PDF = os.path.join(RESULTS_DIR, "p2p_clip_psnr_scatter.pdf")
CFG_SCHEDULE_PATH = os.path.join(
    PROJECT_ROOT, "outputs", "reconstruction", "skip_inv_dt5e9_fc20", "cfg_schedules.pt"
)
COMBINE_METHOD = "combine A7-7和A1-7"
PLOT_STYLE = {
    "font.size": 15,
    "axes.labelsize": 17,
    "axes.titlesize": 19,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
}


def method_color(method_name):
    if method_name == COMBINE_METHOD:
        return "#E74C3C"
    if method_name.startswith("aidi_gs1"):
        return "#3498DB"
    if method_name.startswith("aidi_gs3"):
        return "#2ECC71"
    if method_name.startswith("aidi_gs5"):
        return "#F39C12"
    if method_name.startswith("aidi_gs7"):
        return "#9B59B6"
    if method_name.startswith("afpi_ldt"):
        return "#16A085"
    if method_name == "upper_bound":
        return "#E74C3C"
    if method_name == "skip_inv_dt5e9_fc20":
        return "#34495E"
    return "#7F8C8D"


def method_label(method_name):
    if method_name == COMBINE_METHOD:
        return "COMBINE A7-7 + A1-7"
    m = re.match(r"^(aidi_gs\d+)_gs(\d+)$", method_name)
    if m:
        return f"{m.group(1).upper()}/GS{m.group(2)}"
    if method_name.startswith("afpi_ldt"):
        return method_name.upper()
    if method_name == "upper_bound":
        return "UPPER_BOUND"
    if method_name == "skip_inv_dt5e9_fc20":
        return "SKIP_INV_DT5E9_FC20"
    return method_name


def method_code(method_name):
    """Short code shown near each point."""
    if method_name == COMBINE_METHOD:
        return ""
    m = re.match(r"^aidi_gs(\d+)_gs(\d+)$", method_name)
    if m:
        return f"A{m.group(1)}-{m.group(2)}"
    m = re.match(r"^afpi_ldt(\d+)$", method_name)
    if m:
        return ""
    if method_name == "upper_bound":
        return "UB"
    if method_name == "skip_inv_dt5e9_fc20":
        return "SkipInv"
    return method_name[:6].upper()


def include_plot_point(method_name):
    if method_name == COMBINE_METHOD:
        return False
    if method_name.startswith("afpi_ldt"):
        return False
    return True


def build_combine_summary():
    if not (os.path.isfile(PER_SAMPLE_JSON) and os.path.isfile(CFG_SCHEDULE_PATH)):
        return None

    with open(PER_SAMPLE_JSON, "r") as f:
        per_sample = json.load(f)
    if "aidi_gs7_gs7" not in per_sample or "aidi_gs1_gs7" not in per_sample:
        return None

    by_7 = {int(s["idx"]): s for s in per_sample["aidi_gs7_gs7"]}
    by_1 = {int(s["idx"]): s for s in per_sample["aidi_gs1_gs7"]}
    cfg_schedules = torch.load(CFG_SCHEDULE_PATH, map_location="cpu")

    all_cfg_7 = {
        i for i, row in enumerate(cfg_schedules)
        if all(float(v) == 7.0 for v in row)
    }
    combined = []
    for i in range(len(cfg_schedules)):
        source = by_7 if i in all_cfg_7 else by_1
        if i in source:
            combined.append(source[i])
    if not combined:
        return None

    clip = np.array([s["clip_score"] for s in combined], dtype=float)
    psnr = np.array([s["psnr"] for s in combined], dtype=float)
    ssim = np.array([s["ssim"] for s in combined], dtype=float)
    lpips = np.array([s["lpips"] for s in combined], dtype=float)
    return {
        "clip_score_mean": float(np.mean(clip)),
        "clip_score_std": float(np.std(clip)),
        "psnr_mean": float(np.mean(psnr)),
        "psnr_std": float(np.std(psnr)),
        "ssim_mean": float(np.mean(ssim)),
        "ssim_std": float(np.std(ssim)),
        "lpips_mean": float(np.mean(lpips)),
        "lpips_std": float(np.std(lpips)),
        "n_samples": len(combined),
    }


def main():
    print("=" * 60)
    print("P2P CLIP vs PSNR Scatter Plot")
    print("=" * 60)

    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"Evaluation summary not found: {INPUT_JSON}")

    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    # Keep only skip_inv_dt5e9_fc20 among skip-inv methods.
    skip_methods = [m for m in data if m.startswith("skip_inv_")]
    for m in skip_methods:
        if m != "skip_inv_dt5e9_fc20":
            data.pop(m, None)

    # Keep only AFPI-LDT09 among F-* methods.
    f_methods = [m for m in data if m.startswith("afpi_ldt")]
    if f_methods:
        best_f = "afpi_ldt09" if "afpi_ldt09" in data else sorted(f_methods)[-1]
        for m in f_methods:
            if m != best_f:
                data.pop(m, None)
        print(f"Using pinned F method only: {best_f}")

    # Add skip_inv_dt5e9_fc20 from skip-inv evaluation result
    if os.path.exists(SKIP_INV_JSON):
        with open(SKIP_INV_JSON, "r") as f:
            skip_data = json.load(f)
        if "skip_inv_dt5e9_fc20" in skip_data:
            data["skip_inv_dt5e9_fc20"] = skip_data["skip_inv_dt5e9_fc20"]
            print("Added skip_inv_dt5e9_fc20 from skip-inv evaluation.")
        else:
            print("Warning: skip_inv_dt5e9_fc20 not found in skip-inv summary.")
    else:
        print(f"Warning: skip-inv summary not found: {SKIP_INV_JSON}")

    combine_summary = build_combine_summary()
    if combine_summary is not None:
        data[COMBINE_METHOD] = combine_summary
        print("Added combine A7-7和A1-7 from cfg_schedules + per-sample metrics.")
    else:
        print("Warning: unable to build combine A7-7和A1-7.")

    methods = sorted(method for method in data if include_plot_point(method))
    if not methods:
        raise RuntimeError("No methods found in evaluation summary.")

    x_vals = [data[m]["clip_score_mean"] for m in methods]
    y_vals = [data[m]["psnr_mean"] for m in methods]

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(11, 8), dpi=150)

        legend_handles = []
        for i, method in enumerate(methods):
            x = x_vals[i]
            y = y_vals[i]
            c = method_color(method)
            point_code = method_code(method)
            ax.scatter(x, y, s=260, alpha=0.78, color=c, edgecolors="black", linewidth=1.8)
            if point_code:
                ax.annotate(
                    point_code,
                    (x, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=11,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                )
            legend_handles.append(
                Line2D(
                    [0], [0],
                    marker="o",
                    color="w",
                    label=f"{point_code}: {method_label(method)}",
                    markerfacecolor=c,
                    markeredgecolor="black",
                    markersize=9,
                )
            )

        ax.set_xlabel("CLIP Score (mean)", fontweight="bold")
        ax.set_ylabel("PSNR (mean)", fontweight="bold")
        ax.set_title("Editing Results Comparison", fontweight="bold", pad=14)
        ax.grid(True, alpha=0.3, linestyle="--")

        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        x_margin = max((x_max - x_min) * 0.15, 0.002)
        y_margin = max((y_max - y_min) * 0.15, 0.8)
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        plt.tight_layout()
        fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
        fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    print(f"✓ Figure saved to: {OUTPUT_PNG}")
    print(f"✓ Figure saved to: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
