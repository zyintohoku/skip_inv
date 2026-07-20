#!/usr/bin/env python3
"""
Visualize reconstruction comparison as a scatter plot.
X-axis: Init↔Inv -log(MSE)
Y-axis: Gen↔Rec -log(MSE)

Included methods:
1) AIDI combinations aidi_gs*/gs*:
   - init/gen/inv from outputs/reconstruction/aidi_gs*
   - rec from outputs/reconstruction/aidi_gs*/gs*
2) afpi_ldt09 and skip_inv_dt5e9_fc20 from their own directories

Naming style follows analysis/plot_p2p_clip_psnr.py with short codes + legend.
"""

import os
import re
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "reconstruction")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "distribution")
OUTPUT_PNG = os.path.join(RESULTS_DIR, "aidi_gs_comparison.png")
OUTPUT_PDF = os.path.join(RESULTS_DIR, "aidi_gs_comparison.pdf")
CFG_SCHEDULE_PATH = os.path.join(OUTPUTS_DIR, "skip_inv_dt5e9_fc20", "cfg_schedules.pt")
OUTPUT_TABLE_MD = os.path.join(RESULTS_DIR, "aidi_all_methods_table.md")
OUTPUT_TABLE_CSV = os.path.join(RESULTS_DIR, "aidi_all_methods_table.csv")
PLOT_STYLE = {
    "font.size": 15,
    "axes.labelsize": 17,
    "axes.titlesize": 19,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
}


def method_color(method_name):
    if method_name == "combine A7-7和A1-7":
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
    if method_name.startswith("skip_inv"):
        return "#34495E"
    return "#7F8C8D"


def method_code(method_name):
    if method_name == "combine A7-7和A1-7":
        return ""
    m = re.match(r"^aidi_gs(\d+)_gs(\d+)$", method_name)
    if m:
        return f"A{m.group(1)}-{m.group(2)}"
    m = re.match(r"^afpi_ldt(\d+)$", method_name)
    if m:
        return ""
    if method_name == "skip_inv_dt5e9_fc20":
        return "SkipInv"
    return method_name[:6].upper()


def method_label(method_name):
    if method_name == "combine A7-7和A1-7":
        return "COMBINE A7-7 + A1-7"
    m = re.match(r"^aidi_gs(\d+)_gs(\d+)$", method_name)
    if m:
        return f"AIDI_GS{m.group(1)}/GS{m.group(2)}"
    if method_name.startswith("afpi_ldt"):
        return method_name.upper()
    if method_name == "skip_inv_dt5e9_fc20":
        return "SKIP_INV_DT5E9_FC20"
    return method_name


def include_plot_point(method_name):
    if method_name == "combine A7-7和A1-7":
        return False
    if method_name.startswith("afpi_ldt"):
        return False
    return True


def compute_metrics(init_inv_gen_dir, rec_dir):
    """Load latents and compute metrics.

    init/inv/gen are loaded from init_inv_gen_dir;
    rec is loaded from rec_dir.
    """
    init_latents = torch.load(os.path.join(init_inv_gen_dir, "init_latents.pt"), map_location="cpu")
    inv_latents = torch.load(os.path.join(init_inv_gen_dir, "inv_latents.pt"), map_location="cpu")
    gen_latents = torch.load(os.path.join(init_inv_gen_dir, "gen_latents.pt"), map_location="cpu")
    rec_latents = torch.load(os.path.join(rec_dir, "rec_latents.pt"), map_location="cpu")

    n = min(len(init_latents), len(inv_latents), len(gen_latents), len(rec_latents))
    init_inv_mse_list = []
    gen_rec_mse_list = []
    for i in range(n):
        init_inv_mse_list.append(F.mse_loss(init_latents[i], inv_latents[i]).item())
        gen_rec_mse_list.append(F.mse_loss(gen_latents[i], rec_latents[i]).item())

    init_inv_nlm = float(np.mean(-np.log(np.array(init_inv_mse_list))))
    gen_rec_nlm = float(np.mean(-np.log(np.array(gen_rec_mse_list))))
    return {
        "init_inv_nlm": init_inv_nlm,
        "gen_rec_nlm": gen_rec_nlm,
        "num_samples": n,
    }


def gather_data():
    """Collect all target methods and metrics."""
    data = {}

    # AIDI: base dir aidi_gs* with rec from subdir gs*
    for base_gs in [1, 3, 5, 7]:
        base_name = f"aidi_gs{base_gs}"
        base_dir = os.path.join(OUTPUTS_DIR, base_name)
        if not os.path.isdir(base_dir):
            continue

        for entry in sorted(os.listdir(base_dir)):
            sub_dir = os.path.join(base_dir, entry)
            if not (os.path.isdir(sub_dir) and entry.startswith("gs")):
                continue
            if not os.path.isfile(os.path.join(sub_dir, "rec_latents.pt")):
                continue

            method_name = f"{base_name}_{entry}"  # e.g. aidi_gs1_gs3
            data[method_name] = compute_metrics(base_dir, sub_dir)

    # Extra methods: afpi_ldt09 and skip_inv_dt5e9_fc20
    for method_name in ["afpi_ldt09", "skip_inv_dt5e9_fc20"]:
        method_dir = os.path.join(OUTPUTS_DIR, method_name)
        if not os.path.isdir(method_dir):
            print(f"Warning: {method_dir} not found")
            continue
        required = ["init_latents.pt", "inv_latents.pt", "gen_latents.pt", "rec_latents.pt"]
        if not all(os.path.isfile(os.path.join(method_dir, f)) for f in required):
            print(f"Warning: missing latent files in {method_dir}")
            continue
        data[method_name] = compute_metrics(method_dir, method_dir)

    combine_name = "combine A7-7和A1-7"
    aidi_gs7_dir = os.path.join(OUTPUTS_DIR, "aidi_gs7")
    aidi_gs1_dir = os.path.join(OUTPUTS_DIR, "aidi_gs1")
    rec_gs7_dir = os.path.join(aidi_gs7_dir, "gs7")
    rec_gs1_dir = os.path.join(aidi_gs1_dir, "gs7")
    combine_required = [
        CFG_SCHEDULE_PATH,
        os.path.join(aidi_gs7_dir, "init_latents.pt"),
        os.path.join(aidi_gs7_dir, "inv_latents.pt"),
        os.path.join(aidi_gs7_dir, "gen_latents.pt"),
        os.path.join(rec_gs7_dir, "rec_latents.pt"),
        os.path.join(aidi_gs1_dir, "init_latents.pt"),
        os.path.join(aidi_gs1_dir, "inv_latents.pt"),
        os.path.join(aidi_gs1_dir, "gen_latents.pt"),
        os.path.join(rec_gs1_dir, "rec_latents.pt"),
    ]
    if all(os.path.isfile(f) for f in combine_required):
        cfg_schedules = torch.load(CFG_SCHEDULE_PATH, map_location="cpu")
        all_cfg_7_ids = {
            i for i, row in enumerate(cfg_schedules)
            if all(float(v) == 7.0 for v in row)
        }

        init7 = torch.load(os.path.join(aidi_gs7_dir, "init_latents.pt"), map_location="cpu")
        inv7 = torch.load(os.path.join(aidi_gs7_dir, "inv_latents.pt"), map_location="cpu")
        gen7 = torch.load(os.path.join(aidi_gs7_dir, "gen_latents.pt"), map_location="cpu")
        rec7 = torch.load(os.path.join(rec_gs7_dir, "rec_latents.pt"), map_location="cpu")

        init1 = torch.load(os.path.join(aidi_gs1_dir, "init_latents.pt"), map_location="cpu")
        inv1 = torch.load(os.path.join(aidi_gs1_dir, "inv_latents.pt"), map_location="cpu")
        gen1 = torch.load(os.path.join(aidi_gs1_dir, "gen_latents.pt"), map_location="cpu")
        rec1 = torch.load(os.path.join(rec_gs1_dir, "rec_latents.pt"), map_location="cpu")

        n = min(
            len(cfg_schedules),
            len(init7), len(inv7), len(gen7), len(rec7),
            len(init1), len(inv1), len(gen1), len(rec1),
        )
        init_inv_mse_list = []
        gen_rec_mse_list = []
        for i in range(n):
            if i in all_cfg_7_ids:
                init_lat, inv_lat, gen_lat, rec_lat = init7[i], inv7[i], gen7[i], rec7[i]
            else:
                init_lat, inv_lat, gen_lat, rec_lat = init1[i], inv1[i], gen1[i], rec1[i]
            init_inv_mse_list.append(F.mse_loss(init_lat, inv_lat).item())
            gen_rec_mse_list.append(F.mse_loss(gen_lat, rec_lat).item())

        data[combine_name] = {
            "init_inv_nlm": float(np.mean(-np.log(np.array(init_inv_mse_list)))),
            "gen_rec_nlm": float(np.mean(-np.log(np.array(gen_rec_mse_list)))),
            "num_samples": n,
        }
    else:
        print("Warning: combine method files not complete, skipping combine A7-7和A1-7")

    return data


def save_results_table(data):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    methods = sorted(data.keys())

    with open(OUTPUT_TABLE_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "method_code", "init_inv_nlm", "gen_rec_nlm", "num_samples"])
        for method in methods:
            m = data[method]
            writer.writerow([
                method,
                method_code(method),
                f"{m['init_inv_nlm']:.6f}",
                f"{m['gen_rec_nlm']:.6f}",
                m["num_samples"],
            ])

    with open(OUTPUT_TABLE_MD, "w") as f:
        f.write("# AIDI/AFPI/Skip-Inv/Combine Metrics\n\n")
        f.write("| Method | Code | Init↔Inv -log(MSE) | Gen↔Rec -log(MSE) | Samples |\n")
        f.write("|---|---|---:|---:|---:|\n")
        for method in methods:
            m = data[method]
            f.write(
                f"| {method} | {method_code(method)} | {m['init_inv_nlm']:.4f} | "
                f"{m['gen_rec_nlm']:.4f} | {m['num_samples']} |\n"
            )


def create_scatter_plot(data, output_path):
    methods = sorted(method for method in data if include_plot_point(method))
    if not methods:
        raise RuntimeError("No valid methods found for plotting.")

    x_vals = [data[m]["init_inv_nlm"] for m in methods]
    y_vals = [data[m]["gen_rec_nlm"] for m in methods]

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(11, 8), dpi=150)
        legend_handles = []

        for i, method in enumerate(methods):
            x, y = x_vals[i], y_vals[i]
            c = method_color(method)
            code = method_code(method)
            label = method_label(method)

            ax.scatter(x, y, s=260, alpha=0.78, color=c, edgecolors="black", linewidth=1.8)
            if code:
                ax.annotate(
                    code,
                    (x, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=11,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                )

            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=f"{code}: {label}",
                    markerfacecolor=c,
                    markeredgecolor="black",
                    markersize=9,
                )
            )

        ax.set_xlabel("Init↔Inv -log(MSE)", fontweight="bold")
        ax.set_ylabel("Gen↔Rec -log(MSE)", fontweight="bold")
        ax.set_title("Reconstruction Results Comparison", fontweight="bold", pad=14)
        ax.grid(True, alpha=0.3, linestyle="--")

        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        x_margin = max((x_max - x_min) * 0.15, 0.1)
        y_margin = max((y_max - y_min) * 0.15, 0.1)
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"✓ Figure saved to: {output_path}")
    print(f"✓ Figure saved to: {OUTPUT_PDF}")


def main():
    print("=" * 60)
    print("AIDI/AFPI/Skip-Inv Scatter Plot")
    print("=" * 60)
    data = gather_data()

    print("\n" + "-" * 80)
    print(f"{'Method':<24} {'Init↔Inv':<15} {'Gen↔Rec':<15} {'Samples':<8}")
    print("-" * 80)
    for method in sorted(data.keys()):
        m = data[method]
        print(f"{method:<24} {m['init_inv_nlm']:<15.4f} {m['gen_rec_nlm']:<15.4f} {m['num_samples']:<8}")
    print("-" * 80)
    save_results_table(data)
    print(f"✓ Table saved to: {OUTPUT_TABLE_MD}")
    print(f"✓ Table saved to: {OUTPUT_TABLE_CSV}")

    create_scatter_plot(data, OUTPUT_PNG)


if __name__ == "__main__":
    main()
