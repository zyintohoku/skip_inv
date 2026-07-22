#!/usr/bin/env python3
"""Plot sensitive prompt pressure against FPI and DDIM reconstruction PSNR."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return {int(row["seed"]): row for row in csv.DictReader(f)}


def mean(values):
    return sum(values) / len(values)


def sample_std(values):
    value_mean = mean(values)
    return math.sqrt(sum((value - value_mean) ** 2 for value in values) / (len(values) - 1))


def pearson(xs, ys):
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_std = sample_std(xs)
    y_std = sample_std(ys)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / ((len(xs) - 1) * x_std * y_std)


def fit_line(xs, ys):
    x_mean = mean(xs)
    y_mean = mean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
    intercept = y_mean - slope * x_mean
    return slope, intercept


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["seed", "prompt_pressure_total", "fpi_psnr", "ddim_psnr"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path, args, fpi_corr, ddim_corr, row_count):
    path.write_text(
        "# Sensitive Prompt Pressure vs PSNR\n\n"
        "This directory contains the formal analysis outputs for the sensitive prompt pressure experiment.\n\n"
        f"- Rows: `{row_count}` seeds\n"
        f"- FPI manifest: `{args.fpi_manifest}`\n"
        f"- DDIM pressure manifest: `{args.ddim_manifest}`\n"
        "- X axis: `prompt_pressure_total` from the DDIM prompt-pressure generation run.\n"
        "- Y axis: reconstruction PSNR.\n"
        f"- FPI Pearson r: `{fpi_corr:.6f}`\n"
        f"- DDIM Pearson r: `{ddim_corr:.6f}`\n",
        encoding="utf-8",
    )


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fpi_rows_by_seed = load_rows(args.fpi_manifest)
    ddim_rows_by_seed = load_rows(args.ddim_manifest)
    seeds = sorted(set(fpi_rows_by_seed) & set(ddim_rows_by_seed))
    if not seeds:
        raise ValueError("No overlapping seeds found between FPI and DDIM manifests.")

    rows = []
    for seed in seeds:
        rows.append(
            {
                "seed": seed,
                "prompt_pressure_total": float(ddim_rows_by_seed[seed]["prompt_pressure_total"]),
                "fpi_psnr": float(fpi_rows_by_seed[seed]["image_psnr"]),
                "ddim_psnr": float(ddim_rows_by_seed[seed]["image_psnr"]),
            }
        )

    pressure = [row["prompt_pressure_total"] for row in rows]
    fpi_psnr = [row["fpi_psnr"] for row in rows]
    ddim_psnr = [row["ddim_psnr"] for row in rows]
    fpi_corr = pearson(pressure, fpi_psnr)
    ddim_corr = pearson(pressure, ddim_psnr)

    csv_path = output_dir / "sensitive_pressure_vs_psnr_fpi_ddim.csv"
    write_csv(csv_path, rows)

    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=180)
    ax.scatter(pressure, fpi_psnr, s=34, alpha=0.78, color="#2563eb", label=f"FPI (r={fpi_corr:.3f})")
    ax.scatter(pressure, ddim_psnr, s=34, alpha=0.78, color="#dc2626", label=f"DDIM inv (r={ddim_corr:.3f})")

    line_x = [min(pressure), max(pressure)]
    for psnr_values, color in [(fpi_psnr, "#2563eb"), (ddim_psnr, "#dc2626")]:
        slope, intercept = fit_line(pressure, psnr_values)
        ax.plot(line_x, [slope * x + intercept for x in line_x], color=color, linewidth=1.8, alpha=0.85)

    ax.set_title("Sensitive prompt: reconstruction PSNR vs total prompt pressure")
    ax.set_xlabel("Total prompt pressure")
    ax.set_ylabel("Reconstruction PSNR")
    ax.grid(True, linewidth=0.6, alpha=0.28)
    ax.legend(frameon=False)
    fig.tight_layout()

    png_path = output_dir / "sensitive_pressure_vs_psnr_fpi_ddim.png"
    pdf_path = output_dir / "sensitive_pressure_vs_psnr_fpi_ddim.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    write_readme(output_dir / "README.md", args, fpi_corr, ddim_corr, len(rows))

    if args.preview_dir:
        preview_dir = Path(args.preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
        for src in [csv_path, png_path, pdf_path, output_dir / "README.md"]:
            (preview_dir / src.name).write_bytes(src.read_bytes())

    print(f"wrote: {output_dir}")
    print(f"rows: {len(rows)}")
    print(f"fpi_corr: {fpi_corr}")
    print(f"ddim_corr: {ddim_corr}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fpi_manifest",
        default="../artifacts/outputs/prompt_group_top1_fpi_100/sensitive/manifest.csv",
    )
    parser.add_argument(
        "--ddim_manifest",
        default="../artifacts/outputs/sensitive_prompt_pressure_ddim_inv_rec/manifest.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="../artifacts/results/sensitive_pressure_psnr_plot",
    )
    parser.add_argument(
        "--preview_dir",
        default="tmp_prompt_pressure_psnr_plot",
        help="Optional src-local copy for quick viewing. Pass an empty string to disable.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
