import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


PLOT_STYLE = {
    "font.size": 15,
    "axes.labelsize": 17,
    "axes.titlesize": 19,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
}
plt.rcParams.update(PLOT_STYLE)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def float_array(rows: List[Dict[str, str]], key: str) -> np.ndarray:
    return np.array([float(row[key]) for row in rows], dtype=np.float64)


def label_rows(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    by_low_psnr = min(rows, key=lambda row: float(row["psnr_mean"]))
    by_high_psnr_std = max(rows, key=lambda row: float(row["psnr_std"]))
    by_low_clip = min(rows, key=lambda row: float(row["clip_image_score_mean"]))
    by_high_clip_std = max(rows, key=lambda row: float(row["clip_image_score_std"]))
    labels = {
        "low_psnr": by_low_psnr,
        "high_psnr_std": by_high_psnr_std,
        "low_clip": by_low_clip,
        "high_clip_std": by_high_clip_std,
    }
    # Deduplicate when the same prompt is selected by multiple criteria.
    return {str(row["sample_id"]): row for row in labels.values()}


def annotate_selected(ax, selected: Dict[str, Dict[str, str]], x_key: str, y_key: str) -> None:
    for row in selected.values():
        ax.annotate(
            str(row["sample_id"]),
            xy=(float(row[x_key]), float(row[y_key])),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            color="#222222",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "#cccccc",
                "alpha": 0.82,
            },
        )


def scatter_mean_std(
    rows: List[Dict[str, str]],
    x_key: str,
    y_key: str,
    color_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
    output_prefix: Path,
    annotate: bool = True,
    uniform_color: str = "",
) -> None:
    x = float_array(rows, x_key)
    y = float_array(rows, y_key)
    median_x = float(np.median(x))
    median_y = float(np.median(y))

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    if uniform_color:
        scatter = ax.scatter(
            x,
            y,
            color=uniform_color,
            s=42,
            alpha=0.78,
            edgecolors="white",
            linewidths=0.45,
        )
    else:
        c = float_array(rows, color_key)
        scatter = ax.scatter(
            x,
            y,
            c=c,
            cmap="viridis",
            s=42,
            alpha=0.78,
            edgecolors="white",
            linewidths=0.45,
        )
    ax.axvline(median_x, color="#555555", linestyle="--", linewidth=1.1, label=f"median x = {median_x:.3f}")
    ax.axhline(median_y, color="#777777", linestyle=":", linewidth=1.2, label=f"median y = {median_y:.3f}")
    if annotate:
        annotate_selected(ax, label_rows(rows), x_key, y_key)

    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.75)
    if annotate:
        ax.legend(loc="best", frameon=True)
    if not uniform_color:
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def scatter_xy(
    rows: List[Dict[str, str]],
    x_key: str,
    y_key: str,
    color_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
    output_prefix: Path,
) -> None:
    x = float_array(rows, x_key)
    y = float_array(rows, y_key)
    c = float_array(rows, color_key)

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    scatter = ax.scatter(
        x,
        y,
        c=c,
        cmap="plasma",
        s=42,
        alpha=0.78,
        edgecolors="white",
        linewidths=0.45,
    )
    annotate_selected(ax, label_rows(rows), x_key, y_key)

    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.75)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def histogram_clip_mean(rows: List[Dict[str, str]], output_prefix: Path) -> None:
    values = float_array(rows, "clip_image_score_mean")
    median = float(np.median(values))
    mean = float(np.mean(values))

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.hist(
        values,
        bins=36,
        color="#4c78a8",
        edgecolor="white",
        linewidth=0.7,
        alpha=0.86,
    )
    ax.axvline(mean, color="#333333", linestyle="--", linewidth=1.2, label=f"mean = {mean:.3f}")
    ax.axvline(median, color="#777777", linestyle=":", linewidth=1.4, label=f"median = {median:.3f}")
    ax.set_title("Distribution of Prompt Mean CLIP Image Score", pad=12)
    ax.set_xlabel("Mean CLIP image score across 10 seeds")
    ax.set_ylabel("Number of prompts")
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.7, alpha=0.75)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def histogram_pair_clip_scores(rows: List[Dict[str, str]], output_prefix: Path) -> None:
    values = np.array([float(row["gen_rec_clip_image_score"]) for row in rows], dtype=np.float64)
    median = float(np.median(values))
    mean = float(np.mean(values))

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.hist(
        values,
        bins=60,
        color="#59a14f",
        edgecolor="white",
        linewidth=0.55,
        alpha=0.86,
    )
    ax.axvline(mean, color="#333333", linestyle="--", linewidth=1.2, label=f"mean = {mean:.3f}")
    ax.axvline(median, color="#777777", linestyle=":", linewidth=1.4, label=f"median = {median:.3f}")
    ax.set_title("Distribution of CLIP Image Score over 7000 Image Pairs", pad=12)
    ax.set_xlabel("Generated-Reconstructed CLIP image score")
    ax.set_ylabel("Number of image pairs")
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.7, alpha=0.75)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def histogram_psnr_mean(rows: List[Dict[str, str]], output_prefix: Path) -> None:
    values = float_array(rows, "psnr_mean")
    median = float(np.median(values))
    mean = float(np.mean(values))

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.hist(
        values,
        bins=36,
        color="#f28e2b",
        edgecolor="white",
        linewidth=0.7,
        alpha=0.86,
    )
    ax.axvline(mean, color="#333333", linestyle="--", linewidth=1.2, label=f"mean = {mean:.2f}")
    ax.axvline(median, color="#777777", linestyle=":", linewidth=1.4, label=f"median = {median:.2f}")
    ax.set_title("Distribution of Prompt Mean FPI Reconstruction PSNR", pad=12)
    ax.set_xlabel("Mean PSNR across 10 seeds (dB)")
    ax.set_ylabel("Number of prompts")
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.7, alpha=0.75)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def scatter_pair_clip_vs_psnr(
    rows: List[Dict[str, str]],
    output_prefix: Path,
    title: str,
    xlim=None,
) -> None:
    x = np.array([float(row["gen_rec_clip_image_score"]) for row in rows], dtype=np.float64)
    y = np.array([float(row["psnr"]) for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    ax.scatter(
        x,
        y,
        color="#4c78a8",
        s=18,
        alpha=0.45,
        edgecolors="none",
    )
    ax.set_title(title, pad=12)
    ax.set_xlabel("Generated-Reconstructed CLIP image score")
    ax.set_ylabel("PSNR (dB)")
    if xlim is not None:
        ax.set_xlim(*xlim)
    y_margin = max(2.0, 0.04 * (float(y.max()) - float(y.min())))
    ax.set_ylim(max(0.0, float(y.min()) - y_margin), float(y.max()) + y_margin)
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.75)
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    rows = read_csv(Path(args.prompt_summary_csv))
    pair_rows = read_csv(Path(args.pair_scores_csv))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(rows) != args.expected_prompts:
        raise ValueError(f"Expected {args.expected_prompts} prompt rows, got {len(rows)}")
    if len(pair_rows) != args.expected_pairs:
        raise ValueError(f"Expected {args.expected_pairs} pair rows, got {len(pair_rows)}")

    scatter_mean_std(
        rows=rows,
        x_key="clip_image_score_mean",
        y_key="clip_image_score_std",
        color_key="clip_image_score_mean",
        title="Prompt CLIP Reconstruction Similarity vs Seed Sensitivity",
        xlabel="Mean CLIP image score across 10 seeds",
        ylabel="CLIP image score std across 10 seeds",
        cbar_label="Mean CLIP image score",
        output_prefix=output_dir / "prompt_clip_mean_vs_std",
        uniform_color="#4c78a8",
    )

    scatter_xy(
        rows=rows,
        x_key="clip_image_score_mean",
        y_key="psnr_mean",
        color_key="clip_image_score_std",
        title="Prompt CLIP Image Score vs Mean PSNR",
        xlabel="Mean CLIP image score across 10 seeds",
        ylabel="Mean PSNR across 10 seeds (dB)",
        cbar_label="CLIP image score std",
        output_prefix=output_dir / "prompt_clip_mean_vs_psnr_mean",
    )

    scatter_mean_std(
        rows=rows,
        x_key="psnr_mean",
        y_key="psnr_std",
        color_key="clip_image_score_mean",
        title="Prompt PSNR Quality vs Seed Sensitivity Colored by CLIP",
        xlabel="Mean PSNR across 10 seeds (dB)",
        ylabel="PSNR std across 10 seeds (dB)",
        cbar_label="Mean CLIP image score",
        output_prefix=output_dir / "prompt_psnr_mean_vs_std_colored_by_clip",
    )

    scatter_mean_std(
        rows=rows,
        x_key="psnr_mean",
        y_key="psnr_std",
        color_key="clip_image_score_mean",
        title="Prompt PSNR Quality vs Seed Sensitivity",
        xlabel="Mean PSNR across 10 seeds (dB)",
        ylabel="PSNR std across 10 seeds (dB)",
        cbar_label="Mean CLIP image score",
        output_prefix=output_dir / "prompt_psnr_mean_vs_std",
        annotate=False,
        uniform_color="#4c78a8",
    )

    histogram_clip_mean(
        rows=rows,
        output_prefix=output_dir / "prompt_clip_mean_distribution",
    )

    histogram_pair_clip_scores(
        rows=pair_rows,
        output_prefix=output_dir / "all_pair_clip_score_distribution",
    )

    histogram_psnr_mean(
        rows=rows,
        output_prefix=output_dir / "prompt_fpi_psnr_mean_distribution",
    )

    scatter_pair_clip_vs_psnr(
        rows=pair_rows,
        output_prefix=output_dir / "all_pair_clip_score_vs_psnr",
        title="CLIP Image Score vs FPI Reconstruction PSNR over 7000 Image Pairs",
    )

    filtered_pair_rows = [
        row for row in pair_rows
        if 0.9 <= float(row["gen_rec_clip_image_score"]) <= 1.0
    ]
    scatter_pair_clip_vs_psnr(
        rows=filtered_pair_rows,
        output_prefix=output_dir / "all_pair_clip_score_0p9_1p0_vs_psnr",
        title="CLIP Image Score vs PSNR for Image Pairs with 0.9 <= CLIP <= 1.0",
        xlim=(0.9, 1.0),
    )

    print(f"saved plots to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all-prompt CLIP and PSNR summary relationships."
    )
    parser.add_argument(
        "--prompt_summary_csv",
        type=str,
        default="results/all_prompt_seed_clip_scores/all_seed_reconstruction_clip_by_prompt.csv",
    )
    parser.add_argument(
        "--pair_scores_csv",
        type=str,
        default="results/all_prompt_seed_clip_scores/all_seed_reconstruction_clip_scores.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/all_prompt_seed_clip_scores/plots",
    )
    parser.add_argument("--expected_prompts", type=int, default=700)
    parser.add_argument("--expected_pairs", type=int, default=7000)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
