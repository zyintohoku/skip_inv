import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: List[Dict], max_rows: int = 40) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "clip_image_score_mean",
        "psnr_mean",
        "abs_latent_mean_mean",
        "abs_1_minus_latent_std_mean",
        "latent_gaussian_deviation_mean",
        "original_prompt",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fields)) + " |\n")
        for row in rows[:max_rows]:
            values = []
            for field in fields:
                value = row[field]
                if isinstance(value, float):
                    if "psnr" in field:
                        value = f"{value:.4f}"
                    else:
                        value = f"{value:.6f}"
                values.append(str(value).replace("|", "\\|"))
            f.write("| " + " | ".join(values) + " |\n")


def parse_seed_spec(seed_spec: str) -> List[int]:
    seeds = []
    for token in seed_spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" not in token:
            seeds.append(int(token))
            continue
        start, end = token.split("-", 1)
        seeds.extend(range(int(start), int(end) + 1))
    if not seeds:
        raise ValueError("No seeds parsed.")
    return seeds


def mean_std(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def load_pair_scores(path: Path) -> Dict[tuple[int, int], Dict[str, str]]:
    pair_rows = read_csv(path)
    return {
        (int(row["sample_id"]), int(row["seed"])): row
        for row in pair_rows
    }


def compute_latent_stats(args: argparse.Namespace, prompt_rows: List[Dict[str, str]]) -> tuple[List[Dict], List[Dict]]:
    seeds = parse_seed_spec(args.seeds)
    prompt_by_sample = {int(row["sample_id"]): row for row in prompt_rows}
    pair_scores = load_pair_scores(Path(args.pair_scores_csv))
    per_seed_rows = []

    for seed in seeds:
        latent_path = Path(args.inv_latent_root_template.format(seed=seed)) / "inv_latents.pt"
        if not latent_path.exists():
            raise FileNotFoundError(f"Missing inv_latents.pt: {latent_path}")
        inv_latents = torch.load(latent_path, map_location="cpu")
        if len(inv_latents) != len(prompt_rows):
            raise ValueError(f"{latent_path} has {len(inv_latents)} latents, expected {len(prompt_rows)}")

        for sample_id, latent in enumerate(inv_latents):
            prompt_row = prompt_by_sample[sample_id]
            pair_row = pair_scores.get((sample_id, seed))
            if pair_row is None:
                raise KeyError(f"Missing pair score for sample_id={sample_id}, seed={seed}")
            latent_f = latent.detach().float()
            latent_mean = float(latent_f.mean().item())
            latent_std = float(latent_f.std(unbiased=False).item())
            abs_latent_mean = abs(latent_mean)
            abs_1_minus_latent_std = abs(1.0 - latent_std)
            per_seed_rows.append(
                {
                    "sample_id": sample_id,
                    "mapping_key": prompt_row["mapping_key"],
                    "seed": seed,
                    "original_prompt": prompt_row["original_prompt"],
                    "clip_image_score_mean": float(prompt_row["clip_image_score_mean"]),
                    "psnr_mean": float(prompt_row["psnr_mean"]),
                    "gen_rec_clip_image_score": float(pair_row["gen_rec_clip_image_score"]),
                    "psnr": float(pair_row["psnr"]),
                    "latent_mean": latent_mean,
                    "latent_std": latent_std,
                    "latent_l2_norm": float(torch.linalg.vector_norm(latent_f.reshape(-1)).item()),
                    "abs_latent_mean": abs_latent_mean,
                    "abs_1_minus_latent_std": abs_1_minus_latent_std,
                    "latent_gaussian_deviation": abs_latent_mean + abs_1_minus_latent_std,
                }
            )

    grouped: Dict[int, List[Dict]] = {}
    for row in per_seed_rows:
        grouped.setdefault(int(row["sample_id"]), []).append(row)

    summary_rows = []
    for sample_id in sorted(grouped):
        rows = grouped[sample_id]
        prompt_row = prompt_by_sample[sample_id]
        abs_mean_stats = mean_std([row["abs_latent_mean"] for row in rows])
        std_dev_stats = mean_std([row["abs_1_minus_latent_std"] for row in rows])
        gauss_dev_stats = mean_std([row["latent_gaussian_deviation"] for row in rows])
        latent_mean_stats = mean_std([row["latent_mean"] for row in rows])
        latent_std_stats = mean_std([row["latent_std"] for row in rows])
        l2_stats = mean_std([row["latent_l2_norm"] for row in rows])

        summary_rows.append(
            {
                "sample_id": sample_id,
                "mapping_key": prompt_row["mapping_key"],
                "original_prompt": prompt_row["original_prompt"],
                "editing_prompt": prompt_row["editing_prompt"],
                "n_seeds": len(rows),
                "clip_image_score_mean": float(prompt_row["clip_image_score_mean"]),
                "clip_image_score_std": float(prompt_row["clip_image_score_std"]),
                "psnr_mean": float(prompt_row["psnr_mean"]),
                "psnr_std": float(prompt_row["psnr_std"]),
                "abs_latent_mean_mean": abs_mean_stats["mean"],
                "abs_latent_mean_std": abs_mean_stats["std"],
                "abs_latent_mean_min": abs_mean_stats["min"],
                "abs_latent_mean_max": abs_mean_stats["max"],
                "abs_1_minus_latent_std_mean": std_dev_stats["mean"],
                "abs_1_minus_latent_std_std": std_dev_stats["std"],
                "abs_1_minus_latent_std_min": std_dev_stats["min"],
                "abs_1_minus_latent_std_max": std_dev_stats["max"],
                "latent_gaussian_deviation_mean": gauss_dev_stats["mean"],
                "latent_gaussian_deviation_std": gauss_dev_stats["std"],
                "latent_gaussian_deviation_min": gauss_dev_stats["min"],
                "latent_gaussian_deviation_max": gauss_dev_stats["max"],
                "latent_mean_mean": latent_mean_stats["mean"],
                "latent_mean_std": latent_mean_stats["std"],
                "latent_std_mean": latent_std_stats["mean"],
                "latent_std_std": latent_std_stats["std"],
                "latent_l2_norm_mean": l2_stats["mean"],
                "latent_l2_norm_std": l2_stats["std"],
            }
        )

    return per_seed_rows, summary_rows


def annotate_extremes(ax, rows: List[Dict], x_key: str, y_key: str, color_key: str) -> None:
    selected = {
        min(rows, key=lambda row: row[x_key])["sample_id"],
        max(rows, key=lambda row: row[x_key])["sample_id"],
        min(rows, key=lambda row: row[y_key])["sample_id"],
        max(rows, key=lambda row: row[y_key])["sample_id"],
        max(rows, key=lambda row: row[color_key])["sample_id"],
    }
    for row in rows:
        if row["sample_id"] not in selected:
            continue
        ax.annotate(
            str(row["sample_id"]),
            xy=(row[x_key], row[y_key]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="#222222",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "#cccccc",
                "alpha": 0.82,
            },
        )


def plot_colored_scatter(
    rows: List[Dict],
    color_key: str,
    cbar_label: str,
    output_prefix: Path,
) -> None:
    x_key = "clip_image_score_mean"
    y_key = "psnr_mean"
    x = np.array([row[x_key] for row in rows], dtype=np.float64)
    y = np.array([row[y_key] for row in rows], dtype=np.float64)
    c = np.array([row[color_key] for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    scatter = ax.scatter(
        x,
        y,
        c=c,
        cmap="viridis",
        s=42,
        alpha=0.80,
        edgecolors="white",
        linewidths=0.45,
    )
    ax.axvline(float(np.median(x)), color="#555555", linestyle="--", linewidth=1.1, label=f"median x = {np.median(x):.3f}")
    ax.axhline(float(np.median(y)), color="#777777", linestyle=":", linewidth=1.2, label=f"median y = {np.median(y):.2f}")
    annotate_extremes(ax, rows, x_key, y_key, color_key)

    ax.set_title("Prompt CLIP Image Score vs Mean PSNR", fontsize=15, pad=12)
    ax.set_xlabel("Mean CLIP image score across 10 seeds")
    ax.set_ylabel("Mean PSNR across 10 seeds (dB)")
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.75)
    ax.legend(loc="best", frameon=True)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def plot_per_seed_colored_scatter(
    rows: List[Dict],
    color_key: str,
    cbar_label: str,
    output_prefix: Path,
) -> None:
    x_key = "gen_rec_clip_image_score"
    y_key = "psnr"
    x = np.array([row[x_key] for row in rows], dtype=np.float64)
    y = np.array([row[y_key] for row in rows], dtype=np.float64)
    c = np.array([row[color_key] for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    scatter = ax.scatter(
        x,
        y,
        c=c,
        cmap="viridis",
        s=14,
        alpha=0.52,
        edgecolors="none",
    )
    ax.axvline(float(np.median(x)), color="#555555", linestyle="--", linewidth=1.1, label=f"median x = {np.median(x):.3f}")
    ax.axhline(float(np.median(y)), color="#777777", linestyle=":", linewidth=1.2, label=f"median y = {np.median(y):.2f}")
    ax.set_title("Sample-Seed CLIP Image Score vs PSNR", fontsize=15, pad=12)
    ax.set_xlabel("Generated-reconstructed CLIP image score")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.75)
    ax.legend(loc="best", frameon=True)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    prompt_rows = read_csv(Path(args.prompt_summary_csv))
    if len(prompt_rows) != args.expected_prompts:
        raise ValueError(f"Expected {args.expected_prompts} prompt rows, got {len(prompt_rows)}")

    output_dir = Path(args.output_dir)
    per_seed_rows, summary_rows = compute_latent_stats(args, prompt_rows)

    write_csv(output_dir / "all_prompt_inv_latent_gaussian_per_seed.csv", per_seed_rows)
    write_csv(output_dir / "all_prompt_inv_latent_gaussian_summary.csv", summary_rows)
    write_csv(output_dir / "all_prompt_clip_psnr_latent_gaussian_summary.csv", summary_rows)

    by_gaussian_deviation = sorted(
        summary_rows,
        key=lambda row: row["latent_gaussian_deviation_mean"],
        reverse=True,
    )
    write_markdown(
        output_dir / "all_prompt_inv_latent_gaussian_summary_top40_by_deviation.md",
        by_gaussian_deviation,
    )

    plots_dir = Path(args.plots_dir)
    plot_colored_scatter(
        summary_rows,
        color_key="abs_latent_mean_mean",
        cbar_label="Mean abs(latent mean) across seeds",
        output_prefix=plots_dir / "prompt_clip_mean_vs_psnr_mean_colored_by_abs_latent_mean",
    )
    plot_colored_scatter(
        summary_rows,
        color_key="abs_1_minus_latent_std_mean",
        cbar_label="Mean abs(1 - latent std) across seeds",
        output_prefix=plots_dir / "prompt_clip_mean_vs_psnr_mean_colored_by_abs_1_minus_latent_std",
    )
    plot_colored_scatter(
        summary_rows,
        color_key="latent_gaussian_deviation_mean",
        cbar_label="Mean abs(mean) + abs(1 - std) across seeds",
        output_prefix=plots_dir / "prompt_clip_mean_vs_psnr_mean_colored_by_latent_gaussian_deviation",
    )
    plot_colored_scatter(
        summary_rows,
        color_key="latent_l2_norm_mean",
        cbar_label="Mean latent L2 norm across seeds",
        output_prefix=plots_dir / "prompt_clip_mean_vs_psnr_mean_colored_by_latent_l2_norm_mean",
    )
    plot_per_seed_colored_scatter(
        per_seed_rows,
        color_key="abs_latent_mean",
        cbar_label="abs(latent mean)",
        output_prefix=plots_dir / "sample_seed_clip_score_vs_psnr_colored_by_abs_latent_mean",
    )
    plot_per_seed_colored_scatter(
        per_seed_rows,
        color_key="abs_1_minus_latent_std",
        cbar_label="abs(1 - latent std)",
        output_prefix=plots_dir / "sample_seed_clip_score_vs_psnr_colored_by_abs_1_minus_latent_std",
    )
    plot_per_seed_colored_scatter(
        per_seed_rows,
        color_key="latent_gaussian_deviation",
        cbar_label="abs(mean) + abs(1 - std)",
        output_prefix=plots_dir / "sample_seed_clip_score_vs_psnr_colored_by_latent_gaussian_deviation",
    )
    plot_per_seed_colored_scatter(
        per_seed_rows,
        color_key="latent_l2_norm",
        cbar_label="latent L2 norm",
        output_prefix=plots_dir / "sample_seed_clip_score_vs_psnr_colored_by_latent_l2_norm",
    )
    print(f"saved latent Gaussian stats to: {output_dir}")
    print(f"saved plots to: {plots_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute all-prompt inversion latent Gaussian stats and plot them on CLIP/PSNR axes."
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
        "--inv_latent_root_template",
        type=str,
        default="outputs/fpi_gs7_seed{seed}_from_saved_latents",
    )
    parser.add_argument("--seeds", type=str, default="1-10")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/all_prompt_seed_clip_scores/all_prompt_inv_latent_gaussian_summary",
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default="results/all_prompt_seed_clip_scores/plots",
    )
    parser.add_argument("--expected_prompts", type=int, default=700)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
