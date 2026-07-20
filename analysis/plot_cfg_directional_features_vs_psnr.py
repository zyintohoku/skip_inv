import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


FEATURES = [
    ("weighted_alignment_C_u", r"$C^{(u)}$"),
    ("orthogonal_bending_ratio_rho_perp_u", r"$\rho_{\perp}^{(u)}$"),
    ("reverse_direction_ratio_rho_minus_u", r"$\rho_{-}^{(u)}$"),
    ("uncond_trajectory_deviation_auc_D_u", r"$D_{\mathrm{auc}}^{(u)}$"),
]


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["sample_id"] = int(row["sample_id"])
        row["seed"] = int(row["seed"])
        row["psnr"] = float(row["psnr"])
        for feature, _ in FEATURES:
            row[feature] = float(row[feature])
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_or_merge(input_root: Path, merged_csv: Path):
    rows = []
    for path in sorted(input_root.glob("sample_*/per_seed_cfg_directional_features.csv")):
        rows.extend(read_rows(path))
    if not rows:
        raise FileNotFoundError(f"No per-seed summaries found under {input_root}")
    write_csv(merged_csv, rows)
    return rows


def sample_colors(rows):
    sample_ids = sorted({row["sample_id"] for row in rows})
    cmap = plt.get_cmap("tab10")
    return {sample_id: cmap(i % 10) for i, sample_id in enumerate(sample_ids)}


def plot_one_feature(rows, feature, label, colors, output_path: Path):
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for sample_id in sorted(colors):
        sample_rows = [row for row in rows if row["sample_id"] == sample_id]
        ax.scatter(
            [row[feature] for row in sample_rows],
            [row["psnr"] for row in sample_rows],
            s=34,
            alpha=0.82,
            color=colors[sample_id],
            edgecolors="none",
            label=str(sample_id),
        )
    ax.set_xlabel(label)
    ax.set_ylabel("Reconstruction PSNR")
    ax.grid(alpha=0.25)
    ax.set_title(f"{label} vs reconstruction PSNR", fontsize=11)
    ax.legend(title="sample_id", fontsize=7, title_fontsize=8, ncol=2, loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_overview(rows, colors, output_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2))
    axes = axes.reshape(-1)
    for ax, (feature, label) in zip(axes, FEATURES):
        for sample_id in sorted(colors):
            sample_rows = [row for row in rows if row["sample_id"] == sample_id]
            ax.scatter(
                [row[feature] for row in sample_rows],
                [row["psnr"] for row in sample_rows],
                s=26,
                alpha=0.82,
                color=colors[sample_id],
                edgecolors="none",
                label=str(sample_id),
            )
        ax.set_xlabel(label)
        ax.set_ylabel("PSNR")
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="sample_id",
        loc="center right",
        fontsize=8,
        title_fontsize=9,
    )
    fig.suptitle("Sensitive prompts: CFG directional features vs reconstruction PSNR", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_root",
        type=str,
        default="outputs/cfg_directional_features_seed_sensitive_top10/seed_sensitive",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/cfg_directional_features_seed_sensitive_top10_analysis/feature_vs_psnr",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rows = load_or_merge(
        Path(args.input_root),
        output_dir / "per_seed_cfg_directional_features_with_psnr.csv",
    )
    colors = sample_colors(rows)

    for feature, label in FEATURES:
        plot_one_feature(rows, feature, label, colors, output_dir / f"{feature}_vs_PSNR.png")
    plot_overview(rows, colors, output_dir / "cfg_directional_features_vs_PSNR_overview.png")

    print(f"saved {len(rows)} points to {output_dir / 'per_seed_cfg_directional_features_with_psnr.csv'}")
    print(f"saved plots to {output_dir}")


if __name__ == "__main__":
    main()
