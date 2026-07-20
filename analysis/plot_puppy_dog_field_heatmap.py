#!/usr/bin/env python3
"""Plot puppy/dog field prompt-grid PSNR heatmaps."""

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_ORDER = [
    ("bare_class", "bare"),
    ("field_np", "in a field"),
    ("field", "sitting in field"),
    ("field_dandelions", "sitting in dandelions"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create puppy/dog field heatmaps.")
    parser.add_argument(
        "--by_prompt_csv",
        default=str(
            PROJECT_ROOT
            / "results"
            / "fpi_gs7_seed_psnr"
            / "puppy_dog_field_fpi"
            / "puppy_dog_field_fpi_psnr_by_prompt.csv"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "results" / "fpi_gs7_seed_psnr" / "puppy_dog_field_fpi"),
    )
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_numeric_pivot(path: Path, labels: List[str], contexts: List[str], matrix: np.ndarray) -> None:
    rows = []
    for i, label in enumerate(labels):
        row = {"label": label}
        for j, context in enumerate(contexts):
            row[context] = f"{matrix[i, j]:.6f}"
        rows.append(row)
    write_csv(path, rows, ["label", *contexts])


def write_text_pivot(path: Path, labels: List[str], contexts: List[str], matrix: List[List[str]]) -> None:
    rows = []
    for i, label in enumerate(labels):
        row = {"label": label}
        for j, context in enumerate(contexts):
            row[context] = matrix[i][j]
        rows.append(row)
    write_csv(path, rows, ["label", *contexts])


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.by_prompt_csv))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    for row in rows:
        label = row["label"]
        if label not in labels:
            labels.append(label)
    contexts = [key for key, _ in CONTEXT_ORDER]
    context_labels = [label for _, label in CONTEXT_ORDER]

    value_by_cell = {}
    cell_rows = []
    for row in rows:
        cell = {
            "label": row["label"],
            "context": row["group"],
            "prompt": row["prompt"],
            "n": row["n"],
            "image_psnr_mean": float(row["image_psnr_mean"]),
            "image_psnr_std": float(row["image_psnr_std"]),
            "image_psnr_var": float(row["image_psnr_std"]) ** 2,
            "image_psnr_min": float(row["image_psnr_min"]),
            "image_psnr_max": float(row["image_psnr_max"]),
        }
        value_by_cell[(cell["label"], cell["context"])] = cell
        cell_rows.append(cell)

    mean_matrix = np.full((len(labels), len(contexts)), np.nan)
    std_matrix = np.full_like(mean_matrix, np.nan)
    var_matrix = np.full_like(mean_matrix, np.nan)
    for i, label in enumerate(labels):
        for j, context in enumerate(contexts):
            cell = value_by_cell[(label, context)]
            mean_matrix[i, j] = cell["image_psnr_mean"]
            std_matrix[i, j] = cell["image_psnr_std"]
            var_matrix[i, j] = cell["image_psnr_var"]

    mean_low = float(np.nanquantile(mean_matrix, 0.25))
    mean_high = float(np.nanquantile(mean_matrix, 0.75))
    std_mid = float(np.nanmedian(std_matrix))
    category_to_code = {"Hard": 0, "Mixed": 1, "Easy": 2}
    category_matrix = [["" for _ in contexts] for _ in labels]
    category_code = np.full_like(mean_matrix, np.nan)
    for i in range(len(labels)):
        for j in range(len(contexts)):
            if mean_matrix[i, j] >= mean_high and std_matrix[i, j] <= std_mid:
                category = "Easy"
            elif mean_matrix[i, j] <= mean_low and std_matrix[i, j] <= std_mid:
                category = "Hard"
            else:
                category = "Mixed"
            category_matrix[i][j] = category
            category_code[i, j] = category_to_code[category]

    cell_fields = [
        "label",
        "context",
        "prompt",
        "n",
        "image_psnr_mean",
        "image_psnr_std",
        "image_psnr_var",
        "image_psnr_min",
        "image_psnr_max",
    ]
    write_csv(output_dir / "puppy_dog_field_class_context_psnr.csv", cell_rows, cell_fields)
    write_numeric_pivot(output_dir / "puppy_dog_field_class_context_mean_matrix.csv", labels, contexts, mean_matrix)
    write_numeric_pivot(output_dir / "puppy_dog_field_class_context_std_matrix.csv", labels, contexts, std_matrix)
    write_numeric_pivot(output_dir / "puppy_dog_field_class_context_var_matrix.csv", labels, contexts, var_matrix)
    write_text_pivot(output_dir / "puppy_dog_field_class_context_category_matrix.csv", labels, contexts, category_matrix)
    write_csv(
        output_dir / "puppy_dog_field_class_context_category_thresholds.csv",
        [
            {
                "mean_low_q25": mean_low,
                "mean_high_q75": mean_high,
                "std_mid_median": std_mid,
                "easy_rule": "mean >= mean_high_q75 and std <= std_mid_median",
                "hard_rule": "mean <= mean_low_q25 and std <= std_mid_median",
                "mixed_rule": "all remaining cells",
            }
        ],
        ["mean_low_q25", "mean_high_q75", "std_mid_median", "easy_rule", "hard_rule", "mixed_rule"],
    )

    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "default"):
        try:
            plt.style.use(style)
            break
        except OSError:
            continue

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
    mean_im = axes[0].imshow(mean_matrix, cmap="viridis", aspect="auto")
    axes[0].set_title("Mean PSNR with std annotation")
    axes[0].set_xticks(np.arange(len(contexts)), context_labels, rotation=20, ha="right")
    axes[0].set_yticks(np.arange(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(contexts)):
            axes[0].text(
                j,
                i,
                f"{mean_matrix[i, j]:.1f}\n±{std_matrix[i, j]:.1f}",
                ha="center",
                va="center",
                color="white" if mean_matrix[i, j] < np.nanmean(mean_matrix) else "black",
                fontsize=8,
            )
    fig.colorbar(mean_im, ax=axes[0], fraction=0.046, pad=0.04, label="Mean PSNR")

    category_cmap = ListedColormap(["#4b3b8f", "#f0c44f", "#2f9e44"])
    cat_im = axes[1].imshow(category_code, cmap=category_cmap, vmin=0, vmax=2, aspect="auto")
    axes[1].set_title("Easy / Mixed / Hard classes")
    axes[1].set_xticks(np.arange(len(contexts)), context_labels, rotation=20, ha="right")
    axes[1].set_yticks(np.arange(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(contexts)):
            category = category_matrix[i][j]
            axes[1].text(
                j,
                i,
                f"{category}\n{mean_matrix[i, j]:.1f}±{std_matrix[i, j]:.1f}",
                ha="center",
                va="center",
                color="white" if category == "Hard" else "black",
                fontsize=7,
            )
    cbar = fig.colorbar(cat_im, ax=axes[1], fraction=0.046, pad=0.04, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Hard", "Mixed", "Easy"])
    cbar.set_label(
        f"thresholds: mean q25={mean_low:.1f}, q75={mean_high:.1f}; std median={std_mid:.1f}"
    )

    for ax in axes:
        ax.set_xlabel("Context")
        ax.set_ylabel("Class label")
        ax.tick_params(axis="both", length=0)

    png_path = output_dir / "puppy_dog_field_class_context_heatmap.png"
    pdf_path = output_dir / "puppy_dog_field_class_context_heatmap.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved heatmap: {png_path}")
    print(f"Saved heatmap: {pdf_path}")


if __name__ == "__main__":
    main()
