#!/usr/bin/env python3
"""Plot class-by-context PSNR heatmaps for modifier prompt-grid FPI results."""

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_ORDER = [
    ("bare", "bare class"),
    ("wooden_fence", "on wooden fence"),
    ("wooden_table", "on wooden table"),
    ("tree_branch", "on tree branch"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create class x context PSNR heatmaps.")
    parser.add_argument(
        "--by_prompt_csv",
        default=str(
            PROJECT_ROOT
            / "results"
            / "fpi_gs7_seed_psnr"
            / "modifier_prompt_grid_fpi"
            / "modifier_prompt_grid_fpi_psnr_by_prompt.csv"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "results" / "fpi_gs7_seed_psnr" / "modifier_prompt_grid_fpi"),
    )
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def context_from_row(row: Dict[str, str]) -> str:
    prompt = row["prompt"]
    if row.get("group") == "bare_class":
        return "bare"
    if prompt.endswith("on a wooden fence"):
        return "wooden_fence"
    if prompt.endswith("on a wooden table"):
        return "wooden_table"
    if prompt.endswith("on a tree branch"):
        return "tree_branch"
    return "other"


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_pivot(path: Path, labels: List[str], contexts: List[str], matrix: np.ndarray) -> None:
    fieldnames = ["label", *contexts]
    rows = []
    for label_idx, label in enumerate(labels):
        row = {"label": label}
        for context_idx, context in enumerate(contexts):
            row[context] = f"{matrix[label_idx, context_idx]:.6f}"
        rows.append(row)
    write_csv(path, rows, fieldnames)


def write_text_pivot(path: Path, labels: List[str], contexts: List[str], matrix: List[List[str]]) -> None:
    fieldnames = ["label", *contexts]
    rows = []
    for label_idx, label in enumerate(labels):
        row = {"label": label}
        for context_idx, context in enumerate(contexts):
            row[context] = matrix[label_idx][context_idx]
        rows.append(row)
    write_csv(path, rows, fieldnames)


def main() -> None:
    args = parse_args()
    by_prompt_csv = Path(args.by_prompt_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(by_prompt_csv)
    labels = []
    for row in rows:
        label = row.get("label", "")
        if label and label not in labels:
            labels.append(label)

    contexts = [key for key, _ in CONTEXT_ORDER]
    context_labels = [label for _, label in CONTEXT_ORDER]
    cell_rows = []
    value_by_cell = {}
    for row in rows:
        label = row.get("label", "")
        context = context_from_row(row)
        if not label or context not in contexts:
            continue
        mean_value = float(row["image_psnr_mean"])
        std_value = float(row["image_psnr_std"])
        out = {
            "label": label,
            "context": context,
            "prompt": row["prompt"],
            "n": row["n"],
            "image_psnr_mean": mean_value,
            "image_psnr_std": std_value,
            "image_psnr_var": std_value**2,
            "image_psnr_min": float(row["image_psnr_min"]),
            "image_psnr_max": float(row["image_psnr_max"]),
        }
        cell_rows.append(out)
        value_by_cell[(label, context)] = out

    mean_matrix = np.full((len(labels), len(contexts)), np.nan, dtype=float)
    std_matrix = np.full_like(mean_matrix, np.nan)
    var_matrix = np.full_like(mean_matrix, np.nan)
    for label_idx, label in enumerate(labels):
        for context_idx, context in enumerate(contexts):
            cell = value_by_cell.get((label, context))
            if cell is None:
                continue
            mean_matrix[label_idx, context_idx] = cell["image_psnr_mean"]
            std_matrix[label_idx, context_idx] = cell["image_psnr_std"]
            var_matrix[label_idx, context_idx] = cell["image_psnr_var"]

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
    write_csv(output_dir / "modifier_prompt_grid_class_context_psnr.csv", cell_rows, cell_fields)
    write_pivot(output_dir / "modifier_prompt_grid_class_context_mean_matrix.csv", labels, contexts, mean_matrix)
    write_pivot(output_dir / "modifier_prompt_grid_class_context_std_matrix.csv", labels, contexts, std_matrix)
    write_pivot(output_dir / "modifier_prompt_grid_class_context_var_matrix.csv", labels, contexts, var_matrix)

    mean_low = float(np.nanquantile(mean_matrix, 0.25))
    mean_high = float(np.nanquantile(mean_matrix, 0.75))
    std_mid = float(np.nanmedian(std_matrix))
    category_matrix = [["" for _ in contexts] for _ in labels]
    category_code = np.full_like(mean_matrix, np.nan)
    category_to_code = {"Hard": 0, "Mixed": 1, "Easy": 2}
    for i in range(len(labels)):
        for j in range(len(contexts)):
            mean_value = mean_matrix[i, j]
            std_value = std_matrix[i, j]
            if np.isnan(mean_value) or np.isnan(std_value):
                continue
            if mean_value >= mean_high and std_value <= std_mid:
                category = "Easy"
            elif mean_value <= mean_low and std_value <= std_mid:
                category = "Hard"
            else:
                category = "Mixed"
            category_matrix[i][j] = category
            category_code[i, j] = category_to_code[category]
    write_text_pivot(
        output_dir / "modifier_prompt_grid_class_context_category_matrix.csv",
        labels,
        contexts,
        category_matrix,
    )
    write_csv(
        output_dir / "modifier_prompt_grid_class_context_category_thresholds.csv",
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
    fig, axes = plt.subplots(1, 2, figsize=(15, max(7, 0.45 * len(labels))), constrained_layout=True)

    mean_im = axes[0].imshow(mean_matrix, cmap="viridis", aspect="auto")
    axes[0].set_title("Mean PSNR with std annotation")
    axes[0].set_xticks(np.arange(len(contexts)), context_labels, rotation=25, ha="right")
    axes[0].set_yticks(np.arange(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(contexts)):
            if np.isnan(mean_matrix[i, j]):
                continue
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
    axes[1].set_xticks(np.arange(len(contexts)), context_labels, rotation=25, ha="right")
    axes[1].set_yticks(np.arange(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(contexts)):
            if np.isnan(category_code[i, j]):
                continue
            category = category_matrix[i][j]
            axes[1].text(
                j,
                i,
                f"{category}\n{mean_matrix[i, j]:.1f}±{std_matrix[i, j]:.1f}",
                ha="center",
                va="center",
                color="white" if category in {"Hard"} else "black",
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

    png_path = output_dir / "modifier_prompt_grid_class_context_heatmap.png"
    pdf_path = output_dir / "modifier_prompt_grid_class_context_heatmap.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved heatmap: {png_path}")
    print(f"Saved heatmap: {pdf_path}")


if __name__ == "__main__":
    main()
