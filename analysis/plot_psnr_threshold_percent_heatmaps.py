#!/usr/bin/env python3
"""Plot class x context heatmaps for percentage of seed runs above PSNR thresholds."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODIFIER_CONTEXT_ORDER = [
    ("bare", "bare"),
    ("wooden_fence", "on wooden fence"),
    ("wooden_table", "on wooden table"),
    ("tree_branch", "on tree branch"),
]
FIELD_CONTEXT_ORDER = [
    ("bare_class", "bare"),
    ("field_np", "in a field"),
    ("field", "sitting in field"),
    ("field_dandelions", "sitting in dandelions"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PSNR threshold-pass percentage heatmaps.")
    parser.add_argument("--detail_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--context_mode", choices=["modifier", "group"], required=True)
    parser.add_argument("--thresholds", default="20,50")
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_thresholds(spec: str) -> List[float]:
    out = []
    for token in spec.split(","):
        token = token.strip()
        if token:
            out.append(float(token))
    if not out:
        raise ValueError("No thresholds parsed")
    return out


def context_from_modifier_prompt(row: Dict[str, str]) -> str:
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


def write_matrix_csv(path: Path, labels: List[str], contexts: List[str], matrix: np.ndarray) -> None:
    rows = []
    for i, label in enumerate(labels):
        row = {"label": label}
        for j, context in enumerate(contexts):
            row[context] = f"{matrix[i, j]:.6f}"
        rows.append(row)
    write_csv(path, rows, ["label", *contexts])


def make_context(row: Dict[str, str], mode: str) -> str:
    if mode == "modifier":
        return context_from_modifier_prompt(row)
    return row.get("group", "")


def main() -> None:
    args = parse_args()
    detail_csv = Path(args.detail_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = parse_thresholds(args.thresholds)

    rows = read_rows(detail_csv)
    labels = []
    for row in rows:
        label = row.get("label", "")
        if label and label not in labels:
            labels.append(label)
    context_order: List[Tuple[str, str]] = MODIFIER_CONTEXT_ORDER if args.context_mode == "modifier" else FIELD_CONTEXT_ORDER
    contexts = [key for key, _ in context_order]
    context_labels = [label for _, label in context_order]

    grouped = defaultdict(list)
    prompt_by_cell = {}
    for row in rows:
        label = row.get("label", "")
        context = make_context(row, args.context_mode)
        if not label or context not in contexts:
            continue
        grouped[(label, context)].append(float(row["image_psnr"]))
        prompt_by_cell.setdefault((label, context), row["prompt"])

    all_cell_rows = []
    for threshold in thresholds:
        matrix = np.full((len(labels), len(contexts)), np.nan)
        threshold_rows = []
        for i, label in enumerate(labels):
            for j, context in enumerate(contexts):
                values = grouped.get((label, context), [])
                if not values:
                    continue
                pct = 100.0 * sum(value > threshold for value in values) / len(values)
                matrix[i, j] = pct
                row = {
                    "threshold": threshold,
                    "label": label,
                    "context": context,
                    "prompt": prompt_by_cell.get((label, context), ""),
                    "n": len(values),
                    "percent_gt_threshold": pct,
                    "count_gt_threshold": sum(value > threshold for value in values),
                }
                threshold_rows.append(row)
                all_cell_rows.append(row)

        threshold_text = str(int(threshold)) if threshold.is_integer() else str(threshold).replace(".", "p")
        write_matrix_csv(
            output_dir / f"{args.name}_percent_gt_{threshold_text}_matrix.csv",
            labels,
            contexts,
            matrix,
        )
        write_csv(
            output_dir / f"{args.name}_percent_gt_{threshold_text}_cells.csv",
            threshold_rows,
            ["threshold", "label", "context", "prompt", "n", "percent_gt_threshold", "count_gt_threshold"],
        )

        for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "default"):
            try:
                plt.style.use(style)
                break
            except OSError:
                continue
        fig, ax = plt.subplots(figsize=(8.5, max(3.8, 0.45 * len(labels))), constrained_layout=True)
        im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
        ax.set_title(f"{args.name}: % of seeds with PSNR > {threshold:g}")
        ax.set_xticks(np.arange(len(contexts)), context_labels, rotation=25, ha="right")
        ax.set_yticks(np.arange(len(labels)), labels)
        for i in range(len(labels)):
            for j in range(len(contexts)):
                if np.isnan(matrix[i, j]):
                    continue
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.0f}%",
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > 55 else "black",
                    fontsize=9,
                )
        ax.set_xlabel("Context")
        ax.set_ylabel("Class label")
        ax.tick_params(axis="both", length=0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"% PSNR > {threshold:g}")
        png_path = output_dir / f"{args.name}_percent_gt_{threshold_text}_heatmap.png"
        pdf_path = output_dir / f"{args.name}_percent_gt_{threshold_text}_heatmap.pdf"
        fig.savefig(png_path, dpi=220)
        fig.savefig(pdf_path)
        plt.close(fig)
        print(f"Saved heatmap: {png_path}")
        print(f"Saved heatmap: {pdf_path}")

    write_csv(
        output_dir / f"{args.name}_threshold_percent_cells.csv",
        all_cell_rows,
        ["threshold", "label", "context", "prompt", "n", "percent_gt_threshold", "count_gt_threshold"],
    )


if __name__ == "__main__":
    main()
