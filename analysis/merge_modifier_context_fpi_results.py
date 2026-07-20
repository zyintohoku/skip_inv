#!/usr/bin/env python3
"""Merge original and extra modifier-context FPI results and plot context heatmaps."""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle


PLOT_STYLE = {
    "font.size": 15,
    "axes.labelsize": 17,
    "axes.titlesize": 19,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "legend.title_fontsize": 14,
}
plt.rcParams.update(PLOT_STYLE)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "fpi_gs7_seed_psnr" / "modifier_prompt_grid_fpi"
CONTEXT_ORDER = [
    ("bare", "bare"),
    ("wooden_fence", "wooden fence"),
    ("wooden_table", "wooden table"),
    ("tree_branch", "tree branch"),
    ("rock", "rock"),
    ("grass", "grass"),
    ("shallow_water", "shallow water"),
    ("snowy_field", "snowy field"),
    ("transparent_glass_jar", "glass jar"),
    ("streetlight_night", "streetlight night"),
]
SUMMARY_LABEL = "summary_mean_std"
SUMMARY_DISPLAY_LABEL = "mean±std"
METRIC_FIELDS = {
    "seed",
    "prompt_id",
    "prompt",
    "method",
    "guidance_scale",
    "num_of_ddim_steps",
    "image_psnr",
    "image_mse",
    "gen_rec_latent_mse",
    "init_inv_latent_mse",
    "inversion_time",
    "inversion_final_loss",
    "inversion_mean_loss",
    "gen_image_path",
    "rec_image_path",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge original and extra modifier-context FPI outputs.")
    parser.add_argument("--results_dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_RESULTS_DIR / "merged_contexts"))
    parser.add_argument("--name", default="modifier_prompt_grid_merged_context_fpi")
    parser.add_argument("--thresholds", default="20,30,40,50")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def finite_float(value: str) -> Optional[float]:
    if value is None or value == "":
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def context_from_row(row: Dict[str, str]) -> str:
    context = row.get("context", "")
    if context:
        return context
    prompt = row["prompt"]
    if row.get("group") == "bare_class":
        return "bare"
    if prompt.endswith("on a wooden fence"):
        return "wooden_fence"
    if prompt.endswith("on a wooden table"):
        return "wooden_table"
    if prompt.endswith("on a tree branch"):
        return "tree_branch"
    raise ValueError(f"Cannot infer context for prompt: {prompt}")


def relpath(path_text: str) -> str:
    if not path_text:
        return ""
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        return os.path.relpath(path, PROJECT_ROOT)
    except ValueError:
        return str(path)


def sort_key(value):
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def parse_thresholds(spec: str) -> List[float]:
    out = []
    for token in spec.split(","):
        token = token.strip()
        if token:
            out.append(float(token))
    if not out:
        raise ValueError("No thresholds parsed")
    return out


def metadata_fields(rows: Iterable[Dict]) -> List[str]:
    fields = []
    for row in rows:
        for key in row:
            if key in METRIC_FIELDS or key in fields:
                continue
            fields.append(key)
    return fields


def summarize(rows: List[Dict], key: str, value_key: str, carry_fields: List[str]) -> List[Dict]:
    grouped = defaultdict(list)
    exemplars: Dict[str, Dict] = {}
    for row in rows:
        group_key = row.get(key, "")
        value = finite_float(row.get(value_key, ""))
        exemplars.setdefault(group_key, row)
        if value is not None:
            grouped[group_key].append(value)

    summaries = []
    for group_key in sorted(exemplars, key=sort_key):
        values = grouped.get(group_key, [])
        exemplar = exemplars[group_key]
        summaries.append(
            {
                key: group_key,
                **{field: exemplar.get(field, "") for field in carry_fields if field != key},
                "n": len(values),
                f"{value_key}_mean": mean(values) if values else "",
                f"{value_key}_std": stdev(values) if len(values) > 1 else 0.0,
                f"{value_key}_min": min(values) if values else "",
                f"{value_key}_max": max(values) if values else "",
            }
        )
    return summaries


def context_prompt_order(by_prompt_rows: List[Tuple[str, Dict[str, str]]]) -> Tuple[List[str], Dict[Tuple[str, str], int]]:
    labels = []
    for _, row in by_prompt_rows:
        label = row.get("label", "")
        if label and label not in labels:
            labels.append(label)

    prompt_id_by_cell = {}
    next_prompt_id = 0
    contexts = [key for key, _ in CONTEXT_ORDER]
    by_cell = {(row.get("label", ""), context_from_row(row)): row for _, row in by_prompt_rows}
    for label in labels:
        for context in contexts:
            if (label, context) not in by_cell:
                continue
            prompt_id_by_cell[(label, context)] = next_prompt_id
            next_prompt_id += 1
    return labels, prompt_id_by_cell


def merge_rows(
    original_detail: List[Dict[str, str]],
    extra_detail: List[Dict[str, str]],
    by_prompt_rows: List[Tuple[str, Dict[str, str]]],
) -> Tuple[List[str], List[Dict]]:
    labels, prompt_id_by_cell = context_prompt_order(by_prompt_rows)
    rows = []
    for result_set, detail_rows in (("original", original_detail), ("extra", extra_detail)):
        for row in detail_rows:
            label = row.get("label", "")
            context = context_from_row(row)
            prompt_id = prompt_id_by_cell[(label, context)]
            out = dict(row)
            out["source_result_set"] = result_set
            out["source_prompt_id"] = row.get("prompt_id", "")
            out["prompt_id"] = prompt_id
            out["context"] = context
            out["gen_image_path"] = relpath(row.get("gen_image_path", ""))
            out["rec_image_path"] = relpath(row.get("rec_image_path", ""))
            rows.append(out)
    rows.sort(key=lambda row: (int(row["prompt_id"]), int(row["seed"])))
    return labels, rows


def write_pivot(path: Path, labels: List[str], contexts: List[str], matrix: np.ndarray) -> None:
    rows = []
    for i, label in enumerate(labels):
        row = {"label": label}
        for j, context in enumerate(contexts):
            value = matrix[i, j]
            row[context] = "" if np.isnan(value) else f"{value:.6f}"
        rows.append(row)
    write_csv(path, rows, ["label", *contexts])


def write_pivot_with_summary(
    path: Path,
    labels: List[str],
    contexts: List[str],
    matrix: np.ndarray,
    subject_stats: Dict[str, Dict[str, float]],
    context_stats: Dict[str, Dict[str, float]],
    overall_stats: Dict[str, float],
    value_key: str,
) -> None:
    fieldnames = ["label", *contexts, f"subject_psnr_{value_key}", "subject_n"]
    rows = []
    for i, label in enumerate(labels):
        row = {"label": label}
        for j, context in enumerate(contexts):
            value = matrix[i, j]
            row[context] = "" if np.isnan(value) else f"{value:.6f}"
        row[f"subject_psnr_{value_key}"] = f"{subject_stats[label][value_key]:.6f}"
        row["subject_n"] = int(subject_stats[label]["n"])
        rows.append(row)

    summary_row = {"label": f"context_psnr_{value_key}"}
    for context in contexts:
        summary_row[context] = f"{context_stats[context][value_key]:.6f}"
    summary_row[f"subject_psnr_{value_key}"] = f"{overall_stats[value_key]:.6f}"
    summary_row["subject_n"] = int(overall_stats["n"])
    rows.append(summary_row)
    write_csv(path, rows, fieldnames)


def write_mean_std_pivot_with_summary(
    path: Path,
    labels: List[str],
    contexts: List[str],
    mean_matrix: np.ndarray,
    std_matrix: np.ndarray,
    subject_stats: Dict[str, Dict[str, float]],
    context_stats: Dict[str, Dict[str, float]],
    overall_stats: Dict[str, float],
) -> None:
    fieldnames = ["label", *contexts, "subject_psnr_mean_std", "subject_n"]
    rows = []
    for i, label in enumerate(labels):
        row = {"label": label}
        for j, context in enumerate(contexts):
            if np.isnan(mean_matrix[i, j]) or np.isnan(std_matrix[i, j]):
                row[context] = ""
            else:
                row[context] = f"{mean_matrix[i, j]:.3f}±{std_matrix[i, j]:.3f}"
        row["subject_psnr_mean_std"] = (
            f"{subject_stats[label]['mean']:.3f}±{subject_stats[label]['std']:.3f}"
        )
        row["subject_n"] = int(subject_stats[label]["n"])
        rows.append(row)

    summary_row = {"label": SUMMARY_LABEL}
    for context in contexts:
        summary_row[context] = f"{context_stats[context]['mean']:.3f}±{context_stats[context]['std']:.3f}"
    summary_row["subject_psnr_mean_std"] = f"{overall_stats['mean']:.3f}±{overall_stats['std']:.3f}"
    summary_row["subject_n"] = int(overall_stats["n"])
    rows.append(summary_row)
    write_csv(path, rows, fieldnames)


def write_text_pivot(path: Path, labels: List[str], contexts: List[str], matrix: List[List[str]]) -> None:
    rows = []
    for i, label in enumerate(labels):
        row = {"label": label}
        for j, context in enumerate(contexts):
            row[context] = matrix[i][j]
        rows.append(row)
    write_csv(path, rows, ["label", *contexts])


def build_cell_tables(
    rows: List[Dict],
    labels: List[str],
    contexts: List[str],
) -> Tuple[List[Dict], np.ndarray, np.ndarray, np.ndarray]:
    grouped = defaultdict(list)
    exemplar = {}
    for row in rows:
        value = finite_float(row.get("image_psnr", ""))
        if value is None:
            continue
        key = (row.get("label", ""), row.get("context", ""))
        grouped[key].append(value)
        exemplar.setdefault(key, row)

    mean_matrix = np.full((len(labels), len(contexts)), np.nan, dtype=float)
    std_matrix = np.full_like(mean_matrix, np.nan)
    var_matrix = np.full_like(mean_matrix, np.nan)
    cell_rows = []
    for i, label in enumerate(labels):
        for j, context in enumerate(contexts):
            values = grouped.get((label, context), [])
            if not values:
                continue
            std_value = stdev(values) if len(values) > 1 else 0.0
            mean_value = mean(values)
            mean_matrix[i, j] = mean_value
            std_matrix[i, j] = std_value
            var_matrix[i, j] = std_value**2
            sample = exemplar[(label, context)]
            cell_rows.append(
                {
                    "label": label,
                    "context": context,
                    "prompt": sample["prompt"],
                    "n": len(values),
                    "image_psnr_mean": mean_value,
                    "image_psnr_std": std_value,
                    "image_psnr_var": std_value**2,
                    "image_psnr_min": min(values),
                    "image_psnr_max": max(values),
                }
            )
    return cell_rows, mean_matrix, std_matrix, var_matrix


def psnr_stats(values: List[float]) -> Dict[str, float]:
    return {
        "n": len(values),
        "mean": mean(values) if values else math.nan,
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values) if values else math.nan,
        "max": max(values) if values else math.nan,
    }


def sort_labels_and_contexts(
    rows: List[Dict],
    labels: List[str],
) -> Tuple[List[str], List[str], List[str], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, float]]:
    values_by_label = defaultdict(list)
    values_by_context = defaultdict(list)
    all_values = []
    for row in rows:
        value = finite_float(row.get("image_psnr", ""))
        if value is None:
            continue
        values_by_label[row["label"]].append(value)
        values_by_context[row["context"]].append(value)
        all_values.append(value)

    subject_stats = {label: psnr_stats(values_by_label[label]) for label in labels}
    context_order = [key for key, _ in CONTEXT_ORDER if key in values_by_context]
    context_stats = {context: psnr_stats(values_by_context[context]) for context in context_order}
    overall_stats = psnr_stats(all_values)

    sorted_labels = sorted(labels, key=lambda label: (subject_stats[label]["mean"], label))
    sorted_contexts = sorted(
        context_order,
        key=lambda context: (context_stats[context]["mean"], context_order.index(context)),
    )
    label_by_context = {key: label for key, label in CONTEXT_ORDER}
    sorted_context_labels = [label_by_context[context] for context in sorted_contexts]
    return sorted_labels, sorted_contexts, sorted_context_labels, subject_stats, context_stats, overall_stats


def write_summary_tables(
    output_dir: Path,
    name: str,
    labels: List[str],
    contexts: List[str],
    subject_stats: Dict[str, Dict[str, float]],
    context_stats: Dict[str, Dict[str, float]],
) -> None:
    write_csv(
        output_dir / f"{name}_subject_psnr_summary.csv",
        [
            {
                "label": label,
                "n": int(subject_stats[label]["n"]),
                "image_psnr_mean": subject_stats[label]["mean"],
                "image_psnr_std": subject_stats[label]["std"],
                "image_psnr_min": subject_stats[label]["min"],
                "image_psnr_max": subject_stats[label]["max"],
            }
            for label in labels
        ],
        ["label", "n", "image_psnr_mean", "image_psnr_std", "image_psnr_min", "image_psnr_max"],
    )
    write_csv(
        output_dir / f"{name}_context_psnr_summary.csv",
        [
            {
                "context": context,
                "n": int(context_stats[context]["n"]),
                "image_psnr_mean": context_stats[context]["mean"],
                "image_psnr_std": context_stats[context]["std"],
                "image_psnr_min": context_stats[context]["min"],
                "image_psnr_max": context_stats[context]["max"],
            }
            for context in contexts
        ],
        ["context", "n", "image_psnr_mean", "image_psnr_std", "image_psnr_min", "image_psnr_max"],
    )


def plot_mean_heatmap(
    output_dir: Path,
    name: str,
    labels: List[str],
    contexts: List[str],
    context_labels: List[str],
    mean_matrix: np.ndarray,
    std_matrix: np.ndarray,
    subject_stats: Dict[str, Dict[str, float]],
    context_stats: Dict[str, Dict[str, float]],
    overall_stats: Dict[str, float],
) -> None:
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

    write_text_pivot(output_dir / f"{name}_class_context_category_matrix.csv", labels, contexts, category_matrix)
    write_csv(
        output_dir / f"{name}_class_context_category_thresholds.csv",
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
    extended_mean = np.full((len(labels) + 1, len(contexts) + 1), np.nan, dtype=float)
    extended_std = np.full_like(extended_mean, np.nan)
    extended_mean[: len(labels), : len(contexts)] = mean_matrix
    extended_std[: len(labels), : len(contexts)] = std_matrix
    for i, label in enumerate(labels):
        extended_mean[i, -1] = subject_stats[label]["mean"]
        extended_std[i, -1] = subject_stats[label]["std"]
    for j, context in enumerate(contexts):
        extended_mean[-1, j] = context_stats[context]["mean"]
        extended_std[-1, j] = context_stats[context]["std"]
    extended_mean[-1, -1] = overall_stats["mean"]
    extended_std[-1, -1] = overall_stats["std"]
    display_labels = [*labels, SUMMARY_DISPLAY_LABEL]
    display_context_labels = [*context_labels, SUMMARY_DISPLAY_LABEL]
    mean_center = float(np.nanmean(extended_mean))
    border_styles = {
        "Easy": {"edgecolor": "#111111", "linewidth": 2.8},
        "Hard": {"edgecolor": "#ef4444", "linewidth": 2.6},
    }

    fig_mean, ax_mean = plt.subplots(
        figsize=(16, max(8.0, 0.58 * (len(labels) + 1))),
        constrained_layout=True,
    )
    mean_panel_im = ax_mean.imshow(extended_mean, cmap="viridis", aspect="auto")
    ax_mean.set_title("Mean PSNR with std annotation", pad=12)
    ax_mean.set_xticks(np.arange(len(display_context_labels)), display_context_labels, rotation=35, ha="right")
    ax_mean.set_yticks(np.arange(len(display_labels)), display_labels)
    for i in range(len(display_labels)):
        for j in range(len(display_context_labels)):
            if np.isnan(extended_mean[i, j]):
                continue
            ax_mean.text(
                j,
                i,
                f"{extended_mean[i, j]:.1f}\n±{extended_std[i, j]:.1f}",
                ha="center",
                va="center",
                color="white" if extended_mean[i, j] < mean_center else "black",
                fontsize=8.7,
            )
    for i in range(len(labels)):
        for j in range(len(contexts)):
            category = category_matrix[i][j]
            if category not in border_styles:
                continue
            inset = 0.02
            ax_mean.add_patch(
                Rectangle(
                    (j - 0.5 + inset, i - 0.5 + inset),
                    1 - 2 * inset,
                    1 - 2 * inset,
                    fill=False,
                    **border_styles[category],
                )
            )
    ax_mean.axhline(len(labels) - 0.5, color="white", linewidth=2.0)
    ax_mean.axvline(len(contexts) - 0.5, color="white", linewidth=2.0)
    ax_mean.legend(
        handles=[
            Patch(facecolor="none", edgecolor=border_styles["Easy"]["edgecolor"], linewidth=2.6, label="Easy"),
            Patch(facecolor="none", edgecolor=border_styles["Hard"]["edgecolor"], linewidth=2.6, label="Hard"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.12),
        ncol=2,
        frameon=False,
    )
    ax_mean.set_xlabel("Context")
    ax_mean.set_ylabel("Subject label")
    ax_mean.tick_params(axis="both", length=0)
    fig_mean.colorbar(mean_panel_im, ax=ax_mean, fraction=0.046, pad=0.04, label="Mean PSNR")
    mean_panel_png = output_dir / f"{name}_class_context_heatmap_mean_psnr_panel.png"
    mean_panel_pdf = output_dir / f"{name}_class_context_heatmap_mean_psnr_panel.pdf"
    fig_mean.savefig(mean_panel_png, dpi=220, bbox_inches="tight")
    fig_mean.savefig(mean_panel_pdf, bbox_inches="tight")
    plt.close(fig_mean)

    fig, axes = plt.subplots(1, 2, figsize=(24, max(7.5, 0.5 * (len(labels) + 1))), constrained_layout=True)
    mean_im = axes[0].imshow(extended_mean, cmap="viridis", aspect="auto")
    axes[0].set_title("Mean PSNR with std annotation")
    axes[0].set_xticks(np.arange(len(display_context_labels)), display_context_labels, rotation=35, ha="right")
    axes[0].set_yticks(np.arange(len(display_labels)), display_labels)
    for i in range(len(display_labels)):
        for j in range(len(display_context_labels)):
            if np.isnan(extended_mean[i, j]):
                continue
            axes[0].text(
                j,
                i,
                f"{extended_mean[i, j]:.1f}\n±{extended_std[i, j]:.1f}",
                ha="center",
                va="center",
                color="white" if extended_mean[i, j] < mean_center else "black",
                fontsize=7.6,
            )
    for i in range(len(labels)):
        for j in range(len(contexts)):
            category = category_matrix[i][j]
            if category not in border_styles:
                continue
            inset = 0.02
            axes[0].add_patch(
                Rectangle(
                    (j - 0.5 + inset, i - 0.5 + inset),
                    1 - 2 * inset,
                    1 - 2 * inset,
                    fill=False,
                    **border_styles[category],
                )
            )
    axes[0].axhline(len(labels) - 0.5, color="white", linewidth=2.0)
    axes[0].axvline(len(contexts) - 0.5, color="white", linewidth=2.0)
    axes[0].legend(
        handles=[
            Patch(facecolor="none", edgecolor=border_styles["Easy"]["edgecolor"], linewidth=2.6, label="Easy"),
            Patch(facecolor="none", edgecolor=border_styles["Hard"]["edgecolor"], linewidth=2.6, label="Hard"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.12),
        ncol=2,
        frameon=False,
    )
    fig.colorbar(mean_im, ax=axes[0], fraction=0.046, pad=0.04, label="Mean PSNR")

    category_cmap = ListedColormap(["#4b3b8f", "#f0c44f", "#2f9e44"])
    cat_im = axes[1].imshow(category_code, cmap=category_cmap, vmin=0, vmax=2, aspect="auto")
    axes[1].set_title("Easy / Mixed / Hard classes")
    axes[1].set_xticks(np.arange(len(contexts)), context_labels, rotation=35, ha="right")
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
                color="white" if category == "Hard" else "black",
                fontsize=7.2,
            )
    cbar = fig.colorbar(cat_im, ax=axes[1], fraction=0.046, pad=0.04, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Hard", "Mixed", "Easy"])
    cbar.set_label(f"thresholds: mean q25={mean_low:.1f}, q75={mean_high:.1f}; std median={std_mid:.1f}")

    for ax in axes:
        ax.set_xlabel("Context")
        ax.set_ylabel("Subject label")
        ax.tick_params(axis="both", length=0)

    png_path = output_dir / f"{name}_class_context_heatmap.png"
    pdf_path = output_dir / f"{name}_class_context_heatmap.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved heatmap: {png_path}")
    print(f"Saved heatmap: {pdf_path}")
    print(f"Saved heatmap panel: {mean_panel_png}")
    print(f"Saved heatmap panel: {mean_panel_pdf}")


def plot_threshold_heatmaps(
    output_dir: Path,
    name: str,
    rows: List[Dict],
    labels: List[str],
    contexts: List[str],
    context_labels: List[str],
    thresholds: List[float],
) -> None:
    grouped = defaultdict(list)
    prompt_by_cell = {}
    for row in rows:
        value = finite_float(row.get("image_psnr", ""))
        if value is None:
            continue
        key = (row.get("label", ""), row.get("context", ""))
        grouped[key].append(value)
        prompt_by_cell.setdefault(key, row.get("prompt", ""))

    all_cell_rows = []
    for threshold in thresholds:
        matrix = np.full((len(labels), len(contexts)), np.nan, dtype=float)
        threshold_rows = []
        for i, label in enumerate(labels):
            for j, context in enumerate(contexts):
                values = grouped.get((label, context), [])
                if not values:
                    continue
                count = sum(value > threshold for value in values)
                pct = 100.0 * count / len(values)
                matrix[i, j] = pct
                row = {
                    "threshold": threshold,
                    "label": label,
                    "context": context,
                    "prompt": prompt_by_cell.get((label, context), ""),
                    "n": len(values),
                    "percent_gt_threshold": pct,
                    "count_gt_threshold": count,
                }
                threshold_rows.append(row)
                all_cell_rows.append(row)

        threshold_text = str(int(threshold)) if threshold.is_integer() else str(threshold).replace(".", "p")
        write_pivot(output_dir / f"{name}_percent_gt_{threshold_text}_matrix.csv", labels, contexts, matrix)
        write_csv(
            output_dir / f"{name}_percent_gt_{threshold_text}_cells.csv",
            threshold_rows,
            ["threshold", "label", "context", "prompt", "n", "percent_gt_threshold", "count_gt_threshold"],
        )

        fig, ax = plt.subplots(figsize=(14, max(4.8, 0.45 * len(labels))), constrained_layout=True)
        bounds = np.arange(0, 110, 10)
        cmap = plt.get_cmap("viridis", len(bounds) - 1)
        norm = BoundaryNorm(bounds, cmap.N)
        im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(f"{name}: % of seeds with PSNR > {threshold:g}")
        ax.set_xticks(np.arange(len(contexts)), context_labels, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(labels)), labels)
        for i in range(len(labels)):
            for j in range(len(contexts)):
                if np.isnan(matrix[i, j]):
                    continue
                rgba = cmap(norm(matrix[i, j]))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.0f}%",
                    ha="center",
                    va="center",
                    color="black" if luminance > 0.55 else "white",
                    fontsize=8,
                )
        ax.set_xlabel("Context")
        ax.set_ylabel("Class label")
        ax.tick_params(axis="both", length=0)
        fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            ticks=bounds,
            label=f"% PSNR > {threshold:g}",
        )
        png_path = output_dir / f"{name}_percent_gt_{threshold_text}_heatmap.png"
        pdf_path = output_dir / f"{name}_percent_gt_{threshold_text}_heatmap.pdf"
        fig.savefig(png_path, dpi=220)
        fig.savefig(pdf_path)
        plt.close(fig)
        print(f"Saved heatmap: {png_path}")
        print(f"Saved heatmap: {pdf_path}")

    write_csv(
        output_dir / f"{name}_threshold_percent_cells.csv",
        all_cell_rows,
        ["threshold", "label", "context", "prompt", "n", "percent_gt_threshold", "count_gt_threshold"],
    )


def write_readme(
    output_dir: Path,
    name: str,
    rows: List[Dict],
    cell_rows: List[Dict],
    summary: Dict,
    labels: List[str],
    contexts: List[str],
    subject_stats: Dict[str, Dict[str, float]],
    context_stats: Dict[str, Dict[str, float]],
) -> None:
    lowest = sorted(cell_rows, key=lambda row: float(row["image_psnr_mean"]))[:8]
    highest = sorted(cell_rows, key=lambda row: float(row["image_psnr_mean"]), reverse=True)[:8]

    lines = [
        "# Merged Modifier Context FPI Results",
        "",
        "This directory merges the original modifier prompt grid with the six extra contexts.",
        "",
        f"- name prefix: `{name}`",
        f"- total pairs: `{summary['n_pairs']}`",
        f"- mean PSNR: `{summary['image_psnr_mean']:.3f}`",
        f"- std PSNR: `{summary['image_psnr_std']:.3f}`",
        f"- min / max PSNR: `{summary['image_psnr_min']:.3f}` / `{summary['image_psnr_max']:.3f}`",
        "",
        "## Contexts",
        "",
    ]
    for context, label in CONTEXT_ORDER:
        lines.append(f"- `{context}`: {label}")
    lines.extend(["", "## Context Summary", ""])
    lines.append("| context | PSNR mean | PSNR std |")
    lines.append("|---|---:|---:|")
    for context in contexts:
        lines.append(
            f"| `{context}` | {context_stats[context]['mean']:.3f} | {context_stats[context]['std']:.3f} |"
        )

    lines.extend(["", "## Subject Summary", ""])
    lines.append("| subject | PSNR mean | PSNR std |")
    lines.append("|---|---:|---:|")
    for label in labels:
        lines.append(
            f"| `{label}` | {subject_stats[label]['mean']:.3f} | {subject_stats[label]['std']:.3f} |"
        )

    lines.extend(["", "## Lowest Mean Cells", ""])
    lines.append("| label | context | prompt | mean PSNR | std |")
    lines.append("|---|---|---|---:|---:|")
    for row in lowest:
        lines.append(
            f"| {row['label']} | `{row['context']}` | `{row['prompt']}` | "
            f"{float(row['image_psnr_mean']):.3f} | {float(row['image_psnr_std']):.3f} |"
        )

    lines.extend(["", "## Highest Mean Cells", ""])
    lines.append("| label | context | prompt | mean PSNR | std |")
    lines.append("|---|---|---|---:|---:|")
    for row in highest:
        lines.append(
            f"| {row['label']} | `{row['context']}` | `{row['prompt']}` | "
            f"{float(row['image_psnr_mean']):.3f} | {float(row['image_psnr_std']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Key Files",
            "",
            f"- `{name}_psnr_detail.csv`",
            f"- `{name}_psnr_by_prompt.csv`",
            f"- `{name}_class_context_mean_matrix.csv`",
            f"- `{name}_class_context_mean_std_matrix.csv`",
            f"- `{name}_subject_psnr_summary.csv`",
            f"- `{name}_context_psnr_summary.csv`",
            f"- `{name}_class_context_heatmap.png`",
            f"- `{name}_percent_gt_20_heatmap.png` through `{name}_percent_gt_50_heatmap.png`",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name
    thresholds = parse_thresholds(args.thresholds)

    original_by_prompt = read_csv(results_dir / "modifier_prompt_grid_fpi_psnr_by_prompt.csv")
    original_detail = read_csv(results_dir / "modifier_prompt_grid_fpi_psnr_detail.csv")
    extra_by_prompt = read_csv(results_dir / "modifier_extra_context_fpi_psnr_by_prompt.csv")
    extra_detail = read_csv(results_dir / "modifier_extra_context_fpi_psnr_detail.csv")
    by_prompt_rows = [("original", row) for row in original_by_prompt] + [("extra", row) for row in extra_by_prompt]

    labels, rows = merge_rows(original_detail, extra_detail, by_prompt_rows)
    (
        labels,
        contexts,
        context_labels,
        subject_stats,
        context_stats,
        overall_stats,
    ) = sort_labels_and_contexts(rows, labels)
    meta_fields = metadata_fields(rows)
    detail_fields = [
        "seed",
        "prompt_id",
        "source_result_set",
        "source_prompt_id",
        "prompt",
        *[field for field in meta_fields if field not in {"source_result_set", "source_prompt_id"}],
        "method",
        "guidance_scale",
        "num_of_ddim_steps",
        "image_psnr",
        "image_mse",
        "gen_rec_latent_mse",
        "init_inv_latent_mse",
        "inversion_time",
        "inversion_final_loss",
        "inversion_mean_loss",
        "gen_image_path",
        "rec_image_path",
    ]
    write_csv(output_dir / f"{name}_psnr_detail.csv", rows, detail_fields)
    write_json(output_dir / f"{name}_psnr_detail.json", rows)

    prompt_summary = summarize(rows, "prompt_id", "image_psnr", ["prompt", *meta_fields])
    seed_summary = summarize(rows, "seed", "image_psnr", [])
    label_summary = summarize(rows, "label", "image_psnr", ["label"])
    group_summary = summarize(rows, "group", "image_psnr", ["group"])
    write_csv(
        output_dir / f"{name}_psnr_by_prompt.csv",
        prompt_summary,
        [
            "prompt_id",
            "prompt",
            *meta_fields,
            "n",
            "image_psnr_mean",
            "image_psnr_std",
            "image_psnr_min",
            "image_psnr_max",
        ],
    )
    write_csv(
        output_dir / f"{name}_psnr_by_seed.csv",
        seed_summary,
        ["seed", "n", "image_psnr_mean", "image_psnr_std", "image_psnr_min", "image_psnr_max"],
    )
    write_csv(
        output_dir / f"{name}_psnr_by_label.csv",
        sorted(label_summary, key=lambda row: (float(row["image_psnr_mean"]), row["label"])),
        ["label", "n", "image_psnr_mean", "image_psnr_std", "image_psnr_min", "image_psnr_max"],
    )
    write_csv(
        output_dir / f"{name}_psnr_by_group.csv",
        group_summary,
        ["group", "n", "image_psnr_mean", "image_psnr_std", "image_psnr_min", "image_psnr_max"],
    )

    all_psnr = [finite_float(row.get("image_psnr", "")) for row in rows]
    all_psnr = [value for value in all_psnr if value is not None]
    summary = {
        "source_results": ["modifier_prompt_grid_fpi", "modifier_extra_context_fpi"],
        "n_pairs": len(rows),
        "n_prompts": len({int(row["prompt_id"]) for row in rows}),
        "n_labels": len(labels),
        "n_contexts": len(CONTEXT_ORDER),
        "image_psnr_mean": mean(all_psnr) if all_psnr else None,
        "image_psnr_std": stdev(all_psnr) if len(all_psnr) > 1 else 0.0,
        "image_psnr_min": min(all_psnr) if all_psnr else None,
        "image_psnr_max": max(all_psnr) if all_psnr else None,
    }
    write_json(output_dir / f"{name}_summary.json", summary)

    write_summary_tables(output_dir, name, labels, contexts, subject_stats, context_stats)
    cell_rows, mean_matrix, std_matrix, var_matrix = build_cell_tables(rows, labels, contexts)
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
    write_csv(output_dir / f"{name}_class_context_psnr.csv", cell_rows, cell_fields)
    write_pivot_with_summary(
        output_dir / f"{name}_class_context_mean_matrix.csv",
        labels,
        contexts,
        mean_matrix,
        subject_stats,
        context_stats,
        overall_stats,
        "mean",
    )
    write_pivot_with_summary(
        output_dir / f"{name}_class_context_std_matrix.csv",
        labels,
        contexts,
        std_matrix,
        subject_stats,
        context_stats,
        overall_stats,
        "std",
    )
    write_pivot(output_dir / f"{name}_class_context_var_matrix.csv", labels, contexts, var_matrix)
    write_mean_std_pivot_with_summary(
        output_dir / f"{name}_class_context_mean_std_matrix.csv",
        labels,
        contexts,
        mean_matrix,
        std_matrix,
        subject_stats,
        context_stats,
        overall_stats,
    )
    plot_mean_heatmap(
        output_dir,
        name,
        labels,
        contexts,
        context_labels,
        mean_matrix,
        std_matrix,
        subject_stats,
        context_stats,
        overall_stats,
    )
    plot_threshold_heatmaps(output_dir, name, rows, labels, contexts, context_labels, thresholds)
    write_readme(output_dir, name, rows, cell_rows, summary, labels, contexts, subject_stats, context_stats)

    print(f"Saved merged results to: {output_dir}")


if __name__ == "__main__":
    main()
