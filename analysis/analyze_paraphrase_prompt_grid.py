#!/usr/bin/env python3
"""Analyze paraphrase prompt-grid FPI results, including partially finished runs."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PLOT_STYLE = {
    "font.size": 15,
    "axes.labelsize": 17,
    "axes.titlesize": 19,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
}
plt.rcParams.update(PLOT_STYLE)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_CSV = PROJECT_ROOT / "results" / "fpi_gs7_seed_psnr" / "paraphrase_prompt_grid" / "paraphrase_prompt_grid.csv"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_RUN_PREFIX = "paraphrase_prompt_grid_fpi_gs7_seed"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "fpi_gs7_seed_psnr" / "paraphrase_prompt_grid" / "analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze paraphrase prompt-grid PSNR behavior.")
    parser.add_argument("--prompt_csv", default=str(DEFAULT_PROMPT_CSV))
    parser.add_argument("--outputs_dir", default=str(DEFAULT_OUTPUTS_DIR))
    parser.add_argument("--run_prefix", default=DEFAULT_RUN_PREFIX)
    parser.add_argument("--results_dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--seeds", default="1-10")
    parser.add_argument("--base_id", default="", help="Optional base_id to plot; default plots all groups.")
    return parser.parse_args()


def parse_int_spec(spec: str) -> List[int]:
    values = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            values.extend(range(start, end + 1))
        else:
            values.append(int(token))
    if not values:
        raise ValueError(f"No seeds parsed from {spec}")
    return values


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32)


def image_psnr(gen_path: Path, rec_path: Path) -> Tuple[float, float]:
    gen = load_rgb(gen_path)
    rec = load_rgb(rec_path)
    mse = float(np.mean((gen - rec) ** 2))
    if mse == 0.0:
        return mse, math.inf
    return mse, float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def finite_float(value: str) -> Optional[float]:
    if value is None or value == "":
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def prompt_lookup(prompt_rows: List[Dict[str, str]]) -> Dict[int, Dict[str, str]]:
    return {idx: {**row, "prompt_id": str(idx)} for idx, row in enumerate(prompt_rows)}


def collect_rows(prompt_rows: List[Dict[str, str]], outputs_dir: Path, run_prefix: str, seeds: List[int]) -> List[Dict]:
    prompts_by_id = prompt_lookup(prompt_rows)
    rows = []
    seen = set()

    for seed in seeds:
        run_dir = outputs_dir / f"{run_prefix}{seed}"
        metrics_path = run_dir / "per_prompt_fpi_metrics.csv"
        if metrics_path.exists():
            for row in read_csv(metrics_path):
                prompt_id = int(row["prompt_id"])
                psnr = finite_float(row.get("image_psnr", ""))
                if psnr is None:
                    continue
                out = {
                    **prompts_by_id[prompt_id],
                    "seed": seed,
                    "prompt_id": prompt_id,
                    "image_psnr": psnr,
                    "image_mse": finite_float(row.get("image_mse", "")),
                    "gen_image_path": row.get("gen_image_path", ""),
                    "rec_image_path": row.get("rec_image_path", ""),
                    "source": "metrics_csv",
                }
                rows.append(out)
                seen.add((seed, prompt_id))

        for prompt_id, prompt_row in prompts_by_id.items():
            if (seed, prompt_id) in seen:
                continue
            gen_path = run_dir / f"{prompt_id}gen.png"
            rec_path = run_dir / f"{prompt_id}rec.png"
            if not gen_path.exists() or not rec_path.exists():
                continue
            mse, psnr = image_psnr(gen_path, rec_path)
            rows.append(
                {
                    **prompt_row,
                    "seed": seed,
                    "prompt_id": prompt_id,
                    "image_psnr": psnr,
                    "image_mse": mse,
                    "gen_image_path": str(gen_path.relative_to(PROJECT_ROOT)),
                    "rec_image_path": str(rec_path.relative_to(PROJECT_ROOT)),
                    "source": "computed_from_images",
                }
            )

    rows.sort(key=lambda row: (row["base_id"], int(row["paraphrase_id"]), int(row["seed"])))
    return rows


def summarize(rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    exemplar = {}
    for row in rows:
        key = (row["base_id"], int(row["paraphrase_id"]))
        grouped[key].append(float(row["image_psnr"]))
        exemplar.setdefault(key, row)

    summaries = []
    for key in sorted(grouped):
        values = grouped[key]
        sample = exemplar[key]
        summaries.append(
            {
                "base_id": sample["base_id"],
                "label": sample["label"],
                "paraphrase_id": int(sample["paraphrase_id"]),
                "prompt": sample["prompt"],
                "n": len(values),
                "image_psnr_mean": mean(values),
                "image_psnr_std": stdev(values) if len(values) > 1 else 0.0,
                "image_psnr_min": min(values),
                "image_psnr_max": max(values),
                "image_psnr_range": max(values) - min(values),
                "percent_gt_20": 100.0 * sum(value > 20 for value in values) / len(values),
                "percent_gt_50": 100.0 * sum(value > 50 for value in values) / len(values),
            }
        )
    return summaries


def short_prompt(text: str, max_chars: int = 58) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 1] + "..."


GROUP_EFFECT_FIELDS = [
    "base_id",
    "n_paraphrases",
    "n_pairs",
    "paraphrase_mean_min",
    "paraphrase_mean_max",
    "paraphrase_mean_range",
    "paraphrase_mean_std",
    "seed_mean_std",
    "overall_psnr_mean",
    "overall_psnr_std",
]


def group_effect_summary(base_id: str, rows: List[Dict], summaries: List[Dict]) -> Dict:
    group_summaries = [row for row in summaries if row["base_id"] == base_id]
    values_by_para = defaultdict(list)
    values_by_seed = defaultdict(list)
    for row in rows:
        if row["base_id"] != base_id:
            continue
        values_by_para[int(row["paraphrase_id"])].append(float(row["image_psnr"]))
        values_by_seed[int(row["seed"])].append(float(row["image_psnr"]))

    para_means = [row["image_psnr_mean"] for row in group_summaries]
    seed_means = [mean(values) for values in values_by_seed.values()]
    all_values = [value for values in values_by_para.values() for value in values]
    return {
        "base_id": base_id,
        "n_paraphrases": len(group_summaries),
        "n_pairs": len(all_values),
        "paraphrase_mean_min": min(para_means),
        "paraphrase_mean_max": max(para_means),
        "paraphrase_mean_range": max(para_means) - min(para_means),
        "paraphrase_mean_std": stdev(para_means) if len(para_means) > 1 else 0.0,
        "seed_mean_std": stdev(seed_means) if len(seed_means) > 1 else 0.0,
        "overall_psnr_mean": mean(all_values),
        "overall_psnr_std": stdev(all_values) if len(all_values) > 1 else 0.0,
    }


def write_group_summary(results_dir: Path, base_id: str, rows: List[Dict], summaries: List[Dict]) -> Dict:
    out = group_effect_summary(base_id, rows, summaries)
    write_csv(
        results_dir / f"{base_id}_group_effect_summary.csv",
        [out],
        GROUP_EFFECT_FIELDS,
    )
    return out


def plot_group_effect_comparison(results_dir: Path, effect_rows: List[Dict]) -> None:
    if not effect_rows:
        return

    ordered = sorted(effect_rows, key=lambda row: row["paraphrase_mean_range"], reverse=True)
    labels = [row["base_id"] for row in ordered]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.bar(
        x - width / 2,
        [row["paraphrase_mean_std"] for row in ordered],
        width,
        label="std of paraphrase means",
        color="#4c78a8",
    )
    ax.bar(
        x + width / 2,
        [row["seed_mean_std"] for row in ordered],
        width,
        label="std of seed means",
        color="#f58518",
    )
    for idx, row in enumerate(ordered):
        ax.text(
            idx,
            max(row["paraphrase_mean_std"], row["seed_mean_std"]) + 0.4,
            f"range={row['paraphrase_mean_range']:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_title("Paraphrase effect vs seed effect")
    ax.set_ylabel("PSNR std")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(results_dir / "paraphrase_effect_vs_seed_effect.png", dpi=220)
    fig.savefig(results_dir / "paraphrase_effect_vs_seed_effect.pdf")
    plt.close(fig)


def plot_combined_stripplot(
    results_dir: Path,
    base_ids: List[str],
    rows: List[Dict],
    summaries: List[Dict],
) -> None:
    if not base_ids:
        return

    fig, axes = plt.subplots(
        len(base_ids),
        1,
        figsize=(10.8, max(5.8, 3.7 * len(base_ids))),
        sharex=True,
        constrained_layout=True,
    )
    if len(base_ids) == 1:
        axes = [axes]

    rng = np.random.default_rng(0)
    all_values = [float(row["image_psnr"]) for row in rows if row["base_id"] in base_ids]
    x_min = max(0.0, min(all_values) - 2.0)
    x_max = max(all_values) + 2.0

    for ax, base_id in zip(axes, base_ids):
        group_rows = [row for row in rows if row["base_id"] == base_id]
        group_summaries = [row for row in summaries if row["base_id"] == base_id]
        ordered = sorted(group_summaries, key=lambda row: (row["image_psnr_mean"], row["paraphrase_id"]))
        value_by_para = defaultdict(list)
        for row in group_rows:
            value_by_para[int(row["paraphrase_id"])].append(float(row["image_psnr"]))

        y_labels = []
        for i, row in enumerate(ordered):
            para_id = int(row["paraphrase_id"])
            values = value_by_para[para_id]
            jitter = rng.uniform(-0.08, 0.08, size=len(values))
            ax.scatter(
                values,
                np.full(len(values), i) + jitter,
                s=36,
                alpha=0.72,
                color="#2f6fbb",
                label=None,
            )
            ax.errorbar(
                row["image_psnr_mean"],
                i,
                xerr=row["image_psnr_std"],
                fmt="D",
                color="#b02a2a",
                ecolor="#b02a2a",
                markersize=5.2,
                capsize=4,
            )
            y_labels.append(f"p{para_id}")

        ax.set_title(base_id, pad=10)
        ax.set_yticks(np.arange(len(ordered)), y_labels)
        ax.set_ylabel("Paraphrase")
        ax.grid(True, axis="x", alpha=0.25)
        ax.set_xlim(x_min, x_max)

    axes[-1].set_xlabel("PSNR")
    fig.savefig(results_dir / "combined_paraphrase_psnr_stripplot.png", dpi=220, bbox_inches="tight")
    fig.savefig(results_dir / "combined_paraphrase_psnr_stripplot.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_group(results_dir: Path, base_id: str, rows: List[Dict], summaries: List[Dict]) -> None:
    group_rows = [row for row in rows if row["base_id"] == base_id]
    group_summaries = [row for row in summaries if row["base_id"] == base_id]
    if not group_rows:
        return

    ordered = sorted(group_summaries, key=lambda row: (row["image_psnr_mean"], row["paraphrase_id"]))
    para_ids = [int(row["paraphrase_id"]) for row in ordered]
    seeds = sorted({int(row["seed"]) for row in group_rows})
    value_by_cell = {
        (int(row["paraphrase_id"]), int(row["seed"])): float(row["image_psnr"]) for row in group_rows
    }
    matrix = np.full((len(para_ids), len(seeds)), np.nan, dtype=float)
    for i, para_id in enumerate(para_ids):
        for j, seed in enumerate(seeds):
            matrix[i, j] = value_by_cell.get((para_id, seed), np.nan)

    ytick_labels = [
        f"p{row['paraphrase_id']}  {row['image_psnr_mean']:.1f}±{row['image_psnr_std']:.1f}  "
        f"{short_prompt(row['prompt'])}"
        for row in ordered
    ]

    fig, ax = plt.subplots(figsize=(10, max(5, 0.48 * len(para_ids))), constrained_layout=True)
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_title(f"{base_id}: PSNR by paraphrase and seed")
    ax.set_xticks(np.arange(len(seeds)), [f"s{seed}" for seed in seeds])
    ax.set_yticks(np.arange(len(para_ids)), ytick_labels)
    for i in range(len(para_ids)):
        for j in range(len(seeds)):
            if np.isnan(matrix[i, j]):
                continue
            rgba = im.cmap(im.norm(matrix[i, j]))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.1f}",
                ha="center",
                va="center",
                color="black" if luminance > 0.55 else "white",
                fontsize=7,
            )
    ax.set_xlabel("Seed")
    ax.set_ylabel("Paraphrase, sorted by mean PSNR")
    ax.tick_params(axis="both", length=0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="PSNR")
    fig.savefig(results_dir / f"{base_id}_paraphrase_seed_psnr_heatmap.png", dpi=220)
    fig.savefig(results_dir / f"{base_id}_paraphrase_seed_psnr_heatmap.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, max(4.8, 0.44 * len(para_ids))), constrained_layout=True)
    rng = np.random.default_rng(0)
    for i, row in enumerate(ordered):
        para_id = int(row["paraphrase_id"])
        values = [value_by_cell[(para_id, seed)] for seed in seeds if (para_id, seed) in value_by_cell]
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(values, np.full(len(values), i) + jitter, s=26, alpha=0.75, color="#2f6fbb", label=None)
        ax.errorbar(
            row["image_psnr_mean"],
            i,
            xerr=row["image_psnr_std"],
            fmt="D",
            color="#b02a2a",
            ecolor="#b02a2a",
            markersize=4,
            capsize=3,
        )
    ax.set_title(f"{base_id}: paraphrase effect vs seed spread")
    ax.set_xlabel("PSNR")
    ax.set_yticks(np.arange(len(para_ids)), ytick_labels)
    ax.set_ylabel("Paraphrase, sorted by mean PSNR")
    ax.grid(True, axis="x", alpha=0.25)
    ax.text(
        0.99,
        0.02,
        "blue dots: seeds; red diamond/errorbar: mean±std",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#333333",
    )
    fig.savefig(results_dir / f"{base_id}_paraphrase_psnr_stripplot.png", dpi=220)
    fig.savefig(results_dir / f"{base_id}_paraphrase_psnr_stripplot.pdf")
    plt.close(fig)

    write_group_summary(results_dir, base_id, rows, summaries)


def main() -> None:
    args = parse_args()
    prompt_csv = Path(args.prompt_csv)
    outputs_dir = Path(args.outputs_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_int_spec(args.seeds)

    prompt_rows = read_csv(prompt_csv)
    rows = collect_rows(prompt_rows, outputs_dir, args.run_prefix, seeds)
    summaries = summarize(rows)

    detail_fields = [
        "seed",
        "prompt_id",
        "group",
        "base_id",
        "label",
        "paraphrase_id",
        "base_prompt",
        "prompt",
        "image_psnr",
        "image_mse",
        "source",
        "gen_image_path",
        "rec_image_path",
    ]
    summary_fields = [
        "base_id",
        "label",
        "paraphrase_id",
        "prompt",
        "n",
        "image_psnr_mean",
        "image_psnr_std",
        "image_psnr_min",
        "image_psnr_max",
        "image_psnr_range",
        "percent_gt_20",
        "percent_gt_50",
    ]
    write_csv(results_dir / "paraphrase_prompt_grid_available_detail.csv", rows, detail_fields)
    write_csv(results_dir / "paraphrase_prompt_grid_available_by_paraphrase.csv", summaries, summary_fields)

    base_ids = sorted({row["base_id"] for row in rows})
    if args.base_id:
        base_ids = [args.base_id]
    effect_rows = []
    for base_id in base_ids:
        plot_group(results_dir, base_id, rows, summaries)
        effect_rows.append(group_effect_summary(base_id, rows, summaries))

    write_csv(results_dir / "paraphrase_effect_summary.csv", effect_rows, GROUP_EFFECT_FIELDS)
    plot_group_effect_comparison(results_dir, effect_rows)
    plot_combined_stripplot(results_dir, base_ids, rows, summaries)

    print(f"Collected {len(rows)} available prompt-seed rows")
    print(f"Saved analysis to: {results_dir}")


if __name__ == "__main__":
    main()
