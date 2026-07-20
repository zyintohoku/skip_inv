#!/usr/bin/env python3
"""Compute and plot FPI-GS7 reconstruction PSNR across saved AIDI seeds."""

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
DEFAULT_MAPPING_FILE = os.path.join(PROJECT_ROOT, "PIE_bench", "mapping_file.json")
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "fpi_gs7_seed_psnr")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute PSNR between saved AIDI generated images and FPI reconstructed images, "
            "then plot prompt mean PSNR vs seed std."
        )
    )
    parser.add_argument("--outputs_dir", default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--mapping_file", default=DEFAULT_MAPPING_FILE)
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--source_prefix", default="aidi_gs7_seed")
    parser.add_argument("--fpi_prefix", default="fpi_gs7_seed")
    parser.add_argument("--fpi_suffix", default="_from_saved_latents")
    return parser.parse_args()


def seed_sort_key_from_name(name, prefix, suffix=""):
    pattern = rf"{re.escape(prefix)}(\d+){re.escape(suffix)}$"
    match = re.fullmatch(pattern, name)
    return int(match.group(1)) if match else None


def find_seed_dirs(outputs_dir, prefix, suffix=""):
    dirs = {}
    for name in os.listdir(outputs_dir):
        path = os.path.join(outputs_dir, name)
        if not os.path.isdir(path):
            continue
        seed = seed_sort_key_from_name(name, prefix, suffix)
        if seed is not None:
            dirs[seed] = path
    return dict(sorted(dirs.items()))


def load_prompts(mapping_file):
    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    prompts = {}
    for sample_id, (mapping_key, item) in enumerate(mapping.items()):
        prompts[sample_id] = {
            "mapping_key": mapping_key,
            "original_prompt": item.get("original_prompt", "").replace("[", "").replace("]", ""),
            "editing_prompt": item.get("editing_prompt", ""),
            "editing_instruction": item.get("editing_instruction", ""),
        }
    return prompts


def load_rgb(path):
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32)


def compute_psnr(gen_path, rec_path):
    gen = load_rgb(gen_path)
    rec = load_rgb(rec_path)
    mse = float(np.mean((gen - rec) ** 2))
    if mse == 0.0:
        return math.inf, mse
    return float(20.0 * math.log10(255.0 / math.sqrt(mse))), mse


def rec_image_ids(seed_dir):
    ids = []
    for name in os.listdir(seed_dir):
        match = re.fullmatch(r"(\d+)rec\.png", name)
        if match:
            ids.append(int(match.group(1)))
    return sorted(ids)


def finite(value):
    return value if math.isfinite(value) else None


def summarize(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        if row["psnr"] is not None and math.isfinite(row["psnr"]):
            grouped[row[key]].append(row["psnr"])

    summaries = []
    for group_key in sorted(grouped):
        values = grouped[group_key]
        summaries.append(
            {
                key: group_key,
                "n": len(values),
                "psnr_mean": mean(values),
                "psnr_std": stdev(values) if len(values) > 1 else 0.0,
                "psnr_min": min(values),
                "psnr_max": max(values),
            }
        )
    return summaries


def prompt_features(prompt):
    words = prompt.split()
    return {
        "prompt_word_count": len(words),
        "prompt_char_len": len(prompt),
        "prompt_comma_count": prompt.count(","),
    }


def keyword_category(text):
    lower = text.lower()
    categories = [
        ("animal", ["cat", "dog", "horse", "bird", "animal", "bear", "zebra", "elephant", "cow", "sheep"]),
        ("person", ["person", "man", "woman", "boy", "girl", "people", "child", "human"]),
        ("vehicle", ["car", "bus", "truck", "bicycle", "bike", "motorcycle", "train", "airplane"]),
        ("food", ["cake", "pizza", "sandwich", "food", "fruit", "banana", "apple", "orange"]),
        ("indoor", ["room", "kitchen", "table", "chair", "bed", "sofa", "desk"]),
        ("outdoor", ["road", "street", "field", "beach", "mountain", "sky", "water", "grass"]),
    ]
    hits = [name for name, keys in categories if any(key in lower for key in keys)]
    return "|".join(hits) if hits else "other"


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def enrich_sample_rows(sample_rows):
    enriched = []
    for row in sample_rows:
        prompt = row["original_prompt"]
        out = dict(row)
        out.update(prompt_features(prompt))
        out["prompt_category"] = keyword_category(prompt)
        enriched.append(out)
    return enriched


def plot_mean_vs_std(rows, results_dir):
    means = np.array([float(row["psnr_mean"]) for row in rows])
    stds = np.array([float(row["psnr_std"]) for row in rows])
    median_mean = float(np.median(means))
    median_std = float(np.median(stds))

    worst_mean = sorted(rows, key=lambda row: float(row["psnr_mean"]))[:8]
    highest_std = sorted(rows, key=lambda row: float(row["psnr_std"]), reverse=True)[:8]
    label_rows = {row["sample_id"]: row for row in worst_mean + highest_std}

    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        try:
            plt.style.use(style)
            break
        except OSError:
            continue
    fig, ax = plt.subplots(figsize=(11, 7))
    scatter = ax.scatter(
        means,
        stds,
        c=means,
        cmap="viridis",
        s=42,
        alpha=0.78,
        edgecolors="white",
        linewidths=0.45,
    )
    ax.axvline(median_mean, color="#555555", linestyle="--", linewidth=1.1, label=f"median mean = {median_mean:.2f}")
    ax.axhline(median_std, color="#777777", linestyle=":", linewidth=1.2, label=f"median std = {median_std:.2f}")

    for row in label_rows.values():
        ax.annotate(
            str(row["sample_id"]),
            xy=(float(row["psnr_mean"]), float(row["psnr_std"])),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="#222222",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.82},
        )

    ax.set_title("FPI Prompt Reconstruction Quality vs Seed Sensitivity", fontsize=15, pad=12)
    ax.set_xlabel("Mean PSNR across 10 seeds (dB)")
    ax.set_ylabel("PSNR std across 10 seeds (dB)")
    ax.legend(loc="upper right", frameon=True)
    ax.text(
        0.01,
        0.98,
        "Each point is one prompt / sample_id\nLabels mark worst mean PSNR and highest seed sensitivity",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9},
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Mean PSNR (dB)")

    png_path = os.path.join(results_dir, "prompt_psnr_mean_vs_std.png")
    pdf_path = os.path.join(results_dir, "prompt_psnr_mean_vs_std.pdf")
    alias_png_path = os.path.join(results_dir, "prompt_psnr_mean_vs.png")
    fig.tight_layout()
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    fig.savefig(alias_png_path, dpi=220)
    plt.close(fig)
    return png_path, pdf_path, alias_png_path


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    source_dirs = find_seed_dirs(args.outputs_dir, args.source_prefix)
    fpi_dirs = find_seed_dirs(args.outputs_dir, args.fpi_prefix, args.fpi_suffix)
    seeds = sorted(set(source_dirs) & set(fpi_dirs))
    if not seeds:
        raise FileNotFoundError(
            f"No matched seeds found for {args.source_prefix}* and {args.fpi_prefix}*{args.fpi_suffix}"
        )

    prompts = load_prompts(args.mapping_file)
    rows = []
    missing_pairs = []
    total_pairs = sum(len(rec_image_ids(fpi_dirs[seed])) for seed in seeds)

    with tqdm(total=total_pairs, desc="Computing FPI PSNR") as progress:
        for seed in seeds:
            source_dir = source_dirs[seed]
            fpi_dir = fpi_dirs[seed]
            for sample_id in rec_image_ids(fpi_dir):
                gen_path = os.path.join(source_dir, f"{sample_id}gen.png")
                rec_path = os.path.join(fpi_dir, f"{sample_id}rec.png")
                if not os.path.exists(gen_path):
                    missing_pairs.append({"seed": seed, "sample_id": sample_id, "missing": gen_path})
                    progress.update(1)
                    continue

                psnr_value, mse_value = compute_psnr(gen_path, rec_path)
                prompt_info = prompts.get(sample_id, {})
                rows.append(
                    {
                        "seed": seed,
                        "sample_id": sample_id,
                        "mapping_key": prompt_info.get("mapping_key", ""),
                        "original_prompt": prompt_info.get("original_prompt", ""),
                        "editing_prompt": prompt_info.get("editing_prompt", ""),
                        "editing_instruction": prompt_info.get("editing_instruction", ""),
                        "psnr": finite(psnr_value),
                        "mse": mse_value,
                        "gen_path": os.path.relpath(gen_path, PROJECT_ROOT),
                        "rec_path": os.path.relpath(rec_path, PROJECT_ROOT),
                    }
                )
                progress.update(1)

    rows.sort(key=lambda row: (row["sample_id"], row["seed"]))
    seed_summary = summarize(rows, "seed")
    sample_summary = summarize(rows, "sample_id")

    for row in sample_summary:
        prompt_info = prompts.get(row["sample_id"], {})
        row["mapping_key"] = prompt_info.get("mapping_key", "")
        row["original_prompt"] = prompt_info.get("original_prompt", "")
        row["editing_prompt"] = prompt_info.get("editing_prompt", "")
        row["editing_instruction"] = prompt_info.get("editing_instruction", "")

    enriched_samples = enrich_sample_rows(sample_summary)
    worst = sorted(enriched_samples, key=lambda row: row["psnr_mean"])[:30]
    best = sorted(enriched_samples, key=lambda row: row["psnr_mean"], reverse=True)[:30]
    unstable = sorted(enriched_samples, key=lambda row: row["psnr_std"], reverse=True)[:30]

    all_psnr = [row["psnr"] for row in rows if row["psnr"] is not None]
    sample_summary_by_id = {row["sample_id"]: row for row in sample_summary}
    summary = {
        "n_seed_dirs": len(seeds),
        "seeds": seeds,
        "n_pairs": len(rows),
        "n_missing_pairs": len(missing_pairs),
        "psnr_mean": mean(all_psnr) if all_psnr else None,
        "psnr_std": stdev(all_psnr) if len(all_psnr) > 1 else 0.0,
        "psnr_min": min(all_psnr) if all_psnr else None,
        "psnr_max": max(all_psnr) if all_psnr else None,
        "worst_sample_id_by_mean_psnr": min(sample_summary_by_id, key=lambda sid: sample_summary_by_id[sid]["psnr_mean"])
        if sample_summary_by_id
        else None,
        "best_sample_id_by_mean_psnr": max(sample_summary_by_id, key=lambda sid: sample_summary_by_id[sid]["psnr_mean"])
        if sample_summary_by_id
        else None,
    }

    detail_fields = [
        "seed",
        "sample_id",
        "mapping_key",
        "original_prompt",
        "editing_prompt",
        "editing_instruction",
        "psnr",
        "mse",
        "gen_path",
        "rec_path",
    ]
    seed_fields = ["seed", "n", "psnr_mean", "psnr_std", "psnr_min", "psnr_max"]
    sample_fields = [
        "sample_id",
        "mapping_key",
        "original_prompt",
        "editing_prompt",
        "editing_instruction",
        "n",
        "psnr_mean",
        "psnr_std",
        "psnr_min",
        "psnr_max",
    ]
    enriched_fields = [
        "sample_id",
        "mapping_key",
        "original_prompt",
        "editing_prompt",
        "editing_instruction",
        "prompt_category",
        "prompt_word_count",
        "prompt_char_len",
        "prompt_comma_count",
        "n",
        "psnr_mean",
        "psnr_std",
        "psnr_min",
        "psnr_max",
    ]

    write_csv(os.path.join(args.results_dir, "fpi_gs7_seed_psnr_detail.csv"), rows, detail_fields)
    write_json(os.path.join(args.results_dir, "fpi_gs7_seed_psnr_detail.json"), rows)
    write_csv(os.path.join(args.results_dir, "fpi_gs7_seed_psnr_by_seed.csv"), seed_summary, seed_fields)
    write_csv(os.path.join(args.results_dir, "fpi_gs7_seed_psnr_by_sample.csv"), sample_summary, sample_fields)
    write_csv(os.path.join(args.results_dir, "prompt_psnr_enriched_by_sample.csv"), enriched_samples, enriched_fields)
    write_csv(os.path.join(args.results_dir, "prompt_psnr_worst30.csv"), worst, enriched_fields)
    write_csv(os.path.join(args.results_dir, "prompt_psnr_best30.csv"), best, enriched_fields)
    write_csv(os.path.join(args.results_dir, "prompt_psnr_most_seed_sensitive30.csv"), unstable, enriched_fields)
    write_json(os.path.join(args.results_dir, "fpi_gs7_seed_psnr_summary.json"), summary)
    write_json(os.path.join(args.results_dir, "fpi_gs7_seed_psnr_missing_pairs.json"), missing_pairs)

    plot_paths = plot_mean_vs_std(enriched_samples, args.results_dir)

    print(f"Saved {len(rows)} FPI PSNR rows to: {args.results_dir}")
    print(f"PSNR mean/std: {summary['psnr_mean']:.4f} +/- {summary['psnr_std']:.4f}")
    print(f"Missing pairs: {len(missing_pairs)}")
    for path in plot_paths:
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
