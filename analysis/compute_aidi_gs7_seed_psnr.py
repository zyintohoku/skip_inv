#!/usr/bin/env python3
"""Compute gen/rec PSNR for outputs/aidi_gs7_seed*."""

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from statistics import mean, stdev

import numpy as np
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
DEFAULT_MAPPING_FILE = os.path.join(PROJECT_ROOT, "PIE_bench", "mapping_file.json")
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "aidi_gs7_seed_psnr")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute PSNR for gen/rec image pairs in outputs/aidi_gs7_seed*."
    )
    parser.add_argument("--outputs_dir", default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--mapping_file", default=DEFAULT_MAPPING_FILE)
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def seed_sort_key(path):
    match = re.search(r"aidi_gs7_seed(\d+)$", os.path.basename(path))
    return int(match.group(1)) if match else os.path.basename(path)


def find_seed_dirs(outputs_dir):
    dirs = []
    for name in os.listdir(outputs_dir):
        path = os.path.join(outputs_dir, name)
        if os.path.isdir(path) and re.fullmatch(r"aidi_gs7_seed\d+", name):
            dirs.append(path)
    return sorted(dirs, key=seed_sort_key)


def load_prompts(mapping_file):
    with open(mapping_file, "r") as f:
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


def image_ids(seed_dir):
    ids = []
    for name in os.listdir(seed_dir):
        match = re.fullmatch(r"(\d+)gen\.png", name)
        if match:
            ids.append(int(match.group(1)))
    return sorted(ids)


def finite(value):
    return value if math.isfinite(value) else None


def summarize(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        if math.isfinite(row["psnr"]):
            grouped[row[key]].append(row["psnr"])

    summaries = []
    for group_key in sorted(grouped):
        values = grouped[group_key]
        summary = {
            key: group_key,
            "n": len(values),
            "psnr_mean": mean(values),
            "psnr_std": stdev(values) if len(values) > 1 else 0.0,
            "psnr_min": min(values),
            "psnr_max": max(values),
        }
        summaries.append(summary)
    return summaries


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    seed_dirs = find_seed_dirs(args.outputs_dir)
    if not seed_dirs:
        raise FileNotFoundError(f"No aidi_gs7_seed* directories found in {args.outputs_dir}")

    prompts = load_prompts(args.mapping_file)
    rows = []
    missing_pairs = []

    total_pairs = sum(len(image_ids(seed_dir)) for seed_dir in seed_dirs)
    with tqdm(total=total_pairs, desc="Computing PSNR") as progress:
        for seed_dir in seed_dirs:
            seed = seed_sort_key(seed_dir)
            for sample_id in image_ids(seed_dir):
                gen_path = os.path.join(seed_dir, f"{sample_id}gen.png")
                rec_path = os.path.join(seed_dir, f"{sample_id}rec.png")
                if not os.path.exists(rec_path):
                    missing_pairs.append({"seed": seed, "sample_id": sample_id, "missing": rec_path})
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
    sample_summary_by_id = {row["sample_id"]: row for row in sample_summary}
    for row in sample_summary:
        prompt_info = prompts.get(row["sample_id"], {})
        row["mapping_key"] = prompt_info.get("mapping_key", "")
        row["original_prompt"] = prompt_info.get("original_prompt", "")
        row["editing_prompt"] = prompt_info.get("editing_prompt", "")
        row["editing_instruction"] = prompt_info.get("editing_instruction", "")

    all_psnr = [row["psnr"] for row in rows if row["psnr"] is not None]
    summary = {
        "n_seed_dirs": len(seed_dirs),
        "seeds": [seed_sort_key(path) for path in seed_dirs],
        "n_pairs": len(rows),
        "n_missing_pairs": len(missing_pairs),
        "psnr_mean": mean(all_psnr) if all_psnr else None,
        "psnr_std": stdev(all_psnr) if len(all_psnr) > 1 else 0.0,
        "psnr_min": min(all_psnr) if all_psnr else None,
        "psnr_max": max(all_psnr) if all_psnr else None,
        "worst_sample_id_by_mean_psnr": min(sample_summary_by_id, key=lambda sid: sample_summary_by_id[sid]["psnr_mean"]) if sample_summary_by_id else None,
        "best_sample_id_by_mean_psnr": max(sample_summary_by_id, key=lambda sid: sample_summary_by_id[sid]["psnr_mean"]) if sample_summary_by_id else None,
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
    summary_fields = ["seed", "n", "psnr_mean", "psnr_std", "psnr_min", "psnr_max"]
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

    write_csv(os.path.join(args.results_dir, "aidi_gs7_seed_psnr_detail.csv"), rows, detail_fields)
    write_json(os.path.join(args.results_dir, "aidi_gs7_seed_psnr_detail.json"), rows)
    write_csv(os.path.join(args.results_dir, "aidi_gs7_seed_psnr_by_seed.csv"), seed_summary, summary_fields)
    write_csv(os.path.join(args.results_dir, "aidi_gs7_seed_psnr_by_sample.csv"), sample_summary, sample_fields)
    write_json(os.path.join(args.results_dir, "aidi_gs7_seed_psnr_summary.json"), summary)
    write_json(os.path.join(args.results_dir, "aidi_gs7_seed_psnr_missing_pairs.json"), missing_pairs)

    print(f"Saved {len(rows)} PSNR rows to: {args.results_dir}")
    print(f"PSNR mean/std: {summary['psnr_mean']:.4f} ± {summary['psnr_std']:.4f}")
    print(f"Missing pairs: {len(missing_pairs)}")


if __name__ == "__main__":
    main()
