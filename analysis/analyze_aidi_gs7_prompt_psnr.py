#!/usr/bin/env python3
"""Analyze how prompt/sample id affects AIDI-GS7 seed PSNR."""

import csv
import json
import math
import os
from collections import defaultdict
from statistics import mean, stdev


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "aidi_gs7_seed_psnr")
DETAIL_CSV = os.path.join(RESULTS_DIR, "aidi_gs7_seed_psnr_detail.csv")
SAMPLE_CSV = os.path.join(RESULTS_DIR, "aidi_gs7_seed_psnr_by_sample.csv")


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(value):
    if value in (None, ""):
        return None
    return float(value)


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs)
    mean_x = mean(xs)
    mean_y = mean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def rank_values(values):
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs)
    return pearson(rank_values(list(xs)), rank_values(list(ys)))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def prompt_features(prompt):
    words = prompt.split()
    return {
        "prompt_char_len": len(prompt),
        "prompt_word_count": len(words),
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


def anova_components(detail_rows):
    values = []
    by_sample = defaultdict(list)
    by_seed = defaultdict(list)
    cell = {}

    for row in detail_rows:
        sample_id = int(row["sample_id"])
        seed = int(row["seed"])
        psnr = to_float(row["psnr"])
        if psnr is None:
            continue
        values.append(psnr)
        by_sample[sample_id].append(psnr)
        by_seed[seed].append(psnr)
        cell[(sample_id, seed)] = psnr

    grand_mean = mean(values)
    sample_means = {sample_id: mean(vs) for sample_id, vs in by_sample.items()}
    seed_means = {seed: mean(vs) for seed, vs in by_seed.items()}
    n_seed = len(seed_means)
    n_sample = len(sample_means)

    ss_total = sum((value - grand_mean) ** 2 for value in values)
    ss_sample = sum(len(by_sample[sample_id]) * (sample_mean - grand_mean) ** 2
                    for sample_id, sample_mean in sample_means.items())
    ss_seed = sum(len(by_seed[seed]) * (seed_mean - grand_mean) ** 2
                  for seed, seed_mean in seed_means.items())
    ss_residual = 0.0
    for (sample_id, seed), value in cell.items():
        fitted = sample_means[sample_id] + seed_means[seed] - grand_mean
        ss_residual += (value - fitted) ** 2

    df_sample = n_sample - 1
    df_seed = n_seed - 1
    df_residual = max((n_sample - 1) * (n_seed - 1), 1)
    ms_sample = ss_sample / df_sample
    ms_seed = ss_seed / df_seed
    ms_residual = ss_residual / df_residual

    return {
        "n_pairs": len(values),
        "n_samples": n_sample,
        "n_seeds": n_seed,
        "grand_mean": grand_mean,
        "total_std": stdev(values),
        "ss_total": ss_total,
        "ss_sample_prompt": ss_sample,
        "ss_seed": ss_seed,
        "ss_residual": ss_residual,
        "eta2_sample_prompt": ss_sample / ss_total if ss_total else None,
        "eta2_seed": ss_seed / ss_total if ss_total else None,
        "eta2_residual_additive": ss_residual / ss_total if ss_total else None,
        "f_sample_prompt": ms_sample / ms_residual if ms_residual else None,
        "f_seed": ms_seed / ms_residual if ms_residual else None,
    }


def main():
    detail_rows = load_csv(DETAIL_CSV)
    sample_rows = load_csv(SAMPLE_CSV)

    components = anova_components(detail_rows)

    enriched_samples = []
    for row in sample_rows:
        prompt = row["original_prompt"]
        enriched = dict(row)
        enriched["sample_id"] = int(row["sample_id"])
        enriched["n"] = int(row["n"])
        enriched["psnr_mean"] = to_float(row["psnr_mean"])
        enriched["psnr_std"] = to_float(row["psnr_std"])
        enriched["psnr_min"] = to_float(row["psnr_min"])
        enriched["psnr_max"] = to_float(row["psnr_max"])
        enriched.update(prompt_features(prompt))
        enriched["prompt_category"] = keyword_category(prompt)
        enriched_samples.append(enriched)

    correlations = {
        "prompt_word_count_vs_psnr_mean_pearson": pearson(
            [row["prompt_word_count"] for row in enriched_samples],
            [row["psnr_mean"] for row in enriched_samples],
        ),
        "prompt_word_count_vs_psnr_mean_spearman": spearman(
            [row["prompt_word_count"] for row in enriched_samples],
            [row["psnr_mean"] for row in enriched_samples],
        ),
        "prompt_char_len_vs_psnr_mean_pearson": pearson(
            [row["prompt_char_len"] for row in enriched_samples],
            [row["psnr_mean"] for row in enriched_samples],
        ),
        "prompt_comma_count_vs_psnr_mean_pearson": pearson(
            [row["prompt_comma_count"] for row in enriched_samples],
            [row["psnr_mean"] for row in enriched_samples],
        ),
        "prompt_word_count_vs_psnr_std_pearson": pearson(
            [row["prompt_word_count"] for row in enriched_samples],
            [row["psnr_std"] for row in enriched_samples],
        ),
    }

    by_category = defaultdict(list)
    for row in enriched_samples:
        by_category[row["prompt_category"]].append(row["psnr_mean"])

    category_rows = []
    for category, values in sorted(by_category.items(), key=lambda item: mean(item[1])):
        category_rows.append(
            {
                "prompt_category": category,
                "n": len(values),
                "psnr_mean": mean(values),
                "psnr_std": stdev(values) if len(values) > 1 else 0.0,
                "psnr_min": min(values),
                "psnr_max": max(values),
            }
        )

    worst = sorted(enriched_samples, key=lambda row: row["psnr_mean"])[:30]
    best = sorted(enriched_samples, key=lambda row: row["psnr_mean"], reverse=True)[:30]
    unstable = sorted(enriched_samples, key=lambda row: row["psnr_std"], reverse=True)[:30]

    sample_fields = [
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
    category_fields = ["prompt_category", "n", "psnr_mean", "psnr_std", "psnr_min", "psnr_max"]

    write_csv(os.path.join(RESULTS_DIR, "prompt_psnr_enriched_by_sample.csv"), enriched_samples, sample_fields)
    write_csv(os.path.join(RESULTS_DIR, "prompt_psnr_worst30.csv"), worst, sample_fields)
    write_csv(os.path.join(RESULTS_DIR, "prompt_psnr_best30.csv"), best, sample_fields)
    write_csv(os.path.join(RESULTS_DIR, "prompt_psnr_most_seed_sensitive30.csv"), unstable, sample_fields)
    write_csv(os.path.join(RESULTS_DIR, "prompt_psnr_by_keyword_category.csv"), category_rows, category_fields)

    analysis = {
        "variance_components": components,
        "text_feature_correlations": correlations,
        "category_summary": category_rows,
        "worst_10_sample_ids": [row["sample_id"] for row in worst[:10]],
        "best_10_sample_ids": [row["sample_id"] for row in best[:10]],
        "most_seed_sensitive_10_sample_ids": [row["sample_id"] for row in unstable[:10]],
    }
    write_json(os.path.join(RESULTS_DIR, "prompt_psnr_analysis_summary.json"), analysis)

    report_path = os.path.join(RESULTS_DIR, "prompt_psnr_analysis.md")
    with open(report_path, "w") as f:
        f.write("# Prompt Effect on AIDI-GS7 PSNR\n\n")
        f.write("PSNR is computed between generated images (`gen`) and reconstructed images (`rec`).\n\n")
        f.write("## Variance Decomposition\n\n")
        f.write("| Component | Share of total PSNR variance |\n")
        f.write("|---|---:|\n")
        f.write(f"| Prompt / sample id | {components['eta2_sample_prompt']:.4f} |\n")
        f.write(f"| Seed | {components['eta2_seed']:.4f} |\n")
        f.write(f"| Residual / prompt-seed interaction | {components['eta2_residual_additive']:.4f} |\n\n")
        f.write(f"Grand mean PSNR: {components['grand_mean']:.4f} ± {components['total_std']:.4f} over {components['n_pairs']} pairs.\n\n")
        f.write("## Text Feature Correlations\n\n")
        f.write("| Feature | Correlation with mean PSNR |\n")
        f.write("|---|---:|\n")
        f.write(f"| Prompt word count, Pearson | {correlations['prompt_word_count_vs_psnr_mean_pearson']:.4f} |\n")
        f.write(f"| Prompt word count, Spearman | {correlations['prompt_word_count_vs_psnr_mean_spearman']:.4f} |\n")
        f.write(f"| Prompt char length, Pearson | {correlations['prompt_char_len_vs_psnr_mean_pearson']:.4f} |\n")
        f.write(f"| Prompt comma count, Pearson | {correlations['prompt_comma_count_vs_psnr_mean_pearson']:.4f} |\n\n")
        f.write("## Worst Prompts by Mean PSNR\n\n")
        f.write("| sample_id | mean PSNR | std | prompt |\n")
        f.write("|---:|---:|---:|---|\n")
        for row in worst[:10]:
            f.write(f"| {row['sample_id']} | {row['psnr_mean']:.4f} | {row['psnr_std']:.4f} | {row['original_prompt']} |\n")
        f.write("\n## Best Prompts by Mean PSNR\n\n")
        f.write("| sample_id | mean PSNR | std | prompt |\n")
        f.write("|---:|---:|---:|---|\n")
        for row in best[:10]:
            f.write(f"| {row['sample_id']} | {row['psnr_mean']:.4f} | {row['psnr_std']:.4f} | {row['original_prompt']} |\n")
        f.write("\n## Most Seed-Sensitive Prompts\n\n")
        f.write("| sample_id | mean PSNR | std | min | max | prompt |\n")
        f.write("|---:|---:|---:|---:|---:|---|\n")
        for row in unstable[:10]:
            f.write(f"| {row['sample_id']} | {row['psnr_mean']:.4f} | {row['psnr_std']:.4f} | {row['psnr_min']:.4f} | {row['psnr_max']:.4f} | {row['original_prompt']} |\n")

    print(f"Saved prompt analysis to: {RESULTS_DIR}")
    print(f"Prompt/sample eta^2: {components['eta2_sample_prompt']:.4f}")
    print(f"Seed eta^2: {components['eta2_seed']:.4f}")
    print(f"Residual eta^2: {components['eta2_residual_additive']:.4f}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
