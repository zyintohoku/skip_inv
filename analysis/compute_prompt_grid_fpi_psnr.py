#!/usr/bin/env python3
"""Aggregate per-seed FPI gen-inv-rec metrics for an arbitrary prompt grid."""

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
    parser = argparse.ArgumentParser(description="Aggregate prompt-grid FPI PSNR results.")
    parser.add_argument("--outputs_dir", type=str, default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--run_prefix", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--name", type=str, default="prompt_grid_fpi")
    parser.add_argument("--seeds", type=str, default="1-10")
    return parser.parse_args()


def parse_int_spec(spec: str) -> List[int]:
    values: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" not in token:
            values.append(int(token))
            continue
        start_text, end_text = token.split("-", 1)
        start = int(start_text)
        end = int(end_text)
        if end < start:
            raise ValueError(f"Invalid range: {token}")
        values.extend(range(start, end + 1))
    if not values:
        raise ValueError(f"No seeds parsed from: {spec}")
    return values


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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


def finite_float(value: str) -> Optional[float]:
    if value is None or value == "":
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def sort_key(value):
    text = str(value)
    return (0, int(text)) if re.fullmatch(r"\d+", text) else (1, text)


def metadata_fields(rows: List[Dict]) -> List[str]:
    fields: List[str] = []
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
        out = {
            key: group_key,
            **{field: exemplar.get(field, "") for field in carry_fields if field != key},
            "n": len(values),
            f"{value_key}_mean": mean(values) if values else "",
            f"{value_key}_std": stdev(values) if len(values) > 1 else 0.0,
            f"{value_key}_min": min(values) if values else "",
            f"{value_key}_max": max(values) if values else "",
        }
        summaries.append(out)
    return summaries


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    results_dir = Path(args.results_dir)
    seeds = parse_int_spec(args.seeds)

    rows: List[Dict] = []
    missing = []
    for seed in seeds:
        run_dir = outputs_dir / f"{args.run_prefix}{seed}"
        metrics_path = run_dir / "per_prompt_fpi_metrics.csv"
        if not metrics_path.exists():
            missing.append({"seed": seed, "missing": str(metrics_path)})
            continue
        for row in read_csv(metrics_path):
            row["seed"] = int(row["seed"])
            row["prompt_id"] = int(row["prompt_id"])
            row["gen_image_path"] = relpath(row.get("gen_image_path", ""))
            row["rec_image_path"] = relpath(row.get("rec_image_path", ""))
            rows.append(row)

    rows.sort(key=lambda row: (int(row["prompt_id"]), int(row["seed"])))
    meta_fields = metadata_fields(rows)
    prompt_carry = ["prompt", *meta_fields]
    prompt_summary = summarize(rows, "prompt_id", "image_psnr", prompt_carry)
    seed_summary = summarize(rows, "seed", "image_psnr", [])
    label_summary = summarize(rows, "label", "image_psnr", ["label"]) if rows and "label" in rows[0] else []
    group_summary = summarize(rows, "group", "image_psnr", ["group"]) if rows and "group" in rows[0] else []

    all_psnr = [finite_float(row.get("image_psnr", "")) for row in rows]
    all_psnr = [value for value in all_psnr if value is not None]

    detail_fields = [
        "seed",
        "prompt_id",
        "prompt",
        *meta_fields,
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
    prompt_fields = [
        "prompt_id",
        "prompt",
        *meta_fields,
        "n",
        "image_psnr_mean",
        "image_psnr_std",
        "image_psnr_min",
        "image_psnr_max",
    ]
    seed_fields = ["seed", "n", "image_psnr_mean", "image_psnr_std", "image_psnr_min", "image_psnr_max"]
    label_fields = ["label", "n", "image_psnr_mean", "image_psnr_std", "image_psnr_min", "image_psnr_max"]
    group_fields = ["group", "n", "image_psnr_mean", "image_psnr_std", "image_psnr_min", "image_psnr_max"]

    name = args.name
    write_csv(results_dir / f"{name}_psnr_detail.csv", rows, detail_fields)
    write_csv(results_dir / f"{name}_psnr_by_prompt.csv", prompt_summary, prompt_fields)
    write_csv(results_dir / f"{name}_psnr_by_seed.csv", seed_summary, seed_fields)
    if label_summary:
        write_csv(results_dir / f"{name}_psnr_by_label.csv", label_summary, label_fields)
    if group_summary:
        write_csv(results_dir / f"{name}_psnr_by_group.csv", group_summary, group_fields)
    write_json(results_dir / f"{name}_psnr_detail.json", rows)
    write_json(results_dir / f"{name}_missing_runs.json", missing)
    write_json(
        results_dir / f"{name}_summary.json",
        {
            "seeds_requested": seeds,
            "n_seed_runs_found": len({int(row["seed"]) for row in rows}),
            "n_pairs": len(rows),
            "n_missing_runs": len(missing),
            "image_psnr_mean": mean(all_psnr) if all_psnr else None,
            "image_psnr_std": stdev(all_psnr) if len(all_psnr) > 1 else 0.0,
            "image_psnr_min": min(all_psnr) if all_psnr else None,
            "image_psnr_max": max(all_psnr) if all_psnr else None,
        },
    )

    print(f"Saved {len(rows)} rows to: {results_dir}")
    if missing:
        print(f"Missing runs: {len(missing)}")


if __name__ == "__main__":
    main()
