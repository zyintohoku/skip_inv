import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


PLOT_STYLE = {
    "font.size": 15,
    "axes.labelsize": 17,
    "axes.titlesize": 19,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
}
plt.rcParams.update(PLOT_STYLE)


METRICS = {
    "psnr": "PSNR",
    "gen_rec_clip_image_score": "CLIP image score",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_metric_matrices(rows: List[Dict[str, str]]) -> Tuple[List[int], List[int], Dict[str, np.ndarray]]:
    sample_ids = sorted({int(row["sample_id"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    sample_index = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    seed_index = {seed: idx for idx, seed in enumerate(seeds)}

    matrices = {
        key: np.full((len(sample_ids), len(seeds)), np.nan, dtype=np.float64)
        for key in METRICS
    }
    for row in rows:
        i = sample_index[int(row["sample_id"])]
        j = seed_index[int(row["seed"])]
        for key in METRICS:
            matrices[key][i, j] = float(row[key])

    for key, matrix in matrices.items():
        if np.isnan(matrix).any():
            raise ValueError(f"Missing values in matrix for {key}")
    return sample_ids, seeds, matrices


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def run_sampling(args: argparse.Namespace) -> Tuple[List[Dict], List[Dict], Dict]:
    rows = read_csv(Path(args.input_csv))
    sample_ids, seeds, matrices = build_metric_matrices(rows)
    if args.sample_size > len(sample_ids):
        raise ValueError(f"sample_size={args.sample_size} exceeds number of prompts={len(sample_ids)}")

    rng = random.Random(args.random_seed)
    sample_to_idx = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    round_rows = []
    per_object_rows = []

    for round_idx in range(args.num_rounds):
        sampled_ids = sorted(rng.sample(sample_ids, args.sample_size))
        sampled_indices = [sample_to_idx[sample_id] for sample_id in sampled_ids]

        round_row = {
            "round": round_idx,
            "sample_size": args.sample_size,
            "sample_ids": " ".join(str(sample_id) for sample_id in sampled_ids),
        }

        for metric_key, matrix in matrices.items():
            sub = matrix[sampled_indices, :]
            prompt_variances = np.var(sub, axis=1, ddof=args.ddof)
            seed_variances = np.var(sub, axis=0, ddof=args.ddof)

            round_row[f"{metric_key}_prompt_object_seed_variance_mean"] = float(np.mean(prompt_variances))
            round_row[f"{metric_key}_prompt_object_seed_variance_std"] = float(np.std(prompt_variances))
            round_row[f"{metric_key}_seed_object_prompt_variance_mean"] = float(np.mean(seed_variances))
            round_row[f"{metric_key}_seed_object_prompt_variance_std"] = float(np.std(seed_variances))
            round_row[f"{metric_key}_prompt_over_seed_variance_ratio"] = (
                float(np.mean(prompt_variances) / np.mean(seed_variances))
                if float(np.mean(seed_variances)) != 0
                else float("inf")
            )

            for sample_id, variance in zip(sampled_ids, prompt_variances):
                per_object_rows.append(
                    {
                        "round": round_idx,
                        "object_type": "prompt",
                        "metric": metric_key,
                        "object_id": sample_id,
                        "variance_across": "seeds",
                        "variance": float(variance),
                    }
                )
            for seed, variance in zip(seeds, seed_variances):
                per_object_rows.append(
                    {
                        "round": round_idx,
                        "object_type": "seed",
                        "metric": metric_key,
                        "object_id": seed,
                        "variance_across": "prompts",
                        "variance": float(variance),
                    }
                )

        round_rows.append(round_row)

    summary = {
        "input_csv": args.input_csv,
        "num_prompts_total": len(sample_ids),
        "num_seeds_total": len(seeds),
        "sample_size": args.sample_size,
        "num_rounds": args.num_rounds,
        "ddof": args.ddof,
        "random_seed": args.random_seed,
        "interpretation": {
            "prompt_object_seed_variance": "For each sampled prompt, variance across seeds; then averaged over sampled prompts and rounds.",
            "seed_object_prompt_variance": "For each seed, variance across sampled prompts; then averaged over seeds and rounds.",
            "larger_variance": "The larger averaged variance indicates the dimension associated with larger reconstruction variability for the sampled setting.",
        },
        "metrics": {},
    }
    for metric_key in METRICS:
        prompt_round_values = [
            row[f"{metric_key}_prompt_object_seed_variance_mean"]
            for row in round_rows
        ]
        seed_round_values = [
            row[f"{metric_key}_seed_object_prompt_variance_mean"]
            for row in round_rows
        ]
        ratios = [
            row[f"{metric_key}_prompt_over_seed_variance_ratio"]
            for row in round_rows
        ]
        prompt_summary = summarize(prompt_round_values)
        seed_summary = summarize(seed_round_values)
        ratio_summary = summarize(ratios)
        summary["metrics"][metric_key] = {
            "metric_label": METRICS[metric_key],
            "prompt_object_seed_variance_round_mean": prompt_summary,
            "seed_object_prompt_variance_round_mean": seed_summary,
            "prompt_over_seed_variance_ratio": ratio_summary,
            "larger_mean_variance": (
                "prompt_across_seeds"
                if prompt_summary["mean"] > seed_summary["mean"]
                else "seed_across_prompts"
            ),
        }

    return round_rows, per_object_rows, summary


def write_summary_csv(path: Path, summary: Dict) -> None:
    rows = []
    for metric_key, item in summary["metrics"].items():
        rows.append(
            {
                "metric": metric_key,
                "metric_label": item["metric_label"],
                "prompt_object_seed_variance_mean": item["prompt_object_seed_variance_round_mean"]["mean"],
                "prompt_object_seed_variance_std_over_rounds": item["prompt_object_seed_variance_round_mean"]["std"],
                "seed_object_prompt_variance_mean": item["seed_object_prompt_variance_round_mean"]["mean"],
                "seed_object_prompt_variance_std_over_rounds": item["seed_object_prompt_variance_round_mean"]["std"],
                "prompt_over_seed_variance_ratio_mean": item["prompt_over_seed_variance_ratio"]["mean"],
                "larger_mean_variance": item["larger_mean_variance"],
            }
        )
    write_csv(path, rows)


def plot_round_variance(round_rows: List[Dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rounds = np.array([int(row["round"]) for row in round_rows], dtype=np.int64)

    for metric_key, metric_label in METRICS.items():
        prompt_values = np.array(
            [float(row[f"{metric_key}_prompt_object_seed_variance_mean"]) for row in round_rows],
            dtype=np.float64,
        )
        seed_values = np.array(
            [float(row[f"{metric_key}_seed_object_prompt_variance_mean"]) for row in round_rows],
            dtype=np.float64,
        )

        fig, ax = plt.subplots(figsize=(9.5, 6.0))
        ax.plot(rounds, prompt_values, color="#4c78a8", linewidth=2.0, label="prompt objects: variance across seeds")
        ax.plot(rounds, seed_values, color="#f58518", linewidth=2.0, label="seed objects: variance across prompts")
        ax.axhline(float(np.mean(prompt_values)), color="#4c78a8", linestyle="--", linewidth=1.3)
        ax.axhline(float(np.mean(seed_values)), color="#f58518", linestyle="--", linewidth=1.3)
        ax.set_title(f"Prompt vs Seed Variance over Random Prompt Samples ({metric_label})", pad=12)
        ax.set_xlabel("Sampling round")
        ax.set_ylabel("Mean variance")
        ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.75)
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(output_dir / f"{metric_key}_prompt_seed_variance_sampling.png", dpi=220)
        fig.savefig(output_dir / f"{metric_key}_prompt_seed_variance_sampling.pdf")
        plt.close(fig)


def write_markdown_summary(path: Path, summary: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Prompt vs Seed Variance Sampling\n\n")
        f.write(f"- prompts total: {summary['num_prompts_total']}\n")
        f.write(f"- seeds total: {summary['num_seeds_total']}\n")
        f.write(f"- sampled prompts per round: {summary['sample_size']}\n")
        f.write(f"- rounds: {summary['num_rounds']}\n")
        f.write(f"- variance ddof: {summary['ddof']}\n")
        f.write(f"- random seed: {summary['random_seed']}\n\n")
        f.write("| metric | prompt object var across seeds | seed object var across prompts | ratio prompt/seed | larger |\n")
        f.write("| --- | ---: | ---: | ---: | --- |\n")
        for metric_key, item in summary["metrics"].items():
            prompt_mean = item["prompt_object_seed_variance_round_mean"]["mean"]
            seed_mean = item["seed_object_prompt_variance_round_mean"]["mean"]
            ratio = item["prompt_over_seed_variance_ratio"]["mean"]
            f.write(
                f"| {item['metric_label']} | {prompt_mean:.6g} | {seed_mean:.6g} | "
                f"{ratio:.6g} | {item['larger_mean_variance']} |\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample prompts and compare prompt-wise seed variance vs seed-wise prompt variance."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="results/all_prompt_seed_clip_scores/all_seed_reconstruction_clip_scores.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/all_prompt_seed_clip_scores/prompt_seed_variance_sampling",
    )
    parser.add_argument("--sample_size", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=100)
    parser.add_argument("--random_seed", type=int, default=20260512)
    parser.add_argument("--ddof", type=int, default=0, help="Variance delta degrees of freedom.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    round_rows, per_object_rows, summary = run_sampling(args)
    write_csv(output_dir / "prompt_seed_variance_sampling_rounds.csv", round_rows)
    write_csv(output_dir / "prompt_seed_variance_sampling_per_object.csv", per_object_rows)
    write_summary_csv(output_dir / "prompt_seed_variance_sampling_summary.csv", summary)
    write_json(output_dir / "prompt_seed_variance_sampling_summary.json", summary)
    write_markdown_summary(output_dir / "prompt_seed_variance_sampling_summary.md", summary)
    plot_round_variance(round_rows, output_dir / "plots")
    print(f"saved prompt/seed variance sampling results to: {output_dir}")


if __name__ == "__main__":
    main(parse_args())
