import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from scipy import stats
except Exception:  # pragma: no cover - fallback for environments without scipy
    stats = None


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


def build_matrix(rows: List[Dict[str, str]], metric: str) -> Tuple[List[int], List[int], np.ndarray]:
    sample_ids = sorted({int(row["sample_id"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    sample_index = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    seed_index = {seed: idx for idx, seed in enumerate(seeds)}

    matrix = np.full((len(sample_ids), len(seeds)), np.nan, dtype=np.float64)
    for row in rows:
        i = sample_index[int(row["sample_id"])]
        j = seed_index[int(row["seed"])]
        matrix[i, j] = float(row[metric])

    if np.isnan(matrix).any():
        raise ValueError(f"Missing values in {metric} matrix.")
    return sample_ids, seeds, matrix


def two_way_anova_without_replication(matrix: np.ndarray) -> Dict[str, float]:
    n_prompts, n_seeds = matrix.shape
    grand_mean = float(np.mean(matrix))
    prompt_means = np.mean(matrix, axis=1)
    seed_means = np.mean(matrix, axis=0)

    ss_total = float(np.sum((matrix - grand_mean) ** 2))
    ss_prompt = float(n_seeds * np.sum((prompt_means - grand_mean) ** 2))
    ss_seed = float(n_prompts * np.sum((seed_means - grand_mean) ** 2))
    ss_residual = float(ss_total - ss_prompt - ss_seed)

    df_prompt = n_prompts - 1
    df_seed = n_seeds - 1
    df_residual = (n_prompts - 1) * (n_seeds - 1)
    df_total = n_prompts * n_seeds - 1

    ms_prompt = ss_prompt / df_prompt
    ms_seed = ss_seed / df_seed
    ms_residual = ss_residual / df_residual

    f_prompt = ms_prompt / ms_residual
    f_seed = ms_seed / ms_residual
    p_prompt = float(stats.f.sf(f_prompt, df_prompt, df_residual)) if stats else float("nan")
    p_seed = float(stats.f.sf(f_seed, df_seed, df_residual)) if stats else float("nan")

    eta_prompt = ss_prompt / ss_total
    eta_seed = ss_seed / ss_total
    eta_residual = ss_residual / ss_total
    partial_eta_prompt = ss_prompt / (ss_prompt + ss_residual)
    partial_eta_seed = ss_seed / (ss_seed + ss_residual)
    omega_prompt = (ss_prompt - df_prompt * ms_residual) / (ss_total + ms_residual)
    omega_seed = (ss_seed - df_seed * ms_residual) / (ss_total + ms_residual)

    return {
        "n_prompts": n_prompts,
        "n_seeds": n_seeds,
        "grand_mean": grand_mean,
        "ss_total": ss_total,
        "ss_prompt": ss_prompt,
        "ss_seed": ss_seed,
        "ss_residual": ss_residual,
        "df_total": df_total,
        "df_prompt": df_prompt,
        "df_seed": df_seed,
        "df_residual": df_residual,
        "ms_prompt": ms_prompt,
        "ms_seed": ms_seed,
        "ms_residual": ms_residual,
        "f_prompt": f_prompt,
        "f_seed": f_seed,
        "p_prompt": p_prompt,
        "p_seed": p_seed,
        "eta_sq_prompt": eta_prompt,
        "eta_sq_seed": eta_seed,
        "eta_sq_residual": eta_residual,
        "partial_eta_sq_prompt": partial_eta_prompt,
        "partial_eta_sq_seed": partial_eta_seed,
        "omega_sq_prompt": omega_prompt,
        "omega_sq_seed": omega_seed,
        "prompt_to_seed_eta_sq_ratio": eta_prompt / eta_seed if eta_seed != 0 else float("inf"),
        "prompt_to_seed_ms_ratio": ms_prompt / ms_seed if ms_seed != 0 else float("inf"),
    }


def make_anova_rows(metric: str, label: str, stats_dict: Dict[str, float]) -> List[Dict]:
    return [
        {
            "metric": metric,
            "metric_label": label,
            "source": "prompt",
            "df": stats_dict["df_prompt"],
            "ss": stats_dict["ss_prompt"],
            "ms": stats_dict["ms_prompt"],
            "F": stats_dict["f_prompt"],
            "p_value": stats_dict["p_prompt"],
            "eta_sq": stats_dict["eta_sq_prompt"],
            "partial_eta_sq": stats_dict["partial_eta_sq_prompt"],
            "omega_sq": stats_dict["omega_sq_prompt"],
        },
        {
            "metric": metric,
            "metric_label": label,
            "source": "seed",
            "df": stats_dict["df_seed"],
            "ss": stats_dict["ss_seed"],
            "ms": stats_dict["ms_seed"],
            "F": stats_dict["f_seed"],
            "p_value": stats_dict["p_seed"],
            "eta_sq": stats_dict["eta_sq_seed"],
            "partial_eta_sq": stats_dict["partial_eta_sq_seed"],
            "omega_sq": stats_dict["omega_sq_seed"],
        },
        {
            "metric": metric,
            "metric_label": label,
            "source": "residual_interaction",
            "df": stats_dict["df_residual"],
            "ss": stats_dict["ss_residual"],
            "ms": stats_dict["ms_residual"],
            "F": "",
            "p_value": "",
            "eta_sq": stats_dict["eta_sq_residual"],
            "partial_eta_sq": "",
            "omega_sq": "",
        },
        {
            "metric": metric,
            "metric_label": label,
            "source": "total",
            "df": stats_dict["df_total"],
            "ss": stats_dict["ss_total"],
            "ms": "",
            "F": "",
            "p_value": "",
            "eta_sq": 1.0,
            "partial_eta_sq": "",
            "omega_sq": "",
        },
    ]


def write_markdown(path: Path, results: Dict[str, Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Two-Way ANOVA: Prompt vs Seed\n\n")
        f.write("Model: `value_ij = grand mean + prompt_i + seed_j + residual_ij`.\n\n")
        f.write(
            "Because each prompt-seed cell has one observation, the residual term includes "
            "prompt-seed interaction plus unexplained reconstruction variability.\n\n"
        )
        f.write("| metric | eta^2 prompt | eta^2 seed | eta^2 residual | F prompt | p prompt | F seed | p seed | prompt/seed eta ratio |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for metric, item in results.items():
            f.write(
                f"| {METRICS[metric]} | {item['eta_sq_prompt']:.6f} | {item['eta_sq_seed']:.6f} | "
                f"{item['eta_sq_residual']:.6f} | {item['f_prompt']:.4f} | {item['p_prompt']:.3e} | "
                f"{item['f_seed']:.4f} | {item['p_seed']:.3e} | {item['prompt_to_seed_eta_sq_ratio']:.2f} |\n"
            )

        f.write("\n## Calculation Details\n\n")
        f.write("- Build a balanced matrix `Y` with shape `700 prompts x 10 seeds`.\n")
        f.write("- Grand mean: mean of all 7000 values.\n")
        f.write("- Prompt SS: `n_seed * sum((prompt_mean_i - grand_mean)^2)`.\n")
        f.write("- Seed SS: `n_prompt * sum((seed_mean_j - grand_mean)^2)`.\n")
        f.write("- Residual SS: `SS_total - SS_prompt - SS_seed`.\n")
        f.write("- Degrees of freedom: prompt `699`, seed `9`, residual `(700-1)*(10-1)=6291`.\n")
        f.write("- Mean square: `MS = SS / df`.\n")
        f.write("- F statistic: `MS_effect / MS_residual`.\n")
        f.write("- Eta squared: `SS_effect / SS_total`, interpreted as variance explained by that factor.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-way ANOVA for prompt and seed effects.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="results/all_prompt_seed_clip_scores/all_seed_reconstruction_clip_scores.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/all_prompt_seed_clip_scores/prompt_seed_anova",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    rows = read_csv(Path(args.input_csv))
    output_dir = Path(args.output_dir)
    all_anova_rows = []
    results = {}

    for metric, label in METRICS.items():
        _, _, matrix = build_matrix(rows, metric)
        stats_dict = two_way_anova_without_replication(matrix)
        results[metric] = stats_dict
        all_anova_rows.extend(make_anova_rows(metric, label, stats_dict))

    write_csv(output_dir / "prompt_seed_two_way_anova.csv", all_anova_rows)
    write_json(output_dir / "prompt_seed_two_way_anova_summary.json", results)
    write_markdown(output_dir / "prompt_seed_two_way_anova_summary.md", results)
    print(f"saved two-way ANOVA results to: {output_dir}")


if __name__ == "__main__":
    main(parse_args())
