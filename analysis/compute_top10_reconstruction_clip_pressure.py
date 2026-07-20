import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


GROUP_SPECS = [
    ("best", "results/fpi_gs7_seed_psnr/prompt_psnr_best30.csv"),
    ("worst", "results/fpi_gs7_seed_psnr/prompt_psnr_worst30.csv"),
    ("most_seed_sensitive", "results/fpi_gs7_seed_psnr/prompt_psnr_most_seed_sensitive30.csv"),
]


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


def load_top10_prompts() -> List[Dict[str, str]]:
    selected = []
    for label, csv_path in GROUP_SPECS:
        rows = read_csv(Path(csv_path))[:10]
        for rank, row in enumerate(rows, start=1):
            selected.append(
                {
                    "label": label,
                    "prompt_rank": rank,
                    "sample_id": int(row["sample_id"]),
                    "mapping_key": row["mapping_key"],
                    "original_prompt": row["original_prompt"],
                    "editing_prompt": row["editing_prompt"],
                    "psnr_mean": float(row["psnr_mean"]),
                    "psnr_std": float(row["psnr_std"]),
                    "psnr_min": float(row["psnr_min"]),
                    "psnr_max": float(row["psnr_max"]),
                }
            )
    return selected


def load_fpi_detail(path: Path) -> Dict[Tuple[int, int], Dict[str, str]]:
    detail = {}
    for row in read_csv(path):
        detail[(int(row["seed"]), int(row["sample_id"]))] = row
    return detail


def load_pressure_metrics(root: Path, seeds: Iterable[int]) -> Dict[Tuple[int, int], Dict[str, str]]:
    metrics = {}
    for seed in seeds:
        path = root / f"seed_{seed:02d}" / "per_sample_prompt_pressure_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing pressure metrics: {path}")
        for row in read_csv(path):
            metrics[(seed, int(row["sample_id"]))] = row
    return metrics


def encode_images(
    image_paths: List[str],
    model,
    preprocess,
    device: torch.device,
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    features = {}
    unique_paths = sorted(set(image_paths))
    for start in range(0, len(unique_paths), batch_size):
        batch_paths = unique_paths[start : start + batch_size]
        images = [
            preprocess(Image.open(path).convert("RGB"))
            for path in batch_paths
        ]
        image_tensor = torch.stack(images).to(device)
        with torch.no_grad():
            batch_features = model.encode_image(image_tensor)
            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
        for path, feature in zip(batch_paths, batch_features):
            features[path] = feature.detach().cpu()
    return features


def compute_clip_scores(rows: List[Dict], args: argparse.Namespace) -> List[Dict]:
    device = torch.device(args.device)
    model, preprocess = clip.load(args.clip_model, device=device)
    model.eval()

    image_paths = []
    for row in rows:
        image_paths.append(row["gen_path"])
        image_paths.append(row["rec_path"])
    features = encode_images(image_paths, model, preprocess, device, args.batch_size)

    scored = []
    for row in rows:
        gen_feature = features[row["gen_path"]]
        rec_feature = features[row["rec_path"]]
        clip_score = float(torch.dot(gen_feature, rec_feature).item())
        out = dict(row)
        out["gen_rec_clip_image_score"] = clip_score
        scored.append(out)
    return scored


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) <= 0 or np.std(y) <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def summarize(rows: List[Dict]) -> List[Dict]:
    summary = []
    labels = sorted(set(row["label"] for row in rows))
    for label in ["all"] + labels:
        subset = rows if label == "all" else [row for row in rows if row["label"] == label]
        x = np.array([float(row["P_t_sum"]) for row in subset], dtype=np.float64)
        y = np.array([float(row["gen_rec_clip_image_score"]) for row in subset], dtype=np.float64)
        summary.append(
            {
                "label": label,
                "n": len(subset),
                "P_t_sum_mean": float(np.mean(x)),
                "P_t_sum_std": float(np.std(x)),
                "clip_image_score_mean": float(np.mean(y)),
                "clip_image_score_std": float(np.std(y)),
                "pearson_P_t_sum_vs_clip": pearson(x, y),
            }
        )
    return summary


def plot_pressure_vs_clip(rows: List[Dict], output_dir: Path) -> None:
    colors = {
        "best": "#2ca02c",
        "worst": "#d62728",
        "most_seed_sensitive": "#9467bd",
    }
    display = {
        "best": "best",
        "worst": "worst",
        "most_seed_sensitive": "seed_sensitive",
    }

    fig, ax = plt.subplots(figsize=(10.0, 6.2))
    for label in ["best", "worst", "most_seed_sensitive"]:
        subset = [row for row in rows if row["label"] == label]
        x = np.array([float(row["P_t_sum"]) for row in subset], dtype=np.float64)
        y = np.array([float(row["gen_rec_clip_image_score"]) for row in subset], dtype=np.float64)
        ax.scatter(
            x,
            y,
            s=34,
            alpha=0.78,
            color=colors[label],
            edgecolor="white",
            linewidth=0.45,
            label=display[label],
        )

    ax.set_xlabel("P_t_sum")
    ax.set_ylabel("CLIP image score")
    ax.set_title("P_t_sum vs CLIP image score")
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.75)
    ax.legend(title="label", frameon=False, fontsize=10, title_fontsize=11)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "P_t_sum_vs_clip_image_score.png", dpi=220)
    fig.savefig(output_dir / "P_t_sum_vs_clip_image_score.pdf")
    plt.close(fig)


def build_rows(args: argparse.Namespace) -> List[Dict]:
    top10 = load_top10_prompts()
    seeds = list(range(args.seed_start, args.seed_end + 1))
    fpi_detail = load_fpi_detail(Path(args.fpi_detail_csv))
    pressure_metrics = load_pressure_metrics(Path(args.pressure_root), seeds)

    rows = []
    for prompt_row in top10:
        sample_id = int(prompt_row["sample_id"])
        for seed in seeds:
            detail_key = (seed, sample_id)
            if detail_key not in fpi_detail:
                raise KeyError(f"Missing FPI detail for seed={seed}, sample_id={sample_id}")
            if detail_key not in pressure_metrics:
                raise KeyError(f"Missing pressure metrics for seed={seed}, sample_id={sample_id}")
            detail = fpi_detail[detail_key]
            pressure = pressure_metrics[detail_key]
            gen_path = detail["gen_path"]
            rec_path = detail["rec_path"]
            if not Path(gen_path).exists():
                raise FileNotFoundError(f"Missing generated image: {gen_path}")
            if not Path(rec_path).exists():
                raise FileNotFoundError(f"Missing reconstructed image: {rec_path}")
            rows.append(
                {
                    "label": prompt_row["label"],
                    "prompt_rank": prompt_row["prompt_rank"],
                    "seed": seed,
                    "sample_id": sample_id,
                    "mapping_key": prompt_row["mapping_key"],
                    "original_prompt": prompt_row["original_prompt"],
                    "editing_prompt": prompt_row["editing_prompt"],
                    "P_t_sum": float(pressure["P_raw_sum"]),
                    "R_t_sum": float(pressure["R_raw_sum"]),
                    "Delta_sum": float(pressure["Delta_raw_sum"]),
                    "PSNR": float(detail["psnr"]),
                    "MSE": float(detail["mse"]),
                    "gen_path": gen_path,
                    "rec_path": rec_path,
                    "prompt_psnr_mean": prompt_row["psnr_mean"],
                    "prompt_psnr_std": prompt_row["psnr_std"],
                    "prompt_psnr_min": prompt_row["psnr_min"],
                    "prompt_psnr_max": prompt_row["psnr_max"],
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute CLIP image scores for best/worst/most-sensitive top10 prompts across seeds."
    )
    parser.add_argument("--output_dir", type=str, default="results/top10_prompt_seed_clip_pressure")
    parser.add_argument("--fpi_detail_csv", type=str, default="results/fpi_gs7_seed_psnr/fpi_gs7_seed_psnr_detail.csv")
    parser.add_argument("--pressure_root", type=str, default="outputs/aidi_gs7_seed_generation_pressure")
    parser.add_argument("--seed_start", type=int, default=1)
    parser.add_argument("--seed_end", type=int, default=10)
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    rows = build_rows(args)
    if len(rows) != 300:
        raise ValueError(f"Expected 300 rows, got {len(rows)}")
    scored = compute_clip_scores(rows, args)
    write_csv(output_dir / "top10_seed_reconstruction_clip_scores.csv", scored)
    write_csv(output_dir / "top10_seed_reconstruction_clip_summary.csv", summarize(scored))
    plot_pressure_vs_clip(scored, output_dir)
    print(f"saved {len(scored)} CLIP-score rows and plots to: {output_dir}")


if __name__ == "__main__":
    main(parse_args())
