#!/usr/bin/env python3
"""Compute CLIP text/image scores for best/worst CFG=1 generations."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image


DEFAULT_ROOT = Path("results/fpi_gs7_seed_psnr/best_worst_top10_cfg1_init_latent_generation")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_clip_backend(args: argparse.Namespace, device: torch.device) -> Dict:
    if args.backend in ("auto", "openai"):
        try:
            import clip

            model, preprocess = clip.load(args.clip_model, device=device)
            model.eval()
            return {"name": "openai", "model": model, "preprocess": preprocess, "clip_module": clip}
        except Exception as exc:
            if args.backend == "openai":
                raise
            print(f"OpenAI clip backend unavailable, falling back to transformers: {exc}")

    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(args.hf_clip_model).to(device)
    processor = CLIPProcessor.from_pretrained(args.hf_clip_model)
    model.eval()
    return {"name": "transformers", "model": model, "processor": processor}


def encode_images(paths: List[str], backend: Dict, device: torch.device) -> torch.Tensor:
    images = [Image.open(path).convert("RGB") for path in paths]
    with torch.no_grad():
        if backend["name"] == "openai":
            tensor = torch.stack([backend["preprocess"](image) for image in images]).to(device)
            features = backend["model"].encode_image(tensor)
        else:
            inputs = backend["processor"](images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            features = backend["model"].get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.detach().cpu()


def encode_texts(texts: List[str], backend: Dict, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        if backend["name"] == "openai":
            tokens = backend["clip_module"].tokenize(texts, truncate=True).to(device)
            features = backend["model"].encode_text(tokens)
        else:
            inputs = backend["processor"](text=texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            features = backend["model"].get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.detach().cpu()


def build_rows(root: Path, cfg7_pattern: str) -> List[Dict]:
    manifest_path = root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest CSV: {manifest_path}")
    rows = []
    for row in read_csv(manifest_path):
        label = row["label"]
        rank = int(row["rank"])
        sample_id = int(row["sample_id"])
        seed = int(row["seed"])
        cfg1_path = Path(row["cfg1_image_path"])
        cfg7_path = Path(cfg7_pattern.format(seed=seed, sample_id=sample_id))
        if not cfg1_path.exists():
            raise FileNotFoundError(f"Missing CFG=1 image: {cfg1_path}")
        if not cfg7_path.exists():
            raise FileNotFoundError(f"Missing CFG=7 comparison image: {cfg7_path}")
        rows.append(
            {
                "label": label,
                "rank": rank,
                "sample_id": sample_id,
                "seed": seed,
                "prompt": row["prompt"],
                "prompt_psnr_mean": float(row["prompt_psnr_mean"]),
                "prompt_psnr_std": float(row["prompt_psnr_std"]) if row["prompt_psnr_std"] != "" else "",
                "cfg1_image_path": str(cfg1_path),
                "cfg7_image_path": str(cfg7_path),
            }
        )
    return sorted(rows, key=lambda item: (item["label"] != "best", item["rank"], item["seed"]))


def score_rows(rows: List[Dict], backend: Dict, device: torch.device, batch_size: int) -> List[Dict]:
    scored = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        cfg1_features = encode_images([row["cfg1_image_path"] for row in batch], backend, device)
        cfg7_features = encode_images([row["cfg7_image_path"] for row in batch], backend, device)
        text_features = encode_texts([row["prompt"] for row in batch], backend, device)

        text_scores = torch.sum(cfg1_features * text_features, dim=-1).tolist()
        image_scores = torch.sum(cfg1_features * cfg7_features, dim=-1).tolist()
        for row, text_score, image_score in zip(batch, text_scores, image_scores):
            out = dict(row)
            out["cfg1_clip_text_score"] = float(text_score)
            out["cfg1_vs_cfg7_clip_image_score"] = float(image_score)
            scored.append(out)
        print(f"scored {min(start + len(batch), len(rows))}/{len(rows)}")
    return scored


def mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def summarize_by_prompt(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, int], List[Dict]] = {}
    for row in rows:
        grouped.setdefault((row["label"], int(row["sample_id"])), []).append(row)

    summary = []
    for key in sorted(grouped, key=lambda item: (item[0] != "best", min(int(r["rank"]) for r in grouped[item]))):
        sample_rows = sorted(grouped[key], key=lambda row: int(row["seed"]))
        text_values = [float(row["cfg1_clip_text_score"]) for row in sample_rows]
        image_values = [float(row["cfg1_vs_cfg7_clip_image_score"]) for row in sample_rows]
        text_mean, text_std = mean_std(text_values)
        image_mean, image_std = mean_std(image_values)
        first = sample_rows[0]
        summary.append(
            {
                "label": first["label"],
                "rank": int(first["rank"]),
                "sample_id": int(first["sample_id"]),
                "n": len(sample_rows),
                "prompt": first["prompt"],
                "cfg1_clip_text_score_mean": text_mean,
                "cfg1_clip_text_score_std": text_std,
                "cfg1_vs_cfg7_clip_image_score_mean": image_mean,
                "cfg1_vs_cfg7_clip_image_score_std": image_std,
                "prompt_psnr_mean": first["prompt_psnr_mean"],
                "prompt_psnr_std": first["prompt_psnr_std"],
            }
        )
    return summary


def format_mean_std(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.4f} +/- {std_value:.4f}"


def write_markdown(path: Path, rows: List[Dict]) -> None:
    lines = [
        "# Best/Worst Top10 CFG=1 CLIP Scores",
        "",
        "Scores are normalized CLIP cosine similarities. Text score compares CFG=1 image to original prompt; image score compares CFG=1 image to the corresponding CFG=7 FPI image.",
        "",
        "| Label | Rank | Sample | Prompt | CFG1 text score | CFG1 vs CFG7 image score |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        prompt = str(row["prompt"]).replace("|", "\\|")
        lines.append(
            "| "
            f"{row['label']} | {row['rank']} | {row['sample_id']} | {prompt} | "
            f"{format_mean_std(row['cfg1_clip_text_score_mean'], row['cfg1_clip_text_score_std'])} | "
            f"{format_mean_std(row['cfg1_vs_cfg7_clip_image_score_mean'], row['cfg1_vs_cfg7_clip_image_score_std'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute prompt text CLIP score and CFG1-vs-CFG7 image CLIP score for best/worst top10."
    )
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument(
        "--cfg7_pattern",
        default="outputs/fpi_gs7_seed{seed}_from_saved_latents/{sample_id}rec.png",
        help="Format string with {seed} and {sample_id}.",
    )
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--backend", choices=["auto", "openai", "transformers"], default="auto")
    parser.add_argument("--clip_model", default="ViT-B/32")
    parser.add_argument("--hf_clip_model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "clip_scores"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(root, args.cfg7_pattern)
    device = torch.device(args.device)
    backend = load_clip_backend(args, device)
    scored = score_rows(rows, backend, device, args.batch_size)
    summary = summarize_by_prompt(scored)

    write_csv(output_dir / "best_worst_cfg1_clip_scores_detail.csv", scored)
    write_csv(output_dir / "best_worst_cfg1_clip_scores_by_prompt.csv", summary)
    write_markdown(output_dir / "best_worst_cfg1_clip_scores_by_prompt.md", summary)
    write_json(
        output_dir / "run_summary.json",
        {
            "root": str(root),
            "cfg7_pattern": args.cfg7_pattern,
            "backend": backend["name"],
            "clip_model": args.clip_model if backend["name"] == "openai" else args.hf_clip_model,
            "device": str(device),
            "batch_size": args.batch_size,
            "n_detail_rows": len(scored),
            "n_prompt_rows": len(summary),
        },
    )
    print(f"saved detail and prompt summary to: {output_dir}")


if __name__ == "__main__":
    main()
