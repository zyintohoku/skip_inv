import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image


KEY_FIELDS = ("seed", "sample_id")


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


def row_key(row: Dict[str, str]) -> Tuple[int, int]:
    return int(row["seed"]), int(row["sample_id"])


def load_done_rows(output_csv: Path) -> Dict[Tuple[int, int], Dict[str, str]]:
    if not output_csv.exists():
        return {}
    done = {}
    for row in read_csv(output_csv):
        if row.get("gen_rec_clip_image_score", "") == "":
            continue
        done[row_key(row)] = row
    return done


def append_rows(path: Path, rows: List[Dict], write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    mode = "w" if write_header else "a"
    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def validate_inputs(rows: Iterable[Dict[str, str]]) -> None:
    for row in rows:
        gen_path = Path(row["gen_path"])
        rec_path = Path(row["rec_path"])
        if not gen_path.exists():
            raise FileNotFoundError(f"Missing generated image: {gen_path}")
        if not rec_path.exists():
            raise FileNotFoundError(f"Missing reconstructed image: {rec_path}")


def load_clip_backend(args: argparse.Namespace, device: torch.device) -> Dict:
    if args.backend in ("auto", "openai"):
        try:
            import clip

            model, preprocess = clip.load(args.clip_model, device=device)
            model.eval()
            print(f"using OpenAI clip backend: {args.clip_model}")
            return {
                "name": "openai",
                "model": model,
                "preprocess": preprocess,
            }
        except Exception as exc:
            if args.backend == "openai":
                raise
            print(f"OpenAI clip backend unavailable, falling back to transformers: {exc}")

    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(args.hf_clip_model).to(device)
    processor = CLIPProcessor.from_pretrained(args.hf_clip_model)
    model.eval()
    print(f"using transformers CLIP backend: {args.hf_clip_model}")
    return {
        "name": "transformers",
        "model": model,
        "processor": processor,
    }


def encode_batch(paths: List[str], backend: Dict, device: torch.device) -> torch.Tensor:
    images = [Image.open(path).convert("RGB") for path in paths]
    with torch.no_grad():
        if backend["name"] == "openai":
            image_tensor = torch.stack([backend["preprocess"](image) for image in images]).to(device)
            features = backend["model"].encode_image(image_tensor)
        else:
            inputs = backend["processor"](images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            features = backend["model"].get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.detach().cpu()


def score_batch(batch: List[Dict[str, str]], backend: Dict, device: torch.device) -> List[Dict]:
    gen_features = encode_batch([row["gen_path"] for row in batch], backend, device)
    rec_features = encode_batch([row["rec_path"] for row in batch], backend, device)
    scores = torch.sum(gen_features * rec_features, dim=-1).tolist()

    out_rows = []
    for row, score in zip(batch, scores):
        out_rows.append(
            {
                "seed": int(row["seed"]),
                "sample_id": int(row["sample_id"]),
                "mapping_key": row["mapping_key"],
                "original_prompt": row["original_prompt"],
                "editing_prompt": row["editing_prompt"],
                "editing_instruction": row["editing_instruction"],
                "psnr": float(row["psnr"]),
                "mse": float(row["mse"]),
                "gen_rec_clip_image_score": float(score),
                "gen_path": row["gen_path"],
                "rec_path": row["rec_path"],
            }
        )
    return out_rows


def mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def summarize_by_seed(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[int, List[Dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["seed"]), []).append(row)

    summary = []
    for seed in sorted(grouped):
        values = [float(row["gen_rec_clip_image_score"]) for row in grouped[seed]]
        psnrs = [float(row["psnr"]) for row in grouped[seed]]
        clip_mean, clip_std = mean_std(values)
        psnr_mean, psnr_std = mean_std(psnrs)
        summary.append(
            {
                "seed": seed,
                "n": len(values),
                "clip_image_score_mean": clip_mean,
                "clip_image_score_std": clip_std,
                "psnr_mean": psnr_mean,
                "psnr_std": psnr_std,
            }
        )
    return summary


def summarize_by_prompt(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[int, List[Dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["sample_id"]), []).append(row)

    summary = []
    for sample_id in sorted(grouped):
        sample_rows = grouped[sample_id]
        values = [float(row["gen_rec_clip_image_score"]) for row in sample_rows]
        psnrs = [float(row["psnr"]) for row in sample_rows]
        clip_mean, clip_std = mean_std(values)
        psnr_mean, psnr_std = mean_std(psnrs)
        first = sample_rows[0]
        summary.append(
            {
                "sample_id": sample_id,
                "mapping_key": first["mapping_key"],
                "original_prompt": first["original_prompt"],
                "editing_prompt": first["editing_prompt"],
                "n": len(values),
                "clip_image_score_mean": clip_mean,
                "clip_image_score_std": clip_std,
                "clip_image_score_min": float(np.min(values)),
                "clip_image_score_max": float(np.max(values)),
                "psnr_mean": psnr_mean,
                "psnr_std": psnr_std,
                "psnr_min": float(np.min(psnrs)),
                "psnr_max": float(np.max(psnrs)),
            }
        )
    return summary


def finalize_outputs(output_csv: Path, output_dir: Path, run_info: Dict) -> None:
    rows = read_csv(output_csv)
    write_csv(output_dir / "all_seed_reconstruction_clip_by_seed.csv", summarize_by_seed(rows))
    write_csv(output_dir / "all_seed_reconstruction_clip_by_prompt.csv", summarize_by_prompt(rows))

    scores = [float(row["gen_rec_clip_image_score"]) for row in rows]
    psnrs = [float(row["psnr"]) for row in rows]
    run_info.update(
        {
            "num_rows": len(rows),
            "clip_image_score_mean": float(np.mean(scores)),
            "clip_image_score_std": float(np.std(scores)),
            "clip_image_score_min": float(np.min(scores)),
            "clip_image_score_max": float(np.max(scores)),
            "psnr_mean": float(np.mean(psnrs)),
            "psnr_std": float(np.std(psnrs)),
        }
    )
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)


def main(args: argparse.Namespace) -> None:
    start_time = time.time()
    detail_csv = Path(args.detail_csv)
    output_dir = Path(args.output_dir)
    output_csv = output_dir / "all_seed_reconstruction_clip_scores.csv"
    rows = read_csv(detail_csv)
    if args.expected_rows is not None and len(rows) != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} rows from {detail_csv}, got {len(rows)}")
    validate_inputs(rows)

    done = load_done_rows(output_csv) if args.resume else {}
    pending = [row for row in rows if row_key(row) not in done]
    print(f"total rows: {len(rows)}")
    print(f"already done: {len(done)}")
    print(f"pending: {len(pending)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume or not output_csv.exists():
        if output_csv.exists() and not args.overwrite:
            raise FileExistsError(f"{output_csv} exists. Use --resume or --overwrite.")
        if output_csv.exists() and args.overwrite:
            output_csv.unlink()
        write_header = True
    else:
        write_header = False

    device = torch.device(args.device)
    backend = load_clip_backend(args, device)

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        out_rows = score_batch(batch, backend, device)
        append_rows(output_csv, out_rows, write_header=write_header)
        write_header = False
        completed = len(done) + min(start + len(batch), len(pending))
        print(f"completed {completed}/{len(rows)}")

    run_info = {
        "detail_csv": str(detail_csv),
        "output_csv": str(output_csv),
        "backend": backend["name"],
        "clip_model": args.clip_model if backend["name"] == "openai" else args.hf_clip_model,
        "device": str(device),
        "batch_size": args.batch_size,
        "resume": args.resume,
        "elapsed_seconds": time.time() - start_time,
    }
    finalize_outputs(output_csv, output_dir, run_info)
    print(f"saved CLIP image scores to: {output_csv}")
    print(f"saved summaries to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute generated/reconstructed CLIP image scores for all FPI-GS7 seed image pairs."
    )
    parser.add_argument("--detail_csv", type=str, default="results/fpi_gs7_seed_psnr/fpi_gs7_seed_psnr_detail.csv")
    parser.add_argument("--output_dir", type=str, default="results/all_prompt_seed_clip_scores")
    parser.add_argument("--backend", choices=["auto", "openai", "transformers"], default="auto")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--hf_clip_model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--expected_rows", type=int, default=7000)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
