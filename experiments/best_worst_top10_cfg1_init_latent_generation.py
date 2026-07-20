#!/usr/bin/env python3
"""Generate CFG=1 images for FPI best/worst prompts from saved init latents."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch
from diffusers import DDIMScheduler
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.inv_methods import MyStableDiffusionPipeline  # noqa: E402


DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "fpi_gs7_seed_psnr"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "best_worst_top10_cfg1_init_latent_generation"
DEFAULT_RANKING_CSV = DEFAULT_RESULTS_DIR / "fpi_gs7_seed_psnr_by_sample.csv"
DEFAULT_SOURCE_PREFIX = PROJECT_ROOT / "outputs" / "aidi_gs7_seed"


def parse_int_set(text: str) -> list[int]:
    values = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(chunk))
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select best/worst top-K prompts from FPI-GS7 seed PSNR results, "
            "copy their saved 10-seed initial latents, and generate CFG=1 images."
        )
    )
    parser.add_argument("--ranking_csv", default=str(DEFAULT_RANKING_CSV))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--source_prefix", default=str(DEFAULT_SOURCE_PREFIX))
    parser.add_argument("--model_name", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--seeds", default="1-10")
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--num_of_ddim_steps", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", choices=["auto", "float32", "float16"], default="float32")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--prepare_only", action="store_true", help="Write prompts/latents/manifest without loading SD.")
    parser.add_argument("--disable_progress_bar", action="store_true")
    return parser.parse_args()


def read_ranked_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in ranking CSV: {path}")
    required = {"sample_id", "original_prompt", "psnr_mean"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Ranking CSV {path} is missing required columns: {sorted(missing)}")

    for row in rows:
        row["sample_id"] = int(row["sample_id"])
        row["psnr_mean"] = float(row["psnr_mean"])
        if "psnr_std" in row and row["psnr_std"] != "":
            row["psnr_std"] = float(row["psnr_std"])
    return rows


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def select_best_worst(rows: list[dict], top_k: int) -> list[dict]:
    best = sorted(rows, key=lambda row: row["psnr_mean"], reverse=True)[:top_k]
    worst = sorted(rows, key=lambda row: row["psnr_mean"])[:top_k]

    selected = []
    for label, label_rows in (("best", best), ("worst", worst)):
        for rank, row in enumerate(label_rows, start=1):
            out = dict(row)
            out["label"] = label
            out["rank"] = rank
            selected.append(out)
    return selected


def torch_dtype_from_arg(value: str):
    if value == "float16":
        return torch.float16
    if value == "float32":
        return torch.float32
    return None


def load_seed_latents(seed: int, source_prefix: Path) -> list[torch.Tensor]:
    path = Path(f"{source_prefix}{seed}") / "init_latents.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing saved init latents for seed {seed}: {path}")
    latents = torch.load(path, map_location="cpu")
    if not isinstance(latents, list):
        raise TypeError(f"Expected a list in {path}, got {type(latents)!r}")
    return latents


def latent_stats(latent: torch.Tensor) -> dict:
    latent = latent.detach().float().cpu()
    return {
        "latent_shape": "x".join(str(dim) for dim in latent.shape),
        "latent_mean": float(latent.mean().item()),
        "latent_std": float(latent.std().item()),
        "latent_l2_norm": float(torch.linalg.vector_norm(latent).item()),
    }


def make_scheduler() -> DDIMScheduler:
    return DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )


def load_pipeline(args: argparse.Namespace) -> MyStableDiffusionPipeline:
    dtype = torch_dtype_from_arg(args.torch_dtype)
    kwargs = {"scheduler": make_scheduler()}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    pipe = MyStableDiffusionPipeline.from_pretrained(args.model_name, **kwargs).to(args.device)
    pipe.set_progress_bar_config(disable=args.disable_progress_bar)
    return pipe


def sample_dir(output_dir: Path, label: str, sample_id: int) -> Path:
    return output_dir / label / f"sample_{sample_id:04d}"


def save_selected_prompts(output_dir: Path, selected: list[dict]) -> None:
    fields = [
        "label",
        "rank",
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
    write_csv(output_dir / "selected_prompts.csv", selected, fields)
    for label in ("best", "worst"):
        rows = [row for row in selected if row["label"] == label]
        write_csv(output_dir / f"{label}_top_prompts.csv", rows, fields)


def prepare_latents(args: argparse.Namespace, selected: list[dict], seeds: list[int]) -> list[dict]:
    output_dir = Path(args.output_dir)
    seed_cache: dict[int, list[torch.Tensor]] = {}
    manifest_rows = []

    for row in selected:
        label = row["label"]
        sid = int(row["sample_id"])
        prompt = row["original_prompt"]
        out_dir = sample_dir(output_dir, label, sid)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            out_dir / "prompt.json",
            {
                "label": label,
                "rank": row["rank"],
                "sample_id": sid,
                "mapping_key": row.get("mapping_key", ""),
                "original_prompt": prompt,
                "editing_prompt": row.get("editing_prompt", ""),
                "editing_instruction": row.get("editing_instruction", ""),
                "psnr_mean": row.get("psnr_mean"),
                "psnr_std": row.get("psnr_std"),
            },
        )

        sample_init_latents = []
        for seed in seeds:
            if seed not in seed_cache:
                seed_cache[seed] = load_seed_latents(seed, Path(args.source_prefix))
            latents = seed_cache[seed]
            if sid >= len(latents):
                raise IndexError(f"sample_id {sid} is outside seed {seed} latent list of length {len(latents)}")

            latent = latents[sid].detach().cpu()
            latent_path = out_dir / f"seed_{seed:02d}_init_latent.pt"
            torch.save(latent, latent_path)
            sample_init_latents.append(latent)

            manifest_row = {
                "label": label,
                "rank": row["rank"],
                "sample_id": sid,
                "seed": seed,
                "prompt": prompt,
                "prompt_psnr_mean": row["psnr_mean"],
                "prompt_psnr_std": row.get("psnr_std", ""),
                "source_init_latents_path": str(Path(f"{args.source_prefix}{seed}") / "init_latents.pt"),
                "source_latent_index": sid,
                "saved_init_latent_path": str(latent_path),
                "cfg1_image_path": str(out_dir / f"seed_{seed:02d}_cfg{args.guidance_scale:g}_gen.png"),
                "cfg1_gen_latent_path": str(out_dir / f"seed_{seed:02d}_cfg{args.guidance_scale:g}_gen_latent.pt"),
            }
            manifest_row.update(latent_stats(latent))
            manifest_rows.append(manifest_row)

        torch.save(sample_init_latents, out_dir / "init_latents.pt")

    fields = [
        "label",
        "rank",
        "sample_id",
        "seed",
        "prompt",
        "prompt_psnr_mean",
        "prompt_psnr_std",
        "source_init_latents_path",
        "source_latent_index",
        "saved_init_latent_path",
        "cfg1_image_path",
        "cfg1_gen_latent_path",
        "latent_shape",
        "latent_mean",
        "latent_std",
        "latent_l2_norm",
    ]
    write_csv(output_dir / "manifest.csv", manifest_rows, fields)
    return manifest_rows


def generate_images(args: argparse.Namespace, manifest_rows: list[dict]) -> None:
    pipe = load_pipeline(args)
    pipe_dtype = next(pipe.unet.parameters()).dtype
    rows_by_sample: dict[tuple[str, int], list[dict]] = defaultdict(list)

    for row in tqdm(manifest_rows, desc="Generating CFG images"):
        rows_by_sample[(row["label"], int(row["sample_id"]))].append(row)
        image_path = Path(row["cfg1_image_path"])
        gen_latent_path = Path(row["cfg1_gen_latent_path"])
        if args.skip_existing and image_path.exists() and gen_latent_path.exists():
            continue

        latent = torch.load(row["saved_init_latent_path"], map_location="cpu")
        latent = latent.to(device=args.device, dtype=pipe_dtype)
        with torch.no_grad():
            images, gen_latent = pipe(
                row["prompt"],
                latents=latent,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_of_ddim_steps,
            )

        image_path.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(image_path)
        torch.save(gen_latent.detach().cpu(), gen_latent_path)

    for (label, sid), sample_rows in rows_by_sample.items():
        sample_rows = sorted(sample_rows, key=lambda item: int(item["seed"]))
        paths = [Path(row["cfg1_gen_latent_path"]) for row in sample_rows]
        if all(path.exists() for path in paths):
            latents = [torch.load(path, map_location="cpu") for path in paths]
            torch.save(latents, sample_dir(Path(args.output_dir), label, sid) / f"cfg{args.guidance_scale:g}_gen_latents.pt")


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available() and not args.prepare_only:
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu or --prepare_only.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_int_set(args.seeds)
    ranked_rows = read_ranked_rows(Path(args.ranking_csv))
    selected = select_best_worst(ranked_rows, args.top_k)

    save_selected_prompts(output_dir, selected)
    manifest_rows = prepare_latents(args, selected, seeds)

    run_config = {
        "ranking_csv": args.ranking_csv,
        "output_dir": args.output_dir,
        "source_prefix": args.source_prefix,
        "model_name": args.model_name,
        "top_k": args.top_k,
        "seeds": seeds,
        "guidance_scale": args.guidance_scale,
        "num_of_ddim_steps": args.num_of_ddim_steps,
        "n_selected_prompts": len(selected),
        "n_manifest_rows": len(manifest_rows),
        "prepare_only": args.prepare_only,
    }
    write_json(output_dir / "run_config.json", run_config)

    if not args.prepare_only:
        generate_images(args, manifest_rows)

    print(f"Saved selected prompt/latent manifest to: {output_dir}")
    print(f"Selected prompts: {len(selected)} ({args.top_k} best + {args.top_k} worst)")
    print(f"Seed latents: {len(manifest_rows)}")
    if args.prepare_only:
        print("prepare_only=True, skipped image generation.")
    else:
        print(f"Generated CFG={args.guidance_scale:g} images by sample_id under: {output_dir}")


if __name__ == "__main__":
    main()
