#!/usr/bin/env python3
"""Run repeated generate-invert-reconstruct trials for selected prompt groups."""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from diffusers import DDIMScheduler

from utils.inv_methods import Inversion, MyStableDiffusionPipeline


GROUP_FILES = {
    "best": "prompt_psnr_best30.csv",
    "worst": "prompt_psnr_worst30.csv",
    "sensitive": "prompt_psnr_most_seed_sensitive30.csv",
}


def parse_int_spec(spec):
    values = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid integer range: {token}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(token))
    if not values:
        raise ValueError(f"No integer values parsed from: {spec}")
    return values


def make_scheduler():
    return DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )


def load_prompt_rows(results_dir, group, top_k):
    csv_path = Path(results_dir) / GROUP_FILES[group]
    if not csv_path.exists():
        raise FileNotFoundError(f"Prompt group CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if top_k is not None:
        rows = rows[:top_k]
    if not rows:
        raise ValueError(f"No prompts loaded from: {csv_path}")
    return rows, csv_path


def clean_prompt(text):
    return text.replace("[", "").replace("]", "")


def image_psnr(image_a, image_b):
    arr_a = np.asarray(image_a).astype(np.float32) / 255.0
    arr_b = np.asarray(image_b).astype(np.float32) / 255.0
    mse = float(np.mean((arr_a - arr_b) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def save_tensor(path, tensor):
    torch.save(tensor.detach().cpu(), path)


@torch.no_grad()
def main(args):
    device = torch.device(args.device)
    output_dir = Path(args.output_dir) / args.group
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, source_csv = load_prompt_rows(args.results_dir, args.group, args.top_k)
    seeds = parse_int_spec(args.seeds)

    pipe = MyStableDiffusionPipeline.from_pretrained(args.model_name, scheduler=make_scheduler()).to(device)
    if args.disable_progress_bar:
        pipe.set_progress_bar_config(disable=True)
    inversion = Inversion(
        pipe,
        num_ddim_steps=args.num_of_ddim_steps,
        delta_threshold=args.delta_threshold,
        method=args.method,
        loss_divergence_threshold=args.loss_divergence_threshold,
    )

    manifest_path = output_dir / "manifest.csv"
    manifest_fields = [
        "group",
        "rank",
        "sample_id",
        "mapping_key",
        "seed",
        "prompt",
        "source_psnr_mean",
        "source_psnr_std",
        "gen_image",
        "rec_image",
        "init_latent",
        "gen_latent",
        "inv_latent",
        "rec_latent",
        "image_psnr",
        "init_inv_mse",
        "gen_rec_mse",
        "invert_time",
        "total_time",
    ]

    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=manifest_fields)
        writer.writeheader()

        for rank, row in enumerate(rows, start=1):
            sample_id = int(row["sample_id"])
            mapping_key = row.get("mapping_key", "")
            prompt = clean_prompt(row["original_prompt"])
            prompt_dir = output_dir / f"rank{rank:03d}_sample{sample_id:03d}"
            prompt_dir.mkdir(parents=True, exist_ok=True)

            for seed in seeds:
                trial_start = time.time()
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)

                init_latent = torch.randn(1, 4, 64, 64).to(device)
                image_gen, gen_latent = pipe(
                    prompt=prompt,
                    latents=init_latent,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_of_ddim_steps,
                )

                invert_start = time.time()
                all_inv_latents, _convergence_losses = inversion.invert(gen_latent, prompt, args.guidance_scale)
                inv_latent = all_inv_latents[-1]
                invert_time = time.time() - invert_start

                image_rec, rec_latent = pipe(
                    prompt=prompt,
                    latents=inv_latent,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_of_ddim_steps,
                )

                seed_stem = f"seed{seed:04d}"
                gen_image_path = prompt_dir / f"{seed_stem}_gen.png"
                rec_image_path = prompt_dir / f"{seed_stem}_rec.png"
                if args.save_images:
                    image_gen[0].save(gen_image_path)
                    image_rec[0].save(rec_image_path)

                init_latent_path = prompt_dir / f"{seed_stem}_init.pt"
                gen_latent_path = prompt_dir / f"{seed_stem}_gen.pt"
                inv_latent_path = prompt_dir / f"{seed_stem}_inv.pt"
                rec_latent_path = prompt_dir / f"{seed_stem}_rec.pt"
                if args.save_latents:
                    save_tensor(init_latent_path, init_latent)
                    save_tensor(gen_latent_path, gen_latent)
                    save_tensor(inv_latent_path, inv_latent)
                    save_tensor(rec_latent_path, rec_latent)

                writer.writerow(
                    {
                        "group": args.group,
                        "rank": rank,
                        "sample_id": sample_id,
                        "mapping_key": mapping_key,
                        "seed": seed,
                        "prompt": prompt,
                        "source_psnr_mean": row.get("psnr_mean", ""),
                        "source_psnr_std": row.get("psnr_std", ""),
                        "gen_image": str(gen_image_path) if args.save_images else "",
                        "rec_image": str(rec_image_path) if args.save_images else "",
                        "init_latent": str(init_latent_path) if args.save_latents else "",
                        "gen_latent": str(gen_latent_path) if args.save_latents else "",
                        "inv_latent": str(inv_latent_path) if args.save_latents else "",
                        "rec_latent": str(rec_latent_path) if args.save_latents else "",
                        "image_psnr": image_psnr(image_gen[0], image_rec[0]),
                        "init_inv_mse": torch.mean((init_latent - inv_latent) ** 2).item(),
                        "gen_rec_mse": torch.mean((gen_latent - rec_latent) ** 2).item(),
                        "invert_time": invert_time,
                        "total_time": time.time() - trial_start,
                    }
                )
                manifest_file.flush()

    config = {
        "group": args.group,
        "results_dir": args.results_dir,
        "source_csv": str(source_csv),
        "output_dir": str(output_dir),
        "model_name": args.model_name,
        "method": args.method,
        "guidance_scale": args.guidance_scale,
        "num_of_ddim_steps": args.num_of_ddim_steps,
        "delta_threshold": args.delta_threshold,
        "loss_divergence_threshold": args.loss_divergence_threshold,
        "top_k": args.top_k,
        "num_prompts": len(rows),
        "seeds": seeds,
        "num_trials_per_prompt": len(seeds),
        "save_images": args.save_images,
        "save_latents": args.save_latents,
        "manifest": str(manifest_path),
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=sorted(GROUP_FILES), required=True)
    parser.add_argument("--results_dir", default=str(REPO_ROOT.parent / "artifacts" / "results" / "fpi_gs7_seed_psnr"))
    parser.add_argument("--output_dir", default="outputs/prompt_group_fpi_100")
    parser.add_argument("--model_name", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--method", default="fpi", choices=["fpi", "afpi", "aidi", "ddim"])
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--num_of_ddim_steps", type=int, default=50)
    parser.add_argument("--delta_threshold", type=float, default=5e-12)
    parser.add_argument("--loss_divergence_threshold", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--seeds", default="0-99")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_latents", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable_progress_bar", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
