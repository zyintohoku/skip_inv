#!/usr/bin/env python3
"""Run most-sensitive prompt generation with per-step prompt pressure tracing."""

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


PRESSURE_FORMULA = {
    "ddim_eta0_update": "x_prev = sqrt(alpha_prev / alpha_t) * x_t + c_t * epsilon_hat",
    "epsilon_coefficient": "c_t = sqrt(1 - alpha_prev) - sqrt(alpha_prev * (1 - alpha_t) / alpha_t)",
    "cfg_noise": "epsilon_hat = epsilon_uncond + guidance_scale * (epsilon_cond - epsilon_uncond)",
    "conditional_prompt_term": "guidance_scale * c_t * epsilon_cond",
    "prompt_pressure": "P_t = || guidance_scale * c_t * epsilon_cond ||_2",
    "total_prompt_pressure": "sum_t P_t",
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


def load_sensitive_prompt_from_csv(results_dir):
    csv_path = Path(results_dir) / "prompt_psnr_most_seed_sensitive30.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Sensitive prompt CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f))
    return {
        "sample_id": int(row["sample_id"]),
        "mapping_key": row.get("mapping_key", ""),
        "prompt": clean_prompt(row["original_prompt"]),
        "source_psnr_mean": row.get("psnr_mean", ""),
        "source_psnr_std": row.get("psnr_std", ""),
        "source": str(csv_path),
    }


def load_sensitive_trial_spec(args):
    manifest_path = Path(args.source_manifest)
    if manifest_path.exists():
        rows = []
        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows = [row for row in rows if row.get("group") == "sensitive"]
        if not rows:
            raise ValueError(f"No sensitive rows found in source manifest: {manifest_path}")
        seeds = sorted({int(row["seed"]) for row in rows})
        if args.seeds:
            requested = set(parse_int_spec(args.seeds))
            seeds = [seed for seed in seeds if seed in requested]
        first = rows[0]
        return {
            "sample_id": int(first["sample_id"]),
            "mapping_key": first.get("mapping_key", ""),
            "prompt": clean_prompt(first["prompt"]),
            "source_psnr_mean": first.get("source_psnr_mean", ""),
            "source_psnr_std": first.get("source_psnr_std", ""),
            "source": str(manifest_path),
            "seeds": seeds,
        }

    spec = load_sensitive_prompt_from_csv(args.results_dir)
    spec["seeds"] = parse_int_spec(args.seeds or "0-99")
    return spec


def ddim_epsilon_coefficient(scheduler, timestep):
    t = int(timestep.item() if hasattr(timestep, "item") else timestep)
    step = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    prev_t = t - step
    alpha_t = scheduler.alphas_cumprod[t].float()
    if prev_t >= 0:
        alpha_prev = scheduler.alphas_cumprod[prev_t].float()
    else:
        alpha_prev = scheduler.final_alpha_cumprod.float()
    coeff = torch.sqrt(1.0 - alpha_prev) - torch.sqrt(alpha_prev * (1.0 - alpha_t) / alpha_t)
    return coeff.to(dtype=torch.float32)


@torch.no_grad()
def generate_with_prompt_pressure(
    pipe,
    prompt,
    init_latent,
    guidance_scale,
    num_inference_steps,
    eta=0.0,
):
    height = pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = pipe.unet.config.sample_size * pipe.vae_scale_factor
    pipe.check_inputs(prompt, height, width, callback_steps=1)

    device = pipe._execution_device
    do_classifier_free_guidance = True
    text_embeddings = pipe._encode_prompt(
        prompt,
        device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=None,
    )

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=eta)
    latents = init_latent.clone().detach().to(device=device, dtype=text_embeddings.dtype)
    trace_rows = []

    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for step_index, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_guided = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            eps_coeff = ddim_epsilon_coefficient(pipe.scheduler, t).to(device=device)
            weighted_cond = guidance_scale * eps_coeff * noise_cond.float()
            prompt_pressure = torch.linalg.vector_norm(weighted_cond).item()
            eps_cond_l2 = torch.linalg.vector_norm(noise_cond.float()).item()
            guidance_delta_l2 = torch.linalg.vector_norm((noise_cond - noise_uncond).float()).item()

            prev_latents = pipe.scheduler.step(noise_guided, t, latents, **extra_step_kwargs).prev_sample
            latent_step_l2 = torch.linalg.vector_norm((prev_latents - latents).float()).item()

            trace_rows.append(
                {
                    "step_index": step_index,
                    "timestep": int(t.item() if hasattr(t, "item") else t),
                    "scheduler_epsilon_coeff": float(eps_coeff.item()),
                    "guidance_scale": float(guidance_scale),
                    "epsilon_cond_l2": eps_cond_l2,
                    "guidance_delta_l2": guidance_delta_l2,
                    "prompt_pressure_P_t": prompt_pressure,
                    "latent_step_l2": latent_step_l2,
                }
            )

            latents = prev_latents
            progress_bar.update()

    image = pipe.decode_latents(latents)
    image = pipe.numpy_to_pil(image)
    return image, latents, trace_rows


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main(args):
    device = torch.device(args.device)
    spec = load_sensitive_trial_spec(args)
    if not spec["seeds"]:
        raise ValueError("No seeds selected for sensitive prompt pressure run.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = output_dir / f"sample{spec['sample_id']:03d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    pipe = MyStableDiffusionPipeline.from_pretrained(args.model_name, scheduler=make_scheduler()).to(device)
    if args.disable_progress_bar:
        pipe.set_progress_bar_config(disable=True)

    inversion = Inversion(
        pipe,
        num_ddim_steps=args.num_of_ddim_steps,
        delta_threshold=args.delta_threshold,
        method="ddim",
        loss_divergence_threshold=args.loss_divergence_threshold,
    )

    manifest_path = output_dir / "manifest.csv"
    manifest_fields = [
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
        "trace_csv",
        "trace_json",
        "image_psnr",
        "init_inv_mse",
        "gen_rec_mse",
        "prompt_pressure_total",
        "prompt_pressure_mean",
        "prompt_pressure_max",
        "invert_time",
        "total_time",
    ]
    trace_fields = [
        "step_index",
        "timestep",
        "scheduler_epsilon_coeff",
        "guidance_scale",
        "epsilon_cond_l2",
        "guidance_delta_l2",
        "prompt_pressure_P_t",
        "latent_step_l2",
    ]

    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=manifest_fields)
        writer.writeheader()

        for seed in spec["seeds"]:
            trial_start = time.time()
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            init_latent = torch.randn(1, 4, 64, 64).to(device)

            image_gen, gen_latent, trace_rows = generate_with_prompt_pressure(
                pipe,
                spec["prompt"],
                init_latent,
                args.guidance_scale,
                args.num_of_ddim_steps,
                eta=args.eta,
            )

            invert_start = time.time()
            all_inv_latents, _convergence_losses = inversion.invert(gen_latent, spec["prompt"], args.guidance_scale)
            inv_latent = all_inv_latents[-1]
            invert_time = time.time() - invert_start

            image_rec, rec_latent = pipe(
                prompt=spec["prompt"],
                latents=inv_latent,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_of_ddim_steps,
            )

            seed_stem = f"seed{seed:04d}"
            gen_image_path = sample_dir / f"{seed_stem}_gen.png"
            rec_image_path = sample_dir / f"{seed_stem}_rec.png"
            trace_csv_path = sample_dir / f"{seed_stem}_prompt_pressure_trace.csv"
            trace_json_path = sample_dir / f"{seed_stem}_prompt_pressure_trace.json"

            if args.save_images:
                image_gen[0].save(gen_image_path)
                image_rec[0].save(rec_image_path)

            init_latent_path = sample_dir / f"{seed_stem}_init.pt"
            gen_latent_path = sample_dir / f"{seed_stem}_gen.pt"
            inv_latent_path = sample_dir / f"{seed_stem}_ddim_inv.pt"
            rec_latent_path = sample_dir / f"{seed_stem}_rec.pt"
            if args.save_latents:
                save_tensor(init_latent_path, init_latent)
                save_tensor(gen_latent_path, gen_latent)
                save_tensor(inv_latent_path, inv_latent)
                save_tensor(rec_latent_path, rec_latent)

            write_csv(trace_csv_path, trace_rows, trace_fields)
            pressure_total = sum(float(row["prompt_pressure_P_t"]) for row in trace_rows)
            with trace_json_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "formula": PRESSURE_FORMULA,
                        "sample_id": spec["sample_id"],
                        "mapping_key": spec["mapping_key"],
                        "seed": seed,
                        "prompt": spec["prompt"],
                        "guidance_scale": args.guidance_scale,
                        "num_of_ddim_steps": args.num_of_ddim_steps,
                        "prompt_pressure_total": pressure_total,
                        "records": trace_rows,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            writer.writerow(
                {
                    "sample_id": spec["sample_id"],
                    "mapping_key": spec["mapping_key"],
                    "seed": seed,
                    "prompt": spec["prompt"],
                    "source_psnr_mean": spec["source_psnr_mean"],
                    "source_psnr_std": spec["source_psnr_std"],
                    "gen_image": str(gen_image_path) if args.save_images else "",
                    "rec_image": str(rec_image_path) if args.save_images else "",
                    "init_latent": str(init_latent_path) if args.save_latents else "",
                    "gen_latent": str(gen_latent_path) if args.save_latents else "",
                    "inv_latent": str(inv_latent_path) if args.save_latents else "",
                    "rec_latent": str(rec_latent_path) if args.save_latents else "",
                    "trace_csv": str(trace_csv_path),
                    "trace_json": str(trace_json_path),
                    "image_psnr": image_psnr(image_gen[0], image_rec[0]),
                    "init_inv_mse": torch.mean((init_latent - inv_latent) ** 2).item(),
                    "gen_rec_mse": torch.mean((gen_latent - rec_latent) ** 2).item(),
                    "prompt_pressure_total": pressure_total,
                    "prompt_pressure_mean": pressure_total / len(trace_rows),
                    "prompt_pressure_max": max(float(row["prompt_pressure_P_t"]) for row in trace_rows),
                    "invert_time": invert_time,
                    "total_time": time.time() - trial_start,
                }
            )
            manifest_file.flush()

    config = {
        "formula": PRESSURE_FORMULA,
        "source": spec["source"],
        "output_dir": str(output_dir),
        "model_name": args.model_name,
        "guidance_scale": args.guidance_scale,
        "num_of_ddim_steps": args.num_of_ddim_steps,
        "inversion_method": "ddim",
        "delta_threshold": args.delta_threshold,
        "loss_divergence_threshold": args.loss_divergence_threshold,
        "seeds": spec["seeds"],
        "num_trials": len(spec["seeds"]),
        "manifest": str(manifest_path),
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_manifest",
        default=str(REPO_ROOT.parent / "artifacts" / "outputs" / "prompt_group_top1_fpi_100" / "sensitive" / "manifest.csv"),
        help="Manifest from the previous top1 sensitive run. Seeds and prompt are reused from this file when present.",
    )
    parser.add_argument("--results_dir", default=str(REPO_ROOT.parent / "artifacts" / "results" / "fpi_gs7_seed_psnr"))
    parser.add_argument(
        "--output_dir",
        default=str(REPO_ROOT.parent / "artifacts" / "outputs" / "sensitive_prompt_pressure_ddim_inv_rec"),
    )
    parser.add_argument("--model_name", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--num_of_ddim_steps", type=int, default=50)
    parser.add_argument("--delta_threshold", type=float, default=5e-12)
    parser.add_argument("--loss_divergence_threshold", type=float, default=1.0)
    parser.add_argument("--seeds", default=None, help="Optional seed subset, e.g. 0-9 or 0,3,7.")
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_latents", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable_progress_bar", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
