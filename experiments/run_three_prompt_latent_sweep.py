#!/usr/bin/env python3
"""Run DDIM/FPI/ours inversion on 100 sampled initial latents for three prompts."""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.inv_methods import Inversion as FixedPointInversion  # noqa: E402
from utils.skip_pipe import Inversion as OursInversion  # noqa: E402
from utils.skip_pipe import MyStableDiffusionPipeline  # noqa: E402


DEFAULT_PROMPTS = [
    {
        "label": "best",
        "sample_id": 74,
        "mapping_key": "000000000074",
        "prompt": "a mountain is covered in clouds and snow",
        "editing_prompt": "a mountain is covered in snow",
        "editing_instruction": "Remove the clouds",
    },
    {
        "label": "worst",
        "sample_id": 270,
        "mapping_key": "222000000000",
        "prompt": "MonaLisa with mysterious smile",
        "editing_prompt": "[a golden painting of] MonaLisa with mysterious smile",
        "editing_instruction": "Add the painting a golden style",
    },
    {
        "label": "sensitive",
        "sample_id": 443,
        "mapping_key": "521000000003",
        "prompt": "a fox is walking in the snow",
        "editing_prompt": "a fox is walking in the snow [with head down]",
        "editing_instruction": "Make the fox's head down",
    },
]


METHOD_CHOICES = ("ddim", "fpi", "ours")


def make_scheduler() -> DDIMScheduler:
    return DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample initial latents, generate images for the best/worst/most-sensitive "
            "FPI prompts, then invert and reconstruct each generated latent with "
            "DDIM, FPI, and ours."
        )
    )
    parser.add_argument("--output", default="outputs/three_prompt_latent_sweep")
    parser.add_argument("--model_name", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--num_initial_latents", type=int, default=100)
    parser.add_argument("--latent_seed", type=int, default=0)
    parser.add_argument("--methods", default="ddim,fpi,ours")
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--num_of_ddim_steps", type=int, default=50)
    parser.add_argument("--delta_threshold", type=float, default=5e-12)
    parser.add_argument("--loss_divergence_threshold", type=float, default=1.0)
    parser.add_argument("--max_iterations", type=int, default=100)
    parser.add_argument("--ours_delta_threshold", type=float, default=5e-9)
    parser.add_argument("--ours_loss_divergence_threshold", type=float, default=0.9)
    parser.add_argument("--ours_force_converge_before_step", type=int, default=10)
    parser.add_argument("--ours_max_iterations", type=int, default=100)
    parser.add_argument("--ours_reset_gs", action="store_true")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", choices=["auto", "float32", "float16"], default="float32")
    parser.add_argument("--negative_prompt", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--disable_progress_bar", action="store_true")
    parser.add_argument("--save_generation_latents", action="store_true")
    return parser.parse_args()


def parse_methods(spec: str) -> list[str]:
    methods = []
    for chunk in spec.split(","):
        method = chunk.strip().lower()
        if not method:
            continue
        if method not in METHOD_CHOICES:
            raise ValueError(f"Unsupported method '{method}'. Choose from: {', '.join(METHOD_CHOICES)}")
        methods.append(method)
    if not methods:
        raise ValueError("At least one method must be selected.")
    return methods


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def torch_dtype_from_arg(value: str):
    if value == "float16":
        return torch.float16
    if value == "float32":
        return torch.float32
    return None


def load_pipeline(args: argparse.Namespace, device: torch.device) -> MyStableDiffusionPipeline:
    kwargs = {
        "scheduler": make_scheduler(),
        "local_files_only": args.local_files_only,
    }
    dtype = torch_dtype_from_arg(args.torch_dtype)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    pipe = MyStableDiffusionPipeline.from_pretrained(args.model_name, **kwargs).to(device)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=args.disable_progress_bar)
    return pipe


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def image_mse_psnr(image_a, image_b) -> tuple[float, float]:
    arr_a = np.asarray(image_a.convert("RGB"), dtype=np.float32)
    arr_b = np.asarray(image_b.convert("RGB"), dtype=np.float32)
    mse = float(np.mean((arr_a - arr_b) ** 2))
    if mse == 0.0:
        return mse, float("inf")
    return mse, 20.0 * math.log10(255.0 / math.sqrt(mse))


def tensor_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.mse_loss(a.detach().float().cpu(), b.detach().float().cpu()).item())


def latent_stats(latent: torch.Tensor) -> dict:
    latent = latent.detach().float().cpu()
    return {
        "shape": list(latent.shape),
        "mean": float(latent.mean().item()),
        "std": float(latent.std().item()),
        "min": float(latent.min().item()),
        "max": float(latent.max().item()),
        "l2_norm": float(torch.linalg.vector_norm(latent).item()),
    }


def sample_initial_latents(
    pipe: MyStableDiffusionPipeline,
    device: torch.device,
    height: int,
    width: int,
    count: int,
    seed: int,
) -> list[torch.Tensor]:
    dtype = next(pipe.unet.parameters()).dtype
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (count, pipe.unet.in_channels, height // 8, width // 8)
    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    return [latents[i : i + 1].detach().clone() for i in range(count)]


class BoundedFixedPointInversion(FixedPointInversion):
    def __init__(self, *args, max_iterations: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self.max_iterations = max_iterations
        self.hit_max_iterations = False
        self.any_hit_max_iterations = False

    def fpi_step(self, init_latent, latent_ztm1, t, guidance_scale):
        optimal_latent = init_latent.clone().detach()
        loss_prev = 1.0
        final_loss = 0.0
        self.hit_max_iterations = False

        for _ in range(self.max_iterations):
            latent_input = torch.cat([optimal_latent] * 2)
            noise_pred = self.get_noise_pred_single(latent_input, t, self.context)
            noise_uncond, noise_cond = noise_pred.chunk(2)
            guided_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            updated_latent = self.next_step(guided_noise, t, latent_ztm1)
            loss = F.mse_loss(updated_latent, optimal_latent).item()
            final_loss = loss
            if loss < self.threshold:
                break
            if loss > loss_prev:
                break
            optimal_latent = updated_latent
            loss_prev = loss
        else:
            self.hit_max_iterations = True
            self.any_hit_max_iterations = True

        return optimal_latent.detach(), final_loss


class BoundedOursInversion(OursInversion):
    def __init__(self, *args, max_iterations: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        if max_iterations < 1:
            raise ValueError("ours_max_iterations must be >= 1")
        self.max_iterations = max_iterations
        self.hit_max_iterations = False
        self.any_hit_max_iterations = False

    def afpi_step(self, init_latent, latent_ztm1, t, guidance_scale, step_index=None):
        optimal_latent = init_latent.clone().detach()
        force_converge = (
            step_index is not None
            and self.force_converge_before_step is not None
            and step_index < self.force_converge_before_step
        )

        alpha = 1.0
        loss_prev = 1.0
        loss = float("inf")
        self.hit_max_iterations = False

        for _ in range(self.max_iterations):
            latent_input = torch.cat([optimal_latent] * 2)
            noise_pred = self.get_noise_pred_single(latent_input, t, self.context)
            noise_uncond, noise_cond = noise_pred.chunk(2)
            guided_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            updated_latent = self.next_step(guided_noise, t, latent_ztm1)
            loss = F.mse_loss(updated_latent, optimal_latent).item()
            if loss > loss_prev and alpha == 0.5:
                break
            if loss > loss_prev * self.loss_divergence_threshold:
                alpha = 0.5
            optimal_latent = (1 - alpha) * optimal_latent + alpha * updated_latent
            loss_prev = loss
        else:
            self.hit_max_iterations = True
            self.any_hit_max_iterations = True

        converged = force_converge or loss < self.threshold
        return optimal_latent.detach(), converged


def make_inverters(
    pipe: MyStableDiffusionPipeline,
    methods: list[str],
    args: argparse.Namespace,
) -> dict[str, object]:
    inverters = {}
    for method in methods:
        if method == "ours":
            inverters[method] = BoundedOursInversion(
                pipe,
                num_ddim_steps=args.num_of_ddim_steps,
                delta_threshold=args.ours_delta_threshold,
                loss_divergence_threshold=args.ours_loss_divergence_threshold,
                reset_gs=args.ours_reset_gs,
                force_converge_before_step=args.ours_force_converge_before_step,
                max_iterations=args.ours_max_iterations,
            )
        else:
            inverters[method] = BoundedFixedPointInversion(
                pipe,
                num_ddim_steps=args.num_of_ddim_steps,
                delta_threshold=args.delta_threshold,
                method=method,
                loss_divergence_threshold=args.loss_divergence_threshold,
                max_iterations=args.max_iterations,
            )
    return inverters


def method_metric_fieldnames() -> list[str]:
    return [
        "record_id",
        "prompt_index",
        "latent_index",
        "label",
        "sample_id",
        "mapping_key",
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
        "hit_max_iterations",
        "gen_image_path",
        "rec_image_path",
    ]


def generation_fieldnames() -> list[str]:
    return [
        "record_id",
        "prompt_index",
        "latent_index",
        "label",
        "sample_id",
        "mapping_key",
        "prompt",
        "editing_prompt",
        "editing_instruction",
        "gen_image_path",
    ]


@torch.no_grad()
def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    device = resolve_device(args.device)

    output_dir = Path(args.output)
    gen_dir = output_dir / "generated"
    trace_path = output_dir / "method_traces.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)
    if trace_path.exists():
        trace_path.unlink()

    pipe = load_pipeline(args, device)
    inverters = make_inverters(pipe, methods, args)
    unique_init_latents = sample_initial_latents(
        pipe,
        device=device,
        height=args.height,
        width=args.width,
        count=args.num_initial_latents,
        seed=args.latent_seed,
    )

    torch.save([latent.detach().cpu() for latent in unique_init_latents], output_dir / "unique_init_latents.pt")
    write_json(
        output_dir / "prompt_config.json",
        {
            "prompts": DEFAULT_PROMPTS,
            "methods": methods,
            "num_initial_latents": args.num_initial_latents,
            "latent_seed": args.latent_seed,
            "guidance_scale": args.guidance_scale,
            "num_of_ddim_steps": args.num_of_ddim_steps,
            "height": args.height,
            "width": args.width,
            "model_name": args.model_name,
            "unique_init_latents_path": str(output_dir / "unique_init_latents.pt"),
            "first_init_latent_stats": latent_stats(unique_init_latents[0]) if unique_init_latents else None,
        },
    )

    method_latents = {
        method: {
            "init": [],
            "gen": [],
            "inv": [],
            "rec": [],
            "cfg_schedules": [],
        }
        for method in methods
    }
    generation_rows = []
    method_rows = []

    total_start = time.time()
    total_records = len(DEFAULT_PROMPTS) * args.num_initial_latents
    for prompt_index, prompt_row in enumerate(DEFAULT_PROMPTS):
        prompt = prompt_row["prompt"]
        label = prompt_row["label"]
        for latent_index, init_latent in enumerate(unique_init_latents):
            record_id = prompt_index * args.num_initial_latents + latent_index
            print(
                f"[{record_id + 1}/{total_records}] "
                f"label={label} latent_index={latent_index} prompt={prompt!r}"
            )

            gen_image_path = gen_dir / f"{record_id:04d}_{label}_z{latent_index:03d}_gen.png"
            gen_images, gen_latent = pipe(
                prompt=prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_of_ddim_steps,
                guidance_scale=args.guidance_scale,
                negative_prompt=args.negative_prompt,
                latents=init_latent.clone(),
            )
            gen_image = gen_images[0]
            gen_image.save(gen_image_path)

            generation_row = {
                "record_id": record_id,
                "prompt_index": prompt_index,
                "latent_index": latent_index,
                "label": label,
                "sample_id": prompt_row["sample_id"],
                "mapping_key": prompt_row["mapping_key"],
                "prompt": prompt,
                "editing_prompt": prompt_row["editing_prompt"],
                "editing_instruction": prompt_row["editing_instruction"],
                "gen_image_path": str(gen_image_path),
            }
            generation_rows.append(generation_row)

            for method in methods:
                method_dir = output_dir / method
                method_dir.mkdir(parents=True, exist_ok=True)

                start = time.time()
                if method == "ours":
                    inverters[method].any_hit_max_iterations = False
                    all_inv_latents, inversion_aux = inverters[method].invert(
                        gen_latent,
                        prompt,
                        args.guidance_scale,
                        sample_id=record_id,
                    )
                else:
                    inverters[method].any_hit_max_iterations = False
                    all_inv_latents, inversion_aux = inverters[method].invert(
                        gen_latent,
                        prompt,
                        args.guidance_scale,
                    )
                inversion_time = time.time() - start
                inv_latent = all_inv_latents[-1]

                if method == "ours":
                    cfg_schedule = [float(x) for x in inversion_aux]
                    convergence_losses = []
                    rec_kwargs = {"cfg_schedule": cfg_schedule}
                else:
                    cfg_schedule = []
                    convergence_losses = [float(x) for x in inversion_aux]
                    rec_kwargs = {}

                rec_images, rec_latent = pipe(
                    prompt=prompt,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_of_ddim_steps,
                    guidance_scale=args.guidance_scale,
                    negative_prompt=args.negative_prompt,
                    latents=inv_latent.clone(),
                    **rec_kwargs,
                )
                rec_image = rec_images[0]
                rec_image_path = method_dir / f"{record_id:04d}_{label}_z{latent_index:03d}_rec.png"
                rec_image.save(rec_image_path)

                image_mse, image_psnr = image_mse_psnr(gen_image, rec_image)
                final_loss = float(convergence_losses[-1]) if convergence_losses else ""
                mean_loss = float(np.mean(convergence_losses)) if convergence_losses else ""
                hit_max_iterations = bool(getattr(inverters[method], "any_hit_max_iterations", False))

                method_latents[method]["init"].append(init_latent.detach().cpu())
                method_latents[method]["gen"].append(gen_latent.detach().cpu())
                method_latents[method]["inv"].append(inv_latent.detach().cpu())
                method_latents[method]["rec"].append(rec_latent.detach().cpu())
                method_latents[method]["cfg_schedules"].append(cfg_schedule)

                metric_row = {
                    **generation_row,
                    "method": method,
                    "guidance_scale": args.guidance_scale,
                    "num_of_ddim_steps": args.num_of_ddim_steps,
                    "image_psnr": image_psnr,
                    "image_mse": image_mse,
                    "gen_rec_latent_mse": tensor_mse(gen_latent, rec_latent),
                    "init_inv_latent_mse": tensor_mse(init_latent, inv_latent),
                    "inversion_time": inversion_time,
                    "inversion_final_loss": final_loss,
                    "inversion_mean_loss": mean_loss,
                    "hit_max_iterations": hit_max_iterations,
                    "rec_image_path": str(rec_image_path),
                }
                method_rows.append(metric_row)

                append_jsonl(
                    trace_path,
                    {
                        "record_id": record_id,
                        "prompt_index": prompt_index,
                        "latent_index": latent_index,
                        "label": label,
                        "sample_id": prompt_row["sample_id"],
                        "mapping_key": prompt_row["mapping_key"],
                        "prompt": prompt,
                        "method": method,
                        "gen_image_path": str(gen_image_path),
                        "rec_image_path": str(rec_image_path),
                        "convergence_losses": convergence_losses,
                        "cfg_schedule": cfg_schedule,
                        "metrics": metric_row,
                    },
                )

    write_csv(output_dir / "generation_manifest.csv", generation_rows, generation_fieldnames())
    write_csv(output_dir / "method_metrics.csv", method_rows, method_metric_fieldnames())

    for method, latents in method_latents.items():
        method_dir = output_dir / method
        torch.save(latents["init"], method_dir / "init_latents.pt")
        torch.save(latents["gen"], method_dir / "gen_latents.pt")
        torch.save(latents["inv"], method_dir / "inv_latents.pt")
        torch.save(latents["rec"], method_dir / "rec_latents.pt")
        if method == "ours":
            torch.save(latents["cfg_schedules"], method_dir / "cfg_schedules.pt")
        if args.save_generation_latents:
            torch.save(
                {
                    "init_latents": latents["init"],
                    "gen_latents": latents["gen"],
                    "record_order_csv": str(output_dir / "generation_manifest.csv"),
                },
                method_dir / "generation_latents.pt",
            )

    total_time = time.time() - total_start
    write_json(
        output_dir / "run_summary.json",
        {
            "output": args.output,
            "model_name": args.model_name,
            "methods": methods,
            "num_prompts": len(DEFAULT_PROMPTS),
            "num_initial_latents": args.num_initial_latents,
            "num_generated_images": len(generation_rows),
            "num_reconstructed_images": len(method_rows),
            "guidance_scale": args.guidance_scale,
            "num_of_ddim_steps": args.num_of_ddim_steps,
            "delta_threshold": args.delta_threshold,
            "loss_divergence_threshold": args.loss_divergence_threshold,
            "max_iterations": args.max_iterations,
            "ours_delta_threshold": args.ours_delta_threshold,
            "ours_loss_divergence_threshold": args.ours_loss_divergence_threshold,
            "ours_force_converge_before_step": args.ours_force_converge_before_step,
            "ours_max_iterations": args.ours_max_iterations,
            "ours_reset_gs": args.ours_reset_gs,
            "latent_seed": args.latent_seed,
            "total_time": total_time,
            "avg_time_per_method_reconstruction": total_time / len(method_rows) if method_rows else None,
        },
    )

    print("total_time:", total_time)
    if method_rows:
        print("avg_time_per_method_reconstruction:", total_time / len(method_rows))
    print("saved results to:", output_dir)


if __name__ == "__main__":
    main()
