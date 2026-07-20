#!/usr/bin/env python3
"""Run prompt-grid generation, inversion, and reconstruction with one shared init latent."""

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

from utils.inv_methods import Inversion, MyStableDiffusionPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate, invert, and reconstruct each prompt in a CSV, while reusing "
            "one initial latent sampled once for the chosen seed."
        )
    )
    parser.add_argument("--prompt_csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default="fpi", choices=["ddim", "fpi", "afpi", "aidi"])
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model_name", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--num_of_ddim_steps", type=int, default=50)
    parser.add_argument("--delta_threshold", type=float, default=5e-12)
    parser.add_argument("--loss_divergence_threshold", type=float, default=0.9)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", choices=["auto", "float32", "float16"], default="float32")
    parser.add_argument("--negative_prompt", default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--disable_progress_bar", action="store_true")
    return parser.parse_args()


def torch_dtype_from_arg(value: str):
    if value == "float16":
        return torch.float16
    if value == "float32":
        return torch.float32
    return None


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
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=args.disable_progress_bar)
    return pipe


def read_prompt_csv(path: Path) -> tuple[list[dict], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if not rows:
        raise ValueError(f"No rows found in prompt CSV: {path}")
    if "prompt" not in fieldnames:
        raise ValueError(f"Prompt CSV must contain a 'prompt' column: {path}")
    return rows, fieldnames


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


def append_jsonl(path: Path, data: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def image_mse_psnr(image_a, image_b) -> tuple[float, float]:
    arr_a = np.asarray(image_a.convert("RGB"), dtype=np.float32)
    arr_b = np.asarray(image_b.convert("RGB"), dtype=np.float32)
    mse = float(np.mean((arr_a - arr_b) ** 2))
    if mse == 0.0:
        return mse, float("inf")
    psnr = 20.0 * math.log10(255.0 / math.sqrt(mse))
    return mse, psnr


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


def sample_initial_latent(pipe: MyStableDiffusionPipeline, args: argparse.Namespace) -> torch.Tensor:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = next(pipe.unet.parameters()).dtype
    generator = torch.Generator(device=device).manual_seed(args.seed)
    shape = (1, pipe.unet.in_channels, args.height // 8, args.width // 8)
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


def output_fieldnames(prompt_fieldnames: list[str]) -> list[str]:
    prompt_extras = [name for name in prompt_fieldnames if name != "prompt"]
    return [
        "seed",
        "prompt_id",
        "prompt",
        *prompt_extras,
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


def should_skip_prompt(output_dir: Path, prompt_id: int, args: argparse.Namespace) -> bool:
    return args.skip_existing and (output_dir / f"{prompt_id}gen.png").exists() and (output_dir / f"{prompt_id}rec.png").exists()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_rows, prompt_fieldnames = read_prompt_csv(Path(args.prompt_csv))
    pipe = load_pipeline(args)
    inverter = Inversion(
        pipe,
        num_ddim_steps=args.num_of_ddim_steps,
        delta_threshold=args.delta_threshold,
        method=args.method,
        loss_divergence_threshold=args.loss_divergence_threshold,
    )

    shared_init_latent = sample_initial_latent(pipe, args)
    shared_init_path = output_dir / "shared_init_latent.pt"
    torch.save(shared_init_latent.detach().cpu(), shared_init_path)
    write_json(
        output_dir / "shared_init_latent_meta.json",
        {
            "seed": args.seed,
            "model_name": args.model_name,
            "height": args.height,
            "width": args.width,
            "guidance_scale": args.guidance_scale,
            "num_of_ddim_steps": args.num_of_ddim_steps,
            "latent_path": str(shared_init_path),
            "latent_stats": latent_stats(shared_init_latent),
            "note": "This one unscaled initial noise latent is reused for every prompt in this run.",
        },
    )

    traces_path = output_dir / "gen_inv_rec_traces.jsonl"
    if traces_path.exists() and not args.skip_existing:
        traces_path.unlink()

    init_latents = []
    gen_latents = []
    inv_latents = []
    rec_latents = []
    metrics_rows = []

    total_start = time.time()
    for prompt_id, prompt_row in enumerate(prompt_rows):
        prompt = prompt_row["prompt"]
        print(f"[{prompt_id + 1}/{len(prompt_rows)}] seed={args.seed} method={args.method} prompt_id={prompt_id}")

        if should_skip_prompt(output_dir, prompt_id, args):
            print(f"  skipping existing prompt_id={prompt_id}")
            continue

        init_latent = shared_init_latent.clone()
        gen_images, gen_latent = pipe(
            prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_of_ddim_steps,
            guidance_scale=args.guidance_scale,
            negative_prompt=args.negative_prompt,
            latents=init_latent.clone(),
        )
        gen_image = gen_images[0]
        gen_image_path = output_dir / f"{prompt_id}gen.png"
        gen_image.save(gen_image_path)

        inv_start = time.time()
        all_inv_latents, convergence_losses = inverter.invert(gen_latent, prompt, args.guidance_scale)
        inversion_time = time.time() - inv_start
        inv_latent = all_inv_latents[-1]

        rec_images, rec_latent = pipe(
            prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_of_ddim_steps,
            guidance_scale=args.guidance_scale,
            negative_prompt=args.negative_prompt,
            latents=inv_latent.clone(),
        )
        rec_image = rec_images[0]
        rec_image_path = output_dir / f"{prompt_id}rec.png"
        rec_image.save(rec_image_path)

        image_mse, image_psnr = image_mse_psnr(gen_image, rec_image)
        final_loss = float(convergence_losses[-1]) if convergence_losses else 0.0
        mean_loss = float(np.mean(convergence_losses)) if convergence_losses else 0.0

        init_latents.append(init_latent.detach().cpu())
        gen_latents.append(gen_latent.detach().cpu())
        inv_latents.append(inv_latent.detach().cpu())
        rec_latents.append(rec_latent.detach().cpu())

        metrics_row = {
            "seed": args.seed,
            "prompt_id": prompt_id,
            "prompt": prompt,
            **{name: prompt_row.get(name, "") for name in prompt_fieldnames if name != "prompt"},
            "method": args.method,
            "guidance_scale": args.guidance_scale,
            "num_of_ddim_steps": args.num_of_ddim_steps,
            "image_psnr": image_psnr,
            "image_mse": image_mse,
            "gen_rec_latent_mse": tensor_mse(gen_latent, rec_latent),
            "init_inv_latent_mse": tensor_mse(init_latent, inv_latent),
            "inversion_time": inversion_time,
            "inversion_final_loss": final_loss,
            "inversion_mean_loss": mean_loss,
            "gen_image_path": str(gen_image_path),
            "rec_image_path": str(rec_image_path),
        }
        metrics_rows.append(metrics_row)

        append_jsonl(
            traces_path,
            {
                "seed": args.seed,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "prompt_row": prompt_row,
                "shared_init_latent_path": str(shared_init_path),
                "gen_image_path": str(gen_image_path),
                "rec_image_path": str(rec_image_path),
                "convergence_losses": [float(x) for x in convergence_losses],
                "metrics": metrics_row,
            },
        )

    torch.save(init_latents, output_dir / "init_latents.pt")
    torch.save(gen_latents, output_dir / "gen_latents.pt")
    torch.save(inv_latents, output_dir / "inv_latents.pt")
    torch.save(rec_latents, output_dir / "rec_latents.pt")
    write_csv(output_dir / "per_prompt_fpi_metrics.csv", metrics_rows, output_fieldnames(prompt_fieldnames))

    total_time = time.time() - total_start
    write_json(
        output_dir / "run_summary.json",
        {
            "prompt_csv": args.prompt_csv,
            "output": args.output,
            "seed": args.seed,
            "method": args.method,
            "guidance_scale": args.guidance_scale,
            "num_of_ddim_steps": args.num_of_ddim_steps,
            "delta_threshold": args.delta_threshold,
            "loss_divergence_threshold": args.loss_divergence_threshold,
            "shared_init_latent_path": str(shared_init_path),
            "num_prompts": len(prompt_rows),
            "num_completed": len(metrics_rows),
            "total_time": total_time,
            "avg_time": total_time / len(metrics_rows) if metrics_rows else None,
        },
    )
    print(f"total_time: {total_time}")
    if metrics_rows:
        print(f"avg_time: {total_time / len(metrics_rows)}")
    print(f"saved shared-init prompt-grid results to: {output_dir}")


if __name__ == "__main__":
    main()
