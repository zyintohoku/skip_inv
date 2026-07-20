#!/usr/bin/env python3
"""Single-prompt SDXL 1.0 base-only Euler inversion/reconstruction test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from utils.sdxl_euler_inv_methods import (
    BASE_MODEL_ID,
    SDXLEulerInversion,
    load_sdxl_base_pipeline,
    psnr_from_mse,
    tensor_mse,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one SDXL base Euler gen-inv-rec example.")
    parser.add_argument(
        "--prompt",
        default="A small ceramic teapot on a wooden table, natural window light, detailed photo",
    )
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--method", choices=["euler", "fpi", "aidi", "afpi"], default="afpi")
    parser.add_argument("--delta-threshold", type=float, default=5e-12)
    parser.add_argument("--loss-divergence-threshold", type=float, default=0.9)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--variant", default="fp16")
    parser.add_argument("--allow-download", dest="allow_download", action="store_true", default=True)
    parser.add_argument("--local-files-only", dest="allow_download", action="store_false")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "euler_inversion_outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    variant = None if args.variant.lower() == "none" else args.variant
    pipe = load_sdxl_base_pipeline(
        model_id=BASE_MODEL_ID,
        local_files_only=not args.allow_download,
        variant=variant,
    )
    inverter = SDXLEulerInversion(
        pipe=pipe,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        delta_threshold=args.delta_threshold,
        method=args.method,
        loss_divergence_threshold=args.loss_divergence_threshold,
        max_iterations=args.max_iterations,
        show_progress=not args.no_progress,
    )
    result = inverter.gen_inv_rec(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
    )

    inverter.decode_latents(result.gen_latent)[0].save(args.output_dir / "gen.png")
    inverter.decode_latents(result.rec_latent)[0].save(args.output_dir / "rec.png")
    torch.save(result.init_latent.detach().cpu(), args.output_dir / "init_latent.pt")
    torch.save(result.inv_latent.detach().cpu(), args.output_dir / "inv_latent.pt")
    torch.save(result.gen_latent.detach().cpu(), args.output_dir / "gen_latent.pt")
    torch.save(result.rec_latent.detach().cpu(), args.output_dir / "rec_latent.pt")
    torch.save([x.detach().cpu() for x in result.gen_trace_latents], args.output_dir / "gen_trace_latents.pt")
    torch.save([x.detach().cpu() for x in result.inv_trace_latents], args.output_dir / "inv_trace_latents.pt")
    torch.save([x.detach().cpu() for x in result.rec_trace_latents], args.output_dir / "rec_trace_latents.pt")

    gen_rec_latent_mse = tensor_mse(result.gen_latent, result.rec_latent)
    summary = {
        "model": BASE_MODEL_ID,
        "scheduler": type(pipe.scheduler).__name__,
        "method": args.method,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "delta_threshold": args.delta_threshold,
        "loss_divergence_threshold": args.loss_divergence_threshold,
        "max_iterations": args.max_iterations,
        "init_inv_latent_mse": tensor_mse(result.init_latent, result.inv_latent),
        "gen_rec_latent_mse": gen_rec_latent_mse,
        "gen_rec_latent_psnr": psnr_from_mse(gen_rec_latent_mse),
        "inversion_time_seconds": result.inversion_time,
        "total_time_seconds": result.total_time,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (args.output_dir / "inversion_trace.json").open("w", encoding="utf-8") as f:
        json.dump(result.inversion_trace, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
