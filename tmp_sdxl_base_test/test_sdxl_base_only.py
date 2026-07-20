#!/usr/bin/env python3
"""Minimal SDXL 1.0 base-only generation smoke test.

This script intentionally loads only:
    stabilityai/stable-diffusion-xl-base-1.0

It does not instantiate or call the SDXL refiner.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline


BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal SDXL 1.0 base-only generation test.")
    parser.add_argument(
        "--prompt",
        default="A small ceramic teapot on a wooden table, natural window light, detailed photo",
        help="Text prompt for SDXL base generation.",
    )
    parser.add_argument(
        "--negative-prompt",
        default="",
        help="Negative prompt.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--steps", type=int, default=25, help="Number of denoising steps.")
    parser.add_argument("--guidance-scale", type=float, default=7.0, help="Classifier-free guidance scale.")
    parser.add_argument("--height", type=int, default=1024, help="Output image height.")
    parser.add_argument("--width", type=int, default=1024, help="Output image width.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "sdxl_base_test.png",
        help="Where to save the generated image.",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Use accelerate CPU offload to reduce VRAM usage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe_kwargs = {
        "torch_dtype": dtype,
        "use_safetensors": True,
    }
    if dtype == torch.float16:
        pipe_kwargs["variant"] = "fp16"

    pipe = StableDiffusionXLPipeline.from_pretrained(BASE_MODEL_ID, **pipe_kwargs)

    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
        generator_device = "cpu"
    else:
        pipe = pipe.to(device)
        generator_device = device

    generator = torch.Generator(device=generator_device).manual_seed(args.seed)

    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        generator=generator,
    ).images[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"Saved SDXL base-only output to: {args.output}")


if __name__ == "__main__":
    main()
