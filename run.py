#!/usr/bin/env python3
"""Generate PIE prompts, invert generated latents, and reconstruct images."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from diffusers import DDIMScheduler

from utils.inv_methods import Inversion, MyStableDiffusionPipeline


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def make_scheduler() -> DDIMScheduler:
    return DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )


def load_mapping(path: str) -> dict:
    mapping_path = Path(path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
    with mapping_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_sample_ids(spec: str, total: int) -> list[int]:
    if spec == "all":
        return list(range(total))

    sample_ids = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid sample id range: {chunk}")
            sample_ids.extend(range(start, end + 1))
        else:
            sample_ids.append(int(chunk))

    for sample_id in sample_ids:
        if sample_id < 0 or sample_id >= total:
            raise IndexError(f"sample_id {sample_id} outside valid range [0, {total - 1}]")
    return sample_ids


def load_shared_init_latent(source_prefix: str, seed: int, device: torch.device) -> tuple[torch.Tensor, Path]:
    path = Path(f"{source_prefix}{seed}") / "init_latents.pt"
    if not path.exists():
        raise FileNotFoundError(f"Initial latent file not found: {path}")

    latents = torch.load(path, map_location="cpu")
    if isinstance(latents, list):
        if not latents:
            raise ValueError(f"No latents found in: {path}")
        latent = latents[0]
    elif torch.is_tensor(latents):
        latent = latents[0:1] if latents.ndim == 4 else latents
    else:
        raise TypeError(f"Expected list or tensor in {path}, got {type(latents)!r}")

    return latent.detach().clone().to(device), path


@torch.no_grad()
def main(
    output_dir: str = "output",
    guidance_scale: float = 7.0,
    num_of_ddim_steps: int = 50,
    delta_threshold: float = 5e-12,
    method: str = "afpi",
    loss_divergence_threshold: float = 1.0,
    mapping_file: str = "PIE_bench/mapping_file.json",
    sample_ids: str = "all",
    model_name: str = "CompVis/stable-diffusion-v1-4",
    seed: int = 0,
    source_init_prefix: str = "outputs/aidi_gs7_seed",
    device: torch.device = DEVICE,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    scheduler = make_scheduler()
    ldm_stable = MyStableDiffusionPipeline.from_pretrained(model_name, scheduler=scheduler).to(device)
    inversion = Inversion(
        ldm_stable,
        num_ddim_steps=num_of_ddim_steps,
        delta_threshold=delta_threshold,
        method=method,
        loss_divergence_threshold=loss_divergence_threshold,
    )

    editing_instruction = load_mapping(mapping_file)
    items = list(editing_instruction.items())
    selected_ids = parse_sample_ids(sample_ids, len(items))

    init_latents, inv_latents, gen_latents, rec_latents = [], [], [], []
    total_time = 0.0
    shared_init_latent, source_init_path = load_shared_init_latent(source_init_prefix, seed, device)
    torch.save(shared_init_latent.detach().cpu(), output_path / "shared_init_latent.pt")

    for position, sample_id in enumerate(selected_ids, start=1):
        _, item = items[sample_id]
        prompt = item["original_prompt"].replace("[", "").replace("]", "")
        print(f"[{position}/{len(selected_ids)}] sample_id={sample_id} method={method}")

        init_latent = shared_init_latent.clone()
        image_gen, gen_latent = ldm_stable(
            prompt=prompt,
            latents=init_latent,
            guidance_scale=guidance_scale,
            num_inference_steps=num_of_ddim_steps,
        )
        image_gen[0].save(output_path / f"{sample_id}gen.png")

        start_time = time.time()
        all_inv_latents, _convergence_losses = inversion.invert(gen_latent, prompt, guidance_scale)
        inv_latent = all_inv_latents[-1]
        total_time += time.time() - start_time

        image_rec, rec_latent = ldm_stable(
            prompt=prompt,
            latents=inv_latent,
            guidance_scale=guidance_scale,
            num_inference_steps=num_of_ddim_steps,
        )
        image_rec[0].save(output_path / f"{sample_id}rec.png")

        init_latents.append(init_latent.detach().cpu())
        inv_latents.append(inv_latent.detach().cpu())
        gen_latents.append(gen_latent.detach().cpu())
        rec_latents.append(rec_latent.detach().cpu())

    avg_time = total_time / len(selected_ids) if selected_ids else 0.0
    print("total_time:", total_time)
    print("avg_time:", avg_time)

    torch.save(init_latents, output_path / "init_latents.pt")
    torch.save(inv_latents, output_path / "inv_latents.pt")
    torch.save(gen_latents, output_path / "gen_latents.pt")
    torch.save(rec_latents, output_path / "rec_latents.pt")

    summary = {
        "output_dir": output_dir,
        "mapping_file": mapping_file,
        "model_name": model_name,
        "method": method,
        "guidance_scale": guidance_scale,
        "num_of_ddim_steps": num_of_ddim_steps,
        "delta_threshold": delta_threshold,
        "loss_divergence_threshold": loss_divergence_threshold,
        "sample_ids": selected_ids,
        "num_samples": len(selected_ids),
        "shared_init_latent_path": str(output_path / "shared_init_latent.pt"),
        "source_init_latent_path": str(source_init_path),
        "source_init_latent_index": 0,
        "total_time": total_time,
        "avg_time": avg_time,
    }
    with (output_path / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_ddim_steps", type=int, default=50)
    parser.add_argument("--delta_threshold", type=float, default=5e-12)
    parser.add_argument("--loss_divergence_threshold", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--output", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", type=str, default="afpi", choices=["ddim", "fpi", "afpi", "aidi"])
    parser.add_argument("--mapping_file", type=str, default="PIE_bench/mapping_file.json")
    parser.add_argument("--sample_ids", type=str, default="all")
    parser.add_argument("--model_name", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--source_init_prefix", type=str, default="outputs/aidi_gs7_seed")

    # Accepted for compatibility with older shell scripts; current Inversion no longer uses them.
    parser.add_argument("--K_round", type=int, default=50, help=argparse.SUPPRESS)
    parser.add_argument("--conv_check", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--fp_th", type=float, default=0.7, help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    main(
        output_dir=args.output,
        guidance_scale=args.guidance_scale,
        num_of_ddim_steps=args.num_of_ddim_steps,
        delta_threshold=args.delta_threshold,
        method=args.method,
        loss_divergence_threshold=args.loss_divergence_threshold,
        mapping_file=args.mapping_file,
        sample_ids=args.sample_ids,
        model_name=args.model_name,
        seed=args.seed,
        source_init_prefix=args.source_init_prefix,
    )
