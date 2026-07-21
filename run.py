#!/usr/bin/env python3
"""Generate PIE prompts, invert generated latents, and reconstruct images."""

import argparse
import json
import os
import time

import torch
from diffusers import DDIMScheduler

from utils.inv_methods import Inversion, MyStableDiffusionPipeline


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def parse_sample_ids(spec, total):
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


@torch.no_grad()
def main(
    output_dir="output",
    guidance_scale=7.0,
    num_of_ddim_steps=50,
    delta_threshold=5e-12,
    method="afpi",
    loss_divergence_threshold=1.0,
    mapping_file="PIE_bench/mapping_file.json",
    sample_ids="all",
    model_name="CompVis/stable-diffusion-v1-4",
    **kwargs,
):
    os.makedirs(output_dir, exist_ok=True)

    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    ldm_stable = MyStableDiffusionPipeline.from_pretrained(model_name, scheduler=scheduler).to(device)
    inversion = Inversion(
        ldm_stable,
        num_ddim_steps=num_of_ddim_steps,
        delta_threshold=delta_threshold,
        method=method,
        loss_divergence_threshold=loss_divergence_threshold,
    )

    with open(mapping_file, "r", encoding="utf-8") as f:
        editing_instruction = json.load(f)

    items = list(editing_instruction.items())
    selected_ids = parse_sample_ids(sample_ids, len(items))

    init_latents, inv_latents, gen_latents, rec_latents = [], [], [], []
    total_time = 0.0
    for sample_id in selected_ids:
        _, item = items[sample_id]
        prompt = item["original_prompt"].replace("[", "").replace("]", "")
        init_latent = torch.randn(1, 4, 64, 64).to(device)
        image_gen, gen_latent = ldm_stable(
            prompt=prompt,
            latents=init_latent,
            guidance_scale=7,
            num_inference_steps=num_of_ddim_steps,
        )
        image_gen[0].save(f"{output_dir}/{sample_id}gen.png")

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
        image_rec[0].save(f"{output_dir}/{sample_id}rec.png")

        init_latents.append(init_latent)
        inv_latents.append(inv_latent)
        gen_latents.append(gen_latent)
        rec_latents.append(rec_latent)

    print("total_time:", total_time)
    print("avg_time:", total_time / len(selected_ids) if selected_ids else 0.0)
    torch.save(init_latents, f"{output_dir}/init_latents.pt")
    torch.save(inv_latents, f"{output_dir}/inv_latents.pt")
    torch.save(gen_latents, f"{output_dir}/gen_latents.pt")
    torch.save(rec_latents, f"{output_dir}/rec_latents.pt")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_ddim_steps", type=int, default=50, help="Blended word needed for P2P")
    parser.add_argument("--delta_threshold", type=float, default=5e-12, help="Delta threshold")
    parser.add_argument("--loss_divergence_threshold", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--output", type=str, default="outputs", help="Save editing results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", type=str, default="afpi", choices=["ddim", "fpi", "afpi", "aidi"])
    parser.add_argument("--mapping_file", type=str, default="PIE_bench/mapping_file.json")
    parser.add_argument("--sample_ids", type=str, default="all")
    parser.add_argument("--model_name", type=str, default="CompVis/stable-diffusion-v1-4")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    params = {}
    params["guidance_scale"] = args.guidance_scale
    params["num_of_ddim_steps"] = args.num_of_ddim_steps
    params["delta_threshold"] = args.delta_threshold
    params["loss_divergence_threshold"] = args.loss_divergence_threshold
    params["output_dir"] = args.output
    params["method"] = args.method
    params["mapping_file"] = args.mapping_file
    params["sample_ids"] = args.sample_ids
    params["model_name"] = args.model_name
    torch.manual_seed(args.seed)
    main(**params)
