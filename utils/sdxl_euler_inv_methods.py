from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from tqdm import tqdm


BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
EulerInversionMethod = Literal["euler", "fpi", "aidi", "afpi"]


@dataclass
class SDXLConditioning:
    prompt_embeds: torch.Tensor
    add_text_embeds: torch.Tensor
    add_time_ids: torch.Tensor
    timestep_cond: torch.Tensor | None


@dataclass
class SDXLEulerRunResult:
    init_latent: torch.Tensor
    gen_latent: torch.Tensor
    inv_latent: torch.Tensor
    rec_latent: torch.Tensor
    gen_trace_latents: List[torch.Tensor]
    inv_trace_latents: List[torch.Tensor]
    rec_trace_latents: List[torch.Tensor]
    inversion_trace: List[dict]
    inversion_time: float
    total_time: float


def load_sdxl_base_pipeline(
    model_id: str = BASE_MODEL_ID,
    local_files_only: bool = False,
    variant: str | None = "fp16",
    device: torch.device | str | None = None,
) -> StableDiffusionXLPipeline:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    pipe_kwargs = {
        "torch_dtype": dtype,
        "use_safetensors": True,
        "local_files_only": local_files_only,
    }
    if variant is not None:
        pipe_kwargs["variant"] = variant

    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, **pipe_kwargs)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def tensor_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.mse_loss(a.detach().float().cpu(), b.detach().float().cpu()).item()


def psnr_from_mse(mse: float, max_value: float = 1.0) -> float:
    if mse <= 0.0:
        return math.inf
    return 20.0 * math.log10(max_value / math.sqrt(mse))


def image_mse_psnr(image_a: Path, image_b: Path) -> Tuple[float, float]:
    with Image.open(image_a) as img_a, Image.open(image_b) as img_b:
        arr_a = np.asarray(img_a.convert("RGB"), dtype=np.float32)
        arr_b = np.asarray(img_b.convert("RGB"), dtype=np.float32)
    mse = float(np.mean((arr_a - arr_b) ** 2))
    return mse, psnr_from_mse(mse, max_value=255.0)


class SDXLEulerInversion:
    def __init__(
        self,
        pipe: StableDiffusionXLPipeline,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.0,
        delta_threshold: float = 5e-12,
        method: EulerInversionMethod = "afpi",
        loss_divergence_threshold: float = 0.9,
        max_iterations: int = 50,
        show_progress: bool = True,
    ) -> None:
        self.pipe = pipe
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.delta_threshold = delta_threshold
        self.method = method
        self.loss_divergence_threshold = loss_divergence_threshold
        self.max_iterations = max_iterations
        self.show_progress = show_progress
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)

    @property
    def device(self) -> torch.device:
        return self.pipe._execution_device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.pipe.unet.parameters()).dtype

    def prepare_conditioning(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = 1024,
        width: int = 1024,
    ) -> SDXLConditioning:
        pipe = self.pipe
        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
            negative_prompt_2=None,
        )

        add_text_embeds = pooled_prompt_embeds
        text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim
        add_time_ids = pipe._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        timestep_cond = None
        if pipe.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(1).to(self.device)
            timestep_cond = pipe.get_guidance_scale_embedding(
                guidance_scale_tensor,
                embedding_dim=pipe.unet.config.time_cond_proj_dim,
            ).to(device=self.device, dtype=prompt_embeds.dtype)

        return SDXLConditioning(
            prompt_embeds=prompt_embeds.to(self.device),
            add_text_embeds=add_text_embeds.to(self.device),
            add_time_ids=add_time_ids.to(self.device),
            timestep_cond=timestep_cond,
        )

    def prepare_initial_latent(
        self,
        height: int,
        width: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=self.pipe.unet.config.in_channels,
            height=height,
            width=width,
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )

    def predict_guided_noise(
        self,
        conditioning: SDXLConditioning,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        sigma = self.pipe.scheduler.sigmas[step_index].to(device=latents.device, dtype=latents.dtype)
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = latent_model_input / torch.sqrt(sigma**2 + 1)
        added_cond_kwargs = {
            "text_embeds": conditioning.add_text_embeds,
            "time_ids": conditioning.add_time_ids,
        }

        noise_pred = self.pipe.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=conditioning.prompt_embeds,
            timestep_cond=conditioning.timestep_cond,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        noise_uncond, noise_text = noise_pred.chunk(2)
        return noise_uncond + self.guidance_scale * (noise_text - noise_uncond)

    def euler_forward_step(
        self,
        noise_pred: torch.Tensor,
        latents: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        sigma = self.pipe.scheduler.sigmas[step_index].to(device=latents.device, dtype=torch.float32)
        sigma_next = self.pipe.scheduler.sigmas[step_index + 1].to(device=latents.device, dtype=torch.float32)
        prev_sample = latents.to(torch.float32) + noise_pred.to(torch.float32) * (sigma_next - sigma)
        return prev_sample.to(noise_pred.dtype)

    def euler_inverse_step(
        self,
        noise_pred: torch.Tensor,
        lower_sigma_latents: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        sigma = self.pipe.scheduler.sigmas[step_index].to(device=lower_sigma_latents.device, dtype=torch.float32)
        sigma_next = self.pipe.scheduler.sigmas[step_index + 1].to(
            device=lower_sigma_latents.device,
            dtype=torch.float32,
        )
        higher_sigma_sample = lower_sigma_latents.to(torch.float32) - noise_pred.to(torch.float32) * (
            sigma_next - sigma
        )
        return higher_sigma_sample.to(noise_pred.dtype)

    @torch.no_grad()
    def generate(
        self,
        conditioning: SDXLConditioning,
        init_latent: torch.Tensor,
        desc: str = "Generate",
    ) -> tuple[torch.Tensor, List[torch.Tensor]]:
        latents = init_latent
        all_latents = [latents.detach().clone()]
        iterator = enumerate(self.pipe.scheduler.timesteps)
        if self.show_progress:
            iterator = enumerate(tqdm(self.pipe.scheduler.timesteps, desc=desc))

        for step_index, timestep in iterator:
            noise_pred = self.predict_guided_noise(conditioning, latents, timestep, step_index)
            latents = self.euler_forward_step(noise_pred, latents, step_index)
            all_latents.append(latents.detach().clone())

        return latents, all_latents

    def fixed_point_inverse_step(
        self,
        conditioning: SDXLConditioning,
        init_latent: torch.Tensor,
        lower_sigma_latents: torch.Tensor,
        timestep: torch.Tensor,
        step_index: int,
    ) -> tuple[torch.Tensor, float, int]:
        optimal_latent = init_latent.clone().detach()
        loss_prev = 1.0
        alpha = 1.0
        final_loss = 0.0

        for iteration in range(1, self.max_iterations + 1):
            noise_pred = self.predict_guided_noise(conditioning, optimal_latent, timestep, step_index)
            updated_latent = self.euler_inverse_step(noise_pred, lower_sigma_latents, step_index)
            loss = F.mse_loss(updated_latent.float(), optimal_latent.float()).item()
            final_loss = loss

            if loss < self.delta_threshold:
                return optimal_latent.detach(), final_loss, iteration
            if loss > loss_prev and self.method in {"fpi", "aidi"}:
                return optimal_latent.detach(), final_loss, iteration
            if self.method == "afpi":
                if loss > loss_prev and alpha == 0.5:
                    return optimal_latent.detach(), final_loss, iteration
                if loss > loss_prev * self.loss_divergence_threshold:
                    alpha = 0.5

            if self.method == "aidi":
                optimal_latent = 0.5 * optimal_latent + 0.5 * updated_latent
            elif self.method == "afpi":
                optimal_latent = (1 - alpha) * optimal_latent + alpha * updated_latent
            else:
                optimal_latent = updated_latent

            loss_prev = loss

        return optimal_latent.detach(), final_loss, self.max_iterations

    @torch.no_grad()
    def invert(
        self,
        conditioning: SDXLConditioning,
        image_latent: torch.Tensor,
        desc: str = "Invert",
    ) -> tuple[torch.Tensor, List[torch.Tensor], List[dict]]:
        latents = image_latent.clone().detach()
        all_latents = [latents.detach().clone()]
        traces: List[dict] = []
        timesteps = self.pipe.scheduler.timesteps
        step_range = range(len(timesteps) - 1, -1, -1)
        if self.show_progress:
            step_range = tqdm(step_range, desc=desc)

        for reverse_i in step_range:
            timestep = timesteps[reverse_i]
            noise_pred = self.predict_guided_noise(conditioning, latents, timestep, reverse_i)
            init_latent = self.euler_inverse_step(noise_pred, latents, reverse_i)

            if self.method == "euler":
                latents = init_latent
                final_loss = 0.0
                iterations = 1
            else:
                latents, final_loss, iterations = self.fixed_point_inverse_step(
                    conditioning=conditioning,
                    init_latent=init_latent,
                    lower_sigma_latents=latents,
                    timestep=timestep,
                    step_index=reverse_i,
                )

            all_latents.append(latents.detach().clone())
            traces.append(
                {
                    "reverse_step": len(timesteps) - 1 - reverse_i,
                    "forward_step_index": reverse_i,
                    "timestep": float(timestep.item()),
                    "sigma": float(self.pipe.scheduler.sigmas[reverse_i].item()),
                    "sigma_next": float(self.pipe.scheduler.sigmas[reverse_i + 1].item()),
                    "final_loss": float(final_loss),
                    "iterations": int(iterations),
                }
            )

        return latents, all_latents, traces

    @torch.no_grad()
    def gen_inv_rec(
        self,
        prompt: str,
        negative_prompt: str,
        height: int,
        width: int,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> SDXLEulerRunResult:
        self.pipe.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        conditioning = self.prepare_conditioning(prompt, negative_prompt, height, width)
        if generator is None:
            if seed is None:
                raise ValueError("Either seed or generator must be provided.")
            generator = torch.Generator(device=self.device).manual_seed(seed)
        init_latent = self.prepare_initial_latent(height, width, generator)

        start_time = time.time()
        gen_latent, gen_trace_latents = self.generate(conditioning, init_latent, desc="Generate")
        inversion_start = time.time()
        inv_latent, inv_trace_latents, inversion_trace = self.invert(conditioning, gen_latent, desc="Invert")
        inversion_time = time.time() - inversion_start
        rec_latent, rec_trace_latents = self.generate(conditioning, inv_latent, desc="Reconstruct")
        total_time = time.time() - start_time

        return SDXLEulerRunResult(
            init_latent=init_latent,
            gen_latent=gen_latent,
            inv_latent=inv_latent,
            rec_latent=rec_latent,
            gen_trace_latents=gen_trace_latents,
            inv_trace_latents=inv_trace_latents,
            rec_trace_latents=rec_trace_latents,
            inversion_trace=inversion_trace,
            inversion_time=inversion_time,
            total_time=total_time,
        )

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor):
        pipe = self.pipe
        needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
        decode_latents_tensor = latents

        if needs_upcasting:
            pipe.upcast_vae()
            decode_latents_tensor = decode_latents_tensor.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
        elif decode_latents_tensor.dtype != pipe.vae.dtype:
            decode_latents_tensor = decode_latents_tensor.to(pipe.vae.dtype)

        has_latents_mean = hasattr(pipe.vae.config, "latents_mean") and pipe.vae.config.latents_mean is not None
        has_latents_std = hasattr(pipe.vae.config, "latents_std") and pipe.vae.config.latents_std is not None
        if has_latents_mean and has_latents_std:
            latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, 4, 1, 1).to(
                decode_latents_tensor.device,
                decode_latents_tensor.dtype,
            )
            latents_std = torch.tensor(pipe.vae.config.latents_std).view(1, 4, 1, 1).to(
                decode_latents_tensor.device,
                decode_latents_tensor.dtype,
            )
            decode_latents_tensor = decode_latents_tensor * latents_std / pipe.vae.config.scaling_factor + latents_mean
        else:
            decode_latents_tensor = decode_latents_tensor / pipe.vae.config.scaling_factor

        image = pipe.vae.decode(decode_latents_tensor, return_dict=False)[0]
        if needs_upcasting:
            pipe.vae.to(dtype=torch.float16)
        return pipe.image_processor.postprocess(image, output_type="pil")
