from diffusers import StableDiffusionPipeline
from typing import Callable, List, Optional, Union
import torch
import torch.nn.functional as F
from utils.inv_methods import Inversion as BaseInversion

class MyStableDiffusionPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        cfg_schedule: List[int] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        self.check_inputs(prompt, height, width, callback_steps)

        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        do_classifier_free_guidance = True

        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        if cfg_schedule is not None:
            # Inversion records CFG from timestep 0 -> 50, while generation runs 50 -> 0.
            # Reverse on a copy so reconstruction aligns schedules without mutating saved data.
            cfg_schedule = list(reversed(cfg_schedule))
            
        for i, t in enumerate(timesteps):
            if cfg_schedule is not None:
                guidance_scale = cfg_schedule[i]

            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

        image = self.decode_latents(latents)
        image = self.numpy_to_pil(image)
        return image, latents

class Inversion(BaseInversion):
    @torch.no_grad()
    def invert(self, latent, prompt, guidance_scale, sample_id=None, verbose=False):
        self.init_prompt(prompt)
        latent = latent.clone().detach()
        all_latent = [latent]
        
        timesteps = self.model.scheduler.timesteps
        total_steps = self.num_ddim_steps
        cfg_schedule = []
        current_gs = guidance_scale
        
        for i in range(total_steps):
            latent_temp = latent.clone().detach()
            t = timesteps[-i - 1]
            if self.reset_gs:
                current_gs = guidance_scale
            
            while current_gs >= 1:
                latent_input = torch.cat([latent_temp] * 2)
                noise_pred = self.get_noise_pred_single(latent_input, t, self.context)
                noise_uncond, noise_cond = noise_pred.chunk(2)
                guided_noise = noise_uncond + current_gs * (noise_cond - noise_uncond)

                latent = self.next_step(guided_noise, t, latent_temp)
                latent, converged = self.afpi_step(latent, latent_temp, t, current_gs, step_index=i)

                if converged:
                    if verbose:
                        print(f"Step {i:2d} | t={t.item():4d} | GS={current_gs} ✓")
                    all_latent.append(latent)
                    cfg_schedule.append(current_gs)
                    break
                elif current_gs == 1:
                    if verbose:
                        print(f"Step {i:2d} | t={t.item():4d} | GS={current_gs} ✗ (failed)")
                    else:
                        print(f"Failed to converge at step {i}, sample_id={sample_id}")
                    all_latent.append(latent)
                    cfg_schedule.append(current_gs)
                    break
                else:
                    if verbose:
                        print(f"Step {i:2d} | t={t.item():4d} | GS={current_gs} → {current_gs-1}")
                    current_gs -= 1
                    
        return all_latent, cfg_schedule

    def afpi_step(self, init_latent, latent_ztm1, t, guidance_scale, step_index=None):
        """
        Adaptive fixed point iteration step.
        
        Args:
            init_latent: Initial latent from DDIM inversion
            latent_ztm1: Latent at time step t-1
            t: Current timestep
            guidance_scale: Guidance scale value
            step_index: Current step index (0-based). If provided and less than 
                       force_converge_before_step, will always return converged=True
        
        Returns:
            optimal_latent: Optimized latent
            converged: Whether the optimization converged
        """
        optimal_latent = init_latent.clone().detach()
        
        # Check if we should force convergence (skip full optimization)
        force_converge = (step_index is not None and 
                         self.force_converge_before_step is not None and 
                         step_index < self.force_converge_before_step)
        
        alpha = 1.0
        loss_prev = 1.0
        converged = False
        while True:
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
        # Force convergence for early steps if specified
        if force_converge or loss < self.threshold:
            converged = True
        else:
            converged = False
            
        return optimal_latent.detach(), converged

    def __init__(self, model, num_ddim_steps=50, delta_threshold=5e-9, loss_divergence_threshold=0.9, reset_gs=False, force_converge_before_step=10):
        """
        Initialize Inversion class.
        
        Args:
            model: Stable Diffusion model
            num_ddim_steps: Number of DDIM steps
            delta_threshold: Convergence threshold for loss
            loss_divergence_threshold: Threshold for detecting loss divergence
            reset_gs: Whether to reset guidance scale at each step
            force_converge_before_step: If set, always return converged=True for steps < this value.
                                       This can speed up early inversion steps where high precision
                                       is less critical. Default is None (no forced convergence).
        """
        self.model = model
        self.scheduler = model.scheduler
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(num_ddim_steps)
        self.prompt = None
        self.context = None
        self.num_ddim_steps = num_ddim_steps
        self.threshold = delta_threshold
        self.loss_divergence_threshold = loss_divergence_threshold
        self.reset_gs = reset_gs
        self.force_converge_before_step = force_converge_before_step
