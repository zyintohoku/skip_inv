from typing import Optional, Union, List, Callable, Tuple
import torch
from diffusers import StableDiffusionPipeline
import numpy as np
import torch.nn.functional as F
from PIL import Image

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def _pil_resample_lanczos():
    return getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def preprocess_image_for_vae(
    image: Union[str, Image.Image],
    height: int = 512,
    width: int = 512,
) -> Tuple[torch.FloatTensor, Image.Image]:
    if isinstance(image, str):
        image = Image.open(image)
    image = image.convert("RGB").resize((width, height), _pil_resample_lanczos())
    array = np.asarray(image).astype(np.float32) / 255.0
    array = array[None].transpose(0, 3, 1, 2)
    tensor = torch.from_numpy(array)
    tensor = 2.0 * tensor - 1.0
    return tensor, image


@torch.no_grad()
def encode_image_to_latent(
    pipe: StableDiffusionPipeline,
    image: Union[str, Image.Image],
    height: int = 512,
    width: int = 512,
    latent_mode: str = "mean",
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.FloatTensor, Image.Image]:
    image_tensor, processed_image = preprocess_image_for_vae(image, height=height, width=width)
    vae_device = next(pipe.vae.parameters()).device
    vae_dtype = next(pipe.vae.parameters()).dtype
    image_tensor = image_tensor.to(device=vae_device, dtype=vae_dtype)
    posterior = pipe.vae.encode(image_tensor).latent_dist
    if latent_mode == "mean":
        latent = posterior.mean
    elif latent_mode == "sample":
        latent = posterior.sample(generator=generator)
    else:
        raise ValueError(f"Unsupported latent_mode: {latent_mode}")
    scaling_factor = getattr(pipe.vae.config, "scaling_factor", 0.18215)
    return latent * scaling_factor, processed_image

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
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        #do_classifier_free_guidance = guidance_scale > 1.0
        do_classifier_free_guidance = True

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
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

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            #losses=[]
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    #losses.append(torch.nn.functional.mse_loss(noise_pred_uncond, noise_pred_text).item())
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        # 8. Post-processing
        #if not output_type == "latent":
        image = self.decode_latents(latents)
        #else:
        #    return latents
        
        # 9. Run safety checker
        #image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)
        # 10. Convert to PIL
        #if output_type == "pil":
        image = self.numpy_to_pil(image)

        #if not return_dict:
        #    return (image, has_nsfw_concept)
        #return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
        return image, latents
    
class Inversion:
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def loop(self, latent, guidance_scale):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        latent = latent.clone().detach()
        all_latent = [latent]
        convergence_losses = []  # Track final loss at each timestep
        
        timesteps = self.model.scheduler.timesteps
        total_steps = self.num_ddim_steps
        for i in range(total_steps):
            t = timesteps[-i - 1]
            latent_input = torch.cat([latent] * 2)

            noise_pred = self.get_noise_pred_single(latent_input, t, self.context)
            noise_uncond, noise_cond = noise_pred.chunk(2)
            guided_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            latent_ztm1 = latent
            latent = self.next_step(guided_noise, t, latent_ztm1)

            ################ optimization steps #################
            final_loss = None
            if self.method=='afpi':
                latent, final_loss = self.afpi_step(latent, latent_ztm1, t, guidance_scale)
            elif self.method=='aidi':
                latent, final_loss = self.aidi_step(latent, latent_ztm1, t, guidance_scale)
            elif self.method=='fpi':
                latent, final_loss = self.fpi_step(latent, latent_ztm1, t, guidance_scale)
            elif self.method=='ddim':
                all_latent.append(latent)
                convergence_losses.append(0.0)  # DDIM doesn't iterate
                continue

            all_latent.append(latent)
            convergence_losses.append(final_loss if final_loss is not None else 0.0)
        
        return all_latent, convergence_losses

    def fpi_step(self, init_latent, latent_ztm1, t, guidance_scale):
        optimal_latent = init_latent.clone().detach()
        
        loss_prev = 1.0
        final_loss = 0.0
        #print(t)
        iterations = 0
        while True:
            iterations += 1
            latent_input = torch.cat([optimal_latent] * 2)
            noise_pred = self.get_noise_pred_single(latent_input, t, self.context)
            noise_uncond, noise_cond = noise_pred.chunk(2)
            guided_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            updated_latent = self.next_step(guided_noise, t, latent_ztm1)
            loss = F.mse_loss(updated_latent, optimal_latent).item()
            final_loss = loss
            #print(loss, F.mse_loss(noise_uncond, noise_cond).item())
            if loss < self.threshold:
                break
            if loss > loss_prev:
                break
            if self.max_iterations is not None and iterations >= self.max_iterations:
                break
            optimal_latent = updated_latent
            loss_prev = loss
        return optimal_latent.detach(), final_loss

    def aidi_step(self, init_latent, latent_ztm1, t, guidance_scale):
        optimal_latent = init_latent.clone().detach()
        
        loss_prev = 1.0
        final_loss = 0.0
        #print()
        while True:
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
            optimal_latent = 0.5 * optimal_latent + 0.5 * updated_latent
            loss_prev = loss
        #print(t.item(), loss)
        return optimal_latent.detach(), final_loss

    def afpi_step(self, init_latent, latent_ztm1, t, guidance_scale):
        optimal_latent = init_latent.clone().detach()
        
        alpha = 1.0
        loss_prev = 1.0
        final_loss = 0.0
        while True:
            latent_input = torch.cat([optimal_latent] * 2)
            noise_pred = self.get_noise_pred_single(latent_input, t, self.context)
            noise_uncond, noise_cond = noise_pred.chunk(2)
            guided_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            updated_latent = self.next_step(guided_noise, t, latent_ztm1)
            loss = F.mse_loss(updated_latent, optimal_latent).item()
            final_loss = loss
            if loss < self.threshold:
                break
            if loss > loss_prev and alpha == 0.5:
                break
            if loss > loss_prev * self.loss_divergence_threshold:
                alpha = 0.5
            optimal_latent = (1 - alpha) * optimal_latent + alpha * updated_latent
            loss_prev = loss
        #print(t.item(), loss)
        return optimal_latent.detach(), final_loss
                
    def invert(self, image_latent, prompt: str, guidance_scale):
        self.init_prompt(prompt)
        all_latent, convergence_losses = self.loop(image_latent, guidance_scale)
        return all_latent, convergence_losses

    def __init__(
        self,
        model,
        num_ddim_steps=50,
        delta_threshold=5e-12,
        method='afpi',
        loss_divergence_threshold=1.0,
        max_iterations=None,
    ):
        self.model = model
        self.scheduler = model.scheduler
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(num_ddim_steps)
        self.prompt = None
        self.context = None
        self.num_ddim_steps = num_ddim_steps
        self.threshold = delta_threshold
        self.method = method
        self.loss_divergence_threshold = loss_divergence_threshold
        self.max_iterations = max_iterations
