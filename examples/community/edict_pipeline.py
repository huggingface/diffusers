from typing import Optional

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDIMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import (
    deprecate,
)


class EDICTPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
        mixing_coeff: float = 0.93,
        leapfrog_steps: bool = True,
    ):
        self.mixing_coeff = mixing_coeff
        self.leapfrog_steps = leapfrog_steps

        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def _encode_prompt(
        self, prompt: str, negative_prompt: Optional[str] = None, do_classifier_free_guidance: bool = False
    ):
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        prompt_embeds = self.text_encoder(text_inputs.input_ids.to(self.device)).last_hidden_state

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=self.device)

        if do_classifier_free_guidance:
            uncond_tokens = "" if negative_prompt is None else negative_prompt

            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            negative_prompt_embeds = self.text_encoder(uncond_input.input_ids.to(self.device)).last_hidden_state

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def denoise_mixing_layer(self, x: torch.Tensor, y: torch.Tensor):
        x = self.mixing_coeff * x + (1 - self.mixing_coeff) * y
        y = self.mixing_coeff * y + (1 - self.mixing_coeff) * x

        return [x, y]

    def noise_mixing_layer(self, x: torch.Tensor, y: torch.Tensor):
        y = (y - (1 - self.mixing_coeff) * x) / self.mixing_coeff
        x = (x - (1 - self.mixing_coeff) * y) / self.mixing_coeff

        return [x, y]

    def _get_alpha_and_beta(self, t: torch.Tensor):
        # as self.alphas_cumprod is always in cpu
        t = int(t)

        alpha_prod = self.scheduler.alphas_cumprod[t] if t >= 0 else self.scheduler.final_alpha_cumprod

        return alpha_prod, 1 - alpha_prod

    def noise_step(
        self,
        base: torch.Tensor,
        model_input: torch.Tensor,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
    ):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps / self.scheduler.num_inference_steps

        alpha_prod_t, beta_prod_t = self._get_alpha_and_beta(timestep)
        alpha_prod_t_prev, beta_prod_t_prev = self._get_alpha_and_beta(prev_timestep)

        a_t = (alpha_prod_t_prev / alpha_prod_t) ** 0.5
        b_t = -a_t * (beta_prod_t**0.5) + beta_prod_t_prev**0.5

        next_model_input = (base - b_t * model_output) / a_t

        return model_input, next_model_input.to(base.dtype)

    def denoise_step(
        self,
        base: torch.Tensor,
        model_input: torch.Tensor,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
    ):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps / self.scheduler.num_inference_steps

        alpha_prod_t, beta_prod_t = self._get_alpha_and_beta(timestep)
        alpha_prod_t_prev, beta_prod_t_prev = self._get_alpha_and_beta(prev_timestep)

        a_t = (alpha_prod_t_prev / alpha_prod_t) ** 0.5
        b_t = -a_t * (beta_prod_t**0.5) + beta_prod_t_prev**0.5
        next_model_input = a_t * base + b_t * model_output

        return model_input, next_model_input.to(base.dtype)

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    @torch.no_grad()
    def prepare_latents(
        self,
        image: Image.Image,
        text_embeds: torch.Tensor,
        timesteps: torch.Tensor,
        guidance_scale: float,
        generator: Optional[torch.Generator] = None,
    ):
        do_classifier_free_guidance = guidance_scale > 1.0

        image = image.to(device=self.device, dtype=text_embeds.dtype)
        latent = self.vae.encode(image).latent_dist.sample(generator)

        latent = self.vae.config.scaling_factor * latent

        coupled_latents = [latent.clone(), latent.clone()]

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            coupled_latents = self.noise_mixing_layer(x=coupled_latents[0], y=coupled_latents[1])

            # j - model_input index, k - base index
            for j in range(2):
                k = j ^ 1

                if self.leapfrog_steps:
                    if i % 2 == 0:
                        k, j = j, k

                model_input = coupled_latents[j]
                base = coupled_latents[k]

                latent_model_input = torch.cat([model_input] * 2) if do_classifier_free_guidance else model_input

                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds).sample

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                base, model_input = self.noise_step(
                    base=base,
                    model_input=model_input,
                    model_output=noise_pred,
                    timestep=t,
                )

                coupled_latents[k] = model_input

        return coupled_latents

    @torch.no_grad()
    def __call__(
        self,
        base_prompt: str,
        target_prompt: str,
        image: Image.Image,
        guidance_scale: float = 3.0,
        num_inference_steps: int = 50,
        strength: float = 0.8,
        negative_prompt: Optional[str] = None,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
    ):
        do_classifier_free_guidance = guidance_scale > 1.0

        image = self.image_processor.preprocess(image)

        base_embeds = self._encode_prompt(base_prompt, negative_prompt, do_classifier_free_guidance)
        target_embeds = self._encode_prompt(target_prompt, negative_prompt, do_classifier_free_guidance)

        self.scheduler.set_timesteps(num_inference_steps, self.device)

        t_limit = num_inference_steps - int(num_inference_steps * strength)
        fwd_timesteps = self.scheduler.timesteps[t_limit:]
        bwd_timesteps = fwd_timesteps.flip(0)

        coupled_latents = self.prepare_latents(image, base_embeds, bwd_timesteps, guidance_scale, generator)

        for i, t in tqdm(enumerate(fwd_timesteps), total=len(fwd_timesteps)):
            # j - model_input index, k - base index
            for k in range(2):
                j = k ^ 1

                if self.leapfrog_steps:
                    if i % 2 == 1:
                        k, j = j, k

                model_input = coupled_latents[j]
                base = coupled_latents[k]

                latent_model_input = torch.cat([model_input] * 2) if do_classifier_free_guidance else model_input

                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=target_embeds).sample

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                base, model_input = self.denoise_step(
                    base=base,
                    model_input=model_input,
                    model_output=noise_pred,
                    timestep=t,
                )

                coupled_latents[k] = model_input

            coupled_latents = self.denoise_mixing_layer(x=coupled_latents[0], y=coupled_latents[1])

        # either one is fine
        final_latent = coupled_latents[0]

        if output_type not in ["latent", "pt", "np", "pil"]:
            deprecation_message = (
                f"the output_type {output_type} is outdated. Please make sure to set it to one of these instead: "
                "`pil`, `np`, `pt`, `latent`"
            )
            deprecate("Unsupported output_type", "1.0.0", deprecation_message, standard_warn=False)
            output_type = "np"

        if output_type == "latent":
            image = final_latent
        else:
            image = self.decode_latents(final_latent)
            image = self.image_processor.postprocess(image, output_type=output_type)

        return image
