from typing import Optional, Union

import numpy as np
import torch
from diffusers import DiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


def preprocess(image):
    if isinstance(image, Image.Image):
        w, h = image.size
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8

        image = np.array(image.resize((w, h), resample=Image.Resampling.LANCZOS))[None, :]
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    else:
        raise TypeError("Expected object of type PIL.Image.Image")
    return image


class EDICTPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
        p: float = 0.93,
        leapfrog_steps: bool = True,
    ):
        self.leapfrog_steps = leapfrog_steps
        self.p = p
        
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )


    def encode_prompt(self, prompt: str, negative_prompt: Optional[str] = None):
        negative_prompt = "" if negative_prompt is None else negative_prompt

        tokens_uncond = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        embeds_uncond = self.text_encoder(
            tokens_uncond.input_ids.to(self.device)
        ).last_hidden_state

        tokens_cond = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        embeds_cond = self.text_encoder(
            tokens_cond.input_ids.to(self.device)
        ).last_hidden_state

        return torch.cat([embeds_uncond, embeds_cond])

    def denoise_mixing_layer(self, x: torch.Tensor, y: torch.Tensor):
        x = self.p * x + (1 - self.p) * y
        y = self.p * y + (1 - self.p) * x

        return [x, y]

    def noise_mixing_layer(self, x: torch.Tensor, y: torch.Tensor):
        y = (y - (1 - self.p) * x) / self.p
        x = (x - (1 - self.p) * y) / self.p

        return [x, y]

    def get_alpha_and_beta(self, t: torch.Tensor):
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

        alpha_prod_t, beta_prod_t = self.get_alpha_and_beta(timestep)
        alpha_prod_t_prev, beta_prod_t_prev = self.get_alpha_and_beta(prev_timestep)

        a_t = (alpha_prod_t_prev / alpha_prod_t) ** 0.5
        b_t = -a_t * (beta_prod_t**0.5) + beta_prod_t_prev ** 0.5

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

        alpha_prod_t, beta_prod_t = self.get_alpha_and_beta(timestep)
        alpha_prod_t_prev, beta_prod_t_prev = self.get_alpha_and_beta(prev_timestep)

        a_t = (alpha_prod_t_prev / alpha_prod_t) ** 0.5
        b_t = -a_t * (beta_prod_t**0.5) + beta_prod_t_prev**0.5
        next_model_input = a_t * base + b_t * model_output

        return model_input, next_model_input.to(base.dtype)
    
    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor):
        # latents = 1 / self.vae.config.scaling_factor * latents
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    @torch.no_grad()
    def prepare_latents(
        self,
        image: Image.Image,
        text_embeds: torch.Tensor,
        timesteps: torch.Tensor,
        guidance_scale: float,
    ):
        generator = torch.cuda.manual_seed(1)
        image = image.to(device=self.device, dtype=text_embeds.dtype)
        latent = self.vae.encode(image).latent_dist.sample(generator)

        # init_latents = self.vae.config.scaling_factor * init_latents
        latent = 0.18215 * latent

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

                latent_model_input = torch.cat([model_input] * 2)

                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds).sample

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

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
    ):

        image = preprocess(image)  # from PIL.Image to torch.Tensor

        base_embeds = self.encode_prompt(base_prompt, negative_prompt)
        target_embeds = self.encode_prompt(target_prompt, negative_prompt)

        self.scheduler.set_timesteps(num_inference_steps, self.device)

        t_limit = num_inference_steps - int(num_inference_steps * strength)
        fwd_timesteps = self.scheduler.timesteps[t_limit:]
        bwd_timesteps = fwd_timesteps.flip(0)

        coupled_latents = self.prepare_latents(image, base_embeds, bwd_timesteps, guidance_scale)

        for i, t in tqdm(enumerate(fwd_timesteps), total=len(fwd_timesteps)):
            # j - model_input index, k - base index
            for k in range(2):
                j = k ^ 1

                if self.leapfrog_steps:
                    if i % 2 == 1:
                        k, j = j, k

                model_input = coupled_latents[j]
                base = coupled_latents[k]

                latent_model_input = torch.cat([model_input] * 2)

                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=target_embeds).sample
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

        image = self.decode_latents(final_latent)
        image = (image[0] * 255).round().astype("uint8")
        pil_image = Image.fromarray(image)

        return pil_image