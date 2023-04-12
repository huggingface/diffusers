from typing import Union

import torch
from PIL import Image
from torchvision import transforms as tfms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
)


class MagicMixPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[PNDMScheduler, LMSDiscreteScheduler, DDIMScheduler],
    ):
        super().__init__()

        self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler)

    # convert PIL image to latents
    def encode(self, img):
        with torch.no_grad():
            latent = self.vae.encode(tfms.ToTensor()(img).unsqueeze(0).to(self.device) * 2 - 1)
            latent = 0.18215 * latent.latent_dist.sample()
        return latent

    # convert latents to PIL image
    def decode(self, latent):
        latent = (1 / 0.18215) * latent
        with torch.no_grad():
            img = self.vae.decode(latent).sample
        img = (img / 2 + 0.5).clamp(0, 1)
        img = img.detach().cpu().permute(0, 2, 3, 1).numpy()
        img = (img * 255).round().astype("uint8")
        return Image.fromarray(img[0])

    # convert prompt into text embeddings, also unconditional embeddings
    def prep_text(self, prompt):
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_embedding = self.text_encoder(text_input.input_ids.to(self.device))[0]

        uncond_input = self.tokenizer(
            "",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        uncond_embedding = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        return torch.cat([uncond_embedding, text_embedding])

    def __call__(
        self,
        img: Image.Image,
        prompt: str,
        kmin: float = 0.3,
        kmax: float = 0.6,
        mix_factor: float = 0.5,
        seed: int = 42,
        steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> Image.Image:
        tmin = steps - int(kmin * steps)
        tmax = steps - int(kmax * steps)

        text_embeddings = self.prep_text(prompt)

        self.scheduler.set_timesteps(steps)

        width, height = img.size
        encoded = self.encode(img)

        torch.manual_seed(seed)
        noise = torch.randn(
            (1, self.unet.config.in_channels, height // 8, width // 8),
        ).to(self.device)

        latents = self.scheduler.add_noise(
            encoded,
            noise,
            timesteps=self.scheduler.timesteps[tmax],
        )

        input = torch.cat([latents] * 2)

        input = self.scheduler.scale_model_input(input, self.scheduler.timesteps[tmax])

        with torch.no_grad():
            pred = self.unet(
                input,
                self.scheduler.timesteps[tmax],
                encoder_hidden_states=text_embeddings,
            ).sample

        pred_uncond, pred_text = pred.chunk(2)
        pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)

        latents = self.scheduler.step(pred, self.scheduler.timesteps[tmax], latents).prev_sample

        for i, t in enumerate(tqdm(self.scheduler.timesteps)):
            if i > tmax:
                if i < tmin:  # layout generation phase
                    orig_latents = self.scheduler.add_noise(
                        encoded,
                        noise,
                        timesteps=t,
                    )

                    input = (mix_factor * latents) + (
                        1 - mix_factor
                    ) * orig_latents  # interpolating between layout noise and conditionally generated noise to preserve layout sematics
                    input = torch.cat([input] * 2)

                else:  # content generation phase
                    input = torch.cat([latents] * 2)

                input = self.scheduler.scale_model_input(input, t)

                with torch.no_grad():
                    pred = self.unet(
                        input,
                        t,
                        encoder_hidden_states=text_embeddings,
                    ).sample

                pred_uncond, pred_text = pred.chunk(2)
                pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)

                latents = self.scheduler.step(pred, t, latents).prev_sample

        return self.decode(latents)
