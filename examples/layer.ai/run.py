import torch
import random

import numpy as np

from PIL import Image
from typing import NamedTuple
from diffusers import (
    EulerAncestralDiscreteScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)


min_seed_value = np.iinfo(np.uint32).min
max_seed_value = np.iinfo(np.uint32).max


def new_seed() -> int:
    return random.randint(min_seed_value, max_seed_value)


def seed_everything(seed: int) -> int:
    if not min_seed_value <= seed <= max_seed_value:
        msg = f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}"
        print(f"> [warning] {msg}")
        seed = new_seed()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed


class GeneratedImage(NamedTuple):
    seed: int
    index: int
    image: Image.Image


class StableDiffusionGenerator:
    DEVICE = "cuda"

    def __init__(
        self,
        path: str,
        *,
        use_half: bool = False,
    ):
        self.use_half = use_half

        # Optimizations based on:
        # https://github.com/modal-labs/modal-examples/blob/main/06_gpu/stable_diffusion_cli.py#L108
        torch.backends.cuda.matmul.allow_tf32 = True
        euler = EulerAncestralDiscreteScheduler.from_pretrained(path, subfolder="scheduler")

        # Init txt2img pipeline
        kwargs = dict(scheduler=euler)
        if use_half:
            kwargs.update(dict(torch_dtype=torch.float16, revision="fp16"))
        self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(path, **kwargs).to(self.DEVICE)
        self.txt2img_pipe.unet.to(memory_format=torch.channels_last)
        self.txt2img_pipe.enable_xformers_memory_efficient_attention()
        # First-time "warmup" pass
        prompt = "a teapot"
        _ = self.txt2img_pipe(prompt, num_inference_steps=1)

        # Init img2img pipeline
        sub_models = {
            k: v for k, v in vars(self.txt2img_pipe).items() if not k.startswith("_") and k not in ["vae_scale_factor"]
        }
        self.img2img_pipe = StableDiffusionImg2ImgPipeline(**sub_models).to(self.DEVICE)
        self.img2img_pipe.unet.to(memory_format=torch.channels_last)
        self.img2img_pipe.enable_xformers_memory_efficient_attention()

    def generate(
        self,
        prompt,
        batch_size=1,
        seed=None,
        width=512,
        height=512,
        num_inference_steps=40,
        guidance_scale=7.5,
        init_image=None,
        image_guidance=0.8,
    ):
        if seed is None or seed < 0:
            seed = new_seed()
        seed_everything(seed)

        if init_image is not None:
            return self._generate_img2img(
                seed=seed,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                batch_size=batch_size,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                init_image=init_image,
                image_guidance=image_guidance,
            )
        else:
            return self._generate_txt2img(
                seed=seed,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                batch_size=batch_size,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
            )

    def _generate_txt2img(
        self,
        seed,
        prompt,
        batch_size,
        width,
        height,
        num_inference_steps,
        guidance_scale,
    ):
        # calculate latents from seed
        latents = None
        for index in range(batch_size):
            image_latents = torch.randn(
                (
                    1,
                    self.img2img_pipe.unet.in_channels,
                    height // 8,
                    width // 8,
                ),
                device=self.DEVICE,
                dtype=torch.float16 if self.use_half else torch.float32
            )
            latents = image_latents if latents is None else torch.cat((latents, image_latents))

        with torch.inference_mode():
            images = self.txt2img_pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=batch_size,
                latents=latents,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
            ).images

        return [GeneratedImage(image=image, seed=seed, index=index) for index, image in enumerate(images)]

    def _generate_img2img(
        self,
        seed,
        prompt,
        batch_size,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        init_image,
        image_guidance,
    ):
        with torch.inference_mode():
            images = self.img2img_pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=batch_size,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                # `init_image` is deprecated, use `image` instead
                image=init_image,
                strength=image_guidance,
            ).images
        return [GeneratedImage(image=image, seed=seed, index=index) for index, image in enumerate(images)]


if __name__ == "__main__":
    g = StableDiffusionGenerator("runwayml/stable-diffusion-v1-5", use_half=True)
    responses = g.generate("A beautiful, fantasy landscape", seed=142857)
    for i, res in enumerate(responses):
        res.image.save(f"test_{i}.png")
