import random

from PIL import Image
from typing import NamedTuple
from dataclasses import dataclass


RANDOM_RANGE_MAX = 2**40


class InferenceParameters(NamedTuple):
    prompt: str
    batch_size: int = 1
    seed: int | None = None
    width: int = 512
    height: int = 512
    num_inference_steps: int = 35
    guidance_scale: float = 7.5
    image_guidance: float = 0.2
    negative_prompt: str | None = None


@dataclass(frozen=True)
class GeneratedImage:
    image: Image.Image
    index: int


class StableDiffusionInpaintGenerator:
    DEVICE = "cuda"

    def __init__(
        self,
        path: str = "runwayml/stable-diffusion-inpainting",
        use_half: bool = True,
    ):
        self.use_half = use_half
        import torch
        from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInpaintPipeline

        # Optimizations based on:
        # https://github.com/modal-labs/modal-examples/blob/main/06_gpu/stable_diffusion_cli.py#L108
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        # logger.info("Loading the scheduler")
        euler = EulerAncestralDiscreteScheduler.from_pretrained(path, subfolder="scheduler")
        # logger.info("Scheduler loaded")

        # logger.info("Initializing inpainting pipeline")
        # Init inpainting pipeline
        kwargs = dict(
            scheduler=euler,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
        )
        if use_half:
            kwargs.update(dict(torch_dtype=torch.float16))
        kwargs.update(dict(torch_dtype=torch.float16))
        self.inpainting_pipe = StableDiffusionInpaintPipeline.from_pretrained(path, **kwargs).to(self.DEVICE)

        # logger.info("Loaded pretrained model, enabling optimizations")
        self.inpainting_pipe.unet.to(memory_format=torch.channels_last)
        self.inpainting_pipe.enable_xformers_memory_efficient_attention()
        self.inpainting_pipe.set_progress_bar_config(disable=True)
        # logger.info("inpainting pipeline initialized")

    def generate(self, params: InferenceParameters, init_image=None, mask_image=None):
        import torch

        generator = torch.Generator(device=self.DEVICE)

        # logger.info("Generating inpainted images")
        return self._inpaint(
            generator=generator,
            seed=random.randrange(0, RANDOM_RANGE_MAX) if params.seed is None else params.seed,
            prompt=params.prompt,
            num_inference_steps=params.num_inference_steps,
            batch_size=params.batch_size,
            guidance_scale=params.guidance_scale,
            width=params.width,
            height=params.height,
            mask_image=mask_image,
            base_image=init_image,
            negative_prompt=params.negative_prompt,
            image_guidance=params.image_guidance,
        )

    def _inpaint(
        self,
        generator,
        seed,
        prompt,
        batch_size,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        base_image,
        mask_image,
        negative_prompt,
        image_guidance,
    ):
        import torch

        generator = generator.manual_seed(seed)
        with torch.inference_mode():
            images = self.inpainting_pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=batch_size,
                generator=generator,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                image=base_image,
                mask_image=mask_image,
                negative_prompt=negative_prompt,
                image_guidance=image_guidance,
            ).images
        return [GeneratedImage(image=image, index=index) for index, image in enumerate(images)]


if __name__ == "__main__":
    import requests
    import numpy as np
    from io import BytesIO

    def download_image(url: str) -> Image.Image:
        raw_data = requests.get(url).content
        return Image.open(BytesIO(raw_data))

    generator = StableDiffusionInpaintGenerator()
    image_url = "https://ailabcdn.nolibox.com/tmp/3cf1b7cc46fc496bb758ed56c20f8195.png"
    mask_url = "https://ailabcdn.nolibox.com/tmp/31c8575a2fd24a9d9a840266073a0523.png"

    image = download_image(image_url)
    mask = Image.fromarray(np.array(download_image(mask_url))[..., -1])
    image.save("original.png")
    mask.save("mask.png")
    results = generator.generate(InferenceParameters("a house", seed=142857, image_guidance=0.0), image, mask)
    for i, rs in enumerate(results):
        rs.image.save(f"out_{i}.png")
