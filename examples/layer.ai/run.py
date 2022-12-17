import math
import torch
import random
import torchvision

import numpy as np

from PIL import Image
from torch import Tensor
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from typing import NamedTuple
from diffusers import (
    SchedulerMixin,
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


def slerp(
    x1: Tensor,
    x2: Tensor,
    r1: Union[float, Tensor],
    r2: Optional[Union[float, Tensor]] = None,
    *,
    dot_threshold: float = 0.9995,
) -> Tensor:
    if r2 is None:
        r2 = 1.0 - r1
    b, *shape = x1.shape
    x1 = x1.view(b, -1)
    x2 = x2.view(b, -1)
    low_norm = x1 / torch.norm(x1, dim=1, keepdim=True)
    high_norm = x2 / torch.norm(x2, dim=1, keepdim=True)
    dot = (low_norm * high_norm).sum(1)
    overflow_mask = dot > dot_threshold
    out = torch.zeros_like(x1)
    out[overflow_mask] = r1 * x1 + r2 * x2
    normal_mask = ~overflow_mask
    omega = torch.acos(dot[normal_mask])
    so = torch.sin(omega)
    x1_part = (torch.sin(r1 * omega) / so).unsqueeze(1) * x1
    x2_part = (torch.sin(r2 * omega) / so).unsqueeze(1) * x2
    out[normal_mask] = x1_part + x2_part
    return out.view(b, *shape)


def set_seed_and_variations(
    seed: int,
    get_noise: Callable[[], Tensor],
    get_new_z: Callable[[Tensor], Tensor],
    variations: Optional[List[Tuple[int, float]]],
) -> Tuple[Tensor, Tensor]:
    seed_everything(seed)
    z_noise = get_noise()
    if variations is not None:
        for v_seed, v_weight in variations:
            seed_everything(v_seed)
            v_noise = get_noise()
            z_noise = slerp(v_noise, z_noise, v_weight)
    z = get_new_z(z_noise)
    # for some samplers (e.g. EulerAncestralDiscreteScheduler), they will
    # re-sample noises during the sampling process, so we need to set back to the
    # original seed to make results consistent
    seed_everything(seed)
    get_noise()
    return z, z_noise


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
        sampler_base: Optional[Type[SchedulerMixin]] = None,
    ):
        self.use_half = use_half

        # Optimizations based on:
        # https://github.com/modal-labs/modal-examples/blob/main/06_gpu/stable_diffusion_cli.py#L108
        torch.backends.cuda.matmul.allow_tf32 = True
        sampler_base = sampler_base or EulerAncestralDiscreteScheduler
        euler = sampler_base.from_pretrained(path, subfolder="scheduler")

        # Init txt2img pipeline
        kwargs = dict(scheduler=euler)
        if use_half:
            kwargs.update(dict(torch_dtype=torch.float16))
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
        variations=None,
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
                variations=variations,
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
        variations,
        prompt,
        batch_size,
        width,
        height,
        num_inference_steps,
        guidance_scale,
    ):
        # calculate latents from seed
        latents = None
        z_shape = (
            1,
            self.txt2img_pipe.unet.in_channels,
            height // 8,
            width // 8,
        )
        z_kwargs = dict(
            device=self.DEVICE,
            dtype=torch.float16 if self.use_half else torch.float32,
        )
        for index in range(batch_size):
            image_latents, _ = set_seed_and_variations(
                seed + index,
                lambda: torch.randn(z_shape, **z_kwargs),
                lambda noise: noise,
                variations,
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


def save_images(arr: Tensor, path: str, n_row: Optional[int] = None) -> None:
    if n_row is None:
        n_row = math.ceil(math.sqrt(len(arr)))
    torchvision.utils.save_image(arr, path, normalize=True, nrow=n_row)


def inference(prompt: str, seed: int, variations=None) -> Tensor:
    image = g.generate(prompt, seed=seed, variations=variations)[0].image
    array = np.array(image).transpose([2, 0, 1])[None, ...].astype(np.float32)
    return torch.from_numpy(array) / 255.0


if __name__ == "__main__":
    g = StableDiffusionGenerator("runwayml/stable-diffusion-v1-5", use_half=True)
    responses = g.generate("A beautiful, fantasy landscape", seed=142857)
    for i, res in enumerate(responses):
        res.image.save(f"test_{i}.png")
