# Copyright 2022 Peter Willemsen <peter@codebuffet.co>. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Callable, List, Optional, Union

import numpy as np
import PIL
import torch
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale import StableDiffusionUpscalePipeline
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, LMSDiscreteScheduler, PNDMScheduler


def make_transparency_mask(size, overlap_pixels, remove_borders=[]):
    size_x = size[0] - overlap_pixels * 2
    size_y = size[1] - overlap_pixels * 2
    for letter in ["l", "r"]:
        if letter in remove_borders:
            size_x += overlap_pixels
    for letter in ["t", "b"]:
        if letter in remove_borders:
            size_y += overlap_pixels
    mask = np.ones((size_y, size_x), dtype=np.uint8) * 255
    mask = np.pad(mask, mode="linear_ramp", pad_width=overlap_pixels, end_values=0)

    if "l" in remove_borders:
        mask = mask[:, overlap_pixels : mask.shape[1]]
    if "r" in remove_borders:
        mask = mask[:, 0 : mask.shape[1] - overlap_pixels]
    if "t" in remove_borders:
        mask = mask[overlap_pixels : mask.shape[0], :]
    if "b" in remove_borders:
        mask = mask[0 : mask.shape[0] - overlap_pixels, :]
    return mask


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


def clamp_rect(rect: [int], min: [int], max: [int]):
    return (
        clamp(rect[0], min[0], max[0]),
        clamp(rect[1], min[1], max[1]),
        clamp(rect[2], min[0], max[0]),
        clamp(rect[3], min[1], max[1]),
    )


def add_overlap_rect(rect: [int], overlap: int, image_size: [int]):
    rect = list(rect)
    rect[0] -= overlap
    rect[1] -= overlap
    rect[2] += overlap
    rect[3] += overlap
    rect = clamp_rect(rect, [0, 0], [image_size[0], image_size[1]])
    return rect


def squeeze_tile(tile, original_image, original_slice, slice_x):
    result = Image.new("RGB", (tile.size[0] + original_slice, tile.size[1]))
    result.paste(
        original_image.resize((tile.size[0], tile.size[1]), Image.BICUBIC).crop(
            (slice_x, 0, slice_x + original_slice, tile.size[1])
        ),
        (0, 0),
    )
    result.paste(tile, (original_slice, 0))
    return result


def unsqueeze_tile(tile, original_image_slice):
    crop_rect = (original_image_slice * 4, 0, tile.size[0], tile.size[1])
    tile = tile.crop(crop_rect)
    return tile


def next_divisible(n, d):
    divisor = n % d
    return n - divisor


class StableDiffusionTiledUpscalePipeline(StableDiffusionUpscalePipeline):
    r"""
    Pipeline for tile-based text-guided image super-resolution using Stable Diffusion 2, trading memory for compute
    to create gigantic images.

    This model inherits from [`StableDiffusionUpscalePipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        low_res_scheduler ([`SchedulerMixin`]):
            A scheduler used to add initial noise to the low res conditioning image. It must be an instance of
            [`DDPMScheduler`].
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        low_res_scheduler: DDPMScheduler,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        max_noise_level: int = 350,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            low_res_scheduler=low_res_scheduler,
            scheduler=scheduler,
            max_noise_level=max_noise_level,
        )

    def _process_tile(self, original_image_slice, x, y, tile_size, tile_border, image, final_image, **kwargs):
        torch.manual_seed(0)
        crop_rect = (
            min(image.size[0] - (tile_size + original_image_slice), x * tile_size),
            min(image.size[1] - (tile_size + original_image_slice), y * tile_size),
            min(image.size[0], (x + 1) * tile_size),
            min(image.size[1], (y + 1) * tile_size),
        )
        crop_rect_with_overlap = add_overlap_rect(crop_rect, tile_border, image.size)
        tile = image.crop(crop_rect_with_overlap)
        translated_slice_x = ((crop_rect[0] + ((crop_rect[2] - crop_rect[0]) / 2)) / image.size[0]) * tile.size[0]
        translated_slice_x = translated_slice_x - (original_image_slice / 2)
        translated_slice_x = max(0, translated_slice_x)
        to_input = squeeze_tile(tile, image, original_image_slice, translated_slice_x)
        orig_input_size = to_input.size
        to_input = to_input.resize((tile_size, tile_size), Image.BICUBIC)
        upscaled_tile = super(StableDiffusionTiledUpscalePipeline, self).__call__(image=to_input, **kwargs).images[0]
        upscaled_tile = upscaled_tile.resize((orig_input_size[0] * 4, orig_input_size[1] * 4), Image.BICUBIC)
        upscaled_tile = unsqueeze_tile(upscaled_tile, original_image_slice)
        upscaled_tile = upscaled_tile.resize((tile.size[0] * 4, tile.size[1] * 4), Image.BICUBIC)
        remove_borders = []
        if x == 0:
            remove_borders.append("l")
        elif crop_rect[2] == image.size[0]:
            remove_borders.append("r")
        if y == 0:
            remove_borders.append("t")
        elif crop_rect[3] == image.size[1]:
            remove_borders.append("b")
        transparency_mask = Image.fromarray(
            make_transparency_mask(
                (upscaled_tile.size[0], upscaled_tile.size[1]), tile_border * 4, remove_borders=remove_borders
            ),
            mode="L",
        )
        final_image.paste(
            upscaled_tile, (crop_rect_with_overlap[0] * 4, crop_rect_with_overlap[1] * 4), transparency_mask
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        num_inference_steps: int = 75,
        guidance_scale: float = 9.0,
        noise_level: int = 50,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        tile_size: int = 128,
        tile_border: int = 32,
        original_image_slice: int = 32,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image (`PIL.Image.Image` or List[`PIL.Image.Image`] or `torch.FloatTensor`):
                `Image`, or tensor representing an image batch which will be upscaled. *
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            tile_size (`int`, *optional*):
                The size of the tiles. Too big can result in an OOM-error.
            tile_border (`int`, *optional*):
                The number of pixels around a tile to consider (bigger means less seams, too big can lead to an OOM-error).
            original_image_slice (`int`, *optional*):
                The amount of pixels of the original image to calculate with the current tile (bigger means more depth
                is preserved, less blur occurs in the final image, too big can lead to an OOM-error or loss in detail).
            callback (`Callable`, *optional*):
                A function that take a callback function with a single argument, a dict,
                that contains the (partially) processed image under "image",
                as well as the progress (0 to 1, where 1 is completed) under "progress".

        Returns: A PIL.Image that is 4 times larger than the original input image.

        """

        final_image = Image.new("RGB", (image.size[0] * 4, image.size[1] * 4))
        tcx = math.ceil(image.size[0] / tile_size)
        tcy = math.ceil(image.size[1] / tile_size)
        total_tile_count = tcx * tcy
        current_count = 0
        for y in range(tcy):
            for x in range(tcx):
                self._process_tile(
                    original_image_slice,
                    x,
                    y,
                    tile_size,
                    tile_border,
                    image,
                    final_image,
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    noise_level=noise_level,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    eta=eta,
                    generator=generator,
                    latents=latents,
                )
                current_count += 1
                if callback is not None:
                    callback({"progress": current_count / total_tile_count, "image": final_image})
        return final_image


def main():
    # Run a demo
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipe = StableDiffusionTiledUpscalePipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    image = Image.open("../../docs/source/imgs/diffusers_library.jpg")

    def callback(obj):
        print(f"progress: {obj['progress']:.4f}")
        obj["image"].save("diffusers_library_progress.jpg")

    final_image = pipe(image=image, prompt="Black font, white background, vector", noise_level=40, callback=callback)
    final_image.save("diffusers_library.jpg")


if __name__ == "__main__":
    main()
