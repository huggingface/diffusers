# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from copy import deepcopy
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from packaging import version
from PIL import Image

from ... import __version__
from ...models import UNet2DConditionModel, VQModel
from ...schedulers import DDPMScheduler
from ...utils import deprecate, logging
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import KandinskyV22InpaintPipeline, KandinskyV22PriorPipeline
        >>> from diffusers.utils import load_image
        >>> import torch
        >>> import numpy as np

        >>> pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
        ... )
        >>> pipe_prior.to("cuda")

        >>> prompt = "a hat"
        >>> image_emb, zero_image_emb = pipe_prior(prompt, return_dict=False)

        >>> pipe = KandinskyV22InpaintPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
        ... )
        >>> pipe.to("cuda")

        >>> init_image = load_image(
        ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
        ...     "/kandinsky/cat.png"
        ... )

        >>> mask = np.zeros((768, 768), dtype=np.float32)
        >>> mask[:250, 250:-250] = 1

        >>> out = pipe(
        ...     image=init_image,
        ...     mask_image=mask,
        ...     image_embeds=image_emb,
        ...     negative_image_embeds=zero_image_emb,
        ...     height=768,
        ...     width=768,
        ...     num_inference_steps=50,
        ... )

        >>> image = out.images[0]
        >>> image.save("cat_with_hat.png")
        ```
"""


# Copied from diffusers.pipelines.kandinsky2_2.pipeline_kandinsky2_2.downscale_height_and_width
def downscale_height_and_width(height, width, scale_factor=8):
    new_height = height // scale_factor**2
    if height % scale_factor**2 != 0:
        new_height += 1
    new_width = width // scale_factor**2
    if width % scale_factor**2 != 0:
        new_width += 1
    return new_height * scale_factor, new_width * scale_factor


# Copied from diffusers.pipelines.kandinsky.pipeline_kandinsky_inpaint.prepare_mask
def prepare_mask(masks):
    prepared_masks = []
    for mask in masks:
        old_mask = deepcopy(mask)
        for i in range(mask.shape[1]):
            for j in range(mask.shape[2]):
                if old_mask[0][i][j] == 1:
                    continue
                if i != 0:
                    mask[:, i - 1, j] = 0
                if j != 0:
                    mask[:, i, j - 1] = 0
                if i != 0 and j != 0:
                    mask[:, i - 1, j - 1] = 0
                if i != mask.shape[1] - 1:
                    mask[:, i + 1, j] = 0
                if j != mask.shape[2] - 1:
                    mask[:, i, j + 1] = 0
                if i != mask.shape[1] - 1 and j != mask.shape[2] - 1:
                    mask[:, i + 1, j + 1] = 0
        prepared_masks.append(mask)
    return torch.stack(prepared_masks, dim=0)


# Copied from diffusers.pipelines.kandinsky.pipeline_kandinsky_inpaint.prepare_mask_and_masked_image
def prepare_mask_and_masked_image(image, mask, height, width):
    r"""
    Prepares a pair (mask, image) to be consumed by the Kandinsky inpaint pipeline. This means that those inputs will
    be converted to ``torch.Tensor`` with shapes ``batch x channels x height x width`` where ``channels`` is ``3`` for
    the ``image`` and ``1`` for the ``mask``.

    The ``image`` will be converted to ``torch.float32`` and normalized to be in ``[-1, 1]``. The ``mask`` will be
    binarized (``mask > 0.5``) and cast to ``torch.float32`` too.

    Args:
        image (Union[np.array, PIL.Image, torch.Tensor]): The image to inpaint.
            It can be a ``PIL.Image``, or a ``height x width x 3`` ``np.array`` or a ``channels x height x width``
            ``torch.Tensor`` or a ``batch x channels x height x width`` ``torch.Tensor``.
        mask (_type_): The mask to apply to the image, i.e. regions to inpaint.
            It can be a ``PIL.Image``, or a ``height x width`` ``np.array`` or a ``1 x height x width``
            ``torch.Tensor`` or a ``batch x 1 x height x width`` ``torch.Tensor``.
        height (`int`, *optional*, defaults to 512):
            The height in pixels of the generated image.
        width (`int`, *optional*, defaults to 512):
            The width in pixels of the generated image.


    Raises:
        ValueError: ``torch.Tensor`` images should be in the ``[-1, 1]`` range. ValueError: ``torch.Tensor`` mask
        should be in the ``[0, 1]`` range. ValueError: ``mask`` and ``image`` should have the same spatial dimensions.
        TypeError: ``mask`` is a ``torch.Tensor`` but ``image`` is not
            (ot the other way around).

    Returns:
        tuple[torch.Tensor]: The pair (mask, image) as ``torch.Tensor`` with 4
            dimensions: ``batch x channels x height x width``.
    """

    if image is None:
        raise ValueError("`image` input cannot be undefined.")

    if mask is None:
        raise ValueError("`mask_image` input cannot be undefined.")

    if isinstance(image, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"`image` is a torch.Tensor but `mask` (type: {type(mask)} is not")

        # Batch single image
        if image.ndim == 3:
            assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
            image = image.unsqueeze(0)

        # Batch and add channel dim for single mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Batch single mask or add channel dim
        if mask.ndim == 3:
            # Single batched mask, no channel dim or single mask not batched but channel dim
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(0)

            # Batched masks no channel dim
            else:
                mask = mask.unsqueeze(1)

        assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
        assert image.shape[-2:] == mask.shape[-2:], "Image and Mask must have the same spatial dimensions"
        assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"

        # Check image is in [-1, 1]
        if image.min() < -1 or image.max() > 1:
            raise ValueError("Image should be in [-1, 1] range")

        # Check mask is in [0, 1]
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("Mask should be in [0, 1] range")

        # Binarize mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # Image as float32
        image = image.to(dtype=torch.float32)
    elif isinstance(mask, torch.Tensor):
        raise TypeError(f"`mask` is a torch.Tensor but `image` (type: {type(image)} is not")
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]

        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            # resize all images w.r.t passed height an width
            image = [i.resize((width, height), resample=Image.BICUBIC, reducing_gap=1) for i in image]
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        # preprocess mask
        if isinstance(mask, (PIL.Image.Image, np.ndarray)):
            mask = [mask]

        if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
            mask = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in mask]
            mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

    mask = 1 - mask

    return mask, image


class KandinskyV22InpaintPipeline(DiffusionPipeline):
    """
    Pipeline for text-guided image inpainting using Kandinsky2.1

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        scheduler ([`DDIMScheduler`]):
            A scheduler to be used in combination with `unet` to generate image latents.
        unet ([`UNet2DConditionModel`]):
            Conditional U-Net architecture to denoise the image embedding.
        movq ([`VQModel`]):
            MoVQ Decoder to generate the image from the latents.
    """

    model_cpu_offload_seq = "unet->movq"
    _callback_tensor_inputs = ["latents", "image_embeds", "negative_image_embeds", "masked_image", "mask_image"]

    def __init__(
        self,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
        movq: VQModel,
    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            movq=movq,
        )
        self.movq_scale_factor = 2 ** (len(self.movq.config.block_out_channels) - 1)
        self._warn_has_been_called = False

    # Copied from diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline.prepare_latents
    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        latents = latents * scheduler.init_noise_sigma
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    def __call__(
        self,
        image_embeds: Union[torch.Tensor, List[torch.Tensor]],
        image: Union[torch.Tensor, PIL.Image.Image],
        mask_image: Union[torch.Tensor, PIL.Image.Image, np.ndarray],
        negative_image_embeds: Union[torch.Tensor, List[torch.Tensor]],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 100,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            image_embeds (`torch.Tensor` or `List[torch.Tensor]`):
                The clip image embeddings for text prompt, that will be used to condition the image generation.
            image (`PIL.Image.Image`):
                `Image`, or tensor representing an image batch which will be inpainted, *i.e.* parts of the image will
                be masked out with `mask_image` and repainted according to `prompt`.
            mask_image (`np.array`):
                Tensor representing an image batch, to mask `image`. White pixels in the mask will be repainted, while
                black pixels will be preserved. If `mask_image` is a PIL image, it will be converted to a single
                channel (luminance) before use. If it's a tensor, it should contain one color channel (L) instead of 3,
                so the expected shape would be `(B, H, W, 1)`.
            negative_image_embeds (`torch.Tensor` or `List[torch.Tensor]`):
                The clip image embeddings for negative text prompt, will be used to condition the image generation.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between: `"pil"` (`PIL.Image.Image`), `"np"`
                (`np.array`) or `"pt"` (`torch.Tensor`).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`
        """
        if not self._warn_has_been_called and version.parse(version.parse(__version__).base_version) < version.parse(
            "0.23.0.dev0"
        ):
            logger.warning(
                "Please note that the expected format of `mask_image` has recently been changed. "
                "Before diffusers == 0.19.0, Kandinsky Inpainting pipelines repainted black pixels and preserved black pixels. "
                "As of diffusers==0.19.0 this behavior has been inverted. Now white pixels are repainted and black pixels are preserved. "
                "This way, Kandinsky's masking behavior is aligned with Stable Diffusion. "
                "THIS means that you HAVE to invert the input mask to have the same behavior as before as explained in https://github.com/huggingface/diffusers/pull/4207. "
                "This warning will be surpressed after the first inference call and will be removed in diffusers>0.23.0"
            )
            self._warn_has_been_called = True

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        self._guidance_scale = guidance_scale

        device = self._execution_device

        if isinstance(image_embeds, list):
            image_embeds = torch.cat(image_embeds, dim=0)
        batch_size = image_embeds.shape[0] * num_images_per_prompt
        if isinstance(negative_image_embeds, list):
            negative_image_embeds = torch.cat(negative_image_embeds, dim=0)

        if self.do_classifier_free_guidance:
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            negative_image_embeds = negative_image_embeds.repeat_interleave(num_images_per_prompt, dim=0)

            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0).to(
                dtype=self.unet.dtype, device=device
            )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # preprocess image and mask
        mask_image, image = prepare_mask_and_masked_image(image, mask_image, height, width)

        image = image.to(dtype=image_embeds.dtype, device=device)
        image = self.movq.encode(image)["latents"]

        mask_image = mask_image.to(dtype=image_embeds.dtype, device=device)

        image_shape = tuple(image.shape[-2:])
        mask_image = F.interpolate(
            mask_image,
            image_shape,
            mode="nearest",
        )
        mask_image = prepare_mask(mask_image)
        masked_image = image * mask_image

        mask_image = mask_image.repeat_interleave(num_images_per_prompt, dim=0)
        masked_image = masked_image.repeat_interleave(num_images_per_prompt, dim=0)
        if self.do_classifier_free_guidance:
            mask_image = mask_image.repeat(2, 1, 1, 1)
            masked_image = masked_image.repeat(2, 1, 1, 1)

        num_channels_latents = self.movq.config.latent_channels

        height, width = downscale_height_and_width(height, width, self.movq_scale_factor)

        # create initial latent
        latents = self.prepare_latents(
            (batch_size, num_channels_latents, height, width),
            image_embeds.dtype,
            device,
            generator,
            latents,
            self.scheduler,
        )
        noise = torch.clone(latents)

        self._num_timesteps = len(timesteps)
        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_model_input = torch.cat([latent_model_input, masked_image, mask_image], dim=1)

            added_cond_kwargs = {"image_embeds": image_embeds}
            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=None,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if self.do_classifier_free_guidance:
                noise_pred, variance_pred = noise_pred.split(latents.shape[1], dim=1)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                _, variance_pred_text = variance_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_pred = torch.cat([noise_pred, variance_pred_text], dim=1)

            if not (
                hasattr(self.scheduler.config, "variance_type")
                and self.scheduler.config.variance_type in ["learned", "learned_range"]
            ):
                noise_pred, _ = noise_pred.split(latents.shape[1], dim=1)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
                generator=generator,
            )[0]
            init_latents_proper = image[:1]
            init_mask = mask_image[:1]

            if i < len(timesteps) - 1:
                noise_timestep = timesteps[i + 1]
                init_latents_proper = self.scheduler.add_noise(
                    init_latents_proper, noise, torch.tensor([noise_timestep])
                )

            latents = init_mask * init_latents_proper + (1 - init_mask) * latents

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                image_embeds = callback_outputs.pop("image_embeds", image_embeds)
                negative_image_embeds = callback_outputs.pop("negative_image_embeds", negative_image_embeds)
                masked_image = callback_outputs.pop("masked_image", masked_image)
                mask_image = callback_outputs.pop("mask_image", mask_image)

            if callback is not None and i % callback_steps == 0:
                step_idx = i // getattr(self.scheduler, "order", 1)
                callback(step_idx, t, latents)

        # post-processing
        latents = mask_image[:1] * image[:1] + (1 - mask_image[:1]) * latents

        if output_type not in ["pt", "np", "pil", "latent"]:
            raise ValueError(
                f"Only the output types `pt`, `pil`, `np` and `latent` are supported not output_type={output_type}"
            )

        if not output_type == "latent":
            image = self.movq.decode(latents, force_not_quantize=True)["sample"]

            if output_type in ["np", "pil"]:
                image = image * 0.5 + 0.5
                image = image.clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()

            if output_type == "pil":
                image = self.numpy_to_pil(image)
        else:
            image = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
