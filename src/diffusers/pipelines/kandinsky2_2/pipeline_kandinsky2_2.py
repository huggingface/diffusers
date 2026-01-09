# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from typing import Callable, Dict, List, Optional, Union

import torch

from ...models import UNet2DConditionModel, VQModel
from ...schedulers import DDPMScheduler
from ...utils import deprecate, is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
        >>> import torch

        >>> pipe_prior = KandinskyV22PriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior")
        >>> pipe_prior.to("cuda")
        >>> prompt = "red cat, 4k photo"
        >>> out = pipe_prior(prompt)
        >>> image_emb = out.image_embeds
        >>> zero_image_emb = out.negative_image_embeds
        >>> pipe = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder")
        >>> pipe.to("cuda")
        >>> image = pipe(
        ...     image_embeds=image_emb,
        ...     negative_image_embeds=zero_image_emb,
        ...     height=768,
        ...     width=768,
        ...     num_inference_steps=50,
        ... ).images
        >>> image[0].save("cat.png")
        ```
"""


def downscale_height_and_width(height, width, scale_factor=8):
    new_height = height // scale_factor**2
    if height % scale_factor**2 != 0:
        new_height += 1
    new_width = width // scale_factor**2
    if width % scale_factor**2 != 0:
        new_width += 1
    return new_height * scale_factor, new_width * scale_factor


class KandinskyV22Pipeline(DiffusionPipeline):
    """
    Pipeline for text-to-image generation using Kandinsky

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        scheduler (Union[`DDIMScheduler`,`DDPMScheduler`]):
            A scheduler to be used in combination with `unet` to generate image latents.
        unet ([`UNet2DConditionModel`]):
            Conditional U-Net architecture to denoise the image embedding.
        movq ([`VQModel`]):
            MoVQ Decoder to generate the image from the latents.
    """

    model_cpu_offload_seq = "unet->movq"
    _callback_tensor_inputs = ["latents", "image_embeds", "negative_image_embeds"]

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
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image_embeds: Union[torch.Tensor, List[torch.Tensor]],
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
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
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

        device = self._execution_device

        self._guidance_scale = guidance_scale

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

        num_channels_latents = self.unet.config.in_channels

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

        self._num_timesteps = len(timesteps)
        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

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

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                image_embeds = callback_outputs.pop("image_embeds", image_embeds)
                negative_image_embeds = callback_outputs.pop("negative_image_embeds", negative_image_embeds)

            if callback is not None and i % callback_steps == 0:
                step_idx = i // getattr(self.scheduler, "order", 1)
                callback(step_idx, t, latents)

            if XLA_AVAILABLE:
                xm.mark_step()

        if output_type not in ["pt", "np", "pil", "latent"]:
            raise ValueError(f"Only the output types `pt`, `pil` and `np` are supported not output_type={output_type}")

        if not output_type == "latent":
            # post-processing
            image = self.movq.decode(latents, force_not_quantize=True)["sample"]
            if output_type in ["np", "pil"]:
                image = image * 0.5 + 0.5
                image = image.clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()

            if output_type == "pil":
                image = self.numpy_to_pil(image)
        else:
            image = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
