# Copyright 2023 Open AI and The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import PIL
import torch
from transformers import CLIPImageProcessor, CLIPVisionModel

from ...models import PriorTransformer
from ...pipelines import DiffusionPipeline
from ...schedulers import HeunDiscreteScheduler
from ...utils import (
    BaseOutput,
    is_accelerate_available,
    logging,
    randn_tensor,
    replace_example_docstring,
)
from .renderer import ShapERenderer


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from PIL import Image
        >>> import torch
        >>> from diffusers import DiffusionPipeline
        >>> from diffusers.utils import export_to_gif, load_image

        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        >>> repo = "openai/shap-e-img2img"
        >>> pipe = DiffusionPipeline.from_pretrained(repo, torch_dtype=torch.float16)
        >>> pipe = pipe.to(device)

        >>> guidance_scale = 3.0
        >>> image_url = "https://hf.co/datasets/diffusers/docs-images/resolve/main/shap-e/corgi.png"
        >>> image = load_image(image_url).convert("RGB")

        >>> images = pipe(
        ...     image,
        ...     guidance_scale=guidance_scale,
        ...     num_inference_steps=64,
        ...     frame_size=256,
        ... ).images

        >>> gif_path = export_to_gif(images[0], "corgi_3d.gif")
        ```
"""


@dataclass
class ShapEPipelineOutput(BaseOutput):
    """
    Output class for ShapEPipeline.

    Args:
        images (`torch.FloatTensor`)
            a list of images for 3D rendering
    """

    images: Union[PIL.Image.Image, np.ndarray]


class ShapEImg2ImgPipeline(DiffusionPipeline):
    """
    Pipeline for generating latent representation of a 3D asset and rendering with NeRF method with Shap-E

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        prior ([`PriorTransformer`]):
            The canonincal unCLIP prior to approximate the image embedding from the text embedding.
        text_encoder ([`CLIPTextModelWithProjection`]):
            Frozen text-encoder.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        scheduler ([`HeunDiscreteScheduler`]):
            A scheduler to be used in combination with `prior` to generate image embedding.
        renderer ([`ShapERenderer`]):
            Shap-E renderer projects the generated latents into parameters of a MLP that's used to create 3D objects
            with the NeRF rendering method
    """

    def __init__(
        self,
        prior: PriorTransformer,
        image_encoder: CLIPVisionModel,
        image_processor: CLIPImageProcessor,
        scheduler: HeunDiscreteScheduler,
        renderer: ShapERenderer,
    ):
        super().__init__()

        self.register_modules(
            prior=prior,
            image_encoder=image_encoder,
            image_processor=image_processor,
            scheduler=scheduler,
            renderer=renderer,
        )

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

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, the pipeline's
        models have their state dicts saved to CPU and then are moved to a `torch.device('meta') and loaded to GPU only
        when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        models = [self.image_encoder, self.prior]
        for cpu_offloaded_model in models:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.image_encoder, "_hf_hook"):
            return self.device
        for module in self.image_encoder.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_image(
        self,
        image,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
    ):
        if isinstance(image, List) and isinstance(image[0], torch.Tensor):
            image = torch.cat(image, axis=0) if image[0].ndim == 4 else torch.stack(image, axis=0)

        if not isinstance(image, torch.Tensor):
            image = self.image_processor(image, return_tensors="pt").pixel_values[0].unsqueeze(0)

        image = image.to(dtype=self.image_encoder.dtype, device=device)

        image_embeds = self.image_encoder(image)["last_hidden_state"]
        image_embeds = image_embeds[:, 1:, :].contiguous()  # batch_size, dim, 256

        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        if do_classifier_free_guidance:
            negative_image_embeds = torch.zeros_like(image_embeds)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeds = torch.cat([negative_image_embeds, image_embeds])

        return image_embeds

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 25,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        guidance_scale: float = 4.0,
        frame_size: int = 64,
        output_type: Optional[str] = "pil",  # pil, np, latent
        return_dict: bool = True,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            frame_size (`int`, *optional*, default to 64):
                the width and height of each image frame of the generated 3d output
            output_type (`str`, *optional*, defaults to `"pt"`):
                The output format of the generate image. Choose between: `"np"` (`np.array`) or `"pt"`
                (`torch.Tensor`).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Examples:

        Returns:
            [`ShapEPipelineOutput`] or `tuple`
        """

        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        elif isinstance(image, list) and isinstance(image[0], (torch.Tensor, PIL.Image.Image)):
            batch_size = len(image)
        else:
            raise ValueError(
                f"`image` has to be of type `PIL.Image.Image`, `torch.Tensor`, `List[PIL.Image.Image]` or `List[torch.Tensor]` but is {type(image)}"
            )

        device = self._execution_device

        batch_size = batch_size * num_images_per_prompt

        do_classifier_free_guidance = guidance_scale > 1.0
        image_embeds = self._encode_image(image, device, num_images_per_prompt, do_classifier_free_guidance)

        # prior

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_embeddings = self.prior.config.num_embeddings
        embedding_dim = self.prior.config.embedding_dim

        latents = self.prepare_latents(
            (batch_size, num_embeddings * embedding_dim),
            image_embeds.dtype,
            device,
            generator,
            latents,
            self.scheduler,
        )

        # YiYi notes: for testing only to match ldm, we can directly create a latents with desired shape: batch_size, num_embeddings, embedding_dim
        latents = latents.reshape(latents.shape[0], num_embeddings, embedding_dim)

        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            scaled_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.prior(
                scaled_model_input,
                timestep=t,
                proj_embedding=image_embeds,
            ).predicted_image_embedding

            # remove the variance
            noise_pred, _ = noise_pred.split(
                scaled_model_input.shape[2], dim=2
            )  # batch_size, num_embeddings, embedding_dim

            if do_classifier_free_guidance is not None:
                noise_pred_uncond, noise_pred = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            latents = self.scheduler.step(
                noise_pred,
                timestep=t,
                sample=latents,
            ).prev_sample

        if output_type == "latent":
            return ShapEPipelineOutput(images=latents)

        images = []
        for i, latent in enumerate(latents):
            print()
            image = self.renderer.decode(
                latent[None, :],
                device,
                size=frame_size,
                ray_batch_size=4096,
                n_coarse_samples=64,
                n_fine_samples=128,
            )

            images.append(image)

        images = torch.stack(images)

        if output_type not in ["np", "pil"]:
            raise ValueError(f"Only the output types `pil` and `np` are supported not output_type={output_type}")

        images = images.cpu().numpy()

        if output_type == "pil":
            images = [self.numpy_to_pil(image) for image in images]

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (images,)

        return ShapEPipelineOutput(images=images)
