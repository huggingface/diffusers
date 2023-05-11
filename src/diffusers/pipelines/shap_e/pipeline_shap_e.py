# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

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


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py

        ```
"""


@dataclass
class ShapEPipelineOutput(BaseOutput):
    """
    Output class for ShapEPipeline.

    Args:
        images (`torch.FloatTensor`)
            3D latent representation
    """

    latents: Union[torch.FloatTensor, np.ndarray]


class ShapEPipeline(DiffusionPipeline):
    """
    Pipeline for generating latent representation of a 3D asset with Shap.E

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
    """

    def __init__(
        self,
        prior: PriorTransformer,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        scheduler: HeunDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            prior=prior,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
        )

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

        models = [
            self.text_encoder,
        ]
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
        if self.device != torch.device("meta") or not hasattr(self.text_encoder, "_hf_hook"):
            return self.device
        for module in self.text_encoder.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
    ):
        len(prompt) if isinstance(prompt, list) else 1

        # YiYi Notes: set pad_token_id to be 0, not sure why I can't set in the config file
        self.tokenizer.pad_token_id = 0
        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        text_encoder_output = self.text_encoder(text_input_ids.to(device))
        prompt_embeds = text_encoder_output.text_embeds

        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        # in Shap-E it normalize the prompt_embeds and then later rescale it, not sure why
        # YiYi TO-DO: move rescale out of prior_transformer and apply it here
        prompt_embeds = prompt_embeds / torch.linalg.norm(prompt_embeds, dim=-1, keepdim=True)

        if do_classifier_free_guidance:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 25,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        guidance_scale: float = 4.0,
        sigma_min: float = 1e-3,
        sigma_max: float = 160.0,
        output_type: Optional[str] = "pt",  # pt only
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
            output_type (`str`, *optional*, defaults to `"pt"`):
                The output format of the generate image. Choose between: `"np"` (`np.array`) or `"pt"`
                (`torch.Tensor`).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Examples:

        Returns:
            [`ShapEPipelineOutput`] or `tuple`
        """

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        device = self._execution_device

        batch_size = batch_size * num_images_per_prompt

        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds = self._encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance)
        # prior

        self.scheduler.set_timesteps(
            num_inference_steps, device=device, sigma_min=sigma_min, sigma_max=sigma_max, use_karras_sigmas=True
        )
        timesteps = self.scheduler.timesteps

        num_embeddings = self.prior.config.num_embeddings
        embedding_dim = self.prior.config.embedding_dim

        latents = self.prepare_latents(
            (batch_size, num_embeddings * embedding_dim),
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            self.scheduler,
        )
        # for testing only to match ldm, we can directly create a latents with desired shape: batch_size, num_embeddings, embedding_dim
        latents = latents.reshape(latents.shape[0], num_embeddings, embedding_dim)

        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            scaled_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.prior(
                scaled_model_input,
                timestep=t,
                proj_embedding=prompt_embeds,
            ).predicted_image_embedding

            # remove the variance
            noise_pred, _ = noise_pred.split(
                scaled_model_input.shape[2], dim=2
            )  # batch_size, num_embeddings, embedding_dim

            # clip between -1 and 1
            noise_pred = noise_pred.clamp(-1, 1)

            if do_classifier_free_guidance is not None:
                noise_pred_uncond, noise_pred = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            latents = self.scheduler.step(
                noise_pred,
                timestep=t,
                sample=latents,
                step_index=i,
            ).prev_sample

        if output_type not in ["pt", "np"]:
            raise ValueError(f"Only the output types `pt` and `np` are supported not output_type={output_type}")

        if output_type == "np":
            latents = latents.cpu().numpy()

        if not return_dict:
            return latents

        return ShapEPipelineOutput(latents)
