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
from transformers import CLIPTokenizer, CLIPTextModel

from ...models import PaellaVQModel
from ...utils import is_accelerate_available, logging, BaseOutput, randn_tensor
from ..pipeline_utils import DiffusionPipeline
from ...schedulers import DDPMScheduler

from .modules import DiffNeXt, Prior, EfficientNetEncoder

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import WuerstchenPipeline

        >>> pipe = WuerstchenPipeline.from_pretrained("kashif/wuerstchen", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> prompt = "an image of a shiba inu, donning a spacesuit and helmet"
        >>> image = pipe(prompt).images[0]
        ```
"""


class WuerstchenPipeline(DiffusionPipeline):
    unet: DiffNeXt
    vqmodel: PaellaVQModel


@dataclass
class WuerstchenPriorPipelineOutput(BaseOutput):
    """
    Output class for WuerstchenPriorPipeline.

    Args:
        image_embeds (`torch.FloatTensor`)
            clip image embeddings for text prompt
        negative_image_embeds (`List[PIL.Image.Image]` or `np.ndarray`)
            clip image embeddings for unconditional tokens
    """

    image_embeds: Union[torch.FloatTensor, np.ndarray]
    text_embeds: Union[torch.FloatTensor, np.ndarray]
    negative_text_embeds: Union[torch.FloatTensor, np.ndarray]


class WuerstchenPriorPipeline(DiffusionPipeline):
    """
    Pipeline for generating image prior for Wuerstchen.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        prior ([`Prior`]):
            The canonical unCLIP prior to approximate the image embedding from the text embedding.
        text_encoder ([`CLIPTextModelWithProjection`]):
            Frozen text-encoder.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        scheduler ([`DDPMScheduler`]):
            A scheduler to be used in combination with `prior` to generate image embedding.
    """

    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        prior: Prior,
        scheduler: DDPMScheduler,
    ) -> None:
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prior=prior,
            scheduler=scheduler,
        )
        self.register_to_config()

    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        latents = latents * scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 100,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pt",  # pt only
        return_dict: bool = True,
    ):
        do_classifier_free_guidance = guidance_scale > 1.0

        clip_tokens = self.tokenizer(
            [prompt] * num_images_per_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        clip_text_embeddings = self.text_encoder(**clip_tokens).last_hidden_state

        if negative_prompt is None:
            negative_prompt = ""

        clip_tokens_uncond = self.tokenizer(
            [negative_prompt] * num_images_per_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        clip_text_embeddings_uncond = self.text_encoder(**clip_tokens_uncond).last_hidden_state

        effnet_features_shape = (num_images_per_prompt, 16, 24, 24)

        device = "cuda"

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        prior_timesteps_tensor = self.scheduler.timesteps

        latents = self.prepare_latents(
            effnet_features_shape,
            clip_text_embeddings.dtype,
            device,
            generator,
            latents,
            self.scheduler,
        )

        cond = torch.cat([clip_text_embeddings, clip_text_embeddings_uncond])

        for i, t in enumerate(self.progress_bar(prior_timesteps_tensor)):
            #  x, r, c
            predicted_image_embedding = self.prior(latents, r=t / num_inference_steps, c=cond)

            if i + 1 == prior_timesteps_tensor.shape[0]:
                prev_timestep = None
            else:
                prev_timestep = prior_timesteps_tensor[i + 1]

            latents = self.scheduler.step(
                predicted_image_embedding,
                timestep=t,
                sample=latents,
                generator=generator,
                prev_timestep=prev_timestep,
            ).prev_sample

        # normalize the latents
        latent = latent * 42.0 - 1.0

        if not return_dict:
            return (latents, clip_text_embeddings, clip_text_embeddings_uncond)

        return WuerstchenPriorPipelineOutput(latents, clip_text_embeddings, clip_text_embeddings_uncond)
