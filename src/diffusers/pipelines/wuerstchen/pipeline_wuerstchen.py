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
        self.multiple = 128
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
    
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
    ):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
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
            text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]

        text_encoder_output = self.text_encoder(text_input_ids.to(device))

        text_encoder_hidden_states = text_encoder_output.last_hidden_state

        text_encoder_hidden_states = text_encoder_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)

        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            negative_prompt_embeds_text_encoder_output = self.text_encoder(uncond_input.input_ids.to(device))

            uncond_text_encoder_hidden_states = negative_prompt_embeds_text_encoder_output.last_hidden_state

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method

            seq_len = uncond_text_encoder_hidden_states.shape[1]
            uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.repeat(1, num_images_per_prompt, 1)
            uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )
            # done duplicates

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_encoder_hidden_states = torch.cat([uncond_text_encoder_hidden_states, text_encoder_hidden_states])

        return text_encoder_hidden_states

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: int = 1024,
        width: int = 1024,
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
        device = self._execution_device
        
        do_classifier_free_guidance = guidance_scale > 1.0

        if negative_prompt is None:
            negative_prompt = ""

        if isinstance(prompt, str):
            prompt = [prompt]
        elif not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        elif not isinstance(negative_prompt, list) and negative_prompt is not None:
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        text_encoder_hidden_states = self._encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt)

        latent_height = 128 * (height / 128) // (1024 / 24)
        latent_width = 128 * (width / 128) // (1024 / 24)
        effnet_features_shape = (num_images_per_prompt, 16, latent_height, latent_width)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        prior_timesteps_tensor = self.scheduler.timesteps if timesteps is None else timesteps

        latents = self.prepare_latents(
            effnet_features_shape,
            text_encoder_hidden_states.dtype,
            device,
            generator,
            latents,
            self.scheduler,
        )

        for i, t in enumerate(self.progress_bar(prior_timesteps_tensor)):
            #  x, r, c
            latents = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            predicted_image_embedding = self.prior(latents, r=t / num_inference_steps, c=text_encoder_hidden_states)

            if do_classifier_free_guidance:
                predicted_image_embedding_uncond, predicted_image_embedding_text = predicted_image_embedding.chunk(2)
                predicted_image_embedding = predicted_image_embedding_uncond + guidance_scale * (
                    predicted_image_embedding_text - predicted_image_embedding_uncond
                )

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

        if output_type not in ["pt", "np"]:
            raise ValueError(f"Only the output types `pt` and `np` are supported not output_type={output_type}")

        if output_type == "np":
            latents = latents.cpu().numpy()

        if not return_dict:
            return (latents, text_encoder_hidden_states)

        return WuerstchenPriorPipelineOutput(latents, text_encoder_hidden_states)
