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

from ...models import VQModelPaella
from ...schedulers import DDPMWuerstchenScheduler
from ...utils import BaseOutput, logging, randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .modules import DiffNeXt, EfficientNetEncoder

# from .diffuzz import Diffuzz


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import WuerstchenPriorPipeline, WuerstchenGeneratorPipeline

        >>> prior_pipe = WuerstchenPriorPipeline.from_pretrained("kashif/wuerstchen-prior", torch_dtype=torch.float16).to("cuda")
        >>> gen_pipe = WuerstchenGeneratorPipeline.from_pretrain("kashif/wuerstchen-gen", torch_dtype=torch.float16).to("cuda")

        >>> prompt = "an image of a shiba inu, donning a spacesuit and helmet"
        >>> prior_output = pipe(prompt)
        >>> images = gen_pipe(prior_output.image_embeds, prior_output.text_embeds)
        ```
"""


default_inference_steps_b = {0.0: 12}


@dataclass
class WuerstchenGeneratorPipelineOutput(BaseOutput):
    """
    Output class for WuerstchenPriorPipeline.

    Args:
        images (`torch.FloatTensor` or `np.ndarray`)
            Generated images for text prompt.
    """

    images: Union[torch.FloatTensor, np.ndarray]


class WuerstchenGeneratorPipeline(DiffusionPipeline):
    """
    Pipeline for generating images from the  Wuerstchen model.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        generator ([`DiffNeXt`]):
            The DiffNeXt unet generator.
        vqgan ([`VQModelPaella`]):
            The VQGAN model.
        efficient_net ([`EfficientNetEncoder`]):
            The EfficientNet encoder.
        scheduler ([`DDPMScheduler`]):
            A scheduler to be used in combination with `prior` to generate image embedding.
    """

    def __init__(
        self,
        generator: DiffNeXt,
        scheduler: DDPMWuerstchenScheduler,
        vqgan: VQModelPaella,
        efficient_net: EfficientNetEncoder,
    ) -> None:
        super().__init__()
        self.multiple = 128
        self.register_modules(
            generator=generator,
            scheduler=scheduler,
            vqgan=vqgan,
            efficient_net=efficient_net,
        )
        # self.diffuzz = Diffuzz(device="cuda")

        self.register_to_config()

    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        return latents

    def check_inputs(
        self, predicted_image_embeddings, text_encoder_hidden_states, do_classifier_free_guidance, device
    ):
        if not isinstance(text_encoder_hidden_states, torch.Tensor):
            raise TypeError(
                f"'text_encoder_hidden_states' must be of type 'torch.Tensor', but got {type(predicted_image_embeddings)}."
            )
        if isinstance(predicted_image_embeddings, np.ndarray):
            predicted_image_embeddings = torch.Tensor(predicted_image_embeddings, device=device).to(
                dtype=text_encoder_hidden_states.dtype
            )
        if not isinstance(predicted_image_embeddings, torch.Tensor):
            raise TypeError(
                f"'predicted_image_embeddings' must be of type 'torch.Tensor' or 'np.array', but got {type(predicted_image_embeddings)}."
            )

        if do_classifier_free_guidance:
            assert (
                predicted_image_embeddings.size(0) == text_encoder_hidden_states.size(0) // 2
            ), f"'text_encoder_hidden_states' must be double the size of 'predicted_image_embeddings' in the first dimension, but {predicted_image_embeddings.size(0)} != {text_encoder_hidden_states.size(0)}."
        else:
            if predicted_image_embeddings.size(0) * 2 == text_encoder_hidden_states.size(0):
                text_encoder_hidden_states = text_encoder_hidden_states.chunk(2)[0]
            assert predicted_image_embeddings.size(0) == text_encoder_hidden_states.size(
                0
            ), f"'text_encoder_hidden_states' must be the size of 'predicted_image_embeddings' in the first dimension, but {predicted_image_embeddings.size(0)} != {text_encoder_hidden_states.size(0)}."

        return predicted_image_embeddings, text_encoder_hidden_states

    @torch.no_grad()
    def encode_image(self, image):
        return self.efficient_net(image)

    @torch.no_grad()
    def __call__(
        self,
        predicted_image_embeddings: torch.Tensor,
        text_encoder_hidden_states: torch.Tensor = None,
        inference_steps: dict = None,
        guidance_scale: float = 3.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pt",  # pt only
        return_dict: bool = True,
    ):
        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        if inference_steps is None:
            inference_steps = default_inference_steps_b

        predicted_image_embeddings, text_encoder_hidden_states = self.check_inputs(
            predicted_image_embeddings, text_encoder_hidden_states, do_classifier_free_guidance, device
        )

        dtype = predicted_image_embeddings.dtype
        latent_height = int(predicted_image_embeddings.size(2) * (256 / 24))
        latent_width = int(predicted_image_embeddings.size(3) * (256 / 24))
        effnet_features_shape = (predicted_image_embeddings.size(0), 4, latent_height, latent_width)

        self.scheduler.set_timesteps(inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        latents = self.prepare_latents(
            effnet_features_shape,
            dtype,
            device,
            generator,
            latents,
            self.scheduler,
        )
        # from transformers import AutoTokenizer, CLIPTextModel
        # text_encoder = CLIPTextModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(device)
        # tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        # clip_tokens = tokenizer([""] * latents.size(0), truncation=True, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").to(device)
        # clip_text_embeddings = text_encoder(**clip_tokens).last_hidden_state.to(dtype)

        for t in self.progress_bar(timesteps[:-1]):
            ratio = t.expand(latents.size(0)).to(dtype)
            effnet = (
                torch.cat([predicted_image_embeddings, torch.zeros_like(predicted_image_embeddings)])
                if do_classifier_free_guidance
                else predicted_image_embeddings
            )
            predicted_latents = self.generator(
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents,
                r=torch.cat([ratio] * 2) if do_classifier_free_guidance else ratio,
                effnet=effnet,
                clip=None,  # torch.cat([text_encoder_hidden_states] * 2) if do_classifier_free_guidance else text_encoder_hidden_states,
            )

            if do_classifier_free_guidance:
                predicted_latents_text, predicted_latents_uncond = predicted_latents.chunk(2)
                predicted_latents = torch.lerp(predicted_latents_uncond, predicted_latents_text, guidance_scale)

            latents = self.scheduler.step(
                model_output=predicted_latents,
                timestep=ratio,
                sample=latents,
                generator=generator,
            ).prediction

        images = self.vqgan.decode(latents).sample.clamp(0, 1)

        if output_type not in ["pt", "np"]:
            raise ValueError(f"Only the output types `pt` and `np` are supported not output_type={output_type}")

        if output_type == "np":
            images = images.permute(0, 2, 3, 1).cpu().numpy()

        if not return_dict:
            return images

        return WuerstchenGeneratorPipelineOutput(images)
