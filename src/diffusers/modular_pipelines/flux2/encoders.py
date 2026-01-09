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

from typing import List, Optional, Tuple, Union

import torch
from transformers import AutoProcessor, Mistral3ForConditionalGeneration

from ...models import AutoencoderKLFlux2
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import Flux2ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def format_text_input(prompts: List[str], system_message: str = None):
    """Format prompts for Mistral3 chat template."""
    cleaned_txt = [prompt.replace("[IMG]", "") for prompt in prompts]

    return [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        for prompt in cleaned_txt
    ]


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class Flux2TextEncoderStep(ModularPipelineBlocks):
    model_name = "flux2"

    # fmt: off
    DEFAULT_SYSTEM_MESSAGE = "You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object attribution and actions without speculation."
    # fmt: on

    @property
    def description(self) -> str:
        return "Text Encoder step that generates text embeddings using Mistral3 to guide the image generation"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Mistral3ForConditionalGeneration),
            ComponentSpec("tokenizer", AutoProcessor),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("prompt"),
            InputParam("prompt_embeds", type_hint=torch.Tensor, required=False),
            InputParam("max_sequence_length", type_hint=int, default=512, required=False),
            InputParam("text_encoder_out_layers", type_hint=Tuple[int], default=(10, 20, 30), required=False),
            InputParam("joint_attention_kwargs"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "prompt_embeds",
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="Text embeddings from Mistral3 used to guide the image generation",
            ),
        ]

    @staticmethod
    def check_inputs(block_state):
        prompt = block_state.prompt
        prompt_embeds = getattr(block_state, "prompt_embeds", None)

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. "
                "Please make sure to only forward one of the two."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

    @staticmethod
    def _get_mistral_3_prompt_embeds(
        text_encoder: Mistral3ForConditionalGeneration,
        tokenizer: AutoProcessor,
        prompt: Union[str, List[str]],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
        # fmt: off
        system_message: str = "You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object attribution and actions without speculation.",
        # fmt: on
        hidden_states_layers: Tuple[int] = (10, 20, 30),
    ):
        dtype = text_encoder.dtype if dtype is None else dtype
        device = text_encoder.device if device is None else device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        messages_batch = format_text_input(prompts=prompt, system_message=system_message)

        inputs = tokenizer.apply_chat_template(
            messages_batch,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
        out = out.to(dtype=dtype, device=device)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

        return prompt_embeds

    @torch.no_grad()
    def __call__(self, components: Flux2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        block_state.device = components._execution_device

        if block_state.prompt_embeds is not None:
            self.set_block_state(state, block_state)
            return components, state

        prompt = block_state.prompt
        if prompt is None:
            prompt = ""
        prompt = [prompt] if isinstance(prompt, str) else prompt

        block_state.prompt_embeds = self._get_mistral_3_prompt_embeds(
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            prompt=prompt,
            device=block_state.device,
            max_sequence_length=block_state.max_sequence_length,
            system_message=self.DEFAULT_SYSTEM_MESSAGE,
            hidden_states_layers=block_state.text_encoder_out_layers,
        )

        self.set_block_state(state, block_state)
        return components, state


class Flux2RemoteTextEncoderStep(ModularPipelineBlocks):
    model_name = "flux2"

    REMOTE_URL = "https://remote-text-encoder-flux-2.huggingface.co/predict"

    @property
    def description(self) -> str:
        return "Text Encoder step that generates text embeddings using a remote API endpoint"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return []

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("prompt"),
            InputParam("prompt_embeds", type_hint=torch.Tensor, required=False),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "prompt_embeds",
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="Text embeddings from remote API used to guide the image generation",
            ),
        ]

    @staticmethod
    def check_inputs(block_state):
        prompt = block_state.prompt
        prompt_embeds = getattr(block_state, "prompt_embeds", None)

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. "
                "Please make sure to only forward one of the two."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

    @torch.no_grad()
    def __call__(self, components: Flux2ModularPipeline, state: PipelineState) -> PipelineState:
        import io

        import requests
        from huggingface_hub import get_token

        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        block_state.device = components._execution_device

        if block_state.prompt_embeds is not None:
            self.set_block_state(state, block_state)
            return components, state

        prompt = block_state.prompt
        if prompt is None:
            prompt = ""
        prompt = [prompt] if isinstance(prompt, str) else prompt

        response = requests.post(
            self.REMOTE_URL,
            json={"prompt": prompt},
            headers={
                "Authorization": f"Bearer {get_token()}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()

        block_state.prompt_embeds = torch.load(io.BytesIO(response.content), weights_only=True)
        block_state.prompt_embeds = block_state.prompt_embeds.to(block_state.device)

        self.set_block_state(state, block_state)
        return components, state


class Flux2VaeEncoderStep(ModularPipelineBlocks):
    model_name = "flux2"

    @property
    def description(self) -> str:
        return "VAE Encoder step that encodes preprocessed images into latent representations for Flux2."

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [ComponentSpec("vae", AutoencoderKLFlux2)]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("condition_images", type_hint=List[torch.Tensor]),
            InputParam("generator"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "image_latents",
                type_hint=List[torch.Tensor],
                description="List of latent representations for each reference image",
            ),
        ]

    @staticmethod
    def _patchify_latents(latents):
        """Convert latents to patchified format for Flux2."""
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
        return latents

    def _encode_vae_image(self, vae: AutoencoderKLFlux2, image: torch.Tensor, generator: torch.Generator):
        """Encode a single image using Flux2 VAE with batch norm normalization."""
        if image.ndim != 4:
            raise ValueError(f"Expected image dims 4, got {image.ndim}.")

        image_latents = retrieve_latents(vae.encode(image), generator=generator, sample_mode="argmax")
        image_latents = self._patchify_latents(image_latents)

        latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(image_latents.device, image_latents.dtype)
        latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
        latents_bn_std = latents_bn_std.to(image_latents.device, image_latents.dtype)
        image_latents = (image_latents - latents_bn_mean) / latents_bn_std

        return image_latents

    @torch.no_grad()
    def __call__(self, components: Flux2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        condition_images = block_state.condition_images

        if condition_images is None:
            return components, state

        device = components._execution_device
        dtype = components.vae.dtype

        image_latents = []
        for image in condition_images:
            image = image.to(device=device, dtype=dtype)
            latent = self._encode_vae_image(
                vae=components.vae,
                image=image,
                generator=block_state.generator,
            )
            image_latents.append(latent)

        block_state.image_latents = image_latents

        self.set_block_state(state, block_state)
        return components, state
