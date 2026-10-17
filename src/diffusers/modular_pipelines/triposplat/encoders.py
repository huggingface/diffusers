# Copyright 2026 The HuggingFace Team. All rights reserved.
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


import torch
from PIL import Image
from torch.nn import functional as F
from transformers import DINOv3ViTModel

from ...configuration_utils import FrozenDict
from ...image_processor import TripoSplatImageProcessor
from ...models import AutoencoderKLFlux2
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipeline, ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


def processor_spec():
    return ComponentSpec(
        "image_processor",
        TripoSplatImageProcessor,
        config=FrozenDict({"canvas_size": 1024}),
        default_creation_method="from_config",
    )


class TripoSplatImagePreprocessStep(ModularPipelineBlocks):
    model_name = "triposplat"

    @property
    def expected_components(self):
        from diffusers import BiRefNetModel

        return [processor_spec(), ComponentSpec("background_remover", BiRefNetModel)]

    @property
    def description(self):
        return "Prepare foreground images on a black square canvas."

    @property
    def inputs(self):
        return [
            InputParam.template("image", required=True),
            InputParam("erode_radius", default=1, type_hint=int, description="Radius of the foreground mask erosion."),
            InputParam(
                "is_preprocessed",
                default=False,
                type_hint=bool,
                description="Whether inputs are already cropped RGB images composited on black.",
            ),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam(
                "preprocessed_images", type_hint=list, description="Prepared RGB images on black square canvases."
            )
        ]

    @torch.no_grad()
    def __call__(self, components: ModularPipeline, state: PipelineState) -> tuple[ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        images = components.image_processor.to_pil(block_state.image)
        if not block_state.is_preprocessed:
            images = components.image_processor.resize_shortest_edge(images)
            for index, item in enumerate(images):
                if item.mode == "RGBA" and item.getchannel("A").getextrema()[0] < 255:
                    continue
                if components.background_remover is None:
                    raise ValueError(
                        "RGB inputs require background_remover, or set is_preprocessed=True for prepared RGB images."
                    )
                pixels = components.image_processor.preprocess([item.convert("RGB")])
                pixels = F.interpolate(
                    pixels,
                    size=(components.background_remover.config.sample_size,) * 2,
                    mode="bilinear",
                    align_corners=True,
                )
                mean = pixels.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = pixels.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                pixels = ((pixels - mean) / std).to(
                    device=components._execution_device, dtype=components.background_remover.dtype
                )
                alpha = components.background_remover(pixels).sample
                alpha = F.interpolate(
                    alpha.float(), size=(item.height, item.width), mode="bilinear", align_corners=True
                )[0, 0]
                alpha = (alpha.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
                item = item.convert("RGB")
                item.putalpha(Image.fromarray(alpha))
                images[index] = item
        block_state.preprocessed_images = components.image_processor.prepare_foreground(
            images, erode_radius=block_state.erode_radius, is_preprocessed=block_state.is_preprocessed
        )
        self.set_block_state(state, block_state)
        return components, state


class TripoSplatImageEncoderStep(ModularPipelineBlocks):
    model_name = "triposplat"

    @property
    def expected_components(self):
        return [processor_spec(), ComponentSpec("image_encoder", DINOv3ViTModel)]

    @property
    def description(self):
        return "Encode prepared images into DINOv3 features."

    @property
    def inputs(self):
        return [InputParam("preprocessed_images", required=True, type_hint=list, description="Prepared RGB images.")]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam(
                "encoder_hidden_states", type_hint=torch.Tensor, description="Normalized DINOv3 image features."
            )
        ]

    @torch.no_grad()
    def __call__(self, components: ModularPipeline, state: PipelineState) -> tuple[ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        pixels = components.image_processor.preprocess(block_state.preprocessed_images).to(
            device=components._execution_device
        )
        mean = pixels.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = pixels.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        pixels = ((pixels - mean) / std).to(components.image_encoder.dtype)
        original_inv_freq = components.image_encoder.rope_embeddings.inv_freq
        try:
            components.image_encoder.rope_embeddings.inv_freq = original_inv_freq.to(
                dtype=components.image_encoder.dtype
            )
            features = components.image_encoder(pixel_values=pixels).last_hidden_state
        finally:
            components.image_encoder.rope_embeddings.inv_freq = original_inv_freq
        block_state.encoder_hidden_states = F.layer_norm(features.float(), features.shape[-1:])
        self.set_block_state(state, block_state)
        return components, state


class TripoSplatVaeEncoderStep(ModularPipelineBlocks):
    model_name = "triposplat"

    @property
    def expected_components(self):
        return [processor_spec(), ComponentSpec("vae", AutoencoderKLFlux2)]

    @property
    def expected_configs(self):
        from ..modular_pipeline_utils import ConfigSpec

        return [ConfigSpec("num_prefix_tokens", default=5)]

    @property
    def description(self):
        return "Sample and normalize packed Flux2 VAE image latents."

    @property
    def inputs(self):
        return [
            InputParam("preprocessed_images", required=True, type_hint=list, description="Prepared RGB images."),
            InputParam.template("generator", type_hint=torch.Generator | list[torch.Generator]),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam(
                "image_latents", type_hint=torch.Tensor, description="Packed VAE latents with zero prefix tokens."
            )
        ]

    @torch.no_grad()
    def __call__(self, components: ModularPipeline, state: PipelineState) -> tuple[ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        generator = block_state.generator
        if isinstance(generator, list):
            if len(generator) % len(block_state.preprocessed_images):
                raise ValueError("The generator list must contain one generator per image or generated sample.")
            generator = generator[:: len(generator) // len(block_state.preprocessed_images)]
        pixels = components.image_processor.preprocess(block_state.preprocessed_images).to(
            device=components._execution_device, dtype=components.vae.dtype
        )
        pixels = components.image_processor.normalize(pixels)
        parameters = components.vae.encode(pixels).latent_dist.parameters
        # TripoSplat samples the posterior variance without the distribution's log-variance clamp.
        mean, logvar = parameters.chunk(2, dim=1)
        noise = randn_tensor(mean.shape, generator=generator, device=components._execution_device, dtype=mean.dtype)
        latents = mean + torch.exp(0.5 * logvar) * noise
        batch, channels, height, width = latents.shape
        latents = latents.view(batch, channels, height // 2, 2, width // 2, 2).permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch, channels * 4, height // 2, width // 2)
        bn_mean = components.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(components.vae.bn.running_var.view(1, -1, 1, 1) + components.vae.bn.eps).to(
            latents.device, latents.dtype
        )
        latents = ((latents - bn_mean) / bn_std).float().flatten(2).transpose(1, 2).contiguous()
        block_state.image_latents = F.pad(latents, (0, 0, components.config.num_prefix_tokens, 0))
        self.set_block_state(state, block_state)
        return components, state
