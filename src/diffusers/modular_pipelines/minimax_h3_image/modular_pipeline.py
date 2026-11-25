# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
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

from ...utils import logging
from ..modular_pipeline import ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Per-row modality tags. They index the transformer's AdaLN table, so the values are a checkpoint contract.
MINIMAX_H3_IMAGE_TAG = 0
MINIMAX_H3_TEXT_TAG = 1


class MiniMaxH3ImageModularPipeline(ModularPipeline):
    """
    A ModularPipeline for MiniMax-H3-Image: text-to-image (`text2image`) and reference-image editing
    (`image_conditioned`).

    MiniMax-H3-Image runs the MiniMax-H3 transformer without audio over one packed sequence of text rows, the rows of
    the reference images and the rows of the generated image, which is a single latent frame. The checkpoint is
    distilled twice over: guidance is baked into the weights, so there is no guider, no `negative_prompt` and no
    `guidance_scale`, and it samples on a fixed grid of six steps with one output head of the transformer per step, so
    there is no `num_inference_steps` either.

    ```py
    import torch
    from diffusers import ModularPipeline

    pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3-Image")
    pipe.load_components(dtype=torch.bfloat16)
    ```

    """

    default_blocks_name = "MiniMaxH3ImageBlocks"

    @property
    def vae_spatial_compression_ratio(self):
        if getattr(self, "vae", None) is not None:
            return self.vae.spatial_compression_ratio
        return 16

    @property
    def vae_latent_channels(self):
        if getattr(self, "vae", None) is not None:
            return self.vae.config.latent_channels
        return 24

    @property
    def patch_size(self):
        if getattr(self, "transformer", None) is not None:
            return tuple(self.transformer.config.patch_size)
        return (1, 2, 2)

    @property
    def canvas_multiple(self):
        r"""What the generated height and width have to be a multiple of, 32 for the released checkpoint."""
        # A canvas has to survive the VAE's spatial compression and still be a whole number of patch rows wide, so
        # the multiple is the product of the two.
        return self.vae_spatial_compression_ratio * self.patch_size[2]

    @property
    def text_encoder_layer(self):
        r"""Which Qwen3-VL hidden state conditions the transformer: `hidden_states[50]`, as in MiniMax-H3."""
        return 50

    @property
    def max_sequence_length(self):
        r"""How many tokens of the prompt, and of each reference label, the presentation keeps."""
        return 4096

    @property
    def pixel_mean(self):
        r"""Per-channel mean the VAE's input is normalized by, ImageNet's."""
        return (0.485, 0.456, 0.406)

    @property
    def pixel_std(self):
        r"""Per-channel standard deviation the VAE's input is normalized by, ImageNet's."""
        return (0.229, 0.224, 0.225)

    @property
    def reference_noise_aug(self):
        r"""The `t` the reference rows of an edit are held at for every step: 0.999, just short of clean."""
        return 0.999

    @property
    def text_tag(self):
        r"""The modality tag of a text row of the packed sequence."""
        return MINIMAX_H3_TEXT_TAG

    @property
    def image_tag(self):
        r"""The modality tag of an image row of the packed sequence, which a vision block's rows also carry."""
        return MINIMAX_H3_IMAGE_TAG
