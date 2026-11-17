# Copyright 2026 Pruna AI, The MiniMax and HuggingFace Teams. All rights reserved.
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


class PH3ImageModularPipeline(ModularPipeline):
    """
    A ModularPipeline for P-H3-Image, an image model based on MiniMax-H3, for text-to-image (`text2image`) and
    instruction-based editing (`image_conditioned`).

    The image is a single latent frame, denoised in one packed sequence with two audio latents that are discarded.

    ```py
    import torch
    from diffusers import ComponentsManager, ModularPipeline

    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device="cuda")

    pipe = ModularPipeline.from_pretrained("PrunaAI/P-H3-Image", components_manager=manager)
    pipe.load_components(dtype=torch.bfloat16)
    prompt = (
        "A German Expressionist oil painting of a Berlin street at night in the manner of Ernst Ludwig Kirchner, "
        "with gaunt figures hurrying past a glowing gas lamp."
    )
    image = pipe(prompt=prompt, height=3360, width=2496, output="images")[0]
    ```
    """

    default_blocks_name = "PH3ImageBlocks"

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
    def audio_latent_channels(self):
        r"""Channels of the audio latents, read off the transformer since there is no audio VAE."""
        if getattr(self, "transformer", None) is not None:
            return self.transformer.config.audio_in_channels
        return 32

    @property
    def num_audio_latents(self):
        r"""Audio latents per channel denoised alongside the image."""
        return 2

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
    def audio_channels(self):
        r"""Channels of the audio latents: MiniMax-H3's soundtrack is stereo, packed channel-major."""
        return 2

    @property
    def pixel_mean(self):
        r"""Per-channel mean the VAE's input is normalized by, ImageNet's."""
        return (0.485, 0.456, 0.406)

    @property
    def pixel_std(self):
        r"""Per-channel standard deviation the VAE's input is normalized by, ImageNet's."""
        return (0.229, 0.224, 0.225)

    @property
    def keyframe_encode_seed(self):
        r"""Seed the conditioning posterior of an edit is sampled under, independently of the request's generator."""
        return 42

    @property
    def keyframe_noise_aug(self):
        r"""The `t` the conditioning rows of an edit are held at: 0.999, just short of clean."""
        return 0.999

    @property
    def text_tag(self):
        r"""The modality tag of a text row of the packed sequence."""
        return 1

    @property
    def video_tag(self):
        r"""The modality tag of an image row of the packed sequence, which a vision block's rows also carry."""
        return 0

    @property
    def audio_tag(self):
        r"""The modality tag of an audio row of the packed sequence."""
        return 2
