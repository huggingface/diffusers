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
from .packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_FPS,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MiniMaxH3ModularPipeline(ModularPipeline):
    """
    A ModularPipeline for joint video + audio generation with MiniMax-H3, covering the `t2va` (text only) and `fl2va`
    (first and/or last keyframe) tasks of the FL2VA checkpoint.

    MiniMax-H3 denoises **one packed sequence** that holds the text conditioning, the keyframe conditioning latents,
    the audio latents and the video latents at once, which is why the blocks pass a row layout around rather than
    per-modality tensors, and why the pipeline carries two schedulers (`shift = 12.0` for video, `shift = 3.0` for
    audio) that are stepped inside a single transformer call.

    The checkpoint is guidance-distilled: guidance is baked into the weights, so there is no guider, no
    `negative_prompt` and no `guidance_scale`, and every step runs exactly one forward pass.

    MiniMax-H3 is modular only: this pipeline and its blocks are the whole integration, there is no
    `DiffusionPipeline` half. This class carries the config-derived geometry the blocks read off the components, the
    packed-sequence geometry lives in `modular_pipelines.minimax_h3.packing`, and the conditioning, encoding and noise
    contracts live on the blocks themselves.

    ```py
    import torch
    from diffusers import ModularPipeline

    pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3")
    pipe.load_components(dtype=torch.bfloat16)
    ```

    > [!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    default_blocks_name = "MiniMaxH3Blocks"

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
    def vae_frames_per_chunk(self):
        if getattr(self, "vae", None) is not None:
            return self.vae.config.clip_length
        return 17

    @property
    def vae_latents_per_chunk(self):
        if getattr(self, "vae", None) is not None:
            return self.vae.tokens_chunk_size
        return 5

    @property
    def audio_sampling_rate(self):
        if getattr(self, "audio_vae", None) is not None:
            return self.audio_vae.config.sampling_rate
        return 32000

    @property
    def audio_latent_channels(self):
        if getattr(self, "audio_vae", None) is not None:
            return self.audio_vae.config.latent_channels
        return 32

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
    def fps(self):
        r"""MiniMax-H3's own frame rate. Everything it generates and conditions on is resampled onto it."""
        return MINIMAX_H3_FPS

    @property
    def min_duration(self):
        r"""Shortest video MiniMax-H3 generates, in seconds."""
        return 5.0

    @property
    def max_duration(self):
        r"""Longest video MiniMax-H3 generates, in seconds."""
        return 15.0

    @property
    def audio_channels(self):
        r"""Channels of the generated soundtrack: MiniMax-H3 is stereo, packed channel-major."""
        return MINIMAX_H3_AUDIO_CHANNELS

    @property
    def text_encoder_layer(self):
        r"""
        Which Qwen3-VL hidden state conditions the transformer.

        MiniMax-H3 reads `hidden_states[50]`, not the final one: the last layer is post-norm and is not the
        conditioning the released weights were trained against.
        """
        return 50

    @property
    def pixel_mean(self):
        r"""Per-channel mean the video VAE's input is normalized by, ImageNet's."""
        return (0.485, 0.456, 0.406)

    @property
    def pixel_std(self):
        r"""Per-channel standard deviation the video VAE's input is normalized by, ImageNet's."""
        return (0.229, 0.224, 0.225)

    @property
    def keyframe_encode_seed(self):
        r"""
        Seed the conditioning posterior is sampled under, independently of the request's own generator.

        Fixed at 42 in the reference implementation, so the same keyframe always encodes to the same anchor.
        """
        return 42

    @property
    def keyframe_noise_aug(self):
        r"""
        The `t` a visual conditioning anchor is held at: 0.999, just short of clean.

        The released model was trained with its anchors very slightly noised, so conditioning on exactly `t = 1.0`
        is off-distribution.
        """
        return 0.999

    @property
    def text_tag(self):
        r"""The modality tag of a text row of the packed sequence."""
        return MINIMAX_H3_TEXT_TAG

    @property
    def video_tag(self):
        r"""The modality tag of a video row of the packed sequence, which a vision block's rows also carry."""
        return MINIMAX_H3_VIDEO_TAG


class MiniMaxH3Ref2VAModularPipeline(MiniMaxH3ModularPipeline):
    """
    A ModularPipeline for joint video + audio generation from omni-references with MiniMax-H3, the `ref2va` task of the
    Ref2VA checkpoint.

    A request carries an ordered list of references — up to 9 images, 3 videos and 3 audio clips, 12 in total — and
    MiniMax-H3 packs one block per reference in front of the generated rows. The order is semantic: it labels the
    references in the prompt presentation and it advances the shared audio/video rotary clock, so reordering the same
    references is a different request.

    The transformer is registered as `transformer_ref`, so one repository can hold both checkpoint partitions
    (`transformer/` for [`MiniMaxH3ModularPipeline`], `transformer_ref/` for this one) and either pipeline loads only
    its own weights. One repository also means one `modular_model_index.json`, which names the `t2va` / `fl2va` half;
    load this one through its blocks instead, which reads the very same file:

    ```py
    pipe = MiniMaxH3Ref2VABlocks().init_pipeline("MiniMaxAI/MiniMax-H3")
    pipe.load_components(dtype=torch.bfloat16)
    ```

    The blocks carry the `ref2va` conditioning, encoding and noise contracts themselves.

    > [!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    default_blocks_name = "MiniMaxH3Ref2VABlocks"

    @property
    def patch_size(self):
        if getattr(self, "transformer_ref", None) is not None:
            return tuple(self.transformer_ref.config.patch_size)
        return (1, 2, 2)
