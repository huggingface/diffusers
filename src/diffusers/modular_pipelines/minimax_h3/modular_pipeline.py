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
