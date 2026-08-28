# Copyright 2026 The HuggingFace Team.
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
"""Shared test fixtures for the LTX2 pipelines.

Every LTX2 pipeline in this directory takes the same set of dummy sub-modules — they differ only in which
optional components they accept and in what `__call__` takes — so the component builder lives here and the
per-pipeline configs subclass `LTX2BaseTesterConfig`. The two scoped tester mixins below carry the
pipeline-family-wide skips so each test file does not restate them.
"""

import pytest
import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

from diffusers import (
    AutoencoderKLLTX2Audio,
    AutoencoderKLLTX2Video,
    FlowMatchEulerDiscreteScheduler,
    LTX2VideoTransformer3DModel,
)
from diffusers.pipelines.ltx2 import LTX2DurationHead, LTX2TextConnectors
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder

from ...testing_utils import torch_device
from ..testing_utils import (
    BasePipelineTesterConfig,
    LoraMemoryTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
)


class LTX2BaseTesterConfig(BasePipelineTesterConfig):
    """Dummy component set shared by every LTX2 pipeline in this directory."""

    # LTX2 is a video pipeline (`num_videos_per_prompt`, not `num_images_per_prompt`) and takes a second latent
    # input for the audio stream. `LTX2HDRPipeline` renders video only and overrides this without `audio_latents`.
    optional_input_params = frozenset(
        [
            "num_inference_steps",
            "num_videos_per_prompt",
            "generator",
            "latents",
            "audio_latents",
            "output_type",
            "return_dict",
        ]
    )
    output_shape = (5, 3, 32, 32)

    base_text_encoder_ckpt_id = "hf-internal-testing/tiny-gemma3"

    # Components the pipeline accepts but that these fast tests leave unset. Which ones exist differs per
    # pipeline — only the pipelines that can predict a duration take a `duration_head`, and only some take an
    # `audio_scheduler` — so each config lists its own set and `get_dummy_components` fills them with `None`.
    unset_components = ("processor", "prompt_enhancer", "duration_head")

    def get_dummy_components(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_text_encoder_ckpt_id)
        text_encoder = Gemma3ForConditionalGeneration.from_pretrained(self.base_text_encoder_ckpt_id)

        torch.manual_seed(0)
        transformer = LTX2VideoTransformer3DModel(
            in_channels=4,
            out_channels=4,
            patch_size=1,
            patch_size_t=1,
            num_attention_heads=2,
            attention_head_dim=8,
            cross_attention_dim=16,
            audio_in_channels=4,
            audio_out_channels=4,
            audio_num_attention_heads=2,
            audio_attention_head_dim=4,
            audio_cross_attention_dim=8,
            num_layers=2,
            qk_norm="rms_norm_across_heads",
            caption_channels=text_encoder.config.text_config.hidden_size,
            rope_double_precision=False,
            rope_type="split",
        )

        torch.manual_seed(0)
        connectors = LTX2TextConnectors(
            caption_channels=text_encoder.config.text_config.hidden_size,
            text_proj_in_factor=text_encoder.config.text_config.num_hidden_layers + 1,
            video_connector_num_attention_heads=4,
            video_connector_attention_head_dim=8,
            video_connector_num_layers=1,
            video_connector_num_learnable_registers=None,
            audio_connector_num_attention_heads=4,
            audio_connector_attention_head_dim=8,
            audio_connector_num_layers=1,
            audio_connector_num_learnable_registers=None,
            connector_rope_base_seq_len=32,
            rope_theta=10000.0,
            rope_double_precision=False,
            causal_temporal_positioning=False,
            rope_type="split",
        )

        torch.manual_seed(0)
        vae = AutoencoderKLLTX2Video(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            block_out_channels=(8,),
            decoder_block_out_channels=(8,),
            layers_per_block=(1,),
            decoder_layers_per_block=(1, 1),
            spatio_temporal_scaling=(True,),
            decoder_spatio_temporal_scaling=(True,),
            decoder_inject_noise=(False, False),
            downsample_type=("spatial",),
            upsample_residual=(False,),
            upsample_factor=(1,),
            timestep_conditioning=False,
            patch_size=1,
            patch_size_t=1,
            encoder_causal=True,
            decoder_causal=False,
        )
        vae.use_framewise_encoding = False
        vae.use_framewise_decoding = False

        torch.manual_seed(0)
        audio_vae = AutoencoderKLLTX2Audio(
            base_channels=4,
            output_channels=2,
            ch_mult=(1,),
            num_res_blocks=1,
            attn_resolutions=None,
            in_channels=2,
            resolution=32,
            latent_channels=2,
            norm_type="pixel",
            causality_axis="height",
            dropout=0.0,
            mid_block_add_attention=False,
            sample_rate=16000,
            mel_hop_length=160,
            is_causal=True,
            mel_bins=8,
        )

        torch.manual_seed(0)
        vocoder = LTX2Vocoder(
            in_channels=audio_vae.config.output_channels * audio_vae.config.mel_bins,
            hidden_channels=32,
            out_channels=2,
            upsample_kernel_sizes=[4, 4],
            upsample_factors=[2, 2],
            resnet_kernel_sizes=[3],
            resnet_dilations=[[1, 3, 5]],
            leaky_relu_negative_slope=0.1,
            output_sampling_rate=16000,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {
            "transformer": transformer,
            "vae": vae,
            "audio_vae": audio_vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "connectors": connectors,
            "vocoder": vocoder,
            **dict.fromkeys(self.unset_components),
        }

    def get_dummy_duration_head(self):
        """A tiny `LTX2DurationHead`, for the pipelines that accept one (`duration_head` in `unset_components`)."""
        torch.manual_seed(0)
        # The dummy connectors emit 4 heads * 8 head_dim = 32 wide output for both streams.
        return LTX2DurationHead(
            video_cross_attention_dim=32,
            audio_cross_attention_dim=32,
            pooler_hidden_dim=8,
            num_queries=1,
            num_pooler_heads=2,
            mlp_hidden_dim=8,
        )

    def get_pipeline_with_duration_head(self):
        components = self.get_dummy_components()
        components["duration_head"] = self.get_dummy_duration_head()
        return self.get_pipeline(**components).to(torch_device)


class LTX2MemoryTesterMixin(MemoryTesterMixin):
    """`MemoryTesterMixin` for the LTX2 pipelines in this directory."""

    # The shared helper only offloads a fixed set of component names and leaves LTX2's extra module components
    # (`connectors`, `audio_vae`, `vocoder`) on CPU, so the forward pass mixes devices. Pipeline-level
    # offloading, which walks every component, is exercised by `test_pipeline_level_group_offloading_inference`.
    @pytest.mark.skip("Using test_pipeline_level_group_offloading_inference instead")
    def test_group_offloading_inference(self):
        pass


class LTX2LoraTesterMixin(LoraTesterMixin):
    """`LoraTesterMixin` for the LTX2 pipelines in this directory.

    Every LTX2 pipeline advertises `connectors` as LoRA-loadable and `load_lora_weights` does handle connector
    LoRAs, but `save_lora_weights` only accepts `transformer_lora_layers` — so connector adapters cannot
    round-trip through the public API these tests drive. Scope the tests to the transformer until that closes.
    """

    lora_loadable_components = ["transformer"]


class LTX2LoraMemoryTesterMixin(LoraMemoryTesterMixin):
    """`LoraMemoryTesterMixin` for the LTX2 pipelines, scoped to the transformer — see `LTX2LoraTesterMixin`."""

    lora_loadable_components = ["transformer"]
