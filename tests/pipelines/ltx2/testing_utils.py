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
per-pipeline configs subclass `LTX2BaseTesterConfig`. The DFR pipelines drive their fixtures through the
module-level `get_dfr_*` helpers instead, because their tests build extra pipelines mid-test rather than only
through a tester config. The scoped tester mixins below carry the pipeline-family-wide skips so each test file
does not restate them.
"""

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

from diffusers import (
    AutoencoderKLLTX2Audio,
    AutoencoderKLLTX2Video,
    FlowMatchEulerDiscreteScheduler,
    LTX2VideoTransformer3DModel,
    LTXEulerAncestralRFScheduler,
)
from diffusers.pipelines.ltx2 import LTX2DurationHead, LTX2LatentUpsamplerModel, LTX2TextConnectors
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder

from ...testing_utils import torch_device
from ..testing_utils import (
    BasePipelineTesterConfig,
    LoraMemoryTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
)


BASE_TEXT_ENCODER_CKPT_ID = "hf-internal-testing/tiny-gemma3"

# Components the LTX2 pipelines accept but that these fast tests leave unset. Which ones exist differs per
# pipeline — only the pipelines that can predict a duration take a `duration_head`, and only some take an
# `audio_scheduler` — so each config lists its own set and `get_ltx2_dummy_components` fills them with `None`.
DEFAULT_UNSET_COMPONENTS = ("processor", "prompt_enhancer", "duration_head")

# The DFR transformer places its keyframes with an absolute positional embedding the other LTX2 pipelines do not
# use, and it reads the VAE strides off its own config to map keyframe positions onto latent frames.
DFR_TRANSFORMER_KWARGS = {"vae_scale_factors": (2, 2, 2), "use_keyframes_abs_pos_embedding": True}


def get_dummy_vae(**overrides):
    """The tiny `AutoencoderKLLTX2Video` every LTX2 test uses; `overrides` go straight to the constructor.

    Seeding is left to the caller so that each fixture keeps the RNG stream the hardcoded slices were recorded
    against.
    """
    kwargs = {
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": 4,
        "block_out_channels": (8,),
        "decoder_block_out_channels": (8,),
        "layers_per_block": (1,),
        "decoder_layers_per_block": (1, 1),
        "spatio_temporal_scaling": (True,),
        "decoder_spatio_temporal_scaling": (True,),
        "decoder_inject_noise": (False, False),
        "downsample_type": ("spatial",),
        "upsample_residual": (False,),
        "upsample_factor": (1,),
        "timestep_conditioning": False,
        "patch_size": 1,
        "patch_size_t": 1,
        "encoder_causal": True,
        "decoder_causal": False,
    }
    kwargs.update(overrides)

    vae = AutoencoderKLLTX2Video(**kwargs)
    vae.use_framewise_encoding = False
    vae.use_framewise_decoding = False
    return vae


def get_dummy_latent_upsampler(**overrides):
    """The tiny `LTX2LatentUpsamplerModel`; `overrides` go straight to the constructor.

    Seeding is left to the caller, as in `get_dummy_vae`.
    """
    kwargs = {"in_channels": 4, "mid_channels": 32, "num_blocks_per_stage": 1}
    kwargs.update(overrides)

    return LTX2LatentUpsamplerModel(**kwargs)


def get_ltx2_dummy_components(*, unset_components=DEFAULT_UNSET_COMPONENTS, transformer_kwargs=None):
    """The dummy sub-modules shared by every LTX2 pipeline.

    `transformer_kwargs` overrides the transformer config for the variants that need a different one (see
    `DFR_TRANSFORMER_KWARGS`); `unset_components` names the optional components to hand back as `None`.
    """
    tokenizer = AutoTokenizer.from_pretrained(BASE_TEXT_ENCODER_CKPT_ID)
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(BASE_TEXT_ENCODER_CKPT_ID)

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
        **(transformer_kwargs or {}),
    )
    if transformer.config.use_keyframes_abs_pos_embedding:
        # Registered zero-initialized, so without this the keyframe marker would be invisible to the transformer
        # and the tests that assert it reaches the forward pass could not tell it apart from an unmarked run.
        torch.nn.init.normal_(transformer.keyframes_abs_pos_embedding, std=0.1)

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
    vae = get_dummy_vae()

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

    return {
        "transformer": transformer,
        "vae": vae,
        "audio_vae": audio_vae,
        "scheduler": FlowMatchEulerDiscreteScheduler(),
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "connectors": connectors,
        "vocoder": vocoder,
        **dict.fromkeys(unset_components),
    }


def get_dummy_duration_head():
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


def get_dfr_dummy_components(*, spatial_upsampler: bool = False, temporal_upsampler: bool = False):
    """The shared LTX2 components with the DFR transformer, plus the upsamplers a multi-stage run needs."""
    components = get_ltx2_dummy_components(transformer_kwargs=DFR_TRANSFORMER_KWARGS)
    if spatial_upsampler:
        torch.manual_seed(0)
        components["latent_upsampler"] = get_dummy_latent_upsampler(
            dims=3, spatial_upsample=True, temporal_upsample=False, use_rational_resampler=False
        )
    if temporal_upsampler:
        torch.manual_seed(0)
        components["temporal_latent_upsampler"] = get_dummy_latent_upsampler(
            dims=3, spatial_upsample=False, temporal_upsample=True
        )
    return components


def get_temporal_dummy_components():
    components = get_dfr_dummy_components(temporal_upsampler=True)
    components["scheduler"] = LTXEulerAncestralRFScheduler(eta=0.5)
    # A refine round never enhances a prompt -- stage 1 already did, and re-enhancing would
    # denoise the canvas under a different prompt than the one that generated it.
    for name in ("duration_head", "processor", "prompt_enhancer"):
        components.pop(name)
    return components


def get_dfr_dummy_inputs(**overrides):
    inputs = {
        "prompt": "a robot dancing",
        "height": 32,
        "width": 32,
        "num_frames": 9,
        "frame_rate": 25.0,
        "sigmas": [1.0, 0.5],
        "use_cross_timestep": False,
        "max_sequence_length": 16,
        "output_type": "pt",
    }
    inputs.update(overrides)
    return inputs


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

    base_text_encoder_ckpt_id = BASE_TEXT_ENCODER_CKPT_ID

    # See `DEFAULT_UNSET_COMPONENTS`; each config narrows or widens this to the components its pipeline accepts.
    unset_components = DEFAULT_UNSET_COMPONENTS

    # `audio_vae` fails at block level for the same reason the other VAEs do: its decode-time convolutions run
    # without the group leader's `forward` having onloaded the group.
    group_offloading_block_level_exclude_modules = ["vae", "audio_vae"]

    def get_dummy_components(self):
        return get_ltx2_dummy_components(unset_components=self.unset_components)

    def get_dummy_duration_head(self):
        """A tiny `LTX2DurationHead`, for the pipelines that accept one (`duration_head` in `unset_components`)."""
        return get_dummy_duration_head()

    def get_pipeline_with_duration_head(self):
        components = self.get_dummy_components()
        components["duration_head"] = self.get_dummy_duration_head()
        return self.get_pipeline(**components).to(torch_device)


class LTX2MemoryTesterMixin(MemoryTesterMixin):
    """`MemoryTesterMixin` for the LTX2 pipelines in this directory."""


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
