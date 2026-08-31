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

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

from diffusers import (
    AutoencoderKLLTX2Audio,
    AutoencoderKLLTX2Video,
    FlowMatchEulerDiscreteScheduler,
    LTX2VideoTransformer3DModel,
    LTXEulerAncestralRFScheduler,
)
from diffusers.pipelines.ltx2 import LTX2LatentUpsamplerModel, LTX2TextConnectors
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder


BASE_TEXT_ENCODER_CKPT_ID = "hf-internal-testing/tiny-gemma3"


def get_dfr_dummy_components(*, spatial_upsampler: bool = False, temporal_upsampler: bool = False):
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
        vae_scale_factors=(2, 2, 2),
        use_keyframes_abs_pos_embedding=True,
    )
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

    components = {
        "transformer": transformer,
        "vae": vae,
        "audio_vae": audio_vae,
        "scheduler": FlowMatchEulerDiscreteScheduler(),
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "connectors": connectors,
        "vocoder": vocoder,
        "processor": None,
        "prompt_enhancer": None,
        "duration_head": None,
    }
    if spatial_upsampler:
        torch.manual_seed(0)
        components["latent_upsampler"] = LTX2LatentUpsamplerModel(
            in_channels=4,
            mid_channels=32,
            num_blocks_per_stage=1,
            dims=3,
            spatial_upsample=True,
            temporal_upsample=False,
            use_rational_resampler=False,
        )
    if temporal_upsampler:
        torch.manual_seed(0)
        components["temporal_latent_upsampler"] = LTX2LatentUpsamplerModel(
            in_channels=4,
            mid_channels=32,
            num_blocks_per_stage=1,
            dims=3,
            spatial_upsample=False,
            temporal_upsample=True,
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
