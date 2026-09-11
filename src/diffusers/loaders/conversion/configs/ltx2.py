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

"""Original configuration helpers and model presets for the ltx2 assembly recipe."""

from typing import Any


def get_ltx2_transformer_config(version: str) -> dict[str, Any]:
    if version == "test":
        # Produces a transformer of the same size as used in test_models_transformer_ltx2.py
        config = {
            "model_id": "diffusers-internal-dev/dummy-ltx2",
            "diffusers_config": {
                "in_channels": 4,
                "out_channels": 4,
                "patch_size": 1,
                "patch_size_t": 1,
                "num_attention_heads": 2,
                "attention_head_dim": 8,
                "cross_attention_dim": 16,
                "vae_scale_factors": (8, 32, 32),
                "pos_embed_max_pos": 20,
                "base_height": 2048,
                "base_width": 2048,
                "audio_in_channels": 4,
                "audio_out_channels": 4,
                "audio_patch_size": 1,
                "audio_patch_size_t": 1,
                "audio_num_attention_heads": 2,
                "audio_attention_head_dim": 4,
                "audio_cross_attention_dim": 8,
                "audio_scale_factor": 4,
                "audio_pos_embed_max_pos": 20,
                "audio_sampling_rate": 16000,
                "audio_hop_length": 160,
                "num_layers": 2,
                "activation_fn": "gelu-approximate",
                "qk_norm": "rms_norm_across_heads",
                "norm_elementwise_affine": False,
                "norm_eps": 1e-6,
                "caption_channels": 16,
                "attention_bias": True,
                "attention_out_bias": True,
                "rope_theta": 10000.0,
                "rope_double_precision": False,
                "causal_offset": 1,
                "timestep_scale_multiplier": 1000,
                "cross_attn_timestep_scale_multiplier": 1,
            },
        }
    elif version == "2.0":
        config = {
            "model_id": "Lightricks/LTX-2",
            "diffusers_config": {
                "in_channels": 128,
                "out_channels": 128,
                "patch_size": 1,
                "patch_size_t": 1,
                "num_attention_heads": 32,
                "attention_head_dim": 128,
                "cross_attention_dim": 4096,
                "vae_scale_factors": (8, 32, 32),
                "pos_embed_max_pos": 20,
                "base_height": 2048,
                "base_width": 2048,
                "gated_attn": False,
                "cross_attn_mod": False,
                "audio_in_channels": 128,
                "audio_out_channels": 128,
                "audio_patch_size": 1,
                "audio_patch_size_t": 1,
                "audio_num_attention_heads": 32,
                "audio_attention_head_dim": 64,
                "audio_cross_attention_dim": 2048,
                "audio_scale_factor": 4,
                "audio_pos_embed_max_pos": 20,
                "audio_sampling_rate": 16000,
                "audio_hop_length": 160,
                "audio_gated_attn": False,
                "audio_cross_attn_mod": False,
                "num_layers": 48,
                "activation_fn": "gelu-approximate",
                "qk_norm": "rms_norm_across_heads",
                "norm_elementwise_affine": False,
                "norm_eps": 1e-6,
                "caption_channels": 3840,
                "attention_bias": True,
                "attention_out_bias": True,
                "rope_theta": 10000.0,
                "rope_double_precision": True,
                "causal_offset": 1,
                "timestep_scale_multiplier": 1000,
                "cross_attn_timestep_scale_multiplier": 1000,
                "rope_type": "split",
                "use_prompt_embeddings": True,
                "perturbed_attn": False,
            },
        }
    elif version == "2.3":
        config = {
            "model_id": "Lightricks/LTX-2.3",
            "diffusers_config": {
                "in_channels": 128,
                "out_channels": 128,
                "patch_size": 1,
                "patch_size_t": 1,
                "num_attention_heads": 32,
                "attention_head_dim": 128,
                "cross_attention_dim": 4096,
                "vae_scale_factors": (8, 32, 32),
                "pos_embed_max_pos": 20,
                "base_height": 2048,
                "base_width": 2048,
                "gated_attn": True,
                "cross_attn_mod": True,
                "audio_in_channels": 128,
                "audio_out_channels": 128,
                "audio_patch_size": 1,
                "audio_patch_size_t": 1,
                "audio_num_attention_heads": 32,
                "audio_attention_head_dim": 64,
                "audio_cross_attention_dim": 2048,
                "audio_scale_factor": 4,
                "audio_pos_embed_max_pos": 20,
                "audio_sampling_rate": 16000,
                "audio_hop_length": 160,
                "audio_gated_attn": True,
                "audio_cross_attn_mod": True,
                "num_layers": 48,
                "activation_fn": "gelu-approximate",
                "qk_norm": "rms_norm_across_heads",
                "norm_elementwise_affine": False,
                "norm_eps": 1e-6,
                "caption_channels": 3840,
                "attention_bias": True,
                "attention_out_bias": True,
                "rope_theta": 10000.0,
                "rope_double_precision": True,
                "causal_offset": 1,
                "timestep_scale_multiplier": 1000,
                "cross_attn_timestep_scale_multiplier": 1000,
                "rope_type": "split",
                "use_prompt_embeddings": False,
                "perturbed_attn": True,
            },
        }
    elif version == "2.5":
        config = {
            "model_id": "Lightricks/LTX-2.5",
            "diffusers_config": {
                "in_channels": 128,
                "out_channels": 128,
                "patch_size": 1,
                "patch_size_t": 1,
                "num_attention_heads": 32,
                "attention_head_dim": 128,
                "cross_attention_dim": 4096,
                "vae_scale_factors": (8, 32, 32),
                "pos_embed_max_pos": 20,
                "base_height": 2048,
                "base_width": 2048,
                "gated_attn": True,
                "cross_attn_mod": True,
                "audio_in_channels": 128,
                "audio_out_channels": 128,
                "audio_patch_size": 1,
                "audio_patch_size_t": 1,
                "audio_num_attention_heads": 32,
                "audio_attention_head_dim": 64,
                "audio_cross_attention_dim": 2048,
                "audio_scale_factor": 4,
                "audio_pos_embed_max_pos": 20,
                "audio_sampling_rate": 16000,
                "audio_hop_length": 160,
                "audio_gated_attn": True,
                "audio_cross_attn_mod": True,
                "num_layers": 48,
                "activation_fn": "gelu-approximate",
                "qk_norm": "rms_norm_across_heads",
                "norm_elementwise_affine": False,
                "norm_eps": 1e-6,
                "caption_channels": 3840,
                "attention_bias": True,
                "attention_out_bias": True,
                "rope_theta": 10000.0,
                "rope_double_precision": True,
                "causal_offset": 1,
                "timestep_scale_multiplier": 1000,
                "cross_attn_timestep_scale_multiplier": 1000,
                "rope_type": "split",
                "use_prompt_embeddings": False,
                "perturbed_attn": True,
                # The only transformer-level deltas from 2.3: the video FFN drops its bias (audio_ff_bias and
                # use_prompt_adaln_single keep their True defaults for this checkpoint), and 2.5 carries a
                # learned keyframe absolute-position embedding.
                "ff_bias": False,
                "use_keyframes_abs_pos_embedding": True,
            },
        }
    return config


def get_ltx2_connectors_config(
    version: str, gemma_text_config: Any | None = None
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if version == "test":
        config = {
            "model_id": "diffusers-internal-dev/dummy-ltx2",
            "diffusers_config": {
                "caption_channels": 16,
                "text_proj_in_factor": 3,
                "video_connector_num_attention_heads": 4,
                "video_connector_attention_head_dim": 8,
                "video_connector_num_layers": 1,
                "video_connector_num_learnable_registers": None,
                "audio_connector_num_attention_heads": 4,
                "audio_connector_attention_head_dim": 8,
                "audio_connector_num_layers": 1,
                "audio_connector_num_learnable_registers": None,
                "connector_rope_base_seq_len": 32,
                "rope_theta": 10000.0,
                "rope_double_precision": False,
                "causal_temporal_positioning": False,
            },
        }
    elif version == "2.0":
        config = {
            "model_id": "Lightricks/LTX-2",
            "diffusers_config": {
                "caption_channels": 3840,
                "text_proj_in_factor": 49,
                "video_connector_num_attention_heads": 30,
                "video_connector_attention_head_dim": 128,
                "video_connector_num_layers": 2,
                "video_connector_num_learnable_registers": 128,
                "video_gated_attn": False,
                "audio_connector_num_attention_heads": 30,
                "audio_connector_attention_head_dim": 128,
                "audio_connector_num_layers": 2,
                "audio_connector_num_learnable_registers": 128,
                "audio_gated_attn": False,
                "connector_rope_base_seq_len": 4096,
                "rope_theta": 10000.0,
                "rope_double_precision": True,
                "causal_temporal_positioning": False,
                "rope_type": "split",
                "per_modality_projections": False,
                "proj_bias": False,
            },
        }
    elif version == "2.3":
        config = {
            "model_id": "Lightricks/LTX-2.3",
            "diffusers_config": {
                "caption_channels": 3840,
                "text_proj_in_factor": 49,
                "video_connector_num_attention_heads": 32,
                "video_connector_attention_head_dim": 128,
                "video_connector_num_layers": 8,
                "video_connector_num_learnable_registers": 128,
                "video_gated_attn": True,
                "audio_connector_num_attention_heads": 32,
                "audio_connector_attention_head_dim": 64,
                "audio_connector_num_layers": 8,
                "audio_connector_num_learnable_registers": 128,
                "audio_gated_attn": True,
                "connector_rope_base_seq_len": 4096,
                "rope_theta": 10000.0,
                "rope_double_precision": True,
                "causal_temporal_positioning": False,
                "rope_type": "split",
                "per_modality_projections": True,
                "video_hidden_dim": 4096,
                "audio_hidden_dim": 2048,
                "proj_bias": True,
            },
        }
    elif version == "2.5":
        if gemma_text_config is None:
            raise ValueError("gemma_text_config is required to derive connector dims for LTX-2.5.")
        config = {
            "model_id": "Lightricks/LTX-2.5",
            "diffusers_config": {
                # Derived from the Gemma 4 text config rather than hardcoded, since (unlike Gemma-3-12B) the
                # 2.5 text encoder isn't a single fixed checkpoint. Formula matches the reference
                # (`encoder_configurator._create_feature_extractor`): hidden_size, and num_hidden_layers + 1
                # for the embedding layer.
                "caption_channels": gemma_text_config.hidden_size,
                "text_proj_in_factor": gemma_text_config.num_hidden_layers + 1,
                "video_connector_num_attention_heads": 32,
                "video_connector_attention_head_dim": 128,
                "video_connector_num_layers": 8,
                "video_connector_num_learnable_registers": 128,
                "video_gated_attn": True,
                "audio_connector_num_attention_heads": 32,
                "audio_connector_attention_head_dim": 64,
                "audio_connector_num_layers": 8,
                "audio_connector_num_learnable_registers": 128,
                "audio_gated_attn": True,
                "connector_rope_base_seq_len": 4096,
                "rope_theta": 10000.0,
                "rope_double_precision": True,
                "causal_temporal_positioning": False,
                "rope_type": "split",
                "per_modality_projections": True,
                "video_hidden_dim": 4096,
                "audio_hidden_dim": 2048,
                "proj_bias": True,
            },
        }

    return config


def get_ltx2_video_vae_config(
    version: str, timestep_conditioning: bool = False
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if version == "test":
        config = {
            "model_id": "diffusers-internal-dev/dummy-ltx2",
            "diffusers_config": {
                "in_channels": 3,
                "out_channels": 3,
                "latent_channels": 128,
                "block_out_channels": (256, 512, 1024, 2048),
                "down_block_types": (
                    "LTX2VideoDownBlock3D",
                    "LTX2VideoDownBlock3D",
                    "LTX2VideoDownBlock3D",
                    "LTX2VideoDownBlock3D",
                ),
                "decoder_block_out_channels": (256, 512, 1024),
                "layers_per_block": (4, 6, 6, 2, 2),
                "decoder_layers_per_block": (5, 5, 5, 5),
                "spatio_temporal_scaling": (True, True, True, True),
                "decoder_spatio_temporal_scaling": (True, True, True),
                "decoder_inject_noise": (False, False, False, False),
                "downsample_type": ("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
                "upsample_residual": (True, True, True),
                "upsample_factor": (2, 2, 2),
                "timestep_conditioning": timestep_conditioning,
                "patch_size": 4,
                "patch_size_t": 1,
                "resnet_norm_eps": 1e-6,
                "encoder_causal": True,
                "decoder_causal": False,
                "encoder_spatial_padding_mode": "zeros",
                "decoder_spatial_padding_mode": "reflect",
                "spatial_compression_ratio": 32,
                "temporal_compression_ratio": 8,
            },
        }
    elif version == "2.0":
        config = {
            "model_id": "Lightricks/LTX-2",
            "diffusers_config": {
                "in_channels": 3,
                "out_channels": 3,
                "latent_channels": 128,
                "block_out_channels": (256, 512, 1024, 2048),
                "down_block_types": (
                    "LTX2VideoDownBlock3D",
                    "LTX2VideoDownBlock3D",
                    "LTX2VideoDownBlock3D",
                    "LTX2VideoDownBlock3D",
                ),
                "decoder_block_out_channels": (256, 512, 1024),
                "layers_per_block": (4, 6, 6, 2, 2),
                "decoder_layers_per_block": (5, 5, 5, 5),
                "spatio_temporal_scaling": (True, True, True, True),
                "decoder_spatio_temporal_scaling": (True, True, True),
                "decoder_inject_noise": (False, False, False, False),
                "downsample_type": ("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
                "upsample_type": ("spatiotemporal", "spatiotemporal", "spatiotemporal"),
                "upsample_residual": (True, True, True),
                "upsample_factor": (2, 2, 2),
                "timestep_conditioning": timestep_conditioning,
                "patch_size": 4,
                "patch_size_t": 1,
                "resnet_norm_eps": 1e-6,
                "encoder_causal": True,
                "decoder_causal": False,
                "encoder_spatial_padding_mode": "zeros",
                "decoder_spatial_padding_mode": "reflect",
                "spatial_compression_ratio": 32,
                "temporal_compression_ratio": 8,
            },
        }
    elif version == "2.3":
        config = {
            "model_id": "Lightricks/LTX-2.3",
            "diffusers_config": {
                "in_channels": 3,
                "out_channels": 3,
                "latent_channels": 128,
                "block_out_channels": (256, 512, 1024, 1024),
                "down_block_types": (
                    "LTX2VideoDownBlock3D",
                    "LTX2VideoDownBlock3D",
                    "LTX2VideoDownBlock3D",
                    "LTX2VideoDownBlock3D",
                ),
                "decoder_block_out_channels": (256, 512, 512, 1024),
                "layers_per_block": (4, 6, 4, 2, 2),
                "decoder_layers_per_block": (4, 6, 4, 2, 2),
                "spatio_temporal_scaling": (True, True, True, True),
                "decoder_spatio_temporal_scaling": (True, True, True, True),
                "decoder_inject_noise": (False, False, False, False, False),
                "downsample_type": ("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
                "upsample_type": ("spatiotemporal", "spatiotemporal", "temporal", "spatial"),
                "upsample_residual": (False, False, False, False),
                "upsample_factor": (2, 2, 1, 2),
                "timestep_conditioning": timestep_conditioning,
                "patch_size": 4,
                "patch_size_t": 1,
                "resnet_norm_eps": 1e-6,
                "encoder_causal": True,
                "decoder_causal": False,
                "encoder_spatial_padding_mode": "zeros",
                "decoder_spatial_padding_mode": "zeros",
                "spatial_compression_ratio": 32,
                "temporal_compression_ratio": 8,
            },
        }
    elif version == "2.5":
        # Same block structure as 2.3 (32x32x8 compression); confirmed against the checkpoint's
        # config["vae"]["encoder_blocks"]/["decoder_blocks"] metadata, which is byte-identical to 2.3's.
        config = {
            "model_id": "Lightricks/LTX-2.5",
            "diffusers_config": {
                "in_channels": 3,
                "out_channels": 3,
                "latent_channels": 128,
                "block_out_channels": (256, 512, 1024, 1024),
                "down_block_types": (
                    "LTX2VideoDownBlock3D",
                    "LTX2VideoDownBlock3D",
                    "LTX2VideoDownBlock3D",
                    "LTX2VideoDownBlock3D",
                ),
                "decoder_block_out_channels": (256, 512, 512, 1024),
                "layers_per_block": (4, 6, 4, 2, 2),
                "decoder_layers_per_block": (4, 6, 4, 2, 2),
                "spatio_temporal_scaling": (True, True, True, True),
                "decoder_spatio_temporal_scaling": (True, True, True, True),
                "decoder_inject_noise": (False, False, False, False, False),
                "downsample_type": ("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
                "upsample_type": ("spatiotemporal", "spatiotemporal", "temporal", "spatial"),
                "upsample_residual": (False, False, False, False),
                "upsample_factor": (2, 2, 1, 2),
                "timestep_conditioning": timestep_conditioning,
                "patch_size": 4,
                "patch_size_t": 1,
                "resnet_norm_eps": 1e-6,
                "encoder_causal": True,
                "decoder_causal": False,
                "encoder_spatial_padding_mode": "zeros",
                "decoder_spatial_padding_mode": "zeros",
                "spatial_compression_ratio": 32,
                "temporal_compression_ratio": 8,
            },
        }
    return config


def get_ltx2_diffusion_video_vae_config(version: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if version != "2.5":
        raise ValueError(
            f"The diffusion decoder was introduced in LTX-2.5, which the converter handles under "
            f"`--version 2.5`; got version {version!r}."
        )
    # The encoder half is 2.5's conv VAE encoder, unchanged (its weights are byte identical to 2.3's), so
    # those entries must stay in sync with `get_ltx2_video_vae_config("2.5")`.
    config = {
        "model_id": "Lightricks/LTX-2.5",
        "diffusers_config": {
            "out_channels": 3,
            "latent_channels": 128,
            "patch_size": 4,
            "decoder_head_dim": 64,
            "decoder_stage_channels": (2048, 1024, 512, 512, 256),
            "decoder_stage_depths": (4, 6, 4, 2, 8),
            "decoder_stage_kernels": ((3, 7, 7), (3, 7, 7), (3, 5, 5), (3, 5, 5)),
            "decoder_upsample_strides": ((1, 2, 2), (2, 1, 1), (2, 2, 2), (2, 2, 2)),
            "decoder_upsample_channel_reductions": (2, 2, 1, 2),
            "decoder_stage5_kernel": (11, 11, 11),
            "decoder_t_emb_dim": 384,
            "decoder_timestep_scale_multiplier": 1000.0,
            "decoder_model_output_type": "x0",
            "decoder_num_inference_steps": 1,
            "spatial_compression_ratio": 32,
            "temporal_compression_ratio": 8,
        },
    }
    return config


def get_ltx2_audio_vae_config(version: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if version == "2.0":
        config = {
            "model_id": "Lightricks/LTX-2",
            "diffusers_config": {
                "base_channels": 128,
                "output_channels": 2,
                "ch_mult": (1, 2, 4),
                "num_res_blocks": 2,
                "attn_resolutions": None,
                "in_channels": 2,
                "resolution": 256,
                "latent_channels": 8,
                "norm_type": "pixel",
                "causality_axis": "height",
                "dropout": 0.0,
                "mid_block_add_attention": False,
                "sample_rate": 16000,
                "mel_hop_length": 160,
                "is_causal": True,
                "mel_bins": 64,
                "double_z": True,
            },
        }
    elif version == "2.3":
        config = {
            "model_id": "Lightricks/LTX-2.3",
            "diffusers_config": {
                "base_channels": 128,
                "output_channels": 2,
                "ch_mult": (1, 2, 4),
                "num_res_blocks": 2,
                "attn_resolutions": None,
                "in_channels": 2,
                "resolution": 256,
                "latent_channels": 8,
                "norm_type": "pixel",
                "causality_axis": "height",
                "dropout": 0.0,
                "mid_block_add_attention": False,
                "sample_rate": 16000,
                "mel_hop_length": 160,
                "is_causal": True,
                "mel_bins": 64,
                "double_z": True,
            },  # Same config as LTX-2.0
        }
    elif version == "2.5":
        config = {
            "model_id": "Lightricks/LTX-2.5",
            "diffusers_config": {
                "base_channels": 128,
                "output_channels": 2,
                "ch_mult": (1, 2, 4),
                "num_res_blocks": 2,
                "attn_resolutions": None,
                "in_channels": 2,
                "resolution": 256,
                "latent_channels": 8,
                "norm_type": "pixel",
                "causality_axis": "height",
                "dropout": 0.0,
                "mid_block_add_attention": False,
                "sample_rate": 16000,
                "mel_hop_length": 160,
                "is_causal": True,
                "mel_bins": 64,
                "double_z": True,
            },  # Same config as LTX-2.0 / 2.3
        }
    return config


def get_ltx2_vocoder_config(version: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if version == "2.0":
        config = {
            "model_id": "Lightricks/LTX-2",
            "diffusers_config": {
                "in_channels": 128,
                "hidden_channels": 1024,
                "out_channels": 2,
                "upsample_kernel_sizes": [16, 15, 8, 4, 4],
                "upsample_factors": [6, 5, 2, 2, 2],
                "resnet_kernel_sizes": [3, 7, 11],
                "resnet_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "act_fn": "leaky_relu",
                "leaky_relu_negative_slope": 0.1,
                "antialias": False,
                "final_act_fn": "tanh",
                "final_bias": True,
                "output_sampling_rate": 24000,
            },
        }
    elif version == "2.3":
        config = {
            "model_id": "Lightricks/LTX-2.3",
            "diffusers_config": {
                "in_channels": 128,
                "hidden_channels": 1536,
                "out_channels": 2,
                "upsample_kernel_sizes": [11, 4, 4, 4, 4, 4],
                "upsample_factors": [5, 2, 2, 2, 2, 2],
                "resnet_kernel_sizes": [3, 7, 11],
                "resnet_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "act_fn": "snakebeta",
                "leaky_relu_negative_slope": 0.1,
                "antialias": True,
                "antialias_ratio": 2,
                "antialias_kernel_size": 12,
                "final_act_fn": None,
                "final_bias": False,
                "bwe_in_channels": 128,
                "bwe_hidden_channels": 512,
                "bwe_out_channels": 2,
                "bwe_upsample_kernel_sizes": [12, 11, 4, 4, 4],
                "bwe_upsample_factors": [6, 5, 2, 2, 2],
                "bwe_resnet_kernel_sizes": [3, 7, 11],
                "bwe_resnet_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "bwe_act_fn": "snakebeta",
                "bwe_leaky_relu_negative_slope": 0.1,
                "bwe_antialias": True,
                "bwe_antialias_ratio": 2,
                "bwe_antialias_kernel_size": 12,
                "bwe_final_act_fn": None,
                "bwe_final_bias": False,
                "filter_length": 512,
                "hop_length": 80,
                "window_length": 512,
                "num_mel_channels": 64,
                "input_sampling_rate": 16000,
                "output_sampling_rate": 48000,
            },
        }
    elif version == "2.5":
        config = {
            "model_id": "Lightricks/LTX-2.5",
            "diffusers_config": {
                "in_channels": 128,
                "hidden_channels": 1536,
                "out_channels": 2,
                "upsample_kernel_sizes": [11, 4, 4, 4, 4, 4],
                "upsample_factors": [5, 2, 2, 2, 2, 2],
                "resnet_kernel_sizes": [3, 7, 11],
                "resnet_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "act_fn": "snakebeta",
                "leaky_relu_negative_slope": 0.1,
                "antialias": True,
                "antialias_ratio": 2,
                "antialias_kernel_size": 12,
                "final_act_fn": None,
                "final_bias": False,
                "bwe_in_channels": 128,
                "bwe_hidden_channels": 512,
                "bwe_out_channels": 2,
                "bwe_upsample_kernel_sizes": [12, 11, 4, 4, 4],
                "bwe_upsample_factors": [6, 5, 2, 2, 2],
                "bwe_resnet_kernel_sizes": [3, 7, 11],
                "bwe_resnet_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "bwe_act_fn": "snakebeta",
                "bwe_leaky_relu_negative_slope": 0.1,
                "bwe_antialias": True,
                "bwe_antialias_ratio": 2,
                "bwe_antialias_kernel_size": 12,
                "bwe_final_act_fn": None,
                "bwe_final_bias": False,
                "filter_length": 512,
                "hop_length": 80,
                "window_length": 512,
                "num_mel_channels": 64,
                "input_sampling_rate": 16000,
                "output_sampling_rate": 48000,
            },  # Same config as LTX-2.3
        }
    return config


def get_ltx2_spatial_latent_upsampler_config(version: str):
    if version == "2.0":
        config = {
            "in_channels": 128,
            "mid_channels": 1024,
            "num_blocks_per_stage": 4,
            "dims": 3,
            "spatial_upsample": True,
            "temporal_upsample": False,
            "rational_spatial_scale": 2.0,
            "use_rational_resampler": True,
        }
    elif version in ("2.3", "2.5"):
        config = {
            "in_channels": 128,
            "mid_channels": 1024,
            "num_blocks_per_stage": 4,
            "dims": 3,
            "spatial_upsample": True,
            "temporal_upsample": False,
            "rational_spatial_scale": 2.0,
            "use_rational_resampler": False,
        }
    else:
        raise ValueError(f"Unsupported version: {version}")
    return config


def get_ltx2_temporal_latent_upsampler_config(version: str):
    if version != "2.5":
        raise ValueError(f"Unsupported version: {version}")
    # The temporal x2 upsampler is narrower than its spatial sibling and pixel-shuffles along time only.
    return {
        "in_channels": 128,
        "mid_channels": 512,
        "num_blocks_per_stage": 4,
        "dims": 3,
        "spatial_upsample": False,
        "temporal_upsample": True,
    }


__all__ = [
    "get_ltx2_audio_vae_config",
    "get_ltx2_connectors_config",
    "get_ltx2_diffusion_video_vae_config",
    "get_ltx2_spatial_latent_upsampler_config",
    "get_ltx2_temporal_latent_upsampler_config",
    "get_ltx2_transformer_config",
    "get_ltx2_video_vae_config",
    "get_ltx2_vocoder_config",
]
