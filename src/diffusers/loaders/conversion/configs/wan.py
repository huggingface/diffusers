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

"""Original configuration helpers and model presets for the wan assembly recipe."""

from typing import Any, Dict


def get_transformer_config(model_type: str) -> Dict[str, Any]:
    if model_type == "Wan-T2V-1.3B":
        config = {
            "model_id": "StevenZhang/Wan2.1-T2V-1.3B-Diff",
            "diffusers_config": {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "in_channels": 16,
                "num_attention_heads": 12,
                "num_layers": 30,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            },
        }
    elif model_type == "Wan-T2V-14B":
        config = {
            "model_id": "StevenZhang/Wan2.1-T2V-14B-Diff",
            "diffusers_config": {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 16,
                "num_attention_heads": 40,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            },
        }
    elif model_type == "Wan-I2V-14B-480p":
        config = {
            "model_id": "StevenZhang/Wan2.1-I2V-14B-480P-Diff",
            "diffusers_config": {
                "image_dim": 1280,
                "added_kv_proj_dim": 5120,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 36,
                "num_attention_heads": 40,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            },
        }
    elif model_type == "Wan-I2V-14B-720p":
        config = {
            "model_id": "StevenZhang/Wan2.1-I2V-14B-720P-Diff",
            "diffusers_config": {
                "image_dim": 1280,
                "added_kv_proj_dim": 5120,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 36,
                "num_attention_heads": 40,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            },
        }
    elif model_type == "Wan-FLF2V-14B-720P":
        config = {
            "model_id": "ypyp/Wan2.1-FLF2V-14B-720P",  # This is just a placeholder
            "diffusers_config": {
                "image_dim": 1280,
                "added_kv_proj_dim": 5120,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 36,
                "num_attention_heads": 40,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
                "rope_max_seq_len": 1024,
                "pos_embed_seq_len": 257 * 2,
            },
        }
    elif model_type == "Wan-VACE-1.3B":
        config = {
            "model_id": "Wan-AI/Wan2.1-VACE-1.3B",
            "diffusers_config": {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "in_channels": 16,
                "num_attention_heads": 12,
                "num_layers": 30,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
                "vace_layers": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
                "vace_in_channels": 96,
            },
        }
    elif model_type == "Wan-VACE-14B":
        config = {
            "model_id": "Wan-AI/Wan2.1-VACE-14B",
            "diffusers_config": {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 16,
                "num_attention_heads": 40,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
                "vace_layers": [0, 5, 10, 15, 20, 25, 30, 35],
                "vace_in_channels": 96,
            },
        }
    elif model_type == "Wan2.2-VACE-Fun-14B":
        config = {
            "model_id": "alibaba-pai/Wan2.2-VACE-Fun-A14B",
            "diffusers_config": {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 16,
                "num_attention_heads": 40,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
                "vace_layers": [0, 5, 10, 15, 20, 25, 30, 35],
                "vace_in_channels": 96,
            },
        }
    elif model_type == "Wan2.2-I2V-14B-720p":
        config = {
            "model_id": "Wan-AI/Wan2.2-I2V-A14B",
            "diffusers_config": {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 36,
                "num_attention_heads": 40,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            },
        }
    elif model_type == "Wan2.2-T2V-A14B":
        config = {
            "model_id": "Wan-AI/Wan2.2-T2V-A14B",
            "diffusers_config": {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 16,
                "num_attention_heads": 40,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            },
        }
    elif model_type == "Wan2.2-TI2V-5B":
        config = {
            "model_id": "Wan-AI/Wan2.2-TI2V-5B",
            "diffusers_config": {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 14336,
                "freq_dim": 256,
                "in_channels": 48,
                "num_attention_heads": 24,
                "num_layers": 30,
                "out_channels": 48,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            },
        }
    elif model_type == "Wan2.2-Animate-14B":
        config = {
            "model_id": "Wan-AI/Wan2.2-Animate-14B",
            "diffusers_config": {
                "image_dim": 1280,
                "added_kv_proj_dim": 5120,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 36,
                "num_attention_heads": 40,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": (1, 2, 2),
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
                "rope_max_seq_len": 1024,
                "pos_embed_seq_len": None,
                "motion_encoder_size": 512,  # Start of Wan Animate-specific configs
                "motion_style_dim": 512,
                "motion_dim": 20,
                "motion_encoder_dim": 512,
                "face_encoder_hidden_dim": 1024,
                "face_encoder_num_heads": 4,
                "inject_face_latents_blocks": 5,
            },
        }
    return config


__all__ = ["get_transformer_config"]
