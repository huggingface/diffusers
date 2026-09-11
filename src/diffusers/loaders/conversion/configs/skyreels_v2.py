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

"""Original configuration helpers and model presets for the skyreels_v2 assembly recipe."""

from typing import Any


def get_transformer_config(model_type: str) -> dict[str, Any]:
    if model_type == "SkyReels-V2-DF-1.3B-540P":
        config = {
            "model_id": "Skywork/SkyReels-V2-DF-1.3B-540P",
            "diffusers_config": {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "in_channels": 16,
                "num_attention_heads": 12,
                "inject_sample_info": True,
                "num_layers": 30,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            },
        }
    elif model_type == "SkyReels-V2-DF-14B-720P":
        config = {
            "model_id": "Skywork/SkyReels-V2-DF-14B-720P",
            "diffusers_config": {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 16,
                "num_attention_heads": 40,
                "inject_sample_info": False,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            },
        }
    elif model_type == "SkyReels-V2-DF-14B-540P":
        config = {
            "model_id": "Skywork/SkyReels-V2-DF-14B-540P",
            "diffusers_config": {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 16,
                "num_attention_heads": 40,
                "inject_sample_info": False,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            },
        }
    elif model_type == "SkyReels-V2-T2V-14B-720P":
        config = {
            "model_id": "Skywork/SkyReels-V2-T2V-14B-720P",
            "diffusers_config": {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 16,
                "num_attention_heads": 40,
                "inject_sample_info": False,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            },
        }
    elif model_type == "SkyReels-V2-T2V-14B-540P":
        config = {
            "model_id": "Skywork/SkyReels-V2-T2V-14B-540P",
            "diffusers_config": {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 16,
                "num_attention_heads": 40,
                "inject_sample_info": False,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            },
        }
    elif model_type == "SkyReels-V2-I2V-1.3B-540P":
        config = {
            "model_id": "Skywork/SkyReels-V2-I2V-1.3B-540P",
            "diffusers_config": {
                "added_kv_proj_dim": 1536,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "in_channels": 36,
                "num_attention_heads": 12,
                "inject_sample_info": False,
                "num_layers": 30,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
                "image_dim": 1280,
            },
        }
    elif model_type == "SkyReels-V2-I2V-14B-540P":
        config = {
            "model_id": "Skywork/SkyReels-V2-I2V-14B-540P",
            "diffusers_config": {
                "added_kv_proj_dim": 5120,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 36,
                "num_attention_heads": 40,
                "inject_sample_info": False,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
                "image_dim": 1280,
            },
        }
    elif model_type == "SkyReels-V2-I2V-14B-720P":
        config = {
            "model_id": "Skywork/SkyReels-V2-I2V-14B-720P",
            "diffusers_config": {
                "added_kv_proj_dim": 5120,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 36,
                "num_attention_heads": 40,
                "inject_sample_info": False,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
                "image_dim": 1280,
            },
        }
    elif model_type == "SkyReels-V2-FLF2V-1.3B-540P":
        config = {
            "model_id": "Skywork/SkyReels-V2-I2V-1.3B-540P",
            "diffusers_config": {
                "added_kv_proj_dim": 1536,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "in_channels": 36,
                "num_attention_heads": 12,
                "inject_sample_info": False,
                "num_layers": 30,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
                "image_dim": 1280,
                "pos_embed_seq_len": 514,
            },
        }
    elif model_type == "SkyReels-V2-FLF2V-14B-540P":
        config = {
            "model_id": "Skywork/SkyReels-V2-I2V-14B-540P",
            "diffusers_config": {
                "added_kv_proj_dim": 5120,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 36,
                "num_attention_heads": 40,
                "inject_sample_info": False,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
                "image_dim": 1280,
                "pos_embed_seq_len": 514,
            },
        }
    elif model_type == "SkyReels-V2-FLF2V-14B-720P":
        config = {
            "model_id": "Skywork/SkyReels-V2-I2V-14B-720P",
            "diffusers_config": {
                "added_kv_proj_dim": 5120,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 36,
                "num_attention_heads": 40,
                "inject_sample_info": False,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
                "image_dim": 1280,
                "pos_embed_seq_len": 514,
            },
        }
    return config


__all__ = ["get_transformer_config"]
