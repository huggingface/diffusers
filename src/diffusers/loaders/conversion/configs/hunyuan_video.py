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

"""Original configuration helpers and model presets for the hunyuan_video assembly recipe."""

TRANSFORMER_CONFIGS = {
    "HYVideo-T/2-cfgdistill": {
        "in_channels": 16,
        "out_channels": 16,
        "num_attention_heads": 24,
        "attention_head_dim": 128,
        "num_layers": 20,
        "num_single_layers": 40,
        "num_refiner_layers": 2,
        "mlp_ratio": 4.0,
        "patch_size": 2,
        "patch_size_t": 1,
        "qk_norm": "rms_norm",
        "guidance_embeds": True,
        "text_embed_dim": 4096,
        "pooled_projection_dim": 768,
        "rope_theta": 256.0,
        "rope_axes_dim": (16, 56, 56),
        "image_condition_type": None,
    },
    "HYVideo-T/2-I2V-33ch": {
        "in_channels": 16 * 2 + 1,
        "out_channels": 16,
        "num_attention_heads": 24,
        "attention_head_dim": 128,
        "num_layers": 20,
        "num_single_layers": 40,
        "num_refiner_layers": 2,
        "mlp_ratio": 4.0,
        "patch_size": 2,
        "patch_size_t": 1,
        "qk_norm": "rms_norm",
        "guidance_embeds": False,
        "text_embed_dim": 4096,
        "pooled_projection_dim": 768,
        "rope_theta": 256.0,
        "rope_axes_dim": (16, 56, 56),
        "image_condition_type": "latent_concat",
    },
    "HYVideo-T/2-I2V-16ch": {
        "in_channels": 16,
        "out_channels": 16,
        "num_attention_heads": 24,
        "attention_head_dim": 128,
        "num_layers": 20,
        "num_single_layers": 40,
        "num_refiner_layers": 2,
        "mlp_ratio": 4.0,
        "patch_size": 2,
        "patch_size_t": 1,
        "qk_norm": "rms_norm",
        "guidance_embeds": True,
        "text_embed_dim": 4096,
        "pooled_projection_dim": 768,
        "rope_theta": 256.0,
        "rope_axes_dim": (16, 56, 56),
        "image_condition_type": "token_replace",
    },
}

__all__ = ["TRANSFORMER_CONFIGS"]
