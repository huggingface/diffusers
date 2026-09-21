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

"""Original configuration helpers and model presets for the flux2 assembly recipe."""

from typing import Any, Dict


def get_flux2_transformer_config(model_type: str) -> Dict[str, Any]:
    if model_type == "flux2-dev":
        config = {
            "model_id": "black-forest-labs/FLUX.2-dev",
            "diffusers_config": {
                "patch_size": 1,
                "in_channels": 128,
                "num_layers": 8,
                "num_single_layers": 48,
                "attention_head_dim": 128,
                "num_attention_heads": 48,
                "joint_attention_dim": 15360,
                "timestep_guidance_channels": 256,
                "mlp_ratio": 3.0,
                "axes_dims_rope": (32, 32, 32, 32),
                "rope_theta": 2000,
                "eps": 1e-6,
            },
        }
    elif model_type == "klein-4b":
        config = {
            "model_id": "diffusers-internal-dev/dummy0115",
            "diffusers_config": {
                "patch_size": 1,
                "in_channels": 128,
                "num_layers": 5,
                "num_single_layers": 20,
                "attention_head_dim": 128,
                "num_attention_heads": 24,
                "joint_attention_dim": 7680,
                "timestep_guidance_channels": 256,
                "mlp_ratio": 3.0,
                "axes_dims_rope": (32, 32, 32, 32),
                "rope_theta": 2000,
                "eps": 1e-6,
                "guidance_embeds": False,
            },
        }

    elif model_type == "klein-9b":
        config = {
            "model_id": "diffusers-internal-dev/dummy0115",
            "diffusers_config": {
                "patch_size": 1,
                "in_channels": 128,
                "num_layers": 8,
                "num_single_layers": 24,
                "attention_head_dim": 128,
                "num_attention_heads": 32,
                "joint_attention_dim": 12288,
                "timestep_guidance_channels": 256,
                "mlp_ratio": 3.0,
                "axes_dims_rope": (32, 32, 32, 32),
                "rope_theta": 2000,
                "eps": 1e-6,
                "guidance_embeds": False,
            },
        }

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from: flux2-dev, klein-4b, klein-9b")

    return config


__all__ = ["get_flux2_transformer_config"]
