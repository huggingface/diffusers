# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    TripoSplatGaussianDecoder,
    TripoSplatTransformer3DModel,
)


def get_triposplat_dummy_components():
    torch.manual_seed(0)
    transformer = TripoSplatTransformer3DModel(
        q_token_length=16,
        model_channels=32,
        cond_channels=32,
        cond2_channels=8,
        num_refiner_blocks=1,
        num_blocks=1,
        num_heads=2,
        mlp_ratio=2.0,
    )
    torch.manual_seed(0)
    decoder = TripoSplatGaussianDecoder(
        model_channels=32, num_octree_blocks=1, num_gaussian_blocks=1, num_heads=2, mlp_ratio=2.0, max_voxel_level=2
    )
    torch.manual_seed(0)
    image_encoder = DINOv3ViTModel(
        DINOv3ViTConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_register_tokens=4,
            patch_size=16,
            use_gated_mlp=True,
            hidden_act="silu",
            layer_norm_eps=1e-5,
            pos_embed_rescale=None,
        )
    )
    torch.manual_seed(0)
    vae = AutoencoderKLFlux2(
        block_out_channels=(4, 8, 8, 8),
        layers_per_block=1,
        latent_channels=2,
        norm_num_groups=1,
        sample_size=32,
        batch_norm_eps=1e-5,
    )
    return {
        "transformer": transformer,
        "decoder": decoder,
        "image_encoder": image_encoder,
        "vae": vae,
        "scheduler": FlowMatchEulerDiscreteScheduler(shift=3.0),
        "background_remover": None,
        "canvas_size": 32,
    }
