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

from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_z_image_lora_to_diffusers


def _z_image_attention_only_state_dict(rank=2, dim=8, blocks=("layers.0", "context_refiner.0")):
    """musubi-tuner LoRA on Z-Image's original attention names: fused qkv + out, nothing else."""
    sd = {}
    for block in blocks:
        flat = block.replace(".", "_")
        sd[f"lora_unet_{flat}_attention_qkv.lora_down.weight"] = torch.randn(rank, dim)
        sd[f"lora_unet_{flat}_attention_qkv.lora_up.weight"] = torch.randn(3 * dim, rank)
        sd[f"lora_unet_{flat}_attention_qkv.alpha"] = torch.tensor(float(rank))
        sd[f"lora_unet_{flat}_attention_out.lora_down.weight"] = torch.randn(rank, dim)
        sd[f"lora_unet_{flat}_attention_out.lora_up.weight"] = torch.randn(dim, rank)
        sd[f"lora_unet_{flat}_attention_out.alpha"] = torch.tensor(float(rank))
    return sd


def test_z_image_attention_only_lora_splits_fused_qkv_and_maps_out():
    sd = _z_image_attention_only_state_dict()
    converted = _convert_non_diffusers_z_image_lora_to_diffusers(dict(sd))
    for block in ("layers.0", "context_refiner.0"):
        for proj in "qkv":
            assert converted[f"transformer.{block}.attention.to_{proj}.lora_A.weight"].shape == (2, 8)
            assert converted[f"transformer.{block}.attention.to_{proj}.lora_B.weight"].shape == (8, 2)
        assert converted[f"transformer.{block}.attention.to_out.0.lora_B.weight"].shape == (8, 2)
    # alpha == rank, so the k chunk of the fused up weight comes through unscaled
    up = sd["lora_unet_layers_0_attention_qkv.lora_up.weight"]
    torch.testing.assert_close(converted["transformer.layers.0.attention.to_k.lora_B.weight"], up[8:16])
    assert len(converted) == 2 * 4 * 2


def test_z_image_fused_qkv_is_skipped_next_to_split_keys():
    sd = _z_image_attention_only_state_dict(blocks=("layers.0",))
    for proj in "qkv":
        sd[f"lora_unet_layers_0_attention_to_{proj}.lora_down.weight"] = torch.randn(2, 8)
        sd[f"lora_unet_layers_0_attention_to_{proj}.lora_up.weight"] = torch.randn(8, 2)
    converted = _convert_non_diffusers_z_image_lora_to_diffusers(dict(sd))
    torch.testing.assert_close(
        converted["transformer.layers.0.attention.to_q.lora_B.weight"],
        sd["lora_unet_layers_0_attention_to_q.lora_up.weight"],
    )
