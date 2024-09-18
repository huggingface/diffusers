# Copyright 2024 HuggingFace Inc.
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


import numpy as np
import torch
from peft import LoraConfig

from diffusers import UNet2DConditionModel
from diffusers.utils.testing_utils import floats_tensor


torch_device = "cuda" if torch.cuda.is_available() else "cpu"


def get_small_unet():
    # from  UNet2DConditionModelTests
    init_dict = {
        "block_out_channels": (4, 8),
        "norm_num_groups": 4,
        "down_block_types": ("CrossAttnDownBlock2D", "DownBlock2D"),
        "up_block_types": ("UpBlock2D", "CrossAttnUpBlock2D"),
        "cross_attention_dim": 8,
        "attention_head_dim": 2,
        "out_channels": 4,
        "in_channels": 4,
        "layers_per_block": 1,
        "sample_size": 16,
    }
    model = UNet2DConditionModel(**init_dict)
    return model


def get_unet_lora_config():
    # from test_models_unet_2d_condition.py
    rank = 4
    unet_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        init_lora_weights=False,
        use_dora=False,
    )
    return unet_lora_config


def get_dummy_input():
    # from UNet2DConditionModelTests
    batch_size = 4
    num_channels = 4
    sizes = (16, 16)

    noise = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)
    time_step = torch.tensor([10]).to(torch_device)
    encoder_hidden_states = floats_tensor((batch_size, 4, 8)).to(torch_device)

    return {"sample": noise, "timestep": time_step, "encoder_hidden_states": encoder_hidden_states}


def check_hotswap(hotswap):
    dummy_input = get_dummy_input()
    unet = get_small_unet()
    # lora_config = get_unet_lora_config()
    # unet.add_adapter(lora_config)
    unet.to(torch_device)

    # Note: When using the compile flag "reduce-overhead", there will be errors of the type
    # > input name: arg861_1. data pointer changed from 139647332027392 to 139647331054592
    unet = torch.compile(unet)

    torch.manual_seed(42)
    out0 = unet(**dummy_input)

    if hotswap:
        unet.load_lora_weights("ybelkada/sd-1.5-pokemon-lora-peft", adapter_name="foo", hotswap=hotswap)
    else:
        # offloading the old and loading the new adapter will result in recompilation
        unet.set_lora_device(adapter_names=["foo"], device="cpu")
        unet.load_lora_weights("ybelkada/sd-1.5-pokemon-lora-peft", adapter_name="bar")

    torch.manual_seed(42)
    out1 = unet(**dummy_input)

    # sanity check: since it's the same LoRA, the results should be identical
    out0, out1 = np.array(out0.images[0]), np.array(out1.images[0])
    assert not (out0 == 0).all()
    assert (out0 == out1).all()


if __name__ == "__main__":
    # check_hotswap(False) will trigger recompilation
    check_hotswap(True)
