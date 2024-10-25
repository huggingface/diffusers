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
"""This is a standalone script that checks that we can hotswap a LoRA adapter on a compiles model

By itself, this script is not super interesting but when we collect the compile logs, we can check that hotswapping
does not trigger recompilation. This is done in the TestLoraHotSwapping class in test_pipelines.py.

Running this script with `check_hotswap(False)` will load the LoRA adapter without hotswapping, which will result in
recompilation.

"""

import os
import sys
import tempfile

import torch
from peft import LoraConfig, get_peft_model_state_dict
from peft.tuners.tuners_utils import BaseTunerLayer

from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.utils.testing_utils import floats_tensor


torch_device = "cuda" if torch.cuda.is_available() else "cpu"


def get_small_unet():
    # from UNet2DConditionModelTests
    torch.manual_seed(0)
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
    return model.to(torch_device)


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


def get_lora_state_dicts(modules_to_save):
    state_dicts = {}
    for module_name, module in modules_to_save.items():
        if module is not None:
            state_dicts[f"{module_name}_lora_layers"] = get_peft_model_state_dict(module)
    return state_dicts


def set_lora_device(model, adapter_names, device):
    # copied from LoraBaseMixin.set_lora_device
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            for adapter_name in adapter_names:
                module.lora_A[adapter_name].to(device)
                module.lora_B[adapter_name].to(device)
                # this is a param, not a module, so device placement is not in-place -> re-assign
                if hasattr(module, "lora_magnitude_vector") and module.lora_magnitude_vector is not None:
                    if adapter_name in module.lora_magnitude_vector:
                        module.lora_magnitude_vector[adapter_name] = module.lora_magnitude_vector[adapter_name].to(
                            device
                        )


def check_hotswap(do_hotswap):
    dummy_input = get_dummy_input()
    unet = get_small_unet()
    lora_config = get_unet_lora_config()
    unet.add_adapter(lora_config)
    torch.manual_seed(42)
    out_base = unet(**dummy_input)["sample"]
    # sanity check
    assert not (out_base == 0).all()

    with tempfile.TemporaryDirectory() as tmp_dirname:
        lora_state_dicts = get_lora_state_dicts({"unet": unet})
        StableDiffusionPipeline.save_lora_weights(
            save_directory=tmp_dirname, safe_serialization=True, **lora_state_dicts
        )
        del unet

        unet = get_small_unet()
        file_name = os.path.join(tmp_dirname, "pytorch_lora_weights.safetensors")
        unet.load_attn_procs(file_name)
        unet = torch.compile(unet, mode="reduce-overhead")

        torch.manual_seed(42)
        out0 = unet(**dummy_input)["sample"]

        # sanity check: still same result
        atol, rtol = 1e-5, 1e-5
        assert torch.allclose(out_base, out0, atol=atol, rtol=rtol)

        if do_hotswap:
            unet.load_attn_procs(file_name, adapter_name="default_0", hotswap=True)
        else:
            # offloading the old and loading the new adapter will result in recompilation
            set_lora_device(unet, adapter_names=["default_0"], device="cpu")
            unet.load_attn_procs(file_name, adapter_name="other_name", hotswap=False)

        torch.manual_seed(42)
        out1 = unet(**dummy_input)["sample"]

        # sanity check: since it's the same LoRA, the results should be identical
        assert torch.allclose(out0, out1, atol=atol, rtol=rtol)


if __name__ == "__main__":
    # check_hotswap(True) does not trigger recompilation
    # check_hotswap(False) triggers recompilation
    check_hotswap(do_hotswap=sys.argv[1] == "1")
