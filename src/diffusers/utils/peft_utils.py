# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"""
PEFT utilities: Utilities related to peft library
"""
import torch


def recurse_replace_peft_layers(model):
    r"""
    Recursively replace all instances of `LoraLayer` with corresponding new layers in `model`.
    """
    from peft.tuners.lora import LoraLayer

    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            recurse_replace_peft_layers(module)
            
        if isinstance(module, LoraLayer) and isinstance(module, torch.nn.Linear):
            new_module = torch.nn.Linear(module.in_features, module.out_features, bias=module.bias is not None).to(module.weight.device)
            new_module.weight = module.weight
            if module.bias is not None:
                new_module.bias = module.bias

            setattr(model, name, new_module)
            del module

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # TODO: do it for Conv2d
    
    return model

def convert_old_state_dict_to_peft(attention_modules, state_dict):
    # Convert from the old naming convention to the new naming convention.
    #
    # Previously, the old LoRA layers were stored on the state dict at the
    # same level as the attention block i.e.
    # `text_model.encoder.layers.11.self_attn.to_out_lora.lora_A.weight`.
    #
    # This is no actual module at that point, they were monkey patched on to the
    # existing module. We want to be able to load them via their actual state dict.
    # They're in `PatchedLoraProjection.lora_linear_layer` now.
    converted_state_dict = {}

    for name, _ in attention_modules:
        converted_state_dict[f"{name}.q_proj.lora_B.weight"] = state_dict.pop(f"{name}.to_q_lora.up.weight")
        converted_state_dict[f"{name}.k_proj.lora_B.weight"] = state_dict.pop(f"{name}.to_k_lora.up.weight")
        converted_state_dict[f"{name}.v_proj.lora_B.weight"] = state_dict.pop(f"{name}.to_v_lora.up.weight")
        converted_state_dict[f"{name}.out_proj.lora_B.weight"] = state_dict.pop(f"{name}.to_out_lora.up.weight")

        converted_state_dict[f"{name}.q_proj.lora_A.weight"] = state_dict.pop(f"{name}.to_q_lora.down.weight")
        converted_state_dict[f"{name}.k_proj.lora_A.weight"] = state_dict.pop(f"{name}.to_k_lora.down.weight")
        converted_state_dict[f"{name}.v_proj.lora_A.weight"] = state_dict.pop(f"{name}.to_v_lora.down.weight")
        converted_state_dict[f"{name}.out_proj.lora_A.weight"] = state_dict.pop(f"{name}.to_out_lora.down.weight")

    return converted_state_dict


def convert_peft_state_dict_to_diffusers(attention_modules, state_dict, adapter_name):
    # Convert from the new naming convention to the diffusers naming convention.
    converted_state_dict = {}

    for name, _ in attention_modules:
        converted_state_dict[f"{name}.q_proj.lora_linear_layer.up.weight"] = state_dict.pop(
            f"{name}.q_proj.lora_B.{adapter_name}.weight"
        )
        converted_state_dict[f"{name}.k_proj.lora_linear_layer.up.weight"] = state_dict.pop(
            f"{name}.k_proj.lora_B.{adapter_name}.weight"
        )
        converted_state_dict[f"{name}.v_proj.lora_linear_layer.up.weight"] = state_dict.pop(
            f"{name}.v_proj.lora_B.{adapter_name}.weight"
        )
        converted_state_dict[f"{name}.out_proj.lora_linear_layer.up.weight"] = state_dict.pop(
            f"{name}.out_proj.lora_B.{adapter_name}.weight"
        )

        converted_state_dict[f"{name}.q_proj.lora_linear_layer.down.weight"] = state_dict.pop(
            f"{name}.q_proj.lora_A.{adapter_name}.weight"
        )
        converted_state_dict[f"{name}.k_proj.lora_linear_layer.down.weight"] = state_dict.pop(
            f"{name}.k_proj.lora_A.{adapter_name}.weight"
        )
        converted_state_dict[f"{name}.v_proj.lora_linear_layer.down.weight"] = state_dict.pop(
            f"{name}.v_proj.lora_A.{adapter_name}.weight"
        )
        converted_state_dict[f"{name}.out_proj.lora_linear_layer.down.weight"] = state_dict.pop(
            f"{name}.out_proj.lora_A.{adapter_name}.weight"
        )

    return converted_state_dict


def convert_diffusers_state_dict_to_peft(attention_modules, state_dict):
    # Convert from the diffusers naming convention to the new naming convention.
    converted_state_dict = {}

    for name, _ in attention_modules:
        converted_state_dict[f"{name}.q_proj.lora_B.weight"] = state_dict.pop(
            f"{name}.q_proj.lora_linear_layer.up.weight"
        )
        converted_state_dict[f"{name}.k_proj.lora_B.weight"] = state_dict.pop(
            f"{name}.k_proj.lora_linear_layer.up.weight"
        )
        converted_state_dict[f"{name}.v_proj.lora_B.weight"] = state_dict.pop(
            f"{name}.v_proj.lora_linear_layer.up.weight"
        )
        converted_state_dict[f"{name}.out_proj.lora_B.weight"] = state_dict.pop(
            f"{name}.out_proj.lora_linear_layer.up.weight"
        )

        converted_state_dict[f"{name}.q_proj.lora_A.weight"] = state_dict.pop(
            f"{name}.q_proj.lora_linear_layer.down.weight"
        )
        converted_state_dict[f"{name}.k_proj.lora_A.weight"] = state_dict.pop(
            f"{name}.k_proj.lora_linear_layer.down.weight"
        )
        converted_state_dict[f"{name}.v_proj.lora_A.weight"] = state_dict.pop(
            f"{name}.v_proj.lora_linear_layer.down.weight"
        )
        converted_state_dict[f"{name}.out_proj.lora_A.weight"] = state_dict.pop(
            f"{name}.out_proj.lora_linear_layer.down.weight"
        )

    return converted_state_dict


def convert_unet_state_dict_to_peft(state_dict):
    converted_state_dict = {}

    patterns = {
        ".to_out_lora": ".to_o",
        ".down": ".lora_A",
        ".up": ".lora_B",
        ".to_q_lora": ".to_q",
        ".to_k_lora": ".to_k",
        ".to_v_lora": ".to_v",
    }

    for k, v in state_dict.items():
        if any(pattern in k for pattern in patterns.keys()):
            for old, new in patterns.items():
                k = k.replace(old, new)

        converted_state_dict[k] = v

    return converted_state_dict
