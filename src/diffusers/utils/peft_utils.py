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
        converted_state_dict[
            f"{name}.q_proj.lora_B.weight"
        ] = state_dict.pop(f"{name}.to_q_lora.up.weight")
        converted_state_dict[
            f"{name}.k_proj.lora_B.weight"
        ] = state_dict.pop(f"{name}.to_k_lora.up.weight")
        converted_state_dict[
            f"{name}.v_proj.lora_B.weight"
        ] = state_dict.pop(f"{name}.to_v_lora.up.weight")
        converted_state_dict[
            f"{name}.out_proj.lora_B.weight"
        ] = state_dict.pop(f"{name}.to_out_lora.up.weight")

        converted_state_dict[
            f"{name}.q_proj.lora_A.weight"
        ] = state_dict.pop(f"{name}.to_q_lora.down.weight")
        converted_state_dict[
            f"{name}.k_proj.lora_A.weight"
        ] = state_dict.pop(f"{name}.to_k_lora.down.weight")
        converted_state_dict[
            f"{name}.v_proj.lora_A.weight"
        ] = state_dict.pop(f"{name}.to_v_lora.down.weight")
        converted_state_dict[
            f"{name}.out_proj.lora_A.weight"
        ] = state_dict.pop(f"{name}.to_out_lora.down.weight")
    
    return converted_state_dict


def convert_peft_state_dict_to_diffusers(attention_modules, state_dict, adapter_name):
    # Convert from the new naming convention to the diffusers naming convention.
    converted_state_dict = {}
    
    for name, _ in attention_modules:
        converted_state_dict[
            f"{name}.q_proj.lora_linear_layer.up.weight"
        ] = state_dict.pop(f"{name}.q_proj.lora_B.{adapter_name}.weight")
        converted_state_dict[
            f"{name}.k_proj.lora_linear_layer.up.weight"
        ] = state_dict.pop(f"{name}.k_proj.lora_B.{adapter_name}.weight")
        converted_state_dict[
            f"{name}.v_proj.lora_linear_layer.up.weight"
        ] = state_dict.pop(f"{name}.v_proj.lora_B.{adapter_name}.weight")
        converted_state_dict[
            f"{name}.out_proj.lora_linear_layer.up.weight"
        ] = state_dict.pop(f"{name}.out_proj.lora_B.{adapter_name}.weight")

        converted_state_dict[
            f"{name}.q_proj.lora_linear_layer.down.weight"
        ] = state_dict.pop(f"{name}.q_proj.lora_A.{adapter_name}.weight")
        converted_state_dict[
            f"{name}.k_proj.lora_linear_layer.down.weight"
        ] = state_dict.pop(f"{name}.k_proj.lora_A.{adapter_name}.weight")
        converted_state_dict[
            f"{name}.v_proj.lora_linear_layer.down.weight"
        ] = state_dict.pop(f"{name}.v_proj.lora_A.{adapter_name}.weight")
        converted_state_dict[
            f"{name}.out_proj.lora_linear_layer.down.weight"
        ] = state_dict.pop(f"{name}.out_proj.lora_A.{adapter_name}.weight")
    
    return converted_state_dict


def convert_diffusers_state_dict_to_peft(attention_modules, state_dict):
    # Convert from the diffusers naming convention to the new naming convention.
    converted_state_dict = {}
    
    for name, _ in attention_modules:
        converted_state_dict[
            f"{name}.q_proj.lora_B.weight"
        ] = state_dict.pop(f"{name}.q_proj.lora_linear_layer.up.weight")
        converted_state_dict[
            f"{name}.k_proj.lora_B.weight"
        ] = state_dict.pop(f"{name}.k_proj.lora_linear_layer.up.weight")
        converted_state_dict[
            f"{name}.v_proj.lora_B.weight"
        ] = state_dict.pop(f"{name}.v_proj.lora_linear_layer.up.weight")
        converted_state_dict[
            f"{name}.out_proj.lora_B.weight"
        ] = state_dict.pop(f"{name}.out_proj.lora_linear_layer.up.weight")

        converted_state_dict[
            f"{name}.q_proj.lora_A.weight"
        ] = state_dict.pop(f"{name}.q_proj.lora_linear_layer.down.weight")
        converted_state_dict[
            f"{name}.k_proj.lora_A.weight"
        ] = state_dict.pop(f"{name}.k_proj.lora_linear_layer.down.weight")
        converted_state_dict[
            f"{name}.v_proj.lora_A.weight"
        ] = state_dict.pop(f"{name}.v_proj.lora_linear_layer.down.weight")
        converted_state_dict[
            f"{name}.out_proj.lora_A.weight"
        ] = state_dict.pop(f"{name}.out_proj.lora_linear_layer.down.weight")
    
    return converted_state_dict