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
State dict utilities: utility methods for converting state dicts easily
"""
import enum


class StateDictType(enum.Enum):
    """
    The mode to use when converting state dicts.
    """

    DIFFUSERS_OLD = "diffusers_old"
    # KOHYA_SS = "kohya_ss" # TODO: implement this
    PEFT = "peft"
    DIFFUSERS = "diffusers"


# We need to define a proper mapping for Unet since it uses different output keys than text encoder
# e.g. to_q_lora -> q_proj / to_q
UNET_TO_DIFFUSERS = {
    ".to_out_lora.up": ".to_out.0.lora_B",
    ".to_out_lora.down": ".to_out.0.lora_A",
    ".to_q_lora.down": ".to_q.lora_A",
    ".to_q_lora.up": ".to_q.lora_B",
    ".to_k_lora.down": ".to_k.lora_A",
    ".to_k_lora.up": ".to_k.lora_B",
    ".to_v_lora.down": ".to_v.lora_A",
    ".to_v_lora.up": ".to_v.lora_B",
    ".lora.up": ".lora_B",
    ".lora.down": ".lora_A",
}


DIFFUSERS_TO_PEFT = {
    ".q_proj.lora_linear_layer.up": ".q_proj.lora_B",
    ".q_proj.lora_linear_layer.down": ".q_proj.lora_A",
    ".k_proj.lora_linear_layer.up": ".k_proj.lora_B",
    ".k_proj.lora_linear_layer.down": ".k_proj.lora_A",
    ".v_proj.lora_linear_layer.up": ".v_proj.lora_B",
    ".v_proj.lora_linear_layer.down": ".v_proj.lora_A",
    ".out_proj.lora_linear_layer.up": ".out_proj.lora_B",
    ".out_proj.lora_linear_layer.down": ".out_proj.lora_A",
    ".lora_linear_layer.up": ".lora_B",
    ".lora_linear_layer.down": ".lora_A",
}

DIFFUSERS_OLD_TO_PEFT = {
    ".to_q_lora.up": ".q_proj.lora_B",
    ".to_q_lora.down": ".q_proj.lora_A",
    ".to_k_lora.up": ".k_proj.lora_B",
    ".to_k_lora.down": ".k_proj.lora_A",
    ".to_v_lora.up": ".v_proj.lora_B",
    ".to_v_lora.down": ".v_proj.lora_A",
    ".to_out_lora.up": ".out_proj.lora_B",
    ".to_out_lora.down": ".out_proj.lora_A",
    ".lora_linear_layer.up": ".lora_B",
    ".lora_linear_layer.down": ".lora_A",
}

PEFT_TO_DIFFUSERS = {
    ".q_proj.lora_B": ".q_proj.lora_linear_layer.up",
    ".q_proj.lora_A": ".q_proj.lora_linear_layer.down",
    ".k_proj.lora_B": ".k_proj.lora_linear_layer.up",
    ".k_proj.lora_A": ".k_proj.lora_linear_layer.down",
    ".v_proj.lora_B": ".v_proj.lora_linear_layer.up",
    ".v_proj.lora_A": ".v_proj.lora_linear_layer.down",
    ".out_proj.lora_B": ".out_proj.lora_linear_layer.up",
    ".out_proj.lora_A": ".out_proj.lora_linear_layer.down",
}

DIFFUSERS_OLD_TO_DIFFUSERS = {
    ".to_q_lora.up": ".q_proj.lora_linear_layer.up",
    ".to_q_lora.down": ".q_proj.lora_linear_layer.down",
    ".to_k_lora.up": ".k_proj.lora_linear_layer.up",
    ".to_k_lora.down": ".k_proj.lora_linear_layer.down",
    ".to_v_lora.up": ".v_proj.lora_linear_layer.up",
    ".to_v_lora.down": ".v_proj.lora_linear_layer.down",
    ".to_out_lora.up": ".out_proj.lora_linear_layer.up",
    ".to_out_lora.down": ".out_proj.lora_linear_layer.down",
}

PEFT_STATE_DICT_MAPPINGS = {
    StateDictType.DIFFUSERS_OLD: DIFFUSERS_OLD_TO_PEFT,
    StateDictType.DIFFUSERS: DIFFUSERS_TO_PEFT,
}

DIFFUSERS_STATE_DICT_MAPPINGS = {
    StateDictType.DIFFUSERS_OLD: DIFFUSERS_OLD_TO_DIFFUSERS,
    StateDictType.PEFT: PEFT_TO_DIFFUSERS,
}

KEYS_TO_ALWAYS_REPLACE = {
    ".processor.": ".",
}


def convert_state_dict(state_dict, mapping):
    r"""
    Simply iterates over the state dict and replaces the patterns in `mapping` with the corresponding values.

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            The state dict to convert.
        mapping (`dict[str, str]`):
            The mapping to use for conversion, the mapping should be a dictionary with the following structure:
                - key: the pattern to replace
                - value: the pattern to replace with

    Returns:
        converted_state_dict (`dict`)
            The converted state dict.
    """
    converted_state_dict = {}
    for k, v in state_dict.items():
        # First, filter out the keys that we always want to replace
        for pattern in KEYS_TO_ALWAYS_REPLACE.keys():
            if pattern in k:
                new_pattern = KEYS_TO_ALWAYS_REPLACE[pattern]
                k = k.replace(pattern, new_pattern)

        for pattern in mapping.keys():
            if pattern in k:
                new_pattern = mapping[pattern]
                k = k.replace(pattern, new_pattern)
                break
        converted_state_dict[k] = v
    return converted_state_dict


def convert_state_dict_to_peft(state_dict, original_type=None, **kwargs):
    r"""
    Converts a state dict to the PEFT format The state dict can be from previous diffusers format (`OLD_DIFFUSERS`), or
    new diffusers format (`DIFFUSERS`). The method only supports the conversion from diffusers old/new to PEFT for now.

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            The state dict to convert.
        original_type (`StateDictType`, *optional*):
            The original type of the state dict, if not provided, the method will try to infer it automatically.
    """
    if original_type is None:
        # Old diffusers to PEFT
        if any("to_out_lora" in k for k in state_dict.keys()):
            original_type = StateDictType.DIFFUSERS_OLD
        elif any("lora_linear_layer" in k for k in state_dict.keys()):
            original_type = StateDictType.DIFFUSERS
        else:
            raise ValueError("Could not automatically infer state dict type")

    if original_type not in PEFT_STATE_DICT_MAPPINGS.keys():
        raise ValueError(f"Original type {original_type} is not supported")

    mapping = PEFT_STATE_DICT_MAPPINGS[original_type]
    return convert_state_dict(state_dict, mapping)


def convert_state_dict_to_diffusers(state_dict, original_type=None, **kwargs):
    r"""
    Converts a state dict to new diffusers format. The state dict can be from previous diffusers format
    (`OLD_DIFFUSERS`), or PEFT format (`PEFT`) or new diffusers format (`DIFFUSERS`). In the last case the method will
    return the state dict as is.

    The method only supports the conversion from diffusers old, PEFT to diffusers new for now.

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            The state dict to convert.
        original_type (`StateDictType`, *optional*):
            The original type of the state dict, if not provided, the method will try to infer it automatically.
        kwargs (`dict`, *args*):
            Additional arguments to pass to the method.

            - **adapter_name**: For example, in case of PEFT, some keys will be pre-pended
                with the adapter name, therefore needs a special handling. By default PEFT also takes care of that in
                `get_peft_model_state_dict` method:
                https://github.com/huggingface/peft/blob/ba0477f2985b1ba311b83459d29895c809404e99/src/peft/utils/save_and_load.py#L92
                but we add it here in case we don't want to rely on that method.
    """
    peft_adapter_name = kwargs.pop("adapter_name", None)
    if peft_adapter_name is not None:
        peft_adapter_name = "." + peft_adapter_name
    else:
        peft_adapter_name = ""

    if original_type is None:
        # Old diffusers to PEFT
        if any("to_out_lora" in k for k in state_dict.keys()):
            original_type = StateDictType.DIFFUSERS_OLD
        elif any(f".lora_A{peft_adapter_name}.weight" in k for k in state_dict.keys()):
            original_type = StateDictType.PEFT
        elif any("lora_linear_layer" in k for k in state_dict.keys()):
            # nothing to do
            return state_dict
        else:
            raise ValueError("Could not automatically infer state dict type")

    if original_type not in DIFFUSERS_STATE_DICT_MAPPINGS.keys():
        raise ValueError(f"Original type {original_type} is not supported")

    mapping = DIFFUSERS_STATE_DICT_MAPPINGS[original_type]
    return convert_state_dict(state_dict, mapping)


def convert_unet_state_dict_to_peft(state_dict):
    r"""
    Converts a state dict from UNet format to diffusers format - i.e. by removing some keys
    """
    mapping = UNET_TO_DIFFUSERS
    return convert_state_dict(state_dict, mapping)
