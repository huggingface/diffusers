# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import json

from .import_utils import is_torch_available
from .logging import get_logger


if is_torch_available():
    import torch


logger = get_logger(__name__)


class StateDictType(enum.Enum):
    """
    The mode to use when converting state dicts.
    """

    DIFFUSERS_OLD = "diffusers_old"
    KOHYA_SS = "kohya_ss"
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
    ".to_out.lora_magnitude_vector": ".to_out.0.lora_magnitude_vector",
}

CONTROL_LORA_TO_DIFFUSERS = {
    ".to_q.down": ".to_q.lora_A.weight",
    ".to_q.up": ".to_q.lora_B.weight",
    ".to_k.down": ".to_k.lora_A.weight",
    ".to_k.up": ".to_k.lora_B.weight",
    ".to_v.down": ".to_v.lora_A.weight",
    ".to_v.up": ".to_v.lora_B.weight",
    ".to_out.0.down": ".to_out.0.lora_A.weight",
    ".to_out.0.up": ".to_out.0.lora_B.weight",
    ".ff.net.0.proj.down": ".ff.net.0.proj.lora_A.weight",
    ".ff.net.0.proj.up": ".ff.net.0.proj.lora_B.weight",
    ".ff.net.2.down": ".ff.net.2.lora_A.weight",
    ".ff.net.2.up": ".ff.net.2.lora_B.weight",
    ".proj_in.down": ".proj_in.lora_A.weight",
    ".proj_in.up": ".proj_in.lora_B.weight",
    ".proj_out.down": ".proj_out.lora_A.weight",
    ".proj_out.up": ".proj_out.lora_B.weight",
    ".conv.down": ".conv.lora_A.weight",
    ".conv.up": ".conv.lora_B.weight",
    **{f".conv{i}.down": f".conv{i}.lora_A.weight" for i in range(1, 3)},
    **{f".conv{i}.up": f".conv{i}.lora_B.weight" for i in range(1, 3)},
    "conv_in.down": "conv_in.lora_A.weight",
    "conv_in.up": "conv_in.lora_B.weight",
    ".conv_shortcut.down": ".conv_shortcut.lora_A.weight",
    ".conv_shortcut.up": ".conv_shortcut.lora_B.weight",
    **{f".linear_{i}.down": f".linear_{i}.lora_A.weight" for i in range(1, 3)},
    **{f".linear_{i}.up": f".linear_{i}.lora_B.weight" for i in range(1, 3)},
    "time_emb_proj.down": "time_emb_proj.lora_A.weight",
    "time_emb_proj.up": "time_emb_proj.lora_B.weight",
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
    "text_projection.lora.down.weight": "text_projection.lora_A.weight",
    "text_projection.lora.up.weight": "text_projection.lora_B.weight",
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
    "to_k.lora_A": "to_k.lora.down",
    "to_k.lora_B": "to_k.lora.up",
    "to_q.lora_A": "to_q.lora.down",
    "to_q.lora_B": "to_q.lora.up",
    "to_v.lora_A": "to_v.lora.down",
    "to_v.lora_B": "to_v.lora.up",
    "to_out.0.lora_A": "to_out.0.lora.down",
    "to_out.0.lora_B": "to_out.0.lora.up",
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
    ".to_k.lora_magnitude_vector": ".k_proj.lora_magnitude_vector",
    ".to_v.lora_magnitude_vector": ".v_proj.lora_magnitude_vector",
    ".to_q.lora_magnitude_vector": ".q_proj.lora_magnitude_vector",
    ".to_out.lora_magnitude_vector": ".out_proj.lora_magnitude_vector",
}

PEFT_TO_KOHYA_SS = {
    "lora_A": "lora_down",
    "lora_B": "lora_up",
    # This is not a comprehensive dict as kohya format requires replacing `.` with `_` in keys,
    # adding prefixes and adding alpha values
    # Check `convert_state_dict_to_kohya` for more
}

PEFT_STATE_DICT_MAPPINGS = {
    StateDictType.DIFFUSERS_OLD: DIFFUSERS_OLD_TO_PEFT,
    StateDictType.DIFFUSERS: DIFFUSERS_TO_PEFT,
}

DIFFUSERS_STATE_DICT_MAPPINGS = {
    StateDictType.DIFFUSERS_OLD: DIFFUSERS_OLD_TO_DIFFUSERS,
    StateDictType.PEFT: PEFT_TO_DIFFUSERS,
}

KOHYA_STATE_DICT_MAPPINGS = {StateDictType.PEFT: PEFT_TO_KOHYA_SS}

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

            - **adapter_name**: For example, in case of PEFT, some keys will be prepended
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


def convert_sai_sd_control_lora_state_dict_to_peft(state_dict):
    def _convert_controlnet_to_diffusers(state_dict):
        is_sdxl = "input_blocks.11.0.in_layers.0.weight" not in state_dict
        logger.info(f"Using ControlNet lora ({'SDXL' if is_sdxl else 'SD15'})")

        # Retrieves the keys for the input blocks only
        num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in state_dict if "input_blocks" in layer})
        input_blocks = {
            layer_id: [key for key in state_dict if f"input_blocks.{layer_id}" in key]
            for layer_id in range(num_input_blocks)
        }
        layers_per_block = 2

        # op blocks
        op_blocks = [key for key in state_dict if "0.op" in key]

        converted_state_dict = {}
        # Conv in layers
        for key in input_blocks[0]:
            diffusers_key = key.replace("input_blocks.0.0", "conv_in")
            converted_state_dict[diffusers_key] = state_dict.get(key)

        # controlnet time embedding blocks
        time_embedding_blocks = [key for key in state_dict if "time_embed" in key]
        for key in time_embedding_blocks:
            diffusers_key = key.replace("time_embed.0", "time_embedding.linear_1").replace(
                "time_embed.2", "time_embedding.linear_2"
            )
            converted_state_dict[diffusers_key] = state_dict.get(key)

        # controlnet label embedding blocks
        label_embedding_blocks = [key for key in state_dict if "label_emb" in key]
        for key in label_embedding_blocks:
            diffusers_key = key.replace("label_emb.0.0", "add_embedding.linear_1").replace(
                "label_emb.0.2", "add_embedding.linear_2"
            )
            converted_state_dict[diffusers_key] = state_dict.get(key)

        # Down blocks
        for i in range(1, num_input_blocks):
            block_id = (i - 1) // (layers_per_block + 1)
            layer_in_block_id = (i - 1) % (layers_per_block + 1)

            resnets = [
                key for key in input_blocks[i] if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
            ]
            for key in resnets:
                diffusers_key = (
                    key.replace("in_layers.0", "norm1")
                    .replace("in_layers.2", "conv1")
                    .replace("out_layers.0", "norm2")
                    .replace("out_layers.3", "conv2")
                    .replace("emb_layers.1", "time_emb_proj")
                    .replace("skip_connection", "conv_shortcut")
                )
                diffusers_key = diffusers_key.replace(
                    f"input_blocks.{i}.0", f"down_blocks.{block_id}.resnets.{layer_in_block_id}"
                )
                converted_state_dict[diffusers_key] = state_dict.get(key)

            if f"input_blocks.{i}.0.op.bias" in state_dict:
                for key in [key for key in op_blocks if f"input_blocks.{i}.0.op" in key]:
                    diffusers_key = key.replace(
                        f"input_blocks.{i}.0.op", f"down_blocks.{block_id}.downsamplers.0.conv"
                    )
                    converted_state_dict[diffusers_key] = state_dict.get(key)

            attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]
            if attentions:
                for key in attentions:
                    diffusers_key = key.replace(
                        f"input_blocks.{i}.1", f"down_blocks.{block_id}.attentions.{layer_in_block_id}"
                    )
                    converted_state_dict[diffusers_key] = state_dict.get(key)

        # controlnet down blocks
        for i in range(num_input_blocks):
            converted_state_dict[f"controlnet_down_blocks.{i}.weight"] = state_dict.get(f"zero_convs.{i}.0.weight")
            converted_state_dict[f"controlnet_down_blocks.{i}.bias"] = state_dict.get(f"zero_convs.{i}.0.bias")

        # Retrieves the keys for the middle blocks only
        num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in state_dict if "middle_block" in layer})
        middle_blocks = {
            layer_id: [key for key in state_dict if f"middle_block.{layer_id}" in key]
            for layer_id in range(num_middle_blocks)
        }

        # Mid blocks
        for key in middle_blocks.keys():
            diffusers_key = max(key - 1, 0)
            if key % 2 == 0:
                for k in middle_blocks[key]:
                    diffusers_key_hf = (
                        k.replace("in_layers.0", "norm1")
                        .replace("in_layers.2", "conv1")
                        .replace("out_layers.0", "norm2")
                        .replace("out_layers.3", "conv2")
                        .replace("emb_layers.1", "time_emb_proj")
                        .replace("skip_connection", "conv_shortcut")
                    )
                    diffusers_key_hf = diffusers_key_hf.replace(
                        f"middle_block.{key}", f"mid_block.resnets.{diffusers_key}"
                    )
                    converted_state_dict[diffusers_key_hf] = state_dict.get(k)
            else:
                for k in middle_blocks[key]:
                    diffusers_key_hf = k.replace(f"middle_block.{key}", f"mid_block.attentions.{diffusers_key}")
                    converted_state_dict[diffusers_key_hf] = state_dict.get(k)

        # mid block
        converted_state_dict["controlnet_mid_block.weight"] = state_dict.get("middle_block_out.0.weight")
        converted_state_dict["controlnet_mid_block.bias"] = state_dict.get("middle_block_out.0.bias")

        # controlnet cond embedding blocks
        cond_embedding_blocks = {
            ".".join(layer.split(".")[:2])
            for layer in state_dict
            if "input_hint_block" in layer
            and ("input_hint_block.0" not in layer)
            and ("input_hint_block.14" not in layer)
        }
        num_cond_embedding_blocks = len(cond_embedding_blocks)

        for idx in range(1, num_cond_embedding_blocks + 1):
            diffusers_idx = idx - 1
            cond_block_id = 2 * idx

            converted_state_dict[f"controlnet_cond_embedding.blocks.{diffusers_idx}.weight"] = state_dict.get(
                f"input_hint_block.{cond_block_id}.weight"
            )
            converted_state_dict[f"controlnet_cond_embedding.blocks.{diffusers_idx}.bias"] = state_dict.get(
                f"input_hint_block.{cond_block_id}.bias"
            )

        for key in [key for key in state_dict if "input_hint_block.0" in key]:
            diffusers_key = key.replace("input_hint_block.0", "controlnet_cond_embedding.conv_in")
            converted_state_dict[diffusers_key] = state_dict.get(key)

        for key in [key for key in state_dict if "input_hint_block.14" in key]:
            diffusers_key = key.replace("input_hint_block.14", "controlnet_cond_embedding.conv_out")
            converted_state_dict[diffusers_key] = state_dict.get(key)

        return converted_state_dict

    state_dict = _convert_controlnet_to_diffusers(state_dict)
    mapping = CONTROL_LORA_TO_DIFFUSERS
    return convert_state_dict(state_dict, mapping)


def convert_all_state_dict_to_peft(state_dict):
    r"""
    Attempts to first `convert_state_dict_to_peft`, and if it doesn't detect `lora_linear_layer` for a valid
    `DIFFUSERS` LoRA for example, attempts to exclusively convert the Unet `convert_unet_state_dict_to_peft`
    """
    try:
        peft_dict = convert_state_dict_to_peft(state_dict)
    except Exception as e:
        if str(e) == "Could not automatically infer state dict type":
            peft_dict = convert_unet_state_dict_to_peft(state_dict)
        else:
            raise

    if not any("lora_A" in key or "lora_B" in key for key in peft_dict.keys()):
        raise ValueError("Your LoRA was not converted to PEFT")

    return peft_dict


def convert_state_dict_to_kohya(state_dict, original_type=None, **kwargs):
    r"""
    Converts a `PEFT` state dict to `Kohya` format that can be used in AUTOMATIC1111, ComfyUI, SD.Next, InvokeAI, etc.
    The method only supports the conversion from PEFT to Kohya for now.

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            The state dict to convert.
        original_type (`StateDictType`, *optional*):
            The original type of the state dict, if not provided, the method will try to infer it automatically.
        kwargs (`dict`, *args*):
            Additional arguments to pass to the method.

            - **adapter_name**: For example, in case of PEFT, some keys will be prepended
                with the adapter name, therefore needs a special handling. By default PEFT also takes care of that in
                `get_peft_model_state_dict` method:
                https://github.com/huggingface/peft/blob/ba0477f2985b1ba311b83459d29895c809404e99/src/peft/utils/save_and_load.py#L92
                but we add it here in case we don't want to rely on that method.
    """
    try:
        import torch
    except ImportError:
        logger.error("Converting PEFT state dicts to Kohya requires torch to be installed.")
        raise

    peft_adapter_name = kwargs.pop("adapter_name", None)
    if peft_adapter_name is not None:
        peft_adapter_name = "." + peft_adapter_name
    else:
        peft_adapter_name = ""

    if original_type is None:
        if any(f".lora_A{peft_adapter_name}.weight" in k for k in state_dict.keys()):
            original_type = StateDictType.PEFT

    if original_type not in KOHYA_STATE_DICT_MAPPINGS.keys():
        raise ValueError(f"Original type {original_type} is not supported")

    # Use the convert_state_dict function with the appropriate mapping
    kohya_ss_partial_state_dict = convert_state_dict(state_dict, KOHYA_STATE_DICT_MAPPINGS[StateDictType.PEFT])
    kohya_ss_state_dict = {}

    # Additional logic for replacing header, alpha parameters `.` with `_` in all keys
    for kohya_key, weight in kohya_ss_partial_state_dict.items():
        if "text_encoder_2." in kohya_key:
            kohya_key = kohya_key.replace("text_encoder_2.", "lora_te2.")
        elif "text_encoder." in kohya_key:
            kohya_key = kohya_key.replace("text_encoder.", "lora_te1.")
        elif "unet" in kohya_key:
            kohya_key = kohya_key.replace("unet", "lora_unet")
        elif "lora_magnitude_vector" in kohya_key:
            kohya_key = kohya_key.replace("lora_magnitude_vector", "dora_scale")

        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_key = kohya_key.replace(peft_adapter_name, "")  # Kohya doesn't take names
        kohya_ss_state_dict[kohya_key] = weight
        if "lora_down" in kohya_key:
            alpha_key = f"{kohya_key.split('.')[0]}.alpha"
            kohya_ss_state_dict[alpha_key] = torch.tensor(len(weight))

    return kohya_ss_state_dict


def state_dict_all_zero(state_dict, filter_str=None):
    if filter_str is not None:
        if isinstance(filter_str, str):
            filter_str = [filter_str]
        state_dict = {k: v for k, v in state_dict.items() if any(f in k for f in filter_str)}

    return all(torch.all(param == 0).item() for param in state_dict.values())


def _load_sft_state_dict_metadata(model_file: str):
    import safetensors.torch

    from ..loaders.lora_base import LORA_ADAPTER_METADATA_KEY

    with safetensors.torch.safe_open(model_file, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}

    metadata.pop("format", None)
    if metadata:
        raw = metadata.get(LORA_ADAPTER_METADATA_KEY)
        return json.loads(raw) if raw else None
    else:
        return None
