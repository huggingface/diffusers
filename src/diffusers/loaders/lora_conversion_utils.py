# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import re
from typing import Any, Dict, Tuple

import torch

from ..utils import is_peft_version, logging


logger = logging.get_logger(__name__)


def _maybe_map_sgm_blocks_to_diffusers(state_dict, unet_config, delimiter="_", block_slice_pos=5):
    # 1. get all state_dict_keys
    all_keys = list(state_dict.keys())
    sgm_patterns = ["input_blocks", "middle_block", "output_blocks"]

    # 2. check if needs remapping, if not return original dict
    is_in_sgm_format = False
    for key in all_keys:
        if any(p in key for p in sgm_patterns):
            is_in_sgm_format = True
            break

    if not is_in_sgm_format:
        return state_dict

    # 3. Else remap from SGM patterns
    new_state_dict = {}
    inner_block_map = ["resnets", "attentions", "upsamplers"]

    # Retrieves # of down, mid and up blocks
    input_block_ids, middle_block_ids, output_block_ids = set(), set(), set()

    for layer in all_keys:
        if "text" in layer:
            new_state_dict[layer] = state_dict.pop(layer)
        else:
            layer_id = int(layer.split(delimiter)[:block_slice_pos][-1])
            if sgm_patterns[0] in layer:
                input_block_ids.add(layer_id)
            elif sgm_patterns[1] in layer:
                middle_block_ids.add(layer_id)
            elif sgm_patterns[2] in layer:
                output_block_ids.add(layer_id)
            else:
                raise ValueError(f"Checkpoint not supported because layer {layer} not supported.")

    input_blocks = {
        layer_id: [key for key in state_dict if f"input_blocks{delimiter}{layer_id}" in key]
        for layer_id in input_block_ids
    }
    middle_blocks = {
        layer_id: [key for key in state_dict if f"middle_block{delimiter}{layer_id}" in key]
        for layer_id in middle_block_ids
    }
    output_blocks = {
        layer_id: [key for key in state_dict if f"output_blocks{delimiter}{layer_id}" in key]
        for layer_id in output_block_ids
    }

    # Rename keys accordingly
    for i in input_block_ids:
        block_id = (i - 1) // (unet_config.layers_per_block + 1)
        layer_in_block_id = (i - 1) % (unet_config.layers_per_block + 1)

        for key in input_blocks[i]:
            inner_block_id = int(key.split(delimiter)[block_slice_pos])
            inner_block_key = inner_block_map[inner_block_id] if "op" not in key else "downsamplers"
            inner_layers_in_block = str(layer_in_block_id) if "op" not in key else "0"
            new_key = delimiter.join(
                key.split(delimiter)[: block_slice_pos - 1]
                + [str(block_id), inner_block_key, inner_layers_in_block]
                + key.split(delimiter)[block_slice_pos + 1 :]
            )
            new_state_dict[new_key] = state_dict.pop(key)

    for i in middle_block_ids:
        key_part = None
        if i == 0:
            key_part = [inner_block_map[0], "0"]
        elif i == 1:
            key_part = [inner_block_map[1], "0"]
        elif i == 2:
            key_part = [inner_block_map[0], "1"]
        else:
            raise ValueError(f"Invalid middle block id {i}.")

        for key in middle_blocks[i]:
            new_key = delimiter.join(
                key.split(delimiter)[: block_slice_pos - 1] + key_part + key.split(delimiter)[block_slice_pos:]
            )
            new_state_dict[new_key] = state_dict.pop(key)

    for i in output_block_ids:
        block_id = i // (unet_config.layers_per_block + 1)
        layer_in_block_id = i % (unet_config.layers_per_block + 1)

        for key in output_blocks[i]:
            inner_block_id = int(key.split(delimiter)[block_slice_pos])
            inner_block_key = inner_block_map[inner_block_id]
            inner_layers_in_block = str(layer_in_block_id) if inner_block_id < 2 else "0"
            new_key = delimiter.join(
                key.split(delimiter)[: block_slice_pos - 1]
                + [str(block_id), inner_block_key, inner_layers_in_block]
                + key.split(delimiter)[block_slice_pos + 1 :]
            )
            new_state_dict[new_key] = state_dict.pop(key)

    if len(state_dict) > 0:
        raise ValueError("At this point all state dict entries have to be converted.")

    return new_state_dict


def _convert_non_diffusers_lora_to_diffusers(
    state_dict: Dict[str, torch.Tensor], unet_name: str = "unet", text_encoder_name: str = "text_encoder"
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    def detect_dora_lora(state_dict: Dict[str, torch.Tensor]) -> Tuple[bool, bool, bool]:
        is_unet_dora_lora = any("dora_scale" in k and "lora_unet_" in k for k in state_dict)
        is_te_dora_lora = any("dora_scale" in k and ("lora_te_" in k or "lora_te1_" in k) for k in state_dict)
        is_te2_dora_lora = any("dora_scale" in k and "lora_te2_" in k for k in state_dict)
        return is_unet_dora_lora, is_te_dora_lora, is_te2_dora_lora

    def check_peft_version(is_unet_dora_lora: bool, is_te_dora_lora: bool, is_te2_dora_lora: bool):
        if is_unet_dora_lora or is_te_dora_lora or is_te2_dora_lora:
            if is_peft_version("<", "0.9.0"):
                raise ValueError(
                    "You need `peft` 0.9.0 at least to use DoRA-enabled LoRAs. Please upgrade your installation of `peft`."
                )

    def rename_keys(
        state_dict: Dict[str, torch.Tensor],
        key: str,
        unet_state_dict: Dict[str, torch.Tensor],
        te_state_dict: Dict[str, torch.Tensor],
        te2_state_dict: Dict[str, torch.Tensor],
        is_unet_dora_lora: bool,
        is_te_dora_lora: bool,
        is_te2_dora_lora: bool,
    ):
        lora_name = key.split(".")[0]
        lora_name_up = lora_name + ".lora_up.weight"
        diffusers_name = key.replace(lora_name + ".", "").replace("_", ".")
        lora_type = lora_name.split("_")[1]

        if lora_type == "unet":
            diffusers_name = _adjust_unet_names(diffusers_name)
            unet_state_dict = _populate_state_dict(
                unet_state_dict, state_dict, key, lora_name_up, diffusers_name, is_unet_dora_lora
            )
        else:
            diffusers_name = _adjust_text_encoder_names(diffusers_name)
            if lora_type in ["te", "te1"]:
                te_state_dict = _populate_state_dict(
                    te_state_dict, state_dict, key, lora_name_up, diffusers_name, is_te_dora_lora
                )
            else:
                te2_state_dict = _populate_state_dict(
                    te2_state_dict, state_dict, key, lora_name_up, diffusers_name, is_te2_dora_lora
                )

        return unet_state_dict, te_state_dict, te2_state_dict

    def _adjust_unet_names(name: str) -> str:
        replacements = [
            ("input.blocks", "down_blocks"),
            ("down.blocks", "down_blocks"),
            ("middle.block", "mid_block"),
            ("mid.block", "mid_block"),
            ("output.blocks", "up_blocks"),
            ("up.blocks", "up_blocks"),
            ("transformer.blocks", "transformer_blocks"),
            ("to.q.lora", "to_q_lora"),
            ("to.k.lora", "to_k_lora"),
            ("to.v.lora", "to_v_lora"),
            ("to.out.0.lora", "to_out_lora"),
            ("proj.in", "proj_in"),
            ("proj.out", "proj_out"),
            ("emb.layers", "time_emb_proj"),
            ("time.emb.proj", "time_emb_proj"),
            ("conv.shortcut", "conv_shortcut"),
            ("skip.connection", "conv_shortcut"),
        ]
        for old, new in replacements:
            name = name.replace(old, new)
        if "emb" in name and "time.emb.proj" not in name:
            pattern = r"\.\d+(?=\D*$)"
            name = re.sub(pattern, "", name, count=1)
        if ".in." in name:
            name = name.replace("in.layers.2", "conv1")
        if ".out." in name:
            name = name.replace("out.layers.3", "conv2")
        if "downsamplers" in name or "upsamplers" in name:
            name = name.replace("op", "conv")
        return name

    def _adjust_text_encoder_names(name: str) -> str:
        replacements = [
            ("text.model", "text_model"),
            ("self.attn", "self_attn"),
            ("q.proj.lora", "to_q_lora"),
            ("k.proj.lora", "to_k_lora"),
            ("v.proj.lora", "to_v_lora"),
            ("out.proj.lora", "to_out_lora"),
            ("text.projection", "text_projection"),
        ]
        for old, new in replacements:
            name = name.replace(old, new)
        return name

    def _populate_state_dict(state_dict, main_dict, down_key, up_key, name, is_dora_lora):
        state_dict[name] = main_dict.pop(down_key)
        state_dict[name.replace(".down.", ".up.")] = main_dict.pop(up_key)
        if is_dora_lora:
            dora_key = down_key.replace("lora_down.weight", "dora_scale")
            scale_key = "_lora.down." if "_lora.down." in name else ".lora.down."
            state_dict[name.replace(scale_key, ".lora_magnitude_vector.")] = main_dict.pop(dora_key)
        return state_dict

    def update_network_alphas(
        state_dict: Dict[str, torch.Tensor],
        network_alphas: Dict[str, float],
        diffusers_name: str,
        lora_name_alpha: str,
    ):
        if lora_name_alpha in state_dict:
            alpha = state_dict.pop(lora_name_alpha).item()
            prefix = (
                "unet."
                if "unet" in lora_name_alpha
                else "text_encoder."
                if "te1" in lora_name_alpha
                else "text_encoder_2."
            )
            new_name = prefix + diffusers_name.split(".lora.")[0] + ".alpha"
            network_alphas.update({new_name: alpha})

    unet_state_dict = {}
    te_state_dict = {}
    te2_state_dict = {}
    network_alphas = {}

    is_unet_dora_lora, is_te_dora_lora, is_te2_dora_lora = detect_dora_lora(state_dict)
    check_peft_version(is_unet_dora_lora, is_te_dora_lora, is_te2_dora_lora)

    lora_keys = [k for k in state_dict.keys() if k.endswith("lora_down.weight")]
    for key in lora_keys:
        unet_state_dict, te_state_dict, te2_state_dict = rename_keys(
            state_dict,
            key,
            unet_state_dict,
            te_state_dict,
            te2_state_dict,
            is_unet_dora_lora,
            is_te_dora_lora,
            is_te2_dora_lora,
        )
        lora_name = key.split(".")[0]
        lora_name_alpha = lora_name + ".alpha"
        diffusers_name = key.replace(lora_name + ".", "").replace("_", ".")
        update_network_alphas(state_dict, network_alphas, diffusers_name, lora_name_alpha)

    if state_dict:
        raise ValueError(f"The following keys have not been correctly renamed: \n\n {', '.join(state_dict.keys())}")

    logger.info("Non-diffusers LoRA checkpoint detected.")
    unet_state_dict = {f"{unet_name}.{module_name}": params for module_name, params in unet_state_dict.items()}
    te_state_dict = {f"{text_encoder_name}.{module_name}": params for module_name, params in te_state_dict.items()}

    if te2_state_dict:
        te2_state_dict = {f"text_encoder_2.{module_name}": params for module_name, params in te2_state_dict.items()}
        te_state_dict.update(te2_state_dict)

    new_state_dict = {**unet_state_dict, **te_state_dict}
    return new_state_dict, network_alphas
