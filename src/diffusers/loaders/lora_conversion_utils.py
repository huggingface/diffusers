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


def _convert_non_diffusers_lora_to_diffusers(state_dict, unet_name="unet", text_encoder_name="text_encoder"):
    """
    Converts a non-Diffusers LoRA state dict to a Diffusers compatible state dict.

    Args:
        state_dict (`dict`): The state dict to convert.
        unet_name (`str`, optional): The name of the U-Net module in the Diffusers model. Defaults to "unet".
        text_encoder_name (`str`, optional): The name of the text encoder module in the Diffusers model. Defaults to
            "text_encoder".

    Returns:
        `tuple`: A tuple containing the converted state dict and a dictionary of alphas.
    """
    unet_state_dict = {}
    te_state_dict = {}
    te2_state_dict = {}
    network_alphas = {}

    # Check for DoRA-enabled LoRAs.
    dora_present_in_unet = any("dora_scale" in k and "lora_unet_" in k for k in state_dict)
    dora_present_in_te = any("dora_scale" in k and ("lora_te_" in k or "lora_te1_" in k) for k in state_dict)
    dora_present_in_te2 = any("dora_scale" in k and "lora_te2_" in k for k in state_dict)
    if dora_present_in_unet or dora_present_in_te or dora_present_in_te2:
        if is_peft_version("<", "0.9.0"):
            raise ValueError(
                "You need `peft` 0.9.0 at least to use DoRA-enabled LoRAs. Please upgrade your installation of `peft`."
            )

    # Iterate over all LoRA weights.
    all_lora_keys = list(state_dict.keys())
    for key in all_lora_keys:
        if not key.endswith("lora_down.weight"):
            continue

        # Extract LoRA name.
        lora_name = key.split(".")[0]

        # Find corresponding up weight and alpha.
        lora_name_up = lora_name + ".lora_up.weight"
        lora_name_alpha = lora_name + ".alpha"

        # Handle U-Net LoRAs.
        if lora_name.startswith("lora_unet_"):
            diffusers_name = _convert_unet_lora_key(key)

            # Store down and up weights.
            unet_state_dict[diffusers_name] = state_dict.pop(key)
            unet_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)

            # Store DoRA scale if present.
            if dora_present_in_unet:
                dora_scale_key_to_replace = "_lora.down." if "_lora.down." in diffusers_name else ".lora.down."
                unet_state_dict[
                    diffusers_name.replace(dora_scale_key_to_replace, ".lora_magnitude_vector.")
                ] = state_dict.pop(key.replace("lora_down.weight", "dora_scale"))

        # Handle text encoder LoRAs.
        elif lora_name.startswith(("lora_te_", "lora_te1_", "lora_te2_")):
            diffusers_name = _convert_text_encoder_lora_key(key, lora_name)

            # Store down and up weights for te or te2.
            if lora_name.startswith(("lora_te_", "lora_te1_")):
                te_state_dict[diffusers_name] = state_dict.pop(key)
                te_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)
            else:
                te2_state_dict[diffusers_name] = state_dict.pop(key)
                te2_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)

            # Store DoRA scale if present.
            if dora_present_in_te or dora_present_in_te2:
                dora_scale_key_to_replace_te = (
                    "_lora.down." if "_lora.down." in diffusers_name else ".lora_linear_layer."
                )
                if lora_name.startswith(("lora_te_", "lora_te1_")):
                    te_state_dict[
                        diffusers_name.replace(dora_scale_key_to_replace_te, ".lora_magnitude_vector.")
                    ] = state_dict.pop(key.replace("lora_down.weight", "dora_scale"))
                elif lora_name.startswith("lora_te2_"):
                    te2_state_dict[
                        diffusers_name.replace(dora_scale_key_to_replace_te, ".lora_magnitude_vector.")
                    ] = state_dict.pop(key.replace("lora_down.weight", "dora_scale"))

        # Store alpha if present.
        if lora_name_alpha in state_dict:
            alpha = state_dict.pop(lora_name_alpha).item()
            network_alphas.update(_get_alpha_name(lora_name_alpha, diffusers_name, alpha))

    # Check if any keys remain.
    if len(state_dict) > 0:
        raise ValueError(f"The following keys have not been correctly renamed: \n\n {', '.join(state_dict.keys())}")

    logger.info("Non-diffusers checkpoint detected.")

    # Construct final state dict.
    unet_state_dict = {f"{unet_name}.{module_name}": params for module_name, params in unet_state_dict.items()}
    te_state_dict = {f"{text_encoder_name}.{module_name}": params for module_name, params in te_state_dict.items()}
    te2_state_dict = (
        {f"text_encoder_2.{module_name}": params for module_name, params in te2_state_dict.items()}
        if len(te2_state_dict) > 0
        else None
    )
    if te2_state_dict is not None:
        te_state_dict.update(te2_state_dict)

    new_state_dict = {**unet_state_dict, **te_state_dict}
    return new_state_dict, network_alphas


def _convert_unet_lora_key(key):
    """
    Converts a U-Net LoRA key to a Diffusers compatible key.
    """
    diffusers_name = key.replace("lora_unet_", "").replace("_", ".")

    # Replace common U-Net naming patterns.
    diffusers_name = diffusers_name.replace("input.blocks", "down_blocks")
    diffusers_name = diffusers_name.replace("down.blocks", "down_blocks")
    diffusers_name = diffusers_name.replace("middle.block", "mid_block")
    diffusers_name = diffusers_name.replace("mid.block", "mid_block")
    diffusers_name = diffusers_name.replace("output.blocks", "up_blocks")
    diffusers_name = diffusers_name.replace("up.blocks", "up_blocks")
    diffusers_name = diffusers_name.replace("transformer.blocks", "transformer_blocks")
    diffusers_name = diffusers_name.replace("to.q.lora", "to_q_lora")
    diffusers_name = diffusers_name.replace("to.k.lora", "to_k_lora")
    diffusers_name = diffusers_name.replace("to.v.lora", "to_v_lora")
    diffusers_name = diffusers_name.replace("to.out.0.lora", "to_out_lora")
    diffusers_name = diffusers_name.replace("proj.in", "proj_in")
    diffusers_name = diffusers_name.replace("proj.out", "proj_out")
    diffusers_name = diffusers_name.replace("emb.layers", "time_emb_proj")

    # SDXL specific conversions.
    if "emb" in diffusers_name and "time.emb.proj" not in diffusers_name:
        pattern = r"\.\d+(?=\D*$)"
        diffusers_name = re.sub(pattern, "", diffusers_name, count=1)
    if ".in." in diffusers_name:
        diffusers_name = diffusers_name.replace("in.layers.2", "conv1")
    if ".out." in diffusers_name:
        diffusers_name = diffusers_name.replace("out.layers.3", "conv2")
    if "downsamplers" in diffusers_name or "upsamplers" in diffusers_name:
        diffusers_name = diffusers_name.replace("op", "conv")
    if "skip" in diffusers_name:
        diffusers_name = diffusers_name.replace("skip.connection", "conv_shortcut")

    # LyCORIS specific conversions.
    if "time.emb.proj" in diffusers_name:
        diffusers_name = diffusers_name.replace("time.emb.proj", "time_emb_proj")
    if "conv.shortcut" in diffusers_name:
        diffusers_name = diffusers_name.replace("conv.shortcut", "conv_shortcut")

    # General conversions.
    if "transformer_blocks" in diffusers_name:
        if "attn1" in diffusers_name or "attn2" in diffusers_name:
            diffusers_name = diffusers_name.replace("attn1", "attn1.processor")
            diffusers_name = diffusers_name.replace("attn2", "attn2.processor")
        elif "ff" in diffusers_name:
            pass
    elif any(key in diffusers_name for key in ("proj_in", "proj_out")):
        pass
    else:
        pass

    return diffusers_name


def _convert_text_encoder_lora_key(key, lora_name):
    """
    Converts a text encoder LoRA key to a Diffusers compatible key.
    """
    if lora_name.startswith(("lora_te_", "lora_te1_")):
        key_to_replace = "lora_te_" if lora_name.startswith("lora_te_") else "lora_te1_"
    else:
        key_to_replace = "lora_te2_"

    diffusers_name = key.replace(key_to_replace, "").replace("_", ".")
    diffusers_name = diffusers_name.replace("text.model", "text_model")
    diffusers_name = diffusers_name.replace("self.attn", "self_attn")
    diffusers_name = diffusers_name.replace("q.proj.lora", "to_q_lora")
    diffusers_name = diffusers_name.replace("k.proj.lora", "to_k_lora")
    diffusers_name = diffusers_name.replace("v.proj.lora", "to_v_lora")
    diffusers_name = diffusers_name.replace("out.proj.lora", "to_out_lora")
    diffusers_name = diffusers_name.replace("text.projection", "text_projection")

    if "self_attn" in diffusers_name or "text_projection" in diffusers_name:
        pass
    elif "mlp" in diffusers_name:
        # Be aware that this is the new diffusers convention and the rest of the code might
        # not utilize it yet.
        diffusers_name = diffusers_name.replace(".lora.", ".lora_linear_layer.")
    return diffusers_name


def _get_alpha_name(lora_name_alpha, diffusers_name, alpha):
    """
    Gets the correct alpha name for the Diffusers model.
    """
    if lora_name_alpha.startswith("lora_unet_"):
        prefix = "unet."
    elif lora_name_alpha.startswith(("lora_te_", "lora_te1_")):
        prefix = "text_encoder."
    else:
        prefix = "text_encoder_2."
    new_name = prefix + diffusers_name.split(".lora.")[0] + ".alpha"
    return {new_name: alpha}
