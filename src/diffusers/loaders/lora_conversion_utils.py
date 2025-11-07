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

import re
from typing import List

import torch

from ..utils import is_peft_version, logging, state_dict_all_zero


logger = logging.get_logger(__name__)


def swap_scale_shift(weight):
    shift, scale = weight.chunk(2, dim=0)
    new_weight = torch.cat([scale, shift], dim=0)
    return new_weight


def _maybe_map_sgm_blocks_to_diffusers(state_dict, unet_config, delimiter="_", block_slice_pos=5):
    # 1. get all state_dict_keys
    all_keys = list(state_dict.keys())
    sgm_patterns = ["input_blocks", "middle_block", "output_blocks"]
    not_sgm_patterns = ["down_blocks", "mid_block", "up_blocks"]

    # check if state_dict contains both patterns
    contains_sgm_patterns = False
    contains_not_sgm_patterns = False
    for key in all_keys:
        if any(p in key for p in sgm_patterns):
            contains_sgm_patterns = True
        elif any(p in key for p in not_sgm_patterns):
            contains_not_sgm_patterns = True

    # if state_dict contains both patterns, remove sgm
    # we can then return state_dict immediately
    if contains_sgm_patterns and contains_not_sgm_patterns:
        for key in all_keys:
            if any(p in key for p in sgm_patterns):
                state_dict.pop(key)
        return state_dict

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

    if state_dict:
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
                unet_state_dict[diffusers_name.replace(dora_scale_key_to_replace, ".lora_magnitude_vector.")] = (
                    state_dict.pop(key.replace("lora_down.weight", "dora_scale"))
                )

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
                    te_state_dict[diffusers_name.replace(dora_scale_key_to_replace_te, ".lora_magnitude_vector.")] = (
                        state_dict.pop(key.replace("lora_down.weight", "dora_scale"))
                    )
                elif lora_name.startswith("lora_te2_"):
                    te2_state_dict[diffusers_name.replace(dora_scale_key_to_replace_te, ".lora_magnitude_vector.")] = (
                        state_dict.pop(key.replace("lora_down.weight", "dora_scale"))
                    )

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


# The utilities under `_convert_kohya_flux_lora_to_diffusers()`
# are adapted from https://github.com/kohya-ss/sd-scripts/blob/a61cf73a5cb5209c3f4d1a3688dd276a4dfd1ecb/networks/convert_flux_lora.py
def _convert_kohya_flux_lora_to_diffusers(state_dict):
    def _convert_to_ai_toolkit(sds_sd, ait_sd, sds_key, ait_key):
        if sds_key + ".lora_down.weight" not in sds_sd:
            return
        down_weight = sds_sd.pop(sds_key + ".lora_down.weight")

        # scale weight by alpha and dim
        rank = down_weight.shape[0]
        default_alpha = torch.tensor(rank, dtype=down_weight.dtype, device=down_weight.device, requires_grad=False)
        alpha = sds_sd.pop(sds_key + ".alpha", default_alpha).item()  # alpha is scalar
        scale = alpha / rank  # LoRA is scaled by 'alpha / rank' in forward pass, so we need to scale it back here

        # calculate scale_down and scale_up to keep the same value. if scale is 4, scale_down is 2 and scale_up is 2
        scale_down = scale
        scale_up = 1.0
        while scale_down * 2 < scale_up:
            scale_down *= 2
            scale_up /= 2

        ait_sd[ait_key + ".lora_A.weight"] = down_weight * scale_down
        ait_sd[ait_key + ".lora_B.weight"] = sds_sd.pop(sds_key + ".lora_up.weight") * scale_up

    def _convert_to_ai_toolkit_cat(sds_sd, ait_sd, sds_key, ait_keys, dims=None):
        if sds_key + ".lora_down.weight" not in sds_sd:
            return
        down_weight = sds_sd.pop(sds_key + ".lora_down.weight")
        up_weight = sds_sd.pop(sds_key + ".lora_up.weight")
        sd_lora_rank = down_weight.shape[0]

        # scale weight by alpha and dim
        default_alpha = torch.tensor(
            sd_lora_rank, dtype=down_weight.dtype, device=down_weight.device, requires_grad=False
        )
        alpha = sds_sd.pop(sds_key + ".alpha", default_alpha)
        scale = alpha / sd_lora_rank

        # calculate scale_down and scale_up
        scale_down = scale
        scale_up = 1.0
        while scale_down * 2 < scale_up:
            scale_down *= 2
            scale_up /= 2

        down_weight = down_weight * scale_down
        up_weight = up_weight * scale_up

        # calculate dims if not provided
        num_splits = len(ait_keys)
        if dims is None:
            dims = [up_weight.shape[0] // num_splits] * num_splits
        else:
            assert sum(dims) == up_weight.shape[0]

        # check upweight is sparse or not
        is_sparse = False
        if sd_lora_rank % num_splits == 0:
            ait_rank = sd_lora_rank // num_splits
            is_sparse = True
            i = 0
            for j in range(len(dims)):
                for k in range(len(dims)):
                    if j == k:
                        continue
                    is_sparse = is_sparse and torch.all(
                        up_weight[i : i + dims[j], k * ait_rank : (k + 1) * ait_rank] == 0
                    )
                i += dims[j]
            if is_sparse:
                logger.info(f"weight is sparse: {sds_key}")

        # make ai-toolkit weight
        ait_down_keys = [k + ".lora_A.weight" for k in ait_keys]
        ait_up_keys = [k + ".lora_B.weight" for k in ait_keys]
        if not is_sparse:
            # down_weight is copied to each split
            ait_sd.update(dict.fromkeys(ait_down_keys, down_weight))

            # up_weight is split to each split
            ait_sd.update({k: v for k, v in zip(ait_up_keys, torch.split(up_weight, dims, dim=0))})  # noqa: C416
        else:
            # down_weight is chunked to each split
            ait_sd.update({k: v for k, v in zip(ait_down_keys, torch.chunk(down_weight, num_splits, dim=0))})  # noqa: C416

            # up_weight is sparse: only non-zero values are copied to each split
            i = 0
            for j in range(len(dims)):
                ait_sd[ait_up_keys[j]] = up_weight[i : i + dims[j], j * ait_rank : (j + 1) * ait_rank].contiguous()
                i += dims[j]

    def _convert_sd_scripts_to_ai_toolkit(sds_sd):
        ait_sd = {}
        for i in range(19):
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_img_attn_proj",
                f"transformer.transformer_blocks.{i}.attn.to_out.0",
            )
            _convert_to_ai_toolkit_cat(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_img_attn_qkv",
                [
                    f"transformer.transformer_blocks.{i}.attn.to_q",
                    f"transformer.transformer_blocks.{i}.attn.to_k",
                    f"transformer.transformer_blocks.{i}.attn.to_v",
                ],
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_img_mlp_0",
                f"transformer.transformer_blocks.{i}.ff.net.0.proj",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_img_mlp_2",
                f"transformer.transformer_blocks.{i}.ff.net.2",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_img_mod_lin",
                f"transformer.transformer_blocks.{i}.norm1.linear",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_txt_attn_proj",
                f"transformer.transformer_blocks.{i}.attn.to_add_out",
            )
            _convert_to_ai_toolkit_cat(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_txt_attn_qkv",
                [
                    f"transformer.transformer_blocks.{i}.attn.add_q_proj",
                    f"transformer.transformer_blocks.{i}.attn.add_k_proj",
                    f"transformer.transformer_blocks.{i}.attn.add_v_proj",
                ],
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_txt_mlp_0",
                f"transformer.transformer_blocks.{i}.ff_context.net.0.proj",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_txt_mlp_2",
                f"transformer.transformer_blocks.{i}.ff_context.net.2",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_txt_mod_lin",
                f"transformer.transformer_blocks.{i}.norm1_context.linear",
            )

        for i in range(38):
            _convert_to_ai_toolkit_cat(
                sds_sd,
                ait_sd,
                f"lora_unet_single_blocks_{i}_linear1",
                [
                    f"transformer.single_transformer_blocks.{i}.attn.to_q",
                    f"transformer.single_transformer_blocks.{i}.attn.to_k",
                    f"transformer.single_transformer_blocks.{i}.attn.to_v",
                    f"transformer.single_transformer_blocks.{i}.proj_mlp",
                ],
                dims=[3072, 3072, 3072, 12288],
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_single_blocks_{i}_linear2",
                f"transformer.single_transformer_blocks.{i}.proj_out",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_single_blocks_{i}_modulation_lin",
                f"transformer.single_transformer_blocks.{i}.norm.linear",
            )

        # TODO: alphas.
        def assign_remaining_weights(assignments, source):
            for lora_key in ["lora_A", "lora_B"]:
                orig_lora_key = "lora_down" if lora_key == "lora_A" else "lora_up"
                for target_fmt, source_fmt, transform in assignments:
                    target_key = target_fmt.format(lora_key=lora_key)
                    source_key = source_fmt.format(orig_lora_key=orig_lora_key)
                    value = source.pop(source_key)
                    if transform:
                        value = transform(value)
                    ait_sd[target_key] = value

        if any("guidance_in" in k for k in sds_sd):
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                "lora_unet_guidance_in_in_layer",
                "time_text_embed.guidance_embedder.linear_1",
            )

            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                "lora_unet_guidance_in_out_layer",
                "time_text_embed.guidance_embedder.linear_2",
            )

        if any("img_in" in k for k in sds_sd):
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                "lora_unet_img_in",
                "x_embedder",
            )

        if any("txt_in" in k for k in sds_sd):
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                "lora_unet_txt_in",
                "context_embedder",
            )

        if any("time_in" in k for k in sds_sd):
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                "lora_unet_time_in_in_layer",
                "time_text_embed.timestep_embedder.linear_1",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                "lora_unet_time_in_out_layer",
                "time_text_embed.timestep_embedder.linear_2",
            )

        if any("vector_in" in k for k in sds_sd):
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                "lora_unet_vector_in_in_layer",
                "time_text_embed.text_embedder.linear_1",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                "lora_unet_vector_in_out_layer",
                "time_text_embed.text_embedder.linear_2",
            )

        if any("final_layer" in k for k in sds_sd):
            # Notice the swap in processing for "final_layer".
            assign_remaining_weights(
                [
                    (
                        "norm_out.linear.{lora_key}.weight",
                        "lora_unet_final_layer_adaLN_modulation_1.{orig_lora_key}.weight",
                        swap_scale_shift,
                    ),
                    ("proj_out.{lora_key}.weight", "lora_unet_final_layer_linear.{orig_lora_key}.weight", None),
                ],
                sds_sd,
            )

        remaining_keys = list(sds_sd.keys())
        te_state_dict = {}
        if remaining_keys:
            if not all(k.startswith(("lora_te", "lora_te1")) for k in remaining_keys):
                raise ValueError(f"Incompatible keys detected: \n\n {', '.join(remaining_keys)}")
            for key in remaining_keys:
                if not key.endswith("lora_down.weight"):
                    continue

                lora_name = key.split(".")[0]
                lora_name_up = f"{lora_name}.lora_up.weight"
                lora_name_alpha = f"{lora_name}.alpha"
                diffusers_name = _convert_text_encoder_lora_key(key, lora_name)

                if lora_name.startswith(("lora_te_", "lora_te1_")):
                    down_weight = sds_sd.pop(key)
                    sd_lora_rank = down_weight.shape[0]
                    te_state_dict[diffusers_name] = down_weight
                    te_state_dict[diffusers_name.replace(".down.", ".up.")] = sds_sd.pop(lora_name_up)

                if lora_name_alpha in sds_sd:
                    alpha = sds_sd.pop(lora_name_alpha).item()
                    scale = alpha / sd_lora_rank

                    scale_down = scale
                    scale_up = 1.0
                    while scale_down * 2 < scale_up:
                        scale_down *= 2
                        scale_up /= 2

                    te_state_dict[diffusers_name] *= scale_down
                    te_state_dict[diffusers_name.replace(".down.", ".up.")] *= scale_up

        if len(sds_sd) > 0:
            logger.warning(f"Unsupported keys for ai-toolkit: {sds_sd.keys()}")

        if te_state_dict:
            te_state_dict = {f"text_encoder.{module_name}": params for module_name, params in te_state_dict.items()}

        new_state_dict = {**ait_sd, **te_state_dict}
        return new_state_dict

    def _convert_mixture_state_dict_to_diffusers(state_dict):
        new_state_dict = {}

        def _convert(original_key, diffusers_key, state_dict, new_state_dict):
            down_key = f"{original_key}.lora_down.weight"
            down_weight = state_dict.pop(down_key)
            lora_rank = down_weight.shape[0]

            up_weight_key = f"{original_key}.lora_up.weight"
            up_weight = state_dict.pop(up_weight_key)

            alpha_key = f"{original_key}.alpha"
            alpha = state_dict.pop(alpha_key)

            # scale weight by alpha and dim
            scale = alpha / lora_rank
            # calculate scale_down and scale_up
            scale_down = scale
            scale_up = 1.0
            while scale_down * 2 < scale_up:
                scale_down *= 2
                scale_up /= 2
            down_weight = down_weight * scale_down
            up_weight = up_weight * scale_up

            diffusers_down_key = f"{diffusers_key}.lora_A.weight"
            new_state_dict[diffusers_down_key] = down_weight
            new_state_dict[diffusers_down_key.replace(".lora_A.", ".lora_B.")] = up_weight

        all_unique_keys = {
            k.replace(".lora_down.weight", "").replace(".lora_up.weight", "").replace(".alpha", "")
            for k in state_dict
            if not k.startswith(("lora_unet_"))
        }
        assert all(k.startswith(("lora_transformer_", "lora_te1_")) for k in all_unique_keys), f"{all_unique_keys=}"

        has_te_keys = False
        for k in all_unique_keys:
            if k.startswith("lora_transformer_single_transformer_blocks_"):
                i = int(k.split("lora_transformer_single_transformer_blocks_")[-1].split("_")[0])
                diffusers_key = f"single_transformer_blocks.{i}"
            elif k.startswith("lora_transformer_transformer_blocks_"):
                i = int(k.split("lora_transformer_transformer_blocks_")[-1].split("_")[0])
                diffusers_key = f"transformer_blocks.{i}"
            elif k.startswith("lora_te1_"):
                has_te_keys = True
                continue
            elif k.startswith("lora_transformer_context_embedder"):
                diffusers_key = "context_embedder"
            elif k.startswith("lora_transformer_norm_out_linear"):
                diffusers_key = "norm_out.linear"
            elif k.startswith("lora_transformer_proj_out"):
                diffusers_key = "proj_out"
            elif k.startswith("lora_transformer_x_embedder"):
                diffusers_key = "x_embedder"
            elif k.startswith("lora_transformer_time_text_embed_guidance_embedder_linear_"):
                i = int(k.split("lora_transformer_time_text_embed_guidance_embedder_linear_")[-1])
                diffusers_key = f"time_text_embed.guidance_embedder.linear_{i}"
            elif k.startswith("lora_transformer_time_text_embed_text_embedder_linear_"):
                i = int(k.split("lora_transformer_time_text_embed_text_embedder_linear_")[-1])
                diffusers_key = f"time_text_embed.text_embedder.linear_{i}"
            elif k.startswith("lora_transformer_time_text_embed_timestep_embedder_linear_"):
                i = int(k.split("lora_transformer_time_text_embed_timestep_embedder_linear_")[-1])
                diffusers_key = f"time_text_embed.timestep_embedder.linear_{i}"
            else:
                raise NotImplementedError(f"Handling for key ({k}) is not implemented.")

            if "attn_" in k:
                if "_to_out_0" in k:
                    diffusers_key += ".attn.to_out.0"
                elif "_to_add_out" in k:
                    diffusers_key += ".attn.to_add_out"
                elif any(qkv in k for qkv in ["to_q", "to_k", "to_v"]):
                    remaining = k.split("attn_")[-1]
                    diffusers_key += f".attn.{remaining}"
                elif any(add_qkv in k for add_qkv in ["add_q_proj", "add_k_proj", "add_v_proj"]):
                    remaining = k.split("attn_")[-1]
                    diffusers_key += f".attn.{remaining}"

            _convert(k, diffusers_key, state_dict, new_state_dict)

        if has_te_keys:
            layer_pattern = re.compile(r"lora_te1_text_model_encoder_layers_(\d+)")
            attn_mapping = {
                "q_proj": ".self_attn.q_proj",
                "k_proj": ".self_attn.k_proj",
                "v_proj": ".self_attn.v_proj",
                "out_proj": ".self_attn.out_proj",
            }
            mlp_mapping = {"fc1": ".mlp.fc1", "fc2": ".mlp.fc2"}
            for k in all_unique_keys:
                if not k.startswith("lora_te1_"):
                    continue

                match = layer_pattern.search(k)
                if not match:
                    continue
                i = int(match.group(1))
                diffusers_key = f"text_model.encoder.layers.{i}"

                if "attn" in k:
                    for key_fragment, suffix in attn_mapping.items():
                        if key_fragment in k:
                            diffusers_key += suffix
                            break
                elif "mlp" in k:
                    for key_fragment, suffix in mlp_mapping.items():
                        if key_fragment in k:
                            diffusers_key += suffix
                            break

                _convert(k, diffusers_key, state_dict, new_state_dict)

        remaining_all_unet = False
        if state_dict:
            remaining_all_unet = all(k.startswith("lora_unet_") for k in state_dict)
        if remaining_all_unet:
            keys = list(state_dict.keys())
            for k in keys:
                state_dict.pop(k)

        if len(state_dict) > 0:
            raise ValueError(
                f"Expected an empty state dict at this point but its has these keys which couldn't be parsed: {list(state_dict.keys())}."
            )

        transformer_state_dict = {
            f"transformer.{k}": v for k, v in new_state_dict.items() if not k.startswith("text_model.")
        }
        te_state_dict = {f"text_encoder.{k}": v for k, v in new_state_dict.items() if k.startswith("text_model.")}
        return {**transformer_state_dict, **te_state_dict}

    # This is  weird.
    # https://huggingface.co/sayakpaul/different-lora-from-civitai/tree/main?show_file_info=sharp_detailed_foot.safetensors
    # has both `peft` and non-peft state dict.
    has_peft_state_dict = any(k.startswith("transformer.") for k in state_dict)
    if has_peft_state_dict:
        state_dict = {
            k.replace("lora_down.weight", "lora_A.weight").replace("lora_up.weight", "lora_B.weight"): v
            for k, v in state_dict.items()
            if k.startswith("transformer.")
        }
        return state_dict

    # Another weird one.
    has_mixture = any(
        k.startswith("lora_transformer_") and ("lora_down" in k or "lora_up" in k or "alpha" in k) for k in state_dict
    )

    # ComfyUI.
    if not has_mixture:
        state_dict = {k.replace("diffusion_model.", "lora_unet_"): v for k, v in state_dict.items()}
        state_dict = {k.replace("text_encoders.clip_l.transformer.", "lora_te_"): v for k, v in state_dict.items()}

        has_position_embedding = any("position_embedding" in k for k in state_dict)
        if has_position_embedding:
            zero_status_pe = state_dict_all_zero(state_dict, "position_embedding")
            if zero_status_pe:
                logger.info(
                    "The `position_embedding` LoRA params are all zeros which make them ineffective. "
                    "So, we will purge them out of the current state dict to make loading possible."
                )

            else:
                logger.info(
                    "The state_dict has position_embedding LoRA params and we currently do not support them. "
                    "Open an issue if you need this supported - https://github.com/huggingface/diffusers/issues/new."
                )
            state_dict = {k: v for k, v in state_dict.items() if "position_embedding" not in k}

        has_t5xxl = any(k.startswith("text_encoders.t5xxl.transformer.") for k in state_dict)
        if has_t5xxl:
            zero_status_t5 = state_dict_all_zero(state_dict, "text_encoders.t5xxl")
            if zero_status_t5:
                logger.info(
                    "The `t5xxl` LoRA params are all zeros which make them ineffective. "
                    "So, we will purge them out of the current state dict to make loading possible."
                )
            else:
                logger.info(
                    "T5-xxl keys found in the state dict, which are currently unsupported. We will filter them out."
                    "Open an issue if this is a problem - https://github.com/huggingface/diffusers/issues/new."
                )
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("text_encoders.t5xxl.transformer.")}

        has_diffb = any("diff_b" in k and k.startswith(("lora_unet_", "lora_te_")) for k in state_dict)
        if has_diffb:
            zero_status_diff_b = state_dict_all_zero(state_dict, ".diff_b")
            if zero_status_diff_b:
                logger.info(
                    "The `diff_b` LoRA params are all zeros which make them ineffective. "
                    "So, we will purge them out of the current state dict to make loading possible."
                )
            else:
                logger.info(
                    "`diff_b` keys found in the state dict which are currently unsupported. "
                    "So, we will filter out those keys. Open an issue if this is a problem - "
                    "https://github.com/huggingface/diffusers/issues/new."
                )
            state_dict = {k: v for k, v in state_dict.items() if ".diff_b" not in k}

        has_norm_diff = any(".norm" in k and ".diff" in k for k in state_dict)
        if has_norm_diff:
            zero_status_diff = state_dict_all_zero(state_dict, ".diff")
            if zero_status_diff:
                logger.info(
                    "The `diff` LoRA params are all zeros which make them ineffective. "
                    "So, we will purge them out of the current state dict to make loading possible."
                )
            else:
                logger.info(
                    "Normalization diff keys found in the state dict which are currently unsupported. "
                    "So, we will filter out those keys. Open an issue if this is a problem - "
                    "https://github.com/huggingface/diffusers/issues/new."
                )
            state_dict = {k: v for k, v in state_dict.items() if ".norm" not in k and ".diff" not in k}

        limit_substrings = ["lora_down", "lora_up"]
        if any("alpha" in k for k in state_dict):
            limit_substrings.append("alpha")

        state_dict = {
            _custom_replace(k, limit_substrings): v
            for k, v in state_dict.items()
            if k.startswith(("lora_unet_", "lora_te_"))
        }

        if any("text_projection" in k for k in state_dict):
            logger.info(
                "`text_projection` keys found in the `state_dict` which are unexpected. "
                "So, we will filter out those keys. Open an issue if this is a problem - "
                "https://github.com/huggingface/diffusers/issues/new."
            )
            state_dict = {k: v for k, v in state_dict.items() if "text_projection" not in k}

    if has_mixture:
        return _convert_mixture_state_dict_to_diffusers(state_dict)

    return _convert_sd_scripts_to_ai_toolkit(state_dict)


# Adapted from https://gist.github.com/Leommm-byte/6b331a1e9bd53271210b26543a7065d6
# Some utilities were reused from
# https://github.com/kohya-ss/sd-scripts/blob/a61cf73a5cb5209c3f4d1a3688dd276a4dfd1ecb/networks/convert_flux_lora.py
def _convert_xlabs_flux_lora_to_diffusers(old_state_dict):
    new_state_dict = {}
    orig_keys = list(old_state_dict.keys())

    def handle_qkv(sds_sd, ait_sd, sds_key, ait_keys, dims=None):
        down_weight = sds_sd.pop(sds_key)
        up_weight = sds_sd.pop(sds_key.replace(".down.weight", ".up.weight"))

        # calculate dims if not provided
        num_splits = len(ait_keys)
        if dims is None:
            dims = [up_weight.shape[0] // num_splits] * num_splits
        else:
            assert sum(dims) == up_weight.shape[0]

        # make ai-toolkit weight
        ait_down_keys = [k + ".lora_A.weight" for k in ait_keys]
        ait_up_keys = [k + ".lora_B.weight" for k in ait_keys]

        # down_weight is copied to each split
        ait_sd.update(dict.fromkeys(ait_down_keys, down_weight))

        # up_weight is split to each split
        ait_sd.update({k: v for k, v in zip(ait_up_keys, torch.split(up_weight, dims, dim=0))})  # noqa: C416

    for old_key in orig_keys:
        # Handle double_blocks
        if old_key.startswith(("diffusion_model.double_blocks", "double_blocks")):
            block_num = re.search(r"double_blocks\.(\d+)", old_key).group(1)
            new_key = f"transformer.transformer_blocks.{block_num}"

            if "processor.proj_lora1" in old_key:
                new_key += ".attn.to_out.0"
            elif "processor.proj_lora2" in old_key:
                new_key += ".attn.to_add_out"
            # Handle text latents.
            elif "processor.qkv_lora2" in old_key and "up" not in old_key:
                handle_qkv(
                    old_state_dict,
                    new_state_dict,
                    old_key,
                    [
                        f"transformer.transformer_blocks.{block_num}.attn.add_q_proj",
                        f"transformer.transformer_blocks.{block_num}.attn.add_k_proj",
                        f"transformer.transformer_blocks.{block_num}.attn.add_v_proj",
                    ],
                )
                # continue
            # Handle image latents.
            elif "processor.qkv_lora1" in old_key and "up" not in old_key:
                handle_qkv(
                    old_state_dict,
                    new_state_dict,
                    old_key,
                    [
                        f"transformer.transformer_blocks.{block_num}.attn.to_q",
                        f"transformer.transformer_blocks.{block_num}.attn.to_k",
                        f"transformer.transformer_blocks.{block_num}.attn.to_v",
                    ],
                )
                # continue

            if "down" in old_key:
                new_key += ".lora_A.weight"
            elif "up" in old_key:
                new_key += ".lora_B.weight"

        # Handle single_blocks
        elif old_key.startswith(("diffusion_model.single_blocks", "single_blocks")):
            block_num = re.search(r"single_blocks\.(\d+)", old_key).group(1)
            new_key = f"transformer.single_transformer_blocks.{block_num}"

            if "proj_lora" in old_key:
                new_key += ".proj_out"
            elif "qkv_lora" in old_key and "up" not in old_key:
                handle_qkv(
                    old_state_dict,
                    new_state_dict,
                    old_key,
                    [
                        f"transformer.single_transformer_blocks.{block_num}.attn.to_q",
                        f"transformer.single_transformer_blocks.{block_num}.attn.to_k",
                        f"transformer.single_transformer_blocks.{block_num}.attn.to_v",
                    ],
                )

            if "down" in old_key:
                new_key += ".lora_A.weight"
            elif "up" in old_key:
                new_key += ".lora_B.weight"

        else:
            # Handle other potential key patterns here
            new_key = old_key

        # Since we already handle qkv above.
        if "qkv" not in old_key:
            new_state_dict[new_key] = old_state_dict.pop(old_key)

    if len(old_state_dict) > 0:
        raise ValueError(f"`old_state_dict` should be at this point but has: {list(old_state_dict.keys())}.")

    return new_state_dict


def _custom_replace(key: str, substrings: List[str]) -> str:
    # Replaces the "."s with "_"s upto the `substrings`.
    # Example:
    # lora_unet.foo.bar.lora_A.weight -> lora_unet_foo_bar.lora_A.weight
    pattern = "(" + "|".join(re.escape(sub) for sub in substrings) + ")"

    match = re.search(pattern, key)
    if match:
        start_sub = match.start()
        if start_sub > 0 and key[start_sub - 1] == ".":
            boundary = start_sub - 1
        else:
            boundary = start_sub
        left = key[:boundary].replace(".", "_")
        right = key[boundary:]
        return left + right
    else:
        return key.replace(".", "_")


def _convert_bfl_flux_control_lora_to_diffusers(original_state_dict):
    converted_state_dict = {}
    original_state_dict_keys = list(original_state_dict.keys())
    num_layers = 19
    num_single_layers = 38
    inner_dim = 3072
    mlp_ratio = 4.0

    for lora_key in ["lora_A", "lora_B"]:
        ## time_text_embed.timestep_embedder <-  time_in
        converted_state_dict[f"time_text_embed.timestep_embedder.linear_1.{lora_key}.weight"] = (
            original_state_dict.pop(f"time_in.in_layer.{lora_key}.weight")
        )
        if f"time_in.in_layer.{lora_key}.bias" in original_state_dict_keys:
            converted_state_dict[f"time_text_embed.timestep_embedder.linear_1.{lora_key}.bias"] = (
                original_state_dict.pop(f"time_in.in_layer.{lora_key}.bias")
            )

        converted_state_dict[f"time_text_embed.timestep_embedder.linear_2.{lora_key}.weight"] = (
            original_state_dict.pop(f"time_in.out_layer.{lora_key}.weight")
        )
        if f"time_in.out_layer.{lora_key}.bias" in original_state_dict_keys:
            converted_state_dict[f"time_text_embed.timestep_embedder.linear_2.{lora_key}.bias"] = (
                original_state_dict.pop(f"time_in.out_layer.{lora_key}.bias")
            )

        ## time_text_embed.text_embedder <- vector_in
        converted_state_dict[f"time_text_embed.text_embedder.linear_1.{lora_key}.weight"] = original_state_dict.pop(
            f"vector_in.in_layer.{lora_key}.weight"
        )
        if f"vector_in.in_layer.{lora_key}.bias" in original_state_dict_keys:
            converted_state_dict[f"time_text_embed.text_embedder.linear_1.{lora_key}.bias"] = original_state_dict.pop(
                f"vector_in.in_layer.{lora_key}.bias"
            )

        converted_state_dict[f"time_text_embed.text_embedder.linear_2.{lora_key}.weight"] = original_state_dict.pop(
            f"vector_in.out_layer.{lora_key}.weight"
        )
        if f"vector_in.out_layer.{lora_key}.bias" in original_state_dict_keys:
            converted_state_dict[f"time_text_embed.text_embedder.linear_2.{lora_key}.bias"] = original_state_dict.pop(
                f"vector_in.out_layer.{lora_key}.bias"
            )

        # guidance
        has_guidance = any("guidance" in k for k in original_state_dict)
        if has_guidance:
            converted_state_dict[f"time_text_embed.guidance_embedder.linear_1.{lora_key}.weight"] = (
                original_state_dict.pop(f"guidance_in.in_layer.{lora_key}.weight")
            )
            if f"guidance_in.in_layer.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"time_text_embed.guidance_embedder.linear_1.{lora_key}.bias"] = (
                    original_state_dict.pop(f"guidance_in.in_layer.{lora_key}.bias")
                )

            converted_state_dict[f"time_text_embed.guidance_embedder.linear_2.{lora_key}.weight"] = (
                original_state_dict.pop(f"guidance_in.out_layer.{lora_key}.weight")
            )
            if f"guidance_in.out_layer.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"time_text_embed.guidance_embedder.linear_2.{lora_key}.bias"] = (
                    original_state_dict.pop(f"guidance_in.out_layer.{lora_key}.bias")
                )

        # context_embedder
        converted_state_dict[f"context_embedder.{lora_key}.weight"] = original_state_dict.pop(
            f"txt_in.{lora_key}.weight"
        )
        if f"txt_in.{lora_key}.bias" in original_state_dict_keys:
            converted_state_dict[f"context_embedder.{lora_key}.bias"] = original_state_dict.pop(
                f"txt_in.{lora_key}.bias"
            )

        # x_embedder
        converted_state_dict[f"x_embedder.{lora_key}.weight"] = original_state_dict.pop(f"img_in.{lora_key}.weight")
        if f"img_in.{lora_key}.bias" in original_state_dict_keys:
            converted_state_dict[f"x_embedder.{lora_key}.bias"] = original_state_dict.pop(f"img_in.{lora_key}.bias")

    # double transformer blocks
    for i in range(num_layers):
        block_prefix = f"transformer_blocks.{i}."

        for lora_key in ["lora_A", "lora_B"]:
            # norms
            converted_state_dict[f"{block_prefix}norm1.linear.{lora_key}.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.img_mod.lin.{lora_key}.weight"
            )
            if f"double_blocks.{i}.img_mod.lin.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"{block_prefix}norm1.linear.{lora_key}.bias"] = original_state_dict.pop(
                    f"double_blocks.{i}.img_mod.lin.{lora_key}.bias"
                )

            converted_state_dict[f"{block_prefix}norm1_context.linear.{lora_key}.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.txt_mod.lin.{lora_key}.weight"
            )
            if f"double_blocks.{i}.txt_mod.lin.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"{block_prefix}norm1_context.linear.{lora_key}.bias"] = original_state_dict.pop(
                    f"double_blocks.{i}.txt_mod.lin.{lora_key}.bias"
                )

            # Q, K, V
            if lora_key == "lora_A":
                sample_lora_weight = original_state_dict.pop(f"double_blocks.{i}.img_attn.qkv.{lora_key}.weight")
                converted_state_dict[f"{block_prefix}attn.to_v.{lora_key}.weight"] = torch.cat([sample_lora_weight])
                converted_state_dict[f"{block_prefix}attn.to_q.{lora_key}.weight"] = torch.cat([sample_lora_weight])
                converted_state_dict[f"{block_prefix}attn.to_k.{lora_key}.weight"] = torch.cat([sample_lora_weight])

                context_lora_weight = original_state_dict.pop(f"double_blocks.{i}.txt_attn.qkv.{lora_key}.weight")
                converted_state_dict[f"{block_prefix}attn.add_q_proj.{lora_key}.weight"] = torch.cat(
                    [context_lora_weight]
                )
                converted_state_dict[f"{block_prefix}attn.add_k_proj.{lora_key}.weight"] = torch.cat(
                    [context_lora_weight]
                )
                converted_state_dict[f"{block_prefix}attn.add_v_proj.{lora_key}.weight"] = torch.cat(
                    [context_lora_weight]
                )
            else:
                sample_q, sample_k, sample_v = torch.chunk(
                    original_state_dict.pop(f"double_blocks.{i}.img_attn.qkv.{lora_key}.weight"), 3, dim=0
                )
                converted_state_dict[f"{block_prefix}attn.to_q.{lora_key}.weight"] = torch.cat([sample_q])
                converted_state_dict[f"{block_prefix}attn.to_k.{lora_key}.weight"] = torch.cat([sample_k])
                converted_state_dict[f"{block_prefix}attn.to_v.{lora_key}.weight"] = torch.cat([sample_v])

                context_q, context_k, context_v = torch.chunk(
                    original_state_dict.pop(f"double_blocks.{i}.txt_attn.qkv.{lora_key}.weight"), 3, dim=0
                )
                converted_state_dict[f"{block_prefix}attn.add_q_proj.{lora_key}.weight"] = torch.cat([context_q])
                converted_state_dict[f"{block_prefix}attn.add_k_proj.{lora_key}.weight"] = torch.cat([context_k])
                converted_state_dict[f"{block_prefix}attn.add_v_proj.{lora_key}.weight"] = torch.cat([context_v])

            if f"double_blocks.{i}.img_attn.qkv.{lora_key}.bias" in original_state_dict_keys:
                sample_q_bias, sample_k_bias, sample_v_bias = torch.chunk(
                    original_state_dict.pop(f"double_blocks.{i}.img_attn.qkv.{lora_key}.bias"), 3, dim=0
                )
                converted_state_dict[f"{block_prefix}attn.to_q.{lora_key}.bias"] = torch.cat([sample_q_bias])
                converted_state_dict[f"{block_prefix}attn.to_k.{lora_key}.bias"] = torch.cat([sample_k_bias])
                converted_state_dict[f"{block_prefix}attn.to_v.{lora_key}.bias"] = torch.cat([sample_v_bias])

            if f"double_blocks.{i}.txt_attn.qkv.{lora_key}.bias" in original_state_dict_keys:
                context_q_bias, context_k_bias, context_v_bias = torch.chunk(
                    original_state_dict.pop(f"double_blocks.{i}.txt_attn.qkv.{lora_key}.bias"), 3, dim=0
                )
                converted_state_dict[f"{block_prefix}attn.add_q_proj.{lora_key}.bias"] = torch.cat([context_q_bias])
                converted_state_dict[f"{block_prefix}attn.add_k_proj.{lora_key}.bias"] = torch.cat([context_k_bias])
                converted_state_dict[f"{block_prefix}attn.add_v_proj.{lora_key}.bias"] = torch.cat([context_v_bias])

            # ff img_mlp
            converted_state_dict[f"{block_prefix}ff.net.0.proj.{lora_key}.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.img_mlp.0.{lora_key}.weight"
            )
            if f"double_blocks.{i}.img_mlp.0.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"{block_prefix}ff.net.0.proj.{lora_key}.bias"] = original_state_dict.pop(
                    f"double_blocks.{i}.img_mlp.0.{lora_key}.bias"
                )

            converted_state_dict[f"{block_prefix}ff.net.2.{lora_key}.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.img_mlp.2.{lora_key}.weight"
            )
            if f"double_blocks.{i}.img_mlp.2.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"{block_prefix}ff.net.2.{lora_key}.bias"] = original_state_dict.pop(
                    f"double_blocks.{i}.img_mlp.2.{lora_key}.bias"
                )

            converted_state_dict[f"{block_prefix}ff_context.net.0.proj.{lora_key}.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.txt_mlp.0.{lora_key}.weight"
            )
            if f"double_blocks.{i}.txt_mlp.0.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"{block_prefix}ff_context.net.0.proj.{lora_key}.bias"] = original_state_dict.pop(
                    f"double_blocks.{i}.txt_mlp.0.{lora_key}.bias"
                )

            converted_state_dict[f"{block_prefix}ff_context.net.2.{lora_key}.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.txt_mlp.2.{lora_key}.weight"
            )
            if f"double_blocks.{i}.txt_mlp.2.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"{block_prefix}ff_context.net.2.{lora_key}.bias"] = original_state_dict.pop(
                    f"double_blocks.{i}.txt_mlp.2.{lora_key}.bias"
                )

            # output projections.
            converted_state_dict[f"{block_prefix}attn.to_out.0.{lora_key}.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.img_attn.proj.{lora_key}.weight"
            )
            if f"double_blocks.{i}.img_attn.proj.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"{block_prefix}attn.to_out.0.{lora_key}.bias"] = original_state_dict.pop(
                    f"double_blocks.{i}.img_attn.proj.{lora_key}.bias"
                )
            converted_state_dict[f"{block_prefix}attn.to_add_out.{lora_key}.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.txt_attn.proj.{lora_key}.weight"
            )
            if f"double_blocks.{i}.txt_attn.proj.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"{block_prefix}attn.to_add_out.{lora_key}.bias"] = original_state_dict.pop(
                    f"double_blocks.{i}.txt_attn.proj.{lora_key}.bias"
                )

        # qk_norm
        converted_state_dict[f"{block_prefix}attn.norm_q.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_attn.norm.query_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_k.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_attn.norm.key_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_added_q.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_attn.norm.query_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_added_k.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_attn.norm.key_norm.scale"
        )

    # single transformer blocks
    for i in range(num_single_layers):
        block_prefix = f"single_transformer_blocks.{i}."

        for lora_key in ["lora_A", "lora_B"]:
            # norm.linear  <- single_blocks.0.modulation.lin
            converted_state_dict[f"{block_prefix}norm.linear.{lora_key}.weight"] = original_state_dict.pop(
                f"single_blocks.{i}.modulation.lin.{lora_key}.weight"
            )
            if f"single_blocks.{i}.modulation.lin.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"{block_prefix}norm.linear.{lora_key}.bias"] = original_state_dict.pop(
                    f"single_blocks.{i}.modulation.lin.{lora_key}.bias"
                )

            # Q, K, V, mlp
            mlp_hidden_dim = int(inner_dim * mlp_ratio)
            split_size = (inner_dim, inner_dim, inner_dim, mlp_hidden_dim)

            if lora_key == "lora_A":
                lora_weight = original_state_dict.pop(f"single_blocks.{i}.linear1.{lora_key}.weight")
                converted_state_dict[f"{block_prefix}attn.to_q.{lora_key}.weight"] = torch.cat([lora_weight])
                converted_state_dict[f"{block_prefix}attn.to_k.{lora_key}.weight"] = torch.cat([lora_weight])
                converted_state_dict[f"{block_prefix}attn.to_v.{lora_key}.weight"] = torch.cat([lora_weight])
                converted_state_dict[f"{block_prefix}proj_mlp.{lora_key}.weight"] = torch.cat([lora_weight])

                if f"single_blocks.{i}.linear1.{lora_key}.bias" in original_state_dict_keys:
                    lora_bias = original_state_dict.pop(f"single_blocks.{i}.linear1.{lora_key}.bias")
                    converted_state_dict[f"{block_prefix}attn.to_q.{lora_key}.bias"] = torch.cat([lora_bias])
                    converted_state_dict[f"{block_prefix}attn.to_k.{lora_key}.bias"] = torch.cat([lora_bias])
                    converted_state_dict[f"{block_prefix}attn.to_v.{lora_key}.bias"] = torch.cat([lora_bias])
                    converted_state_dict[f"{block_prefix}proj_mlp.{lora_key}.bias"] = torch.cat([lora_bias])
            else:
                q, k, v, mlp = torch.split(
                    original_state_dict.pop(f"single_blocks.{i}.linear1.{lora_key}.weight"), split_size, dim=0
                )
                converted_state_dict[f"{block_prefix}attn.to_q.{lora_key}.weight"] = torch.cat([q])
                converted_state_dict[f"{block_prefix}attn.to_k.{lora_key}.weight"] = torch.cat([k])
                converted_state_dict[f"{block_prefix}attn.to_v.{lora_key}.weight"] = torch.cat([v])
                converted_state_dict[f"{block_prefix}proj_mlp.{lora_key}.weight"] = torch.cat([mlp])

                if f"single_blocks.{i}.linear1.{lora_key}.bias" in original_state_dict_keys:
                    q_bias, k_bias, v_bias, mlp_bias = torch.split(
                        original_state_dict.pop(f"single_blocks.{i}.linear1.{lora_key}.bias"), split_size, dim=0
                    )
                    converted_state_dict[f"{block_prefix}attn.to_q.{lora_key}.bias"] = torch.cat([q_bias])
                    converted_state_dict[f"{block_prefix}attn.to_k.{lora_key}.bias"] = torch.cat([k_bias])
                    converted_state_dict[f"{block_prefix}attn.to_v.{lora_key}.bias"] = torch.cat([v_bias])
                    converted_state_dict[f"{block_prefix}proj_mlp.{lora_key}.bias"] = torch.cat([mlp_bias])

            # output projections.
            converted_state_dict[f"{block_prefix}proj_out.{lora_key}.weight"] = original_state_dict.pop(
                f"single_blocks.{i}.linear2.{lora_key}.weight"
            )
            if f"single_blocks.{i}.linear2.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"{block_prefix}proj_out.{lora_key}.bias"] = original_state_dict.pop(
                    f"single_blocks.{i}.linear2.{lora_key}.bias"
                )

        # qk norm
        converted_state_dict[f"{block_prefix}attn.norm_q.weight"] = original_state_dict.pop(
            f"single_blocks.{i}.norm.query_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_k.weight"] = original_state_dict.pop(
            f"single_blocks.{i}.norm.key_norm.scale"
        )

    for lora_key in ["lora_A", "lora_B"]:
        converted_state_dict[f"proj_out.{lora_key}.weight"] = original_state_dict.pop(
            f"final_layer.linear.{lora_key}.weight"
        )
        if f"final_layer.linear.{lora_key}.bias" in original_state_dict_keys:
            converted_state_dict[f"proj_out.{lora_key}.bias"] = original_state_dict.pop(
                f"final_layer.linear.{lora_key}.bias"
            )

        converted_state_dict[f"norm_out.linear.{lora_key}.weight"] = swap_scale_shift(
            original_state_dict.pop(f"final_layer.adaLN_modulation.1.{lora_key}.weight")
        )
        if f"final_layer.adaLN_modulation.1.{lora_key}.bias" in original_state_dict_keys:
            converted_state_dict[f"norm_out.linear.{lora_key}.bias"] = swap_scale_shift(
                original_state_dict.pop(f"final_layer.adaLN_modulation.1.{lora_key}.bias")
            )

    if len(original_state_dict) > 0:
        raise ValueError(f"`original_state_dict` should be empty at this point but has {original_state_dict.keys()=}.")

    for key in list(converted_state_dict.keys()):
        converted_state_dict[f"transformer.{key}"] = converted_state_dict.pop(key)

    return converted_state_dict


def _convert_fal_kontext_lora_to_diffusers(original_state_dict):
    converted_state_dict = {}
    original_state_dict_keys = list(original_state_dict.keys())
    num_layers = 19
    num_single_layers = 38
    inner_dim = 3072
    mlp_ratio = 4.0

    # double transformer blocks
    for i in range(num_layers):
        block_prefix = f"transformer_blocks.{i}."
        original_block_prefix = "base_model.model."

        for lora_key in ["lora_A", "lora_B"]:
            # norms
            converted_state_dict[f"{block_prefix}norm1.linear.{lora_key}.weight"] = original_state_dict.pop(
                f"{original_block_prefix}double_blocks.{i}.img_mod.lin.{lora_key}.weight"
            )
            if f"double_blocks.{i}.img_mod.lin.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"{block_prefix}norm1.linear.{lora_key}.bias"] = original_state_dict.pop(
                    f"{original_block_prefix}double_blocks.{i}.img_mod.lin.{lora_key}.bias"
                )

            converted_state_dict[f"{block_prefix}norm1_context.linear.{lora_key}.weight"] = original_state_dict.pop(
                f"{original_block_prefix}double_blocks.{i}.txt_mod.lin.{lora_key}.weight"
            )

            # Q, K, V
            if lora_key == "lora_A":
                sample_lora_weight = original_state_dict.pop(
                    f"{original_block_prefix}double_blocks.{i}.img_attn.qkv.{lora_key}.weight"
                )
                converted_state_dict[f"{block_prefix}attn.to_v.{lora_key}.weight"] = torch.cat([sample_lora_weight])
                converted_state_dict[f"{block_prefix}attn.to_q.{lora_key}.weight"] = torch.cat([sample_lora_weight])
                converted_state_dict[f"{block_prefix}attn.to_k.{lora_key}.weight"] = torch.cat([sample_lora_weight])

                context_lora_weight = original_state_dict.pop(
                    f"{original_block_prefix}double_blocks.{i}.txt_attn.qkv.{lora_key}.weight"
                )
                converted_state_dict[f"{block_prefix}attn.add_q_proj.{lora_key}.weight"] = torch.cat(
                    [context_lora_weight]
                )
                converted_state_dict[f"{block_prefix}attn.add_k_proj.{lora_key}.weight"] = torch.cat(
                    [context_lora_weight]
                )
                converted_state_dict[f"{block_prefix}attn.add_v_proj.{lora_key}.weight"] = torch.cat(
                    [context_lora_weight]
                )
            else:
                sample_q, sample_k, sample_v = torch.chunk(
                    original_state_dict.pop(
                        f"{original_block_prefix}double_blocks.{i}.img_attn.qkv.{lora_key}.weight"
                    ),
                    3,
                    dim=0,
                )
                converted_state_dict[f"{block_prefix}attn.to_q.{lora_key}.weight"] = torch.cat([sample_q])
                converted_state_dict[f"{block_prefix}attn.to_k.{lora_key}.weight"] = torch.cat([sample_k])
                converted_state_dict[f"{block_prefix}attn.to_v.{lora_key}.weight"] = torch.cat([sample_v])

                context_q, context_k, context_v = torch.chunk(
                    original_state_dict.pop(
                        f"{original_block_prefix}double_blocks.{i}.txt_attn.qkv.{lora_key}.weight"
                    ),
                    3,
                    dim=0,
                )
                converted_state_dict[f"{block_prefix}attn.add_q_proj.{lora_key}.weight"] = torch.cat([context_q])
                converted_state_dict[f"{block_prefix}attn.add_k_proj.{lora_key}.weight"] = torch.cat([context_k])
                converted_state_dict[f"{block_prefix}attn.add_v_proj.{lora_key}.weight"] = torch.cat([context_v])

            if f"double_blocks.{i}.img_attn.qkv.{lora_key}.bias" in original_state_dict_keys:
                sample_q_bias, sample_k_bias, sample_v_bias = torch.chunk(
                    original_state_dict.pop(f"{original_block_prefix}double_blocks.{i}.img_attn.qkv.{lora_key}.bias"),
                    3,
                    dim=0,
                )
                converted_state_dict[f"{block_prefix}attn.to_q.{lora_key}.bias"] = torch.cat([sample_q_bias])
                converted_state_dict[f"{block_prefix}attn.to_k.{lora_key}.bias"] = torch.cat([sample_k_bias])
                converted_state_dict[f"{block_prefix}attn.to_v.{lora_key}.bias"] = torch.cat([sample_v_bias])

            if f"double_blocks.{i}.txt_attn.qkv.{lora_key}.bias" in original_state_dict_keys:
                context_q_bias, context_k_bias, context_v_bias = torch.chunk(
                    original_state_dict.pop(f"{original_block_prefix}double_blocks.{i}.txt_attn.qkv.{lora_key}.bias"),
                    3,
                    dim=0,
                )
                converted_state_dict[f"{block_prefix}attn.add_q_proj.{lora_key}.bias"] = torch.cat([context_q_bias])
                converted_state_dict[f"{block_prefix}attn.add_k_proj.{lora_key}.bias"] = torch.cat([context_k_bias])
                converted_state_dict[f"{block_prefix}attn.add_v_proj.{lora_key}.bias"] = torch.cat([context_v_bias])

            # ff img_mlp
            converted_state_dict[f"{block_prefix}ff.net.0.proj.{lora_key}.weight"] = original_state_dict.pop(
                f"{original_block_prefix}double_blocks.{i}.img_mlp.0.{lora_key}.weight"
            )
            if f"{original_block_prefix}double_blocks.{i}.img_mlp.0.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"{block_prefix}ff.net.0.proj.{lora_key}.bias"] = original_state_dict.pop(
                    f"{original_block_prefix}double_blocks.{i}.img_mlp.0.{lora_key}.bias"
                )

            converted_state_dict[f"{block_prefix}ff.net.2.{lora_key}.weight"] = original_state_dict.pop(
                f"{original_block_prefix}double_blocks.{i}.img_mlp.2.{lora_key}.weight"
            )
            if f"{original_block_prefix}double_blocks.{i}.img_mlp.2.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"{block_prefix}ff.net.2.{lora_key}.bias"] = original_state_dict.pop(
                    f"{original_block_prefix}double_blocks.{i}.img_mlp.2.{lora_key}.bias"
                )

            converted_state_dict[f"{block_prefix}ff_context.net.0.proj.{lora_key}.weight"] = original_state_dict.pop(
                f"{original_block_prefix}double_blocks.{i}.txt_mlp.0.{lora_key}.weight"
            )
            if f"{original_block_prefix}double_blocks.{i}.txt_mlp.0.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"{block_prefix}ff_context.net.0.proj.{lora_key}.bias"] = original_state_dict.pop(
                    f"{original_block_prefix}double_blocks.{i}.txt_mlp.0.{lora_key}.bias"
                )

            converted_state_dict[f"{block_prefix}ff_context.net.2.{lora_key}.weight"] = original_state_dict.pop(
                f"{original_block_prefix}double_blocks.{i}.txt_mlp.2.{lora_key}.weight"
            )
            if f"{original_block_prefix}double_blocks.{i}.txt_mlp.2.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"{block_prefix}ff_context.net.2.{lora_key}.bias"] = original_state_dict.pop(
                    f"{original_block_prefix}double_blocks.{i}.txt_mlp.2.{lora_key}.bias"
                )

            # output projections.
            converted_state_dict[f"{block_prefix}attn.to_out.0.{lora_key}.weight"] = original_state_dict.pop(
                f"{original_block_prefix}double_blocks.{i}.img_attn.proj.{lora_key}.weight"
            )
            if f"{original_block_prefix}double_blocks.{i}.img_attn.proj.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"{block_prefix}attn.to_out.0.{lora_key}.bias"] = original_state_dict.pop(
                    f"{original_block_prefix}double_blocks.{i}.img_attn.proj.{lora_key}.bias"
                )
            converted_state_dict[f"{block_prefix}attn.to_add_out.{lora_key}.weight"] = original_state_dict.pop(
                f"{original_block_prefix}double_blocks.{i}.txt_attn.proj.{lora_key}.weight"
            )
            if f"{original_block_prefix}double_blocks.{i}.txt_attn.proj.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"{block_prefix}attn.to_add_out.{lora_key}.bias"] = original_state_dict.pop(
                    f"{original_block_prefix}double_blocks.{i}.txt_attn.proj.{lora_key}.bias"
                )

    # single transformer blocks
    for i in range(num_single_layers):
        block_prefix = f"single_transformer_blocks.{i}."

        for lora_key in ["lora_A", "lora_B"]:
            # norm.linear  <- single_blocks.0.modulation.lin
            converted_state_dict[f"{block_prefix}norm.linear.{lora_key}.weight"] = original_state_dict.pop(
                f"{original_block_prefix}single_blocks.{i}.modulation.lin.{lora_key}.weight"
            )
            if f"{original_block_prefix}single_blocks.{i}.modulation.lin.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"{block_prefix}norm.linear.{lora_key}.bias"] = original_state_dict.pop(
                    f"{original_block_prefix}single_blocks.{i}.modulation.lin.{lora_key}.bias"
                )

            # Q, K, V, mlp
            mlp_hidden_dim = int(inner_dim * mlp_ratio)
            split_size = (inner_dim, inner_dim, inner_dim, mlp_hidden_dim)

            if lora_key == "lora_A":
                lora_weight = original_state_dict.pop(
                    f"{original_block_prefix}single_blocks.{i}.linear1.{lora_key}.weight"
                )
                converted_state_dict[f"{block_prefix}attn.to_q.{lora_key}.weight"] = torch.cat([lora_weight])
                converted_state_dict[f"{block_prefix}attn.to_k.{lora_key}.weight"] = torch.cat([lora_weight])
                converted_state_dict[f"{block_prefix}attn.to_v.{lora_key}.weight"] = torch.cat([lora_weight])
                converted_state_dict[f"{block_prefix}proj_mlp.{lora_key}.weight"] = torch.cat([lora_weight])

                if f"{original_block_prefix}single_blocks.{i}.linear1.{lora_key}.bias" in original_state_dict_keys:
                    lora_bias = original_state_dict.pop(f"single_blocks.{i}.linear1.{lora_key}.bias")
                    converted_state_dict[f"{block_prefix}attn.to_q.{lora_key}.bias"] = torch.cat([lora_bias])
                    converted_state_dict[f"{block_prefix}attn.to_k.{lora_key}.bias"] = torch.cat([lora_bias])
                    converted_state_dict[f"{block_prefix}attn.to_v.{lora_key}.bias"] = torch.cat([lora_bias])
                    converted_state_dict[f"{block_prefix}proj_mlp.{lora_key}.bias"] = torch.cat([lora_bias])
            else:
                q, k, v, mlp = torch.split(
                    original_state_dict.pop(f"{original_block_prefix}single_blocks.{i}.linear1.{lora_key}.weight"),
                    split_size,
                    dim=0,
                )
                converted_state_dict[f"{block_prefix}attn.to_q.{lora_key}.weight"] = torch.cat([q])
                converted_state_dict[f"{block_prefix}attn.to_k.{lora_key}.weight"] = torch.cat([k])
                converted_state_dict[f"{block_prefix}attn.to_v.{lora_key}.weight"] = torch.cat([v])
                converted_state_dict[f"{block_prefix}proj_mlp.{lora_key}.weight"] = torch.cat([mlp])

                if f"{original_block_prefix}single_blocks.{i}.linear1.{lora_key}.bias" in original_state_dict_keys:
                    q_bias, k_bias, v_bias, mlp_bias = torch.split(
                        original_state_dict.pop(f"{original_block_prefix}single_blocks.{i}.linear1.{lora_key}.bias"),
                        split_size,
                        dim=0,
                    )
                    converted_state_dict[f"{block_prefix}attn.to_q.{lora_key}.bias"] = torch.cat([q_bias])
                    converted_state_dict[f"{block_prefix}attn.to_k.{lora_key}.bias"] = torch.cat([k_bias])
                    converted_state_dict[f"{block_prefix}attn.to_v.{lora_key}.bias"] = torch.cat([v_bias])
                    converted_state_dict[f"{block_prefix}proj_mlp.{lora_key}.bias"] = torch.cat([mlp_bias])

            # output projections.
            converted_state_dict[f"{block_prefix}proj_out.{lora_key}.weight"] = original_state_dict.pop(
                f"{original_block_prefix}single_blocks.{i}.linear2.{lora_key}.weight"
            )
            if f"{original_block_prefix}single_blocks.{i}.linear2.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"{block_prefix}proj_out.{lora_key}.bias"] = original_state_dict.pop(
                    f"{original_block_prefix}single_blocks.{i}.linear2.{lora_key}.bias"
                )

    for lora_key in ["lora_A", "lora_B"]:
        converted_state_dict[f"proj_out.{lora_key}.weight"] = original_state_dict.pop(
            f"{original_block_prefix}final_layer.linear.{lora_key}.weight"
        )
        if f"{original_block_prefix}final_layer.linear.{lora_key}.bias" in original_state_dict_keys:
            converted_state_dict[f"proj_out.{lora_key}.bias"] = original_state_dict.pop(
                f"{original_block_prefix}final_layer.linear.{lora_key}.bias"
            )

    if len(original_state_dict) > 0:
        raise ValueError(f"`original_state_dict` should be empty at this point but has {original_state_dict.keys()=}.")

    for key in list(converted_state_dict.keys()):
        converted_state_dict[f"transformer.{key}"] = converted_state_dict.pop(key)

    return converted_state_dict


def _convert_hunyuan_video_lora_to_diffusers(original_state_dict):
    converted_state_dict = {k: original_state_dict.pop(k) for k in list(original_state_dict.keys())}

    def remap_norm_scale_shift_(key, state_dict):
        weight = state_dict.pop(key)
        shift, scale = weight.chunk(2, dim=0)
        new_weight = torch.cat([scale, shift], dim=0)
        state_dict[key.replace("final_layer.adaLN_modulation.1", "norm_out.linear")] = new_weight

    def remap_txt_in_(key, state_dict):
        def rename_key(key):
            new_key = key.replace("individual_token_refiner.blocks", "token_refiner.refiner_blocks")
            new_key = new_key.replace("adaLN_modulation.1", "norm_out.linear")
            new_key = new_key.replace("txt_in", "context_embedder")
            new_key = new_key.replace("t_embedder.mlp.0", "time_text_embed.timestep_embedder.linear_1")
            new_key = new_key.replace("t_embedder.mlp.2", "time_text_embed.timestep_embedder.linear_2")
            new_key = new_key.replace("c_embedder", "time_text_embed.text_embedder")
            new_key = new_key.replace("mlp", "ff")
            return new_key

        if "self_attn_qkv" in key:
            weight = state_dict.pop(key)
            to_q, to_k, to_v = weight.chunk(3, dim=0)
            state_dict[rename_key(key.replace("self_attn_qkv", "attn.to_q"))] = to_q
            state_dict[rename_key(key.replace("self_attn_qkv", "attn.to_k"))] = to_k
            state_dict[rename_key(key.replace("self_attn_qkv", "attn.to_v"))] = to_v
        else:
            state_dict[rename_key(key)] = state_dict.pop(key)

    def remap_img_attn_qkv_(key, state_dict):
        weight = state_dict.pop(key)
        if "lora_A" in key:
            state_dict[key.replace("img_attn_qkv", "attn.to_q")] = weight
            state_dict[key.replace("img_attn_qkv", "attn.to_k")] = weight
            state_dict[key.replace("img_attn_qkv", "attn.to_v")] = weight
        else:
            to_q, to_k, to_v = weight.chunk(3, dim=0)
            state_dict[key.replace("img_attn_qkv", "attn.to_q")] = to_q
            state_dict[key.replace("img_attn_qkv", "attn.to_k")] = to_k
            state_dict[key.replace("img_attn_qkv", "attn.to_v")] = to_v

    def remap_txt_attn_qkv_(key, state_dict):
        weight = state_dict.pop(key)
        if "lora_A" in key:
            state_dict[key.replace("txt_attn_qkv", "attn.add_q_proj")] = weight
            state_dict[key.replace("txt_attn_qkv", "attn.add_k_proj")] = weight
            state_dict[key.replace("txt_attn_qkv", "attn.add_v_proj")] = weight
        else:
            to_q, to_k, to_v = weight.chunk(3, dim=0)
            state_dict[key.replace("txt_attn_qkv", "attn.add_q_proj")] = to_q
            state_dict[key.replace("txt_attn_qkv", "attn.add_k_proj")] = to_k
            state_dict[key.replace("txt_attn_qkv", "attn.add_v_proj")] = to_v

    def remap_single_transformer_blocks_(key, state_dict):
        hidden_size = 3072

        if "linear1.lora_A.weight" in key or "linear1.lora_B.weight" in key:
            linear1_weight = state_dict.pop(key)
            if "lora_A" in key:
                new_key = key.replace("single_blocks", "single_transformer_blocks").removesuffix(
                    ".linear1.lora_A.weight"
                )
                state_dict[f"{new_key}.attn.to_q.lora_A.weight"] = linear1_weight
                state_dict[f"{new_key}.attn.to_k.lora_A.weight"] = linear1_weight
                state_dict[f"{new_key}.attn.to_v.lora_A.weight"] = linear1_weight
                state_dict[f"{new_key}.proj_mlp.lora_A.weight"] = linear1_weight
            else:
                split_size = (hidden_size, hidden_size, hidden_size, linear1_weight.size(0) - 3 * hidden_size)
                q, k, v, mlp = torch.split(linear1_weight, split_size, dim=0)
                new_key = key.replace("single_blocks", "single_transformer_blocks").removesuffix(
                    ".linear1.lora_B.weight"
                )
                state_dict[f"{new_key}.attn.to_q.lora_B.weight"] = q
                state_dict[f"{new_key}.attn.to_k.lora_B.weight"] = k
                state_dict[f"{new_key}.attn.to_v.lora_B.weight"] = v
                state_dict[f"{new_key}.proj_mlp.lora_B.weight"] = mlp

        elif "linear1.lora_A.bias" in key or "linear1.lora_B.bias" in key:
            linear1_bias = state_dict.pop(key)
            if "lora_A" in key:
                new_key = key.replace("single_blocks", "single_transformer_blocks").removesuffix(
                    ".linear1.lora_A.bias"
                )
                state_dict[f"{new_key}.attn.to_q.lora_A.bias"] = linear1_bias
                state_dict[f"{new_key}.attn.to_k.lora_A.bias"] = linear1_bias
                state_dict[f"{new_key}.attn.to_v.lora_A.bias"] = linear1_bias
                state_dict[f"{new_key}.proj_mlp.lora_A.bias"] = linear1_bias
            else:
                split_size = (hidden_size, hidden_size, hidden_size, linear1_bias.size(0) - 3 * hidden_size)
                q_bias, k_bias, v_bias, mlp_bias = torch.split(linear1_bias, split_size, dim=0)
                new_key = key.replace("single_blocks", "single_transformer_blocks").removesuffix(
                    ".linear1.lora_B.bias"
                )
                state_dict[f"{new_key}.attn.to_q.lora_B.bias"] = q_bias
                state_dict[f"{new_key}.attn.to_k.lora_B.bias"] = k_bias
                state_dict[f"{new_key}.attn.to_v.lora_B.bias"] = v_bias
                state_dict[f"{new_key}.proj_mlp.lora_B.bias"] = mlp_bias

        else:
            new_key = key.replace("single_blocks", "single_transformer_blocks")
            new_key = new_key.replace("linear2", "proj_out")
            new_key = new_key.replace("q_norm", "attn.norm_q")
            new_key = new_key.replace("k_norm", "attn.norm_k")
            state_dict[new_key] = state_dict.pop(key)

    TRANSFORMER_KEYS_RENAME_DICT = {
        "img_in": "x_embedder",
        "time_in.mlp.0": "time_text_embed.timestep_embedder.linear_1",
        "time_in.mlp.2": "time_text_embed.timestep_embedder.linear_2",
        "guidance_in.mlp.0": "time_text_embed.guidance_embedder.linear_1",
        "guidance_in.mlp.2": "time_text_embed.guidance_embedder.linear_2",
        "vector_in.in_layer": "time_text_embed.text_embedder.linear_1",
        "vector_in.out_layer": "time_text_embed.text_embedder.linear_2",
        "double_blocks": "transformer_blocks",
        "img_attn_q_norm": "attn.norm_q",
        "img_attn_k_norm": "attn.norm_k",
        "img_attn_proj": "attn.to_out.0",
        "txt_attn_q_norm": "attn.norm_added_q",
        "txt_attn_k_norm": "attn.norm_added_k",
        "txt_attn_proj": "attn.to_add_out",
        "img_mod.linear": "norm1.linear",
        "img_norm1": "norm1.norm",
        "img_norm2": "norm2",
        "img_mlp": "ff",
        "txt_mod.linear": "norm1_context.linear",
        "txt_norm1": "norm1.norm",
        "txt_norm2": "norm2_context",
        "txt_mlp": "ff_context",
        "self_attn_proj": "attn.to_out.0",
        "modulation.linear": "norm.linear",
        "pre_norm": "norm.norm",
        "final_layer.norm_final": "norm_out.norm",
        "final_layer.linear": "proj_out",
        "fc1": "net.0.proj",
        "fc2": "net.2",
        "input_embedder": "proj_in",
    }

    TRANSFORMER_SPECIAL_KEYS_REMAP = {
        "txt_in": remap_txt_in_,
        "img_attn_qkv": remap_img_attn_qkv_,
        "txt_attn_qkv": remap_txt_attn_qkv_,
        "single_blocks": remap_single_transformer_blocks_,
        "final_layer.adaLN_modulation.1": remap_norm_scale_shift_,
    }

    # Some folks attempt to make their state dict compatible with diffusers by adding "transformer." prefix to all keys
    # and use their custom code. To make sure both "original" and "attempted diffusers" loras work as expected, we make
    # sure that both follow the same initial format by stripping off the "transformer." prefix.
    for key in list(converted_state_dict.keys()):
        if key.startswith("transformer."):
            converted_state_dict[key[len("transformer.") :]] = converted_state_dict.pop(key)
        if key.startswith("diffusion_model."):
            converted_state_dict[key[len("diffusion_model.") :]] = converted_state_dict.pop(key)

    # Rename and remap the state dict keys
    for key in list(converted_state_dict.keys()):
        new_key = key[:]
        for replace_key, rename_key in TRANSFORMER_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)
        converted_state_dict[new_key] = converted_state_dict.pop(key)

    for key in list(converted_state_dict.keys()):
        for special_key, handler_fn_inplace in TRANSFORMER_SPECIAL_KEYS_REMAP.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, converted_state_dict)

    # Add back the "transformer." prefix
    for key in list(converted_state_dict.keys()):
        converted_state_dict[f"transformer.{key}"] = converted_state_dict.pop(key)

    return converted_state_dict


def _convert_non_diffusers_lumina2_lora_to_diffusers(state_dict):
    # Remove "diffusion_model." prefix from keys.
    state_dict = {k[len("diffusion_model.") :]: v for k, v in state_dict.items()}
    converted_state_dict = {}

    def get_num_layers(keys, pattern):
        layers = set()
        for key in keys:
            match = re.search(pattern, key)
            if match:
                layers.add(int(match.group(1)))
        return len(layers)

    def process_block(prefix, index, convert_norm):
        # Process attention qkv: pop lora_A and lora_B weights.
        lora_down = state_dict.pop(f"{prefix}.{index}.attention.qkv.lora_A.weight")
        lora_up = state_dict.pop(f"{prefix}.{index}.attention.qkv.lora_B.weight")
        for attn_key in ["to_q", "to_k", "to_v"]:
            converted_state_dict[f"{prefix}.{index}.attn.{attn_key}.lora_A.weight"] = lora_down
        for attn_key, weight in zip(["to_q", "to_k", "to_v"], torch.split(lora_up, [2304, 768, 768], dim=0)):
            converted_state_dict[f"{prefix}.{index}.attn.{attn_key}.lora_B.weight"] = weight

        # Process attention out weights.
        converted_state_dict[f"{prefix}.{index}.attn.to_out.0.lora_A.weight"] = state_dict.pop(
            f"{prefix}.{index}.attention.out.lora_A.weight"
        )
        converted_state_dict[f"{prefix}.{index}.attn.to_out.0.lora_B.weight"] = state_dict.pop(
            f"{prefix}.{index}.attention.out.lora_B.weight"
        )

        # Process feed-forward weights for layers 1, 2, and 3.
        for layer in range(1, 4):
            converted_state_dict[f"{prefix}.{index}.feed_forward.linear_{layer}.lora_A.weight"] = state_dict.pop(
                f"{prefix}.{index}.feed_forward.w{layer}.lora_A.weight"
            )
            converted_state_dict[f"{prefix}.{index}.feed_forward.linear_{layer}.lora_B.weight"] = state_dict.pop(
                f"{prefix}.{index}.feed_forward.w{layer}.lora_B.weight"
            )

        if convert_norm:
            converted_state_dict[f"{prefix}.{index}.norm1.linear.lora_A.weight"] = state_dict.pop(
                f"{prefix}.{index}.adaLN_modulation.1.lora_A.weight"
            )
            converted_state_dict[f"{prefix}.{index}.norm1.linear.lora_B.weight"] = state_dict.pop(
                f"{prefix}.{index}.adaLN_modulation.1.lora_B.weight"
            )

    noise_refiner_pattern = r"noise_refiner\.(\d+)\."
    num_noise_refiner_layers = get_num_layers(state_dict.keys(), noise_refiner_pattern)
    for i in range(num_noise_refiner_layers):
        process_block("noise_refiner", i, convert_norm=True)

    context_refiner_pattern = r"context_refiner\.(\d+)\."
    num_context_refiner_layers = get_num_layers(state_dict.keys(), context_refiner_pattern)
    for i in range(num_context_refiner_layers):
        process_block("context_refiner", i, convert_norm=False)

    core_transformer_pattern = r"layers\.(\d+)\."
    num_core_transformer_layers = get_num_layers(state_dict.keys(), core_transformer_pattern)
    for i in range(num_core_transformer_layers):
        process_block("layers", i, convert_norm=True)

    if len(state_dict) > 0:
        raise ValueError(f"`state_dict` should be empty at this point but has {state_dict.keys()=}")

    for key in list(converted_state_dict.keys()):
        converted_state_dict[f"transformer.{key}"] = converted_state_dict.pop(key)

    return converted_state_dict


def _convert_non_diffusers_wan_lora_to_diffusers(state_dict):
    converted_state_dict = {}
    original_state_dict = {k[len("diffusion_model.") :]: v for k, v in state_dict.items()}

    block_numbers = {int(k.split(".")[1]) for k in original_state_dict if k.startswith("blocks.")}
    min_block = min(block_numbers)
    max_block = max(block_numbers)

    is_i2v_lora = any("k_img" in k for k in original_state_dict) and any("v_img" in k for k in original_state_dict)
    lora_down_key = "lora_A" if any("lora_A" in k for k in original_state_dict) else "lora_down"
    lora_up_key = "lora_B" if any("lora_B" in k for k in original_state_dict) else "lora_up"
    has_time_projection_weight = any(
        k.startswith("time_projection") and k.endswith(".weight") for k in original_state_dict
    )

    def get_alpha_scales(down_weight, alpha_key):
        rank = down_weight.shape[0]
        alpha = original_state_dict.pop(alpha_key).item()
        scale = alpha / rank  # LoRA is scaled by 'alpha / rank' in forward pass, so we need to scale it back here
        scale_down = scale
        scale_up = 1.0
        while scale_down * 2 < scale_up:
            scale_down *= 2
            scale_up /= 2
        return scale_down, scale_up

    for key in list(original_state_dict.keys()):
        if key.endswith((".diff", ".diff_b")) and "norm" in key:
            # NOTE: we don't support this because norm layer diff keys are just zeroed values. We can support it
            # in future if needed and they are not zeroed.
            original_state_dict.pop(key)
            logger.debug(f"Removing {key} key from the state dict as it is a norm diff key. This is unsupported.")

        if "time_projection" in key and not has_time_projection_weight:
            # AccVideo lora has diff bias keys but not the weight keys. This causes a weird problem where
            # our lora config adds the time proj lora layers, but we don't have the weights for them.
            # CausVid lora has the weight keys and the bias keys.
            original_state_dict.pop(key)

    # For the `diff_b` keys, we treat them as lora_bias.
    # https://huggingface.co/docs/peft/main/en/package_reference/lora#peft.LoraConfig.lora_bias

    for i in range(min_block, max_block + 1):
        # Self-attention
        for o, c in zip(["q", "k", "v", "o"], ["to_q", "to_k", "to_v", "to_out.0"]):
            alpha_key = f"blocks.{i}.self_attn.{o}.alpha"
            has_alpha = alpha_key in original_state_dict
            original_key_A = f"blocks.{i}.self_attn.{o}.{lora_down_key}.weight"
            converted_key_A = f"blocks.{i}.attn1.{c}.lora_A.weight"

            original_key_B = f"blocks.{i}.self_attn.{o}.{lora_up_key}.weight"
            converted_key_B = f"blocks.{i}.attn1.{c}.lora_B.weight"

            if has_alpha:
                down_weight = original_state_dict.pop(original_key_A)
                up_weight = original_state_dict.pop(original_key_B)
                scale_down, scale_up = get_alpha_scales(down_weight, alpha_key)
                converted_state_dict[converted_key_A] = down_weight * scale_down
                converted_state_dict[converted_key_B] = up_weight * scale_up

            else:
                if original_key_A in original_state_dict:
                    converted_state_dict[converted_key_A] = original_state_dict.pop(original_key_A)
                if original_key_B in original_state_dict:
                    converted_state_dict[converted_key_B] = original_state_dict.pop(original_key_B)

            original_key = f"blocks.{i}.self_attn.{o}.diff_b"
            converted_key = f"blocks.{i}.attn1.{c}.lora_B.bias"
            if original_key in original_state_dict:
                converted_state_dict[converted_key] = original_state_dict.pop(original_key)

        # Cross-attention
        for o, c in zip(["q", "k", "v", "o"], ["to_q", "to_k", "to_v", "to_out.0"]):
            alpha_key = f"blocks.{i}.cross_attn.{o}.alpha"
            has_alpha = alpha_key in original_state_dict
            original_key_A = f"blocks.{i}.cross_attn.{o}.{lora_down_key}.weight"
            converted_key_A = f"blocks.{i}.attn2.{c}.lora_A.weight"

            original_key_B = f"blocks.{i}.cross_attn.{o}.{lora_up_key}.weight"
            converted_key_B = f"blocks.{i}.attn2.{c}.lora_B.weight"

            if original_key_A in original_state_dict:
                down_weight = original_state_dict.pop(original_key_A)
                converted_state_dict[converted_key_A] = down_weight
            if original_key_B in original_state_dict:
                up_weight = original_state_dict.pop(original_key_B)
                converted_state_dict[converted_key_B] = up_weight
            if has_alpha:
                scale_down, scale_up = get_alpha_scales(down_weight, alpha_key)
                converted_state_dict[converted_key_A] *= scale_down
                converted_state_dict[converted_key_B] *= scale_up

            original_key = f"blocks.{i}.cross_attn.{o}.diff_b"
            converted_key = f"blocks.{i}.attn2.{c}.lora_B.bias"
            if original_key in original_state_dict:
                converted_state_dict[converted_key] = original_state_dict.pop(original_key)

        if is_i2v_lora:
            for o, c in zip(["k_img", "v_img"], ["add_k_proj", "add_v_proj"]):
                alpha_key = f"blocks.{i}.cross_attn.{o}.alpha"
                has_alpha = alpha_key in original_state_dict
                original_key_A = f"blocks.{i}.cross_attn.{o}.{lora_down_key}.weight"
                converted_key_A = f"blocks.{i}.attn2.{c}.lora_A.weight"

                original_key_B = f"blocks.{i}.cross_attn.{o}.{lora_up_key}.weight"
                converted_key_B = f"blocks.{i}.attn2.{c}.lora_B.weight"

                if original_key_A in original_state_dict:
                    down_weight = original_state_dict.pop(original_key_A)
                    converted_state_dict[converted_key_A] = down_weight
                if original_key_B in original_state_dict:
                    up_weight = original_state_dict.pop(original_key_B)
                    converted_state_dict[converted_key_B] = up_weight
                if has_alpha:
                    scale_down, scale_up = get_alpha_scales(down_weight, alpha_key)
                    converted_state_dict[converted_key_A] *= scale_down
                    converted_state_dict[converted_key_B] *= scale_up

                original_key = f"blocks.{i}.cross_attn.{o}.diff_b"
                converted_key = f"blocks.{i}.attn2.{c}.lora_B.bias"
                if original_key in original_state_dict:
                    converted_state_dict[converted_key] = original_state_dict.pop(original_key)

        # FFN
        for o, c in zip(["ffn.0", "ffn.2"], ["net.0.proj", "net.2"]):
            alpha_key = f"blocks.{i}.{o}.alpha"
            has_alpha = alpha_key in original_state_dict
            original_key_A = f"blocks.{i}.{o}.{lora_down_key}.weight"
            converted_key_A = f"blocks.{i}.ffn.{c}.lora_A.weight"

            original_key_B = f"blocks.{i}.{o}.{lora_up_key}.weight"
            converted_key_B = f"blocks.{i}.ffn.{c}.lora_B.weight"

            if original_key_A in original_state_dict:
                down_weight = original_state_dict.pop(original_key_A)
                converted_state_dict[converted_key_A] = down_weight
            if original_key_B in original_state_dict:
                up_weight = original_state_dict.pop(original_key_B)
                converted_state_dict[converted_key_B] = up_weight
            if has_alpha:
                scale_down, scale_up = get_alpha_scales(down_weight, alpha_key)
                converted_state_dict[converted_key_A] *= scale_down
                converted_state_dict[converted_key_B] *= scale_up

            original_key = f"blocks.{i}.{o}.diff_b"
            converted_key = f"blocks.{i}.ffn.{c}.lora_B.bias"
            if original_key in original_state_dict:
                converted_state_dict[converted_key] = original_state_dict.pop(original_key)

    # Remaining.
    if original_state_dict:
        if any("time_projection" in k for k in original_state_dict):
            original_key = f"time_projection.1.{lora_down_key}.weight"
            converted_key = "condition_embedder.time_proj.lora_A.weight"
            if original_key in original_state_dict:
                converted_state_dict[converted_key] = original_state_dict.pop(original_key)

            original_key = f"time_projection.1.{lora_up_key}.weight"
            converted_key = "condition_embedder.time_proj.lora_B.weight"
            if original_key in original_state_dict:
                converted_state_dict[converted_key] = original_state_dict.pop(original_key)

            if "time_projection.1.diff_b" in original_state_dict:
                converted_state_dict["condition_embedder.time_proj.lora_B.bias"] = original_state_dict.pop(
                    "time_projection.1.diff_b"
                )

        if any("head.head" in k for k in original_state_dict):
            if any(f"head.head.{lora_down_key}.weight" in k for k in state_dict):
                converted_state_dict["proj_out.lora_A.weight"] = original_state_dict.pop(
                    f"head.head.{lora_down_key}.weight"
                )
            if any(f"head.head.{lora_up_key}.weight" in k for k in state_dict):
                converted_state_dict["proj_out.lora_B.weight"] = original_state_dict.pop(
                    f"head.head.{lora_up_key}.weight"
                )
            if "head.head.diff_b" in original_state_dict:
                converted_state_dict["proj_out.lora_B.bias"] = original_state_dict.pop("head.head.diff_b")

            # Notes: https://huggingface.co/lightx2v/Wan2.2-Distill-Loras
            # This is my (sayakpaul) assumption that this particular key belongs to the down matrix.
            # Since for this particular LoRA, we don't have the corresponding up matrix, I will use
            # an identity.
            if any("head.head" in k and k.endswith(".diff") for k in state_dict):
                if f"head.head.{lora_down_key}.weight" in state_dict:
                    logger.info(
                        f"The state dict seems to be have both `head.head.diff` and `head.head.{lora_down_key}.weight` keys, which is unexpected."
                    )
                converted_state_dict["proj_out.lora_A.weight"] = original_state_dict.pop("head.head.diff")
                down_matrix_head = converted_state_dict["proj_out.lora_A.weight"]
                up_matrix_shape = (down_matrix_head.shape[0], converted_state_dict["proj_out.lora_B.bias"].shape[0])
                converted_state_dict["proj_out.lora_B.weight"] = torch.eye(
                    *up_matrix_shape, dtype=down_matrix_head.dtype, device=down_matrix_head.device
                ).T

        for text_time in ["text_embedding", "time_embedding"]:
            if any(text_time in k for k in original_state_dict):
                for b_n in [0, 2]:
                    diffusers_b_n = 1 if b_n == 0 else 2
                    diffusers_name = (
                        "condition_embedder.text_embedder"
                        if text_time == "text_embedding"
                        else "condition_embedder.time_embedder"
                    )
                    if any(f"{text_time}.{b_n}" in k for k in original_state_dict):
                        converted_state_dict[f"{diffusers_name}.linear_{diffusers_b_n}.lora_A.weight"] = (
                            original_state_dict.pop(f"{text_time}.{b_n}.{lora_down_key}.weight")
                        )
                        converted_state_dict[f"{diffusers_name}.linear_{diffusers_b_n}.lora_B.weight"] = (
                            original_state_dict.pop(f"{text_time}.{b_n}.{lora_up_key}.weight")
                        )
                    if f"{text_time}.{b_n}.diff_b" in original_state_dict:
                        converted_state_dict[f"{diffusers_name}.linear_{diffusers_b_n}.lora_B.bias"] = (
                            original_state_dict.pop(f"{text_time}.{b_n}.diff_b")
                        )

        for img_ours, img_theirs in [
            ("ff.net.0.proj", "img_emb.proj.1"),
            ("ff.net.2", "img_emb.proj.3"),
        ]:
            original_key = f"{img_theirs}.{lora_down_key}.weight"
            converted_key = f"condition_embedder.image_embedder.{img_ours}.lora_A.weight"
            if original_key in original_state_dict:
                converted_state_dict[converted_key] = original_state_dict.pop(original_key)

            original_key = f"{img_theirs}.{lora_up_key}.weight"
            converted_key = f"condition_embedder.image_embedder.{img_ours}.lora_B.weight"
            if original_key in original_state_dict:
                converted_state_dict[converted_key] = original_state_dict.pop(original_key)
                bias_key_theirs = original_key.removesuffix(f".{lora_up_key}.weight") + ".diff_b"
                if bias_key_theirs in original_state_dict:
                    bias_key = converted_key.removesuffix(".weight") + ".bias"
                    converted_state_dict[bias_key] = original_state_dict.pop(bias_key_theirs)

    if len(original_state_dict) > 0:
        diff = all(".diff" in k for k in original_state_dict)
        if diff:
            diff_keys = {k for k in original_state_dict if k.endswith(".diff")}
            if not all("lora" not in k for k in diff_keys):
                raise ValueError
            logger.info(
                "The remaining `state_dict` contains `diff` keys which we do not handle yet. If you see performance issues, please file an issue: "
                "https://github.com/huggingface/diffusers//issues/new"
            )
        else:
            raise ValueError(f"`state_dict` should be empty at this point but has {original_state_dict.keys()=}")

    for key in list(converted_state_dict.keys()):
        converted_state_dict[f"transformer.{key}"] = converted_state_dict.pop(key)

    return converted_state_dict


def _convert_musubi_wan_lora_to_diffusers(state_dict):
    # https://github.com/kohya-ss/musubi-tuner
    converted_state_dict = {}
    original_state_dict = {k[len("lora_unet_") :]: v for k, v in state_dict.items()}

    num_blocks = len({k.split("blocks_")[1].split("_")[0] for k in original_state_dict})
    is_i2v_lora = any("k_img" in k for k in original_state_dict) and any("v_img" in k for k in original_state_dict)

    def get_alpha_scales(down_weight, key):
        rank = down_weight.shape[0]
        alpha = original_state_dict.pop(key + ".alpha").item()
        scale = alpha / rank  # LoRA is scaled by 'alpha / rank' in forward pass, so we need to scale it back here
        scale_down = scale
        scale_up = 1.0
        while scale_down * 2 < scale_up:
            scale_down *= 2
            scale_up /= 2
        return scale_down, scale_up

    for i in range(num_blocks):
        # Self-attention
        for o, c in zip(["q", "k", "v", "o"], ["to_q", "to_k", "to_v", "to_out.0"]):
            down_weight = original_state_dict.pop(f"blocks_{i}_self_attn_{o}.lora_down.weight")
            up_weight = original_state_dict.pop(f"blocks_{i}_self_attn_{o}.lora_up.weight")
            scale_down, scale_up = get_alpha_scales(down_weight, f"blocks_{i}_self_attn_{o}")
            converted_state_dict[f"blocks.{i}.attn1.{c}.lora_A.weight"] = down_weight * scale_down
            converted_state_dict[f"blocks.{i}.attn1.{c}.lora_B.weight"] = up_weight * scale_up

        # Cross-attention
        for o, c in zip(["q", "k", "v", "o"], ["to_q", "to_k", "to_v", "to_out.0"]):
            down_weight = original_state_dict.pop(f"blocks_{i}_cross_attn_{o}.lora_down.weight")
            up_weight = original_state_dict.pop(f"blocks_{i}_cross_attn_{o}.lora_up.weight")
            scale_down, scale_up = get_alpha_scales(down_weight, f"blocks_{i}_cross_attn_{o}")
            converted_state_dict[f"blocks.{i}.attn2.{c}.lora_A.weight"] = down_weight * scale_down
            converted_state_dict[f"blocks.{i}.attn2.{c}.lora_B.weight"] = up_weight * scale_up

        if is_i2v_lora:
            for o, c in zip(["k_img", "v_img"], ["add_k_proj", "add_v_proj"]):
                down_weight = original_state_dict.pop(f"blocks_{i}_cross_attn_{o}.lora_down.weight")
                up_weight = original_state_dict.pop(f"blocks_{i}_cross_attn_{o}.lora_up.weight")
                scale_down, scale_up = get_alpha_scales(down_weight, f"blocks_{i}_cross_attn_{o}")
                converted_state_dict[f"blocks.{i}.attn2.{c}.lora_A.weight"] = down_weight * scale_down
                converted_state_dict[f"blocks.{i}.attn2.{c}.lora_B.weight"] = up_weight * scale_up

        # FFN
        for o, c in zip(["ffn_0", "ffn_2"], ["net.0.proj", "net.2"]):
            down_weight = original_state_dict.pop(f"blocks_{i}_{o}.lora_down.weight")
            up_weight = original_state_dict.pop(f"blocks_{i}_{o}.lora_up.weight")
            scale_down, scale_up = get_alpha_scales(down_weight, f"blocks_{i}_{o}")
            converted_state_dict[f"blocks.{i}.ffn.{c}.lora_A.weight"] = down_weight * scale_down
            converted_state_dict[f"blocks.{i}.ffn.{c}.lora_B.weight"] = up_weight * scale_up

    if len(original_state_dict) > 0:
        raise ValueError(f"`state_dict` should be empty at this point but has {original_state_dict.keys()=}")

    for key in list(converted_state_dict.keys()):
        converted_state_dict[f"transformer.{key}"] = converted_state_dict.pop(key)

    return converted_state_dict


def _convert_non_diffusers_hidream_lora_to_diffusers(state_dict, non_diffusers_prefix="diffusion_model"):
    if not all(k.startswith(non_diffusers_prefix) for k in state_dict):
        raise ValueError("Invalid LoRA state dict for HiDream.")
    converted_state_dict = {k.removeprefix(f"{non_diffusers_prefix}."): v for k, v in state_dict.items()}
    converted_state_dict = {f"transformer.{k}": v for k, v in converted_state_dict.items()}
    return converted_state_dict


def _convert_non_diffusers_ltxv_lora_to_diffusers(state_dict, non_diffusers_prefix="diffusion_model"):
    if not all(k.startswith(f"{non_diffusers_prefix}.") for k in state_dict):
        raise ValueError("Invalid LoRA state dict for LTX-Video.")
    converted_state_dict = {k.removeprefix(f"{non_diffusers_prefix}."): v for k, v in state_dict.items()}
    converted_state_dict = {f"transformer.{k}": v for k, v in converted_state_dict.items()}
    return converted_state_dict


def _convert_non_diffusers_qwen_lora_to_diffusers(state_dict):
    has_diffusion_model = any(k.startswith("diffusion_model.") for k in state_dict)
    if has_diffusion_model:
        state_dict = {k.removeprefix("diffusion_model."): v for k, v in state_dict.items()}

    has_lora_unet = any(k.startswith("lora_unet_") for k in state_dict)
    if has_lora_unet:
        state_dict = {k.removeprefix("lora_unet_"): v for k, v in state_dict.items()}

        def convert_key(key: str) -> str:
            prefix = "transformer_blocks"
            if "." in key:
                base, suffix = key.rsplit(".", 1)
            else:
                base, suffix = key, ""

            start = f"{prefix}_"
            rest = base[len(start) :]

            if "." in rest:
                head, tail = rest.split(".", 1)
                tail = "." + tail
            else:
                head, tail = rest, ""

            # Protected n-grams that must keep their internal underscores
            protected = {
                # pairs
                ("to", "q"),
                ("to", "k"),
                ("to", "v"),
                ("to", "out"),
                ("add", "q"),
                ("add", "k"),
                ("add", "v"),
                ("txt", "mlp"),
                ("img", "mlp"),
                ("txt", "mod"),
                ("img", "mod"),
                # triplets
                ("add", "q", "proj"),
                ("add", "k", "proj"),
                ("add", "v", "proj"),
                ("to", "add", "out"),
            }

            prot_by_len = {}
            for ng in protected:
                prot_by_len.setdefault(len(ng), set()).add(ng)

            parts = head.split("_")
            merged = []
            i = 0
            lengths_desc = sorted(prot_by_len.keys(), reverse=True)

            while i < len(parts):
                matched = False
                for L in lengths_desc:
                    if i + L <= len(parts) and tuple(parts[i : i + L]) in prot_by_len[L]:
                        merged.append("_".join(parts[i : i + L]))
                        i += L
                        matched = True
                        break
                if not matched:
                    merged.append(parts[i])
                    i += 1

            head_converted = ".".join(merged)
            converted_base = f"{prefix}.{head_converted}{tail}"
            return converted_base + (("." + suffix) if suffix else "")

        state_dict = {convert_key(k): v for k, v in state_dict.items()}

    has_default = any("default." in k for k in state_dict)
    if has_default:
        state_dict = {k.replace("default.", ""): v for k, v in state_dict.items()}

    converted_state_dict = {}
    all_keys = list(state_dict.keys())
    down_key = ".lora_down.weight"
    up_key = ".lora_up.weight"
    a_key = ".lora_A.weight"
    b_key = ".lora_B.weight"

    has_non_diffusers_lora_id = any(down_key in k or up_key in k for k in all_keys)
    has_diffusers_lora_id = any(a_key in k or b_key in k for k in all_keys)

    if has_non_diffusers_lora_id:

        def get_alpha_scales(down_weight, alpha_key):
            rank = down_weight.shape[0]
            alpha = state_dict.pop(alpha_key).item()
            scale = alpha / rank  # LoRA is scaled by 'alpha / rank' in forward pass, so we need to scale it back here
            scale_down = scale
            scale_up = 1.0
            while scale_down * 2 < scale_up:
                scale_down *= 2
                scale_up /= 2
            return scale_down, scale_up

        for k in all_keys:
            if k.endswith(down_key):
                diffusers_down_key = k.replace(down_key, ".lora_A.weight")
                diffusers_up_key = k.replace(down_key, up_key).replace(up_key, ".lora_B.weight")
                alpha_key = k.replace(down_key, ".alpha")

                down_weight = state_dict.pop(k)
                up_weight = state_dict.pop(k.replace(down_key, up_key))
                scale_down, scale_up = get_alpha_scales(down_weight, alpha_key)
                converted_state_dict[diffusers_down_key] = down_weight * scale_down
                converted_state_dict[diffusers_up_key] = up_weight * scale_up

    # Already in diffusers format (lora_A/lora_B), just pop
    elif has_diffusers_lora_id:
        for k in all_keys:
            if a_key in k or b_key in k:
                converted_state_dict[k] = state_dict.pop(k)
            elif ".alpha" in k:
                state_dict.pop(k)

    if len(state_dict) > 0:
        raise ValueError(f"`state_dict` should be empty at this point but has {state_dict.keys()=}")

    converted_state_dict = {f"transformer.{k}": v for k, v in converted_state_dict.items()}
    return converted_state_dict
