import argparse
import os
import pathlib
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import safetensors.torch
import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download

from diffusers import Flux2Transformer2DModel
from diffusers.utils.import_utils import is_accelerate_available
from transformers import Mistral3ForConditionalGeneration, AutoProcessor


"""
# Transformer
"""


CTX = init_empty_weights if is_accelerate_available() else nullcontext


FLUX2_TRANSFORMER_KEYS_RENAME_DICT ={
    # Image and text input projections
    "img_in": "x_embedder",
    "txt_in": "context_embedder",
    # Timestep and guidance embeddings
    "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
    "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
    "guidance_in.in_layer": "time_guidance_embed.guidance_embedder.linear_1",
    "guidance_in.out_layer": "time_guidance_embed.guidance_embedder.linear_2",
    # Modulation parameters
    "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
    "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
    "single_stream_modulation.lin": "single_stream_modulation.linear",
    # Final output layer
    # "final_layer.adaLN_modulation.1": "norm_out.linear",  # Handle separately since we need to swap mod params
    "final_layer.linear": "proj_out",
}


FLUX2_TRANSFORMER_ADA_LAYER_NORM_KEY_MAP = {
    "final_layer.adaLN_modulation.1": "norm_out.linear",
}


FLUX2_TRANSFORMER_DOUBLE_BLOCK_KEY_MAP = {
    # Handle fused QKV projections separately as we need to break into Q, K, V projections
    "img_attn.norm.query_norm": "attn.norm_q",
    "img_attn.norm.key_norm": "attn.norm_k",
    "img_attn.proj": "attn.to_out.0",
    "img_mlp.0": "ff.linear_in",
    "img_mlp.2": "ff.linear_out",
    "txt_attn.norm.query_norm": "attn.norm_added_q",
    "txt_attn.norm.key_norm": "attn.norm_added_k",
    "txt_attn.proj": "attn.to_add_out",
    "txt_mlp.0": "ff_context.linear_in",
    "txt_mlp.2": "ff_context.linear_out",
}


FLUX2_TRANSFORMER_SINGLE_BLOCK_KEY_MAP = {
    "linear1": "attn.to_qkv_mlp_proj",
    "linear2": "attn.to_out",
    "norm.query_norm": "attn.norm_q",
    "norm.key_norm": "attn.norm_k",
}


# in SD3 original implementation of AdaLayerNormContinuous, it split linear projection output into shift, scale;
# while in diffusers it split into scale, shift. Here we swap the linear projection weights in order to be able to use
# diffusers implementation
def swap_scale_shift(weight):
    shift, scale = weight.chunk(2, dim=0)
    new_weight = torch.cat([scale, shift], dim=0)
    return new_weight


def convert_ada_layer_norm_weights(key: str, state_dict: Dict[str, Any]) -> None:
    # Skip if not a weight
    if ".weight" not in key:
        return
    
    # If adaLN_modulation is in the key, swap scale and shift parameters
    # Original implementation is (shift, scale); diffusers implementation is (scale, shift)
    if "adaLN_modulation" in key:
        key_without_param_type, param_type = key.rsplit(".", maxsplit=1)
        # Assume all such keys are in the AdaLayerNorm key map
        new_key_without_param_type = FLUX2_TRANSFORMER_ADA_LAYER_NORM_KEY_MAP[key_without_param_type]
        new_key = ".".join([new_key_without_param_type, param_type])

        swapped_weight = swap_scale_shift(state_dict.pop(key))
        state_dict[new_key] = swapped_weight
    return


def convert_flux2_double_stream_blocks(key: str, state_dict: Dict[str, Any]) -> None:
    # Skip if not a weight, bias, or scale
    if ".weight" not in key and ".bias" not in key and ".scale" not in key:
        return
    
    new_prefix = "transformer_blocks"
    if "double_blocks." in key:
        parts = key.split(".")
        block_idx = parts[1]
        modality_block_name = parts[2]  # img_attn, img_mlp, txt_attn, txt_mlp
        within_block_name = ".".join(parts[2:-1])
        param_type = parts[-1]

        if param_type == "scale":
            param_type = "weight"
        
        if "qkv" in within_block_name:
            fused_qkv_weight = state_dict.pop(key)
            to_q_weight, to_k_weight, to_v_weight = torch.chunk(fused_qkv_weight, 3, dim=0)
            if "img" in modality_block_name:
                # double_blocks.{N}.img_attn.qkv --> transformer_blocks.{N}.attn.{to_q|to_k|to_v}
                to_q_weight, to_k_weight, to_v_weight = torch.chunk(fused_qkv_weight, 3, dim=0)
                new_q_name = "attn.to_q"
                new_k_name = "attn.to_k"
                new_v_name = "attn.to_v"
            elif "txt" in modality_block_name:
                # double_blocks.{N}.txt_attn.qkv --> transformer_blocks.{N}.attn.{add_q_proj|add_k_proj|add_v_proj}
                to_q_weight, to_k_weight, to_v_weight = torch.chunk(fused_qkv_weight, 3, dim=0)
                new_q_name = "attn.add_q_proj"
                new_k_name = "attn.add_k_proj"
                new_v_name = "attn.add_v_proj"
            new_q_key = ".".join([new_prefix, block_idx, new_q_name, param_type])
            new_k_key = ".".join([new_prefix, block_idx, new_k_name, param_type])
            new_v_key = ".".join([new_prefix, block_idx, new_v_name, param_type])
            state_dict[new_q_key] = to_q_weight
            state_dict[new_k_key] = to_k_weight
            state_dict[new_v_key] = to_v_weight
        else:
            new_within_block_name = FLUX2_TRANSFORMER_DOUBLE_BLOCK_KEY_MAP[within_block_name]
            new_key = ".".join([new_prefix, block_idx, new_within_block_name, param_type])

            param = state_dict.pop(key)
            state_dict[new_key] = param
    return


def convert_flux2_single_stream_blocks(key: str, state_dict: Dict[str, Any]) -> None:
    # Skip if not a weight, bias, or scale
    if ".weight" not in key and ".bias" not in key and ".scale" not in key:
        return
    
    # Mapping:
    #     - single_blocks.{N}.linear1               --> single_transformer_blocks.{N}.attn.to_qkv_mlp_proj
    #     - single_blocks.{N}.linear2               --> single_transformer_blocks.{N}.attn.to_out
    #     - single_blocks.{N}.norm.query_norm.scale --> single_transformer_blocks.{N}.attn.norm_q.weight
    #     - single_blocks.{N}.norm.key_norm.scale   --> single_transformer_blocks.{N}.attn.norm_k.weight
    new_prefix = "single_transformer_blocks"
    if "single_blocks." in key:
        parts = key.split(".")
        block_idx = parts[1]
        within_block_name = ".".join(parts[2:-1])
        param_type = parts[-1]

        if param_type == "scale":
            param_type = "weight"

        new_within_block_name = FLUX2_TRANSFORMER_SINGLE_BLOCK_KEY_MAP[within_block_name]
        new_key = ".".join([new_prefix, block_idx, new_within_block_name, param_type])

        param = state_dict.pop(key)
        state_dict[new_key] = param
    return


TRANSFORMER_SPECIAL_KEYS_REMAP = {
    "adaLN_modulation": convert_ada_layer_norm_weights,
    "double_blocks": convert_flux2_double_stream_blocks,
    "single_blocks": convert_flux2_single_stream_blocks,
}


def load_original_checkpoint(
    repo_id: Optional[str], model_file: Optional[str], checkpoint_path: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    if repo_id is not None:
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=model_file)
    elif checkpoint_path is not None:
        ckpt_path = checkpoint_path
    else:
        raise ValueError("Please provide either `repo_id` or a local `checkpoint_path`")

    if "safetensors" in model_file:
        original_state_dict = safetensors.torch.load_file(ckpt_path)
    else:
        original_state_dict = torch.load(ckpt_path, map_location="cpu")
    return original_state_dict


def update_state_dict(state_dict: Dict[str, Any], old_key: str, new_key: str) -> None:
    state_dict[new_key] = state_dict.pop(old_key)


def get_flux2_transformer_config(model_type: str) -> Tuple[Dict[str, Any], ...]:
    if model_type == "test" or model_type == "dummy-flux2":
        config = {
            "model_id": "diffusers-internal-dev/dummy-flux2",
            "diffusers_config": {
                "patch_size": 1,
                "in_channels": 128,
                "num_layers": 8,
                "num_single_layers": 48,
                "attention_head_dim": 128,
                "num_attention_heads": 48,
                "joint_attention_dim": 15360,
                "timestep_guidance_channels": 256,
                "mlp_ratio": 3.0,
                "axes_dims_rope": (32, 32, 32, 32),
                "rope_theta": 2000,
                "eps": 1e-6,
            }
        }
        rename_dict = FLUX2_TRANSFORMER_KEYS_RENAME_DICT
        special_keys_remap = TRANSFORMER_SPECIAL_KEYS_REMAP
    return config, rename_dict, special_keys_remap


def convert_flux2_transformer_to_diffusers(original_state_dict: Dict[str, torch.Tensor], model_type: str):
    config, rename_dict, special_keys_remap = get_flux2_transformer_config(model_type)

    diffusers_config = config["diffusers_config"]

    with init_empty_weights():
        transformer = Flux2Transformer2DModel.from_config(diffusers_config)

    # Handle official code --> diffusers key remapping via the remap dict
    for key in list(original_state_dict.keys()):
        new_key = key[:]
        for replace_key, rename_key in rename_dict.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_state_dict(original_state_dict, key, new_key)

    # Handle any special logic which can't be expressed by a simple 1:1 remapping with the handlers in
    # special_keys_remap
    for key in list(original_state_dict.keys()):
        for special_key, handler_fn_inplace in special_keys_remap.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, original_state_dict)

    transformer.load_state_dict(original_state_dict, strict=True, assign=True)
    return transformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_state_dict_repo_id", default="diffusers-internal-dev/dummy-flux2", type=str)
    parser.add_argument("--filename", default="flux.safetensors", type=str)
    parser.add_argument("--checkpoint_path", default=None, type=str)

    parser.add_argument("--model_type", type=str, default="test")
    parser.add_argument("--vae", action="store_true")
    parser.add_argument("--transformer", action="store_true")

    parser.add_argument("--dtype", type=str, default="bf16")

    parser.add_argument("--output_path", type=str)

    args = parser.parse_args()
    args.dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    return args


def main(args):
    original_ckpt = load_original_checkpoint(args.original_state_dict_repo_id, args.filename, args.checkpoint_path)

    if args.transformer:
        transformer = convert_flux2_transformer_to_diffusers(original_ckpt, args.model_type)
        transformer.to(args.dtype).save_pretrained(os.path.join(args.output_path, "transformer"))


if __name__ == "__main__":
    args = parse_args()
    main(args)
