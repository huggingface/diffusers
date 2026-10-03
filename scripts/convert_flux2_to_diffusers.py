import argparse
from contextlib import nullcontext
from typing import Any, Dict, Tuple

import safetensors.torch
import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, GenerationConfig, Mistral3ForConditionalGeneration

from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2Pipeline, Flux2Transformer2DModel
from diffusers.utils.import_utils import is_accelerate_available


"""
# VAE

python scripts/convert_flux2_to_diffusers.py  \
--original_state_dict_repo_id "diffusers-internal-dev/new-model-image" \
--vae_filename "flux2-vae.sft" \
--output_path "/raid/yiyi/dummy-flux2-diffusers" \
--vae

# DiT

python scripts/convert_flux2_to_diffusers.py \
  --original_state_dict_repo_id diffusers-internal-dev/new-model-image \
  --dit_filename flux-dev-dummy.sft \
  --dit \
  --output_path .

# Full pipe

python scripts/convert_flux2_to_diffusers.py \
  --original_state_dict_repo_id diffusers-internal-dev/new-model-image \
  --dit_filename flux-dev-dummy.sft \
  --vae_filename "flux2-vae.sft" \
  --dit --vae --full_pipe \
  --output_path .
"""

CTX = init_empty_weights if is_accelerate_available() else nullcontext

parser = argparse.ArgumentParser()
parser.add_argument("--original_state_dict_repo_id", default=None, type=str)
parser.add_argument("--vae_filename", default="flux2-vae.sft", type=str)
parser.add_argument("--dit_filename", default="flux2-dev.safetensors", type=str)
parser.add_argument("--vae", action="store_true")
parser.add_argument("--dit", action="store_true")
parser.add_argument("--vae_dtype", type=str, default="fp32")
parser.add_argument("--dit_dtype", type=str, default="bf16")
parser.add_argument("--checkpoint_path", default=None, type=str)
parser.add_argument("--full_pipe", action="store_true")
parser.add_argument("--output_path", type=str)

args = parser.parse_args()


def load_original_checkpoint(args, filename):
    if args.original_state_dict_repo_id is not None:
        ckpt_path = hf_hub_download(repo_id=args.original_state_dict_repo_id, filename=filename)
    elif args.checkpoint_path is not None:
        ckpt_path = args.checkpoint_path
    else:
        raise ValueError(" please provide either `original_state_dict_repo_id` or a local `checkpoint_path`")

    original_state_dict = safetensors.torch.load_file(ckpt_path)
    return original_state_dict


DIFFUSERS_VAE_TO_FLUX2_MAPPING = {
    "encoder.conv_in.weight": "encoder.conv_in.weight",
    "encoder.conv_in.bias": "encoder.conv_in.bias",
    "encoder.conv_out.weight": "encoder.conv_out.weight",
    "encoder.conv_out.bias": "encoder.conv_out.bias",
    "encoder.conv_norm_out.weight": "encoder.norm_out.weight",
    "encoder.conv_norm_out.bias": "encoder.norm_out.bias",
    "decoder.conv_in.weight": "decoder.conv_in.weight",
    "decoder.conv_in.bias": "decoder.conv_in.bias",
    "decoder.conv_out.weight": "decoder.conv_out.weight",
    "decoder.conv_out.bias": "decoder.conv_out.bias",
    "decoder.conv_norm_out.weight": "decoder.norm_out.weight",
    "decoder.conv_norm_out.bias": "decoder.norm_out.bias",
    "quant_conv.weight": "encoder.quant_conv.weight",
    "quant_conv.bias": "encoder.quant_conv.bias",
    "post_quant_conv.weight": "decoder.post_quant_conv.weight",
    "post_quant_conv.bias": "decoder.post_quant_conv.bias",
    "bn.running_mean": "bn.running_mean",
    "bn.running_var": "bn.running_var",
}


# Copied from diffusers.pipelines.stable_diffusion.convert_from_ckpt.conv_attn_to_linear
def conv_attn_to_linear(checkpoint):
    keys = list(checkpoint.keys())
    attn_keys = ["query.weight", "key.weight", "value.weight"]
    for key in keys:
        if ".".join(key.split(".")[-2:]) in attn_keys:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0, 0]
        elif "proj_attn.weight" in key:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0]


def update_vae_resnet_ldm_to_diffusers(keys, new_checkpoint, checkpoint, mapping):
    for ldm_key in keys:
        diffusers_key = ldm_key.replace(mapping["old"], mapping["new"]).replace("nin_shortcut", "conv_shortcut")
        new_checkpoint[diffusers_key] = checkpoint.get(ldm_key)


def update_vae_attentions_ldm_to_diffusers(keys, new_checkpoint, checkpoint, mapping):
    for ldm_key in keys:
        diffusers_key = (
            ldm_key.replace(mapping["old"], mapping["new"])
            .replace("norm.weight", "group_norm.weight")
            .replace("norm.bias", "group_norm.bias")
            .replace("q.weight", "to_q.weight")
            .replace("q.bias", "to_q.bias")
            .replace("k.weight", "to_k.weight")
            .replace("k.bias", "to_k.bias")
            .replace("v.weight", "to_v.weight")
            .replace("v.bias", "to_v.bias")
            .replace("proj_out.weight", "to_out.0.weight")
            .replace("proj_out.bias", "to_out.0.bias")
        )
        new_checkpoint[diffusers_key] = checkpoint.get(ldm_key)

        # proj_attn.weight has to be converted from conv 1D to linear
        shape = new_checkpoint[diffusers_key].shape

        if len(shape) == 3:
            new_checkpoint[diffusers_key] = new_checkpoint[diffusers_key][:, :, 0]
        elif len(shape) == 4:
            new_checkpoint[diffusers_key] = new_checkpoint[diffusers_key][:, :, 0, 0]


def convert_flux2_vae_checkpoint_to_diffusers(vae_state_dict, config):
    new_checkpoint = {}
    for diffusers_key, ldm_key in DIFFUSERS_VAE_TO_FLUX2_MAPPING.items():
        if ldm_key not in vae_state_dict:
            continue
        new_checkpoint[diffusers_key] = vae_state_dict[ldm_key]

    # Retrieves the keys for the encoder down blocks only
    num_down_blocks = len(config["down_block_types"])
    down_blocks = {
        layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key] for layer_id in range(num_down_blocks)
    }

    for i in range(num_down_blocks):
        resnets = [key for key in down_blocks[i] if f"down.{i}" in key and f"down.{i}.downsample" not in key]
        update_vae_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            vae_state_dict,
            mapping={"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"},
        )
        if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.get(
                f"encoder.down.{i}.downsample.conv.weight"
            )
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.get(
                f"encoder.down.{i}.downsample.conv.bias"
            )

    mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]
        update_vae_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            vae_state_dict,
            mapping={"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"},
        )

    mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
    update_vae_attentions_ldm_to_diffusers(
        mid_attentions, new_checkpoint, vae_state_dict, mapping={"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    )

    # Retrieves the keys for the decoder up blocks only
    num_up_blocks = len(config["up_block_types"])
    up_blocks = {
        layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key] for layer_id in range(num_up_blocks)
    }

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [
            key for key in up_blocks[block_id] if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
        ]
        update_vae_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            vae_state_dict,
            mapping={"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"},
        )
        if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.weight"
            ]
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.bias"
            ]

    mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]
        update_vae_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            vae_state_dict,
            mapping={"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"},
        )

    mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
    update_vae_attentions_ldm_to_diffusers(
        mid_attentions, new_checkpoint, vae_state_dict, mapping={"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    )
    conv_attn_to_linear(new_checkpoint)

    return new_checkpoint


FLUX2_TRANSFORMER_KEYS_RENAME_DICT = {
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


def update_state_dict(state_dict: Dict[str, Any], old_key: str, new_key: str) -> None:
    state_dict[new_key] = state_dict.pop(old_key)


def get_flux2_transformer_config(model_type: str) -> Tuple[Dict[str, Any], ...]:
    if model_type == "flux2-dev":
        config = {
            "model_id": "black-forest-labs/FLUX.2-dev",
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
            },
        }
        rename_dict = FLUX2_TRANSFORMER_KEYS_RENAME_DICT
        special_keys_remap = TRANSFORMER_SPECIAL_KEYS_REMAP
    elif model_type == "klein-4b":
        config = {
            "model_id": "diffusers-internal-dev/dummy0115",
            "diffusers_config": {
                "patch_size": 1,
                "in_channels": 128,
                "num_layers": 5,
                "num_single_layers": 20,
                "attention_head_dim": 128,
                "num_attention_heads": 24,
                "joint_attention_dim": 7680,
                "timestep_guidance_channels": 256,
                "mlp_ratio": 3.0,
                "axes_dims_rope": (32, 32, 32, 32),
                "rope_theta": 2000,
                "eps": 1e-6,
                "guidance_embeds": False,
            },
        }
        rename_dict = FLUX2_TRANSFORMER_KEYS_RENAME_DICT
        special_keys_remap = TRANSFORMER_SPECIAL_KEYS_REMAP

    elif model_type == "klein-9b":
        config = {
            "model_id": "diffusers-internal-dev/dummy0115",
            "diffusers_config": {
                "patch_size": 1,
                "in_channels": 128,
                "num_layers": 8,
                "num_single_layers": 24,
                "attention_head_dim": 128,
                "num_attention_heads": 32,
                "joint_attention_dim": 12288,
                "timestep_guidance_channels": 256,
                "mlp_ratio": 3.0,
                "axes_dims_rope": (32, 32, 32, 32),
                "rope_theta": 2000,
                "eps": 1e-6,
                "guidance_embeds": False,
            },
        }
        rename_dict = FLUX2_TRANSFORMER_KEYS_RENAME_DICT
        special_keys_remap = TRANSFORMER_SPECIAL_KEYS_REMAP

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from: flux2-dev, klein-4b, klein-9b")

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


def main(args):
    if args.vae:
        original_vae_ckpt = load_original_checkpoint(args, filename=args.vae_filename)
        vae = AutoencoderKLFlux2()
        converted_vae_state_dict = convert_flux2_vae_checkpoint_to_diffusers(original_vae_ckpt, vae.config)
        vae.load_state_dict(converted_vae_state_dict, strict=True)
        if not args.full_pipe:
            vae_dtype = torch.bfloat16 if args.vae_dtype == "bf16" else torch.float32
            vae.to(vae_dtype).save_pretrained(f"{args.output_path}/vae")

    if args.dit:
        original_dit_ckpt = load_original_checkpoint(args, filename=args.dit_filename)

        if "klein-4b" in args.dit_filename:
            model_type = "klein-4b"
        elif "klein-9b" in args.dit_filename:
            model_type = "klein-9b"
        else:
            model_type = "flux2-dev"
        transformer = convert_flux2_transformer_to_diffusers(original_dit_ckpt, model_type)
        if not args.full_pipe:
            dit_dtype = torch.bfloat16 if args.dit_dtype == "bf16" else torch.float32
            transformer.to(dit_dtype).save_pretrained(f"{args.output_path}/transformer")

    if args.full_pipe:
        tokenizer_id = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
        text_encoder_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
        generate_config = GenerationConfig.from_pretrained(text_encoder_id)
        generate_config.do_sample = True
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            text_encoder_id, generation_config=generate_config, torch_dtype=torch.bfloat16
        )
        tokenizer = AutoProcessor.from_pretrained(tokenizer_id)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            "black-forest-labs/FLUX.1-dev", subfolder="scheduler"
        )

        if_distilled = "base" not in args.dit_filename

        pipe = Flux2Pipeline(
            vae=vae,
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            if_distilled=if_distilled,
        )
        pipe.save_pretrained(args.output_path)


if __name__ == "__main__":
    main(args)
