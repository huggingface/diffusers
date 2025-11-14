import argparse
from contextlib import nullcontext

import safetensors.torch
import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download

from diffusers import AutoencoderKLFlux2
from diffusers.utils.import_utils import is_accelerate_available
from transformers import Mistral3ForConditionalGeneration, AutoProcessor


"""
# VAE

python scripts/convert_flux2_to_diffusers.py  \
--original_state_dict_repo_id "diffusers-internal-dev/dummy-flux2" \
--filename "ae.pt" \
--output_path "/raid/yiyi/dummy-flux2-diffusers" \
--dtype fp32 \
--vae
"""

CTX = init_empty_weights if is_accelerate_available() else nullcontext

parser = argparse.ArgumentParser()
parser.add_argument("--original_state_dict_repo_id", default=None, type=str)
parser.add_argument("--filename", default="flux.safetensors", type=str)
parser.add_argument("--checkpoint_path", default=None, type=str)
parser.add_argument("--vae", action="store_true")
parser.add_argument("--output_path", type=str)
parser.add_argument("--dtype", type=str, default="bf16")

args = parser.parse_args()
dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32


def load_original_checkpoint(args):
    if args.original_state_dict_repo_id is not None:
        ckpt_path = hf_hub_download(repo_id=args.original_state_dict_repo_id, filename=args.filename)
    elif args.checkpoint_path is not None:
        ckpt_path = args.checkpoint_path
    else:
        raise ValueError(" please provide either `original_state_dict_repo_id` or a local `checkpoint_path`")

    if ckpt_path.endswith(".pt"):
        original_state_dict = torch.load(ckpt_path, map_location="cpu")
    elif ckpt_path.endswith(".safetensors"):
        original_state_dict = safetensors.torch.load_file(ckpt_path)
    else:
        raise ValueError(f"Unsupported file extension: {ckpt_path}")
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


def main(args):
    original_ckpt = load_original_checkpoint(args)

    if args.vae:
        vae = AutoencoderKLFlux2()
        if "model" in original_ckpt:
            # YiYi Notes: remove this depends on if it has "model" key
            original_ckpt = original_ckpt["model"]
        converted_vae_state_dict = convert_flux2_vae_checkpoint_to_diffusers(original_ckpt, vae.config)
        vae.load_state_dict(converted_vae_state_dict, strict=True)
        vae.to(dtype).save_pretrained(f"{args.output_path}/vae")

    if args.full_pipe:
        tokenizer_id = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
        text_encoder_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(text_encoder_id, torch_dtype=torch.bfloat16)
        tokenizer = AutoProcessor.from_pretrained(tokenizer_id)

        # TODO: collate denoiser, vae, text encoder, tokenizer here.

if __name__ == "__main__":
    main(args)
