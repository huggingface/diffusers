import argparse
from typing import Any, Dict, Tuple

import torch
from accelerate import init_empty_weights
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

from diffusers import (
    AutoencoderKLWan,
    JoyImageEditPlusPipeline,
)
from diffusers.models.transformers.transformer_joyimage_edit_plus import JoyImageEditPlusTransformer3DModel
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)


# VAE conversion reused from convert_joyimage_edit_to_diffusers.py (identical VAE)
def convert_vae(vae_ckpt_path):
    old_state_dict = torch.load(vae_ckpt_path, weights_only=True)
    new_state_dict = {}

    middle_key_mapping = {
        "encoder.middle.0.residual.0.gamma": "encoder.mid_block.resnets.0.norm1.gamma",
        "encoder.middle.0.residual.2.bias": "encoder.mid_block.resnets.0.conv1.bias",
        "encoder.middle.0.residual.2.weight": "encoder.mid_block.resnets.0.conv1.weight",
        "encoder.middle.0.residual.3.gamma": "encoder.mid_block.resnets.0.norm2.gamma",
        "encoder.middle.0.residual.6.bias": "encoder.mid_block.resnets.0.conv2.bias",
        "encoder.middle.0.residual.6.weight": "encoder.mid_block.resnets.0.conv2.weight",
        "encoder.middle.2.residual.0.gamma": "encoder.mid_block.resnets.1.norm1.gamma",
        "encoder.middle.2.residual.2.bias": "encoder.mid_block.resnets.1.conv1.bias",
        "encoder.middle.2.residual.2.weight": "encoder.mid_block.resnets.1.conv1.weight",
        "encoder.middle.2.residual.3.gamma": "encoder.mid_block.resnets.1.norm2.gamma",
        "encoder.middle.2.residual.6.bias": "encoder.mid_block.resnets.1.conv2.bias",
        "encoder.middle.2.residual.6.weight": "encoder.mid_block.resnets.1.conv2.weight",
        "decoder.middle.0.residual.0.gamma": "decoder.mid_block.resnets.0.norm1.gamma",
        "decoder.middle.0.residual.2.bias": "decoder.mid_block.resnets.0.conv1.bias",
        "decoder.middle.0.residual.2.weight": "decoder.mid_block.resnets.0.conv1.weight",
        "decoder.middle.0.residual.3.gamma": "decoder.mid_block.resnets.0.norm2.gamma",
        "decoder.middle.0.residual.6.bias": "decoder.mid_block.resnets.0.conv2.bias",
        "decoder.middle.0.residual.6.weight": "decoder.mid_block.resnets.0.conv2.weight",
        "decoder.middle.2.residual.0.gamma": "decoder.mid_block.resnets.1.norm1.gamma",
        "decoder.middle.2.residual.2.bias": "decoder.mid_block.resnets.1.conv1.bias",
        "decoder.middle.2.residual.2.weight": "decoder.mid_block.resnets.1.conv1.weight",
        "decoder.middle.2.residual.3.gamma": "decoder.mid_block.resnets.1.norm2.gamma",
        "decoder.middle.2.residual.6.bias": "decoder.mid_block.resnets.1.conv2.bias",
        "decoder.middle.2.residual.6.weight": "decoder.mid_block.resnets.1.conv2.weight",
    }

    attention_mapping = {
        "encoder.middle.1.norm.gamma": "encoder.mid_block.attentions.0.norm.gamma",
        "encoder.middle.1.to_qkv.weight": "encoder.mid_block.attentions.0.to_qkv.weight",
        "encoder.middle.1.to_qkv.bias": "encoder.mid_block.attentions.0.to_qkv.bias",
        "encoder.middle.1.proj.weight": "encoder.mid_block.attentions.0.proj.weight",
        "encoder.middle.1.proj.bias": "encoder.mid_block.attentions.0.proj.bias",
        "decoder.middle.1.norm.gamma": "decoder.mid_block.attentions.0.norm.gamma",
        "decoder.middle.1.to_qkv.weight": "decoder.mid_block.attentions.0.to_qkv.weight",
        "decoder.middle.1.to_qkv.bias": "decoder.mid_block.attentions.0.to_qkv.bias",
        "decoder.middle.1.proj.weight": "decoder.mid_block.attentions.0.proj.weight",
        "decoder.middle.1.proj.bias": "decoder.mid_block.attentions.0.proj.bias",
    }

    head_mapping = {
        "encoder.head.0.gamma": "encoder.norm_out.gamma",
        "encoder.head.2.bias": "encoder.conv_out.bias",
        "encoder.head.2.weight": "encoder.conv_out.weight",
        "decoder.head.0.gamma": "decoder.norm_out.gamma",
        "decoder.head.2.bias": "decoder.conv_out.bias",
        "decoder.head.2.weight": "decoder.conv_out.weight",
    }

    quant_mapping = {
        "conv1.weight": "quant_conv.weight",
        "conv1.bias": "quant_conv.bias",
        "conv2.weight": "post_quant_conv.weight",
        "conv2.bias": "post_quant_conv.bias",
    }

    for key, value in old_state_dict.items():
        if key in middle_key_mapping:
            new_state_dict[middle_key_mapping[key]] = value
        elif key in attention_mapping:
            new_state_dict[attention_mapping[key]] = value
        elif key in head_mapping:
            new_state_dict[head_mapping[key]] = value
        elif key in quant_mapping:
            new_state_dict[quant_mapping[key]] = value
        elif key == "encoder.conv1.weight":
            new_state_dict["encoder.conv_in.weight"] = value
        elif key == "encoder.conv1.bias":
            new_state_dict["encoder.conv_in.bias"] = value
        elif key == "decoder.conv1.weight":
            new_state_dict["decoder.conv_in.weight"] = value
        elif key == "decoder.conv1.bias":
            new_state_dict["decoder.conv_in.bias"] = value
        elif key.startswith("encoder.downsamples."):
            new_key = key.replace("encoder.downsamples.", "encoder.down_blocks.")
            if ".residual.0.gamma" in new_key:
                new_key = new_key.replace(".residual.0.gamma", ".norm1.gamma")
            elif ".residual.2.bias" in new_key:
                new_key = new_key.replace(".residual.2.bias", ".conv1.bias")
            elif ".residual.2.weight" in new_key:
                new_key = new_key.replace(".residual.2.weight", ".conv1.weight")
            elif ".residual.3.gamma" in new_key:
                new_key = new_key.replace(".residual.3.gamma", ".norm2.gamma")
            elif ".residual.6.bias" in new_key:
                new_key = new_key.replace(".residual.6.bias", ".conv2.bias")
            elif ".residual.6.weight" in new_key:
                new_key = new_key.replace(".residual.6.weight", ".conv2.weight")
            elif ".shortcut.bias" in new_key:
                new_key = new_key.replace(".shortcut.bias", ".conv_shortcut.bias")
            elif ".shortcut.weight" in new_key:
                new_key = new_key.replace(".shortcut.weight", ".conv_shortcut.weight")
            new_state_dict[new_key] = value
        elif key.startswith("decoder.upsamples."):
            parts = key.split(".")
            block_idx = int(parts[2])

            if "residual" in key:
                if block_idx in [0, 1, 2]:
                    new_block_idx = 0
                    resnet_idx = block_idx
                elif block_idx in [4, 5, 6]:
                    new_block_idx = 1
                    resnet_idx = block_idx - 4
                elif block_idx in [8, 9, 10]:
                    new_block_idx = 2
                    resnet_idx = block_idx - 8
                elif block_idx in [12, 13, 14]:
                    new_block_idx = 3
                    resnet_idx = block_idx - 12
                else:
                    new_state_dict[key] = value
                    continue

                if ".residual.0.gamma" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.norm1.gamma"
                elif ".residual.2.bias" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.conv1.bias"
                elif ".residual.2.weight" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.conv1.weight"
                elif ".residual.3.gamma" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.norm2.gamma"
                elif ".residual.6.bias" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.conv2.bias"
                elif ".residual.6.weight" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.conv2.weight"
                else:
                    new_key = key
                new_state_dict[new_key] = value

            elif ".shortcut." in key:
                if block_idx == 4:
                    new_key = key.replace(".shortcut.", ".resnets.0.conv_shortcut.")
                    new_key = new_key.replace("decoder.upsamples.4", "decoder.up_blocks.1")
                else:
                    new_key = key.replace("decoder.upsamples.", "decoder.up_blocks.")
                    new_key = new_key.replace(".shortcut.", ".conv_shortcut.")
                new_state_dict[new_key] = value

            elif ".resample." in key or ".time_conv." in key:
                if block_idx == 3:
                    new_key = key.replace(f"decoder.upsamples.{block_idx}", "decoder.up_blocks.0.upsamplers.0")
                elif block_idx == 7:
                    new_key = key.replace(f"decoder.upsamples.{block_idx}", "decoder.up_blocks.1.upsamplers.0")
                elif block_idx == 11:
                    new_key = key.replace(f"decoder.upsamples.{block_idx}", "decoder.up_blocks.2.upsamplers.0")
                else:
                    new_key = key.replace("decoder.upsamples.", "decoder.up_blocks.")
                new_state_dict[new_key] = value
            else:
                new_key = key.replace("decoder.upsamples.", "decoder.up_blocks.")
                new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    with init_empty_weights():
        vae = AutoencoderKLWan()
    vae.load_state_dict(new_state_dict, strict=True, assign=True)
    return vae


def get_transformer_config() -> Dict[str, Any]:
    return {
        "hidden_size": 4096,
        "in_channels": 16,
        "num_attention_heads": 32,
        "num_layers": 40,
        "out_channels": 16,
        "patch_size": [1, 2, 2],
        "rope_dim_list": [16, 56, 56],
        "text_dim": 4096,
        "rope_type": "rope",
        "theta": 10000,
    }


def convert_transformer(ckpt_path: str):
    checkpoint = torch.load(ckpt_path, weights_only=True)
    if "model" in checkpoint:
        original_state_dict = checkpoint["model"]
    else:
        original_state_dict = checkpoint

    attn_suffixes = (
        "img_attn_qkv.",
        "img_attn_q_norm.",
        "img_attn_k_norm.",
        "img_attn_proj.",
        "txt_attn_qkv.",
        "txt_attn_q_norm.",
        "txt_attn_k_norm.",
        "txt_attn_proj.",
    )
    remapped = {}
    for key, value in original_state_dict.items():
        new_key = key
        if key.startswith("double_blocks."):
            for suffix in attn_suffixes:
                if "." + suffix in key and ".attn." + suffix not in key:
                    new_key = key.replace("." + suffix, ".attn." + suffix)
                    break
        remapped[new_key] = value

    config = get_transformer_config()
    with init_empty_weights():
        transformer = JoyImageEditPlusTransformer3DModel(**config)
    transformer.load_state_dict(remapped, strict=True, assign=True)
    return transformer


def get_args():
    parser = argparse.ArgumentParser(description="Convert JoyImage Edit Plus checkpoints to diffusers format")
    parser.add_argument("--transformer_ckpt_path", type=str, default=None)
    parser.add_argument("--vae_ckpt_path", type=str, default=None)
    parser.add_argument("--text_encoder_path", type=str, default=None)
    parser.add_argument("--save_pipeline", action="store_true")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dtype", default="bf16", help="Torch dtype (fp32, fp16, bf16)")
    parser.add_argument("--flow_shift", type=float, default=1.5)
    return parser.parse_args()


DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

if __name__ == "__main__":
    args = get_args()
    transformer = None
    vae = None
    dtype = DTYPE_MAPPING[args.dtype]

    if args.save_pipeline:
        assert args.transformer_ckpt_path is not None and args.vae_ckpt_path is not None
        assert args.text_encoder_path is not None

    if args.transformer_ckpt_path is not None:
        transformer = convert_transformer(args.transformer_ckpt_path)
        transformer = transformer.to(dtype=dtype)
        if not args.save_pipeline:
            transformer.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")

    if args.vae_ckpt_path is not None:
        vae = convert_vae(args.vae_ckpt_path)
        vae = vae.to(dtype=dtype)
        if not args.save_pipeline:
            vae.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")

    if args.save_pipeline:
        processor = AutoProcessor.from_pretrained(args.text_encoder_path)
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            args.text_encoder_path, torch_dtype=torch.bfloat16
        ).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_path)
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.flow_shift)
        transformer = transformer.to("cuda")
        vae = vae.to("cuda")
        pipe = JoyImageEditPlusPipeline(
            processor=processor,
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
        ).to("cuda")
        pipe.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")
        processor.save_pretrained(f"{args.output_path}/processor")
