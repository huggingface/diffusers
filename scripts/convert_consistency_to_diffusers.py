import argparse

import torch

from diffusers.models.unet_2d import UNet2DModel

TEST_UNET_CONFIG = {
    "sample_size": 32,
    "in_channels": 3,
    "out_channels": 3,
    "layers_per_block": 2,
    "num_class_embeds": 10,
    "block_out_channels": [32, 64],
    "attention_head_dim": 8,
    "down_block_types": [
        "ResnetDownsampleBlock2D",
        "AttnDownsampleBlock2D",
    ],
    "up_block_types": [
        "AttnUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
    ],
    "resnet_time_scale_shift": "scale_shift",
}

UNET_CONFIG = {
    "sample_size": 64,
    "in_channels": 3,
    "out_channels": 3,
    "layers_per_block": 3,
    "num_class_embeds": 1000,
    "block_out_channels": [192, 192 * 2, 192 * 3, 192 * 4],
    "attention_head_dim": 64,
    "down_block_types": [
        "ResnetDownsampleBlock2D",
        "AttnDownsampleBlock2D",
        "AttnDownsampleBlock2D",
        "AttnDownsampleBlock2D",
    ],
    "up_block_types": [
        "AttnUpsampleBlock2D",
        "AttnUpsampleBlock2D",
        "AttnUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
    ],
    "resnet_time_scale_shift": "scale_shift",
}


def convert_resnet(checkpoint, new_checkpoint, old_prefix, new_prefix, has_skip=False):
    new_checkpoint[f"{new_prefix}.norm1.weight"] = checkpoint[f"{old_prefix}.in_layers.0.weight"]
    new_checkpoint[f"{new_prefix}.norm1.bias"] = checkpoint[f"{old_prefix}.in_layers.0.bias"]
    new_checkpoint[f"{new_prefix}.conv1.weight"] = checkpoint[f"{old_prefix}.in_layers.2.weight"]
    new_checkpoint[f"{new_prefix}.conv1.bias"] = checkpoint[f"{old_prefix}.in_layers.2.bias"]
    new_checkpoint[f"{new_prefix}.time_emb_proj.weight"] = checkpoint[f"{old_prefix}.emb_layers.1.weight"]
    new_checkpoint[f"{new_prefix}.time_emb_proj.bias"] = checkpoint[f"{old_prefix}.emb_layers.1.bias"]
    new_checkpoint[f"{new_prefix}.norm2.weight"] = checkpoint[f"{old_prefix}.out_layers.0.weight"]
    new_checkpoint[f"{new_prefix}.norm2.bias"] = checkpoint[f"{old_prefix}.out_layers.0.bias"]
    new_checkpoint[f"{new_prefix}.conv2.weight"] = checkpoint[f"{old_prefix}.out_layers.3.weight"]
    new_checkpoint[f"{new_prefix}.conv2.bias"] = checkpoint[f"{old_prefix}.out_layers.3.bias"]

    if has_skip:
        new_checkpoint[f"{new_prefix}.conv_shortcut.weight"] = checkpoint[f"{old_prefix}.skip_connection.weight"]
        new_checkpoint[f"{new_prefix}.conv_shortcut.bias"] = checkpoint[f"{old_prefix}.skip_connection.bias"]

    return new_checkpoint


def convert_attention(checkpoint, new_checkpoint, old_prefix, new_prefix, attention_head_dim=64):
    c, _, _, _ = checkpoint[f"{old_prefix}.qkv.weight"].shape
    n_heads = c // (attention_head_dim * 3)
    old_weights = checkpoint[f"{old_prefix}.qkv.weight"].reshape(n_heads, attention_head_dim * 3, -1, 1, 1)
    old_biases = checkpoint[f"{old_prefix}.qkv.bias"].reshape(n_heads, attention_head_dim * 3, -1, 1, 1)

    weight_q, weight_k, weight_v = old_weights.chunk(3, dim=1)
    weight_q = weight_q.reshape(n_heads * attention_head_dim, -1, 1, 1)
    weight_k = weight_k.reshape(n_heads * attention_head_dim, -1, 1, 1)
    weight_v = weight_v.reshape(n_heads * attention_head_dim, -1, 1, 1)

    bias_q, bias_k, bias_v = old_biases.chunk(3, dim=1)
    bias_q = bias_q.reshape(n_heads * attention_head_dim, -1, 1, 1)
    bias_k = bias_k.reshape(n_heads * attention_head_dim, -1, 1, 1)
    bias_v = bias_v.reshape(n_heads * attention_head_dim, -1, 1, 1)

    new_checkpoint[f"{new_prefix}.group_norm.weight"] = checkpoint[f"{old_prefix}.norm.weight"]
    new_checkpoint[f"{new_prefix}.group_norm.bias"] = checkpoint[f"{old_prefix}.norm.bias"]

    new_checkpoint[f"{new_prefix}.to_q.weight"] = torch.squeeze(weight_q)
    new_checkpoint[f"{new_prefix}.to_q.bias"] = torch.squeeze(bias_q)
    new_checkpoint[f"{new_prefix}.to_k.weight"] = torch.squeeze(weight_k)
    new_checkpoint[f"{new_prefix}.to_k.bias"] = torch.squeeze(bias_k)
    new_checkpoint[f"{new_prefix}.to_v.weight"] = torch.squeeze(weight_v)
    new_checkpoint[f"{new_prefix}.to_v.bias"] = torch.squeeze(bias_v)

    new_checkpoint[f"{new_prefix}.to_out.0.weight"] = (
        checkpoint[f"{old_prefix}.proj_out.weight"].squeeze(-1).squeeze(-1)
    )
    new_checkpoint[f"{new_prefix}.to_out.0.bias"] = checkpoint[f"{old_prefix}.proj_out.bias"].squeeze(-1).squeeze(-1)

    return new_checkpoint


def con_pt_to_diffuser(checkpoint_path: str, unet_config):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    new_checkpoint = {}

    new_checkpoint["time_embedding.linear_1.weight"] = checkpoint["time_embed.0.weight"]
    new_checkpoint["time_embedding.linear_1.bias"] = checkpoint["time_embed.0.bias"]
    new_checkpoint["time_embedding.linear_2.weight"] = checkpoint["time_embed.2.weight"]
    new_checkpoint["time_embedding.linear_2.bias"] = checkpoint["time_embed.2.bias"]

    if unet_config["num_class_embeds"] is not None:
        new_checkpoint["class_embedding.weight"] = checkpoint["label_emb.weight"]

    new_checkpoint["conv_in.weight"] = checkpoint["input_blocks.0.0.weight"]
    new_checkpoint["conv_in.bias"] = checkpoint["input_blocks.0.0.bias"]

    down_block_types = unet_config["down_block_types"]
    layers_per_block = unet_config["layers_per_block"]
    attention_head_dim = unet_config["attention_head_dim"]
    current_layer = 1

    for i, layer_type in enumerate(down_block_types):
        if layer_type == "ResnetDownsampleBlock2D":
            for j in range(layers_per_block):
                new_prefix = f"down_blocks.{i}.resnets.{j}"
                old_prefix = f"input_blocks.{current_layer}.0"
                new_checkpoint = convert_resnet(checkpoint, new_checkpoint, old_prefix, new_prefix)
                current_layer += 1

        elif layer_type == "AttnDownsampleBlock2D":
            for j in range(layers_per_block):
                new_prefix = f"down_blocks.{i}.resnets.{j}"
                old_prefix = f"input_blocks.{current_layer}.0"
                has_skip = True if j == 0 else False
                new_checkpoint = convert_resnet(checkpoint, new_checkpoint, old_prefix, new_prefix, has_skip)
                new_prefix = f"down_blocks.{i}.attentions.{j}"
                old_prefix = f"input_blocks.{current_layer}.1"
                new_checkpoint = convert_attention(
                    checkpoint, new_checkpoint, old_prefix, new_prefix, attention_head_dim
                )
                current_layer += 1

        if i != len(down_block_types) - 1:
            new_prefix = f"down_blocks.{i}.downsamplers.0"
            old_prefix = f"input_blocks.{current_layer}.0"
            new_checkpoint = convert_resnet(checkpoint, new_checkpoint, old_prefix, new_prefix)
            current_layer += 1

    # hardcoded the mid-block for now
    new_prefix = "mid_block.resnets.0"
    old_prefix = "middle_block.0"
    new_checkpoint = convert_resnet(checkpoint, new_checkpoint, old_prefix, new_prefix)
    new_prefix = "mid_block.attentions.0"
    old_prefix = "middle_block.1"
    new_checkpoint = convert_attention(checkpoint, new_checkpoint, old_prefix, new_prefix, attention_head_dim)
    new_prefix = "mid_block.resnets.1"
    old_prefix = "middle_block.2"
    new_checkpoint = convert_resnet(checkpoint, new_checkpoint, old_prefix, new_prefix)

    current_layer = 0
    up_block_types = unet_config["up_block_types"]

    for i, layer_type in enumerate(up_block_types):
        if layer_type == "ResnetUpsampleBlock2D":
            for j in range(layers_per_block + 1):
                new_prefix = f"up_blocks.{i}.resnets.{j}"
                old_prefix = f"output_blocks.{current_layer}.0"
                new_checkpoint = convert_resnet(checkpoint, new_checkpoint, old_prefix, new_prefix, has_skip=True)
                current_layer += 1
        elif layer_type == "AttnUpsampleBlock2D":
            for j in range(layers_per_block + 1):
                new_prefix = f"up_blocks.{i}.resnets.{j}"
                old_prefix = f"output_blocks.{current_layer}.0"
                new_checkpoint = convert_resnet(checkpoint, new_checkpoint, old_prefix, new_prefix, has_skip=True)
                new_prefix = f"up_blocks.{i}.attentions.{j}"
                old_prefix = f"output_blocks.{current_layer}.1"
                new_checkpoint = convert_attention(
                    checkpoint, new_checkpoint, old_prefix, new_prefix, attention_head_dim
                )
                current_layer += 1

            new_prefix = f"up_blocks.{i}.upsamplers.0"
            old_prefix = f"output_blocks.{current_layer-1}.2"
            # print(new_prefix)
            # print(old_prefix)
            new_checkpoint = convert_resnet(checkpoint, new_checkpoint, old_prefix, new_prefix)

    new_checkpoint["conv_norm_out.weight"] = checkpoint["out.0.weight"]
    new_checkpoint["conv_norm_out.bias"] = checkpoint["out.0.bias"]
    new_checkpoint["conv_out.weight"] = checkpoint["out.2.weight"]
    new_checkpoint["conv_out.bias"] = checkpoint["out.2.bias"]

    return new_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--unet_path", default=None, type=str, required=True, help="Path to the unet.pt to convert.")
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to output the converted UNet model.")
    parser.add_argument("--checkpoint_name", default="cd_imagenet64_l2", type=str, help="Checkpoint to convert.")

    args = parser.parse_args()

    if args.checkpoint_name == "cd_imagenet64_l2":
        unet_config = UNET_CONFIG
    elif args.checkpoint_name == "test":
        unet_config = TEST_UNET_CONFIG
        unet_config["num_class_embeds"] = None
    elif args.checkpoint_name == "test_class_cond":
        unet_config = TEST_UNET_CONFIG
    else:
        raise ValueError(
            f"Checkpoint type {args.checkpoint_name} is not currently supported."
        )

    converted_unet_ckpt = con_pt_to_diffuser(args.unet_path, unet_config)

    image_unet = UNet2DModel(**unet_config)
    # print(image_unet)
    # exit()
    image_unet.load_state_dict(converted_unet_ckpt)
    image_unet.save_pretrained(args.dump_path)
