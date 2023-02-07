import argparse

import huggingface_hub
import k_diffusion as K
import torch

from diffusers import UNet2DConditionModel


UPSCALER_REPO = "pcuenq/k-upscaler"


def resnet_to_diffusers_checkpoint(resnet, checkpoint, *, diffusers_resnet_prefix, resnet_prefix):
    rv = {
        # norm1
        f"{diffusers_resnet_prefix}.norm1.linear.weight": checkpoint[f"{resnet_prefix}.main.0.mapper.weight"],
        f"{diffusers_resnet_prefix}.norm1.linear.bias": checkpoint[f"{resnet_prefix}.main.0.mapper.bias"],
        # conv1
        f"{diffusers_resnet_prefix}.conv1.weight": checkpoint[f"{resnet_prefix}.main.2.weight"],
        f"{diffusers_resnet_prefix}.conv1.bias": checkpoint[f"{resnet_prefix}.main.2.bias"],
        # norm2
        f"{diffusers_resnet_prefix}.norm2.linear.weight": checkpoint[f"{resnet_prefix}.main.4.mapper.weight"],
        f"{diffusers_resnet_prefix}.norm2.linear.bias": checkpoint[f"{resnet_prefix}.main.4.mapper.bias"],
        # conv2
        f"{diffusers_resnet_prefix}.conv2.weight": checkpoint[f"{resnet_prefix}.main.6.weight"],
        f"{diffusers_resnet_prefix}.conv2.bias": checkpoint[f"{resnet_prefix}.main.6.bias"],
    }

    if resnet.conv_shortcut is not None:
        rv.update(
            {
                f"{diffusers_resnet_prefix}.conv_shortcut.weight": checkpoint[f"{resnet_prefix}.skip.weight"],
            }
        )

    return rv


def self_attn_to_diffusers_checkpoint(checkpoint, *, diffusers_attention_prefix, attention_prefix):
    weight_q, weight_k, weight_v = checkpoint[f"{attention_prefix}.qkv_proj.weight"].chunk(3, dim=0)
    bias_q, bias_k, bias_v = checkpoint[f"{attention_prefix}.qkv_proj.bias"].chunk(3, dim=0)
    rv = {
        # norm
        f"{diffusers_attention_prefix}.norm1.linear.weight": checkpoint[f"{attention_prefix}.norm_in.mapper.weight"],
        f"{diffusers_attention_prefix}.norm1.linear.bias": checkpoint[f"{attention_prefix}.norm_in.mapper.bias"],
        # to_q
        f"{diffusers_attention_prefix}.attn1.to_q.weight": weight_q.squeeze(-1).squeeze(-1),
        f"{diffusers_attention_prefix}.attn1.to_q.bias": bias_q,
        # to_k
        f"{diffusers_attention_prefix}.attn1.to_k.weight": weight_k.squeeze(-1).squeeze(-1),
        f"{diffusers_attention_prefix}.attn1.to_k.bias": bias_k,
        # to_v
        f"{diffusers_attention_prefix}.attn1.to_v.weight": weight_v.squeeze(-1).squeeze(-1),
        f"{diffusers_attention_prefix}.attn1.to_v.bias": bias_v,
        # to_out
        f"{diffusers_attention_prefix}.attn1.to_out.0.weight": checkpoint[f"{attention_prefix}.out_proj.weight"]
        .squeeze(-1)
        .squeeze(-1),
        f"{diffusers_attention_prefix}.attn1.to_out.0.bias": checkpoint[f"{attention_prefix}.out_proj.bias"],
    }

    return rv


def cross_attn_to_diffusers_checkpoint(
    checkpoint, *, diffusers_attention_prefix, diffusers_attention_index, attention_prefix
):
    weight_k, weight_v = checkpoint[f"{attention_prefix}.kv_proj.weight"].chunk(2, dim=0)
    bias_k, bias_v = checkpoint[f"{attention_prefix}.kv_proj.bias"].chunk(2, dim=0)

    rv = {
        # norm2 (ada groupnorm)
        f"{diffusers_attention_prefix}.norm{diffusers_attention_index}.linear.weight": checkpoint[
            f"{attention_prefix}.norm_dec.mapper.weight"
        ],
        f"{diffusers_attention_prefix}.norm{diffusers_attention_index}.linear.bias": checkpoint[
            f"{attention_prefix}.norm_dec.mapper.bias"
        ],
        # layernorm on encoder_hidden_state
        f"{diffusers_attention_prefix}.attn{diffusers_attention_index}.norm_cross.weight": checkpoint[
            f"{attention_prefix}.norm_enc.weight"
        ],
        f"{diffusers_attention_prefix}.attn{diffusers_attention_index}.norm_cross.bias": checkpoint[
            f"{attention_prefix}.norm_enc.bias"
        ],
        # to_q
        f"{diffusers_attention_prefix}.attn{diffusers_attention_index}.to_q.weight": checkpoint[
            f"{attention_prefix}.q_proj.weight"
        ]
        .squeeze(-1)
        .squeeze(-1),
        f"{diffusers_attention_prefix}.attn{diffusers_attention_index}.to_q.bias": checkpoint[
            f"{attention_prefix}.q_proj.bias"
        ],
        # to_k
        f"{diffusers_attention_prefix}.attn{diffusers_attention_index}.to_k.weight": weight_k.squeeze(-1).squeeze(-1),
        f"{diffusers_attention_prefix}.attn{diffusers_attention_index}.to_k.bias": bias_k,
        # to_v
        f"{diffusers_attention_prefix}.attn{diffusers_attention_index}.to_v.weight": weight_v.squeeze(-1).squeeze(-1),
        f"{diffusers_attention_prefix}.attn{diffusers_attention_index}.to_v.bias": bias_v,
        # to_out
        f"{diffusers_attention_prefix}.attn{diffusers_attention_index}.to_out.0.weight": checkpoint[
            f"{attention_prefix}.out_proj.weight"
        ]
        .squeeze(-1)
        .squeeze(-1),
        f"{diffusers_attention_prefix}.attn{diffusers_attention_index}.to_out.0.bias": checkpoint[
            f"{attention_prefix}.out_proj.bias"
        ],
    }

    return rv


def block_to_diffusers_checkpoint(block, checkpoint, block_idx, block_type):
    block_prefix = "inner_model.u_net.u_blocks" if block_type == "up" else "inner_model.u_net.d_blocks"
    block_prefix = f"{block_prefix}.{block_idx}"

    diffusers_checkpoint = {}

    if not hasattr(block, "attentions"):
        n = 1  # resnet only
    elif not block.attentions[0].add_self_attention:
        n = 2  # resnet -> cross-attention
    else:
        n = 3  # resnet -> self-attention -> cross-attention)

    for resnet_idx, resnet in enumerate(block.resnets):
        # diffusers_resnet_prefix = f"{diffusers_up_block_prefix}.resnets.{resnet_idx}"
        diffusers_resnet_prefix = f"{block_type}_blocks.{block_idx}.resnets.{resnet_idx}"
        idx = n * resnet_idx if block_type == "up" else n * resnet_idx + 1
        resnet_prefix = f"{block_prefix}.{idx}" if block_type == "up" else f"{block_prefix}.{idx}"

        diffusers_checkpoint.update(
            resnet_to_diffusers_checkpoint(
                resnet, checkpoint, diffusers_resnet_prefix=diffusers_resnet_prefix, resnet_prefix=resnet_prefix
            )
        )

    if hasattr(block, "attentions"):
        for attention_idx, attention in enumerate(block.attentions):
            diffusers_attention_prefix = f"{block_type}_blocks.{block_idx}.attentions.{attention_idx}"
            idx = n * attention_idx + 1 if block_type == "up" else n * attention_idx + 2
            self_attention_prefix = f"{block_prefix}.{idx}"
            cross_attention_prefix = f"{block_prefix}.{idx }"
            cross_attention_index = 1 if not attention.add_self_attention else 2
            idx = (
                n * attention_idx + cross_attention_index
                if block_type == "up"
                else n * attention_idx + cross_attention_index + 1
            )
            cross_attention_prefix = f"{block_prefix}.{idx }"

            diffusers_checkpoint.update(
                cross_attn_to_diffusers_checkpoint(
                    checkpoint,
                    diffusers_attention_prefix=diffusers_attention_prefix,
                    diffusers_attention_index=2,
                    attention_prefix=cross_attention_prefix,
                )
            )

            if attention.add_self_attention is True:
                diffusers_checkpoint.update(
                    self_attn_to_diffusers_checkpoint(
                        checkpoint,
                        diffusers_attention_prefix=diffusers_attention_prefix,
                        attention_prefix=self_attention_prefix,
                    )
                )

    return diffusers_checkpoint


def unet_to_diffusers_checkpoint(model, checkpoint):
    diffusers_checkpoint = {}

    # pre-processing
    diffusers_checkpoint.update(
        {
            "conv_in.weight": checkpoint["inner_model.proj_in.weight"],
            "conv_in.bias": checkpoint["inner_model.proj_in.bias"],
        }
    )

    # timestep and class embedding
    diffusers_checkpoint.update(
        {
            "time_proj.weight": checkpoint["inner_model.timestep_embed.weight"].squeeze(-1),
            "time_embedding.linear_1.weight": checkpoint["inner_model.mapping.0.weight"],
            "time_embedding.linear_1.bias": checkpoint["inner_model.mapping.0.bias"],
            "time_embedding.linear_2.weight": checkpoint["inner_model.mapping.2.weight"],
            "time_embedding.linear_2.bias": checkpoint["inner_model.mapping.2.bias"],
            "time_embedding.cond_proj.weight": checkpoint["inner_model.mapping_cond.weight"],
        }
    )

    # down_blocks
    for down_block_idx, down_block in enumerate(model.down_blocks):
        diffusers_checkpoint.update(block_to_diffusers_checkpoint(down_block, checkpoint, down_block_idx, "down"))

    # up_blocks
    for up_block_idx, up_block in enumerate(model.up_blocks):
        diffusers_checkpoint.update(block_to_diffusers_checkpoint(up_block, checkpoint, up_block_idx, "up"))

    # post-processing
    diffusers_checkpoint.update(
        {
            "conv_out.weight": checkpoint["inner_model.proj_out.weight"],
            "conv_out.bias": checkpoint["inner_model.proj_out.bias"],
        }
    )

    return diffusers_checkpoint


def unet_model_from_original_config(original_config):
    in_channels = original_config["input_channels"] + original_config["unet_cond_dim"]
    out_channels = original_config["input_channels"] + (1 if original_config["has_variance"] else 0)

    block_out_channels = original_config["channels"]

    assert (
        len(set(original_config["depths"])) == 1
    ), "UNet2DConditionModel currently do not support blocks with different number of layers"
    layers_per_block = original_config["depths"][0]

    class_labels_dim = original_config["mapping_cond_dim"]
    cross_attention_dim = original_config["cross_cond_dim"]

    attn1_types = []
    attn2_types = []
    for s, c in zip(original_config["self_attn_depths"], original_config["cross_attn_depths"]):
        if s:
            a1 = "self"
            a2 = "cross" if c else None
        elif c:
            a1 = "cross"
            a2 = None
        else:
            a1 = None
            a2 = None
        attn1_types.append(a1)
        attn2_types.append(a2)

    unet = UNet2DConditionModel(
        in_channels=in_channels,
        out_channels=out_channels,
        down_block_types=("KDownBlock2D", "KCrossAttnDownBlock2D", "KCrossAttnDownBlock2D", "KCrossAttnDownBlock2D"),
        mid_block_type=None,
        up_block_types=("KCrossAttnUpBlock2D", "KCrossAttnUpBlock2D", "KCrossAttnUpBlock2D", "KUpBlock2D"),
        block_out_channels=block_out_channels,
        layers_per_block=layers_per_block,
        act_fn="gelu",
        norm_num_groups=None,
        cross_attention_dim=cross_attention_dim,
        attention_head_dim=64,
        time_cond_proj_dim=class_labels_dim,
        resnet_time_scale_shift="scale_shift",
        time_embedding_type="fourier",
        timestep_post_act="gelu",
        conv_in_kernel=1,
        conv_out_kernel=1,
    )

    return unet


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    orig_config_path = huggingface_hub.hf_hub_download(UPSCALER_REPO, "config_laion_text_cond_latent_upscaler_2.json")
    orig_weights_path = huggingface_hub.hf_hub_download(
        UPSCALER_REPO, "laion_text_cond_latent_upscaler_2_1_00470000_slim.pth"
    )
    print(f"loading original model configuration from {orig_config_path}")
    print(f"loading original model checkpoint from {orig_weights_path}")

    print("converting to diffusers unet")
    orig_config = K.config.load_config(open(orig_config_path))["model"]
    model = unet_model_from_original_config(orig_config)

    orig_checkpoint = torch.load(orig_weights_path, map_location=device)["model_ema"]
    converted_checkpoint = unet_to_diffusers_checkpoint(model, orig_checkpoint)

    model.load_state_dict(converted_checkpoint, strict=True)
    model.save_pretrained(args.dump_path)
    print(f"saving converted unet model in {args.dump_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")
    args = parser.parse_args()

    main(args)
