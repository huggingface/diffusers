import argparse
import tempfile
import os
import torch
from accelerate import load_checkpoint_and_dispatch
from diffusers.models.vq_model import VQModel

"""
Example - From the diffusers root directory:

Download weights:
```sh
$ wget https://huggingface.co/ai-forever/Kandinsky_2.1/blob/main/movq_fp16.ckpt
```

Convert the model:
```sh
python scripts/convert_kandinsky_movq_to_diffusers.py --movq_checkpoint_path D:/Kandinsky-2/weights/2_1/movq_final.ckpt --dump_path ./kandinsky_model --debug movq
```
"""# movq

movq_ORIGINAL_PREFIX = "model"

# Uses default arguments
movq_CONFIG = {}


def movq_model_from_original_config():
    movq = VQModel(
                    in_channels=3, 
                    out_channels=3, 
                    latent_channels=4, 
                    use_spatial_norm=True, 
                    down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "AttnDownEncoderBlock2D"), 
                    up_block_types=("AttnUpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
                    num_vq_embeddings=16384,
                    block_out_channels=(128, 256, 256, 512),
                    vq_embed_dim=4,
                    layers_per_block=2
               )
    return movq


def vqvae_encoder_to_diffusers_checkpoint(model, checkpoint):
    diffusers_checkpoint = {}

    # conv_in
    diffusers_checkpoint.update(
        {
            "encoder.conv_in.weight": checkpoint["encoder.conv_in.weight"],
            "encoder.conv_in.bias": checkpoint["encoder.conv_in.bias"],
        }
    )

    # down_blocks
    for down_block_idx, down_block in enumerate(model.encoder.down_blocks):
        diffusers_down_block_prefix = f"encoder.down_blocks.{down_block_idx}"
        down_block_prefix = f"encoder.down.{down_block_idx}"

        # resnets
        for resnet_idx, resnet in enumerate(down_block.resnets):
            diffusers_resnet_prefix = f"{diffusers_down_block_prefix}.resnets.{resnet_idx}"
            resnet_prefix = f"{down_block_prefix}.block.{resnet_idx}"

            diffusers_checkpoint.update(
                vqvae_resnet_to_diffusers_checkpoint(
                    resnet, checkpoint, diffusers_resnet_prefix=diffusers_resnet_prefix, resnet_prefix=resnet_prefix
                )
            )

        # downsample

        # do not include the downsample when on the last down block
        # There is no downsample on the last down block
        if down_block_idx != len(model.encoder.down_blocks) - 1:
            # There's a single downsample in the original checkpoint but a list of downsamples
            # in the diffusers model.
            diffusers_downsample_prefix = f"{diffusers_down_block_prefix}.downsamplers.0.conv"
            downsample_prefix = f"{down_block_prefix}.downsample.conv"
            diffusers_checkpoint.update(
                {
                    f"{diffusers_downsample_prefix}.weight": checkpoint[f"{downsample_prefix}.weight"],
                    f"{diffusers_downsample_prefix}.bias": checkpoint[f"{downsample_prefix}.bias"],
                }
            )

        # attentions

        if hasattr(down_block, "attentions"):
            for attention_idx, _ in enumerate(down_block.attentions):
                diffusers_attention_prefix = f"{diffusers_down_block_prefix}.attentions.{attention_idx}"
                attention_prefix = f"{down_block_prefix}.attn.{attention_idx}"
                diffusers_checkpoint.update(
                    vqvae_attention_to_diffusers_checkpoint(
                        checkpoint,
                        diffusers_attention_prefix=diffusers_attention_prefix,
                        attention_prefix=attention_prefix,
                    )
                )

    # mid block

    # mid block attentions

    # There is a single hardcoded attention block in the middle of the VQ-diffusion encoder
    diffusers_attention_prefix = "encoder.mid_block.attentions.0"
    attention_prefix = "encoder.mid.attn_1"
    diffusers_checkpoint.update(
        vqvae_attention_to_diffusers_checkpoint(
            checkpoint, diffusers_attention_prefix=diffusers_attention_prefix, attention_prefix=attention_prefix
        )
    )

    # mid block resnets

    for diffusers_resnet_idx, resnet in enumerate(model.encoder.mid_block.resnets):
        diffusers_resnet_prefix = f"encoder.mid_block.resnets.{diffusers_resnet_idx}"

        # the hardcoded prefixes to `block_` are 1 and 2
        orig_resnet_idx = diffusers_resnet_idx + 1
        # There are two hardcoded resnets in the middle of the VQ-diffusion encoder
        resnet_prefix = f"encoder.mid.block_{orig_resnet_idx}"

        diffusers_checkpoint.update(
            vqvae_resnet_to_diffusers_checkpoint(
                resnet, checkpoint, diffusers_resnet_prefix=diffusers_resnet_prefix, resnet_prefix=resnet_prefix
            )
        )

    diffusers_checkpoint.update(
        {
            # conv_norm_out
            "encoder.conv_norm_out.weight": checkpoint["encoder.norm_out.weight"],
            "encoder.conv_norm_out.bias": checkpoint["encoder.norm_out.bias"],
            # conv_out
            "encoder.conv_out.weight": checkpoint["encoder.conv_out.weight"],
            "encoder.conv_out.bias": checkpoint["encoder.conv_out.bias"],
        }
    )

    return diffusers_checkpoint


def vqvae_decoder_to_diffusers_checkpoint(model, checkpoint):
    diffusers_checkpoint = {}

    # conv in
    diffusers_checkpoint.update(
        {
            "decoder.conv_in.weight": checkpoint["decoder.conv_in.weight"],
            "decoder.conv_in.bias": checkpoint["decoder.conv_in.bias"],
        }
    )

    # up_blocks

    for diffusers_up_block_idx, up_block in enumerate(model.decoder.up_blocks):
        # up_blocks are stored in reverse order in the VQ-diffusion checkpoint
        orig_up_block_idx = len(model.decoder.up_blocks) - 1 - diffusers_up_block_idx

        diffusers_up_block_prefix = f"decoder.up_blocks.{diffusers_up_block_idx}"
        up_block_prefix = f"decoder.up.{orig_up_block_idx}"

        # resnets
        for resnet_idx, resnet in enumerate(up_block.resnets):
            diffusers_resnet_prefix = f"{diffusers_up_block_prefix}.resnets.{resnet_idx}"
            resnet_prefix = f"{up_block_prefix}.block.{resnet_idx}"

            diffusers_checkpoint.update(
                vqvae_resnet_to_diffusers_checkpoint_spatial_norm(
                    resnet, checkpoint, diffusers_resnet_prefix=diffusers_resnet_prefix, resnet_prefix=resnet_prefix
                )
            )

        # upsample

        # there is no up sample on the last up block
        if diffusers_up_block_idx != len(model.decoder.up_blocks) - 1:
            # There's a single upsample in the VQ-diffusion checkpoint but a list of downsamples
            # in the diffusers model.
            diffusers_downsample_prefix = f"{diffusers_up_block_prefix}.upsamplers.0.conv"
            downsample_prefix = f"{up_block_prefix}.upsample.conv"
            diffusers_checkpoint.update(
                {
                    f"{diffusers_downsample_prefix}.weight": checkpoint[f"{downsample_prefix}.weight"],
                    f"{diffusers_downsample_prefix}.bias": checkpoint[f"{downsample_prefix}.bias"],
                }
            )

        # attentions

        if hasattr(up_block, "attentions"):
            for attention_idx, _ in enumerate(up_block.attentions):
                diffusers_attention_prefix = f"{diffusers_up_block_prefix}.attentions.{attention_idx}"
                attention_prefix = f"{up_block_prefix}.attn.{attention_idx}"
                diffusers_checkpoint.update(
                    vqvae_attention_to_diffusers_checkpoint_spatial_norm(
                        checkpoint,
                        diffusers_attention_prefix=diffusers_attention_prefix,
                        attention_prefix=attention_prefix,
                    )
                )

    # mid block

    # mid block attentions

    # There is a single hardcoded attention block in the middle of the VQ-diffusion decoder
    diffusers_attention_prefix = "decoder.mid_block.attentions.0"
    attention_prefix = "decoder.mid.attn_1"
    diffusers_checkpoint.update(
        vqvae_attention_to_diffusers_checkpoint_spatial_norm(
            checkpoint, diffusers_attention_prefix=diffusers_attention_prefix, attention_prefix=attention_prefix
        )
    )

    # mid block resnets

    for diffusers_resnet_idx, resnet in enumerate(model.encoder.mid_block.resnets):
        diffusers_resnet_prefix = f"decoder.mid_block.resnets.{diffusers_resnet_idx}"

        # the hardcoded prefixes to `block_` are 1 and 2
        orig_resnet_idx = diffusers_resnet_idx + 1
        # There are two hardcoded resnets in the middle of the VQ-diffusion decoder
        resnet_prefix = f"decoder.mid.block_{orig_resnet_idx}"

        diffusers_checkpoint.update(
            vqvae_resnet_to_diffusers_checkpoint_spatial_norm(
                resnet, checkpoint, diffusers_resnet_prefix=diffusers_resnet_prefix, resnet_prefix=resnet_prefix
            )
        )

    diffusers_checkpoint.update(
        {
            # conv_norm_out
            "decoder.conv_norm_out.norm_layer.weight": checkpoint["decoder.norm_out.norm_layer.weight"],
            "decoder.conv_norm_out.norm_layer.bias": checkpoint["decoder.norm_out.norm_layer.bias"],
            "decoder.conv_norm_out.conv_y.weight": checkpoint["decoder.norm_out.conv_y.weight"],
            "decoder.conv_norm_out.conv_y.bias": checkpoint["decoder.norm_out.conv_y.bias"],
            "decoder.conv_norm_out.conv_b.weight": checkpoint["decoder.norm_out.conv_b.weight"],
            "decoder.conv_norm_out.conv_b.bias": checkpoint["decoder.norm_out.conv_b.bias"],
            # conv_out
            "decoder.conv_out.weight": checkpoint["decoder.conv_out.weight"],
            "decoder.conv_out.bias": checkpoint["decoder.conv_out.bias"],
        }
    )

    return diffusers_checkpoint


def vqvae_resnet_to_diffusers_checkpoint(resnet, checkpoint, *, diffusers_resnet_prefix, resnet_prefix):
    rv = {
        # norm1
        f"{diffusers_resnet_prefix}.norm1.weight": checkpoint[f"{resnet_prefix}.norm1.weight"],
        f"{diffusers_resnet_prefix}.norm1.bias": checkpoint[f"{resnet_prefix}.norm1.bias"],
        # conv1
        f"{diffusers_resnet_prefix}.conv1.weight": checkpoint[f"{resnet_prefix}.conv1.weight"],
        f"{diffusers_resnet_prefix}.conv1.bias": checkpoint[f"{resnet_prefix}.conv1.bias"],
        # norm2
        f"{diffusers_resnet_prefix}.norm2.weight": checkpoint[f"{resnet_prefix}.norm2.weight"],
        f"{diffusers_resnet_prefix}.norm2.bias": checkpoint[f"{resnet_prefix}.norm2.bias"],
        # conv2
        f"{diffusers_resnet_prefix}.conv2.weight": checkpoint[f"{resnet_prefix}.conv2.weight"],
        f"{diffusers_resnet_prefix}.conv2.bias": checkpoint[f"{resnet_prefix}.conv2.bias"],
    }

    if resnet.conv_shortcut is not None:
        rv.update(
            {
                f"{diffusers_resnet_prefix}.conv_shortcut.weight": checkpoint[f"{resnet_prefix}.nin_shortcut.weight"],
                f"{diffusers_resnet_prefix}.conv_shortcut.bias": checkpoint[f"{resnet_prefix}.nin_shortcut.bias"],
            }
        )

    return rv

def vqvae_resnet_to_diffusers_checkpoint_spatial_norm(resnet, checkpoint, *, diffusers_resnet_prefix, resnet_prefix):
    rv = {
        # norm1
        f"{diffusers_resnet_prefix}.norm1.norm_layer.weight": checkpoint[f"{resnet_prefix}.norm1.norm_layer.weight"],
        f"{diffusers_resnet_prefix}.norm1.norm_layer.bias": checkpoint[f"{resnet_prefix}.norm1.norm_layer.bias"],
        f"{diffusers_resnet_prefix}.norm1.conv_y.weight": checkpoint[f"{resnet_prefix}.norm1.conv_y.weight"],
        f"{diffusers_resnet_prefix}.norm1.conv_y.bias": checkpoint[f"{resnet_prefix}.norm1.conv_y.bias"],
        f"{diffusers_resnet_prefix}.norm1.conv_b.weight": checkpoint[f"{resnet_prefix}.norm1.conv_b.weight"],
        f"{diffusers_resnet_prefix}.norm1.conv_b.bias": checkpoint[f"{resnet_prefix}.norm1.conv_b.bias"],
        # conv1
        f"{diffusers_resnet_prefix}.conv1.weight": checkpoint[f"{resnet_prefix}.conv1.weight"],
        f"{diffusers_resnet_prefix}.conv1.bias": checkpoint[f"{resnet_prefix}.conv1.bias"],
        # norm2
        f"{diffusers_resnet_prefix}.norm2.norm_layer.weight": checkpoint[f"{resnet_prefix}.norm2.norm_layer.weight"],
        f"{diffusers_resnet_prefix}.norm2.norm_layer.bias": checkpoint[f"{resnet_prefix}.norm2.norm_layer.bias"],
        f"{diffusers_resnet_prefix}.norm2.conv_y.weight": checkpoint[f"{resnet_prefix}.norm2.conv_y.weight"],
        f"{diffusers_resnet_prefix}.norm2.conv_y.bias": checkpoint[f"{resnet_prefix}.norm2.conv_y.bias"],
        f"{diffusers_resnet_prefix}.norm2.conv_b.weight": checkpoint[f"{resnet_prefix}.norm2.conv_b.weight"],
        f"{diffusers_resnet_prefix}.norm2.conv_b.bias": checkpoint[f"{resnet_prefix}.norm2.conv_b.bias"],
        # conv2
        f"{diffusers_resnet_prefix}.conv2.weight": checkpoint[f"{resnet_prefix}.conv2.weight"],
        f"{diffusers_resnet_prefix}.conv2.bias": checkpoint[f"{resnet_prefix}.conv2.bias"],
    }

    if resnet.conv_shortcut is not None:
        rv.update(
            {
                f"{diffusers_resnet_prefix}.conv_shortcut.weight": checkpoint[f"{resnet_prefix}.nin_shortcut.weight"],
                f"{diffusers_resnet_prefix}.conv_shortcut.bias": checkpoint[f"{resnet_prefix}.nin_shortcut.bias"],
            }
        )

    return rv



def vqvae_attention_to_diffusers_checkpoint(checkpoint, *, diffusers_attention_prefix, attention_prefix):
    return {
        # norm
        f"{diffusers_attention_prefix}.norm.weight": checkpoint[f"{attention_prefix}.norm.weight"],
        f"{diffusers_attention_prefix}.norm.bias": checkpoint[f"{attention_prefix}.norm.bias"],
        # query
        f"{diffusers_attention_prefix}.query.weight": checkpoint[f"{attention_prefix}.q.weight"][:, :, 0, 0],
        f"{diffusers_attention_prefix}.query.bias": checkpoint[f"{attention_prefix}.q.bias"],
        # key
        f"{diffusers_attention_prefix}.key.weight": checkpoint[f"{attention_prefix}.k.weight"][:, :, 0, 0],
        f"{diffusers_attention_prefix}.key.bias": checkpoint[f"{attention_prefix}.k.bias"],
        # value
        f"{diffusers_attention_prefix}.value.weight": checkpoint[f"{attention_prefix}.v.weight"][:, :, 0, 0],
        f"{diffusers_attention_prefix}.value.bias": checkpoint[f"{attention_prefix}.v.bias"],
        # proj_attn
        f"{diffusers_attention_prefix}.proj_attn.weight": checkpoint[f"{attention_prefix}.proj_out.weight"][
            :, :, 0, 0
        ],
        f"{diffusers_attention_prefix}.proj_attn.bias": checkpoint[f"{attention_prefix}.proj_out.bias"],
    }

def vqvae_attention_to_diffusers_checkpoint_spatial_norm(checkpoint, *, diffusers_attention_prefix, attention_prefix):
    return {
        # norm
        f"{diffusers_attention_prefix}.norm.norm_layer.weight": checkpoint[f"{attention_prefix}.norm.norm_layer.weight"],
        f"{diffusers_attention_prefix}.norm.norm_layer.bias": checkpoint[f"{attention_prefix}.norm.norm_layer.bias"],
        f"{diffusers_attention_prefix}.norm.conv_y.weight": checkpoint[f"{attention_prefix}.norm.conv_y.weight"],
        f"{diffusers_attention_prefix}.norm.conv_y.bias": checkpoint[f"{attention_prefix}.norm.conv_y.bias"],
        f"{diffusers_attention_prefix}.norm.conv_b.weight": checkpoint[f"{attention_prefix}.norm.conv_b.weight"],
        f"{diffusers_attention_prefix}.norm.conv_b.bias": checkpoint[f"{attention_prefix}.norm.conv_b.bias"],
        # query
        f"{diffusers_attention_prefix}.query.weight": checkpoint[f"{attention_prefix}.q.weight"][:, :, 0, 0],
        f"{diffusers_attention_prefix}.query.bias": checkpoint[f"{attention_prefix}.q.bias"],
        # key
        f"{diffusers_attention_prefix}.key.weight": checkpoint[f"{attention_prefix}.k.weight"][:, :, 0, 0],
        f"{diffusers_attention_prefix}.key.bias": checkpoint[f"{attention_prefix}.k.bias"],
        # value
        f"{diffusers_attention_prefix}.value.weight": checkpoint[f"{attention_prefix}.v.weight"][:, :, 0, 0],
        f"{diffusers_attention_prefix}.value.bias": checkpoint[f"{attention_prefix}.v.bias"],
        # proj_attn
        f"{diffusers_attention_prefix}.proj_attn.weight": checkpoint[f"{attention_prefix}.proj_out.weight"][
            :, :, 0, 0
        ],
        f"{diffusers_attention_prefix}.proj_attn.bias": checkpoint[f"{attention_prefix}.proj_out.bias"],
    }





def movq_original_checkpoint_to_diffusers_checkpoint(model, checkpoint):
    diffusers_checkpoint = {}
    diffusers_checkpoint.update(vqvae_encoder_to_diffusers_checkpoint(model, checkpoint))


    # quant_conv

    diffusers_checkpoint.update(
        {
            "quant_conv.weight": checkpoint["quant_conv.weight"],
            "quant_conv.bias": checkpoint["quant_conv.bias"],
        }
    )

    # quantize
    diffusers_checkpoint.update({"quantize.embedding.weight": checkpoint["quantize.embedding.weight"]})

    # post_quant_conv
    diffusers_checkpoint.update(
        {
            "post_quant_conv.weight": checkpoint["post_quant_conv.weight"],
            "post_quant_conv.bias": checkpoint["post_quant_conv.bias"],
        }
    )

    # decoder
    diffusers_checkpoint.update(vqvae_decoder_to_diffusers_checkpoint(model, checkpoint))



    for keys in diffusers_checkpoint.keys():
        print(keys)

    return diffusers_checkpoint



def movq(*, args, checkpoint_map_location):
    print("loading movq")

    movq_checkpoint = torch.load(args.movq_checkpoint_path, map_location=checkpoint_map_location)
    movq_model = movq_model_from_original_config()

    movq_diffusers_checkpoint = movq_original_checkpoint_to_diffusers_checkpoint(
        movq_model, movq_checkpoint
    )

    del movq_checkpoint
    load_checkpoint_to_model(movq_diffusers_checkpoint, movq_model, strict=True)

    print("done loading movq")

    return movq_model

def load_checkpoint_to_model(checkpoint, model, strict=False):
    with tempfile.NamedTemporaryFile(delete=False) as file:
        torch.save(checkpoint, file.name)
        del checkpoint
        if strict:
            model.load_state_dict(torch.load(file.name), strict=True)
        else:
            load_checkpoint_and_dispatch(model, file.name, device_map="auto")
    os.remove(file.name)
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dump_path", default="./kandinsky_model", type=str, required=False, help="Path to the output model.")

    parser.add_argument(
        "--movq_checkpoint_path",
        default="D:/Kandinsky-2/weights/2_1/movq_final.ckpt",
        type=str,
        required=False,
        help="Path to the movq checkpoint to convert.",
    )
    parser.add_argument(
        "--checkpoint_load_device",
        default="cpu",
        type=str,
        required=False,
        help="The device passed to `map_location` when loading checkpoints.",
    )

    parser.add_argument(
        "--debug",
        default=None,
        type=str,
        required=False,
        help="Only run a specific stage of the convert script. Used for debugging",
    )

    args = parser.parse_args()

    print(f"loading checkpoints to {args.checkpoint_load_device}")

    checkpoint_map_location = torch.device(args.checkpoint_load_device)


    movq_model = movq(args=args, checkpoint_map_location=checkpoint_map_location)
