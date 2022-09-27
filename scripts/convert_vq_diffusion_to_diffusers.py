"""
This script ports models from VQ-diffusion (https://github.com/microsoft/VQ-Diffusion) to diffusers.

It currently only supports porting the VQVAE for the ITHQ dataset.

VQVAE for the ITHQ dataset:
```sh
# From the root directory in diffusers.

# Download the checkpoint
$ wget https://facevcstandard.blob.core.windows.net/v-zhictang/Improved-VQ-Diffusion_model_release/ithq_vqvae.pth?sv=2020-10-02&st=2022-05-30T15%3A17%3A18Z&se=2030-05-31T15%3A17%3A00Z&sr=b&sp=r&sig=1jVavHFPpUjDs%2FTO1V3PTezaNbPp2Nx8MxiWI7y6fEY%3D -O ithq_vqvae.pth

# Download the config
# NOTE that in VQ-diffusion the documented file is `configs/ithq.yaml` but the target class
# `image_synthesis.modeling.codecs.image_codec.ema_vqvae.PatchVQVAE`
# loads `OUTPUT/pretrained_model/taming_dvae/config.yaml`
$ wget https://raw.githubusercontent.com/microsoft/VQ-Diffusion/main/OUTPUT/pretrained_model/taming_dvae/config.yaml -O ithq_vqvae.yaml

# run the convert script
$ python ./scripts/convert_vq_diffusion_to_diffusers.py \
    --checkpoint_path ./ithq_vqvae.pth \
    --original_config_file ./ithq_vqvae.yaml \
    --dump_path <path to save pre-trained `VQDiffusionPipeline`> \
    --only-vqvae
```
"""

import argparse
import tempfile

import torch

import yaml
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from diffusers import VQModel
from diffusers.pipelines import VQDiffusionPipeline
from yaml.loader import FullLoader


try:
    from omegaconf import OmegaConf
except ImportError:
    raise ImportError(
        "OmegaConf is required to convert the VQ Diffusion checkpoints. Please install it with `pip install"
        " OmegaConf`."
    )

# vqvae model

PORTED_VQVAES = ["image_synthesis.modeling.codecs.image_codec.patch_vqgan.PatchVQGAN"]


def vqvae_model_from_original_config(original_config):
    assert original_config.target in PORTED_VQVAES, f"{original_config.target} has not yet been ported to diffusers."

    original_config = original_config.params

    original_encoder_config = original_config.encoder_config.params
    original_decoder_config = original_config.decoder_config.params

    in_channels = original_encoder_config.in_channels
    out_channels = original_decoder_config.out_ch

    down_block_types = get_down_block_types(original_encoder_config)
    up_block_types = get_up_block_types(original_decoder_config)

    assert original_encoder_config.ch == original_decoder_config.ch
    assert original_encoder_config.ch_mult == original_decoder_config.ch_mult
    block_out_channels = tuple(
        [original_encoder_config.ch * a_ch_mult for a_ch_mult in original_encoder_config.ch_mult]
    )

    assert original_encoder_config.num_res_blocks == original_decoder_config.num_res_blocks
    layers_per_block = original_encoder_config.num_res_blocks

    assert original_encoder_config.z_channels == original_decoder_config.z_channels
    latent_channels = original_encoder_config.z_channels

    num_vq_embeddings = original_config.n_embed

    # Hard coded value for ResnetBlock.GoupNorm(num_groups) in VQ-diffusion
    norm_num_groups = 32

    e_dim = original_config.embed_dim

    model = VQModel(
        in_channels=in_channels,
        out_channels=out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        block_out_channels=block_out_channels,
        layers_per_block=layers_per_block,
        latent_channels=latent_channels,
        num_vq_embeddings=num_vq_embeddings,
        norm_num_groups=norm_num_groups,
        e_dim=e_dim,
    )

    return model


def get_down_block_types(original_encoder_config):
    attn_resolutions = coerce_attn_resolutions(original_encoder_config.attn_resolutions)
    num_resolutions = len(original_encoder_config.ch_mult)
    resolution = coerce_resolution(original_encoder_config.resolution)

    curr_res = resolution
    down_block_types = []

    for _ in range(num_resolutions):
        if curr_res in attn_resolutions:
            down_block_type = "AttnDownEncoderBlock2D"
        else:
            down_block_type = "DownEncoderBlock2D"

        down_block_types.append(down_block_type)

        curr_res = [r // 2 for r in curr_res]

    return down_block_types


def get_up_block_types(original_decoder_config):
    attn_resolutions = coerce_attn_resolutions(original_decoder_config.attn_resolutions)
    num_resolutions = len(original_decoder_config.ch_mult)
    resolution = coerce_resolution(original_decoder_config.resolution)

    curr_res = [r // 2 ** (num_resolutions - 1) for r in resolution]
    up_block_types = []

    for _ in reversed(range(num_resolutions)):
        if curr_res in attn_resolutions:
            up_block_type = "AttnUpDecoderBlock2D"
        else:
            up_block_type = "UpDecoderBlock2D"

        up_block_types.append(up_block_type)

        curr_res = [r * 2 for r in curr_res]

    return up_block_types


def coerce_attn_resolutions(attn_resolutions):
    attn_resolutions = OmegaConf.to_object(attn_resolutions)
    attn_resolutions_ = []
    for ar in attn_resolutions:
        if isinstance(ar, (list, tuple)):
            attn_resolutions_.append(list(ar))
        else:
            attn_resolutions_.append([ar, ar])
    return attn_resolutions_


def coerce_resolution(resolution):
    resolution = OmegaConf.to_object(resolution)
    if isinstance(resolution, int):
        resolution = [resolution, resolution]  # H, W
    elif isinstance(resolution, (tuple, list)):
        resolution = list(resolution)
    else:
        raise ValueError("Unknown type of resolution:", resolution)
    return resolution


# done vqvae model

# vqvae checkpoint


def vqvae_original_checkpoint_to_diffusers_checkpoint(model, checkpoint):
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
    diffusers_checkpoint.update({"quantize.embedding.weight": checkpoint["quantize.embedding"]})

    # post_quant_conv
    diffusers_checkpoint.update(
        {
            "post_quant_conv.weight": checkpoint["post_quant_conv.weight"],
            "post_quant_conv.bias": checkpoint["post_quant_conv.bias"],
        }
    )

    # decoder
    diffusers_checkpoint.update(vqvae_decoder_to_diffusers_checkpoint(model, checkpoint))

    return diffusers_checkpoint


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
                vqvae_resnet_to_diffusers_checkpoint(
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
                    vqvae_attention_to_diffusers_checkpoint(
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
        vqvae_attention_to_diffusers_checkpoint(
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
            vqvae_resnet_to_diffusers_checkpoint(
                resnet, checkpoint, diffusers_resnet_prefix=diffusers_resnet_prefix, resnet_prefix=resnet_prefix
            )
        )

    diffusers_checkpoint.update(
        {
            # conv_norm_out
            "decoder.conv_norm_out.weight": checkpoint["decoder.norm_out.weight"],
            "decoder.conv_norm_out.bias": checkpoint["decoder.norm_out.bias"],
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


def vqvae_attention_to_diffusers_checkpoint(checkpoint, *, diffusers_attention_prefix, attention_prefix):
    return {
        # group_norm
        f"{diffusers_attention_prefix}.group_norm.weight": checkpoint[f"{attention_prefix}.norm.weight"],
        f"{diffusers_attention_prefix}.group_norm.bias": checkpoint[f"{attention_prefix}.norm.bias"],
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


# done vqvae checkpoint


def read_config_file(filename):
    # The yaml file contains annotations that certain values should
    # loaded as tuples. By default, OmegaConf will panic when reading
    # these. Instead, we can manually read the yaml with the FullLoader and then
    # construct the OmegaConf object.
    with open(filename) as f:
        original_config = yaml.load(f, FullLoader)

    return OmegaConf.create(original_config)


# We take separate arguments for the vqvae because the ITHQ vqvae config file
# is separate from the config file for the rest of the model.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vqvae_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the vqvae checkpoint to convert.",
    )

    parser.add_argument(
        "--vqvae_original_config_file",
        default=None,
        type=str,
        required=True,
        help="The YAML config file corresponding to the original architecture for the vqvae.",
    )

    parser.add_argument(
        "--checkpoint_path", default=None, type=str, required=True, help="Path to the checkpoint to convert."
    )

    parser.add_argument(
        "--original_config_file",
        default=None,
        type=str,
        required=True,
        help="The YAML config file corresponding to the original architecture.",
    )

    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")

    parser.add_argument(
        "--checkpoint_load_device",
        default="cpu",
        type=str,
        required=False,
        help="The device passed to `map_location` when loading checkpoints.",
    )

    args = parser.parse_args()

    print(f"loading checkpoints to {args.checkpoint_load_device}")

    checkpoint_map_location = torch.device(args.checkpoint_load_device)

    # vqvae_model

    print(f"loading vqvae, config: {args.vqvae_original_config_file}, checkpoint: {args.vqvae_checkpoint_path}")

    vqvae_original_config = read_config_file(args.vqvae_original_config_file).model
    vqvae_checkpoint = torch.load(args.vqvae_checkpoint_path, map_location=checkpoint_map_location)["model"]

    with init_empty_weights():
        vqvae_model = vqvae_model_from_original_config(vqvae_original_config)

    vqvae_diffusers_checkpoint = vqvae_original_checkpoint_to_diffusers_checkpoint(vqvae_model, vqvae_checkpoint)

    with tempfile.NamedTemporaryFile() as vqvae_diffusers_checkpoint_file:
        torch.save(vqvae_diffusers_checkpoint, vqvae_diffusers_checkpoint_file.name)
        del vqvae_diffusers_checkpoint
        del vqvae_checkpoint
        load_checkpoint_and_dispatch(vqvae_model, vqvae_diffusers_checkpoint_file.name, device_map="auto")

    print("done loading vqvae")

    # done vqvae_model

    print(f"saving VQ diffusion model, path: {args.dump_path}")

    pipe = VQDiffusionPipeline(vqvae=vqvae_model)
    pipe.save_pretrained(args.dump_path)

    print("done writing VQ diffusion model")
