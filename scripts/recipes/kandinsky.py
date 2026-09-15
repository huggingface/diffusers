import argparse
import os
import tempfile

import torch
from accelerate import load_checkpoint_and_dispatch

from diffusers import UNet2DConditionModel
from diffusers.loaders.conversion import get_conversion
from diffusers.loaders.conversion.configs.kandinsky import INPAINT_UNET_CONFIG, MOVQ_CONFIG, PRIOR_CONFIG, UNET_CONFIG
from diffusers.models.transformers.prior_transformer import PriorTransformer
from diffusers.models.vq_model import VQModel


"""
Example - From the diffusers root directory:

Download weights:
```sh
$ wget https://huggingface.co/ai-forever/Kandinsky_2.1/blob/main/prior_fp16.ckpt
```

Convert the model:
```sh
python scripts/recipes/kandinsky.py \
      --prior_checkpoint_path /home/yiyi_huggingface_co/Kandinsky-2/checkpoints_Kandinsky_2.1/prior_fp16.ckpt \
      --clip_stat_path  /home/yiyi_huggingface_co/Kandinsky-2/checkpoints_Kandinsky_2.1/ViT-L-14_stats.th \
      --text2img_checkpoint_path /home/yiyi_huggingface_co/Kandinsky-2/checkpoints_Kandinsky_2.1/decoder_fp16.ckpt \
      --inpaint_text2img_checkpoint_path /home/yiyi_huggingface_co/Kandinsky-2/checkpoints_Kandinsky_2.1/inpainting_fp16.ckpt \
      --movq_checkpoint_path /home/yiyi_huggingface_co/Kandinsky-2/checkpoints_Kandinsky_2.1/movq_final.ckpt \
      --dump_path /home/yiyi_huggingface_co/dump \
      --debug decoder
```
"""


# prior


# Uses default arguments


def prior_model_from_original_config():
    model = PriorTransformer(**PRIOR_CONFIG)

    return model


def prior_original_checkpoint_to_diffusers_checkpoint(model, checkpoint, clip_stats_checkpoint):
    state = dict(checkpoint)
    state["clip_stats.mean"], state["clip_stats.std"] = clip_stats_checkpoint
    return get_conversion("PriorTransformer", {**dict(model.config), "original_format": "kandinsky"}).to_diffusers(
        state
    )


# done prior

# unet

# We are hardcoding the model configuration for now. If we need to generalize to more model configurations, we can
# update then.


def unet_model_from_original_config():
    model = UNet2DConditionModel(**UNET_CONFIG)

    return model


def unet_original_checkpoint_to_diffusers_checkpoint(model, checkpoint):
    return get_conversion("UNet2DConditionModel", dict(model.config)).to_diffusers(checkpoint)


# done unet

# inpaint unet

# We are hardcoding the model configuration for now. If we need to generalize to more model configurations, we can
# update then.


def inpaint_unet_model_from_original_config():
    model = UNet2DConditionModel(**INPAINT_UNET_CONFIG)

    return model


def inpaint_unet_original_checkpoint_to_diffusers_checkpoint(model, checkpoint):
    return get_conversion("UNet2DConditionModel", dict(model.config)).to_diffusers(checkpoint)


# done inpaint unet


# unet utils


# <original>.time_embed -> <diffusers>.time_embedding


# <original>.input_blocks.0 -> <diffusers>.conv_in


# <original>.out.0 -> <diffusers>.conv_norm_out


# <original>.out.2 -> <diffusers>.conv_out


# <original>.input_blocks -> <diffusers>.down_blocks


# <original>.middle_block -> <diffusers>.mid_block


# <original>.output_blocks -> <diffusers>.up_blocks


# TODO maybe document and/or can do more efficiently (build indices in for loop and extract once for each split?)


# done unet utils


def prior(*, args, checkpoint_map_location):
    print("loading prior")

    prior_checkpoint = torch.load(args.prior_checkpoint_path, map_location=checkpoint_map_location)

    clip_stats_checkpoint = torch.load(args.clip_stat_path, map_location=checkpoint_map_location)

    prior_model = prior_model_from_original_config()

    prior_diffusers_checkpoint = prior_original_checkpoint_to_diffusers_checkpoint(
        prior_model, prior_checkpoint, clip_stats_checkpoint
    )

    del prior_checkpoint
    del clip_stats_checkpoint

    load_checkpoint_to_model(prior_diffusers_checkpoint, prior_model, strict=True)

    print("done loading prior")

    return prior_model


def text2img(*, args, checkpoint_map_location):
    print("loading text2img")

    text2img_checkpoint = torch.load(args.text2img_checkpoint_path, map_location=checkpoint_map_location)

    unet_model = unet_model_from_original_config()

    unet_diffusers_checkpoint = unet_original_checkpoint_to_diffusers_checkpoint(unet_model, text2img_checkpoint)

    del text2img_checkpoint

    load_checkpoint_to_model(unet_diffusers_checkpoint, unet_model, strict=True)

    print("done loading text2img")

    return unet_model


def inpaint_text2img(*, args, checkpoint_map_location):
    print("loading inpaint text2img")

    inpaint_text2img_checkpoint = torch.load(
        args.inpaint_text2img_checkpoint_path, map_location=checkpoint_map_location
    )

    inpaint_unet_model = inpaint_unet_model_from_original_config()

    inpaint_unet_diffusers_checkpoint = inpaint_unet_original_checkpoint_to_diffusers_checkpoint(
        inpaint_unet_model, inpaint_text2img_checkpoint
    )

    del inpaint_text2img_checkpoint

    load_checkpoint_to_model(inpaint_unet_diffusers_checkpoint, inpaint_unet_model, strict=True)

    print("done loading inpaint text2img")

    return inpaint_unet_model


# movq


def movq_model_from_original_config():
    movq = VQModel(**MOVQ_CONFIG)
    return movq


def movq_original_checkpoint_to_diffusers_checkpoint(model, checkpoint):
    return get_conversion("VQModel", dict(model.config)).to_diffusers(checkpoint)


def movq(*, args, checkpoint_map_location):
    print("loading movq")

    movq_checkpoint = torch.load(args.movq_checkpoint_path, map_location=checkpoint_map_location)

    movq_model = movq_model_from_original_config()

    movq_diffusers_checkpoint = movq_original_checkpoint_to_diffusers_checkpoint(movq_model, movq_checkpoint)

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

    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")

    parser.add_argument(
        "--prior_checkpoint_path",
        default=None,
        type=str,
        required=False,
        help="Path to the prior checkpoint to convert.",
    )
    parser.add_argument(
        "--clip_stat_path",
        default=None,
        type=str,
        required=False,
        help="Path to the clip stats checkpoint to convert.",
    )
    parser.add_argument(
        "--text2img_checkpoint_path",
        default=None,
        type=str,
        required=False,
        help="Path to the text2img checkpoint to convert.",
    )
    parser.add_argument(
        "--movq_checkpoint_path",
        default=None,
        type=str,
        required=False,
        help="Path to the text2img checkpoint to convert.",
    )
    parser.add_argument(
        "--inpaint_text2img_checkpoint_path",
        default=None,
        type=str,
        required=False,
        help="Path to the inpaint text2img checkpoint to convert.",
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

    if args.debug is not None:
        print(f"debug: only executing {args.debug}")

    if args.debug is None:
        print("to-do")
    elif args.debug == "prior":
        prior_model = prior(args=args, checkpoint_map_location=checkpoint_map_location)
        prior_model.save_pretrained(args.dump_path)
    elif args.debug == "text2img":
        unet_model = text2img(args=args, checkpoint_map_location=checkpoint_map_location)
        unet_model.save_pretrained(f"{args.dump_path}/unet")
    elif args.debug == "inpaint_text2img":
        inpaint_unet_model = inpaint_text2img(args=args, checkpoint_map_location=checkpoint_map_location)
        inpaint_unet_model.save_pretrained(f"{args.dump_path}/inpaint_unet")
    elif args.debug == "decoder":
        decoder = movq(args=args, checkpoint_map_location=checkpoint_map_location)
        decoder.save_pretrained(f"{args.dump_path}/decoder")
    else:
        raise ValueError(f"unknown debug value : {args.debug}")
