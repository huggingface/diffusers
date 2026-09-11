import argparse
import tempfile

import torch
from accelerate import load_checkpoint_and_dispatch
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import UnCLIPPipeline, UNet2DConditionModel, UNet2DModel
from diffusers.loaders.conversion import get_conversion
from diffusers.loaders.conversion.configs.unclip import (
    DECODER_CONFIG,
    PRIOR_CONFIG,
    SUPER_RES_UNET_FIRST_STEPS_CONFIG,
    SUPER_RES_UNET_LAST_STEP_CONFIG,
)
from diffusers.models.transformers.prior_transformer import PriorTransformer
from diffusers.pipelines.deprecated.unclip.text_proj import UnCLIPTextProjModel
from diffusers.schedulers.scheduling_unclip import UnCLIPScheduler


r"""
Example - From the diffusers root directory:

Download weights:
```sh
$ wget https://arena.kakaocdn.net/brainrepo/models/karlo-public/v1.0.0.alpha/efdf6206d8ed593961593dc029a8affa/decoder-ckpt-step%3D01000000-of-01000000.ckpt
$ wget https://arena.kakaocdn.net/brainrepo/models/karlo-public/v1.0.0.alpha/4226b831ae0279020d134281f3c31590/improved-sr-ckpt-step%3D1.2M.ckpt
$ wget https://arena.kakaocdn.net/brainrepo/models/karlo-public/v1.0.0.alpha/85626483eaca9f581e2a78d31ff905ca/prior-ckpt-step%3D01000000-of-01000000.ckpt
$ wget https://arena.kakaocdn.net/brainrepo/models/karlo-public/v1.0.0.alpha/0b62380a75e56f073e2844ab5199153d/ViT-L-14_stats.th
```

Convert the model:
```sh
$ python scripts/recipes/unclip.py \
      --decoder_checkpoint_path ./decoder-ckpt-step\=01000000-of-01000000.ckpt \
      --super_res_unet_checkpoint_path ./improved-sr-ckpt-step\=1.2M.ckpt \
      --prior_checkpoint_path ./prior-ckpt-step\=01000000-of-01000000.ckpt \
      --clip_stat_path ./ViT-L-14_stats.th \
      --dump_path <path where to save model>
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
    return get_conversion("PriorTransformer", {**dict(model.config), "original_format": "unclip"}).to_diffusers(state)


# done prior


# decoder


# We are hardcoding the model configuration for now. If we need to generalize to more model configurations, we can
# update then.


def decoder_model_from_original_config():
    model = UNet2DConditionModel(**DECODER_CONFIG)

    return model


def decoder_original_checkpoint_to_diffusers_checkpoint(model, checkpoint):
    projection_prefixes = ("text_seq_proj.", "clip_tok_proj.", "text_feat_proj.", "clip_emb.", "cf_param")
    state = {key.removeprefix("model."): value for key, value in checkpoint.items() if key.startswith("model.")}
    state = {key: value for key, value in state.items() if not key.startswith(projection_prefixes)}
    return get_conversion("UNet2DConditionModel", dict(model.config)).to_diffusers(state)


# done decoder

# text proj


def text_proj_from_original_config():
    # From the conditional unet constructor where the dimension of the projected time embeddings is
    # constructed
    time_embed_dim = DECODER_CONFIG["block_out_channels"][0] * 4

    cross_attention_dim = DECODER_CONFIG["cross_attention_dim"]

    model = UnCLIPTextProjModel(time_embed_dim=time_embed_dim, cross_attention_dim=cross_attention_dim)

    return model


# Note that the input checkpoint is the original decoder checkpoint
def text_proj_original_checkpoint_to_diffusers_checkpoint(checkpoint):
    prefixes = (
        "model.text_seq_proj.",
        "model.clip_tok_proj.",
        "model.text_feat_proj.",
        "model.clip_emb.",
        "model.cf_param",
    )
    state = {key.removeprefix("model."): value for key, value in checkpoint.items() if key.startswith(prefixes)}
    return get_conversion("UnCLIPTextProjModel", {}).to_diffusers(state)


# done text proj

# super res unet first steps


def super_res_unet_first_steps_model_from_original_config():
    model = UNet2DModel(**SUPER_RES_UNET_FIRST_STEPS_CONFIG)

    return model


def super_res_unet_first_steps_original_checkpoint_to_diffusers_checkpoint(model, checkpoint):
    state = {
        key.removeprefix("model_first_steps."): value
        for key, value in checkpoint.items()
        if key.startswith("model_first_steps.")
    }
    return get_conversion("UNet2DModel", {**dict(model.config), "original_format": "ldm"}).to_diffusers(state)


# done super res unet first steps

# super res unet last step


def super_res_unet_last_step_model_from_original_config():
    model = UNet2DModel(**SUPER_RES_UNET_LAST_STEP_CONFIG)

    return model


def super_res_unet_last_step_original_checkpoint_to_diffusers_checkpoint(model, checkpoint):
    state = {
        key.removeprefix("model_last_step."): value
        for key, value in checkpoint.items()
        if key.startswith("model_last_step.")
    }
    return get_conversion("UNet2DModel", {**dict(model.config), "original_format": "ldm"}).to_diffusers(state)


# done super res unet last step


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


# Driver functions


def text_encoder():
    print("loading CLIP text encoder")

    clip_name = "openai/clip-vit-large-patch14"

    # sets pad_value to 0
    pad_token = "!"

    tokenizer_model = CLIPTokenizer.from_pretrained(clip_name, pad_token=pad_token, device_map="auto")

    assert tokenizer_model.convert_tokens_to_ids(pad_token) == 0

    text_encoder_model = CLIPTextModelWithProjection.from_pretrained(
        clip_name,
        # `CLIPTextModel` does not support device_map="auto"
        # device_map="auto"
    )

    print("done loading CLIP text encoder")

    return text_encoder_model, tokenizer_model


def prior(*, args, checkpoint_map_location):
    print("loading prior")

    prior_checkpoint = torch.load(args.prior_checkpoint_path, map_location=checkpoint_map_location)
    prior_checkpoint = prior_checkpoint["state_dict"]

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


def decoder(*, args, checkpoint_map_location):
    print("loading decoder")

    decoder_checkpoint = torch.load(args.decoder_checkpoint_path, map_location=checkpoint_map_location)
    decoder_checkpoint = decoder_checkpoint["state_dict"]

    decoder_model = decoder_model_from_original_config()

    decoder_diffusers_checkpoint = decoder_original_checkpoint_to_diffusers_checkpoint(
        decoder_model, decoder_checkpoint
    )

    # text proj interlude

    # The original decoder implementation includes a set of parameters that are used
    # for creating the `encoder_hidden_states` which are what the U-net is conditioned
    # on. The diffusers conditional unet directly takes the encoder_hidden_states. We pull
    # the parameters into the UnCLIPTextProjModel class
    text_proj_model = text_proj_from_original_config()

    text_proj_checkpoint = text_proj_original_checkpoint_to_diffusers_checkpoint(decoder_checkpoint)

    load_checkpoint_to_model(text_proj_checkpoint, text_proj_model, strict=True)

    # done text proj interlude

    del decoder_checkpoint

    load_checkpoint_to_model(decoder_diffusers_checkpoint, decoder_model, strict=True)

    print("done loading decoder")

    return decoder_model, text_proj_model


def super_res_unet(*, args, checkpoint_map_location):
    print("loading super resolution unet")

    super_res_checkpoint = torch.load(args.super_res_unet_checkpoint_path, map_location=checkpoint_map_location)
    super_res_checkpoint = super_res_checkpoint["state_dict"]

    # model_first_steps

    super_res_first_model = super_res_unet_first_steps_model_from_original_config()

    super_res_first_steps_checkpoint = super_res_unet_first_steps_original_checkpoint_to_diffusers_checkpoint(
        super_res_first_model, super_res_checkpoint
    )

    # model_last_step
    super_res_last_model = super_res_unet_last_step_model_from_original_config()

    super_res_last_step_checkpoint = super_res_unet_last_step_original_checkpoint_to_diffusers_checkpoint(
        super_res_last_model, super_res_checkpoint
    )

    del super_res_checkpoint

    load_checkpoint_to_model(super_res_first_steps_checkpoint, super_res_first_model, strict=True)

    load_checkpoint_to_model(super_res_last_step_checkpoint, super_res_last_model, strict=True)

    print("done loading super resolution unet")

    return super_res_first_model, super_res_last_model


def load_checkpoint_to_model(checkpoint, model, strict=False):
    with tempfile.NamedTemporaryFile() as file:
        torch.save(checkpoint, file.name)
        del checkpoint
        if strict:
            model.load_state_dict(torch.load(file.name), strict=True)
        else:
            load_checkpoint_and_dispatch(model, file.name, device_map="auto")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")

    parser.add_argument(
        "--prior_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the prior checkpoint to convert.",
    )

    parser.add_argument(
        "--decoder_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the decoder checkpoint to convert.",
    )

    parser.add_argument(
        "--super_res_unet_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the super resolution checkpoint to convert.",
    )

    parser.add_argument(
        "--clip_stat_path", default=None, type=str, required=True, help="Path to the clip stats checkpoint to convert."
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
        text_encoder_model, tokenizer_model = text_encoder()

        prior_model = prior(args=args, checkpoint_map_location=checkpoint_map_location)

        decoder_model, text_proj_model = decoder(args=args, checkpoint_map_location=checkpoint_map_location)

        super_res_first_model, super_res_last_model = super_res_unet(
            args=args, checkpoint_map_location=checkpoint_map_location
        )

        prior_scheduler = UnCLIPScheduler(
            variance_type="fixed_small_log",
            prediction_type="sample",
            num_train_timesteps=1000,
            clip_sample_range=5.0,
        )

        decoder_scheduler = UnCLIPScheduler(
            variance_type="learned_range",
            prediction_type="epsilon",
            num_train_timesteps=1000,
        )

        super_res_scheduler = UnCLIPScheduler(
            variance_type="fixed_small_log",
            prediction_type="epsilon",
            num_train_timesteps=1000,
        )

        print(f"saving Kakao Brain unCLIP to {args.dump_path}")

        pipe = UnCLIPPipeline(
            prior=prior_model,
            decoder=decoder_model,
            text_proj=text_proj_model,
            tokenizer=tokenizer_model,
            text_encoder=text_encoder_model,
            super_res_first=super_res_first_model,
            super_res_last=super_res_last_model,
            prior_scheduler=prior_scheduler,
            decoder_scheduler=decoder_scheduler,
            super_res_scheduler=super_res_scheduler,
        )
        pipe.save_pretrained(args.dump_path)

        print("done writing Kakao Brain unCLIP")
    elif args.debug == "text_encoder":
        text_encoder_model, tokenizer_model = text_encoder()
    elif args.debug == "prior":
        prior_model = prior(args=args, checkpoint_map_location=checkpoint_map_location)
    elif args.debug == "decoder":
        decoder_model, text_proj_model = decoder(args=args, checkpoint_map_location=checkpoint_map_location)
    elif args.debug == "super_res_unet":
        super_res_first_model, super_res_last_model = super_res_unet(
            args=args, checkpoint_map_location=checkpoint_map_location
        )
    else:
        raise ValueError(f"unknown debug value : {args.debug}")
