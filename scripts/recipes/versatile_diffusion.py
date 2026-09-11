# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Conversion script for the Versatile Stable Diffusion checkpoints."""

import argparse

import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
    VersatileDiffusionPipeline,
)
from diffusers.loaders.conversion import get_conversion
from diffusers.loaders.conversion.configs.versatile_diffusion import (
    AUTOENCODER_CONFIG,
    IMAGE_UNET_CONFIG,
    SCHEDULER_CONFIG,
    TEXT_UNET_CONFIG,
    create_image_unet_diffusers_config,
    create_text_unet_diffusers_config,
    create_vae_diffusers_config,
)
from diffusers.pipelines.deprecated.versatile_diffusion.modeling_text_unet import UNetFlatConditionModel


def convert_vd_unet_checkpoint(checkpoint, config, unet_key, extract_ema=False):
    state = {
        key: value
        for key, value in checkpoint.items()
        if key.startswith((unet_key, "model.diffusion_model.time_embed."))
    }
    if extract_ema:
        state = {key: checkpoint["model_ema." + "".join(key.split(".")[1:])] for key in state}
    cls = "UNetFlatConditionModel" if "unet_text." in unet_key else "UNet2DConditionModel"
    config = dict(config)
    if cls == "UNet2DConditionModel":
        config["original_format"] = "versatile_image"
    return get_conversion(cls, config).to_diffusers(state)


def convert_vd_vae_checkpoint(checkpoint, config):
    state = {key.removeprefix("first_stage_model."): value for key, value in checkpoint.items()}
    return get_conversion("AutoencoderKL", config).to_diffusers(state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--unet_checkpoint_path", default=None, type=str, required=False, help="Path to the checkpoint to convert."
    )
    parser.add_argument(
        "--vae_checkpoint_path", default=None, type=str, required=False, help="Path to the checkpoint to convert."
    )
    parser.add_argument(
        "--optimus_checkpoint_path", default=None, type=str, required=False, help="Path to the checkpoint to convert."
    )
    parser.add_argument(
        "--scheduler_type",
        default="pndm",
        type=str,
        help="Type of scheduler to use. Should be one of ['pndm', 'lms', 'ddim', 'euler', 'euler-ancestral', 'dpm']",
    )
    parser.add_argument(
        "--extract_ema",
        action="store_true",
        help=(
            "Only relevant for checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights"
            " or not. Defaults to `False`. Add `--extract_ema` to extract the EMA weights. EMA weights usually yield"
            " higher quality images for inference. Non-EMA weights are usually better to continue fine-tuning."
        ),
    )
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")

    args = parser.parse_args()

    scheduler_config = SCHEDULER_CONFIG

    num_train_timesteps = scheduler_config.timesteps
    beta_start = scheduler_config.beta_linear_start
    beta_end = scheduler_config.beta_linear_end
    if args.scheduler_type == "pndm":
        scheduler = PNDMScheduler(
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            beta_start=beta_start,
            num_train_timesteps=num_train_timesteps,
            skip_prk_steps=True,
            steps_offset=1,
        )
    elif args.scheduler_type == "lms":
        scheduler = LMSDiscreteScheduler(beta_start=beta_start, beta_end=beta_end, beta_schedule="scaled_linear")
    elif args.scheduler_type == "euler":
        scheduler = EulerDiscreteScheduler(beta_start=beta_start, beta_end=beta_end, beta_schedule="scaled_linear")
    elif args.scheduler_type == "euler-ancestral":
        scheduler = EulerAncestralDiscreteScheduler(
            beta_start=beta_start, beta_end=beta_end, beta_schedule="scaled_linear"
        )
    elif args.scheduler_type == "dpm":
        scheduler = DPMSolverMultistepScheduler(
            beta_start=beta_start, beta_end=beta_end, beta_schedule="scaled_linear"
        )
    elif args.scheduler_type == "ddim":
        scheduler = DDIMScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
    else:
        raise ValueError(f"Scheduler of type {args.scheduler_type} doesn't exist!")

    # Convert the UNet2DConditionModel models.
    if args.unet_checkpoint_path is not None:
        # image UNet
        image_unet_config = create_image_unet_diffusers_config(IMAGE_UNET_CONFIG)
        checkpoint = torch.load(args.unet_checkpoint_path)
        converted_image_unet_checkpoint = convert_vd_unet_checkpoint(
            checkpoint, image_unet_config, unet_key="model.diffusion_model.unet_image.", extract_ema=args.extract_ema
        )
        image_unet = UNet2DConditionModel(**image_unet_config)
        image_unet.load_state_dict(converted_image_unet_checkpoint)

        # text UNet
        text_unet_config = create_text_unet_diffusers_config(TEXT_UNET_CONFIG)
        converted_text_unet_checkpoint = convert_vd_unet_checkpoint(
            checkpoint, text_unet_config, unet_key="model.diffusion_model.unet_text.", extract_ema=args.extract_ema
        )
        text_unet = UNetFlatConditionModel(**text_unet_config)
        text_unet.load_state_dict(converted_text_unet_checkpoint)

    # Convert the VAE model.
    if args.vae_checkpoint_path is not None:
        vae_config = create_vae_diffusers_config(AUTOENCODER_CONFIG)
        checkpoint = torch.load(args.vae_checkpoint_path)
        converted_vae_checkpoint = convert_vd_vae_checkpoint(checkpoint, vae_config)

        vae = AutoencoderKL(**vae_config)
        vae.load_state_dict(converted_vae_checkpoint)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    image_feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")

    pipe = VersatileDiffusionPipeline(
        scheduler=scheduler,
        tokenizer=tokenizer,
        image_feature_extractor=image_feature_extractor,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        image_unet=image_unet,
        text_unet=text_unet,
        vae=vae,
    )
    pipe.save_pretrained(args.dump_path)
