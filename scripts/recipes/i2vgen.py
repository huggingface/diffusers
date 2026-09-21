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
"""Conversion script for the LDM checkpoints."""

import argparse

import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers import DDIMScheduler, I2VGenXLPipeline, I2VGenXLUNet, StableDiffusionPipeline
from diffusers.loaders.conversion.checkpoint import convert_component_checkpoint


CLIP_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


def convert_ldm_unet_checkpoint(checkpoint, config, path=None, extract_ema=False, **kwargs):
    return convert_component_checkpoint(checkpoint, config, "I2VGenXLUNet", extract_ema=extract_ema)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--unet_checkpoint_path", default=None, type=str, required=True, help="Path to the checkpoint to convert."
    )
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()

    # UNet
    unet_checkpoint = torch.load(args.unet_checkpoint_path, map_location="cpu")
    unet_checkpoint = unet_checkpoint["state_dict"]
    unet = I2VGenXLUNet(sample_size=32)

    converted_ckpt = convert_ldm_unet_checkpoint(unet_checkpoint, unet.config)

    diff_0 = set(unet.state_dict().keys()) - set(converted_ckpt.keys())
    diff_1 = set(converted_ckpt.keys()) - set(unet.state_dict().keys())

    assert len(diff_0) == len(diff_1) == 0, "Converted weights don't match"

    unet.load_state_dict(converted_ckpt, strict=True)

    # vae
    temp_pipe = StableDiffusionPipeline.from_single_file(
        "https://huggingface.co/ali-vilab/i2vgen-xl/blob/main/models/v2-1_512-ema-pruned.ckpt"
    )
    vae = temp_pipe.vae
    del temp_pipe

    # text encoder and tokenizer
    text_encoder = CLIPTextModel.from_pretrained(CLIP_ID)
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_ID)

    # image encoder and feature extractor
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(CLIP_ID)
    feature_extractor = CLIPImageProcessor.from_pretrained(CLIP_ID)

    # scheduler
    # https://github.com/ali-vilab/i2vgen-xl/blob/main/configs/i2vgen_xl_train.yaml
    scheduler = DDIMScheduler(
        beta_schedule="squaredcos_cap_v2",
        rescale_betas_zero_snr=True,
        set_alpha_to_one=True,
        clip_sample=False,
        steps_offset=1,
        timestep_spacing="leading",
        prediction_type="v_prediction",
    )

    # final
    pipeline = I2VGenXLPipeline(
        unet=unet,
        vae=vae,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
    )

    pipeline.save_pretrained(args.dump_path, push_to_hub=args.push_to_hub)
