"""
This script requires you to build `LAVIS` from source, since the pip version doesn't have BLIP Diffusion. Follow instructions here: https://github.com/salesforce/LAVIS/tree/main.
"""

import argparse
import os
import tempfile

import torch
from lavis.models import load_model_and_preprocess
from transformers import CLIPTokenizer
from transformers.models.blip_2.configuration_blip_2 import Blip2Config

from diffusers import (
    AutoencoderKL,
    PNDMScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders.conversion import get_conversion
from diffusers.loaders.conversion.configs.blip_diffusion import BLIP2_CONFIG
from diffusers.pipelines import BlipDiffusionPipeline
from diffusers.pipelines.deprecated.blip_diffusion.blip_image_processing import BlipImageProcessor
from diffusers.pipelines.deprecated.blip_diffusion.modeling_blip2 import Blip2QFormerModel
from diffusers.pipelines.deprecated.blip_diffusion.modeling_ctx_clip import ContextCLIPTextModel


blip2config = Blip2Config(**BLIP2_CONFIG)


def qformer_model_from_original_config():
    qformer = Blip2QFormerModel(blip2config)
    return qformer


def qformer_original_checkpoint_to_diffusers_checkpoint(model):
    prefixes = ("blip.Qformer.bert.", "blip.query_tokens", "blip.visual_encoder.", "blip.ln_vision.", "proj_layer.")
    state = {key: value for key, value in model.items() if key.startswith(prefixes)}
    # The source position IDs are deterministic, and the shared conversion recreates them.
    state.pop("blip.Qformer.bert.embeddings.position_ids", None)
    return get_conversion("Blip2QFormerModel", blip2config.to_dict()).to_diffusers(state)


def get_qformer(model):
    print("loading qformer")

    qformer = qformer_model_from_original_config()
    qformer_diffusers_checkpoint = qformer_original_checkpoint_to_diffusers_checkpoint(model)

    load_checkpoint_to_model(qformer_diffusers_checkpoint, qformer)

    print("done loading qformer")
    return qformer


def load_checkpoint_to_model(checkpoint, model):
    with tempfile.NamedTemporaryFile(delete=False) as file:
        torch.save(checkpoint, file.name)
        del checkpoint
        model.load_state_dict(torch.load(file.name), strict=False)

    os.remove(file.name)


def save_blip_diffusion_model(model, args):
    qformer = get_qformer(model)
    qformer.eval()

    text_encoder = ContextCLIPTextModel.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="unet")
    vae.eval()
    text_encoder.eval()
    scheduler = PNDMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        set_alpha_to_one=False,
        skip_prk_steps=True,
    )
    tokenizer = CLIPTokenizer.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="tokenizer")
    image_processor = BlipImageProcessor()
    blip_diffusion = BlipDiffusionPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        qformer=qformer,
        image_processor=image_processor,
    )
    blip_diffusion.save_pretrained(args.checkpoint_path)


def main(args):
    model, _, _ = load_model_and_preprocess("blip_diffusion", "base", device="cpu", is_eval=True)
    save_blip_diffusion_model(model.state_dict(), args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", default=None, type=str, required=True, help="Path to the output model.")
    args = parser.parse_args()

    main(args)
