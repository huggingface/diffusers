# Convert the original UniDiffuser checkpoints into diffusers equivalents.

import argparse

import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
    GPT2Tokenizer,
)

from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    UniDiffuserModel,
    UniDiffuserPipeline,
    UniDiffuserTextDecoder,
)
from diffusers.loaders.conversion.checkpoint import convert_component_checkpoint
from diffusers.loaders.conversion.configs.unidiffuser import (
    SCHEDULER_CONFIG,
    create_text_decoder_config,
    create_unidiffuser_unet_config,
    create_vae_diffusers_config,
)


# Modified from diffusers.pipelines.stable_diffusion.convert_from_ckpt.assign_to_checkpoint
# config.num_head_channels => num_head_channels


# Hardcoded configs for test versions of the UniDiffuser models, corresponding to those in the fast default tests.


# Hardcoded configs for the UniDiffuser V1 model at https://huggingface.co/thu-ml/unidiffuser-v1
# See also https://github.com/thu-ml/unidiffuser/blob/main/configs/sample_unidiffuser_v1.py


# From https://huggingface.co/gpt2/blob/main/config.json, the GPT2 checkpoint used by UniDiffuser


# Based on diffusers.pipelines.stable_diffusion.convert_from_ckpt.convert_ldm_vae_checkpoint
def convert_vae_to_diffusers(ckpt, diffusers_model, **kwargs):
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    converted = convert_component_checkpoint(state, dict(diffusers_model.config), "AutoencoderKL")
    diffusers_model.load_state_dict(converted, strict=True)
    return diffusers_model


def convert_uvit_to_diffusers(ckpt, diffusers_model, **kwargs):
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    converted = convert_component_checkpoint(state, dict(diffusers_model.config), "UniDiffuserModel")
    diffusers_model.load_state_dict(converted, strict=True)
    return diffusers_model


def convert_caption_decoder_to_diffusers(ckpt, diffusers_model, **kwargs):
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    converted = convert_component_checkpoint(state, dict(diffusers_model.config), "UniDiffuserTextDecoder")
    diffusers_model.load_state_dict(converted, strict=True)
    return diffusers_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--caption_decoder_checkpoint_path",
        default=None,
        type=str,
        required=False,
        help="Path to caption decoder checkpoint to convert.",
    )
    parser.add_argument(
        "--uvit_checkpoint_path", default=None, type=str, required=False, help="Path to U-ViT checkpoint to convert."
    )
    parser.add_argument(
        "--vae_checkpoint_path",
        default=None,
        type=str,
        required=False,
        help="Path to VAE checkpoint to convert.",
    )
    parser.add_argument(
        "--pipeline_output_path",
        default=None,
        type=str,
        required=True,
        help="Path to save the output pipeline to.",
    )
    parser.add_argument(
        "--config_type",
        default="test",
        type=str,
        help=(
            "Config type to use. Should be 'test' to create small models for testing or 'big' to convert a full"
            " checkpoint."
        ),
    )
    parser.add_argument(
        "--version",
        default=0,
        type=int,
        help="The UniDiffuser model type to convert to. Should be 0 for UniDiffuser-v0 and 1 for UniDiffuser-v1.",
    )
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        help="Whether to use safetensors/safe seialization when saving the pipeline.",
    )

    args = parser.parse_args()

    # Convert the VAE model.
    if args.vae_checkpoint_path is not None:
        vae_config = create_vae_diffusers_config(args.config_type)
        vae = AutoencoderKL(**vae_config)
        vae = convert_vae_to_diffusers(args.vae_checkpoint_path, vae)

    # Convert the U-ViT ("unet") model.
    if args.uvit_checkpoint_path is not None:
        unet_config = create_unidiffuser_unet_config(args.config_type, args.version)
        unet = UniDiffuserModel(**unet_config)
        unet = convert_uvit_to_diffusers(args.uvit_checkpoint_path, unet)

    # Convert the caption decoder ("text_decoder") model.
    if args.caption_decoder_checkpoint_path is not None:
        text_decoder_config = create_text_decoder_config(args.config_type)
        text_decoder = UniDiffuserTextDecoder(**text_decoder_config)
        text_decoder = convert_caption_decoder_to_diffusers(args.caption_decoder_checkpoint_path, text_decoder)

    # Scheduler is the same for both the test and big models.
    scheduler_config = SCHEDULER_CONFIG
    scheduler = DPMSolverMultistepScheduler(
        beta_start=scheduler_config.beta_start,
        beta_end=scheduler_config.beta_end,
        beta_schedule=scheduler_config.beta_schedule,
        solver_order=scheduler_config.solver_order,
    )

    if args.config_type == "test":
        # Make a small random CLIPTextModel
        torch.manual_seed(0)
        clip_text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(clip_text_encoder_config)
        clip_tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        # Make a small random CLIPVisionModel and accompanying CLIPImageProcessor
        torch.manual_seed(0)
        clip_image_encoder_config = CLIPVisionConfig(
            image_size=32,
            patch_size=2,
            num_channels=3,
            hidden_size=32,
            projection_dim=32,
            num_hidden_layers=5,
            num_attention_heads=4,
            intermediate_size=37,
            dropout=0.1,
            attention_dropout=0.1,
            initializer_range=0.02,
        )
        image_encoder = CLIPVisionModelWithProjection(clip_image_encoder_config)
        image_processor = CLIPImageProcessor(crop_size=32, size=32)

        # Note that the text_decoder should already have its token embeddings resized.
        text_tokenizer = GPT2Tokenizer.from_pretrained("hf-internal-testing/tiny-random-GPT2Model")
        eos = "<|EOS|>"
        special_tokens_dict = {"eos_token": eos}
        text_tokenizer.add_special_tokens(special_tokens_dict)
    elif args.config_type == "big":
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Note that the text_decoder should already have its token embeddings resized.
        text_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        eos = "<|EOS|>"
        special_tokens_dict = {"eos_token": eos}
        text_tokenizer.add_special_tokens(special_tokens_dict)
    else:
        raise NotImplementedError(
            f"Config type {args.config_type} is not implemented, currently only config types"
            " 'test' and 'big' are available."
        )

    pipeline = UniDiffuserPipeline(
        vae=vae,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        clip_image_processor=image_processor,
        clip_tokenizer=clip_tokenizer,
        text_decoder=text_decoder,
        text_tokenizer=text_tokenizer,
        unet=unet,
        scheduler=scheduler,
    )
    pipeline.save_pretrained(args.pipeline_output_path, safe_serialization=args.safe_serialization)
