import argparse

import torch
import yaml
from transformers import (
    CLIPProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionGLIGENPipeline,
    StableDiffusionGLIGENTextImagePipeline,
    UNet2DConditionModel,
)
from diffusers.loaders.conversion.checkpoint import convert_component_checkpoint
from diffusers.loaders.conversion.configs.gligen import create_unet_config, create_vae_config


def convert_open_clip_checkpoint(checkpoint):
    model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    state = checkpoint["text_encoder"]
    config = model.config.to_dict()
    if any(key.startswith("transformer.text_model.") for key in state):
        state = {key.removeprefix("transformer."): value for key, value in state.items()}
        config["original_format"] = "clip"
    else:
        config["original_format"] = "openclip"
    state = dict(state)
    state.pop("text_model.embeddings.position_ids", None)
    converted = convert_component_checkpoint(state, config, "CLIPTextModel")
    model.load_state_dict(converted, strict=True)
    return model


def convert_gligen_vae_checkpoint(checkpoint, config):
    return convert_component_checkpoint(checkpoint["autoencoder"], config, "AutoencoderKL")


def convert_gligen_unet_checkpoint(checkpoint, config, path=None, extract_ema=False):
    return convert_component_checkpoint(checkpoint["model"], config, "UNet2DConditionModel", extract_ema=extract_ema)


def convert_gligen_to_diffusers(
    checkpoint_path: str,
    original_config_file: str,
    attention_type: str,
    image_size: int = 512,
    extract_ema: bool = False,
    num_in_channels: int = None,
    device: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if "global_step" in checkpoint:
        checkpoint["global_step"]
    else:
        print("global_step key not found in model")

    original_config = yaml.safe_load(original_config_file)

    if num_in_channels is not None:
        original_config["model"]["params"]["in_channels"] = num_in_channels

    num_train_timesteps = original_config["diffusion"]["params"]["timesteps"]
    beta_start = original_config["diffusion"]["params"]["linear_start"]
    beta_end = original_config["diffusion"]["params"]["linear_end"]

    scheduler = DDIMScheduler(
        beta_end=beta_end,
        beta_schedule="scaled_linear",
        beta_start=beta_start,
        num_train_timesteps=num_train_timesteps,
        steps_offset=1,
        clip_sample=False,
        set_alpha_to_one=False,
        prediction_type="epsilon",
    )

    # Convert the UNet2DConditionalModel model
    unet_config = create_unet_config(original_config, image_size, attention_type)
    unet = UNet2DConditionModel(**unet_config)

    converted_unet_checkpoint = convert_gligen_unet_checkpoint(
        checkpoint, unet_config, path=checkpoint_path, extract_ema=extract_ema
    )

    unet.load_state_dict(converted_unet_checkpoint)

    # Convert the VAE model
    vae_config = create_vae_config(original_config, image_size)
    converted_vae_checkpoint = convert_gligen_vae_checkpoint(checkpoint, vae_config)

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(converted_vae_checkpoint)

    # Convert the text model
    text_encoder = convert_open_clip_checkpoint(checkpoint)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    if attention_type == "gated-text-image":
        image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        pipe = StableDiffusionGLIGENTextImagePipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            image_encoder=image_encoder,
            processor=processor,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
        )
    elif attention_type == "gated":
        pipe = StableDiffusionGLIGENPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
        )

    return pipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path", default=None, type=str, required=True, help="Path to the checkpoint to convert."
    )
    parser.add_argument(
        "--original_config_file",
        default=None,
        type=str,
        required=True,
        help="The YAML config file corresponding to the gligen architecture.",
    )
    parser.add_argument(
        "--num_in_channels",
        default=None,
        type=int,
        help="The number of input channels. If `None` number of input channels will be automatically inferred.",
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
    parser.add_argument(
        "--attention_type",
        default=None,
        type=str,
        required=True,
        help="Type of attention ex: gated or gated-text-image",
    )
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")
    parser.add_argument("--device", type=str, help="Device to use.")
    parser.add_argument("--half", action="store_true", help="Save weights in half precision.")

    args = parser.parse_args()

    pipe = convert_gligen_to_diffusers(
        checkpoint_path=args.checkpoint_path,
        original_config_file=args.original_config_file,
        attention_type=args.attention_type,
        extract_ema=args.extract_ema,
        num_in_channels=args.num_in_channels,
        device=args.device,
    )

    if args.half:
        pipe.to(dtype=torch.float16)

    pipe.save_pretrained(args.dump_path)
