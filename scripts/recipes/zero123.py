"""
This script modified from
https://github.com/huggingface/diffusers/blob/bc691231360a4cbc7d19a58742ebb8ed0f05e027/scripts/convert_original_stable_diffusion_to_diffusers.py

Convert original Zero1to3 checkpoint to diffusers checkpoint.

# run the convert script
$ python recipes/zero123.py \
   --checkpoint_path /path/zero123/105000.ckpt \
   --dump_path ./zero1to3 \
   --original_config_file /path/zero123/configs/sd-objaverse-finetune-c_concat-256.yaml
```
"""

import argparse

import torch
import yaml
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from pipeline_zero1to3 import CCProjection, Zero1to3StableDiffusionPipeline
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)

from diffusers.loaders.conversion import get_conversion
from diffusers.loaders.conversion.checkpoint import convert_component_checkpoint
from diffusers.loaders.conversion.configs.zero123 import create_unet_diffusers_config, create_vae_diffusers_config
from diffusers.models import (
    AutoencoderKL,
    UNet2DConditionModel,
)
from diffusers.schedulers import DDIMScheduler
from diffusers.utils import logging


logger = logging.get_logger(__name__)


def convert_ldm_unet_checkpoint(checkpoint, config, path=None, extract_ema=False, **kwargs):
    return convert_component_checkpoint(checkpoint, config, "UNet2DConditionModel", extract_ema=extract_ema)


def convert_ldm_vae_checkpoint(checkpoint, config):
    return convert_component_checkpoint(checkpoint, config, "AutoencoderKL")


def convert_from_original_zero123_ckpt(checkpoint_path, original_config_file, extract_ema, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt["global_step"]
    checkpoint = ckpt["state_dict"]
    del ckpt
    torch.cuda.empty_cache()

    original_config = yaml.safe_load(original_config_file)
    original_config["model"]["params"]["cond_stage_config"]["target"].split(".")[-1]
    num_in_channels = 8
    original_config["model"]["params"]["unet_config"]["params"]["in_channels"] = num_in_channels
    prediction_type = "epsilon"
    image_size = 256
    num_train_timesteps = getattr(original_config["model"]["params"], "timesteps", None) or 1000

    beta_start = getattr(original_config["model"]["params"], "linear_start", None) or 0.02
    beta_end = getattr(original_config["model"]["params"], "linear_end", None) or 0.085
    scheduler = DDIMScheduler(
        beta_end=beta_end,
        beta_schedule="scaled_linear",
        beta_start=beta_start,
        num_train_timesteps=num_train_timesteps,
        steps_offset=1,
        clip_sample=False,
        set_alpha_to_one=False,
        prediction_type=prediction_type,
    )
    scheduler.register_to_config(clip_sample=False)

    # Convert the UNet2DConditionModel model.
    upcast_attention = None
    unet_config = create_unet_diffusers_config(original_config, image_size=image_size)
    unet_config["upcast_attention"] = upcast_attention
    with init_empty_weights():
        unet = UNet2DConditionModel(**unet_config)
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(
        checkpoint, unet_config, path=None, extract_ema=extract_ema
    )
    for param_name, param in converted_unet_checkpoint.items():
        set_module_tensor_to_device(unet, param_name, "cpu", value=param)

    # Convert the VAE model.
    vae_config = create_vae_diffusers_config(original_config, image_size=image_size)
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)

    if (
        "model" in original_config
        and "params" in original_config["model"]
        and "scale_factor" in original_config["model"]["params"]
    ):
        vae_scaling_factor = original_config["model"]["params"]["scale_factor"]
    else:
        vae_scaling_factor = 0.18215  # default SD scaling factor

    vae_config["scaling_factor"] = vae_scaling_factor

    with init_empty_weights():
        vae = AutoencoderKL(**vae_config)

    for param_name, param in converted_vae_checkpoint.items():
        set_module_tensor_to_device(vae, param_name, "cpu", value=param)

    feature_extractor = CLIPImageProcessor.from_pretrained(
        "lambdalabs/sd-image-variations-diffusers", subfolder="feature_extractor"
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "lambdalabs/sd-image-variations-diffusers", subfolder="image_encoder"
    )

    cc_projection = CCProjection()
    cc_projection.load_state_dict(
        get_conversion("CCProjection", {}).to_diffusers(
            {key: value.cpu() for key, value in checkpoint.items() if key.startswith("cc_projection.")}
        )
    )

    pipe = Zero1to3StableDiffusionPipeline(
        vae, image_encoder, unet, scheduler, None, feature_extractor, cc_projection, requires_safety_checker=False
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
        help="The YAML config file corresponding to the original architecture.",
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
        "--to_safetensors",
        action="store_true",
        help="Whether to store pipeline in safetensors format or not.",
    )
    parser.add_argument("--half", action="store_true", help="Save weights in half precision.")
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")
    parser.add_argument("--device", type=str, help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)")
    args = parser.parse_args()

    pipe = convert_from_original_zero123_ckpt(
        checkpoint_path=args.checkpoint_path,
        original_config_file=args.original_config_file,
        extract_ema=args.extract_ema,
        device=args.device,
    )

    if args.half:
        pipe.to(dtype=torch.float16)

    pipe.save_pretrained(args.dump_path, safe_serialization=args.to_safetensors)
