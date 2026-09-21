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
"""Conversion script for the AudioLDM checkpoints."""

import argparse

import torch
import yaml
from transformers import (
    AutoTokenizer,
    ClapTextConfig,
    ClapTextModelWithProjection,
    SpeechT5HifiGan,
    SpeechT5HifiGanConfig,
)

from diffusers import (
    AudioLDMPipeline,
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders.conversion.checkpoint import convert_component_checkpoint
from diffusers.loaders.conversion.configs.audioldm import (
    DEFAULT_CONFIG,
    create_transformers_vocoder_config,
    create_unet_diffusers_config,
    create_vae_diffusers_config,
)


# Adapted from diffusers.pipelines.stable_diffusion.convert_from_ckpt.create_vae_diffusers_config


# Adapted from diffusers.pipelines.stable_diffusion.convert_from_ckpt.convert_ldm_unet_checkpoint
def convert_ldm_unet_checkpoint(checkpoint, config, path=None, extract_ema=False, **kwargs):
    return convert_component_checkpoint(checkpoint, config, "UNet2DConditionModel", extract_ema=extract_ema)


def convert_ldm_vae_checkpoint(checkpoint, config):
    return convert_component_checkpoint(checkpoint, config, "AutoencoderKL")


CLAP_EXPECTED_MISSING_KEYS = ["text_model.embeddings.token_type_ids"]


def convert_open_clap_checkpoint(checkpoint, config):
    prefix = "cond_stage_model.model."
    state = {
        key.removeprefix(prefix): value
        for key, value in checkpoint.items()
        if key.startswith(prefix) and key.startswith(prefix + "text_")
    }
    return convert_component_checkpoint(state, config.to_dict(), "ClapTextModelWithProjection")


def convert_hifigan_checkpoint(checkpoint, config):
    state = {
        key.removeprefix("first_stage_model.vocoder."): value
        for key, value in checkpoint.items()
        if key.startswith("first_stage_model.vocoder.")
    }
    return convert_component_checkpoint(state, config.to_dict(), "SpeechT5HifiGan")


# Adapted from https://huggingface.co/spaces/haoheliu/audioldm-text-to-audio-generation/blob/84a0384742a22bd80c44e903e241f0623e874f1d/audioldm/utils.py#L72-L73


def load_pipeline_from_original_audioldm_ckpt(
    checkpoint_path: str,
    original_config_file: str = None,
    image_size: int = 512,
    prediction_type: str = None,
    extract_ema: bool = False,
    scheduler_type: str = "ddim",
    num_in_channels: int = None,
    model_channels: int = None,
    num_head_channels: int = None,
    device: str = None,
    from_safetensors: bool = False,
) -> AudioLDMPipeline:
    """
    Load an AudioLDM pipeline object from a `.ckpt`/`.safetensors` file and (ideally) a `.yaml` config file.

    Although many of the arguments can be automatically inferred, some of these rely on brittle checks against the
    global step count, which will likely fail for models that have undergone further fine-tuning. Therefore, it is
    recommended that you override the default values and/or supply an `original_config_file` wherever possible.

    Args:
        checkpoint_path (`str`): Path to `.ckpt` file.
        original_config_file (`str`):
            Path to `.yaml` config file corresponding to the original architecture. If `None`, will be automatically
            set to the audioldm-s-full-v2 config.
        image_size (`int`, *optional*, defaults to 512):
            The image size that the model was trained on.
        prediction_type (`str`, *optional*):
            The prediction type that the model was trained on. If `None`, will be automatically
            inferred by looking for a key in the config. For the default config, the prediction type is `'epsilon'`.
        num_in_channels (`int`, *optional*, defaults to None):
            The number of UNet input channels. If `None`, it will be automatically inferred from the config.
        model_channels (`int`, *optional*, defaults to None):
            The number of UNet model channels. If `None`, it will be automatically inferred from the config. Override
            to 128 for the small checkpoints, 192 for the medium checkpoints and 256 for the large.
        num_head_channels (`int`, *optional*, defaults to None):
            The number of UNet head channels. If `None`, it will be automatically inferred from the config. Override
            to 32 for the small and medium checkpoints, and 64 for the large.
        scheduler_type (`str`, *optional*, defaults to 'pndm'):
            Type of scheduler to use. Should be one of `["pndm", "lms", "heun", "euler", "euler-ancestral", "dpm",
            "ddim"]`.
        extract_ema (`bool`, *optional*, defaults to `False`): Only relevant for
            checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights or not. Defaults to
            `False`. Pass `True` to extract the EMA weights. EMA weights usually yield higher quality images for
            inference. Non-EMA weights are usually better to continue fine-tuning.
        device (`str`, *optional*, defaults to `None`):
            The device to use. Pass `None` to determine automatically.
        from_safetensors (`str`, *optional*, defaults to `False`):
            If `checkpoint_path` is in `safetensors` format, load checkpoint with safetensors instead of PyTorch.
        return: An AudioLDMPipeline object representing the passed-in `.ckpt`/`.safetensors` file.
    """

    if from_safetensors:
        from safetensors import safe_open

        checkpoint = {}
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                checkpoint[key] = f.get_tensor(key)
    else:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            checkpoint = torch.load(checkpoint_path, map_location=device)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)

    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if original_config_file is None:
        original_config = DEFAULT_CONFIG
    else:
        original_config = yaml.safe_load(original_config_file)

    if num_in_channels is not None:
        original_config["model"]["params"]["unet_config"]["params"]["in_channels"] = num_in_channels

    if model_channels is not None:
        original_config["model"]["params"]["unet_config"]["params"]["model_channels"] = model_channels

    if num_head_channels is not None:
        original_config["model"]["params"]["unet_config"]["params"]["num_head_channels"] = num_head_channels

    if (
        "parameterization" in original_config["model"]["params"]
        and original_config["model"]["params"]["parameterization"] == "v"
    ):
        if prediction_type is None:
            prediction_type = "v_prediction"
    else:
        if prediction_type is None:
            prediction_type = "epsilon"

    if image_size is None:
        image_size = 512

    num_train_timesteps = original_config["model"]["params"]["timesteps"]
    beta_start = original_config["model"]["params"]["linear_start"]
    beta_end = original_config["model"]["params"]["linear_end"]

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
    # make sure scheduler works correctly with DDIM
    scheduler.register_to_config(clip_sample=False)

    if scheduler_type == "pndm":
        config = dict(scheduler.config)
        config["skip_prk_steps"] = True
        scheduler = PNDMScheduler.from_config(config)
    elif scheduler_type == "lms":
        scheduler = LMSDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == "heun":
        scheduler = HeunDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == "euler":
        scheduler = EulerDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == "euler-ancestral":
        scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == "dpm":
        scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
    elif scheduler_type == "ddim":
        scheduler = scheduler
    else:
        raise ValueError(f"Scheduler of type {scheduler_type} doesn't exist!")

    # Convert the UNet2DModel
    unet_config = create_unet_diffusers_config(original_config, image_size=image_size)
    unet = UNet2DConditionModel(**unet_config)

    converted_unet_checkpoint = convert_ldm_unet_checkpoint(
        checkpoint, unet_config, path=checkpoint_path, extract_ema=extract_ema
    )

    unet.load_state_dict(converted_unet_checkpoint)

    # Convert the VAE model
    vae_config = create_vae_diffusers_config(original_config, checkpoint=checkpoint, image_size=image_size)
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(converted_vae_checkpoint)

    # Convert the text model
    # AudioLDM uses the same configuration and tokenizer as the original CLAP model
    config = ClapTextConfig.from_pretrained("laion/clap-htsat-unfused")
    tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

    converted_text_model = convert_open_clap_checkpoint(checkpoint, config)
    text_model = ClapTextModelWithProjection(config)

    missing_keys, unexpected_keys = text_model.load_state_dict(converted_text_model, strict=False)
    # we expect not to have token_type_ids in our original state dict so let's ignore them
    missing_keys = list(set(missing_keys) - set(CLAP_EXPECTED_MISSING_KEYS))

    if len(unexpected_keys) > 0:
        raise ValueError(f"Unexpected keys when loading CLAP model: {unexpected_keys}")

    if len(missing_keys) > 0:
        raise ValueError(f"Missing keys when loading CLAP model: {missing_keys}")

    # Convert the vocoder model
    vocoder_config = create_transformers_vocoder_config(original_config)
    vocoder_config = SpeechT5HifiGanConfig(**vocoder_config)
    converted_vocoder_checkpoint = convert_hifigan_checkpoint(checkpoint, vocoder_config)

    vocoder = SpeechT5HifiGan(vocoder_config)
    vocoder.load_state_dict(converted_vocoder_checkpoint)

    # Instantiate the diffusers pipeline
    pipe = AudioLDMPipeline(
        vae=vae,
        text_encoder=text_model,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        vocoder=vocoder,
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
        "--num_in_channels",
        default=None,
        type=int,
        help="The number of input channels. If `None` number of input channels will be automatically inferred.",
    )
    parser.add_argument(
        "--model_channels",
        default=None,
        type=int,
        help="The number of UNet model channels. If `None`, it will be automatically inferred from the config. Override"
        " to 128 for the small checkpoints, 192 for the medium checkpoints and 256 for the large.",
    )
    parser.add_argument(
        "--num_head_channels",
        default=None,
        type=int,
        help="The number of UNet head channels. If `None`, it will be automatically inferred from the config. Override"
        " to 32 for the small and medium checkpoints, and 64 for the large.",
    )
    parser.add_argument(
        "--scheduler_type",
        default="ddim",
        type=str,
        help="Type of scheduler to use. Should be one of ['pndm', 'lms', 'ddim', 'euler', 'euler-ancestral', 'dpm']",
    )
    parser.add_argument(
        "--image_size",
        default=None,
        type=int,
        help=("The image size that the model was trained on."),
    )
    parser.add_argument(
        "--prediction_type",
        default=None,
        type=str,
        help=("The prediction type that the model was trained on."),
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
        "--from_safetensors",
        action="store_true",
        help="If `--checkpoint_path` is in `safetensors` format, load checkpoint with safetensors instead of PyTorch.",
    )
    parser.add_argument(
        "--to_safetensors",
        action="store_true",
        help="Whether to store pipeline in safetensors format or not.",
    )
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")
    parser.add_argument("--device", type=str, help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)")
    args = parser.parse_args()

    pipe = load_pipeline_from_original_audioldm_ckpt(
        checkpoint_path=args.checkpoint_path,
        original_config_file=args.original_config_file,
        image_size=args.image_size,
        prediction_type=args.prediction_type,
        extract_ema=args.extract_ema,
        scheduler_type=args.scheduler_type,
        num_in_channels=args.num_in_channels,
        model_channels=args.model_channels,
        num_head_channels=args.num_head_channels,
        from_safetensors=args.from_safetensors,
        device=args.device,
    )
    pipe.save_pretrained(args.dump_path, safe_serialization=args.to_safetensors)
