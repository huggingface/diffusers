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
"""Conversion script for the AudioLDM2 checkpoints."""

import argparse
from typing import List, Union

import torch
import yaml
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    ClapConfig,
    ClapModel,
    GPT2Config,
    GPT2Model,
    SpeechT5HifiGan,
    SpeechT5HifiGanConfig,
    T5Config,
    T5EncoderModel,
)

from diffusers import (
    AudioLDM2Pipeline,
    AudioLDM2ProjectionModel,
    AudioLDM2UNet2DConditionModel,
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.loaders.conversion.checkpoint import convert_component_checkpoint
from diffusers.loaders.conversion.configs.audioldm2 import (
    DEFAULT_CONFIG,
    create_transformers_vocoder_config,
    create_unet_diffusers_config,
    create_vae_diffusers_config,
)
from diffusers.utils import is_safetensors_available
from diffusers.utils.import_utils import BACKENDS_MAPPING


# Adapted from diffusers.pipelines.stable_diffusion.convert_from_ckpt.create_vae_diffusers_config


def convert_ldm_unet_checkpoint(checkpoint, config, path=None, extract_ema=False, **kwargs):
    return convert_component_checkpoint(checkpoint, config, "AudioLDM2UNet2DConditionModel", extract_ema=extract_ema)


def convert_ldm_vae_checkpoint(checkpoint, config):
    return convert_component_checkpoint(checkpoint, config, "AutoencoderKL")


CLAP_EXPECTED_MISSING_KEYS = ["text_model.embeddings.token_type_ids"]


def convert_open_clap_checkpoint(checkpoint, config):
    prefix = "clap.model."
    state = {key.removeprefix(prefix): value for key, value in checkpoint.items() if key.startswith(prefix)}
    return convert_component_checkpoint(state, config.to_dict(), "ClapModel")


def extract_sub_model(checkpoint, key_prefix):
    """
    Takes a state dict and returns the state dict for a particular sub-model.
    """

    sub_model_state_dict = {}
    keys = list(checkpoint.keys())
    for key in keys:
        if key.startswith(key_prefix):
            sub_model_state_dict[key.replace(key_prefix, "")] = checkpoint.get(key)

    return sub_model_state_dict


def convert_hifigan_checkpoint(checkpoint, config):
    state = {
        key.removeprefix("first_stage_model.vocoder."): value
        for key, value in checkpoint.items()
        if key.startswith("first_stage_model.vocoder.")
    }
    return convert_component_checkpoint(state, config.to_dict(), "SpeechT5HifiGan")


def convert_projection_checkpoint(checkpoint):
    prefix = "cond_stage_models.0."
    modules = ("start_of_sequence_tokens.", "end_of_sequence_tokens.", "input_sequence_embed_linear.")
    state = {
        key.removeprefix(prefix): value
        for key, value in checkpoint.items()
        if key.startswith(prefix) and key.removeprefix(prefix).startswith(modules)
    }
    return convert_component_checkpoint(state, {}, "AudioLDM2ProjectionModel")


# Adapted from https://github.com/haoheliu/AudioLDM2/blob/81ad2c6ce015c1310387695e2dae975a7d2ed6fd/audioldm2/utils.py#L143


def load_pipeline_from_original_AudioLDM2_ckpt(
    checkpoint_path: str,
    original_config_file: str = None,
    image_size: int = 1024,
    prediction_type: str = None,
    extract_ema: bool = False,
    scheduler_type: str = "ddim",
    cross_attention_dim: Union[List, List[List]] = None,
    transformer_layers_per_block: int = None,
    device: str = None,
    from_safetensors: bool = False,
) -> AudioLDM2Pipeline:
    """
    Load an AudioLDM2 pipeline object from a `.ckpt`/`.safetensors` file and (ideally) a `.yaml` config file.

    Although many of the arguments can be automatically inferred, some of these rely on brittle checks against the
    global step count, which will likely fail for models that have undergone further fine-tuning. Therefore, it is
    recommended that you override the default values and/or supply an `original_config_file` wherever possible.

    Args:
        checkpoint_path (`str`): Path to `.ckpt` file.
        original_config_file (`str`):
            Path to `.yaml` config file corresponding to the original architecture. If `None`, will be automatically
            set to the AudioLDM2 base config.
        image_size (`int`, *optional*, defaults to 1024):
            The image size that the model was trained on.
        prediction_type (`str`, *optional*):
            The prediction type that the model was trained on. If `None`, will be automatically
            inferred by looking for a key in the config. For the default config, the prediction type is `'epsilon'`.
        scheduler_type (`str`, *optional*, defaults to 'ddim'):
            Type of scheduler to use. Should be one of `["pndm", "lms", "heun", "euler", "euler-ancestral", "dpm",
            "ddim"]`.
        cross_attention_dim (`list`, *optional*, defaults to `None`):
            The dimension of the cross-attention layers. If `None`, the cross-attention dimension will be
            automatically inferred. Set to `[768, 1024]` for the base model, or `[768, 1024, None]` for the large model.
        transformer_layers_per_block (`int`, *optional*, defaults to `None`):
            The number of transformer layers in each transformer block. If `None`, number of layers will be "
             "automatically inferred. Set to `1` for the base model, or `2` for the large model.
        extract_ema (`bool`, *optional*, defaults to `False`): Only relevant for
            checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights or not. Defaults to
            `False`. Pass `True` to extract the EMA weights. EMA weights usually yield higher quality images for
            inference. Non-EMA weights are usually better to continue fine-tuning.
        device (`str`, *optional*, defaults to `None`):
            The device to use. Pass `None` to determine automatically.
        from_safetensors (`str`, *optional*, defaults to `False`):
            If `checkpoint_path` is in `safetensors` format, load checkpoint with safetensors instead of PyTorch.
        return: An AudioLDM2Pipeline object representing the passed-in `.ckpt`/`.safetensors` file.
    """

    if from_safetensors:
        if not is_safetensors_available():
            raise ValueError(BACKENDS_MAPPING["safetensors"][1])

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

    if image_size is not None:
        original_config["model"]["params"]["unet_config"]["params"]["image_size"] = image_size

    if cross_attention_dim is not None:
        original_config["model"]["params"]["unet_config"]["params"]["context_dim"] = cross_attention_dim

    if transformer_layers_per_block is not None:
        original_config["model"]["params"]["unet_config"]["params"]["transformer_depth"] = transformer_layers_per_block

    if (
        "parameterization" in original_config["model"]["params"]
        and original_config["model"]["params"]["parameterization"] == "v"
    ):
        if prediction_type is None:
            prediction_type = "v_prediction"
    else:
        if prediction_type is None:
            prediction_type = "epsilon"

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
    unet = AudioLDM2UNet2DConditionModel(**unet_config)

    converted_unet_checkpoint = convert_ldm_unet_checkpoint(
        checkpoint, unet_config, path=checkpoint_path, extract_ema=extract_ema
    )

    unet.load_state_dict(converted_unet_checkpoint)

    # Convert the VAE model
    vae_config = create_vae_diffusers_config(original_config, checkpoint=checkpoint, image_size=image_size)
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(converted_vae_checkpoint)

    # Convert the joint audio-text encoding model
    clap_config = ClapConfig.from_pretrained("laion/clap-htsat-unfused")
    clap_config.audio_config.update(
        {
            "patch_embeds_hidden_size": 128,
            "hidden_size": 1024,
            "depths": [2, 2, 12, 2],
        }
    )
    # AudioLDM2 uses the same tokenizer and feature extractor as the original CLAP model
    clap_tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")
    clap_feature_extractor = AutoFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")

    converted_clap_model = convert_open_clap_checkpoint(checkpoint, clap_config)
    clap_model = ClapModel(clap_config)

    missing_keys, unexpected_keys = clap_model.load_state_dict(converted_clap_model, strict=False)
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

    # Convert the Flan-T5 encoder model: AudioLDM2 uses the same configuration and tokenizer as the original Flan-T5 large model
    t5_config = T5Config.from_pretrained("google/flan-t5-large")
    converted_t5_checkpoint = extract_sub_model(checkpoint, key_prefix="cond_stage_models.1.model.")

    t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    # hard-coded in the original implementation (i.e. not retrievable from the config)
    t5_tokenizer.model_max_length = 128
    t5_model = T5EncoderModel(t5_config)
    t5_model.load_state_dict(converted_t5_checkpoint)

    # Convert the GPT2 encoder model: AudioLDM2 uses the same configuration as the original GPT2 base model
    gpt2_config = GPT2Config.from_pretrained("gpt2")
    gpt2_model = GPT2Model(gpt2_config)
    gpt2_model.config.max_new_tokens = original_config["model"]["params"]["cond_stage_config"][
        "crossattn_audiomae_generated"
    ]["params"]["sequence_gen_length"]

    converted_gpt2_checkpoint = extract_sub_model(checkpoint, key_prefix="cond_stage_models.0.model.")
    gpt2_model.load_state_dict(converted_gpt2_checkpoint)

    # Convert the extra embedding / projection layers
    projection_model = AudioLDM2ProjectionModel(clap_config.projection_dim, t5_config.d_model, gpt2_config.n_embd)

    converted_projection_checkpoint = convert_projection_checkpoint(checkpoint)
    projection_model.load_state_dict(converted_projection_checkpoint)

    # Instantiate the diffusers pipeline
    pipe = AudioLDM2Pipeline(
        vae=vae,
        text_encoder=clap_model,
        text_encoder_2=t5_model,
        projection_model=projection_model,
        language_model=gpt2_model,
        tokenizer=clap_tokenizer,
        tokenizer_2=t5_tokenizer,
        feature_extractor=clap_feature_extractor,
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
        "--cross_attention_dim",
        default=None,
        type=int,
        nargs="+",
        help="The dimension of the cross-attention layers. If `None`, the cross-attention dimension will be "
        "automatically inferred. Set to `768+1024` for the base model, or `768+1024+640` for the large model",
    )
    parser.add_argument(
        "--transformer_layers_per_block",
        default=None,
        type=int,
        help="The number of transformer layers in each transformer block. If `None`, number of layers will be "
        "automatically inferred. Set to `1` for the base model, or `2` for the large model.",
    )
    parser.add_argument(
        "--scheduler_type",
        default="ddim",
        type=str,
        help="Type of scheduler to use. Should be one of ['pndm', 'lms', 'ddim', 'euler', 'euler-ancestral', 'dpm']",
    )
    parser.add_argument(
        "--image_size",
        default=1048,
        type=int,
        help="The image size that the model was trained on.",
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

    pipe = load_pipeline_from_original_AudioLDM2_ckpt(
        checkpoint_path=args.checkpoint_path,
        original_config_file=args.original_config_file,
        image_size=args.image_size,
        prediction_type=args.prediction_type,
        extract_ema=args.extract_ema,
        scheduler_type=args.scheduler_type,
        cross_attention_dim=args.cross_attention_dim,
        transformer_layers_per_block=args.transformer_layers_per_block,
        from_safetensors=args.from_safetensors,
        device=args.device,
    )
    pipe.save_pretrained(args.dump_path, safe_serialization=args.to_safetensors)
