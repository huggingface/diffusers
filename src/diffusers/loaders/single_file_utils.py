# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
""" Conversion script for the Stable Diffusion checkpoints."""

import os
import re
from contextlib import nullcontext
from io import BytesIO
from urllib.parse import urlparse

import requests
import yaml

from ..models.modeling_utils import load_state_dict
from ..schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EDMDPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from ..utils import is_accelerate_available, is_transformers_available, logging
from ..utils.hub_utils import _get_model_file


if is_transformers_available():
    from transformers import (
        CLIPTextConfig,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPTokenizer,
    )

if is_accelerate_available():
    from accelerate import init_empty_weights

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

CONFIG_URLS = {
    "v1": "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml",
    "v2": "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml",
    "xl": "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml",
    "xl_refiner": "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_refiner.yaml",
    "upscale": "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/x4-upscaling.yaml",
    "controlnet": "https://raw.githubusercontent.com/lllyasviel/ControlNet/main/models/cldm_v15.yaml",
}

CHECKPOINT_KEY_NAMES = {
    "v2": "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight",
    "xl_base": "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias",
    "xl_refiner": "conditioner.embedders.0.model.transformer.resblocks.9.mlp.c_proj.bias",
}

SCHEDULER_DEFAULT_CONFIG = {
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "interpolation_type": "linear",
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "sample_max_value": 1.0,
    "set_alpha_to_one": False,
    "skip_prk_steps": True,
    "steps_offset": 1,
    "timestep_spacing": "leading",
}

DIFFUSERS_TO_LDM_MAPPING = {
    "unet": {
        "layers": {
            "time_embedding.linear_1.weight": "time_embed.0.weight",
            "time_embedding.linear_1.bias": "time_embed.0.bias",
            "time_embedding.linear_2.weight": "time_embed.2.weight",
            "time_embedding.linear_2.bias": "time_embed.2.bias",
            "conv_in.weight": "input_blocks.0.0.weight",
            "conv_in.bias": "input_blocks.0.0.bias",
            "conv_norm_out.weight": "out.0.weight",
            "conv_norm_out.bias": "out.0.bias",
            "conv_out.weight": "out.2.weight",
            "conv_out.bias": "out.2.bias",
        },
        "class_embed_type": {
            "class_embedding.linear_1.weight": "label_emb.0.0.weight",
            "class_embedding.linear_1.bias": "label_emb.0.0.bias",
            "class_embedding.linear_2.weight": "label_emb.0.2.weight",
            "class_embedding.linear_2.bias": "label_emb.0.2.bias",
        },
        "addition_embed_type": {
            "add_embedding.linear_1.weight": "label_emb.0.0.weight",
            "add_embedding.linear_1.bias": "label_emb.0.0.bias",
            "add_embedding.linear_2.weight": "label_emb.0.2.weight",
            "add_embedding.linear_2.bias": "label_emb.0.2.bias",
        },
    },
    "controlnet": {
        "layers": {
            "time_embedding.linear_1.weight": "time_embed.0.weight",
            "time_embedding.linear_1.bias": "time_embed.0.bias",
            "time_embedding.linear_2.weight": "time_embed.2.weight",
            "time_embedding.linear_2.bias": "time_embed.2.bias",
            "conv_in.weight": "input_blocks.0.0.weight",
            "conv_in.bias": "input_blocks.0.0.bias",
            "controlnet_cond_embedding.conv_in.weight": "input_hint_block.0.weight",
            "controlnet_cond_embedding.conv_in.bias": "input_hint_block.0.bias",
            "controlnet_cond_embedding.conv_out.weight": "input_hint_block.14.weight",
            "controlnet_cond_embedding.conv_out.bias": "input_hint_block.14.bias",
        },
        "class_embed_type": {
            "class_embedding.linear_1.weight": "label_emb.0.0.weight",
            "class_embedding.linear_1.bias": "label_emb.0.0.bias",
            "class_embedding.linear_2.weight": "label_emb.0.2.weight",
            "class_embedding.linear_2.bias": "label_emb.0.2.bias",
        },
        "addition_embed_type": {
            "add_embedding.linear_1.weight": "label_emb.0.0.weight",
            "add_embedding.linear_1.bias": "label_emb.0.0.bias",
            "add_embedding.linear_2.weight": "label_emb.0.2.weight",
            "add_embedding.linear_2.bias": "label_emb.0.2.bias",
        },
    },
    "vae": {
        "encoder.conv_in.weight": "encoder.conv_in.weight",
        "encoder.conv_in.bias": "encoder.conv_in.bias",
        "encoder.conv_out.weight": "encoder.conv_out.weight",
        "encoder.conv_out.bias": "encoder.conv_out.bias",
        "encoder.conv_norm_out.weight": "encoder.norm_out.weight",
        "encoder.conv_norm_out.bias": "encoder.norm_out.bias",
        "decoder.conv_in.weight": "decoder.conv_in.weight",
        "decoder.conv_in.bias": "decoder.conv_in.bias",
        "decoder.conv_out.weight": "decoder.conv_out.weight",
        "decoder.conv_out.bias": "decoder.conv_out.bias",
        "decoder.conv_norm_out.weight": "decoder.norm_out.weight",
        "decoder.conv_norm_out.bias": "decoder.norm_out.bias",
        "quant_conv.weight": "quant_conv.weight",
        "quant_conv.bias": "quant_conv.bias",
        "post_quant_conv.weight": "post_quant_conv.weight",
        "post_quant_conv.bias": "post_quant_conv.bias",
    },
    "openclip": {
        "layers": {
            "text_model.embeddings.position_embedding.weight": "positional_embedding",
            "text_model.embeddings.token_embedding.weight": "token_embedding.weight",
            "text_model.final_layer_norm.weight": "ln_final.weight",
            "text_model.final_layer_norm.bias": "ln_final.bias",
            "text_projection.weight": "text_projection",
        },
        "transformer": {
            "text_model.encoder.layers.": "resblocks.",
            "layer_norm1": "ln_1",
            "layer_norm2": "ln_2",
            ".fc1.": ".c_fc.",
            ".fc2.": ".c_proj.",
            ".self_attn": ".attn",
            "transformer.text_model.final_layer_norm.": "ln_final.",
            "transformer.text_model.embeddings.token_embedding.weight": "token_embedding.weight",
            "transformer.text_model.embeddings.position_embedding.weight": "positional_embedding",
        },
    },
}

LDM_VAE_KEY = "first_stage_model."
LDM_VAE_DEFAULT_SCALING_FACTOR = 0.18215
PLAYGROUND_VAE_SCALING_FACTOR = 0.5
LDM_UNET_KEY = "model.diffusion_model."
LDM_CONTROLNET_KEY = "control_model."
LDM_CLIP_PREFIX_TO_REMOVE = ["cond_stage_model.transformer.", "conditioner.embedders.0.transformer."]
LDM_OPEN_CLIP_TEXT_PROJECTION_DIM = 1024

SD_2_TEXT_ENCODER_KEYS_TO_IGNORE = [
    "cond_stage_model.model.transformer.resblocks.23.attn.in_proj_bias",
    "cond_stage_model.model.transformer.resblocks.23.attn.in_proj_weight",
    "cond_stage_model.model.transformer.resblocks.23.attn.out_proj.bias",
    "cond_stage_model.model.transformer.resblocks.23.attn.out_proj.weight",
    "cond_stage_model.model.transformer.resblocks.23.ln_1.bias",
    "cond_stage_model.model.transformer.resblocks.23.ln_1.weight",
    "cond_stage_model.model.transformer.resblocks.23.ln_2.bias",
    "cond_stage_model.model.transformer.resblocks.23.ln_2.weight",
    "cond_stage_model.model.transformer.resblocks.23.mlp.c_fc.bias",
    "cond_stage_model.model.transformer.resblocks.23.mlp.c_fc.weight",
    "cond_stage_model.model.transformer.resblocks.23.mlp.c_proj.bias",
    "cond_stage_model.model.transformer.resblocks.23.mlp.c_proj.weight",
    "cond_stage_model.model.text_projection",
]


VALID_URL_PREFIXES = ["https://huggingface.co/", "huggingface.co/", "hf.co/", "https://hf.co/"]


def _extract_repo_id_and_weights_name(pretrained_model_name_or_path):
    pattern = r"([^/]+)/([^/]+)/(?:blob/main/)?(.+)"
    weights_name = None
    repo_id = (None,)
    for prefix in VALID_URL_PREFIXES:
        pretrained_model_name_or_path = pretrained_model_name_or_path.replace(prefix, "")
    match = re.match(pattern, pretrained_model_name_or_path)
    if not match:
        return repo_id, weights_name

    repo_id = f"{match.group(1)}/{match.group(2)}"
    weights_name = match.group(3)

    return repo_id, weights_name


def fetch_ldm_config_and_checkpoint(
    pretrained_model_link_or_path,
    class_name,
    original_config_file=None,
    resume_download=False,
    force_download=False,
    proxies=None,
    token=None,
    cache_dir=None,
    local_files_only=None,
    revision=None,
):
    if os.path.isfile(pretrained_model_link_or_path):
        checkpoint = load_state_dict(pretrained_model_link_or_path)

    else:
        repo_id, weights_name = _extract_repo_id_and_weights_name(pretrained_model_link_or_path)
        checkpoint_path = _get_model_file(
            repo_id,
            weights_name=weights_name,
            force_download=force_download,
            cache_dir=cache_dir,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
        )
        checkpoint = load_state_dict(checkpoint_path)

    # some checkpoints contain the model state dict under a "state_dict" key
    while "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    original_config = fetch_original_config(class_name, checkpoint, original_config_file)

    return original_config, checkpoint


def infer_original_config_file(class_name, checkpoint):
    if CHECKPOINT_KEY_NAMES["v2"] in checkpoint and checkpoint[CHECKPOINT_KEY_NAMES["v2"]].shape[-1] == 1024:
        config_url = CONFIG_URLS["v2"]

    elif CHECKPOINT_KEY_NAMES["xl_base"] in checkpoint:
        config_url = CONFIG_URLS["xl"]

    elif CHECKPOINT_KEY_NAMES["xl_refiner"] in checkpoint:
        config_url = CONFIG_URLS["xl_refiner"]

    elif class_name == "StableDiffusionUpscalePipeline":
        config_url = CONFIG_URLS["upscale"]

    elif class_name == "ControlNetModel":
        config_url = CONFIG_URLS["controlnet"]

    else:
        config_url = CONFIG_URLS["v1"]

    original_config_file = BytesIO(requests.get(config_url).content)

    return original_config_file


def fetch_original_config(pipeline_class_name, checkpoint, original_config_file=None):
    def is_valid_url(url):
        result = urlparse(url)
        if result.scheme and result.netloc:
            return True

        return False

    if original_config_file is None:
        original_config_file = infer_original_config_file(pipeline_class_name, checkpoint)

    elif os.path.isfile(original_config_file):
        with open(original_config_file, "r") as fp:
            original_config_file = fp.read()

    elif is_valid_url(original_config_file):
        original_config_file = BytesIO(requests.get(original_config_file).content)

    else:
        raise ValueError("Invalid `original_config_file` provided. Please set it to a valid file path or URL.")

    original_config = yaml.safe_load(original_config_file)

    return original_config


def infer_model_type(original_config, checkpoint=None, model_type=None):
    if model_type is not None:
        return model_type

    has_cond_stage_config = (
        "cond_stage_config" in original_config["model"]["params"]
        and original_config["model"]["params"]["cond_stage_config"] is not None
    )
    has_network_config = (
        "network_config" in original_config["model"]["params"]
        and original_config["model"]["params"]["network_config"] is not None
    )

    if has_cond_stage_config:
        model_type = original_config["model"]["params"]["cond_stage_config"]["target"].split(".")[-1]

    elif has_network_config:
        context_dim = original_config["model"]["params"]["network_config"]["params"]["context_dim"]
        if "edm_mean" in checkpoint and "edm_std" in checkpoint:
            model_type = "Playground"
        elif context_dim == 2048:
            model_type = "SDXL"
        else:
            model_type = "SDXL-Refiner"
    else:
        raise ValueError("Unable to infer model type from config")

    logger.debug(f"No `model_type` given, `model_type` inferred as: {model_type}")

    return model_type


def get_default_scheduler_config():
    return SCHEDULER_DEFAULT_CONFIG


def set_image_size(pipeline_class_name, original_config, checkpoint, image_size=None, model_type=None):
    if image_size:
        return image_size

    global_step = checkpoint["global_step"] if "global_step" in checkpoint else None
    model_type = infer_model_type(original_config, checkpoint, model_type)

    if pipeline_class_name == "StableDiffusionUpscalePipeline":
        image_size = original_config["model"]["params"]["unet_config"]["params"]["image_size"]
        return image_size

    elif model_type in ["SDXL", "SDXL-Refiner", "Playground"]:
        image_size = 1024
        return image_size

    elif (
        "parameterization" in original_config["model"]["params"]
        and original_config["model"]["params"]["parameterization"] == "v"
    ):
        # NOTE: For stable diffusion 2 base one has to pass `image_size==512`
        # as it relies on a brittle global step parameter here
        image_size = 512 if global_step == 875000 else 768
        return image_size

    else:
        image_size = 512
        return image_size


# Copied from diffusers.pipelines.stable_diffusion.convert_from_ckpt.conv_attn_to_linear
def conv_attn_to_linear(checkpoint):
    keys = list(checkpoint.keys())
    attn_keys = ["query.weight", "key.weight", "value.weight"]
    for key in keys:
        if ".".join(key.split(".")[-2:]) in attn_keys:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0, 0]
        elif "proj_attn.weight" in key:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0]


def create_unet_diffusers_config(original_config, image_size: int):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    if (
        "unet_config" in original_config["model"]["params"]
        and original_config["model"]["params"]["unet_config"] is not None
    ):
        unet_params = original_config["model"]["params"]["unet_config"]["params"]
    else:
        unet_params = original_config["model"]["params"]["network_config"]["params"]

    vae_params = original_config["model"]["params"]["first_stage_config"]["params"]["ddconfig"]
    block_out_channels = [unet_params["model_channels"] * mult for mult in unet_params["channel_mult"]]

    down_block_types = []
    resolution = 1
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnDownBlock2D" if resolution in unet_params["attention_resolutions"] else "DownBlock2D"
        down_block_types.append(block_type)
        if i != len(block_out_channels) - 1:
            resolution *= 2

    up_block_types = []
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnUpBlock2D" if resolution in unet_params["attention_resolutions"] else "UpBlock2D"
        up_block_types.append(block_type)
        resolution //= 2

    if unet_params["transformer_depth"] is not None:
        transformer_layers_per_block = (
            unet_params["transformer_depth"]
            if isinstance(unet_params["transformer_depth"], int)
            else list(unet_params["transformer_depth"])
        )
    else:
        transformer_layers_per_block = 1

    vae_scale_factor = 2 ** (len(vae_params["ch_mult"]) - 1)

    head_dim = unet_params["num_heads"] if "num_heads" in unet_params else None
    use_linear_projection = (
        unet_params["use_linear_in_transformer"] if "use_linear_in_transformer" in unet_params else False
    )
    if use_linear_projection:
        # stable diffusion 2-base-512 and 2-768
        if head_dim is None:
            head_dim_mult = unet_params["model_channels"] // unet_params["num_head_channels"]
            head_dim = [head_dim_mult * c for c in list(unet_params["channel_mult"])]

    class_embed_type = None
    addition_embed_type = None
    addition_time_embed_dim = None
    projection_class_embeddings_input_dim = None
    context_dim = None

    if unet_params["context_dim"] is not None:
        context_dim = (
            unet_params["context_dim"]
            if isinstance(unet_params["context_dim"], int)
            else unet_params["context_dim"][0]
        )

    if "num_classes" in unet_params:
        if unet_params["num_classes"] == "sequential":
            if context_dim in [2048, 1280]:
                # SDXL
                addition_embed_type = "text_time"
                addition_time_embed_dim = 256
            else:
                class_embed_type = "projection"
            assert "adm_in_channels" in unet_params
            projection_class_embeddings_input_dim = unet_params["adm_in_channels"]

    config = {
        "sample_size": image_size // vae_scale_factor,
        "in_channels": unet_params["in_channels"],
        "down_block_types": down_block_types,
        "block_out_channels": block_out_channels,
        "layers_per_block": unet_params["num_res_blocks"],
        "cross_attention_dim": context_dim,
        "attention_head_dim": head_dim,
        "use_linear_projection": use_linear_projection,
        "class_embed_type": class_embed_type,
        "addition_embed_type": addition_embed_type,
        "addition_time_embed_dim": addition_time_embed_dim,
        "projection_class_embeddings_input_dim": projection_class_embeddings_input_dim,
        "transformer_layers_per_block": transformer_layers_per_block,
    }

    if "disable_self_attentions" in unet_params:
        config["only_cross_attention"] = unet_params["disable_self_attentions"]

    if "num_classes" in unet_params and isinstance(unet_params["num_classes"], int):
        config["num_class_embeds"] = unet_params["num_classes"]

    config["out_channels"] = unet_params["out_channels"]
    config["up_block_types"] = up_block_types

    return config


def create_controlnet_diffusers_config(original_config, image_size: int):
    unet_params = original_config["model"]["params"]["control_stage_config"]["params"]
    diffusers_unet_config = create_unet_diffusers_config(original_config, image_size=image_size)

    controlnet_config = {
        "conditioning_channels": unet_params["hint_channels"],
        "in_channels": diffusers_unet_config["in_channels"],
        "down_block_types": diffusers_unet_config["down_block_types"],
        "block_out_channels": diffusers_unet_config["block_out_channels"],
        "layers_per_block": diffusers_unet_config["layers_per_block"],
        "cross_attention_dim": diffusers_unet_config["cross_attention_dim"],
        "attention_head_dim": diffusers_unet_config["attention_head_dim"],
        "use_linear_projection": diffusers_unet_config["use_linear_projection"],
        "class_embed_type": diffusers_unet_config["class_embed_type"],
        "addition_embed_type": diffusers_unet_config["addition_embed_type"],
        "addition_time_embed_dim": diffusers_unet_config["addition_time_embed_dim"],
        "projection_class_embeddings_input_dim": diffusers_unet_config["projection_class_embeddings_input_dim"],
        "transformer_layers_per_block": diffusers_unet_config["transformer_layers_per_block"],
    }

    return controlnet_config


def create_vae_diffusers_config(original_config, image_size, scaling_factor=None, latents_mean=None, latents_std=None):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    vae_params = original_config["model"]["params"]["first_stage_config"]["params"]["ddconfig"]
    if (scaling_factor is None) and (latents_mean is not None) and (latents_std is not None):
        scaling_factor = PLAYGROUND_VAE_SCALING_FACTOR
    elif (scaling_factor is None) and ("scale_factor" in original_config["model"]["params"]):
        scaling_factor = original_config["model"]["params"]["scale_factor"]
    elif scaling_factor is None:
        scaling_factor = LDM_VAE_DEFAULT_SCALING_FACTOR

    block_out_channels = [vae_params["ch"] * mult for mult in vae_params["ch_mult"]]
    down_block_types = ["DownEncoderBlock2D"] * len(block_out_channels)
    up_block_types = ["UpDecoderBlock2D"] * len(block_out_channels)

    config = {
        "sample_size": image_size,
        "in_channels": vae_params["in_channels"],
        "out_channels": vae_params["out_ch"],
        "down_block_types": down_block_types,
        "up_block_types": up_block_types,
        "block_out_channels": block_out_channels,
        "latent_channels": vae_params["z_channels"],
        "layers_per_block": vae_params["num_res_blocks"],
        "scaling_factor": scaling_factor,
    }
    if latents_mean is not None and latents_std is not None:
        config.update({"latents_mean": latents_mean, "latents_std": latents_std})

    return config


def update_unet_resnet_ldm_to_diffusers(ldm_keys, new_checkpoint, checkpoint, mapping=None):
    for ldm_key in ldm_keys:
        diffusers_key = (
            ldm_key.replace("in_layers.0", "norm1")
            .replace("in_layers.2", "conv1")
            .replace("out_layers.0", "norm2")
            .replace("out_layers.3", "conv2")
            .replace("emb_layers.1", "time_emb_proj")
            .replace("skip_connection", "conv_shortcut")
        )
        if mapping:
            diffusers_key = diffusers_key.replace(mapping["old"], mapping["new"])
        new_checkpoint[diffusers_key] = checkpoint.pop(ldm_key)


def update_unet_attention_ldm_to_diffusers(ldm_keys, new_checkpoint, checkpoint, mapping):
    for ldm_key in ldm_keys:
        diffusers_key = ldm_key.replace(mapping["old"], mapping["new"])
        new_checkpoint[diffusers_key] = checkpoint.pop(ldm_key)


def convert_ldm_unet_checkpoint(checkpoint, config, extract_ema=False):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """
    # extract state_dict for UNet
    unet_state_dict = {}
    keys = list(checkpoint.keys())
    unet_key = LDM_UNET_KEY

    # at least a 100 parameters have to start with `model_ema` in order for the checkpoint to be EMA
    if sum(k.startswith("model_ema") for k in keys) > 100 and extract_ema:
        logger.warning("Checkpoint has both EMA and non-EMA weights.")
        logger.warning(
            "In this conversion only the EMA weights are extracted. If you want to instead extract the non-EMA"
            " weights (useful to continue fine-tuning), please make sure to remove the `--extract_ema` flag."
        )
        for key in keys:
            if key.startswith("model.diffusion_model"):
                flat_ema_key = "model_ema." + "".join(key.split(".")[1:])
                unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(flat_ema_key)
    else:
        if sum(k.startswith("model_ema") for k in keys) > 100:
            logger.warning(
                "In this conversion only the non-EMA weights are extracted. If you want to instead extract the EMA"
                " weights (usually better for inference), please make sure to add the `--extract_ema` flag."
            )
        for key in keys:
            if key.startswith(unet_key):
                unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(key)

    new_checkpoint = {}
    ldm_unet_keys = DIFFUSERS_TO_LDM_MAPPING["unet"]["layers"]
    for diffusers_key, ldm_key in ldm_unet_keys.items():
        if ldm_key not in unet_state_dict:
            continue
        new_checkpoint[diffusers_key] = unet_state_dict[ldm_key]

    if ("class_embed_type" in config) and (config["class_embed_type"] in ["timestep", "projection"]):
        class_embed_keys = DIFFUSERS_TO_LDM_MAPPING["unet"]["class_embed_type"]
        for diffusers_key, ldm_key in class_embed_keys.items():
            new_checkpoint[diffusers_key] = unet_state_dict[ldm_key]

    if ("addition_embed_type" in config) and (config["addition_embed_type"] == "text_time"):
        addition_embed_keys = DIFFUSERS_TO_LDM_MAPPING["unet"]["addition_embed_type"]
        for diffusers_key, ldm_key in addition_embed_keys.items():
            new_checkpoint[diffusers_key] = unet_state_dict[ldm_key]

    # Relevant to StableDiffusionUpscalePipeline
    if "num_class_embeds" in config:
        if (config["num_class_embeds"] is not None) and ("label_emb.weight" in unet_state_dict):
            new_checkpoint["class_embedding.weight"] = unet_state_dict["label_emb.weight"]

    # Retrieves the keys for the input blocks only
    num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "input_blocks" in layer})
    input_blocks = {
        layer_id: [key for key in unet_state_dict if f"input_blocks.{layer_id}" in key]
        for layer_id in range(num_input_blocks)
    }

    # Retrieves the keys for the middle blocks only
    num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "middle_block" in layer})
    middle_blocks = {
        layer_id: [key for key in unet_state_dict if f"middle_block.{layer_id}" in key]
        for layer_id in range(num_middle_blocks)
    }

    # Retrieves the keys for the output blocks only
    num_output_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "output_blocks" in layer})
    output_blocks = {
        layer_id: [key for key in unet_state_dict if f"output_blocks.{layer_id}" in key]
        for layer_id in range(num_output_blocks)
    }

    # Down blocks
    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (config["layers_per_block"] + 1)
        layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

        resnets = [
            key for key in input_blocks[i] if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
        ]
        update_unet_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            unet_state_dict,
            {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"},
        )

        if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.weight"
            )
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.bias"
            )

        attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]
        if attentions:
            update_unet_attention_ldm_to_diffusers(
                attentions,
                new_checkpoint,
                unet_state_dict,
                {"old": f"input_blocks.{i}.1", "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}"},
            )

    # Mid blocks
    resnet_0 = middle_blocks[0]
    attentions = middle_blocks[1]
    resnet_1 = middle_blocks[2]

    update_unet_resnet_ldm_to_diffusers(
        resnet_0, new_checkpoint, unet_state_dict, mapping={"old": "middle_block.0", "new": "mid_block.resnets.0"}
    )
    update_unet_resnet_ldm_to_diffusers(
        resnet_1, new_checkpoint, unet_state_dict, mapping={"old": "middle_block.2", "new": "mid_block.resnets.1"}
    )
    update_unet_attention_ldm_to_diffusers(
        attentions, new_checkpoint, unet_state_dict, mapping={"old": "middle_block.1", "new": "mid_block.attentions.0"}
    )

    # Up Blocks
    for i in range(num_output_blocks):
        block_id = i // (config["layers_per_block"] + 1)
        layer_in_block_id = i % (config["layers_per_block"] + 1)

        resnets = [
            key for key in output_blocks[i] if f"output_blocks.{i}.0" in key and f"output_blocks.{i}.0.op" not in key
        ]
        update_unet_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            unet_state_dict,
            {"old": f"output_blocks.{i}.0", "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}"},
        )

        attentions = [
            key for key in output_blocks[i] if f"output_blocks.{i}.1" in key and f"output_blocks.{i}.1.conv" not in key
        ]
        if attentions:
            update_unet_attention_ldm_to_diffusers(
                attentions,
                new_checkpoint,
                unet_state_dict,
                {"old": f"output_blocks.{i}.1", "new": f"up_blocks.{block_id}.attentions.{layer_in_block_id}"},
            )

        if f"output_blocks.{i}.1.conv.weight" in unet_state_dict:
            new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_state_dict[
                f"output_blocks.{i}.1.conv.weight"
            ]
            new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_state_dict[
                f"output_blocks.{i}.1.conv.bias"
            ]
        if f"output_blocks.{i}.2.conv.weight" in unet_state_dict:
            new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_state_dict[
                f"output_blocks.{i}.2.conv.weight"
            ]
            new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_state_dict[
                f"output_blocks.{i}.2.conv.bias"
            ]

    return new_checkpoint


def convert_controlnet_checkpoint(
    checkpoint,
    config,
):
    # Some controlnet ckpt files are distributed independently from the rest of the
    # model components i.e. https://huggingface.co/thibaud/controlnet-sd21/
    if "time_embed.0.weight" in checkpoint:
        controlnet_state_dict = checkpoint

    else:
        controlnet_state_dict = {}
        keys = list(checkpoint.keys())
        controlnet_key = LDM_CONTROLNET_KEY
        for key in keys:
            if key.startswith(controlnet_key):
                controlnet_state_dict[key.replace(controlnet_key, "")] = checkpoint.pop(key)

    new_checkpoint = {}
    ldm_controlnet_keys = DIFFUSERS_TO_LDM_MAPPING["controlnet"]["layers"]
    for diffusers_key, ldm_key in ldm_controlnet_keys.items():
        if ldm_key not in controlnet_state_dict:
            continue
        new_checkpoint[diffusers_key] = controlnet_state_dict[ldm_key]

    # Retrieves the keys for the input blocks only
    num_input_blocks = len(
        {".".join(layer.split(".")[:2]) for layer in controlnet_state_dict if "input_blocks" in layer}
    )
    input_blocks = {
        layer_id: [key for key in controlnet_state_dict if f"input_blocks.{layer_id}" in key]
        for layer_id in range(num_input_blocks)
    }

    # Down blocks
    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (config["layers_per_block"] + 1)
        layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

        resnets = [
            key for key in input_blocks[i] if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
        ]
        update_unet_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            controlnet_state_dict,
            {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"},
        )

        if f"input_blocks.{i}.0.op.weight" in controlnet_state_dict:
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = controlnet_state_dict.pop(
                f"input_blocks.{i}.0.op.weight"
            )
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = controlnet_state_dict.pop(
                f"input_blocks.{i}.0.op.bias"
            )

        attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]
        if attentions:
            update_unet_attention_ldm_to_diffusers(
                attentions,
                new_checkpoint,
                controlnet_state_dict,
                {"old": f"input_blocks.{i}.1", "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}"},
            )

    # controlnet down blocks
    for i in range(num_input_blocks):
        new_checkpoint[f"controlnet_down_blocks.{i}.weight"] = controlnet_state_dict.pop(f"zero_convs.{i}.0.weight")
        new_checkpoint[f"controlnet_down_blocks.{i}.bias"] = controlnet_state_dict.pop(f"zero_convs.{i}.0.bias")

    # Retrieves the keys for the middle blocks only
    num_middle_blocks = len(
        {".".join(layer.split(".")[:2]) for layer in controlnet_state_dict if "middle_block" in layer}
    )
    middle_blocks = {
        layer_id: [key for key in controlnet_state_dict if f"middle_block.{layer_id}" in key]
        for layer_id in range(num_middle_blocks)
    }
    if middle_blocks:
        resnet_0 = middle_blocks[0]
        attentions = middle_blocks[1]
        resnet_1 = middle_blocks[2]

        update_unet_resnet_ldm_to_diffusers(
            resnet_0,
            new_checkpoint,
            controlnet_state_dict,
            mapping={"old": "middle_block.0", "new": "mid_block.resnets.0"},
        )
        update_unet_resnet_ldm_to_diffusers(
            resnet_1,
            new_checkpoint,
            controlnet_state_dict,
            mapping={"old": "middle_block.2", "new": "mid_block.resnets.1"},
        )
        update_unet_attention_ldm_to_diffusers(
            attentions,
            new_checkpoint,
            controlnet_state_dict,
            mapping={"old": "middle_block.1", "new": "mid_block.attentions.0"},
        )

    # mid block
    new_checkpoint["controlnet_mid_block.weight"] = controlnet_state_dict.pop("middle_block_out.0.weight")
    new_checkpoint["controlnet_mid_block.bias"] = controlnet_state_dict.pop("middle_block_out.0.bias")

    # controlnet cond embedding blocks
    cond_embedding_blocks = {
        ".".join(layer.split(".")[:2])
        for layer in controlnet_state_dict
        if "input_hint_block" in layer and ("input_hint_block.0" not in layer) and ("input_hint_block.14" not in layer)
    }
    num_cond_embedding_blocks = len(cond_embedding_blocks)

    for idx in range(1, num_cond_embedding_blocks + 1):
        diffusers_idx = idx - 1
        cond_block_id = 2 * idx

        new_checkpoint[f"controlnet_cond_embedding.blocks.{diffusers_idx}.weight"] = controlnet_state_dict.pop(
            f"input_hint_block.{cond_block_id}.weight"
        )
        new_checkpoint[f"controlnet_cond_embedding.blocks.{diffusers_idx}.bias"] = controlnet_state_dict.pop(
            f"input_hint_block.{cond_block_id}.bias"
        )

    return new_checkpoint


def create_diffusers_controlnet_model_from_ldm(
    pipeline_class_name, original_config, checkpoint, upcast_attention=False, image_size=None, torch_dtype=None
):
    # import here to avoid circular imports
    from ..models import ControlNetModel

    image_size = set_image_size(pipeline_class_name, original_config, checkpoint, image_size=image_size)

    diffusers_config = create_controlnet_diffusers_config(original_config, image_size=image_size)
    diffusers_config["upcast_attention"] = upcast_attention

    diffusers_format_controlnet_checkpoint = convert_controlnet_checkpoint(checkpoint, diffusers_config)

    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    with ctx():
        controlnet = ControlNetModel(**diffusers_config)

    if is_accelerate_available():
        from ..models.modeling_utils import load_model_dict_into_meta

        unexpected_keys = load_model_dict_into_meta(
            controlnet, diffusers_format_controlnet_checkpoint, dtype=torch_dtype
        )
        if controlnet._keys_to_ignore_on_load_unexpected is not None:
            for pat in controlnet._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        if len(unexpected_keys) > 0:
            logger.warn(
                f"Some weights of the model checkpoint were not used when initializing {controlnet.__name__}: \n {[', '.join(unexpected_keys)]}"
            )
    else:
        controlnet.load_state_dict(diffusers_format_controlnet_checkpoint)

    if torch_dtype is not None:
        controlnet = controlnet.to(torch_dtype)

    return {"controlnet": controlnet}


def update_vae_resnet_ldm_to_diffusers(keys, new_checkpoint, checkpoint, mapping):
    for ldm_key in keys:
        diffusers_key = ldm_key.replace(mapping["old"], mapping["new"]).replace("nin_shortcut", "conv_shortcut")
        new_checkpoint[diffusers_key] = checkpoint.pop(ldm_key)


def update_vae_attentions_ldm_to_diffusers(keys, new_checkpoint, checkpoint, mapping):
    for ldm_key in keys:
        diffusers_key = (
            ldm_key.replace(mapping["old"], mapping["new"])
            .replace("norm.weight", "group_norm.weight")
            .replace("norm.bias", "group_norm.bias")
            .replace("q.weight", "to_q.weight")
            .replace("q.bias", "to_q.bias")
            .replace("k.weight", "to_k.weight")
            .replace("k.bias", "to_k.bias")
            .replace("v.weight", "to_v.weight")
            .replace("v.bias", "to_v.bias")
            .replace("proj_out.weight", "to_out.0.weight")
            .replace("proj_out.bias", "to_out.0.bias")
        )
        new_checkpoint[diffusers_key] = checkpoint.pop(ldm_key)

        # proj_attn.weight has to be converted from conv 1D to linear
        shape = new_checkpoint[diffusers_key].shape

        if len(shape) == 3:
            new_checkpoint[diffusers_key] = new_checkpoint[diffusers_key][:, :, 0]
        elif len(shape) == 4:
            new_checkpoint[diffusers_key] = new_checkpoint[diffusers_key][:, :, 0, 0]


def convert_ldm_vae_checkpoint(checkpoint, config):
    # extract state dict for VAE
    # remove the LDM_VAE_KEY prefix from the ldm checkpoint keys so that it is easier to map them to diffusers keys
    vae_state_dict = {}
    keys = list(checkpoint.keys())
    vae_key = LDM_VAE_KEY if any(k.startswith(LDM_VAE_KEY) for k in keys) else ""
    for key in keys:
        if key.startswith(vae_key):
            vae_state_dict[key.replace(vae_key, "")] = checkpoint.get(key)

    new_checkpoint = {}
    vae_diffusers_ldm_map = DIFFUSERS_TO_LDM_MAPPING["vae"]
    for diffusers_key, ldm_key in vae_diffusers_ldm_map.items():
        if ldm_key not in vae_state_dict:
            continue
        new_checkpoint[diffusers_key] = vae_state_dict[ldm_key]

    # Retrieves the keys for the encoder down blocks only
    num_down_blocks = len(config["down_block_types"])
    down_blocks = {
        layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key] for layer_id in range(num_down_blocks)
    }

    for i in range(num_down_blocks):
        resnets = [key for key in down_blocks[i] if f"down.{i}" in key and f"down.{i}.downsample" not in key]
        update_vae_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            vae_state_dict,
            mapping={"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"},
        )
        if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.weight"
            )
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.bias"
            )

    mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]
        update_vae_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            vae_state_dict,
            mapping={"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"},
        )

    mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
    update_vae_attentions_ldm_to_diffusers(
        mid_attentions, new_checkpoint, vae_state_dict, mapping={"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    )

    # Retrieves the keys for the decoder up blocks only
    num_up_blocks = len(config["up_block_types"])
    up_blocks = {
        layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key] for layer_id in range(num_up_blocks)
    }

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [
            key for key in up_blocks[block_id] if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
        ]
        update_vae_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            vae_state_dict,
            mapping={"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"},
        )
        if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.weight"
            ]
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.bias"
            ]

    mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]
        update_vae_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            vae_state_dict,
            mapping={"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"},
        )

    mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
    update_vae_attentions_ldm_to_diffusers(
        mid_attentions, new_checkpoint, vae_state_dict, mapping={"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    )
    conv_attn_to_linear(new_checkpoint)

    return new_checkpoint


def create_text_encoder_from_ldm_clip_checkpoint(config_name, checkpoint, local_files_only=False, torch_dtype=None):
    try:
        config = CLIPTextConfig.from_pretrained(config_name, local_files_only=local_files_only)
    except Exception:
        raise ValueError(
            f"With local_files_only set to {local_files_only}, you must first locally save the configuration in the following path: 'openai/clip-vit-large-patch14'."
        )

    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    with ctx():
        text_model = CLIPTextModel(config)

    keys = list(checkpoint.keys())
    text_model_dict = {}

    remove_prefixes = LDM_CLIP_PREFIX_TO_REMOVE

    for key in keys:
        for prefix in remove_prefixes:
            if key.startswith(prefix):
                diffusers_key = key.replace(prefix, "")
                text_model_dict[diffusers_key] = checkpoint[key]

    if is_accelerate_available():
        from ..models.modeling_utils import load_model_dict_into_meta

        unexpected_keys = load_model_dict_into_meta(text_model, text_model_dict, dtype=torch_dtype)
        if text_model._keys_to_ignore_on_load_unexpected is not None:
            for pat in text_model._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        if len(unexpected_keys) > 0:
            logger.warn(
                f"Some weights of the model checkpoint were not used when initializing {text_model.__class__.__name__}: \n {[', '.join(unexpected_keys)]}"
            )
    else:
        if not (hasattr(text_model, "embeddings") and hasattr(text_model.embeddings.position_ids)):
            text_model_dict.pop("text_model.embeddings.position_ids", None)

        text_model.load_state_dict(text_model_dict)

    if torch_dtype is not None:
        text_model = text_model.to(torch_dtype)

    return text_model


def create_text_encoder_from_open_clip_checkpoint(
    config_name,
    checkpoint,
    prefix="cond_stage_model.model.",
    has_projection=False,
    local_files_only=False,
    torch_dtype=None,
    **config_kwargs,
):
    try:
        config = CLIPTextConfig.from_pretrained(config_name, **config_kwargs, local_files_only=local_files_only)
    except Exception:
        raise ValueError(
            f"With local_files_only set to {local_files_only}, you must first locally save the configuration in the following path: '{config_name}'."
        )

    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    with ctx():
        text_model = CLIPTextModelWithProjection(config) if has_projection else CLIPTextModel(config)

    text_model_dict = {}
    text_proj_key = prefix + "text_projection"
    text_proj_dim = (
        int(checkpoint[text_proj_key].shape[0]) if text_proj_key in checkpoint else LDM_OPEN_CLIP_TEXT_PROJECTION_DIM
    )
    text_model_dict["text_model.embeddings.position_ids"] = text_model.text_model.embeddings.get_buffer("position_ids")

    keys = list(checkpoint.keys())
    keys_to_ignore = SD_2_TEXT_ENCODER_KEYS_TO_IGNORE

    openclip_diffusers_ldm_map = DIFFUSERS_TO_LDM_MAPPING["openclip"]["layers"]
    for diffusers_key, ldm_key in openclip_diffusers_ldm_map.items():
        ldm_key = prefix + ldm_key
        if ldm_key not in checkpoint:
            continue
        if ldm_key in keys_to_ignore:
            continue
        if ldm_key.endswith("text_projection"):
            text_model_dict[diffusers_key] = checkpoint[ldm_key].T.contiguous()
        else:
            text_model_dict[diffusers_key] = checkpoint[ldm_key]

    for key in keys:
        if key in keys_to_ignore:
            continue

        if not key.startswith(prefix + "transformer."):
            continue

        diffusers_key = key.replace(prefix + "transformer.", "")
        transformer_diffusers_to_ldm_map = DIFFUSERS_TO_LDM_MAPPING["openclip"]["transformer"]
        for new_key, old_key in transformer_diffusers_to_ldm_map.items():
            diffusers_key = (
                diffusers_key.replace(old_key, new_key).replace(".in_proj_weight", "").replace(".in_proj_bias", "")
            )

        if key.endswith(".in_proj_weight"):
            weight_value = checkpoint[key]

            text_model_dict[diffusers_key + ".q_proj.weight"] = weight_value[:text_proj_dim, :]
            text_model_dict[diffusers_key + ".k_proj.weight"] = weight_value[text_proj_dim : text_proj_dim * 2, :]
            text_model_dict[diffusers_key + ".v_proj.weight"] = weight_value[text_proj_dim * 2 :, :]

        elif key.endswith(".in_proj_bias"):
            weight_value = checkpoint[key]
            text_model_dict[diffusers_key + ".q_proj.bias"] = weight_value[:text_proj_dim]
            text_model_dict[diffusers_key + ".k_proj.bias"] = weight_value[text_proj_dim : text_proj_dim * 2]
            text_model_dict[diffusers_key + ".v_proj.bias"] = weight_value[text_proj_dim * 2 :]
        else:
            text_model_dict[diffusers_key] = checkpoint[key]

    if is_accelerate_available():
        from ..models.modeling_utils import load_model_dict_into_meta

        unexpected_keys = load_model_dict_into_meta(text_model, text_model_dict, dtype=torch_dtype)
        if text_model._keys_to_ignore_on_load_unexpected is not None:
            for pat in text_model._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        if len(unexpected_keys) > 0:
            logger.warn(
                f"Some weights of the model checkpoint were not used when initializing {text_model.__class__.__name__}: \n {[', '.join(unexpected_keys)]}"
            )

    else:
        if not (hasattr(text_model, "embeddings") and hasattr(text_model.embeddings.position_ids)):
            text_model_dict.pop("text_model.embeddings.position_ids", None)

        text_model.load_state_dict(text_model_dict)

    if torch_dtype is not None:
        text_model = text_model.to(torch_dtype)

    return text_model


def create_diffusers_unet_model_from_ldm(
    pipeline_class_name,
    original_config,
    checkpoint,
    num_in_channels=None,
    upcast_attention=False,
    extract_ema=False,
    image_size=None,
    torch_dtype=None,
    model_type=None,
):
    from ..models import UNet2DConditionModel

    if num_in_channels is None:
        if pipeline_class_name in [
            "StableDiffusionInpaintPipeline",
            "StableDiffusionControlNetInpaintPipeline",
            "StableDiffusionXLInpaintPipeline",
            "StableDiffusionXLControlNetInpaintPipeline",
        ]:
            num_in_channels = 9

        elif pipeline_class_name == "StableDiffusionUpscalePipeline":
            num_in_channels = 7

        else:
            num_in_channels = 4

    image_size = set_image_size(
        pipeline_class_name, original_config, checkpoint, image_size=image_size, model_type=model_type
    )
    unet_config = create_unet_diffusers_config(original_config, image_size=image_size)
    unet_config["in_channels"] = num_in_channels
    unet_config["upcast_attention"] = upcast_attention

    diffusers_format_unet_checkpoint = convert_ldm_unet_checkpoint(checkpoint, unet_config, extract_ema=extract_ema)
    ctx = init_empty_weights if is_accelerate_available() else nullcontext

    with ctx():
        unet = UNet2DConditionModel(**unet_config)

    if is_accelerate_available():
        from ..models.modeling_utils import load_model_dict_into_meta

        unexpected_keys = load_model_dict_into_meta(unet, diffusers_format_unet_checkpoint, dtype=torch_dtype)
        if unet._keys_to_ignore_on_load_unexpected is not None:
            for pat in unet._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        if len(unexpected_keys) > 0:
            logger.warn(
                f"Some weights of the model checkpoint were not used when initializing {unet.__name__}: \n {[', '.join(unexpected_keys)]}"
            )
    else:
        unet.load_state_dict(diffusers_format_unet_checkpoint)

    if torch_dtype is not None:
        unet = unet.to(torch_dtype)

    return {"unet": unet}


def create_diffusers_vae_model_from_ldm(
    pipeline_class_name,
    original_config,
    checkpoint,
    image_size=None,
    scaling_factor=None,
    torch_dtype=None,
    model_type=None,
):
    # import here to avoid circular imports
    from ..models import AutoencoderKL

    image_size = set_image_size(
        pipeline_class_name, original_config, checkpoint, image_size=image_size, model_type=model_type
    )
    model_type = infer_model_type(original_config, checkpoint, model_type)

    if model_type == "Playground":
        edm_mean = (
            checkpoint["edm_mean"].to(dtype=torch_dtype).tolist() if torch_dtype else checkpoint["edm_mean"].tolist()
        )
        edm_std = (
            checkpoint["edm_std"].to(dtype=torch_dtype).tolist() if torch_dtype else checkpoint["edm_std"].tolist()
        )
    else:
        edm_mean = None
        edm_std = None

    vae_config = create_vae_diffusers_config(
        original_config,
        image_size=image_size,
        scaling_factor=scaling_factor,
        latents_mean=edm_mean,
        latents_std=edm_std,
    )
    diffusers_format_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)
    ctx = init_empty_weights if is_accelerate_available() else nullcontext

    with ctx():
        vae = AutoencoderKL(**vae_config)

    if is_accelerate_available():
        from ..models.modeling_utils import load_model_dict_into_meta

        unexpected_keys = load_model_dict_into_meta(vae, diffusers_format_vae_checkpoint, dtype=torch_dtype)
        if vae._keys_to_ignore_on_load_unexpected is not None:
            for pat in vae._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        if len(unexpected_keys) > 0:
            logger.warn(
                f"Some weights of the model checkpoint were not used when initializing {vae.__name__}: \n {[', '.join(unexpected_keys)]}"
            )
    else:
        vae.load_state_dict(diffusers_format_vae_checkpoint)

    if torch_dtype is not None:
        vae = vae.to(torch_dtype)

    return {"vae": vae}


def create_text_encoders_and_tokenizers_from_ldm(
    original_config,
    checkpoint,
    model_type=None,
    local_files_only=False,
    torch_dtype=None,
):
    model_type = infer_model_type(original_config, checkpoint=checkpoint, model_type=model_type)

    if model_type == "FrozenOpenCLIPEmbedder":
        config_name = "stabilityai/stable-diffusion-2"
        config_kwargs = {"subfolder": "text_encoder"}

        try:
            text_encoder = create_text_encoder_from_open_clip_checkpoint(
                config_name, checkpoint, local_files_only=local_files_only, torch_dtype=torch_dtype, **config_kwargs
            )
            tokenizer = CLIPTokenizer.from_pretrained(
                config_name, subfolder="tokenizer", local_files_only=local_files_only
            )
        except Exception:
            raise ValueError(
                f"With local_files_only set to {local_files_only}, you must first locally save the text_encoder in the following path: '{config_name}'."
            )
        else:
            return {"text_encoder": text_encoder, "tokenizer": tokenizer}

    elif model_type == "FrozenCLIPEmbedder":
        try:
            config_name = "openai/clip-vit-large-patch14"
            text_encoder = create_text_encoder_from_ldm_clip_checkpoint(
                config_name,
                checkpoint,
                local_files_only=local_files_only,
                torch_dtype=torch_dtype,
            )
            tokenizer = CLIPTokenizer.from_pretrained(config_name, local_files_only=local_files_only)

        except Exception:
            raise ValueError(
                f"With local_files_only set to {local_files_only}, you must first locally save the tokenizer in the following path: '{config_name}'."
            )
        else:
            return {"text_encoder": text_encoder, "tokenizer": tokenizer}

    elif model_type == "SDXL-Refiner":
        config_name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
        config_kwargs = {"projection_dim": 1280}
        prefix = "conditioner.embedders.0.model."

        try:
            tokenizer_2 = CLIPTokenizer.from_pretrained(config_name, pad_token="!", local_files_only=local_files_only)
            text_encoder_2 = create_text_encoder_from_open_clip_checkpoint(
                config_name,
                checkpoint,
                prefix=prefix,
                has_projection=True,
                local_files_only=local_files_only,
                torch_dtype=torch_dtype,
                **config_kwargs,
            )
        except Exception:
            raise ValueError(
                f"With local_files_only set to {local_files_only}, you must first locally save the text_encoder_2 and tokenizer_2 in the following path: {config_name} with `pad_token` set to '!'."
            )

        else:
            return {
                "text_encoder": None,
                "tokenizer": None,
                "tokenizer_2": tokenizer_2,
                "text_encoder_2": text_encoder_2,
            }

    elif model_type in ["SDXL", "Playground"]:
        try:
            config_name = "openai/clip-vit-large-patch14"
            tokenizer = CLIPTokenizer.from_pretrained(config_name, local_files_only=local_files_only)
            text_encoder = create_text_encoder_from_ldm_clip_checkpoint(
                config_name, checkpoint, local_files_only=local_files_only, torch_dtype=torch_dtype
            )

        except Exception:
            raise ValueError(
                f"With local_files_only set to {local_files_only}, you must first locally save the text_encoder and tokenizer in the following path: 'openai/clip-vit-large-patch14'."
            )

        try:
            config_name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
            config_kwargs = {"projection_dim": 1280}
            prefix = "conditioner.embedders.1.model."
            tokenizer_2 = CLIPTokenizer.from_pretrained(config_name, pad_token="!", local_files_only=local_files_only)
            text_encoder_2 = create_text_encoder_from_open_clip_checkpoint(
                config_name,
                checkpoint,
                prefix=prefix,
                has_projection=True,
                local_files_only=local_files_only,
                torch_dtype=torch_dtype,
                **config_kwargs,
            )
        except Exception:
            raise ValueError(
                f"With local_files_only set to {local_files_only}, you must first locally save the text_encoder_2 and tokenizer_2 in the following path: {config_name} with `pad_token` set to '!'."
            )

        return {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "tokenizer_2": tokenizer_2,
            "text_encoder_2": text_encoder_2,
        }

    return


def create_scheduler_from_ldm(
    pipeline_class_name,
    original_config,
    checkpoint,
    prediction_type=None,
    scheduler_type="ddim",
    model_type=None,
):
    scheduler_config = get_default_scheduler_config()
    model_type = infer_model_type(original_config, checkpoint=checkpoint, model_type=model_type)

    global_step = checkpoint["global_step"] if "global_step" in checkpoint else None

    num_train_timesteps = getattr(original_config["model"]["params"], "timesteps", None) or 1000
    scheduler_config["num_train_timesteps"] = num_train_timesteps

    if (
        "parameterization" in original_config["model"]["params"]
        and original_config["model"]["params"]["parameterization"] == "v"
    ):
        if prediction_type is None:
            # NOTE: For stable diffusion 2 base it is recommended to pass `prediction_type=="epsilon"`
            # as it relies on a brittle global step parameter here
            prediction_type = "epsilon" if global_step == 875000 else "v_prediction"

    else:
        prediction_type = prediction_type or "epsilon"

    scheduler_config["prediction_type"] = prediction_type

    if model_type in ["SDXL", "SDXL-Refiner"]:
        scheduler_type = "euler"
    elif model_type == "Playground":
        scheduler_type = "edm_dpm_solver_multistep"
    else:
        beta_start = original_config["model"]["params"].get("linear_start", 0.02)
        beta_end = original_config["model"]["params"].get("linear_end", 0.085)
        scheduler_config["beta_start"] = beta_start
        scheduler_config["beta_end"] = beta_end
        scheduler_config["beta_schedule"] = "scaled_linear"
        scheduler_config["clip_sample"] = False
        scheduler_config["set_alpha_to_one"] = False

    if scheduler_type == "pndm":
        scheduler_config["skip_prk_steps"] = True
        scheduler = PNDMScheduler.from_config(scheduler_config)

    elif scheduler_type == "lms":
        scheduler = LMSDiscreteScheduler.from_config(scheduler_config)

    elif scheduler_type == "heun":
        scheduler = HeunDiscreteScheduler.from_config(scheduler_config)

    elif scheduler_type == "euler":
        scheduler = EulerDiscreteScheduler.from_config(scheduler_config)

    elif scheduler_type == "euler-ancestral":
        scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler_config)

    elif scheduler_type == "dpm":
        scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config)

    elif scheduler_type == "ddim":
        scheduler = DDIMScheduler.from_config(scheduler_config)

    elif scheduler_type == "edm_dpm_solver_multistep":
        scheduler_config = {
            "algorithm_type": "dpmsolver++",
            "dynamic_thresholding_ratio": 0.995,
            "euler_at_final": False,
            "final_sigmas_type": "zero",
            "lower_order_final": True,
            "num_train_timesteps": 1000,
            "prediction_type": "epsilon",
            "rho": 7.0,
            "sample_max_value": 1.0,
            "sigma_data": 0.5,
            "sigma_max": 80.0,
            "sigma_min": 0.002,
            "solver_order": 2,
            "solver_type": "midpoint",
            "thresholding": False,
        }
        scheduler = EDMDPMSolverMultistepScheduler(**scheduler_config)

    else:
        raise ValueError(f"Scheduler of type {scheduler_type} doesn't exist!")

    if pipeline_class_name == "StableDiffusionUpscalePipeline":
        scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", subfolder="scheduler")
        low_res_scheduler = DDPMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler", subfolder="low_res_scheduler"
        )

        return {
            "scheduler": scheduler,
            "low_res_scheduler": low_res_scheduler,
        }

    return {"scheduler": scheduler}
