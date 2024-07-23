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
"""Conversion script for the Stable Diffusion checkpoints."""

import os
import re
from contextlib import nullcontext
from io import BytesIO
from urllib.parse import urlparse

import requests
import torch
import yaml

from ..models.modeling_utils import load_state_dict
from ..schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EDMDPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from ..utils import (
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_NAME,
    deprecate,
    is_accelerate_available,
    is_transformers_available,
    logging,
)
from ..utils.hub_utils import _get_model_file


if is_transformers_available():
    from transformers import AutoImageProcessor

if is_accelerate_available():
    from accelerate import init_empty_weights

    from ..models.modeling_utils import load_model_dict_into_meta

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

CHECKPOINT_KEY_NAMES = {
    "v2": "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight",
    "xl_base": "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias",
    "xl_refiner": "conditioner.embedders.0.model.transformer.resblocks.9.mlp.c_proj.bias",
    "upscale": "model.diffusion_model.input_blocks.10.0.skip_connection.bias",
    "controlnet": "control_model.time_embed.0.weight",
    "playground-v2-5": "edm_mean",
    "inpainting": "model.diffusion_model.input_blocks.0.0.weight",
    "clip": "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight",
    "clip_sdxl": "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight",
    "clip_sd3": "text_encoders.clip_l.transformer.text_model.embeddings.position_embedding.weight",
    "open_clip": "cond_stage_model.model.token_embedding.weight",
    "open_clip_sdxl": "conditioner.embedders.1.model.positional_embedding",
    "open_clip_sdxl_refiner": "conditioner.embedders.0.model.text_projection",
    "open_clip_sd3": "text_encoders.clip_g.transformer.text_model.embeddings.position_embedding.weight",
    "stable_cascade_stage_b": "down_blocks.1.0.channelwise.0.weight",
    "stable_cascade_stage_c": "clip_txt_mapper.weight",
    "sd3": "model.diffusion_model.joint_blocks.0.context_block.adaLN_modulation.1.bias",
    "animatediff": "down_blocks.0.motion_modules.0.temporal_transformer.transformer_blocks.0.attention_blocks.1.pos_encoder.pe",
    "animatediff_v2": "mid_block.motion_modules.0.temporal_transformer.norm.bias",
    "animatediff_sdxl_beta": "up_blocks.2.motion_modules.0.temporal_transformer.norm.weight",
}

DIFFUSERS_DEFAULT_PIPELINE_PATHS = {
    "xl_base": {"pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0"},
    "xl_refiner": {"pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-refiner-1.0"},
    "xl_inpaint": {"pretrained_model_name_or_path": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"},
    "playground-v2-5": {"pretrained_model_name_or_path": "playgroundai/playground-v2.5-1024px-aesthetic"},
    "upscale": {"pretrained_model_name_or_path": "stabilityai/stable-diffusion-x4-upscaler"},
    "inpainting": {"pretrained_model_name_or_path": "runwayml/stable-diffusion-inpainting"},
    "inpainting_v2": {"pretrained_model_name_or_path": "stabilityai/stable-diffusion-2-inpainting"},
    "controlnet": {"pretrained_model_name_or_path": "lllyasviel/control_v11p_sd15_canny"},
    "v2": {"pretrained_model_name_or_path": "stabilityai/stable-diffusion-2-1"},
    "v1": {"pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5"},
    "stable_cascade_stage_b": {"pretrained_model_name_or_path": "stabilityai/stable-cascade", "subfolder": "decoder"},
    "stable_cascade_stage_b_lite": {
        "pretrained_model_name_or_path": "stabilityai/stable-cascade",
        "subfolder": "decoder_lite",
    },
    "stable_cascade_stage_c": {
        "pretrained_model_name_or_path": "stabilityai/stable-cascade-prior",
        "subfolder": "prior",
    },
    "stable_cascade_stage_c_lite": {
        "pretrained_model_name_or_path": "stabilityai/stable-cascade-prior",
        "subfolder": "prior_lite",
    },
    "sd3": {
        "pretrained_model_name_or_path": "stabilityai/stable-diffusion-3-medium-diffusers",
    },
    "animatediff_v1": {"pretrained_model_name_or_path": "guoyww/animatediff-motion-adapter-v1-5"},
    "animatediff_v2": {"pretrained_model_name_or_path": "guoyww/animatediff-motion-adapter-v1-5-2"},
    "animatediff_v3": {"pretrained_model_name_or_path": "guoyww/animatediff-motion-adapter-v1-5-3"},
    "animatediff_sdxl_beta": {"pretrained_model_name_or_path": "guoyww/animatediff-motion-adapter-sdxl-beta"},
}

# Use to configure model sample size when original config is provided
DIFFUSERS_TO_LDM_DEFAULT_IMAGE_SIZE_MAP = {
    "xl_base": 1024,
    "xl_refiner": 1024,
    "xl_inpaint": 1024,
    "playground-v2-5": 1024,
    "upscale": 512,
    "inpainting": 512,
    "inpainting_v2": 512,
    "controlnet": 512,
    "v2": 768,
    "v1": 512,
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

# To support legacy scheduler_type argument
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

LDM_VAE_KEY = "first_stage_model."
LDM_VAE_DEFAULT_SCALING_FACTOR = 0.18215
PLAYGROUND_VAE_SCALING_FACTOR = 0.5
LDM_UNET_KEY = "model.diffusion_model."
LDM_CONTROLNET_KEY = "control_model."
LDM_CLIP_PREFIX_TO_REMOVE = [
    "cond_stage_model.transformer.",
    "conditioner.embedders.0.transformer.",
]
OPEN_CLIP_PREFIX = "conditioner.embedders.0.model."
LDM_OPEN_CLIP_TEXT_PROJECTION_DIM = 1024

VALID_URL_PREFIXES = ["https://huggingface.co/", "huggingface.co/", "hf.co/", "https://hf.co/"]


class SingleFileComponentError(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(self.message)


def is_valid_url(url):
    result = urlparse(url)
    if result.scheme and result.netloc:
        return True

    return False


def _extract_repo_id_and_weights_name(pretrained_model_name_or_path):
    if not is_valid_url(pretrained_model_name_or_path):
        raise ValueError("Invalid `pretrained_model_name_or_path` provided. Please set it to a valid URL.")

    pattern = r"([^/]+)/([^/]+)/(?:blob/main/)?(.+)"
    weights_name = None
    repo_id = (None,)
    for prefix in VALID_URL_PREFIXES:
        pretrained_model_name_or_path = pretrained_model_name_or_path.replace(prefix, "")
    match = re.match(pattern, pretrained_model_name_or_path)
    if not match:
        logger.warning("Unable to identify the repo_id and weights_name from the provided URL.")
        return repo_id, weights_name

    repo_id = f"{match.group(1)}/{match.group(2)}"
    weights_name = match.group(3)

    return repo_id, weights_name


def _is_model_weights_in_cached_folder(cached_folder, name):
    pretrained_model_name_or_path = os.path.join(cached_folder, name)
    weights_exist = False

    for weights_name in [WEIGHTS_NAME, SAFETENSORS_WEIGHTS_NAME]:
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, weights_name)):
            weights_exist = True

    return weights_exist


def load_single_file_checkpoint(
    pretrained_model_link_or_path,
    force_download=False,
    proxies=None,
    token=None,
    cache_dir=None,
    local_files_only=None,
    revision=None,
):
    if os.path.isfile(pretrained_model_link_or_path):
        pretrained_model_link_or_path = pretrained_model_link_or_path

    else:
        repo_id, weights_name = _extract_repo_id_and_weights_name(pretrained_model_link_or_path)
        pretrained_model_link_or_path = _get_model_file(
            repo_id,
            weights_name=weights_name,
            force_download=force_download,
            cache_dir=cache_dir,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
        )

    checkpoint = load_state_dict(pretrained_model_link_or_path)

    # some checkpoints contain the model state dict under a "state_dict" key
    while "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    return checkpoint


def fetch_original_config(original_config_file, local_files_only=False):
    if os.path.isfile(original_config_file):
        with open(original_config_file, "r") as fp:
            original_config_file = fp.read()

    elif is_valid_url(original_config_file):
        if local_files_only:
            raise ValueError(
                "`local_files_only` is set to True, but a URL was provided as `original_config_file`. "
                "Please provide a valid local file path."
            )

        original_config_file = BytesIO(requests.get(original_config_file).content)

    else:
        raise ValueError("Invalid `original_config_file` provided. Please set it to a valid file path or URL.")

    original_config = yaml.safe_load(original_config_file)

    return original_config


def is_clip_model(checkpoint):
    if CHECKPOINT_KEY_NAMES["clip"] in checkpoint:
        return True

    return False


def is_clip_sdxl_model(checkpoint):
    if CHECKPOINT_KEY_NAMES["clip_sdxl"] in checkpoint:
        return True

    return False


def is_clip_sd3_model(checkpoint):
    if CHECKPOINT_KEY_NAMES["clip_sd3"] in checkpoint:
        return True

    return False


def is_open_clip_model(checkpoint):
    if CHECKPOINT_KEY_NAMES["open_clip"] in checkpoint:
        return True

    return False


def is_open_clip_sdxl_model(checkpoint):
    if CHECKPOINT_KEY_NAMES["open_clip_sdxl"] in checkpoint:
        return True

    return False


def is_open_clip_sd3_model(checkpoint):
    if CHECKPOINT_KEY_NAMES["open_clip_sd3"] in checkpoint:
        return True

    return False


def is_open_clip_sdxl_refiner_model(checkpoint):
    if CHECKPOINT_KEY_NAMES["open_clip_sdxl_refiner"] in checkpoint:
        return True

    return False


def is_clip_model_in_single_file(class_obj, checkpoint):
    is_clip_in_checkpoint = any(
        [
            is_clip_model(checkpoint),
            is_clip_sd3_model(checkpoint),
            is_open_clip_model(checkpoint),
            is_open_clip_sdxl_model(checkpoint),
            is_open_clip_sdxl_refiner_model(checkpoint),
            is_open_clip_sd3_model(checkpoint),
        ]
    )
    if (
        class_obj.__name__ == "CLIPTextModel" or class_obj.__name__ == "CLIPTextModelWithProjection"
    ) and is_clip_in_checkpoint:
        return True

    return False


def infer_diffusers_model_type(checkpoint):
    if (
        CHECKPOINT_KEY_NAMES["inpainting"] in checkpoint
        and checkpoint[CHECKPOINT_KEY_NAMES["inpainting"]].shape[1] == 9
    ):
        if CHECKPOINT_KEY_NAMES["v2"] in checkpoint and checkpoint[CHECKPOINT_KEY_NAMES["v2"]].shape[-1] == 1024:
            model_type = "inpainting_v2"
        else:
            model_type = "inpainting"

    elif CHECKPOINT_KEY_NAMES["v2"] in checkpoint and checkpoint[CHECKPOINT_KEY_NAMES["v2"]].shape[-1] == 1024:
        model_type = "v2"

    elif CHECKPOINT_KEY_NAMES["playground-v2-5"] in checkpoint:
        model_type = "playground-v2-5"

    elif CHECKPOINT_KEY_NAMES["xl_base"] in checkpoint:
        model_type = "xl_base"

    elif CHECKPOINT_KEY_NAMES["xl_refiner"] in checkpoint:
        model_type = "xl_refiner"

    elif CHECKPOINT_KEY_NAMES["upscale"] in checkpoint:
        model_type = "upscale"

    elif CHECKPOINT_KEY_NAMES["controlnet"] in checkpoint:
        model_type = "controlnet"

    elif (
        CHECKPOINT_KEY_NAMES["stable_cascade_stage_c"] in checkpoint
        and checkpoint[CHECKPOINT_KEY_NAMES["stable_cascade_stage_c"]].shape[0] == 1536
    ):
        model_type = "stable_cascade_stage_c_lite"

    elif (
        CHECKPOINT_KEY_NAMES["stable_cascade_stage_c"] in checkpoint
        and checkpoint[CHECKPOINT_KEY_NAMES["stable_cascade_stage_c"]].shape[0] == 2048
    ):
        model_type = "stable_cascade_stage_c"

    elif (
        CHECKPOINT_KEY_NAMES["stable_cascade_stage_b"] in checkpoint
        and checkpoint[CHECKPOINT_KEY_NAMES["stable_cascade_stage_b"]].shape[-1] == 576
    ):
        model_type = "stable_cascade_stage_b_lite"

    elif (
        CHECKPOINT_KEY_NAMES["stable_cascade_stage_b"] in checkpoint
        and checkpoint[CHECKPOINT_KEY_NAMES["stable_cascade_stage_b"]].shape[-1] == 640
    ):
        model_type = "stable_cascade_stage_b"

    elif CHECKPOINT_KEY_NAMES["sd3"] in checkpoint:
        model_type = "sd3"

    elif CHECKPOINT_KEY_NAMES["animatediff"] in checkpoint:
        if CHECKPOINT_KEY_NAMES["animatediff_v2"] in checkpoint:
            model_type = "animatediff_v2"

        elif checkpoint[CHECKPOINT_KEY_NAMES["animatediff_sdxl_beta"]].shape[-1] == 320:
            model_type = "animatediff_sdxl_beta"

        elif checkpoint[CHECKPOINT_KEY_NAMES["animatediff"]].shape[1] == 24:
            model_type = "animatediff_v1"

        else:
            model_type = "animatediff_v3"

    else:
        model_type = "v1"

    return model_type


def fetch_diffusers_config(checkpoint):
    model_type = infer_diffusers_model_type(checkpoint)
    model_path = DIFFUSERS_DEFAULT_PIPELINE_PATHS[model_type]

    return model_path


def set_image_size(checkpoint, image_size=None):
    if image_size:
        return image_size

    model_type = infer_diffusers_model_type(checkpoint)
    image_size = DIFFUSERS_TO_LDM_DEFAULT_IMAGE_SIZE_MAP[model_type]

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


def create_unet_diffusers_config_from_ldm(
    original_config, checkpoint, image_size=None, upcast_attention=None, num_in_channels=None
):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    if image_size is not None:
        deprecation_message = (
            "Configuring UNet2DConditionModel with the `image_size` argument to `from_single_file`"
            "is deprecated and will be ignored in future versions."
        )
        deprecate("image_size", "1.0.0", deprecation_message)

    image_size = set_image_size(checkpoint, image_size=image_size)

    if (
        "unet_config" in original_config["model"]["params"]
        and original_config["model"]["params"]["unet_config"] is not None
    ):
        unet_params = original_config["model"]["params"]["unet_config"]["params"]
    else:
        unet_params = original_config["model"]["params"]["network_config"]["params"]

    if num_in_channels is not None:
        deprecation_message = (
            "Configuring UNet2DConditionModel with the `num_in_channels` argument to `from_single_file`"
            "is deprecated and will be ignored in future versions."
        )
        deprecate("image_size", "1.0.0", deprecation_message)
        in_channels = num_in_channels
    else:
        in_channels = unet_params["in_channels"]

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
        "in_channels": in_channels,
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

    if upcast_attention is not None:
        deprecation_message = (
            "Configuring UNet2DConditionModel with the `upcast_attention` argument to `from_single_file`"
            "is deprecated and will be ignored in future versions."
        )
        deprecate("image_size", "1.0.0", deprecation_message)
        config["upcast_attention"] = upcast_attention

    if "disable_self_attentions" in unet_params:
        config["only_cross_attention"] = unet_params["disable_self_attentions"]

    if "num_classes" in unet_params and isinstance(unet_params["num_classes"], int):
        config["num_class_embeds"] = unet_params["num_classes"]

    config["out_channels"] = unet_params["out_channels"]
    config["up_block_types"] = up_block_types

    return config


def create_controlnet_diffusers_config_from_ldm(original_config, checkpoint, image_size=None, **kwargs):
    if image_size is not None:
        deprecation_message = (
            "Configuring ControlNetModel with the `image_size` argument"
            "is deprecated and will be ignored in future versions."
        )
        deprecate("image_size", "1.0.0", deprecation_message)

    image_size = set_image_size(checkpoint, image_size=image_size)

    unet_params = original_config["model"]["params"]["control_stage_config"]["params"]
    diffusers_unet_config = create_unet_diffusers_config_from_ldm(original_config, image_size=image_size)

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


def create_vae_diffusers_config_from_ldm(original_config, checkpoint, image_size=None, scaling_factor=None):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    if image_size is not None:
        deprecation_message = (
            "Configuring AutoencoderKL with the `image_size` argument"
            "is deprecated and will be ignored in future versions."
        )
        deprecate("image_size", "1.0.0", deprecation_message)

    image_size = set_image_size(checkpoint, image_size=image_size)

    if "edm_mean" in checkpoint and "edm_std" in checkpoint:
        latents_mean = checkpoint["edm_mean"]
        latents_std = checkpoint["edm_std"]
    else:
        latents_mean = None
        latents_std = None

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
        new_checkpoint[diffusers_key] = checkpoint.get(ldm_key)


def update_unet_attention_ldm_to_diffusers(ldm_keys, new_checkpoint, checkpoint, mapping):
    for ldm_key in ldm_keys:
        diffusers_key = ldm_key.replace(mapping["old"], mapping["new"])
        new_checkpoint[diffusers_key] = checkpoint.get(ldm_key)


def update_vae_resnet_ldm_to_diffusers(keys, new_checkpoint, checkpoint, mapping):
    for ldm_key in keys:
        diffusers_key = ldm_key.replace(mapping["old"], mapping["new"]).replace("nin_shortcut", "conv_shortcut")
        new_checkpoint[diffusers_key] = checkpoint.get(ldm_key)


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
        new_checkpoint[diffusers_key] = checkpoint.get(ldm_key)

        # proj_attn.weight has to be converted from conv 1D to linear
        shape = new_checkpoint[diffusers_key].shape

        if len(shape) == 3:
            new_checkpoint[diffusers_key] = new_checkpoint[diffusers_key][:, :, 0]
        elif len(shape) == 4:
            new_checkpoint[diffusers_key] = new_checkpoint[diffusers_key][:, :, 0, 0]


def convert_stable_cascade_unet_single_file_to_diffusers(checkpoint, **kwargs):
    is_stage_c = "clip_txt_mapper.weight" in checkpoint

    if is_stage_c:
        state_dict = {}
        for key in checkpoint.keys():
            if key.endswith("in_proj_weight"):
                weights = checkpoint[key].chunk(3, 0)
                state_dict[key.replace("attn.in_proj_weight", "to_q.weight")] = weights[0]
                state_dict[key.replace("attn.in_proj_weight", "to_k.weight")] = weights[1]
                state_dict[key.replace("attn.in_proj_weight", "to_v.weight")] = weights[2]
            elif key.endswith("in_proj_bias"):
                weights = checkpoint[key].chunk(3, 0)
                state_dict[key.replace("attn.in_proj_bias", "to_q.bias")] = weights[0]
                state_dict[key.replace("attn.in_proj_bias", "to_k.bias")] = weights[1]
                state_dict[key.replace("attn.in_proj_bias", "to_v.bias")] = weights[2]
            elif key.endswith("out_proj.weight"):
                weights = checkpoint[key]
                state_dict[key.replace("attn.out_proj.weight", "to_out.0.weight")] = weights
            elif key.endswith("out_proj.bias"):
                weights = checkpoint[key]
                state_dict[key.replace("attn.out_proj.bias", "to_out.0.bias")] = weights
            else:
                state_dict[key] = checkpoint[key]
    else:
        state_dict = {}
        for key in checkpoint.keys():
            if key.endswith("in_proj_weight"):
                weights = checkpoint[key].chunk(3, 0)
                state_dict[key.replace("attn.in_proj_weight", "to_q.weight")] = weights[0]
                state_dict[key.replace("attn.in_proj_weight", "to_k.weight")] = weights[1]
                state_dict[key.replace("attn.in_proj_weight", "to_v.weight")] = weights[2]
            elif key.endswith("in_proj_bias"):
                weights = checkpoint[key].chunk(3, 0)
                state_dict[key.replace("attn.in_proj_bias", "to_q.bias")] = weights[0]
                state_dict[key.replace("attn.in_proj_bias", "to_k.bias")] = weights[1]
                state_dict[key.replace("attn.in_proj_bias", "to_v.bias")] = weights[2]
            elif key.endswith("out_proj.weight"):
                weights = checkpoint[key]
                state_dict[key.replace("attn.out_proj.weight", "to_out.0.weight")] = weights
            elif key.endswith("out_proj.bias"):
                weights = checkpoint[key]
                state_dict[key.replace("attn.out_proj.bias", "to_out.0.bias")] = weights
            # rename clip_mapper to clip_txt_pooled_mapper
            elif key.endswith("clip_mapper.weight"):
                weights = checkpoint[key]
                state_dict[key.replace("clip_mapper.weight", "clip_txt_pooled_mapper.weight")] = weights
            elif key.endswith("clip_mapper.bias"):
                weights = checkpoint[key]
                state_dict[key.replace("clip_mapper.bias", "clip_txt_pooled_mapper.bias")] = weights
            else:
                state_dict[key] = checkpoint[key]

    return state_dict


def convert_ldm_unet_checkpoint(checkpoint, config, extract_ema=False, **kwargs):
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
                unet_state_dict[key.replace(unet_key, "")] = checkpoint.get(flat_ema_key)
    else:
        if sum(k.startswith("model_ema") for k in keys) > 100:
            logger.warning(
                "In this conversion only the non-EMA weights are extracted. If you want to instead extract the EMA"
                " weights (usually better for inference), please make sure to add the `--extract_ema` flag."
            )
        for key in keys:
            if key.startswith(unet_key):
                unet_state_dict[key.replace(unet_key, "")] = checkpoint.get(key)

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
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = unet_state_dict.get(
                f"input_blocks.{i}.0.op.weight"
            )
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = unet_state_dict.get(
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
    for key in middle_blocks.keys():
        diffusers_key = max(key - 1, 0)
        if key % 2 == 0:
            update_unet_resnet_ldm_to_diffusers(
                middle_blocks[key],
                new_checkpoint,
                unet_state_dict,
                mapping={"old": f"middle_block.{key}", "new": f"mid_block.resnets.{diffusers_key}"},
            )
        else:
            update_unet_attention_ldm_to_diffusers(
                middle_blocks[key],
                new_checkpoint,
                unet_state_dict,
                mapping={"old": f"middle_block.{key}", "new": f"mid_block.attentions.{diffusers_key}"},
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
    **kwargs,
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
                controlnet_state_dict[key.replace(controlnet_key, "")] = checkpoint.get(key)

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
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = controlnet_state_dict.get(
                f"input_blocks.{i}.0.op.weight"
            )
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = controlnet_state_dict.get(
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
        new_checkpoint[f"controlnet_down_blocks.{i}.weight"] = controlnet_state_dict.get(f"zero_convs.{i}.0.weight")
        new_checkpoint[f"controlnet_down_blocks.{i}.bias"] = controlnet_state_dict.get(f"zero_convs.{i}.0.bias")

    # Retrieves the keys for the middle blocks only
    num_middle_blocks = len(
        {".".join(layer.split(".")[:2]) for layer in controlnet_state_dict if "middle_block" in layer}
    )
    middle_blocks = {
        layer_id: [key for key in controlnet_state_dict if f"middle_block.{layer_id}" in key]
        for layer_id in range(num_middle_blocks)
    }

    # Mid blocks
    for key in middle_blocks.keys():
        diffusers_key = max(key - 1, 0)
        if key % 2 == 0:
            update_unet_resnet_ldm_to_diffusers(
                middle_blocks[key],
                new_checkpoint,
                controlnet_state_dict,
                mapping={"old": f"middle_block.{key}", "new": f"mid_block.resnets.{diffusers_key}"},
            )
        else:
            update_unet_attention_ldm_to_diffusers(
                middle_blocks[key],
                new_checkpoint,
                controlnet_state_dict,
                mapping={"old": f"middle_block.{key}", "new": f"mid_block.attentions.{diffusers_key}"},
            )

    # mid block
    new_checkpoint["controlnet_mid_block.weight"] = controlnet_state_dict.get("middle_block_out.0.weight")
    new_checkpoint["controlnet_mid_block.bias"] = controlnet_state_dict.get("middle_block_out.0.bias")

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

        new_checkpoint[f"controlnet_cond_embedding.blocks.{diffusers_idx}.weight"] = controlnet_state_dict.get(
            f"input_hint_block.{cond_block_id}.weight"
        )
        new_checkpoint[f"controlnet_cond_embedding.blocks.{diffusers_idx}.bias"] = controlnet_state_dict.get(
            f"input_hint_block.{cond_block_id}.bias"
        )

    return new_checkpoint


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
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.get(
                f"encoder.down.{i}.downsample.conv.weight"
            )
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.get(
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


def convert_ldm_clip_checkpoint(checkpoint, remove_prefix=None):
    keys = list(checkpoint.keys())
    text_model_dict = {}

    remove_prefixes = []
    remove_prefixes.extend(LDM_CLIP_PREFIX_TO_REMOVE)
    if remove_prefix:
        remove_prefixes.append(remove_prefix)

    for key in keys:
        for prefix in remove_prefixes:
            if key.startswith(prefix):
                diffusers_key = key.replace(prefix, "")
                text_model_dict[diffusers_key] = checkpoint.get(key)

    return text_model_dict


def convert_open_clip_checkpoint(
    text_model,
    checkpoint,
    prefix="cond_stage_model.model.",
):
    text_model_dict = {}
    text_proj_key = prefix + "text_projection"

    if text_proj_key in checkpoint:
        text_proj_dim = int(checkpoint[text_proj_key].shape[0])
    elif hasattr(text_model.config, "projection_dim"):
        text_proj_dim = text_model.config.projection_dim
    else:
        text_proj_dim = LDM_OPEN_CLIP_TEXT_PROJECTION_DIM

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
            weight_value = checkpoint.get(key)

            text_model_dict[diffusers_key + ".q_proj.weight"] = weight_value[:text_proj_dim, :].clone().detach()
            text_model_dict[diffusers_key + ".k_proj.weight"] = (
                weight_value[text_proj_dim : text_proj_dim * 2, :].clone().detach()
            )
            text_model_dict[diffusers_key + ".v_proj.weight"] = weight_value[text_proj_dim * 2 :, :].clone().detach()

        elif key.endswith(".in_proj_bias"):
            weight_value = checkpoint.get(key)
            text_model_dict[diffusers_key + ".q_proj.bias"] = weight_value[:text_proj_dim].clone().detach()
            text_model_dict[diffusers_key + ".k_proj.bias"] = (
                weight_value[text_proj_dim : text_proj_dim * 2].clone().detach()
            )
            text_model_dict[diffusers_key + ".v_proj.bias"] = weight_value[text_proj_dim * 2 :].clone().detach()
        else:
            text_model_dict[diffusers_key] = checkpoint.get(key)

    return text_model_dict


def create_diffusers_clip_model_from_ldm(
    cls,
    checkpoint,
    subfolder="",
    config=None,
    torch_dtype=None,
    local_files_only=None,
    is_legacy_loading=False,
):
    if config:
        config = {"pretrained_model_name_or_path": config}
    else:
        config = fetch_diffusers_config(checkpoint)

    # For backwards compatibility
    # Older versions of `from_single_file` expected CLIP configs to be placed in their original transformers model repo
    # in the cache_dir, rather than in a subfolder of the Diffusers model
    if is_legacy_loading:
        logger.warning(
            (
                "Detected legacy CLIP loading behavior. Please run `from_single_file` with `local_files_only=False once to update "
                "the local cache directory with the necessary CLIP model config files. "
                "Attempting to load CLIP model from legacy cache directory."
            )
        )

        if is_clip_model(checkpoint) or is_clip_sdxl_model(checkpoint):
            clip_config = "openai/clip-vit-large-patch14"
            config["pretrained_model_name_or_path"] = clip_config
            subfolder = ""

        elif is_open_clip_model(checkpoint):
            clip_config = "stabilityai/stable-diffusion-2"
            config["pretrained_model_name_or_path"] = clip_config
            subfolder = "text_encoder"

        else:
            clip_config = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
            config["pretrained_model_name_or_path"] = clip_config
            subfolder = ""

    model_config = cls.config_class.from_pretrained(**config, subfolder=subfolder, local_files_only=local_files_only)
    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    with ctx():
        model = cls(model_config)

    position_embedding_dim = model.text_model.embeddings.position_embedding.weight.shape[-1]

    if is_clip_model(checkpoint):
        diffusers_format_checkpoint = convert_ldm_clip_checkpoint(checkpoint)

    elif (
        is_clip_sdxl_model(checkpoint)
        and checkpoint[CHECKPOINT_KEY_NAMES["clip_sdxl"]].shape[-1] == position_embedding_dim
    ):
        diffusers_format_checkpoint = convert_ldm_clip_checkpoint(checkpoint)

    elif (
        is_clip_sd3_model(checkpoint)
        and checkpoint[CHECKPOINT_KEY_NAMES["clip_sd3"]].shape[-1] == position_embedding_dim
    ):
        diffusers_format_checkpoint = convert_ldm_clip_checkpoint(checkpoint, "text_encoders.clip_l.transformer.")
        diffusers_format_checkpoint["text_projection.weight"] = torch.eye(position_embedding_dim)

    elif is_open_clip_model(checkpoint):
        prefix = "cond_stage_model.model."
        diffusers_format_checkpoint = convert_open_clip_checkpoint(model, checkpoint, prefix=prefix)

    elif (
        is_open_clip_sdxl_model(checkpoint)
        and checkpoint[CHECKPOINT_KEY_NAMES["open_clip_sdxl"]].shape[-1] == position_embedding_dim
    ):
        prefix = "conditioner.embedders.1.model."
        diffusers_format_checkpoint = convert_open_clip_checkpoint(model, checkpoint, prefix=prefix)

    elif is_open_clip_sdxl_refiner_model(checkpoint):
        prefix = "conditioner.embedders.0.model."
        diffusers_format_checkpoint = convert_open_clip_checkpoint(model, checkpoint, prefix=prefix)

    elif (
        is_open_clip_sd3_model(checkpoint)
        and checkpoint[CHECKPOINT_KEY_NAMES["open_clip_sd3"]].shape[-1] == position_embedding_dim
    ):
        diffusers_format_checkpoint = convert_ldm_clip_checkpoint(checkpoint, "text_encoders.clip_g.transformer.")

    else:
        raise ValueError("The provided checkpoint does not seem to contain a valid CLIP model.")

    if is_accelerate_available():
        unexpected_keys = load_model_dict_into_meta(model, diffusers_format_checkpoint, dtype=torch_dtype)
    else:
        _, unexpected_keys = model.load_state_dict(diffusers_format_checkpoint, strict=False)

    if model._keys_to_ignore_on_load_unexpected is not None:
        for pat in model._keys_to_ignore_on_load_unexpected:
            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

    if len(unexpected_keys) > 0:
        logger.warning(
            f"Some weights of the model checkpoint were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}"
        )

    if torch_dtype is not None:
        model.to(torch_dtype)

    model.eval()

    return model


def _legacy_load_scheduler(
    cls,
    checkpoint,
    component_name,
    original_config=None,
    **kwargs,
):
    scheduler_type = kwargs.get("scheduler_type", None)
    prediction_type = kwargs.get("prediction_type", None)

    if scheduler_type is not None:
        deprecation_message = (
            "Please pass an instance of a Scheduler object directly to the `scheduler` argument in `from_single_file`."
        )
        deprecate("scheduler_type", "1.0.0", deprecation_message)

    if prediction_type is not None:
        deprecation_message = (
            "Please configure an instance of a Scheduler with the appropriate `prediction_type` "
            "and pass the object directly to the `scheduler` argument in `from_single_file`."
        )
        deprecate("prediction_type", "1.0.0", deprecation_message)

    scheduler_config = SCHEDULER_DEFAULT_CONFIG
    model_type = infer_diffusers_model_type(checkpoint=checkpoint)

    global_step = checkpoint["global_step"] if "global_step" in checkpoint else None

    if original_config:
        num_train_timesteps = getattr(original_config["model"]["params"], "timesteps", 1000)
    else:
        num_train_timesteps = 1000

    scheduler_config["num_train_timesteps"] = num_train_timesteps

    if model_type == "v2":
        if prediction_type is None:
            # NOTE: For stable diffusion 2 base it is recommended to pass `prediction_type=="epsilon"` # as it relies on a brittle global step parameter here
            prediction_type = "epsilon" if global_step == 875000 else "v_prediction"

    else:
        prediction_type = prediction_type or "epsilon"

    scheduler_config["prediction_type"] = prediction_type

    if model_type in ["xl_base", "xl_refiner"]:
        scheduler_type = "euler"
    elif model_type == "playground":
        scheduler_type = "edm_dpm_solver_multistep"
    else:
        if original_config:
            beta_start = original_config["model"]["params"].get("linear_start")
            beta_end = original_config["model"]["params"].get("linear_end")

        else:
            beta_start = 0.02
            beta_end = 0.085

        scheduler_config["beta_start"] = beta_start
        scheduler_config["beta_end"] = beta_end
        scheduler_config["beta_schedule"] = "scaled_linear"
        scheduler_config["clip_sample"] = False
        scheduler_config["set_alpha_to_one"] = False

    # to deal with an edge case StableDiffusionUpscale pipeline has two schedulers
    if component_name == "low_res_scheduler":
        return cls.from_config(
            {
                "beta_end": 0.02,
                "beta_schedule": "scaled_linear",
                "beta_start": 0.0001,
                "clip_sample": True,
                "num_train_timesteps": 1000,
                "prediction_type": "epsilon",
                "trained_betas": None,
                "variance_type": "fixed_small",
            }
        )

    if scheduler_type is None:
        return cls.from_config(scheduler_config)

    elif scheduler_type == "pndm":
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

    return scheduler


def _legacy_load_clip_tokenizer(cls, checkpoint, config=None, local_files_only=False):
    if config:
        config = {"pretrained_model_name_or_path": config}
    else:
        config = fetch_diffusers_config(checkpoint)

    if is_clip_model(checkpoint) or is_clip_sdxl_model(checkpoint):
        clip_config = "openai/clip-vit-large-patch14"
        config["pretrained_model_name_or_path"] = clip_config
        subfolder = ""

    elif is_open_clip_model(checkpoint):
        clip_config = "stabilityai/stable-diffusion-2"
        config["pretrained_model_name_or_path"] = clip_config
        subfolder = "tokenizer"

    else:
        clip_config = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
        config["pretrained_model_name_or_path"] = clip_config
        subfolder = ""

    tokenizer = cls.from_pretrained(**config, subfolder=subfolder, local_files_only=local_files_only)

    return tokenizer


def _legacy_load_safety_checker(local_files_only, torch_dtype):
    # Support for loading safety checker components using the deprecated
    # `load_safety_checker` argument.

    from ..pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

    feature_extractor = AutoImageProcessor.from_pretrained(
        "CompVis/stable-diffusion-safety-checker", local_files_only=local_files_only, torch_dtype=torch_dtype
    )
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        "CompVis/stable-diffusion-safety-checker", local_files_only=local_files_only, torch_dtype=torch_dtype
    )

    return {"safety_checker": safety_checker, "feature_extractor": feature_extractor}


# in SD3 original implementation of AdaLayerNormContinuous, it split linear projection output into shift, scale;
# while in diffusers it split into scale, shift. Here we swap the linear projection weights in order to be able to use diffusers implementation
def swap_scale_shift(weight, dim):
    shift, scale = weight.chunk(2, dim=0)
    new_weight = torch.cat([scale, shift], dim=0)
    return new_weight


def convert_sd3_transformer_checkpoint_to_diffusers(checkpoint, **kwargs):
    converted_state_dict = {}
    keys = list(checkpoint.keys())
    for k in keys:
        if "model.diffusion_model." in k:
            checkpoint[k.replace("model.diffusion_model.", "")] = checkpoint.pop(k)

    num_layers = list(set(int(k.split(".", 2)[1]) for k in checkpoint if "joint_blocks" in k))[-1] + 1  # noqa: C401
    caption_projection_dim = 1536

    # Positional and patch embeddings.
    converted_state_dict["pos_embed.pos_embed"] = checkpoint.pop("pos_embed")
    converted_state_dict["pos_embed.proj.weight"] = checkpoint.pop("x_embedder.proj.weight")
    converted_state_dict["pos_embed.proj.bias"] = checkpoint.pop("x_embedder.proj.bias")

    # Timestep embeddings.
    converted_state_dict["time_text_embed.timestep_embedder.linear_1.weight"] = checkpoint.pop(
        "t_embedder.mlp.0.weight"
    )
    converted_state_dict["time_text_embed.timestep_embedder.linear_1.bias"] = checkpoint.pop("t_embedder.mlp.0.bias")
    converted_state_dict["time_text_embed.timestep_embedder.linear_2.weight"] = checkpoint.pop(
        "t_embedder.mlp.2.weight"
    )
    converted_state_dict["time_text_embed.timestep_embedder.linear_2.bias"] = checkpoint.pop("t_embedder.mlp.2.bias")

    # Context projections.
    converted_state_dict["context_embedder.weight"] = checkpoint.pop("context_embedder.weight")
    converted_state_dict["context_embedder.bias"] = checkpoint.pop("context_embedder.bias")

    # Pooled context projection.
    converted_state_dict["time_text_embed.text_embedder.linear_1.weight"] = checkpoint.pop("y_embedder.mlp.0.weight")
    converted_state_dict["time_text_embed.text_embedder.linear_1.bias"] = checkpoint.pop("y_embedder.mlp.0.bias")
    converted_state_dict["time_text_embed.text_embedder.linear_2.weight"] = checkpoint.pop("y_embedder.mlp.2.weight")
    converted_state_dict["time_text_embed.text_embedder.linear_2.bias"] = checkpoint.pop("y_embedder.mlp.2.bias")

    # Transformer blocks 🎸.
    for i in range(num_layers):
        # Q, K, V
        sample_q, sample_k, sample_v = torch.chunk(
            checkpoint.pop(f"joint_blocks.{i}.x_block.attn.qkv.weight"), 3, dim=0
        )
        context_q, context_k, context_v = torch.chunk(
            checkpoint.pop(f"joint_blocks.{i}.context_block.attn.qkv.weight"), 3, dim=0
        )
        sample_q_bias, sample_k_bias, sample_v_bias = torch.chunk(
            checkpoint.pop(f"joint_blocks.{i}.x_block.attn.qkv.bias"), 3, dim=0
        )
        context_q_bias, context_k_bias, context_v_bias = torch.chunk(
            checkpoint.pop(f"joint_blocks.{i}.context_block.attn.qkv.bias"), 3, dim=0
        )

        converted_state_dict[f"transformer_blocks.{i}.attn.to_q.weight"] = torch.cat([sample_q])
        converted_state_dict[f"transformer_blocks.{i}.attn.to_q.bias"] = torch.cat([sample_q_bias])
        converted_state_dict[f"transformer_blocks.{i}.attn.to_k.weight"] = torch.cat([sample_k])
        converted_state_dict[f"transformer_blocks.{i}.attn.to_k.bias"] = torch.cat([sample_k_bias])
        converted_state_dict[f"transformer_blocks.{i}.attn.to_v.weight"] = torch.cat([sample_v])
        converted_state_dict[f"transformer_blocks.{i}.attn.to_v.bias"] = torch.cat([sample_v_bias])

        converted_state_dict[f"transformer_blocks.{i}.attn.add_q_proj.weight"] = torch.cat([context_q])
        converted_state_dict[f"transformer_blocks.{i}.attn.add_q_proj.bias"] = torch.cat([context_q_bias])
        converted_state_dict[f"transformer_blocks.{i}.attn.add_k_proj.weight"] = torch.cat([context_k])
        converted_state_dict[f"transformer_blocks.{i}.attn.add_k_proj.bias"] = torch.cat([context_k_bias])
        converted_state_dict[f"transformer_blocks.{i}.attn.add_v_proj.weight"] = torch.cat([context_v])
        converted_state_dict[f"transformer_blocks.{i}.attn.add_v_proj.bias"] = torch.cat([context_v_bias])

        # output projections.
        converted_state_dict[f"transformer_blocks.{i}.attn.to_out.0.weight"] = checkpoint.pop(
            f"joint_blocks.{i}.x_block.attn.proj.weight"
        )
        converted_state_dict[f"transformer_blocks.{i}.attn.to_out.0.bias"] = checkpoint.pop(
            f"joint_blocks.{i}.x_block.attn.proj.bias"
        )
        if not (i == num_layers - 1):
            converted_state_dict[f"transformer_blocks.{i}.attn.to_add_out.weight"] = checkpoint.pop(
                f"joint_blocks.{i}.context_block.attn.proj.weight"
            )
            converted_state_dict[f"transformer_blocks.{i}.attn.to_add_out.bias"] = checkpoint.pop(
                f"joint_blocks.{i}.context_block.attn.proj.bias"
            )

        # norms.
        converted_state_dict[f"transformer_blocks.{i}.norm1.linear.weight"] = checkpoint.pop(
            f"joint_blocks.{i}.x_block.adaLN_modulation.1.weight"
        )
        converted_state_dict[f"transformer_blocks.{i}.norm1.linear.bias"] = checkpoint.pop(
            f"joint_blocks.{i}.x_block.adaLN_modulation.1.bias"
        )
        if not (i == num_layers - 1):
            converted_state_dict[f"transformer_blocks.{i}.norm1_context.linear.weight"] = checkpoint.pop(
                f"joint_blocks.{i}.context_block.adaLN_modulation.1.weight"
            )
            converted_state_dict[f"transformer_blocks.{i}.norm1_context.linear.bias"] = checkpoint.pop(
                f"joint_blocks.{i}.context_block.adaLN_modulation.1.bias"
            )
        else:
            converted_state_dict[f"transformer_blocks.{i}.norm1_context.linear.weight"] = swap_scale_shift(
                checkpoint.pop(f"joint_blocks.{i}.context_block.adaLN_modulation.1.weight"),
                dim=caption_projection_dim,
            )
            converted_state_dict[f"transformer_blocks.{i}.norm1_context.linear.bias"] = swap_scale_shift(
                checkpoint.pop(f"joint_blocks.{i}.context_block.adaLN_modulation.1.bias"),
                dim=caption_projection_dim,
            )

        # ffs.
        converted_state_dict[f"transformer_blocks.{i}.ff.net.0.proj.weight"] = checkpoint.pop(
            f"joint_blocks.{i}.x_block.mlp.fc1.weight"
        )
        converted_state_dict[f"transformer_blocks.{i}.ff.net.0.proj.bias"] = checkpoint.pop(
            f"joint_blocks.{i}.x_block.mlp.fc1.bias"
        )
        converted_state_dict[f"transformer_blocks.{i}.ff.net.2.weight"] = checkpoint.pop(
            f"joint_blocks.{i}.x_block.mlp.fc2.weight"
        )
        converted_state_dict[f"transformer_blocks.{i}.ff.net.2.bias"] = checkpoint.pop(
            f"joint_blocks.{i}.x_block.mlp.fc2.bias"
        )
        if not (i == num_layers - 1):
            converted_state_dict[f"transformer_blocks.{i}.ff_context.net.0.proj.weight"] = checkpoint.pop(
                f"joint_blocks.{i}.context_block.mlp.fc1.weight"
            )
            converted_state_dict[f"transformer_blocks.{i}.ff_context.net.0.proj.bias"] = checkpoint.pop(
                f"joint_blocks.{i}.context_block.mlp.fc1.bias"
            )
            converted_state_dict[f"transformer_blocks.{i}.ff_context.net.2.weight"] = checkpoint.pop(
                f"joint_blocks.{i}.context_block.mlp.fc2.weight"
            )
            converted_state_dict[f"transformer_blocks.{i}.ff_context.net.2.bias"] = checkpoint.pop(
                f"joint_blocks.{i}.context_block.mlp.fc2.bias"
            )

    # Final blocks.
    converted_state_dict["proj_out.weight"] = checkpoint.pop("final_layer.linear.weight")
    converted_state_dict["proj_out.bias"] = checkpoint.pop("final_layer.linear.bias")
    converted_state_dict["norm_out.linear.weight"] = swap_scale_shift(
        checkpoint.pop("final_layer.adaLN_modulation.1.weight"), dim=caption_projection_dim
    )
    converted_state_dict["norm_out.linear.bias"] = swap_scale_shift(
        checkpoint.pop("final_layer.adaLN_modulation.1.bias"), dim=caption_projection_dim
    )

    return converted_state_dict


def is_t5_in_single_file(checkpoint):
    if "text_encoders.t5xxl.transformer.shared.weight" in checkpoint:
        return True

    return False


def convert_sd3_t5_checkpoint_to_diffusers(checkpoint):
    keys = list(checkpoint.keys())
    text_model_dict = {}

    remove_prefixes = ["text_encoders.t5xxl.transformer."]

    for key in keys:
        for prefix in remove_prefixes:
            if key.startswith(prefix):
                diffusers_key = key.replace(prefix, "")
                text_model_dict[diffusers_key] = checkpoint.get(key)

    return text_model_dict


def create_diffusers_t5_model_from_checkpoint(
    cls,
    checkpoint,
    subfolder="",
    config=None,
    torch_dtype=None,
    local_files_only=None,
):
    if config:
        config = {"pretrained_model_name_or_path": config}
    else:
        config = fetch_diffusers_config(checkpoint)

    model_config = cls.config_class.from_pretrained(**config, subfolder=subfolder, local_files_only=local_files_only)
    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    with ctx():
        model = cls(model_config)

    diffusers_format_checkpoint = convert_sd3_t5_checkpoint_to_diffusers(checkpoint)

    if is_accelerate_available():
        unexpected_keys = load_model_dict_into_meta(model, diffusers_format_checkpoint, dtype=torch_dtype)
        if model._keys_to_ignore_on_load_unexpected is not None:
            for pat in model._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}"
            )

    else:
        model.load_state_dict(diffusers_format_checkpoint)

    use_keep_in_fp32_modules = (cls._keep_in_fp32_modules is not None) and (torch_dtype == torch.float16)
    if use_keep_in_fp32_modules:
        keep_in_fp32_modules = model._keep_in_fp32_modules
    else:
        keep_in_fp32_modules = []

    if keep_in_fp32_modules is not None:
        for name, param in model.named_parameters():
            if any(module_to_keep_in_fp32 in name.split(".") for module_to_keep_in_fp32 in keep_in_fp32_modules):
                # param = param.to(torch.float32) does not work here as only in the local scope.
                param.data = param.data.to(torch.float32)

    return model


def convert_animatediff_checkpoint_to_diffusers(checkpoint, **kwargs):
    converted_state_dict = {}
    for k, v in checkpoint.items():
        if "pos_encoder" in k:
            continue

        else:
            converted_state_dict[
                k.replace(".norms.0", ".norm1")
                .replace(".norms.1", ".norm2")
                .replace(".ff_norm", ".norm3")
                .replace(".attention_blocks.0", ".attn1")
                .replace(".attention_blocks.1", ".attn2")
                .replace(".temporal_transformer", "")
            ] = v

    return converted_state_dict
