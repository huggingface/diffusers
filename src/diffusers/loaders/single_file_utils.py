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
"""Checkpoint discovery, configuration and component loading for single-file pipelines."""

import copy
import os
import re
from collections.abc import Mapping
from contextlib import nullcontext
from io import BytesIO
from typing import Any
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
from ..utils.constants import DIFFUSERS_REQUEST_TIMEOUT
from ..utils.hub_utils import _get_model_file
from ..utils.torch_utils import empty_device_cache


if is_transformers_available():
    from transformers import AutoImageProcessor

if is_accelerate_available():
    from accelerate import init_empty_weights

    from ..models.model_loading_utils import load_model_dict_into_meta

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def convert_model_checkpoint(
    checkpoint: Mapping[str, Any],
    config: dict[str, Any],
    extract_ema: bool = False,
    return_config: bool = False,
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Convert a resolved single-file model using the shared bidirectional component definition.

    Model loading and original-format export share the config-driven definitions in `loaders.conversion`.
    """
    from .conversion.checkpoint import convert_component_checkpoint

    return convert_component_checkpoint(
        checkpoint, config, config["_class_name"], extract_ema=extract_ema, return_config=return_config
    )


CHECKPOINT_KEY_NAMES = {
    "v1": "model.diffusion_model.output_blocks.11.0.skip_connection.weight",
    "v2": "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight",
    "xl_base": "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias",
    "xl_refiner": "conditioner.embedders.0.model.transformer.resblocks.9.mlp.c_proj.bias",
    "upscale": "model.diffusion_model.input_blocks.10.0.skip_connection.bias",
    "controlnet": [
        "control_model.time_embed.0.weight",
        "controlnet_cond_embedding.conv_in.weight",
    ],
    # TODO: find non-Diffusers keys for controlnet_xl
    "controlnet_xl": "add_embedding.linear_1.weight",
    "controlnet_xl_large": "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k.weight",
    "controlnet_xl_mid": "down_blocks.1.attentions.0.norm.weight",
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
    "sd3": [
        "joint_blocks.0.context_block.adaLN_modulation.1.bias",
        "model.diffusion_model.joint_blocks.0.context_block.adaLN_modulation.1.bias",
    ],
    "sd35_large": [
        "joint_blocks.37.x_block.mlp.fc1.weight",
        "model.diffusion_model.joint_blocks.37.x_block.mlp.fc1.weight",
    ],
    "animatediff": "down_blocks.0.motion_modules.0.temporal_transformer.transformer_blocks.0.attention_blocks.0.pos_encoder.pe",
    "animatediff_v2": "mid_block.motion_modules.0.temporal_transformer.norm.bias",
    "animatediff_sdxl_beta": "up_blocks.2.motion_modules.0.temporal_transformer.norm.weight",
    "animatediff_scribble": "controlnet_cond_embedding.conv_in.weight",
    "animatediff_rgb": "controlnet_cond_embedding.weight",
    "auraflow": [
        "double_layers.0.attn.w2q.weight",
        "double_layers.0.attn.w1q.weight",
        "cond_seq_linear.weight",
        "t_embedder.mlp.0.weight",
    ],
    "flux": [
        "double_blocks.0.img_attn.norm.key_norm.scale",
        "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale",
    ],
    "ltx-video": [
        "model.diffusion_model.patchify_proj.weight",
        "model.diffusion_model.transformer_blocks.27.scale_shift_table",
        "patchify_proj.weight",
        "transformer_blocks.27.scale_shift_table",
        "vae.decoder.last_scale_shift_table",  # 0.9.1, 0.9.5, 0.9.7, 0.9.8
        "vae.decoder.up_blocks.9.res_blocks.0.conv1.conv.weight",  # 0.9.0
    ],
    "autoencoder-dc": "decoder.stages.1.op_list.0.main.conv.conv.bias",
    "autoencoder-dc-sana": "encoder.project_in.conv.bias",
    "mochi-1-preview": ["model.diffusion_model.blocks.0.attn.qkv_x.weight", "blocks.0.attn.qkv_x.weight"],
    "hunyuan-video": "txt_in.individual_token_refiner.blocks.0.adaLN_modulation.1.bias",
    "instruct-pix2pix": "model.diffusion_model.input_blocks.0.0.weight",
    "lumina2": ["model.diffusion_model.cap_embedder.0.weight", "cap_embedder.0.weight"],
    "z-image-turbo": [
        "model.diffusion_model.layers.0.adaLN_modulation.0.weight",
        "layers.0.adaLN_modulation.0.weight",
    ],
    "z-image-turbo-controlnet": "control_all_x_embedder.2-1.weight",
    "z-image-turbo-controlnet-2.x": "control_layers.14.adaLN_modulation.0.weight",
    "sana": [
        "blocks.0.cross_attn.q_linear.weight",
        "blocks.0.cross_attn.q_linear.bias",
        "blocks.0.cross_attn.kv_linear.weight",
        "blocks.0.cross_attn.kv_linear.bias",
    ],
    "wan": ["model.diffusion_model.head.modulation", "head.modulation"],
    "wan_vae": "decoder.middle.0.residual.0.gamma",
    "wan_vace": "vace_blocks.0.after_proj.bias",
    "wan_animate": "motion_encoder.dec.direction.weight",
    "hidream": "double_stream_blocks.0.block.adaLN_modulation.1.bias",
    "cosmos-1.0": [
        "net.x_embedder.proj.1.weight",
        "net.blocks.block1.blocks.0.block.attn.to_q.0.weight",
        "net.extra_pos_embedder.pos_emb_h",
    ],
    "cosmos-2.0": [
        "net.x_embedder.proj.1.weight",
        "net.blocks.0.self_attn.q_proj.weight",
        "net.pos_embedder.dim_spatial_range",
    ],
    "flux2": ["model.diffusion_model.single_stream_modulation.lin.weight", "single_stream_modulation.lin.weight"],
    "chroma": [
        "model.diffusion_model.distilled_guidance_layer.in_proj.bias",
        "distilled_guidance_layer.in_proj.bias",
    ],
    "ltx2": [
        "model.diffusion_model.av_ca_a2v_gate_adaln_single.emb.timestep_embedder.linear_1.weight",
        "vae.per_channel_statistics.mean-of-means",
        "audio_vae.per_channel_statistics.mean-of-means",
    ],
}

DIFFUSERS_DEFAULT_PIPELINE_PATHS = {
    "xl_base": {"pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0"},
    "xl_refiner": {"pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-refiner-1.0"},
    "xl_inpaint": {"pretrained_model_name_or_path": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"},
    "playground-v2-5": {"pretrained_model_name_or_path": "playgroundai/playground-v2.5-1024px-aesthetic"},
    "upscale": {"pretrained_model_name_or_path": "stabilityai/stable-diffusion-x4-upscaler"},
    "inpainting": {"pretrained_model_name_or_path": "stable-diffusion-v1-5/stable-diffusion-inpainting"},
    "inpainting_v2": {"pretrained_model_name_or_path": "stabilityai/stable-diffusion-2-inpainting"},
    "controlnet": {"pretrained_model_name_or_path": "lllyasviel/control_v11p_sd15_canny"},
    "controlnet_xl_large": {"pretrained_model_name_or_path": "diffusers/controlnet-canny-sdxl-1.0"},
    "controlnet_xl_mid": {"pretrained_model_name_or_path": "diffusers/controlnet-canny-sdxl-1.0-mid"},
    "controlnet_xl_small": {"pretrained_model_name_or_path": "diffusers/controlnet-canny-sdxl-1.0-small"},
    "v2": {"pretrained_model_name_or_path": "stabilityai/stable-diffusion-2-1"},
    "v1": {"pretrained_model_name_or_path": "stable-diffusion-v1-5/stable-diffusion-v1-5"},
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
    "sd35_large": {
        "pretrained_model_name_or_path": "stabilityai/stable-diffusion-3.5-large",
    },
    "sd35_medium": {
        "pretrained_model_name_or_path": "stabilityai/stable-diffusion-3.5-medium",
    },
    "animatediff_v1": {"pretrained_model_name_or_path": "guoyww/animatediff-motion-adapter-v1-5"},
    "animatediff_v2": {"pretrained_model_name_or_path": "guoyww/animatediff-motion-adapter-v1-5-2"},
    "animatediff_v3": {"pretrained_model_name_or_path": "guoyww/animatediff-motion-adapter-v1-5-3"},
    "animatediff_sdxl_beta": {"pretrained_model_name_or_path": "guoyww/animatediff-motion-adapter-sdxl-beta"},
    "animatediff_scribble": {"pretrained_model_name_or_path": "guoyww/animatediff-sparsectrl-scribble"},
    "animatediff_rgb": {"pretrained_model_name_or_path": "guoyww/animatediff-sparsectrl-rgb"},
    "auraflow": {"pretrained_model_name_or_path": "fal/AuraFlow-v0.3"},
    "flux-dev": {"pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev"},
    "flux-fill": {"pretrained_model_name_or_path": "black-forest-labs/FLUX.1-Fill-dev"},
    "flux-depth": {"pretrained_model_name_or_path": "black-forest-labs/FLUX.1-Depth-dev"},
    "flux-schnell": {"pretrained_model_name_or_path": "black-forest-labs/FLUX.1-schnell"},
    "flux-2-dev": {"pretrained_model_name_or_path": "black-forest-labs/FLUX.2-dev"},
    "chroma": {"pretrained_model_name_or_path": "lodestones/Chroma1-HD"},
    "ltx-video": {"pretrained_model_name_or_path": "diffusers/LTX-Video-0.9.0"},
    "ltx-video-0.9.1": {"pretrained_model_name_or_path": "diffusers/LTX-Video-0.9.1"},
    "ltx-video-0.9.5": {"pretrained_model_name_or_path": "Lightricks/LTX-Video-0.9.5"},
    "ltx-video-0.9.7": {"pretrained_model_name_or_path": "Lightricks/LTX-Video-0.9.7-dev"},
    "autoencoder-dc-f128c512": {"pretrained_model_name_or_path": "mit-han-lab/dc-ae-f128c512-mix-1.0-diffusers"},
    "autoencoder-dc-f64c128": {"pretrained_model_name_or_path": "mit-han-lab/dc-ae-f64c128-mix-1.0-diffusers"},
    "autoencoder-dc-f32c32": {"pretrained_model_name_or_path": "mit-han-lab/dc-ae-f32c32-mix-1.0-diffusers"},
    "autoencoder-dc-f32c32-sana": {"pretrained_model_name_or_path": "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers"},
    "mochi-1-preview": {"pretrained_model_name_or_path": "genmo/mochi-1-preview"},
    "hunyuan-video": {"pretrained_model_name_or_path": "hunyuanvideo-community/HunyuanVideo"},
    "instruct-pix2pix": {"pretrained_model_name_or_path": "timbrooks/instruct-pix2pix"},
    "lumina2": {"pretrained_model_name_or_path": "Alpha-VLLM/Lumina-Image-2.0"},
    "sana": {"pretrained_model_name_or_path": "Efficient-Large-Model/Sana_1600M_1024px_diffusers"},
    "wan-t2v-1.3B": {"pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"},
    "wan-t2v-14B": {"pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-14B-Diffusers"},
    "wan-i2v-14B": {"pretrained_model_name_or_path": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"},
    "wan-animate-14B": {"pretrained_model_name_or_path": "Wan-AI/Wan2.2-Animate-14B-Diffusers"},
    "wan-vace-1.3B": {"pretrained_model_name_or_path": "Wan-AI/Wan2.1-VACE-1.3B-diffusers"},
    "wan-vace-14B": {"pretrained_model_name_or_path": "Wan-AI/Wan2.1-VACE-14B-diffusers"},
    "hidream": {"pretrained_model_name_or_path": "HiDream-ai/HiDream-I1-Dev"},
    "cosmos-1.0-t2w-7B": {"pretrained_model_name_or_path": "nvidia/Cosmos-1.0-Diffusion-7B-Text2World"},
    "cosmos-1.0-t2w-14B": {"pretrained_model_name_or_path": "nvidia/Cosmos-1.0-Diffusion-14B-Text2World"},
    "cosmos-1.0-v2w-7B": {"pretrained_model_name_or_path": "nvidia/Cosmos-1.0-Diffusion-7B-Video2World"},
    "cosmos-1.0-v2w-14B": {"pretrained_model_name_or_path": "nvidia/Cosmos-1.0-Diffusion-14B-Video2World"},
    "cosmos-2.0-t2i-2B": {"pretrained_model_name_or_path": "nvidia/Cosmos-Predict2-2B-Text2Image"},
    "cosmos-2.0-t2i-14B": {"pretrained_model_name_or_path": "nvidia/Cosmos-Predict2-14B-Text2Image"},
    "cosmos-2.0-v2w-2B": {"pretrained_model_name_or_path": "nvidia/Cosmos-Predict2-2B-Video2World"},
    "cosmos-2.0-v2w-14B": {"pretrained_model_name_or_path": "nvidia/Cosmos-Predict2-14B-Video2World"},
    "z-image-turbo": {"pretrained_model_name_or_path": "Tongyi-MAI/Z-Image-Turbo"},
    "z-image-turbo-controlnet": {"pretrained_model_name_or_path": "hlky/Z-Image-Turbo-Fun-Controlnet-Union"},
    "z-image-turbo-controlnet-2.0": {"pretrained_model_name_or_path": "hlky/Z-Image-Turbo-Fun-Controlnet-Union-2.0"},
    "z-image-turbo-controlnet-2.1": {"pretrained_model_name_or_path": "hlky/Z-Image-Turbo-Fun-Controlnet-Union-2.1"},
    "ltx2-dev": {"pretrained_model_name_or_path": "Lightricks/LTX-2"},
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
    "instruct-pix2pix": 512,
    "v2": 768,
    "v1": 512,
}


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

LDM_VAE_DEFAULT_SCALING_FACTOR = 0.18215
PLAYGROUND_VAE_SCALING_FACTOR = 0.5
LDM_CLIP_PREFIX_TO_REMOVE = [
    "cond_stage_model.transformer.",
    "conditioner.embedders.0.transformer.",
]
SCHEDULER_LEGACY_KWARGS = ["prediction_type", "scheduler_type"]

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


def _is_single_file_path_or_url(pretrained_model_name_or_path):
    if os.path.isfile(pretrained_model_name_or_path):
        return True

    if not is_valid_url(pretrained_model_name_or_path):
        return False

    repo_id, weight_name = _extract_repo_id_and_weights_name(pretrained_model_name_or_path)
    return bool(repo_id and weight_name)


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


def _is_legacy_scheduler_kwargs(kwargs):
    return any(k in SCHEDULER_LEGACY_KWARGS for k in kwargs.keys())


def load_single_file_checkpoint(
    pretrained_model_link_or_path,
    force_download=False,
    proxies=None,
    token=None,
    cache_dir=None,
    local_files_only=None,
    revision=None,
    disable_mmap=False,
    user_agent=None,
):
    if user_agent is None:
        user_agent = {"file_type": "single_file", "framework": "pytorch"}

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
            user_agent=user_agent,
        )

    checkpoint = load_state_dict(pretrained_model_link_or_path, disable_mmap=disable_mmap)

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

        original_config_file = BytesIO(requests.get(original_config_file, timeout=DIFFUSERS_REQUEST_TIMEOUT).content)

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
        elif CHECKPOINT_KEY_NAMES["xl_base"] in checkpoint:
            model_type = "xl_inpaint"
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

    elif any(key in checkpoint for key in CHECKPOINT_KEY_NAMES["controlnet"]):
        if CHECKPOINT_KEY_NAMES["controlnet_xl"] in checkpoint:
            if CHECKPOINT_KEY_NAMES["controlnet_xl_large"] in checkpoint:
                model_type = "controlnet_xl_large"
            elif CHECKPOINT_KEY_NAMES["controlnet_xl_mid"] in checkpoint:
                model_type = "controlnet_xl_mid"
            else:
                model_type = "controlnet_xl_small"
        else:
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

    elif any(key in checkpoint for key in CHECKPOINT_KEY_NAMES["sd3"]) and any(
        checkpoint[key].shape[-1] == 9216 if key in checkpoint else False for key in CHECKPOINT_KEY_NAMES["sd3"]
    ):
        if "model.diffusion_model.pos_embed" in checkpoint:
            key = "model.diffusion_model.pos_embed"
        else:
            key = "pos_embed"

        if checkpoint[key].shape[1] == 36864:
            model_type = "sd3"
        elif checkpoint[key].shape[1] == 147456:
            model_type = "sd35_medium"

    elif any(key in checkpoint for key in CHECKPOINT_KEY_NAMES["sd35_large"]):
        model_type = "sd35_large"

    elif CHECKPOINT_KEY_NAMES["animatediff"] in checkpoint:
        if CHECKPOINT_KEY_NAMES["animatediff_scribble"] in checkpoint:
            model_type = "animatediff_scribble"

        elif CHECKPOINT_KEY_NAMES["animatediff_rgb"] in checkpoint:
            model_type = "animatediff_rgb"

        elif CHECKPOINT_KEY_NAMES["animatediff_v2"] in checkpoint:
            model_type = "animatediff_v2"

        elif checkpoint[CHECKPOINT_KEY_NAMES["animatediff_sdxl_beta"]].shape[-1] == 320:
            model_type = "animatediff_sdxl_beta"

        elif checkpoint[CHECKPOINT_KEY_NAMES["animatediff"]].shape[1] == 24:
            model_type = "animatediff_v1"

        else:
            model_type = "animatediff_v3"

    elif any(key in checkpoint for key in CHECKPOINT_KEY_NAMES["flux2"]):
        model_type = "flux-2-dev"

    elif any(key in checkpoint for key in CHECKPOINT_KEY_NAMES["chroma"]):
        model_type = "chroma"

    elif any(key in checkpoint for key in CHECKPOINT_KEY_NAMES["flux"]):
        if any(
            g in checkpoint for g in ["guidance_in.in_layer.bias", "model.diffusion_model.guidance_in.in_layer.bias"]
        ):
            if "model.diffusion_model.img_in.weight" in checkpoint:
                key = "model.diffusion_model.img_in.weight"
            else:
                key = "img_in.weight"

            if checkpoint[key].shape[1] == 384:
                model_type = "flux-fill"
            elif checkpoint[key].shape[1] == 128:
                model_type = "flux-depth"
            else:
                model_type = "flux-dev"
        else:
            model_type = "flux-schnell"

    elif any(key in checkpoint for key in CHECKPOINT_KEY_NAMES["ltx-video"]):
        has_vae = "vae.encoder.conv_in.conv.bias" in checkpoint
        if any(key.endswith("transformer_blocks.47.scale_shift_table") for key in checkpoint):
            model_type = "ltx-video-0.9.7"
        elif has_vae and checkpoint["vae.encoder.conv_out.conv.weight"].shape[1] == 2048:
            model_type = "ltx-video-0.9.5"
        elif "vae.decoder.last_time_embedder.timestep_embedder.linear_1.weight" in checkpoint:
            model_type = "ltx-video-0.9.1"
        else:
            model_type = "ltx-video"

    elif CHECKPOINT_KEY_NAMES["autoencoder-dc"] in checkpoint:
        encoder_key = "encoder.project_in.conv.conv.bias"
        decoder_key = "decoder.project_in.main.conv.weight"

        if CHECKPOINT_KEY_NAMES["autoencoder-dc-sana"] in checkpoint:
            model_type = "autoencoder-dc-f32c32-sana"

        elif checkpoint[encoder_key].shape[-1] == 64 and checkpoint[decoder_key].shape[1] == 32:
            model_type = "autoencoder-dc-f32c32"

        elif checkpoint[encoder_key].shape[-1] == 64 and checkpoint[decoder_key].shape[1] == 128:
            model_type = "autoencoder-dc-f64c128"

        else:
            model_type = "autoencoder-dc-f128c512"

    elif any(key in checkpoint for key in CHECKPOINT_KEY_NAMES["mochi-1-preview"]):
        model_type = "mochi-1-preview"

    elif CHECKPOINT_KEY_NAMES["hunyuan-video"] in checkpoint:
        model_type = "hunyuan-video"

    elif all(key in checkpoint for key in CHECKPOINT_KEY_NAMES["auraflow"]):
        model_type = "auraflow"

    elif (
        CHECKPOINT_KEY_NAMES["instruct-pix2pix"] in checkpoint
        and checkpoint[CHECKPOINT_KEY_NAMES["instruct-pix2pix"]].shape[1] == 8
    ):
        model_type = "instruct-pix2pix"

    elif any(key in checkpoint for key in CHECKPOINT_KEY_NAMES["z-image-turbo"]):
        model_type = "z-image-turbo"

    elif any(key in checkpoint for key in CHECKPOINT_KEY_NAMES["lumina2"]):
        model_type = "lumina2"

    elif any(key in checkpoint for key in CHECKPOINT_KEY_NAMES["sana"]):
        model_type = "sana"

    elif any(key in checkpoint for key in CHECKPOINT_KEY_NAMES["wan"]):
        if "model.diffusion_model.patch_embedding.weight" in checkpoint:
            prefix = "model.diffusion_model."
        else:
            prefix = ""

        target_key = f"{prefix}patch_embedding.weight"

        if f"{prefix}{CHECKPOINT_KEY_NAMES['wan_vace']}" in checkpoint:
            if checkpoint[target_key].shape[0] == 1536:
                model_type = "wan-vace-1.3B"
            elif checkpoint[target_key].shape[0] == 5120:
                model_type = "wan-vace-14B"

        elif f"{prefix}{CHECKPOINT_KEY_NAMES['wan_animate']}" in checkpoint:
            model_type = "wan-animate-14B"

        elif checkpoint[target_key].shape[0] == 1536:
            model_type = "wan-t2v-1.3B"
        elif checkpoint[target_key].shape[0] == 5120 and checkpoint[target_key].shape[1] == 16:
            model_type = "wan-t2v-14B"
        else:
            model_type = "wan-i2v-14B"

    elif CHECKPOINT_KEY_NAMES["wan_vae"] in checkpoint:
        # All Wan models use the same VAE so we can use the same default model repo to fetch the config
        model_type = "wan-t2v-14B"

    elif CHECKPOINT_KEY_NAMES["hidream"] in checkpoint:
        model_type = "hidream"

    elif all(key in checkpoint for key in CHECKPOINT_KEY_NAMES["cosmos-1.0"]):
        x_embedder_shape = checkpoint[CHECKPOINT_KEY_NAMES["cosmos-1.0"][0]].shape
        if x_embedder_shape[1] == 68:
            model_type = "cosmos-1.0-t2w-7B" if x_embedder_shape[0] == 4096 else "cosmos-1.0-t2w-14B"
        elif x_embedder_shape[1] == 72:
            model_type = "cosmos-1.0-v2w-7B" if x_embedder_shape[0] == 4096 else "cosmos-1.0-v2w-14B"
        else:
            raise ValueError(f"Unexpected x_embedder shape: {x_embedder_shape} when loading Cosmos 1.0 model.")

    elif all(key in checkpoint for key in CHECKPOINT_KEY_NAMES["cosmos-2.0"]):
        x_embedder_shape = checkpoint[CHECKPOINT_KEY_NAMES["cosmos-2.0"][0]].shape
        if x_embedder_shape[1] == 68:
            model_type = "cosmos-2.0-t2i-2B" if x_embedder_shape[0] == 2048 else "cosmos-2.0-t2i-14B"
        elif x_embedder_shape[1] == 72:
            model_type = "cosmos-2.0-v2w-2B" if x_embedder_shape[0] == 2048 else "cosmos-2.0-v2w-14B"
        else:
            raise ValueError(f"Unexpected x_embedder shape: {x_embedder_shape} when loading Cosmos 2.0 model.")

    elif CHECKPOINT_KEY_NAMES["z-image-turbo-controlnet-2.x"] in checkpoint:
        before_proj_weight = checkpoint.get("control_noise_refiner.0.before_proj.weight", None)
        if before_proj_weight is None:
            model_type = "z-image-turbo-controlnet-2.0"
        elif before_proj_weight is not None and torch.all(before_proj_weight == 0.0):
            model_type = "z-image-turbo-controlnet-2.0"
        else:
            model_type = "z-image-turbo-controlnet-2.1"

    elif CHECKPOINT_KEY_NAMES["z-image-turbo-controlnet"] in checkpoint:
        model_type = "z-image-turbo-controlnet"

    elif any(key in checkpoint for key in CHECKPOINT_KEY_NAMES["ltx2"]):
        model_type = "ltx2-dev"

    else:
        model_type = "v1"

    return model_type


def fetch_diffusers_config(checkpoint):
    model_type = infer_diffusers_model_type(checkpoint)
    model_path = DIFFUSERS_DEFAULT_PIPELINE_PATHS[model_type]
    model_path = copy.deepcopy(model_path)

    return model_path


def set_image_size(checkpoint, image_size=None):
    if image_size:
        return image_size

    model_type = infer_diffusers_model_type(checkpoint)
    image_size = DIFFUSERS_TO_LDM_DEFAULT_IMAGE_SIZE_MAP[model_type]

    return image_size


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


def convert_open_clip_checkpoint(text_model, checkpoint, prefix="cond_stage_model.model."):
    """Select the OpenCLIP text component and apply its shared layout definition."""
    from .conversion import get_conversion

    config = text_model.config.to_dict()
    config["original_format"] = "openclip"
    model_class = type(text_model).__name__
    state = {key.removeprefix(prefix): tensor for key, tensor in checkpoint.items() if key.startswith(prefix)}
    state.pop("logit_scale", None)
    if model_class != "CLIPTextModelWithProjection":
        state.pop("text_projection", None)
    if prefix == "cond_stage_model.model." and config["num_hidden_layers"] == 23:
        # Stable Diffusion 2 conditions on the penultimate layer of the 24-layer OpenCLIP encoder.
        state = {key: value for key, value in state.items() if not key.startswith("transformer.resblocks.23.")}
    return get_conversion(model_class, config).to_diffusers(state)


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

    # `CLIPTextModel` was flattened in transformers >=5.6; `CLIPTextModelWithProjection` still wraps via `text_model`.
    has_text_model_wrapper = hasattr(model, "text_model")
    text_model = model.text_model if has_text_model_wrapper else model
    position_embedding_dim = text_model.embeddings.position_embedding.weight.shape[-1]

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

    if not has_text_model_wrapper:
        diffusers_format_checkpoint = {
            k.removeprefix("text_model."): v for k, v in diffusers_format_checkpoint.items()
        }

    if is_accelerate_available():
        load_model_dict_into_meta(model, diffusers_format_checkpoint, dtype=torch_dtype)
        empty_device_cache()
    else:
        model.load_state_dict(diffusers_format_checkpoint, strict=False)

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
            "Please pass an instance of a Scheduler object directly to the `scheduler` argument in `from_single_file`\n\n"
            "Example:\n\n"
            "from diffusers import StableDiffusionPipeline, DDIMScheduler\n\n"
            "scheduler = DDIMScheduler()\n"
            "pipe = StableDiffusionPipeline.from_single_file(<checkpoint path>, scheduler=scheduler)\n"
        )
        deprecate("scheduler_type", "1.0.0", deprecation_message)

    if prediction_type is not None:
        deprecation_message = (
            "Please configure an instance of a Scheduler with the appropriate `prediction_type` and "
            "pass the object directly to the `scheduler` argument in `from_single_file`.\n\n"
            "Example:\n\n"
            "from diffusers import StableDiffusionPipeline, DDIMScheduler\n\n"
            'scheduler = DDIMScheduler(prediction_type="v_prediction")\n'
            "pipe = StableDiffusionPipeline.from_single_file(<checkpoint path>, scheduler=scheduler)\n"
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
        load_model_dict_into_meta(model, diffusers_format_checkpoint, dtype=torch_dtype)
        empty_device_cache()
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
