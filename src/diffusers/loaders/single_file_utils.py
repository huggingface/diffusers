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
"""Conversion script for the Stable Diffusion checkpoints."""

import copy
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
from ..utils.constants import DIFFUSERS_REQUEST_TIMEOUT
from ..utils.hub_utils import _get_model_file
from ..utils.torch_utils import empty_device_cache


if is_transformers_available():
    from transformers import AutoImageProcessor

if is_accelerate_available():
    from accelerate import init_empty_weights

    from ..models.model_loading_utils import load_model_dict_into_meta

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

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
        "vae.per_channel_statistics.mean-of-means",
    ],
    "autoencoder-dc": "decoder.stages.1.op_list.0.main.conv.conv.bias",
    "autoencoder-dc-sana": "encoder.project_in.conv.bias",
    "mochi-1-preview": ["model.diffusion_model.blocks.0.attn.qkv_x.weight", "blocks.0.attn.qkv_x.weight"],
    "hunyuan-video": "txt_in.individual_token_refiner.blocks.0.adaLN_modulation.1.bias",
    "instruct-pix2pix": "model.diffusion_model.input_blocks.0.0.weight",
    "lumina2": ["model.diffusion_model.cap_embedder.0.weight", "cap_embedder.0.weight"],
    "sana": [
        "blocks.0.cross_attn.q_linear.weight",
        "blocks.0.cross_attn.q_linear.bias",
        "blocks.0.cross_attn.kv_linear.weight",
        "blocks.0.cross_attn.kv_linear.bias",
    ],
    "wan": ["model.diffusion_model.head.modulation", "head.modulation"],
    "wan_vae": "decoder.middle.0.residual.0.gamma",
    "wan_vace": "vace_blocks.0.after_proj.bias",
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

LDM_VAE_KEYS = ["first_stage_model.", "vae."]
LDM_VAE_DEFAULT_SCALING_FACTOR = 0.18215
PLAYGROUND_VAE_SCALING_FACTOR = 0.5
LDM_UNET_KEY = "model.diffusion_model."
LDM_CONTROLNET_KEY = "control_model."
LDM_CLIP_PREFIX_TO_REMOVE = [
    "cond_stage_model.transformer.",
    "conditioner.embedders.0.transformer.",
]
LDM_OPEN_CLIP_TEXT_PROJECTION_DIM = 1024
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

    elif any(key in checkpoint for key in CHECKPOINT_KEY_NAMES["lumina2"]):
        model_type = "lumina2"

    elif any(key in checkpoint for key in CHECKPOINT_KEY_NAMES["sana"]):
        model_type = "sana"

    elif any(key in checkpoint for key in CHECKPOINT_KEY_NAMES["wan"]):
        if "model.diffusion_model.patch_embedding.weight" in checkpoint:
            target_key = "model.diffusion_model.patch_embedding.weight"
        else:
            target_key = "patch_embedding.weight"

        if CHECKPOINT_KEY_NAMES["wan_vace"] in checkpoint:
            if checkpoint[target_key].shape[0] == 1536:
                model_type = "wan-vace-1.3B"
            elif checkpoint[target_key].shape[0] == 5120:
                model_type = "wan-vace-14B"

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
    # Return checkpoint if it's already been converted
    if "time_embedding.linear_1.weight" in checkpoint:
        return checkpoint
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
    vae_key = ""
    for ldm_vae_key in LDM_VAE_KEYS:
        if any(k.startswith(ldm_vae_key) for k in keys):
            vae_key = ldm_vae_key

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
    elif hasattr(text_model.config, "hidden_size"):
        text_proj_dim = text_model.config.hidden_size
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
def swap_scale_shift(weight, dim):
    shift, scale = weight.chunk(2, dim=0)
    new_weight = torch.cat([scale, shift], dim=0)
    return new_weight


def swap_proj_gate(weight):
    proj, gate = weight.chunk(2, dim=0)
    new_weight = torch.cat([gate, proj], dim=0)
    return new_weight


def get_attn2_layers(state_dict):
    attn2_layers = []
    for key in state_dict.keys():
        if "attn2." in key:
            # Extract the layer number from the key
            layer_num = int(key.split(".")[1])
            attn2_layers.append(layer_num)

    return tuple(sorted(set(attn2_layers)))


def get_caption_projection_dim(state_dict):
    caption_projection_dim = state_dict["context_embedder.weight"].shape[0]
    return caption_projection_dim


def convert_sd3_transformer_checkpoint_to_diffusers(checkpoint, **kwargs):
    converted_state_dict = {}
    keys = list(checkpoint.keys())
    for k in keys:
        if "model.diffusion_model." in k:
            checkpoint[k.replace("model.diffusion_model.", "")] = checkpoint.pop(k)

    num_layers = list(set(int(k.split(".", 2)[1]) for k in checkpoint if "joint_blocks" in k))[-1] + 1  # noqa: C401
    dual_attention_layers = get_attn2_layers(checkpoint)

    caption_projection_dim = get_caption_projection_dim(checkpoint)
    has_qk_norm = any("ln_q" in key for key in checkpoint.keys())

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

        # qk norm
        if has_qk_norm:
            converted_state_dict[f"transformer_blocks.{i}.attn.norm_q.weight"] = checkpoint.pop(
                f"joint_blocks.{i}.x_block.attn.ln_q.weight"
            )
            converted_state_dict[f"transformer_blocks.{i}.attn.norm_k.weight"] = checkpoint.pop(
                f"joint_blocks.{i}.x_block.attn.ln_k.weight"
            )
            converted_state_dict[f"transformer_blocks.{i}.attn.norm_added_q.weight"] = checkpoint.pop(
                f"joint_blocks.{i}.context_block.attn.ln_q.weight"
            )
            converted_state_dict[f"transformer_blocks.{i}.attn.norm_added_k.weight"] = checkpoint.pop(
                f"joint_blocks.{i}.context_block.attn.ln_k.weight"
            )

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

        if i in dual_attention_layers:
            # Q, K, V
            sample_q2, sample_k2, sample_v2 = torch.chunk(
                checkpoint.pop(f"joint_blocks.{i}.x_block.attn2.qkv.weight"), 3, dim=0
            )
            sample_q2_bias, sample_k2_bias, sample_v2_bias = torch.chunk(
                checkpoint.pop(f"joint_blocks.{i}.x_block.attn2.qkv.bias"), 3, dim=0
            )
            converted_state_dict[f"transformer_blocks.{i}.attn2.to_q.weight"] = torch.cat([sample_q2])
            converted_state_dict[f"transformer_blocks.{i}.attn2.to_q.bias"] = torch.cat([sample_q2_bias])
            converted_state_dict[f"transformer_blocks.{i}.attn2.to_k.weight"] = torch.cat([sample_k2])
            converted_state_dict[f"transformer_blocks.{i}.attn2.to_k.bias"] = torch.cat([sample_k2_bias])
            converted_state_dict[f"transformer_blocks.{i}.attn2.to_v.weight"] = torch.cat([sample_v2])
            converted_state_dict[f"transformer_blocks.{i}.attn2.to_v.bias"] = torch.cat([sample_v2_bias])

            # qk norm
            if has_qk_norm:
                converted_state_dict[f"transformer_blocks.{i}.attn2.norm_q.weight"] = checkpoint.pop(
                    f"joint_blocks.{i}.x_block.attn2.ln_q.weight"
                )
                converted_state_dict[f"transformer_blocks.{i}.attn2.norm_k.weight"] = checkpoint.pop(
                    f"joint_blocks.{i}.x_block.attn2.ln_k.weight"
                )

            # output projections.
            converted_state_dict[f"transformer_blocks.{i}.attn2.to_out.0.weight"] = checkpoint.pop(
                f"joint_blocks.{i}.x_block.attn2.proj.weight"
            )
            converted_state_dict[f"transformer_blocks.{i}.attn2.to_out.0.bias"] = checkpoint.pop(
                f"joint_blocks.{i}.x_block.attn2.proj.bias"
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


def convert_flux_transformer_checkpoint_to_diffusers(checkpoint, **kwargs):
    converted_state_dict = {}
    keys = list(checkpoint.keys())

    for k in keys:
        if "model.diffusion_model." in k:
            checkpoint[k.replace("model.diffusion_model.", "")] = checkpoint.pop(k)

    num_layers = list(set(int(k.split(".", 2)[1]) for k in checkpoint if "double_blocks." in k))[-1] + 1  # noqa: C401
    num_single_layers = list(set(int(k.split(".", 2)[1]) for k in checkpoint if "single_blocks." in k))[-1] + 1  # noqa: C401
    mlp_ratio = 4.0
    inner_dim = 3072

    # in SD3 original implementation of AdaLayerNormContinuous, it split linear projection output into shift, scale;
    # while in diffusers it split into scale, shift. Here we swap the linear projection weights in order to be able to use diffusers implementation
    def swap_scale_shift(weight):
        shift, scale = weight.chunk(2, dim=0)
        new_weight = torch.cat([scale, shift], dim=0)
        return new_weight

    ## time_text_embed.timestep_embedder <-  time_in
    converted_state_dict["time_text_embed.timestep_embedder.linear_1.weight"] = checkpoint.pop(
        "time_in.in_layer.weight"
    )
    converted_state_dict["time_text_embed.timestep_embedder.linear_1.bias"] = checkpoint.pop("time_in.in_layer.bias")
    converted_state_dict["time_text_embed.timestep_embedder.linear_2.weight"] = checkpoint.pop(
        "time_in.out_layer.weight"
    )
    converted_state_dict["time_text_embed.timestep_embedder.linear_2.bias"] = checkpoint.pop("time_in.out_layer.bias")

    ## time_text_embed.text_embedder <- vector_in
    converted_state_dict["time_text_embed.text_embedder.linear_1.weight"] = checkpoint.pop("vector_in.in_layer.weight")
    converted_state_dict["time_text_embed.text_embedder.linear_1.bias"] = checkpoint.pop("vector_in.in_layer.bias")
    converted_state_dict["time_text_embed.text_embedder.linear_2.weight"] = checkpoint.pop(
        "vector_in.out_layer.weight"
    )
    converted_state_dict["time_text_embed.text_embedder.linear_2.bias"] = checkpoint.pop("vector_in.out_layer.bias")

    # guidance
    has_guidance = any("guidance" in k for k in checkpoint)
    if has_guidance:
        converted_state_dict["time_text_embed.guidance_embedder.linear_1.weight"] = checkpoint.pop(
            "guidance_in.in_layer.weight"
        )
        converted_state_dict["time_text_embed.guidance_embedder.linear_1.bias"] = checkpoint.pop(
            "guidance_in.in_layer.bias"
        )
        converted_state_dict["time_text_embed.guidance_embedder.linear_2.weight"] = checkpoint.pop(
            "guidance_in.out_layer.weight"
        )
        converted_state_dict["time_text_embed.guidance_embedder.linear_2.bias"] = checkpoint.pop(
            "guidance_in.out_layer.bias"
        )

    # context_embedder
    converted_state_dict["context_embedder.weight"] = checkpoint.pop("txt_in.weight")
    converted_state_dict["context_embedder.bias"] = checkpoint.pop("txt_in.bias")

    # x_embedder
    converted_state_dict["x_embedder.weight"] = checkpoint.pop("img_in.weight")
    converted_state_dict["x_embedder.bias"] = checkpoint.pop("img_in.bias")

    # double transformer blocks
    for i in range(num_layers):
        block_prefix = f"transformer_blocks.{i}."
        # norms.
        ## norm1
        converted_state_dict[f"{block_prefix}norm1.linear.weight"] = checkpoint.pop(
            f"double_blocks.{i}.img_mod.lin.weight"
        )
        converted_state_dict[f"{block_prefix}norm1.linear.bias"] = checkpoint.pop(
            f"double_blocks.{i}.img_mod.lin.bias"
        )
        ## norm1_context
        converted_state_dict[f"{block_prefix}norm1_context.linear.weight"] = checkpoint.pop(
            f"double_blocks.{i}.txt_mod.lin.weight"
        )
        converted_state_dict[f"{block_prefix}norm1_context.linear.bias"] = checkpoint.pop(
            f"double_blocks.{i}.txt_mod.lin.bias"
        )
        # Q, K, V
        sample_q, sample_k, sample_v = torch.chunk(checkpoint.pop(f"double_blocks.{i}.img_attn.qkv.weight"), 3, dim=0)
        context_q, context_k, context_v = torch.chunk(
            checkpoint.pop(f"double_blocks.{i}.txt_attn.qkv.weight"), 3, dim=0
        )
        sample_q_bias, sample_k_bias, sample_v_bias = torch.chunk(
            checkpoint.pop(f"double_blocks.{i}.img_attn.qkv.bias"), 3, dim=0
        )
        context_q_bias, context_k_bias, context_v_bias = torch.chunk(
            checkpoint.pop(f"double_blocks.{i}.txt_attn.qkv.bias"), 3, dim=0
        )
        converted_state_dict[f"{block_prefix}attn.to_q.weight"] = torch.cat([sample_q])
        converted_state_dict[f"{block_prefix}attn.to_q.bias"] = torch.cat([sample_q_bias])
        converted_state_dict[f"{block_prefix}attn.to_k.weight"] = torch.cat([sample_k])
        converted_state_dict[f"{block_prefix}attn.to_k.bias"] = torch.cat([sample_k_bias])
        converted_state_dict[f"{block_prefix}attn.to_v.weight"] = torch.cat([sample_v])
        converted_state_dict[f"{block_prefix}attn.to_v.bias"] = torch.cat([sample_v_bias])
        converted_state_dict[f"{block_prefix}attn.add_q_proj.weight"] = torch.cat([context_q])
        converted_state_dict[f"{block_prefix}attn.add_q_proj.bias"] = torch.cat([context_q_bias])
        converted_state_dict[f"{block_prefix}attn.add_k_proj.weight"] = torch.cat([context_k])
        converted_state_dict[f"{block_prefix}attn.add_k_proj.bias"] = torch.cat([context_k_bias])
        converted_state_dict[f"{block_prefix}attn.add_v_proj.weight"] = torch.cat([context_v])
        converted_state_dict[f"{block_prefix}attn.add_v_proj.bias"] = torch.cat([context_v_bias])
        # qk_norm
        converted_state_dict[f"{block_prefix}attn.norm_q.weight"] = checkpoint.pop(
            f"double_blocks.{i}.img_attn.norm.query_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_k.weight"] = checkpoint.pop(
            f"double_blocks.{i}.img_attn.norm.key_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_added_q.weight"] = checkpoint.pop(
            f"double_blocks.{i}.txt_attn.norm.query_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_added_k.weight"] = checkpoint.pop(
            f"double_blocks.{i}.txt_attn.norm.key_norm.scale"
        )
        # ff img_mlp
        converted_state_dict[f"{block_prefix}ff.net.0.proj.weight"] = checkpoint.pop(
            f"double_blocks.{i}.img_mlp.0.weight"
        )
        converted_state_dict[f"{block_prefix}ff.net.0.proj.bias"] = checkpoint.pop(f"double_blocks.{i}.img_mlp.0.bias")
        converted_state_dict[f"{block_prefix}ff.net.2.weight"] = checkpoint.pop(f"double_blocks.{i}.img_mlp.2.weight")
        converted_state_dict[f"{block_prefix}ff.net.2.bias"] = checkpoint.pop(f"double_blocks.{i}.img_mlp.2.bias")
        converted_state_dict[f"{block_prefix}ff_context.net.0.proj.weight"] = checkpoint.pop(
            f"double_blocks.{i}.txt_mlp.0.weight"
        )
        converted_state_dict[f"{block_prefix}ff_context.net.0.proj.bias"] = checkpoint.pop(
            f"double_blocks.{i}.txt_mlp.0.bias"
        )
        converted_state_dict[f"{block_prefix}ff_context.net.2.weight"] = checkpoint.pop(
            f"double_blocks.{i}.txt_mlp.2.weight"
        )
        converted_state_dict[f"{block_prefix}ff_context.net.2.bias"] = checkpoint.pop(
            f"double_blocks.{i}.txt_mlp.2.bias"
        )
        # output projections.
        converted_state_dict[f"{block_prefix}attn.to_out.0.weight"] = checkpoint.pop(
            f"double_blocks.{i}.img_attn.proj.weight"
        )
        converted_state_dict[f"{block_prefix}attn.to_out.0.bias"] = checkpoint.pop(
            f"double_blocks.{i}.img_attn.proj.bias"
        )
        converted_state_dict[f"{block_prefix}attn.to_add_out.weight"] = checkpoint.pop(
            f"double_blocks.{i}.txt_attn.proj.weight"
        )
        converted_state_dict[f"{block_prefix}attn.to_add_out.bias"] = checkpoint.pop(
            f"double_blocks.{i}.txt_attn.proj.bias"
        )

    # single transformer blocks
    for i in range(num_single_layers):
        block_prefix = f"single_transformer_blocks.{i}."
        # norm.linear  <- single_blocks.0.modulation.lin
        converted_state_dict[f"{block_prefix}norm.linear.weight"] = checkpoint.pop(
            f"single_blocks.{i}.modulation.lin.weight"
        )
        converted_state_dict[f"{block_prefix}norm.linear.bias"] = checkpoint.pop(
            f"single_blocks.{i}.modulation.lin.bias"
        )
        # Q, K, V, mlp
        mlp_hidden_dim = int(inner_dim * mlp_ratio)
        split_size = (inner_dim, inner_dim, inner_dim, mlp_hidden_dim)
        q, k, v, mlp = torch.split(checkpoint.pop(f"single_blocks.{i}.linear1.weight"), split_size, dim=0)
        q_bias, k_bias, v_bias, mlp_bias = torch.split(
            checkpoint.pop(f"single_blocks.{i}.linear1.bias"), split_size, dim=0
        )
        converted_state_dict[f"{block_prefix}attn.to_q.weight"] = torch.cat([q])
        converted_state_dict[f"{block_prefix}attn.to_q.bias"] = torch.cat([q_bias])
        converted_state_dict[f"{block_prefix}attn.to_k.weight"] = torch.cat([k])
        converted_state_dict[f"{block_prefix}attn.to_k.bias"] = torch.cat([k_bias])
        converted_state_dict[f"{block_prefix}attn.to_v.weight"] = torch.cat([v])
        converted_state_dict[f"{block_prefix}attn.to_v.bias"] = torch.cat([v_bias])
        converted_state_dict[f"{block_prefix}proj_mlp.weight"] = torch.cat([mlp])
        converted_state_dict[f"{block_prefix}proj_mlp.bias"] = torch.cat([mlp_bias])
        # qk norm
        converted_state_dict[f"{block_prefix}attn.norm_q.weight"] = checkpoint.pop(
            f"single_blocks.{i}.norm.query_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_k.weight"] = checkpoint.pop(
            f"single_blocks.{i}.norm.key_norm.scale"
        )
        # output projections.
        converted_state_dict[f"{block_prefix}proj_out.weight"] = checkpoint.pop(f"single_blocks.{i}.linear2.weight")
        converted_state_dict[f"{block_prefix}proj_out.bias"] = checkpoint.pop(f"single_blocks.{i}.linear2.bias")

    converted_state_dict["proj_out.weight"] = checkpoint.pop("final_layer.linear.weight")
    converted_state_dict["proj_out.bias"] = checkpoint.pop("final_layer.linear.bias")
    converted_state_dict["norm_out.linear.weight"] = swap_scale_shift(
        checkpoint.pop("final_layer.adaLN_modulation.1.weight")
    )
    converted_state_dict["norm_out.linear.bias"] = swap_scale_shift(
        checkpoint.pop("final_layer.adaLN_modulation.1.bias")
    )

    return converted_state_dict


def convert_ltx_transformer_checkpoint_to_diffusers(checkpoint, **kwargs):
    converted_state_dict = {key: checkpoint.pop(key) for key in list(checkpoint.keys()) if "vae" not in key}

    TRANSFORMER_KEYS_RENAME_DICT = {
        "model.diffusion_model.": "",
        "patchify_proj": "proj_in",
        "adaln_single": "time_embed",
        "q_norm": "norm_q",
        "k_norm": "norm_k",
    }

    TRANSFORMER_SPECIAL_KEYS_REMAP = {}

    for key in list(converted_state_dict.keys()):
        new_key = key
        for replace_key, rename_key in TRANSFORMER_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)
        converted_state_dict[new_key] = converted_state_dict.pop(key)

    for key in list(converted_state_dict.keys()):
        for special_key, handler_fn_inplace in TRANSFORMER_SPECIAL_KEYS_REMAP.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, converted_state_dict)

    return converted_state_dict


def convert_ltx_vae_checkpoint_to_diffusers(checkpoint, **kwargs):
    converted_state_dict = {key: checkpoint.pop(key) for key in list(checkpoint.keys()) if "vae." in key}

    def remove_keys_(key: str, state_dict):
        state_dict.pop(key)

    VAE_KEYS_RENAME_DICT = {
        # common
        "vae.": "",
        # decoder
        "up_blocks.0": "mid_block",
        "up_blocks.1": "up_blocks.0",
        "up_blocks.2": "up_blocks.1.upsamplers.0",
        "up_blocks.3": "up_blocks.1",
        "up_blocks.4": "up_blocks.2.conv_in",
        "up_blocks.5": "up_blocks.2.upsamplers.0",
        "up_blocks.6": "up_blocks.2",
        "up_blocks.7": "up_blocks.3.conv_in",
        "up_blocks.8": "up_blocks.3.upsamplers.0",
        "up_blocks.9": "up_blocks.3",
        # encoder
        "down_blocks.0": "down_blocks.0",
        "down_blocks.1": "down_blocks.0.downsamplers.0",
        "down_blocks.2": "down_blocks.0.conv_out",
        "down_blocks.3": "down_blocks.1",
        "down_blocks.4": "down_blocks.1.downsamplers.0",
        "down_blocks.5": "down_blocks.1.conv_out",
        "down_blocks.6": "down_blocks.2",
        "down_blocks.7": "down_blocks.2.downsamplers.0",
        "down_blocks.8": "down_blocks.3",
        "down_blocks.9": "mid_block",
        # common
        "conv_shortcut": "conv_shortcut.conv",
        "res_blocks": "resnets",
        "norm3.norm": "norm3",
        "per_channel_statistics.mean-of-means": "latents_mean",
        "per_channel_statistics.std-of-means": "latents_std",
    }

    VAE_091_RENAME_DICT = {
        # decoder
        "up_blocks.0": "mid_block",
        "up_blocks.1": "up_blocks.0.upsamplers.0",
        "up_blocks.2": "up_blocks.0",
        "up_blocks.3": "up_blocks.1.upsamplers.0",
        "up_blocks.4": "up_blocks.1",
        "up_blocks.5": "up_blocks.2.upsamplers.0",
        "up_blocks.6": "up_blocks.2",
        "up_blocks.7": "up_blocks.3.upsamplers.0",
        "up_blocks.8": "up_blocks.3",
        # common
        "last_time_embedder": "time_embedder",
        "last_scale_shift_table": "scale_shift_table",
    }

    VAE_095_RENAME_DICT = {
        # decoder
        "up_blocks.0": "mid_block",
        "up_blocks.1": "up_blocks.0.upsamplers.0",
        "up_blocks.2": "up_blocks.0",
        "up_blocks.3": "up_blocks.1.upsamplers.0",
        "up_blocks.4": "up_blocks.1",
        "up_blocks.5": "up_blocks.2.upsamplers.0",
        "up_blocks.6": "up_blocks.2",
        "up_blocks.7": "up_blocks.3.upsamplers.0",
        "up_blocks.8": "up_blocks.3",
        # encoder
        "down_blocks.0": "down_blocks.0",
        "down_blocks.1": "down_blocks.0.downsamplers.0",
        "down_blocks.2": "down_blocks.1",
        "down_blocks.3": "down_blocks.1.downsamplers.0",
        "down_blocks.4": "down_blocks.2",
        "down_blocks.5": "down_blocks.2.downsamplers.0",
        "down_blocks.6": "down_blocks.3",
        "down_blocks.7": "down_blocks.3.downsamplers.0",
        "down_blocks.8": "mid_block",
        # common
        "last_time_embedder": "time_embedder",
        "last_scale_shift_table": "scale_shift_table",
    }

    VAE_SPECIAL_KEYS_REMAP = {
        "per_channel_statistics.channel": remove_keys_,
        "per_channel_statistics.mean-of-means": remove_keys_,
        "per_channel_statistics.mean-of-stds": remove_keys_,
    }

    if converted_state_dict["vae.encoder.conv_out.conv.weight"].shape[1] == 2048:
        VAE_KEYS_RENAME_DICT.update(VAE_095_RENAME_DICT)
    elif "vae.decoder.last_time_embedder.timestep_embedder.linear_1.weight" in converted_state_dict:
        VAE_KEYS_RENAME_DICT.update(VAE_091_RENAME_DICT)

    for key in list(converted_state_dict.keys()):
        new_key = key
        for replace_key, rename_key in VAE_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)
        converted_state_dict[new_key] = converted_state_dict.pop(key)

    for key in list(converted_state_dict.keys()):
        for special_key, handler_fn_inplace in VAE_SPECIAL_KEYS_REMAP.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, converted_state_dict)

    return converted_state_dict


def convert_autoencoder_dc_checkpoint_to_diffusers(checkpoint, **kwargs):
    converted_state_dict = {key: checkpoint.pop(key) for key in list(checkpoint.keys())}

    def remap_qkv_(key: str, state_dict):
        qkv = state_dict.pop(key)
        q, k, v = torch.chunk(qkv, 3, dim=0)
        parent_module, _, _ = key.rpartition(".qkv.conv.weight")
        state_dict[f"{parent_module}.to_q.weight"] = q.squeeze()
        state_dict[f"{parent_module}.to_k.weight"] = k.squeeze()
        state_dict[f"{parent_module}.to_v.weight"] = v.squeeze()

    def remap_proj_conv_(key: str, state_dict):
        parent_module, _, _ = key.rpartition(".proj.conv.weight")
        state_dict[f"{parent_module}.to_out.weight"] = state_dict.pop(key).squeeze()

    AE_KEYS_RENAME_DICT = {
        # common
        "main.": "",
        "op_list.": "",
        "context_module": "attn",
        "local_module": "conv_out",
        # NOTE: The below two lines work because scales in the available configs only have a tuple length of 1
        # If there were more scales, there would be more layers, so a loop would be better to handle this
        "aggreg.0.0": "to_qkv_multiscale.0.proj_in",
        "aggreg.0.1": "to_qkv_multiscale.0.proj_out",
        "depth_conv.conv": "conv_depth",
        "inverted_conv.conv": "conv_inverted",
        "point_conv.conv": "conv_point",
        "point_conv.norm": "norm",
        "conv.conv.": "conv.",
        "conv1.conv": "conv1",
        "conv2.conv": "conv2",
        "conv2.norm": "norm",
        "proj.norm": "norm_out",
        # encoder
        "encoder.project_in.conv": "encoder.conv_in",
        "encoder.project_out.0.conv": "encoder.conv_out",
        "encoder.stages": "encoder.down_blocks",
        # decoder
        "decoder.project_in.conv": "decoder.conv_in",
        "decoder.project_out.0": "decoder.norm_out",
        "decoder.project_out.2.conv": "decoder.conv_out",
        "decoder.stages": "decoder.up_blocks",
    }

    AE_F32C32_F64C128_F128C512_KEYS = {
        "encoder.project_in.conv": "encoder.conv_in.conv",
        "decoder.project_out.2.conv": "decoder.conv_out.conv",
    }

    AE_SPECIAL_KEYS_REMAP = {
        "qkv.conv.weight": remap_qkv_,
        "proj.conv.weight": remap_proj_conv_,
    }
    if "encoder.project_in.conv.bias" not in converted_state_dict:
        AE_KEYS_RENAME_DICT.update(AE_F32C32_F64C128_F128C512_KEYS)

    for key in list(converted_state_dict.keys()):
        new_key = key[:]
        for replace_key, rename_key in AE_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)
        converted_state_dict[new_key] = converted_state_dict.pop(key)

    for key in list(converted_state_dict.keys()):
        for special_key, handler_fn_inplace in AE_SPECIAL_KEYS_REMAP.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, converted_state_dict)

    return converted_state_dict


def convert_mochi_transformer_checkpoint_to_diffusers(checkpoint, **kwargs):
    converted_state_dict = {}

    # Comfy checkpoints add this prefix
    keys = list(checkpoint.keys())
    for k in keys:
        if "model.diffusion_model." in k:
            checkpoint[k.replace("model.diffusion_model.", "")] = checkpoint.pop(k)

    # Convert patch_embed
    converted_state_dict["patch_embed.proj.weight"] = checkpoint.pop("x_embedder.proj.weight")
    converted_state_dict["patch_embed.proj.bias"] = checkpoint.pop("x_embedder.proj.bias")

    # Convert time_embed
    converted_state_dict["time_embed.timestep_embedder.linear_1.weight"] = checkpoint.pop("t_embedder.mlp.0.weight")
    converted_state_dict["time_embed.timestep_embedder.linear_1.bias"] = checkpoint.pop("t_embedder.mlp.0.bias")
    converted_state_dict["time_embed.timestep_embedder.linear_2.weight"] = checkpoint.pop("t_embedder.mlp.2.weight")
    converted_state_dict["time_embed.timestep_embedder.linear_2.bias"] = checkpoint.pop("t_embedder.mlp.2.bias")
    converted_state_dict["time_embed.pooler.to_kv.weight"] = checkpoint.pop("t5_y_embedder.to_kv.weight")
    converted_state_dict["time_embed.pooler.to_kv.bias"] = checkpoint.pop("t5_y_embedder.to_kv.bias")
    converted_state_dict["time_embed.pooler.to_q.weight"] = checkpoint.pop("t5_y_embedder.to_q.weight")
    converted_state_dict["time_embed.pooler.to_q.bias"] = checkpoint.pop("t5_y_embedder.to_q.bias")
    converted_state_dict["time_embed.pooler.to_out.weight"] = checkpoint.pop("t5_y_embedder.to_out.weight")
    converted_state_dict["time_embed.pooler.to_out.bias"] = checkpoint.pop("t5_y_embedder.to_out.bias")
    converted_state_dict["time_embed.caption_proj.weight"] = checkpoint.pop("t5_yproj.weight")
    converted_state_dict["time_embed.caption_proj.bias"] = checkpoint.pop("t5_yproj.bias")

    # Convert transformer blocks
    num_layers = 48
    for i in range(num_layers):
        block_prefix = f"transformer_blocks.{i}."
        old_prefix = f"blocks.{i}."

        # norm1
        converted_state_dict[block_prefix + "norm1.linear.weight"] = checkpoint.pop(old_prefix + "mod_x.weight")
        converted_state_dict[block_prefix + "norm1.linear.bias"] = checkpoint.pop(old_prefix + "mod_x.bias")
        if i < num_layers - 1:
            converted_state_dict[block_prefix + "norm1_context.linear.weight"] = checkpoint.pop(
                old_prefix + "mod_y.weight"
            )
            converted_state_dict[block_prefix + "norm1_context.linear.bias"] = checkpoint.pop(
                old_prefix + "mod_y.bias"
            )
        else:
            converted_state_dict[block_prefix + "norm1_context.linear_1.weight"] = checkpoint.pop(
                old_prefix + "mod_y.weight"
            )
            converted_state_dict[block_prefix + "norm1_context.linear_1.bias"] = checkpoint.pop(
                old_prefix + "mod_y.bias"
            )

        # Visual attention
        qkv_weight = checkpoint.pop(old_prefix + "attn.qkv_x.weight")
        q, k, v = qkv_weight.chunk(3, dim=0)

        converted_state_dict[block_prefix + "attn1.to_q.weight"] = q
        converted_state_dict[block_prefix + "attn1.to_k.weight"] = k
        converted_state_dict[block_prefix + "attn1.to_v.weight"] = v
        converted_state_dict[block_prefix + "attn1.norm_q.weight"] = checkpoint.pop(
            old_prefix + "attn.q_norm_x.weight"
        )
        converted_state_dict[block_prefix + "attn1.norm_k.weight"] = checkpoint.pop(
            old_prefix + "attn.k_norm_x.weight"
        )
        converted_state_dict[block_prefix + "attn1.to_out.0.weight"] = checkpoint.pop(
            old_prefix + "attn.proj_x.weight"
        )
        converted_state_dict[block_prefix + "attn1.to_out.0.bias"] = checkpoint.pop(old_prefix + "attn.proj_x.bias")

        # Context attention
        qkv_weight = checkpoint.pop(old_prefix + "attn.qkv_y.weight")
        q, k, v = qkv_weight.chunk(3, dim=0)

        converted_state_dict[block_prefix + "attn1.add_q_proj.weight"] = q
        converted_state_dict[block_prefix + "attn1.add_k_proj.weight"] = k
        converted_state_dict[block_prefix + "attn1.add_v_proj.weight"] = v
        converted_state_dict[block_prefix + "attn1.norm_added_q.weight"] = checkpoint.pop(
            old_prefix + "attn.q_norm_y.weight"
        )
        converted_state_dict[block_prefix + "attn1.norm_added_k.weight"] = checkpoint.pop(
            old_prefix + "attn.k_norm_y.weight"
        )
        if i < num_layers - 1:
            converted_state_dict[block_prefix + "attn1.to_add_out.weight"] = checkpoint.pop(
                old_prefix + "attn.proj_y.weight"
            )
            converted_state_dict[block_prefix + "attn1.to_add_out.bias"] = checkpoint.pop(
                old_prefix + "attn.proj_y.bias"
            )

        # MLP
        converted_state_dict[block_prefix + "ff.net.0.proj.weight"] = swap_proj_gate(
            checkpoint.pop(old_prefix + "mlp_x.w1.weight")
        )
        converted_state_dict[block_prefix + "ff.net.2.weight"] = checkpoint.pop(old_prefix + "mlp_x.w2.weight")
        if i < num_layers - 1:
            converted_state_dict[block_prefix + "ff_context.net.0.proj.weight"] = swap_proj_gate(
                checkpoint.pop(old_prefix + "mlp_y.w1.weight")
            )
            converted_state_dict[block_prefix + "ff_context.net.2.weight"] = checkpoint.pop(
                old_prefix + "mlp_y.w2.weight"
            )

    # Output layers
    converted_state_dict["norm_out.linear.weight"] = swap_scale_shift(checkpoint.pop("final_layer.mod.weight"), dim=0)
    converted_state_dict["norm_out.linear.bias"] = swap_scale_shift(checkpoint.pop("final_layer.mod.bias"), dim=0)
    converted_state_dict["proj_out.weight"] = checkpoint.pop("final_layer.linear.weight")
    converted_state_dict["proj_out.bias"] = checkpoint.pop("final_layer.linear.bias")

    converted_state_dict["pos_frequencies"] = checkpoint.pop("pos_frequencies")

    return converted_state_dict


def convert_hunyuan_video_transformer_to_diffusers(checkpoint, **kwargs):
    def remap_norm_scale_shift_(key, state_dict):
        weight = state_dict.pop(key)
        shift, scale = weight.chunk(2, dim=0)
        new_weight = torch.cat([scale, shift], dim=0)
        state_dict[key.replace("final_layer.adaLN_modulation.1", "norm_out.linear")] = new_weight

    def remap_txt_in_(key, state_dict):
        def rename_key(key):
            new_key = key.replace("individual_token_refiner.blocks", "token_refiner.refiner_blocks")
            new_key = new_key.replace("adaLN_modulation.1", "norm_out.linear")
            new_key = new_key.replace("txt_in", "context_embedder")
            new_key = new_key.replace("t_embedder.mlp.0", "time_text_embed.timestep_embedder.linear_1")
            new_key = new_key.replace("t_embedder.mlp.2", "time_text_embed.timestep_embedder.linear_2")
            new_key = new_key.replace("c_embedder", "time_text_embed.text_embedder")
            new_key = new_key.replace("mlp", "ff")
            return new_key

        if "self_attn_qkv" in key:
            weight = state_dict.pop(key)
            to_q, to_k, to_v = weight.chunk(3, dim=0)
            state_dict[rename_key(key.replace("self_attn_qkv", "attn.to_q"))] = to_q
            state_dict[rename_key(key.replace("self_attn_qkv", "attn.to_k"))] = to_k
            state_dict[rename_key(key.replace("self_attn_qkv", "attn.to_v"))] = to_v
        else:
            state_dict[rename_key(key)] = state_dict.pop(key)

    def remap_img_attn_qkv_(key, state_dict):
        weight = state_dict.pop(key)
        to_q, to_k, to_v = weight.chunk(3, dim=0)
        state_dict[key.replace("img_attn_qkv", "attn.to_q")] = to_q
        state_dict[key.replace("img_attn_qkv", "attn.to_k")] = to_k
        state_dict[key.replace("img_attn_qkv", "attn.to_v")] = to_v

    def remap_txt_attn_qkv_(key, state_dict):
        weight = state_dict.pop(key)
        to_q, to_k, to_v = weight.chunk(3, dim=0)
        state_dict[key.replace("txt_attn_qkv", "attn.add_q_proj")] = to_q
        state_dict[key.replace("txt_attn_qkv", "attn.add_k_proj")] = to_k
        state_dict[key.replace("txt_attn_qkv", "attn.add_v_proj")] = to_v

    def remap_single_transformer_blocks_(key, state_dict):
        hidden_size = 3072

        if "linear1.weight" in key:
            linear1_weight = state_dict.pop(key)
            split_size = (hidden_size, hidden_size, hidden_size, linear1_weight.size(0) - 3 * hidden_size)
            q, k, v, mlp = torch.split(linear1_weight, split_size, dim=0)
            new_key = key.replace("single_blocks", "single_transformer_blocks").removesuffix(".linear1.weight")
            state_dict[f"{new_key}.attn.to_q.weight"] = q
            state_dict[f"{new_key}.attn.to_k.weight"] = k
            state_dict[f"{new_key}.attn.to_v.weight"] = v
            state_dict[f"{new_key}.proj_mlp.weight"] = mlp

        elif "linear1.bias" in key:
            linear1_bias = state_dict.pop(key)
            split_size = (hidden_size, hidden_size, hidden_size, linear1_bias.size(0) - 3 * hidden_size)
            q_bias, k_bias, v_bias, mlp_bias = torch.split(linear1_bias, split_size, dim=0)
            new_key = key.replace("single_blocks", "single_transformer_blocks").removesuffix(".linear1.bias")
            state_dict[f"{new_key}.attn.to_q.bias"] = q_bias
            state_dict[f"{new_key}.attn.to_k.bias"] = k_bias
            state_dict[f"{new_key}.attn.to_v.bias"] = v_bias
            state_dict[f"{new_key}.proj_mlp.bias"] = mlp_bias

        else:
            new_key = key.replace("single_blocks", "single_transformer_blocks")
            new_key = new_key.replace("linear2", "proj_out")
            new_key = new_key.replace("q_norm", "attn.norm_q")
            new_key = new_key.replace("k_norm", "attn.norm_k")
            state_dict[new_key] = state_dict.pop(key)

    TRANSFORMER_KEYS_RENAME_DICT = {
        "img_in": "x_embedder",
        "time_in.mlp.0": "time_text_embed.timestep_embedder.linear_1",
        "time_in.mlp.2": "time_text_embed.timestep_embedder.linear_2",
        "guidance_in.mlp.0": "time_text_embed.guidance_embedder.linear_1",
        "guidance_in.mlp.2": "time_text_embed.guidance_embedder.linear_2",
        "vector_in.in_layer": "time_text_embed.text_embedder.linear_1",
        "vector_in.out_layer": "time_text_embed.text_embedder.linear_2",
        "double_blocks": "transformer_blocks",
        "img_attn_q_norm": "attn.norm_q",
        "img_attn_k_norm": "attn.norm_k",
        "img_attn_proj": "attn.to_out.0",
        "txt_attn_q_norm": "attn.norm_added_q",
        "txt_attn_k_norm": "attn.norm_added_k",
        "txt_attn_proj": "attn.to_add_out",
        "img_mod.linear": "norm1.linear",
        "img_norm1": "norm1.norm",
        "img_norm2": "norm2",
        "img_mlp": "ff",
        "txt_mod.linear": "norm1_context.linear",
        "txt_norm1": "norm1.norm",
        "txt_norm2": "norm2_context",
        "txt_mlp": "ff_context",
        "self_attn_proj": "attn.to_out.0",
        "modulation.linear": "norm.linear",
        "pre_norm": "norm.norm",
        "final_layer.norm_final": "norm_out.norm",
        "final_layer.linear": "proj_out",
        "fc1": "net.0.proj",
        "fc2": "net.2",
        "input_embedder": "proj_in",
    }

    TRANSFORMER_SPECIAL_KEYS_REMAP = {
        "txt_in": remap_txt_in_,
        "img_attn_qkv": remap_img_attn_qkv_,
        "txt_attn_qkv": remap_txt_attn_qkv_,
        "single_blocks": remap_single_transformer_blocks_,
        "final_layer.adaLN_modulation.1": remap_norm_scale_shift_,
    }

    def update_state_dict_(state_dict, old_key, new_key):
        state_dict[new_key] = state_dict.pop(old_key)

    for key in list(checkpoint.keys()):
        new_key = key[:]
        for replace_key, rename_key in TRANSFORMER_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_state_dict_(checkpoint, key, new_key)

    for key in list(checkpoint.keys()):
        for special_key, handler_fn_inplace in TRANSFORMER_SPECIAL_KEYS_REMAP.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, checkpoint)

    return checkpoint


def convert_auraflow_transformer_checkpoint_to_diffusers(checkpoint, **kwargs):
    converted_state_dict = {}
    state_dict_keys = list(checkpoint.keys())

    # Handle register tokens and positional embeddings
    converted_state_dict["register_tokens"] = checkpoint.pop("register_tokens", None)

    # Handle time step projection
    converted_state_dict["time_step_proj.linear_1.weight"] = checkpoint.pop("t_embedder.mlp.0.weight", None)
    converted_state_dict["time_step_proj.linear_1.bias"] = checkpoint.pop("t_embedder.mlp.0.bias", None)
    converted_state_dict["time_step_proj.linear_2.weight"] = checkpoint.pop("t_embedder.mlp.2.weight", None)
    converted_state_dict["time_step_proj.linear_2.bias"] = checkpoint.pop("t_embedder.mlp.2.bias", None)

    # Handle context embedder
    converted_state_dict["context_embedder.weight"] = checkpoint.pop("cond_seq_linear.weight", None)

    # Calculate the number of layers
    def calculate_layers(keys, key_prefix):
        layers = set()
        for k in keys:
            if key_prefix in k:
                layer_num = int(k.split(".")[1])  # get the layer number
                layers.add(layer_num)
        return len(layers)

    mmdit_layers = calculate_layers(state_dict_keys, key_prefix="double_layers")
    single_dit_layers = calculate_layers(state_dict_keys, key_prefix="single_layers")

    # MMDiT blocks
    for i in range(mmdit_layers):
        # Feed-forward
        path_mapping = {"mlpX": "ff", "mlpC": "ff_context"}
        weight_mapping = {"c_fc1": "linear_1", "c_fc2": "linear_2", "c_proj": "out_projection"}
        for orig_k, diffuser_k in path_mapping.items():
            for k, v in weight_mapping.items():
                converted_state_dict[f"joint_transformer_blocks.{i}.{diffuser_k}.{v}.weight"] = checkpoint.pop(
                    f"double_layers.{i}.{orig_k}.{k}.weight", None
                )

        # Norms
        path_mapping = {"modX": "norm1", "modC": "norm1_context"}
        for orig_k, diffuser_k in path_mapping.items():
            converted_state_dict[f"joint_transformer_blocks.{i}.{diffuser_k}.linear.weight"] = checkpoint.pop(
                f"double_layers.{i}.{orig_k}.1.weight", None
            )

        # Attentions
        x_attn_mapping = {"w2q": "to_q", "w2k": "to_k", "w2v": "to_v", "w2o": "to_out.0"}
        context_attn_mapping = {"w1q": "add_q_proj", "w1k": "add_k_proj", "w1v": "add_v_proj", "w1o": "to_add_out"}
        for attn_mapping in [x_attn_mapping, context_attn_mapping]:
            for k, v in attn_mapping.items():
                converted_state_dict[f"joint_transformer_blocks.{i}.attn.{v}.weight"] = checkpoint.pop(
                    f"double_layers.{i}.attn.{k}.weight", None
                )

    # Single-DiT blocks
    for i in range(single_dit_layers):
        # Feed-forward
        mapping = {"c_fc1": "linear_1", "c_fc2": "linear_2", "c_proj": "out_projection"}
        for k, v in mapping.items():
            converted_state_dict[f"single_transformer_blocks.{i}.ff.{v}.weight"] = checkpoint.pop(
                f"single_layers.{i}.mlp.{k}.weight", None
            )

        # Norms
        converted_state_dict[f"single_transformer_blocks.{i}.norm1.linear.weight"] = checkpoint.pop(
            f"single_layers.{i}.modCX.1.weight", None
        )

        # Attentions
        x_attn_mapping = {"w1q": "to_q", "w1k": "to_k", "w1v": "to_v", "w1o": "to_out.0"}
        for k, v in x_attn_mapping.items():
            converted_state_dict[f"single_transformer_blocks.{i}.attn.{v}.weight"] = checkpoint.pop(
                f"single_layers.{i}.attn.{k}.weight", None
            )
    # Final blocks
    converted_state_dict["proj_out.weight"] = checkpoint.pop("final_linear.weight", None)

    # Handle the final norm layer
    norm_weight = checkpoint.pop("modF.1.weight", None)
    if norm_weight is not None:
        converted_state_dict["norm_out.linear.weight"] = swap_scale_shift(norm_weight, dim=None)
    else:
        converted_state_dict["norm_out.linear.weight"] = None

    converted_state_dict["pos_embed.pos_embed"] = checkpoint.pop("positional_encoding")
    converted_state_dict["pos_embed.proj.weight"] = checkpoint.pop("init_x_linear.weight")
    converted_state_dict["pos_embed.proj.bias"] = checkpoint.pop("init_x_linear.bias")

    return converted_state_dict


def convert_lumina2_to_diffusers(checkpoint, **kwargs):
    converted_state_dict = {}

    # Original Lumina-Image-2 has an extra norm parameter that is unused
    # We just remove it here
    checkpoint.pop("norm_final.weight", None)

    # Comfy checkpoints add this prefix
    keys = list(checkpoint.keys())
    for k in keys:
        if "model.diffusion_model." in k:
            checkpoint[k.replace("model.diffusion_model.", "")] = checkpoint.pop(k)

    LUMINA_KEY_MAP = {
        "cap_embedder": "time_caption_embed.caption_embedder",
        "t_embedder.mlp.0": "time_caption_embed.timestep_embedder.linear_1",
        "t_embedder.mlp.2": "time_caption_embed.timestep_embedder.linear_2",
        "attention": "attn",
        ".out.": ".to_out.0.",
        "k_norm": "norm_k",
        "q_norm": "norm_q",
        "w1": "linear_1",
        "w2": "linear_2",
        "w3": "linear_3",
        "adaLN_modulation.1": "norm1.linear",
    }
    ATTENTION_NORM_MAP = {
        "attention_norm1": "norm1.norm",
        "attention_norm2": "norm2",
    }
    CONTEXT_REFINER_MAP = {
        "context_refiner.0.attention_norm1": "context_refiner.0.norm1",
        "context_refiner.0.attention_norm2": "context_refiner.0.norm2",
        "context_refiner.1.attention_norm1": "context_refiner.1.norm1",
        "context_refiner.1.attention_norm2": "context_refiner.1.norm2",
    }
    FINAL_LAYER_MAP = {
        "final_layer.adaLN_modulation.1": "norm_out.linear_1",
        "final_layer.linear": "norm_out.linear_2",
    }

    def convert_lumina_attn_to_diffusers(tensor, diffusers_key):
        q_dim = 2304
        k_dim = v_dim = 768

        to_q, to_k, to_v = torch.split(tensor, [q_dim, k_dim, v_dim], dim=0)

        return {
            diffusers_key.replace("qkv", "to_q"): to_q,
            diffusers_key.replace("qkv", "to_k"): to_k,
            diffusers_key.replace("qkv", "to_v"): to_v,
        }

    for key in keys:
        diffusers_key = key
        for k, v in CONTEXT_REFINER_MAP.items():
            diffusers_key = diffusers_key.replace(k, v)
        for k, v in FINAL_LAYER_MAP.items():
            diffusers_key = diffusers_key.replace(k, v)
        for k, v in ATTENTION_NORM_MAP.items():
            diffusers_key = diffusers_key.replace(k, v)
        for k, v in LUMINA_KEY_MAP.items():
            diffusers_key = diffusers_key.replace(k, v)

        if "qkv" in diffusers_key:
            converted_state_dict.update(convert_lumina_attn_to_diffusers(checkpoint.pop(key), diffusers_key))
        else:
            converted_state_dict[diffusers_key] = checkpoint.pop(key)

    return converted_state_dict


def convert_sana_transformer_to_diffusers(checkpoint, **kwargs):
    converted_state_dict = {}
    keys = list(checkpoint.keys())
    for k in keys:
        if "model.diffusion_model." in k:
            checkpoint[k.replace("model.diffusion_model.", "")] = checkpoint.pop(k)

    num_layers = list(set(int(k.split(".", 2)[1]) for k in checkpoint if "blocks" in k))[-1] + 1  # noqa: C401

    # Positional and patch embeddings.
    checkpoint.pop("pos_embed")
    converted_state_dict["patch_embed.proj.weight"] = checkpoint.pop("x_embedder.proj.weight")
    converted_state_dict["patch_embed.proj.bias"] = checkpoint.pop("x_embedder.proj.bias")

    # Timestep embeddings.
    converted_state_dict["time_embed.emb.timestep_embedder.linear_1.weight"] = checkpoint.pop(
        "t_embedder.mlp.0.weight"
    )
    converted_state_dict["time_embed.emb.timestep_embedder.linear_1.bias"] = checkpoint.pop("t_embedder.mlp.0.bias")
    converted_state_dict["time_embed.emb.timestep_embedder.linear_2.weight"] = checkpoint.pop(
        "t_embedder.mlp.2.weight"
    )
    converted_state_dict["time_embed.emb.timestep_embedder.linear_2.bias"] = checkpoint.pop("t_embedder.mlp.2.bias")
    converted_state_dict["time_embed.linear.weight"] = checkpoint.pop("t_block.1.weight")
    converted_state_dict["time_embed.linear.bias"] = checkpoint.pop("t_block.1.bias")

    # Caption Projection.
    checkpoint.pop("y_embedder.y_embedding")
    converted_state_dict["caption_projection.linear_1.weight"] = checkpoint.pop("y_embedder.y_proj.fc1.weight")
    converted_state_dict["caption_projection.linear_1.bias"] = checkpoint.pop("y_embedder.y_proj.fc1.bias")
    converted_state_dict["caption_projection.linear_2.weight"] = checkpoint.pop("y_embedder.y_proj.fc2.weight")
    converted_state_dict["caption_projection.linear_2.bias"] = checkpoint.pop("y_embedder.y_proj.fc2.bias")
    converted_state_dict["caption_norm.weight"] = checkpoint.pop("attention_y_norm.weight")

    for i in range(num_layers):
        converted_state_dict[f"transformer_blocks.{i}.scale_shift_table"] = checkpoint.pop(
            f"blocks.{i}.scale_shift_table"
        )

        # Self-Attention
        sample_q, sample_k, sample_v = torch.chunk(checkpoint.pop(f"blocks.{i}.attn.qkv.weight"), 3, dim=0)
        converted_state_dict[f"transformer_blocks.{i}.attn1.to_q.weight"] = torch.cat([sample_q])
        converted_state_dict[f"transformer_blocks.{i}.attn1.to_k.weight"] = torch.cat([sample_k])
        converted_state_dict[f"transformer_blocks.{i}.attn1.to_v.weight"] = torch.cat([sample_v])

        # Output Projections
        converted_state_dict[f"transformer_blocks.{i}.attn1.to_out.0.weight"] = checkpoint.pop(
            f"blocks.{i}.attn.proj.weight"
        )
        converted_state_dict[f"transformer_blocks.{i}.attn1.to_out.0.bias"] = checkpoint.pop(
            f"blocks.{i}.attn.proj.bias"
        )

        # Cross-Attention
        converted_state_dict[f"transformer_blocks.{i}.attn2.to_q.weight"] = checkpoint.pop(
            f"blocks.{i}.cross_attn.q_linear.weight"
        )
        converted_state_dict[f"transformer_blocks.{i}.attn2.to_q.bias"] = checkpoint.pop(
            f"blocks.{i}.cross_attn.q_linear.bias"
        )

        linear_sample_k, linear_sample_v = torch.chunk(
            checkpoint.pop(f"blocks.{i}.cross_attn.kv_linear.weight"), 2, dim=0
        )
        linear_sample_k_bias, linear_sample_v_bias = torch.chunk(
            checkpoint.pop(f"blocks.{i}.cross_attn.kv_linear.bias"), 2, dim=0
        )
        converted_state_dict[f"transformer_blocks.{i}.attn2.to_k.weight"] = linear_sample_k
        converted_state_dict[f"transformer_blocks.{i}.attn2.to_v.weight"] = linear_sample_v
        converted_state_dict[f"transformer_blocks.{i}.attn2.to_k.bias"] = linear_sample_k_bias
        converted_state_dict[f"transformer_blocks.{i}.attn2.to_v.bias"] = linear_sample_v_bias

        # Output Projections
        converted_state_dict[f"transformer_blocks.{i}.attn2.to_out.0.weight"] = checkpoint.pop(
            f"blocks.{i}.cross_attn.proj.weight"
        )
        converted_state_dict[f"transformer_blocks.{i}.attn2.to_out.0.bias"] = checkpoint.pop(
            f"blocks.{i}.cross_attn.proj.bias"
        )

        # MLP
        converted_state_dict[f"transformer_blocks.{i}.ff.conv_inverted.weight"] = checkpoint.pop(
            f"blocks.{i}.mlp.inverted_conv.conv.weight"
        )
        converted_state_dict[f"transformer_blocks.{i}.ff.conv_inverted.bias"] = checkpoint.pop(
            f"blocks.{i}.mlp.inverted_conv.conv.bias"
        )
        converted_state_dict[f"transformer_blocks.{i}.ff.conv_depth.weight"] = checkpoint.pop(
            f"blocks.{i}.mlp.depth_conv.conv.weight"
        )
        converted_state_dict[f"transformer_blocks.{i}.ff.conv_depth.bias"] = checkpoint.pop(
            f"blocks.{i}.mlp.depth_conv.conv.bias"
        )
        converted_state_dict[f"transformer_blocks.{i}.ff.conv_point.weight"] = checkpoint.pop(
            f"blocks.{i}.mlp.point_conv.conv.weight"
        )

    # Final layer
    converted_state_dict["proj_out.weight"] = checkpoint.pop("final_layer.linear.weight")
    converted_state_dict["proj_out.bias"] = checkpoint.pop("final_layer.linear.bias")
    converted_state_dict["scale_shift_table"] = checkpoint.pop("final_layer.scale_shift_table")

    return converted_state_dict


def convert_wan_transformer_to_diffusers(checkpoint, **kwargs):
    converted_state_dict = {}

    keys = list(checkpoint.keys())
    for k in keys:
        if "model.diffusion_model." in k:
            checkpoint[k.replace("model.diffusion_model.", "")] = checkpoint.pop(k)

    TRANSFORMER_KEYS_RENAME_DICT = {
        "time_embedding.0": "condition_embedder.time_embedder.linear_1",
        "time_embedding.2": "condition_embedder.time_embedder.linear_2",
        "text_embedding.0": "condition_embedder.text_embedder.linear_1",
        "text_embedding.2": "condition_embedder.text_embedder.linear_2",
        "time_projection.1": "condition_embedder.time_proj",
        "cross_attn": "attn2",
        "self_attn": "attn1",
        ".o.": ".to_out.0.",
        ".q.": ".to_q.",
        ".k.": ".to_k.",
        ".v.": ".to_v.",
        ".k_img.": ".add_k_proj.",
        ".v_img.": ".add_v_proj.",
        ".norm_k_img.": ".norm_added_k.",
        "head.modulation": "scale_shift_table",
        "head.head": "proj_out",
        "modulation": "scale_shift_table",
        "ffn.0": "ffn.net.0.proj",
        "ffn.2": "ffn.net.2",
        # Hack to swap the layer names
        # The original model calls the norms in following order: norm1, norm3, norm2
        # We convert it to: norm1, norm2, norm3
        "norm2": "norm__placeholder",
        "norm3": "norm2",
        "norm__placeholder": "norm3",
        # For the I2V model
        "img_emb.proj.0": "condition_embedder.image_embedder.norm1",
        "img_emb.proj.1": "condition_embedder.image_embedder.ff.net.0.proj",
        "img_emb.proj.3": "condition_embedder.image_embedder.ff.net.2",
        "img_emb.proj.4": "condition_embedder.image_embedder.norm2",
        # For the VACE model
        "before_proj": "proj_in",
        "after_proj": "proj_out",
    }

    for key in list(checkpoint.keys()):
        new_key = key[:]
        for replace_key, rename_key in TRANSFORMER_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)

        converted_state_dict[new_key] = checkpoint.pop(key)

    return converted_state_dict


def convert_wan_vae_to_diffusers(checkpoint, **kwargs):
    converted_state_dict = {}

    # Create mappings for specific components
    middle_key_mapping = {
        # Encoder middle block
        "encoder.middle.0.residual.0.gamma": "encoder.mid_block.resnets.0.norm1.gamma",
        "encoder.middle.0.residual.2.bias": "encoder.mid_block.resnets.0.conv1.bias",
        "encoder.middle.0.residual.2.weight": "encoder.mid_block.resnets.0.conv1.weight",
        "encoder.middle.0.residual.3.gamma": "encoder.mid_block.resnets.0.norm2.gamma",
        "encoder.middle.0.residual.6.bias": "encoder.mid_block.resnets.0.conv2.bias",
        "encoder.middle.0.residual.6.weight": "encoder.mid_block.resnets.0.conv2.weight",
        "encoder.middle.2.residual.0.gamma": "encoder.mid_block.resnets.1.norm1.gamma",
        "encoder.middle.2.residual.2.bias": "encoder.mid_block.resnets.1.conv1.bias",
        "encoder.middle.2.residual.2.weight": "encoder.mid_block.resnets.1.conv1.weight",
        "encoder.middle.2.residual.3.gamma": "encoder.mid_block.resnets.1.norm2.gamma",
        "encoder.middle.2.residual.6.bias": "encoder.mid_block.resnets.1.conv2.bias",
        "encoder.middle.2.residual.6.weight": "encoder.mid_block.resnets.1.conv2.weight",
        # Decoder middle block
        "decoder.middle.0.residual.0.gamma": "decoder.mid_block.resnets.0.norm1.gamma",
        "decoder.middle.0.residual.2.bias": "decoder.mid_block.resnets.0.conv1.bias",
        "decoder.middle.0.residual.2.weight": "decoder.mid_block.resnets.0.conv1.weight",
        "decoder.middle.0.residual.3.gamma": "decoder.mid_block.resnets.0.norm2.gamma",
        "decoder.middle.0.residual.6.bias": "decoder.mid_block.resnets.0.conv2.bias",
        "decoder.middle.0.residual.6.weight": "decoder.mid_block.resnets.0.conv2.weight",
        "decoder.middle.2.residual.0.gamma": "decoder.mid_block.resnets.1.norm1.gamma",
        "decoder.middle.2.residual.2.bias": "decoder.mid_block.resnets.1.conv1.bias",
        "decoder.middle.2.residual.2.weight": "decoder.mid_block.resnets.1.conv1.weight",
        "decoder.middle.2.residual.3.gamma": "decoder.mid_block.resnets.1.norm2.gamma",
        "decoder.middle.2.residual.6.bias": "decoder.mid_block.resnets.1.conv2.bias",
        "decoder.middle.2.residual.6.weight": "decoder.mid_block.resnets.1.conv2.weight",
    }

    # Create a mapping for attention blocks
    attention_mapping = {
        # Encoder middle attention
        "encoder.middle.1.norm.gamma": "encoder.mid_block.attentions.0.norm.gamma",
        "encoder.middle.1.to_qkv.weight": "encoder.mid_block.attentions.0.to_qkv.weight",
        "encoder.middle.1.to_qkv.bias": "encoder.mid_block.attentions.0.to_qkv.bias",
        "encoder.middle.1.proj.weight": "encoder.mid_block.attentions.0.proj.weight",
        "encoder.middle.1.proj.bias": "encoder.mid_block.attentions.0.proj.bias",
        # Decoder middle attention
        "decoder.middle.1.norm.gamma": "decoder.mid_block.attentions.0.norm.gamma",
        "decoder.middle.1.to_qkv.weight": "decoder.mid_block.attentions.0.to_qkv.weight",
        "decoder.middle.1.to_qkv.bias": "decoder.mid_block.attentions.0.to_qkv.bias",
        "decoder.middle.1.proj.weight": "decoder.mid_block.attentions.0.proj.weight",
        "decoder.middle.1.proj.bias": "decoder.mid_block.attentions.0.proj.bias",
    }

    # Create a mapping for the head components
    head_mapping = {
        # Encoder head
        "encoder.head.0.gamma": "encoder.norm_out.gamma",
        "encoder.head.2.bias": "encoder.conv_out.bias",
        "encoder.head.2.weight": "encoder.conv_out.weight",
        # Decoder head
        "decoder.head.0.gamma": "decoder.norm_out.gamma",
        "decoder.head.2.bias": "decoder.conv_out.bias",
        "decoder.head.2.weight": "decoder.conv_out.weight",
    }

    # Create a mapping for the quant components
    quant_mapping = {
        "conv1.weight": "quant_conv.weight",
        "conv1.bias": "quant_conv.bias",
        "conv2.weight": "post_quant_conv.weight",
        "conv2.bias": "post_quant_conv.bias",
    }

    # Process each key in the state dict
    for key, value in checkpoint.items():
        # Handle middle block keys using the mapping
        if key in middle_key_mapping:
            new_key = middle_key_mapping[key]
            converted_state_dict[new_key] = value
        # Handle attention blocks using the mapping
        elif key in attention_mapping:
            new_key = attention_mapping[key]
            converted_state_dict[new_key] = value
        # Handle head keys using the mapping
        elif key in head_mapping:
            new_key = head_mapping[key]
            converted_state_dict[new_key] = value
        # Handle quant keys using the mapping
        elif key in quant_mapping:
            new_key = quant_mapping[key]
            converted_state_dict[new_key] = value
        # Handle encoder conv1
        elif key == "encoder.conv1.weight":
            converted_state_dict["encoder.conv_in.weight"] = value
        elif key == "encoder.conv1.bias":
            converted_state_dict["encoder.conv_in.bias"] = value
        # Handle decoder conv1
        elif key == "decoder.conv1.weight":
            converted_state_dict["decoder.conv_in.weight"] = value
        elif key == "decoder.conv1.bias":
            converted_state_dict["decoder.conv_in.bias"] = value
        # Handle encoder downsamples
        elif key.startswith("encoder.downsamples."):
            # Convert to down_blocks
            new_key = key.replace("encoder.downsamples.", "encoder.down_blocks.")

            # Convert residual block naming but keep the original structure
            if ".residual.0.gamma" in new_key:
                new_key = new_key.replace(".residual.0.gamma", ".norm1.gamma")
            elif ".residual.2.bias" in new_key:
                new_key = new_key.replace(".residual.2.bias", ".conv1.bias")
            elif ".residual.2.weight" in new_key:
                new_key = new_key.replace(".residual.2.weight", ".conv1.weight")
            elif ".residual.3.gamma" in new_key:
                new_key = new_key.replace(".residual.3.gamma", ".norm2.gamma")
            elif ".residual.6.bias" in new_key:
                new_key = new_key.replace(".residual.6.bias", ".conv2.bias")
            elif ".residual.6.weight" in new_key:
                new_key = new_key.replace(".residual.6.weight", ".conv2.weight")
            elif ".shortcut.bias" in new_key:
                new_key = new_key.replace(".shortcut.bias", ".conv_shortcut.bias")
            elif ".shortcut.weight" in new_key:
                new_key = new_key.replace(".shortcut.weight", ".conv_shortcut.weight")

            converted_state_dict[new_key] = value

        # Handle decoder upsamples
        elif key.startswith("decoder.upsamples."):
            # Convert to up_blocks
            parts = key.split(".")
            block_idx = int(parts[2])

            # Group residual blocks
            if "residual" in key:
                if block_idx in [0, 1, 2]:
                    new_block_idx = 0
                    resnet_idx = block_idx
                elif block_idx in [4, 5, 6]:
                    new_block_idx = 1
                    resnet_idx = block_idx - 4
                elif block_idx in [8, 9, 10]:
                    new_block_idx = 2
                    resnet_idx = block_idx - 8
                elif block_idx in [12, 13, 14]:
                    new_block_idx = 3
                    resnet_idx = block_idx - 12
                else:
                    # Keep as is for other blocks
                    converted_state_dict[key] = value
                    continue

                # Convert residual block naming
                if ".residual.0.gamma" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.norm1.gamma"
                elif ".residual.2.bias" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.conv1.bias"
                elif ".residual.2.weight" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.conv1.weight"
                elif ".residual.3.gamma" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.norm2.gamma"
                elif ".residual.6.bias" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.conv2.bias"
                elif ".residual.6.weight" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.conv2.weight"
                else:
                    new_key = key

                converted_state_dict[new_key] = value

            # Handle shortcut connections
            elif ".shortcut." in key:
                if block_idx == 4:
                    new_key = key.replace(".shortcut.", ".resnets.0.conv_shortcut.")
                    new_key = new_key.replace("decoder.upsamples.4", "decoder.up_blocks.1")
                else:
                    new_key = key.replace("decoder.upsamples.", "decoder.up_blocks.")
                    new_key = new_key.replace(".shortcut.", ".conv_shortcut.")

                converted_state_dict[new_key] = value

            # Handle upsamplers
            elif ".resample." in key or ".time_conv." in key:
                if block_idx == 3:
                    new_key = key.replace(f"decoder.upsamples.{block_idx}", "decoder.up_blocks.0.upsamplers.0")
                elif block_idx == 7:
                    new_key = key.replace(f"decoder.upsamples.{block_idx}", "decoder.up_blocks.1.upsamplers.0")
                elif block_idx == 11:
                    new_key = key.replace(f"decoder.upsamples.{block_idx}", "decoder.up_blocks.2.upsamplers.0")
                else:
                    new_key = key.replace("decoder.upsamples.", "decoder.up_blocks.")

                converted_state_dict[new_key] = value
            else:
                new_key = key.replace("decoder.upsamples.", "decoder.up_blocks.")
                converted_state_dict[new_key] = value
        else:
            # Keep other keys unchanged
            converted_state_dict[key] = value

    return converted_state_dict


def convert_hidream_transformer_to_diffusers(checkpoint, **kwargs):
    keys = list(checkpoint.keys())
    for k in keys:
        if "model.diffusion_model." in k:
            checkpoint[k.replace("model.diffusion_model.", "")] = checkpoint.pop(k)

    return checkpoint


def convert_chroma_transformer_checkpoint_to_diffusers(checkpoint, **kwargs):
    converted_state_dict = {}
    keys = list(checkpoint.keys())

    for k in keys:
        if "model.diffusion_model." in k:
            checkpoint[k.replace("model.diffusion_model.", "")] = checkpoint.pop(k)

    num_layers = list(set(int(k.split(".", 2)[1]) for k in checkpoint if "double_blocks." in k))[-1] + 1  # noqa: C401
    num_single_layers = list(set(int(k.split(".", 2)[1]) for k in checkpoint if "single_blocks." in k))[-1] + 1  # noqa: C401
    num_guidance_layers = (
        list(set(int(k.split(".", 3)[2]) for k in checkpoint if "distilled_guidance_layer.layers." in k))[-1] + 1  # noqa: C401
    )
    mlp_ratio = 4.0
    inner_dim = 3072

    # in SD3 original implementation of AdaLayerNormContinuous, it split linear projection output into shift, scale;
    # while in diffusers it split into scale, shift. Here we swap the linear projection weights in order to be able to use diffusers implementation
    def swap_scale_shift(weight):
        shift, scale = weight.chunk(2, dim=0)
        new_weight = torch.cat([scale, shift], dim=0)
        return new_weight

    # guidance
    converted_state_dict["distilled_guidance_layer.in_proj.bias"] = checkpoint.pop(
        "distilled_guidance_layer.in_proj.bias"
    )
    converted_state_dict["distilled_guidance_layer.in_proj.weight"] = checkpoint.pop(
        "distilled_guidance_layer.in_proj.weight"
    )
    converted_state_dict["distilled_guidance_layer.out_proj.bias"] = checkpoint.pop(
        "distilled_guidance_layer.out_proj.bias"
    )
    converted_state_dict["distilled_guidance_layer.out_proj.weight"] = checkpoint.pop(
        "distilled_guidance_layer.out_proj.weight"
    )
    for i in range(num_guidance_layers):
        block_prefix = f"distilled_guidance_layer.layers.{i}."
        converted_state_dict[f"{block_prefix}linear_1.bias"] = checkpoint.pop(
            f"distilled_guidance_layer.layers.{i}.in_layer.bias"
        )
        converted_state_dict[f"{block_prefix}linear_1.weight"] = checkpoint.pop(
            f"distilled_guidance_layer.layers.{i}.in_layer.weight"
        )
        converted_state_dict[f"{block_prefix}linear_2.bias"] = checkpoint.pop(
            f"distilled_guidance_layer.layers.{i}.out_layer.bias"
        )
        converted_state_dict[f"{block_prefix}linear_2.weight"] = checkpoint.pop(
            f"distilled_guidance_layer.layers.{i}.out_layer.weight"
        )
        converted_state_dict[f"distilled_guidance_layer.norms.{i}.weight"] = checkpoint.pop(
            f"distilled_guidance_layer.norms.{i}.scale"
        )

    # context_embedder
    converted_state_dict["context_embedder.weight"] = checkpoint.pop("txt_in.weight")
    converted_state_dict["context_embedder.bias"] = checkpoint.pop("txt_in.bias")

    # x_embedder
    converted_state_dict["x_embedder.weight"] = checkpoint.pop("img_in.weight")
    converted_state_dict["x_embedder.bias"] = checkpoint.pop("img_in.bias")

    # double transformer blocks
    for i in range(num_layers):
        block_prefix = f"transformer_blocks.{i}."
        # Q, K, V
        sample_q, sample_k, sample_v = torch.chunk(checkpoint.pop(f"double_blocks.{i}.img_attn.qkv.weight"), 3, dim=0)
        context_q, context_k, context_v = torch.chunk(
            checkpoint.pop(f"double_blocks.{i}.txt_attn.qkv.weight"), 3, dim=0
        )
        sample_q_bias, sample_k_bias, sample_v_bias = torch.chunk(
            checkpoint.pop(f"double_blocks.{i}.img_attn.qkv.bias"), 3, dim=0
        )
        context_q_bias, context_k_bias, context_v_bias = torch.chunk(
            checkpoint.pop(f"double_blocks.{i}.txt_attn.qkv.bias"), 3, dim=0
        )
        converted_state_dict[f"{block_prefix}attn.to_q.weight"] = torch.cat([sample_q])
        converted_state_dict[f"{block_prefix}attn.to_q.bias"] = torch.cat([sample_q_bias])
        converted_state_dict[f"{block_prefix}attn.to_k.weight"] = torch.cat([sample_k])
        converted_state_dict[f"{block_prefix}attn.to_k.bias"] = torch.cat([sample_k_bias])
        converted_state_dict[f"{block_prefix}attn.to_v.weight"] = torch.cat([sample_v])
        converted_state_dict[f"{block_prefix}attn.to_v.bias"] = torch.cat([sample_v_bias])
        converted_state_dict[f"{block_prefix}attn.add_q_proj.weight"] = torch.cat([context_q])
        converted_state_dict[f"{block_prefix}attn.add_q_proj.bias"] = torch.cat([context_q_bias])
        converted_state_dict[f"{block_prefix}attn.add_k_proj.weight"] = torch.cat([context_k])
        converted_state_dict[f"{block_prefix}attn.add_k_proj.bias"] = torch.cat([context_k_bias])
        converted_state_dict[f"{block_prefix}attn.add_v_proj.weight"] = torch.cat([context_v])
        converted_state_dict[f"{block_prefix}attn.add_v_proj.bias"] = torch.cat([context_v_bias])
        # qk_norm
        converted_state_dict[f"{block_prefix}attn.norm_q.weight"] = checkpoint.pop(
            f"double_blocks.{i}.img_attn.norm.query_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_k.weight"] = checkpoint.pop(
            f"double_blocks.{i}.img_attn.norm.key_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_added_q.weight"] = checkpoint.pop(
            f"double_blocks.{i}.txt_attn.norm.query_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_added_k.weight"] = checkpoint.pop(
            f"double_blocks.{i}.txt_attn.norm.key_norm.scale"
        )
        # ff img_mlp
        converted_state_dict[f"{block_prefix}ff.net.0.proj.weight"] = checkpoint.pop(
            f"double_blocks.{i}.img_mlp.0.weight"
        )
        converted_state_dict[f"{block_prefix}ff.net.0.proj.bias"] = checkpoint.pop(f"double_blocks.{i}.img_mlp.0.bias")
        converted_state_dict[f"{block_prefix}ff.net.2.weight"] = checkpoint.pop(f"double_blocks.{i}.img_mlp.2.weight")
        converted_state_dict[f"{block_prefix}ff.net.2.bias"] = checkpoint.pop(f"double_blocks.{i}.img_mlp.2.bias")
        converted_state_dict[f"{block_prefix}ff_context.net.0.proj.weight"] = checkpoint.pop(
            f"double_blocks.{i}.txt_mlp.0.weight"
        )
        converted_state_dict[f"{block_prefix}ff_context.net.0.proj.bias"] = checkpoint.pop(
            f"double_blocks.{i}.txt_mlp.0.bias"
        )
        converted_state_dict[f"{block_prefix}ff_context.net.2.weight"] = checkpoint.pop(
            f"double_blocks.{i}.txt_mlp.2.weight"
        )
        converted_state_dict[f"{block_prefix}ff_context.net.2.bias"] = checkpoint.pop(
            f"double_blocks.{i}.txt_mlp.2.bias"
        )
        # output projections.
        converted_state_dict[f"{block_prefix}attn.to_out.0.weight"] = checkpoint.pop(
            f"double_blocks.{i}.img_attn.proj.weight"
        )
        converted_state_dict[f"{block_prefix}attn.to_out.0.bias"] = checkpoint.pop(
            f"double_blocks.{i}.img_attn.proj.bias"
        )
        converted_state_dict[f"{block_prefix}attn.to_add_out.weight"] = checkpoint.pop(
            f"double_blocks.{i}.txt_attn.proj.weight"
        )
        converted_state_dict[f"{block_prefix}attn.to_add_out.bias"] = checkpoint.pop(
            f"double_blocks.{i}.txt_attn.proj.bias"
        )

    # single transformer blocks
    for i in range(num_single_layers):
        block_prefix = f"single_transformer_blocks.{i}."
        # Q, K, V, mlp
        mlp_hidden_dim = int(inner_dim * mlp_ratio)
        split_size = (inner_dim, inner_dim, inner_dim, mlp_hidden_dim)
        q, k, v, mlp = torch.split(checkpoint.pop(f"single_blocks.{i}.linear1.weight"), split_size, dim=0)
        q_bias, k_bias, v_bias, mlp_bias = torch.split(
            checkpoint.pop(f"single_blocks.{i}.linear1.bias"), split_size, dim=0
        )
        converted_state_dict[f"{block_prefix}attn.to_q.weight"] = torch.cat([q])
        converted_state_dict[f"{block_prefix}attn.to_q.bias"] = torch.cat([q_bias])
        converted_state_dict[f"{block_prefix}attn.to_k.weight"] = torch.cat([k])
        converted_state_dict[f"{block_prefix}attn.to_k.bias"] = torch.cat([k_bias])
        converted_state_dict[f"{block_prefix}attn.to_v.weight"] = torch.cat([v])
        converted_state_dict[f"{block_prefix}attn.to_v.bias"] = torch.cat([v_bias])
        converted_state_dict[f"{block_prefix}proj_mlp.weight"] = torch.cat([mlp])
        converted_state_dict[f"{block_prefix}proj_mlp.bias"] = torch.cat([mlp_bias])
        # qk norm
        converted_state_dict[f"{block_prefix}attn.norm_q.weight"] = checkpoint.pop(
            f"single_blocks.{i}.norm.query_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_k.weight"] = checkpoint.pop(
            f"single_blocks.{i}.norm.key_norm.scale"
        )
        # output projections.
        converted_state_dict[f"{block_prefix}proj_out.weight"] = checkpoint.pop(f"single_blocks.{i}.linear2.weight")
        converted_state_dict[f"{block_prefix}proj_out.bias"] = checkpoint.pop(f"single_blocks.{i}.linear2.bias")

    converted_state_dict["proj_out.weight"] = checkpoint.pop("final_layer.linear.weight")
    converted_state_dict["proj_out.bias"] = checkpoint.pop("final_layer.linear.bias")

    return converted_state_dict


def convert_cosmos_transformer_checkpoint_to_diffusers(checkpoint, **kwargs):
    converted_state_dict = {key: checkpoint.pop(key) for key in list(checkpoint.keys())}

    def remove_keys_(key: str, state_dict):
        state_dict.pop(key)

    def rename_transformer_blocks_(key: str, state_dict):
        block_index = int(key.split(".")[1].removeprefix("block"))
        new_key = key
        old_prefix = f"blocks.block{block_index}"
        new_prefix = f"transformer_blocks.{block_index}"
        new_key = new_prefix + new_key.removeprefix(old_prefix)
        state_dict[new_key] = state_dict.pop(key)

    TRANSFORMER_KEYS_RENAME_DICT_COSMOS_1_0 = {
        "t_embedder.1": "time_embed.t_embedder",
        "affline_norm": "time_embed.norm",
        ".blocks.0.block.attn": ".attn1",
        ".blocks.1.block.attn": ".attn2",
        ".blocks.2.block": ".ff",
        ".blocks.0.adaLN_modulation.1": ".norm1.linear_1",
        ".blocks.0.adaLN_modulation.2": ".norm1.linear_2",
        ".blocks.1.adaLN_modulation.1": ".norm2.linear_1",
        ".blocks.1.adaLN_modulation.2": ".norm2.linear_2",
        ".blocks.2.adaLN_modulation.1": ".norm3.linear_1",
        ".blocks.2.adaLN_modulation.2": ".norm3.linear_2",
        "to_q.0": "to_q",
        "to_q.1": "norm_q",
        "to_k.0": "to_k",
        "to_k.1": "norm_k",
        "to_v.0": "to_v",
        "layer1": "net.0.proj",
        "layer2": "net.2",
        "proj.1": "proj",
        "x_embedder": "patch_embed",
        "extra_pos_embedder": "learnable_pos_embed",
        "final_layer.adaLN_modulation.1": "norm_out.linear_1",
        "final_layer.adaLN_modulation.2": "norm_out.linear_2",
        "final_layer.linear": "proj_out",
    }

    TRANSFORMER_SPECIAL_KEYS_REMAP_COSMOS_1_0 = {
        "blocks.block": rename_transformer_blocks_,
        "logvar.0.freqs": remove_keys_,
        "logvar.0.phases": remove_keys_,
        "logvar.1.weight": remove_keys_,
        "pos_embedder.seq": remove_keys_,
    }

    TRANSFORMER_KEYS_RENAME_DICT_COSMOS_2_0 = {
        "t_embedder.1": "time_embed.t_embedder",
        "t_embedding_norm": "time_embed.norm",
        "blocks": "transformer_blocks",
        "adaln_modulation_self_attn.1": "norm1.linear_1",
        "adaln_modulation_self_attn.2": "norm1.linear_2",
        "adaln_modulation_cross_attn.1": "norm2.linear_1",
        "adaln_modulation_cross_attn.2": "norm2.linear_2",
        "adaln_modulation_mlp.1": "norm3.linear_1",
        "adaln_modulation_mlp.2": "norm3.linear_2",
        "self_attn": "attn1",
        "cross_attn": "attn2",
        "q_proj": "to_q",
        "k_proj": "to_k",
        "v_proj": "to_v",
        "output_proj": "to_out.0",
        "q_norm": "norm_q",
        "k_norm": "norm_k",
        "mlp.layer1": "ff.net.0.proj",
        "mlp.layer2": "ff.net.2",
        "x_embedder.proj.1": "patch_embed.proj",
        "final_layer.adaln_modulation.1": "norm_out.linear_1",
        "final_layer.adaln_modulation.2": "norm_out.linear_2",
        "final_layer.linear": "proj_out",
    }

    TRANSFORMER_SPECIAL_KEYS_REMAP_COSMOS_2_0 = {
        "accum_video_sample_counter": remove_keys_,
        "accum_image_sample_counter": remove_keys_,
        "accum_iteration": remove_keys_,
        "accum_train_in_hours": remove_keys_,
        "pos_embedder.seq": remove_keys_,
        "pos_embedder.dim_spatial_range": remove_keys_,
        "pos_embedder.dim_temporal_range": remove_keys_,
        "_extra_state": remove_keys_,
    }

    PREFIX_KEY = "net."
    if "net.blocks.block1.blocks.0.block.attn.to_q.0.weight" in checkpoint:
        TRANSFORMER_KEYS_RENAME_DICT = TRANSFORMER_KEYS_RENAME_DICT_COSMOS_1_0
        TRANSFORMER_SPECIAL_KEYS_REMAP = TRANSFORMER_SPECIAL_KEYS_REMAP_COSMOS_1_0
    else:
        TRANSFORMER_KEYS_RENAME_DICT = TRANSFORMER_KEYS_RENAME_DICT_COSMOS_2_0
        TRANSFORMER_SPECIAL_KEYS_REMAP = TRANSFORMER_SPECIAL_KEYS_REMAP_COSMOS_2_0

    state_dict_keys = list(converted_state_dict.keys())
    for key in state_dict_keys:
        new_key = key[:]
        if new_key.startswith(PREFIX_KEY):
            new_key = new_key.removeprefix(PREFIX_KEY)
        for replace_key, rename_key in TRANSFORMER_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)
        converted_state_dict[new_key] = converted_state_dict.pop(key)

    state_dict_keys = list(converted_state_dict.keys())
    for key in state_dict_keys:
        for special_key, handler_fn_inplace in TRANSFORMER_SPECIAL_KEYS_REMAP.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, converted_state_dict)

    return converted_state_dict
