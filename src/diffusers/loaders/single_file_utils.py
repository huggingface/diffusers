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

from contextlib import nullcontext

import torch

from ..utils import deprecate
from .single_file.single_file_utils import (
    CHECKPOINT_KEY_NAMES,
    DIFFUSERS_TO_LDM_MAPPING,
    LDM_CLIP_PREFIX_TO_REMOVE,
    LDM_OPEN_CLIP_TEXT_PROJECTION_DIM,
    SD_2_TEXT_ENCODER_KEYS_TO_IGNORE,
    SingleFileComponentError,
)


class SingleFileComponentError(SingleFileComponentError):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `SingleFileComponentError` from diffusers.loaders.single_file_utils has been deprecated. Please use `from diffusers.loaders.single_file.single_files_utils import SingleFileComponentError` instead."
        deprecate("diffusers.loaders.single_file_utils. ", "0.36", deprecation_message)
        super().__init__(*args, **kwargs)


def is_valid_url(url):
    from .single_file.single_file_utils import is_valid_url

    deprecation_message = "Importing `is_valid_url()` from diffusers.loaders.single_file_utils has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import is_valid_url` instead."
    deprecate("diffusers.loaders.single_file_utils.is_valid_url", "0.36", deprecation_message)

    return is_valid_url(url)


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
    from .single_file.single_file_utils import load_single_file_checkpoint

    deprecation_message = "Importing `load_single_file_checkpoint()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import load_single_file_checkpoint` instead."
    deprecate("diffusers.loaders.single_file_utils.load_single_file_checkpoint", "0.36", deprecation_message)

    return load_single_file_checkpoint(
        pretrained_model_link_or_path,
    force_download,
    proxies,
    token,
    cache_dir,
    local_files_only,
    revision,
    disable_mmap,
    user_agent,
    )


def fetch_original_config(original_config_file, local_files_only=False):
    from .single_file.single_file_utils import fetch_original_config

    deprecation_message = "Importing `fetch_original_config()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import fetch_original_config` instead."
    deprecate("diffusers.loaders.single_file_utils.fetch_original_config", "0.36", deprecation_message)

    return fetch_original_config(original_config_file, local_files_only)


def is_clip_model(checkpoint):
    from .single_file.single_file_utils import is_clip_model

    deprecation_message = "Importing `is_clip_model()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import is_clip_model` instead."
    deprecate("diffusers.loaders.single_file_utils.is_clip_model", "0.36", deprecation_message)

    return is_clip_model(checkpoint)


def is_clip_sdxl_model(checkpoint):
    from .single_file.single_file_utils import is_clip_sdxl_model

    deprecation_message = "Importing `is_clip_sdxl_model()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import is_clip_sdxl_model` instead."
    deprecate("diffusers.loaders.single_file_utils.is_clip_sdxl_model", "0.36", deprecation_message)

    return is_clip_sdxl_model(checkpoint)


def is_clip_sd3_model(checkpoint):
    from .single_file.single_file_utils import is_clip_sd3_model

    deprecation_message = "Importing `is_clip_sd3_model()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import is_clip_sd3_model` instead."
    deprecate("diffusers.loaders.single_file_utils.is_clip_sd3_model", "0.36", deprecation_message)

    return is_clip_sd3_model(checkpoint)


def is_open_clip_model(checkpoint):

    deprecation_message = "Importing `is_open_clip_model()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import is_open_clip_model` instead."
    deprecate("diffusers.loaders.single_file_utils.is_open_clip_model", "0.36", deprecation_message)

    return is_open_clip_model(checkpoint)


def is_open_clip_sdxl_model(checkpoint):
    from .single_file.single_file_utils import is_open_clip_sdxl_model

    deprecation_message = "Importing `is_open_clip_sdxl_model()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import is_open_clip_sdxl_model` instead."
    deprecate("diffusers.loaders.single_file_utils.is_open_clip_sdxl_model", "0.36", deprecation_message)

    return is_open_clip_sdxl_model(checkpoint)

def is_open_clip_sd3_model(checkpoint):
    from .single_file.single_file_utils import is_open_clip_sd3_model

    deprecation_message = "Importing `is_open_clip_sd3_model()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import is_open_clip_sd3_model` instead."
    deprecate("diffusers.loaders.single_file_utils.is_open_clip_sd3_model", "0.36", deprecation_message)

    return is_open_clip_sd3_model(checkpoint)


def is_open_clip_sdxl_refiner_model(checkpoint):
    from .single_file.single_file_utils import is_open_clip_sdxl_refiner_model

    deprecation_message = "Importing `is_open_clip_sdxl_refiner_model()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import is_open_clip_sdxl_refiner_model` instead."
    deprecate("diffusers.loaders.single_file_utils.is_open_clip_sdxl_refiner_model", "0.36", deprecation_message)

    return is_open_clip_sdxl_refiner_model(checkpoint)


def is_clip_model_in_single_file(class_obj, checkpoint):
    from .single_file.single_file_utils import is_clip_model_in_single_file

    deprecation_message = "Importing `is_clip_model_in_single_file()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import is_clip_model_in_single_file` instead."
    deprecate("diffusers.loaders.single_file_utils.is_clip_model_in_single_file", "0.36", deprecation_message)

    return is_clip_model_in_single_file(class_obj, checkpoint)


def infer_diffusers_model_type(checkpoint):
    from .single_file.single_file_utils import infer_diffusers_model_type

    deprecation_message = "Importing `infer_diffusers_model_type()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import infer_diffusers_model_type` instead."
    deprecate("diffusers.loaders.single_file_utils.infer_diffusers_model_type", "0.36", deprecation_message)

    return infer_diffusers_model_type(checkpoint)



def fetch_diffusers_config(checkpoint):
    from .single_file.single_file_utils import fetch_diffusers_config

    deprecation_message = "Importing `fetch_diffusers_config()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import fetch_diffusers_config` instead."
    deprecate("diffusers.loaders.single_file_utils.fetch_diffusers_config", "0.36", deprecation_message)

    return fetch_diffusers_config(checkpoint)


def set_image_size(checkpoint, image_size=None):
    from .single_file.single_file_utils import set_image_size

    deprecation_message = "Importing `set_image_size()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import set_image_size` instead."
    deprecate("diffusers.loaders.single_file_utils.set_image_size", "0.36", deprecation_message)

    return set_image_size(checkpoint, image_size)


def conv_attn_to_linear(checkpoint):
    from .single_file.single_file_utils import conv_attn_to_linear

    deprecation_message = "Importing `conv_attn_to_linear()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import conv_attn_to_linear` instead."
    deprecate("diffusers.loaders.single_file_utils.conv_attn_to_linear", "0.36", deprecation_message)

    return conv_attn_to_linear(checkpoint)



def create_unet_diffusers_config_from_ldm(
    original_config, checkpoint, image_size=None, upcast_attention=None, num_in_channels=None
):
    from .single_file.single_file_utils import create_unet_diffusers_config_from_ldm

    deprecation_message = "Importing `create_unet_diffusers_config_from_ldm()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import create_unet_diffusers_config_from_ldm` instead."
    deprecate("diffusers.loaders.single_file_utils.create_unet_diffusers_config_from_ldm", "0.36", deprecation_message)

    return create_unet_diffusers_config_from_ldm(original_config, checkpoint, image_size, upcast_attention, num_in_channels)


def create_controlnet_diffusers_config_from_ldm(original_config, checkpoint, image_size=None, **kwargs):
    from .single_file.single_file_utils import create_controlnet_diffusers_config_from_ldm

    deprecation_message = "Importing `create_controlnet_diffusers_config_from_ldm()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import create_controlnet_diffusers_config_from_ldm` instead."
    deprecate("diffusers.loaders.single_file_utils.create_controlnet_diffusers_config_from_ldm", "0.36", deprecation_message)
    return create_controlnet_diffusers_config_from_ldm(original_config, checkpoint, image_size, **kwargs)

def create_vae_diffusers_config_from_ldm(original_config, checkpoint, image_size=None, scaling_factor=None):
    from .single_file.single_file_utils import create_vae_diffusers_config_from_ldm

    deprecation_message = "Importing `create_vae_diffusers_config_from_ldm()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import create_vae_diffusers_config_from_ldm` instead."
    deprecate("diffusers.loaders.single_file_utils.create_vae_diffusers_config_from_ldm", "0.36", deprecation_message)
    return create_vae_diffusers_config_from_ldm(original_config, checkpoint, image_size, scaling_factor)

def update_unet_resnet_ldm_to_diffusers(ldm_keys, new_checkpoint, checkpoint, mapping=None):
    from .single_file.single_file_utils import update_unet_resnet_ldm_to_diffusers

    deprecation_message = "Importing `update_unet_resnet_ldm_to_diffusers()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import update_unet_resnet_ldm_to_diffusers` instead."
    deprecate("diffusers.loaders.single_file_utils.update_unet_resnet_ldm_to_diffusers", "0.36", deprecation_message)

    return update_unet_resnet_ldm_to_diffusers(ldm_keys, new_checkpoint, checkpoint, mapping)


def update_unet_attention_ldm_to_diffusers(ldm_keys, new_checkpoint, checkpoint, mapping):
    from .single_file.single_file_utils import update_unet_attention_ldm_to_diffusers

    deprecation_message = "Importing `update_unet_attention_ldm_to_diffusers()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import update_unet_attention_ldm_to_diffusers` instead."
    deprecate("diffusers.loaders.single_file_utils.update_unet_attention_ldm_to_diffusers", "0.36", deprecation_message)

    return update_unet_attention_ldm_to_diffusers(ldm_keys, new_checkpoint, checkpoint, mapping)


def update_vae_resnet_ldm_to_diffusers(keys, new_checkpoint, checkpoint, mapping):
    from .single_file.single_file_utils import update_vae_resnet_ldm_to_diffusers

    deprecation_message = "Importing `update_vae_resnet_ldm_to_diffusers()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import update_vae_resnet_ldm_to_diffusers` instead."
    deprecate("diffusers.loaders.single_file_utils.update_vae_resnet_ldm_to_diffusers", "0.36", deprecation_message)

    return update_vae_resnet_ldm_to_diffusers(keys, new_checkpoint, checkpoint, mapping)


def update_vae_attentions_ldm_to_diffusers(keys, new_checkpoint, checkpoint, mapping):
    from .single_file.single_file_utils import update_vae_attentions_ldm_to_diffusers

    deprecation_message = "Importing `update_vae_attentions_ldm_to_diffusers()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import update_vae_attentions_ldm_to_diffusers` instead."
    deprecate("diffusers.loaders.single_file_utils.update_vae_attentions_ldm_to_diffusers", "0.36", deprecation_message)

    return update_vae_attentions_ldm_to_diffusers(keys, new_checkpoint, checkpoint, mapping)

def convert_stable_cascade_unet_single_file_to_diffusers(checkpoint, **kwargs):
    from .single_file.single_file_utils import convert_stable_cascade_unet_single_file_to_diffusers
    deprecation_message = "Importing `convert_stable_cascade_unet_single_file_to_diffusers()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import convert_stable_cascade_unet_single_file_to_diffusers` instead."
    deprecate("diffusers.loaders.single_file_utils.convert_stable_cascade_unet_single_file_to_diffusers", "0.36", deprecation_message)
    return convert_stable_cascade_unet_single_file_to_diffusers(checkpoint, **kwargs)


def convert_ldm_unet_checkpoint(checkpoint, config, extract_ema=False, **kwargs):
    from .single_file.single_file_utils import convert_ldm_unet_checkpoint
    deprecation_message = "Importing `convert_ldm_unet_checkpoint()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import convert_ldm_unet_checkpoint` instead."
    deprecate("diffusers.loaders.single_file_utils.convert_ldm_unet_checkpoint", "0.36", deprecation_message)
    return convert_ldm_unet_checkpoint(checkpoint, config, extract_ema, **kwargs)


def convert_controlnet_checkpoint(
    checkpoint,
    config,
    **kwargs,
):
    from .single_file.single_file_utils import convert_controlnet_checkpoint
    deprecation_message = "Importing `convert_controlnet_checkpoint()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import convert_controlnet_checkpoint` instead."
    deprecate("diffusers.loaders.single_file_utils.convert_controlnet_checkpoint", "0.36", deprecation_message)
    return convert_controlnet_checkpoint(checkpoint, config, **kwargs)



def convert_ldm_vae_checkpoint(checkpoint, config):
    from .single_file.single_file_utils import convert_ldm_vae_checkpoint
    deprecation_message = "Importing `convert_ldm_vae_checkpoint()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import convert_ldm_vae_checkpoint` instead."
    deprecate("diffusers.loaders.single_file_utils.convert_ldm_vae_checkpoint", "0.36", deprecation_message)
    return convert_ldm_vae_checkpoint(checkpoint, config, config)


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
    else:
        model.load_state_dict(diffusers_format_checkpoint, strict=False)

    if torch_dtype is not None:
        model.to(torch_dtype)

    model.eval()

    return model


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

    # single transfomer blocks
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

    # Original Lumina-Image-2 has an extra norm paramter that is unused
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
