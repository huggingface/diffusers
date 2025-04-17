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


from ..utils import deprecate
from .single_file.single_file_utils import SingleFileComponentError


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

    return create_unet_diffusers_config_from_ldm(
        original_config, checkpoint, image_size, upcast_attention, num_in_channels
    )


def create_controlnet_diffusers_config_from_ldm(original_config, checkpoint, image_size=None, **kwargs):
    from .single_file.single_file_utils import create_controlnet_diffusers_config_from_ldm

    deprecation_message = "Importing `create_controlnet_diffusers_config_from_ldm()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import create_controlnet_diffusers_config_from_ldm` instead."
    deprecate(
        "diffusers.loaders.single_file_utils.create_controlnet_diffusers_config_from_ldm", "0.36", deprecation_message
    )
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
    deprecate(
        "diffusers.loaders.single_file_utils.update_unet_attention_ldm_to_diffusers", "0.36", deprecation_message
    )

    return update_unet_attention_ldm_to_diffusers(ldm_keys, new_checkpoint, checkpoint, mapping)


def update_vae_resnet_ldm_to_diffusers(keys, new_checkpoint, checkpoint, mapping):
    from .single_file.single_file_utils import update_vae_resnet_ldm_to_diffusers

    deprecation_message = "Importing `update_vae_resnet_ldm_to_diffusers()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import update_vae_resnet_ldm_to_diffusers` instead."
    deprecate("diffusers.loaders.single_file_utils.update_vae_resnet_ldm_to_diffusers", "0.36", deprecation_message)

    return update_vae_resnet_ldm_to_diffusers(keys, new_checkpoint, checkpoint, mapping)


def update_vae_attentions_ldm_to_diffusers(keys, new_checkpoint, checkpoint, mapping):
    from .single_file.single_file_utils import update_vae_attentions_ldm_to_diffusers

    deprecation_message = "Importing `update_vae_attentions_ldm_to_diffusers()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import update_vae_attentions_ldm_to_diffusers` instead."
    deprecate(
        "diffusers.loaders.single_file_utils.update_vae_attentions_ldm_to_diffusers", "0.36", deprecation_message
    )

    return update_vae_attentions_ldm_to_diffusers(keys, new_checkpoint, checkpoint, mapping)


def convert_stable_cascade_unet_single_file_to_diffusers(checkpoint, **kwargs):
    from .single_file.single_file_utils import convert_stable_cascade_unet_single_file_to_diffusers

    deprecation_message = "Importing `convert_stable_cascade_unet_single_file_to_diffusers()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import convert_stable_cascade_unet_single_file_to_diffusers` instead."
    deprecate(
        "diffusers.loaders.single_file_utils.convert_stable_cascade_unet_single_file_to_diffusers",
        "0.36",
        deprecation_message,
    )
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
    from .single_file.single_file_utils import convert_ldm_clip_checkpoint

    deprecation_message = "Importing `convert_ldm_clip_checkpoint()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import convert_ldm_clip_checkpoint` instead."
    deprecate("diffusers.loaders.single_file_utils.convert_ldm_clip_checkpoint", "0.36", deprecation_message)
    return convert_ldm_clip_checkpoint(checkpoint, remove_prefix)


def convert_open_clip_checkpoint(
    text_model,
    checkpoint,
    prefix="cond_stage_model.model.",
):
    from .single_file.single_file_utils import convert_open_clip_checkpoint

    deprecation_message = "Importing `convert_open_clip_checkpoint()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import convert_open_clip_checkpoint` instead."
    deprecate("diffusers.loaders.single_file_utils.convert_open_clip_checkpoint", "0.36", deprecation_message)
    return convert_open_clip_checkpoint(text_model, checkpoint, prefix)


def create_diffusers_clip_model_from_ldm(
    cls,
    checkpoint,
    subfolder="",
    config=None,
    torch_dtype=None,
    local_files_only=None,
    is_legacy_loading=False,
):
    from .single_file.single_file_utils import create_diffusers_clip_model_from_ldm

    deprecation_message = "Importing `create_diffusers_clip_model_from_ldm()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import create_diffusers_clip_model_from_ldm` instead."
    deprecate("diffusers.loaders.single_file_utils.create_diffusers_clip_model_from_ldm", "0.36", deprecation_message)
    return create_diffusers_clip_model_from_ldm(
        cls, checkpoint, subfolder, config, torch_dtype, local_files_only, is_legacy_loading
    )


# in SD3 original implementation of AdaLayerNormContinuous, it split linear projection output into shift, scale;
# while in diffusers it split into scale, shift. Here we swap the linear projection weights in order to be able to use diffusers implementation
def swap_scale_shift(weight, dim):
    from .single_file.single_file_utils import swap_scale_shift

    deprecation_message = "Importing `swap_scale_shift()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import swap_scale_shift` instead."
    deprecate("diffusers.loaders.single_file_utils.swap_scale_shift", "0.36", deprecation_message)
    return swap_scale_shift(weight, dim)


def swap_proj_gate(weight):
    from .single_file.single_file_utils import swap_proj_gate

    deprecation_message = "Importing `swap_proj_gate()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import swap_proj_gate` instead."
    deprecate("diffusers.loaders.single_file_utils.swap_proj_gate", "0.36", deprecation_message)
    return swap_proj_gate(weight)


def get_attn2_layers(state_dict):
    from .single_file.single_file_utils import get_attn2_layers

    deprecation_message = "Importing `get_attn2_layers()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import get_attn2_layers` instead."
    deprecate("diffusers.loaders.single_file_utils.get_attn2_layers", "0.36", deprecation_message)
    return get_attn2_layers(state_dict)


def get_caption_projection_dim(state_dict):
    from .single_file.single_file_utils import get_caption_projection_dim

    deprecation_message = "Importing `get_caption_projection_dim()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import get_caption_projection_dim` instead."
    deprecate("diffusers.loaders.single_file_utils.get_caption_projection_dim", "0.36", deprecation_message)
    return get_caption_projection_dim(state_dict)


def convert_sd3_transformer_checkpoint_to_diffusers(checkpoint, **kwargs):
    from .single_file.single_file_utils import convert_sd3_transformer_checkpoint_to_diffusers

    deprecation_message = "Importing `convert_sd3_transformer_checkpoint_to_diffusers()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import convert_sd3_transformer_checkpoint_to_diffusers` instead."
    deprecate(
        "diffusers.loaders.single_file_utils.convert_sd3_transformer_checkpoint_to_diffusers",
        "0.36",
        deprecation_message,
    )
    return convert_sd3_transformer_checkpoint_to_diffusers(checkpoint, **kwargs)


def is_t5_in_single_file(checkpoint):
    from .single_file.single_file_utils import is_t5_in_single_file

    deprecation_message = "Importing `is_t5_in_single_file()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import is_t5_in_single_file` instead."
    deprecate("diffusers.loaders.single_file_utils.is_t5_in_single_file", "0.36", deprecation_message)
    return is_t5_in_single_file(checkpoint)


def convert_sd3_t5_checkpoint_to_diffusers(checkpoint):
    from .single_file.single_file_utils import convert_sd3_t5_checkpoint_to_diffusers

    deprecation_message = "Importing `convert_sd3_t5_checkpoint_to_diffusers()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import convert_sd3_t5_checkpoint_to_diffusers` instead."
    deprecate(
        "diffusers.loaders.single_file_utils.convert_sd3_t5_checkpoint_to_diffusers", "0.36", deprecation_message
    )
    return convert_sd3_t5_checkpoint_to_diffusers(checkpoint)


def create_diffusers_t5_model_from_checkpoint(
    cls,
    checkpoint,
    subfolder="",
    config=None,
    torch_dtype=None,
    local_files_only=None,
):
    from .single_file.single_file_utils import create_diffusers_t5_model_from_checkpoint

    deprecation_message = "Importing `create_diffusers_t5_model_from_checkpoint()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import create_diffusers_t5_model_from_checkpoint` instead."
    deprecate(
        "diffusers.loaders.single_file_utils.create_diffusers_t5_model_from_checkpoint", "0.36", deprecation_message
    )
    return create_diffusers_t5_model_from_checkpoint(cls, checkpoint, subfolder, config, torch_dtype, local_files_only)


def convert_animatediff_checkpoint_to_diffusers(checkpoint, **kwargs):
    from .single_file.single_file_utils import convert_animatediff_checkpoint_to_diffusers

    deprecation_message = "Importing `convert_animatediff_checkpoint_to_diffusers()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import convert_animatediff_checkpoint_to_diffusers` instead."
    deprecate(
        "diffusers.loaders.single_file_utils.convert_animatediff_checkpoint_to_diffusers", "0.36", deprecation_message
    )
    return convert_animatediff_checkpoint_to_diffusers(checkpoint, **kwargs)


def convert_flux_transformer_checkpoint_to_diffusers(checkpoint, **kwargs):
    from .single_file.single_file_utils import convert_flux_transformer_checkpoint_to_diffusers

    deprecation_message = "Importing `convert_flux_transformer_checkpoint_to_diffusers()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import convert_flux_transformer_checkpoint_to_diffusers` instead."
    deprecate(
        "diffusers.loaders.single_file_utils.convert_flux_transformer_checkpoint_to_diffusers",
        "0.36",
        deprecation_message,
    )
    return convert_flux_transformer_checkpoint_to_diffusers(checkpoint, **kwargs)


def convert_ltx_transformer_checkpoint_to_diffusers(checkpoint, **kwargs):
    from .single_file.single_file_utils import convert_ltx_transformer_checkpoint_to_diffusers

    deprecation_message = "Importing `convert_ltx_transformer_checkpoint_to_diffusers()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import convert_ltx_transformer_checkpoint_to_diffusers` instead."
    deprecate(
        "diffusers.loaders.single_file_utils.convert_ltx_transformer_checkpoint_to_diffusers",
        "0.36",
        deprecation_message,
    )
    return convert_ltx_transformer_checkpoint_to_diffusers(checkpoint, **kwargs)


def convert_ltx_vae_checkpoint_to_diffusers(checkpoint, **kwargs):
    from .single_file.single_file_utils import convert_ltx_vae_checkpoint_to_diffusers

    deprecation_message = "Importing `convert_ltx_vae_checkpoint_to_diffusers()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import convert_ltx_vae_checkpoint_to_diffusers` instead."
    deprecate(
        "diffusers.loaders.single_file_utils.convert_ltx_vae_checkpoint_to_diffusers", "0.36", deprecation_message
    )
    return convert_ltx_vae_checkpoint_to_diffusers(checkpoint, **kwargs)


def convert_autoencoder_dc_checkpoint_to_diffusers(checkpoint, **kwargs):
    from .single_file.single_file_utils import convert_autoencoder_dc_checkpoint_to_diffusers

    deprecation_message = "Importing `convert_autoencoder_dc_checkpoint_to_diffusers()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import convert_autoencoder_dc_checkpoint_to_diffusers` instead."
    deprecate(
        "diffusers.loaders.single_file_utils.convert_autoencoder_dc_checkpoint_to_diffusers",
        "0.36",
        deprecation_message,
    )
    return convert_autoencoder_dc_checkpoint_to_diffusers(checkpoint, **kwargs)


def convert_mochi_transformer_checkpoint_to_diffusers(checkpoint, **kwargs):
    from .single_file.single_file_utils import convert_mochi_transformer_checkpoint_to_diffusers

    deprecation_message = "Importing `convert_mochi_transformer_checkpoint_to_diffusers()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import convert_mochi_transformer_checkpoint_to_diffusers` instead."
    deprecate(
        "diffusers.loaders.single_file_utils.convert_mochi_transformer_checkpoint_to_diffusers",
        "0.36",
        deprecation_message,
    )
    return convert_mochi_transformer_checkpoint_to_diffusers(checkpoint, **kwargs)


def convert_hunyuan_video_transformer_to_diffusers(checkpoint, **kwargs):
    from .single_file.single_file_utils import convert_hunyuan_video_transformer_to_diffusers

    deprecation_message = "Importing `convert_hunyuan_video_transformer_to_diffusers()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import convert_hunyuan_video_transformer_to_diffusers` instead."
    deprecate(
        "diffusers.loaders.single_file_utils.convert_hunyuan_video_transformer_to_diffusers",
        "0.36",
        deprecation_message,
    )
    return convert_hunyuan_video_transformer_to_diffusers(checkpoint, **kwargs)


def convert_auraflow_transformer_checkpoint_to_diffusers(checkpoint, **kwargs):
    from .single_file.single_file_utils import convert_auraflow_transformer_checkpoint_to_diffusers

    deprecation_message = "Importing `convert_auraflow_transformer_checkpoint_to_diffusers()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import convert_auraflow_transformer_checkpoint_to_diffusers` instead."
    deprecate(
        "diffusers.loaders.single_file_utils.convert_auraflow_transformer_checkpoint_to_diffusers",
        "0.36",
        deprecation_message,
    )
    return convert_auraflow_transformer_checkpoint_to_diffusers(checkpoint, **kwargs)


def convert_lumina2_to_diffusers(checkpoint, **kwargs):
    from .single_file.single_file_utils import convert_lumina2_to_diffusers

    deprecation_message = "Importing `convert_lumina2_to_diffusers()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import convert_lumina2_to_diffusers` instead."
    deprecate("diffusers.loaders.single_file_utils.convert_lumina2_to_diffusers", "0.36", deprecation_message)
    return convert_lumina2_to_diffusers(checkpoint, **kwargs)


def convert_sana_transformer_to_diffusers(checkpoint, **kwargs):
    from .single_file.single_file_utils import convert_sana_transformer_to_diffusers

    deprecation_message = "Importing `convert_sana_transformer_to_diffusers()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import convert_sana_transformer_to_diffusers` instead."
    deprecate("diffusers.loaders.single_file_utils.convert_sana_transformer_to_diffusers", "0.36", deprecation_message)
    return convert_sana_transformer_to_diffusers(checkpoint, **kwargs)


def convert_wan_transformer_to_diffusers(checkpoint, **kwargs):
    from .single_file.single_file_utils import convert_wan_transformer_to_diffusers

    deprecation_message = "Importing `convert_wan_transformer_to_diffusers()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import convert_wan_transformer_to_diffusers` instead."
    deprecate("diffusers.loaders.single_file_utils.convert_wan_transformer_to_diffusers", "0.36", deprecation_message)
    return convert_wan_transformer_to_diffusers(checkpoint, **kwargs)


def convert_wan_vae_to_diffusers(checkpoint, **kwargs):
    from .single_file.single_file_utils import convert_wan_vae_to_diffusers

    deprecation_message = "Importing `convert_wan_vae_to_diffusers()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file_utils import convert_wan_vae_to_diffusers` instead."
    deprecate("diffusers.loaders.single_file_utils.convert_wan_vae_to_diffusers", "0.36", deprecation_message)
    return convert_wan_vae_to_diffusers(checkpoint, **kwargs)
