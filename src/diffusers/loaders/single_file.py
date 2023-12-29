# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import inspect
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import validate_hf_hub_args
from safetensors.torch import load_file as safe_load

from ..utils import (
    deprecate,
    is_accelerate_available,
    is_omegaconf_available,
    is_transformers_available,
    logging,
)
from ..utils.import_utils import BACKENDS_MAPPING
from .single_file_utils import (
    create_controlnet_model,
    create_paint_by_example_components,
    create_scheduler,
    create_stable_unclip_components,
    create_text_encoders_and_tokenizers,
    create_unet_model,
    create_vae_model,
    fetch_original_config,
    infer_model_type,
)


if is_transformers_available():
    pass

if is_accelerate_available():
    from accelerate import init_empty_weights

logger = logging.get_logger(__name__)


VALID_URL_PREFIXES = ["https://huggingface.co/", "huggingface.co/", "hf.co/", "https://hf.co/"]
TEXT_ENCODER_FROM_PIPELINE_CLASS = {
    "StableUnCLIPPipeline": "FrozenOpenCLIPEmbedder",
    "StableUnCLIPImg2ImgPipeline": "FrozenOpenCLIPEmbedder",
    "LDMTextToImagePipeline": "LDMTextToImage",
    "PaintByExamplePipeline": "PaintByExample",
    "StableDiffusion": "stable-diffusion",
}


def extract_pipeline_component_names(pipeline_class):
    components = inspect.signature(pipeline_class).parameters.keys()
    return components


def check_valid_url(pretrained_model_link_or_path):
    # remove huggingface url
    has_valid_url_prefix = False
    for prefix in VALID_URL_PREFIXES:
        if pretrained_model_link_or_path.startswith(prefix):
            pretrained_model_link_or_path = pretrained_model_link_or_path[len(prefix) :]
            has_valid_url_prefix = True

    return has_valid_url_prefix


def download_model_checkpoint(
    ckpt_path,
    cache_dir=None,
    resume_download=False,
    force_download=False,
    proxies=None,
    local_files_only=None,
    token=None,
    revision=None,
):
    # get repo_id and (potentially nested) file path of ckpt in repo
    repo_id = "/".join(ckpt_path.parts[:2])
    file_path = "/".join(ckpt_path.parts[2:])

    if file_path.startswith("blob/"):
        file_path = file_path[len("blob/") :]

    if file_path.startswith("main/"):
        file_path = file_path[len("main/") :]

    path = hf_hub_download(
        repo_id,
        filename=file_path,
        cache_dir=cache_dir,
        resume_download=resume_download,
        proxies=proxies,
        local_files_only=local_files_only,
        token=token,
        revision=revision,
        force_download=force_download,
    )

    return path


def load_checkpoint(checkpoint_path_or_dict, device=None, from_safetensors=True):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(checkpoint_path_or_dict, str):
        if from_safetensors:
            checkpoint = safe_load(checkpoint_path_or_dict, device="cpu")

        else:
            checkpoint = torch.load(checkpoint_path_or_dict, map_location=device)

    elif isinstance(checkpoint_path_or_dict, dict):
        checkpoint = checkpoint_path_or_dict

    return checkpoint


def build_component(
    pipeline_components,
    pipeline_class_name,
    component_name,
    original_config,
    checkpoint,
    checkpoint_path_or_dict,
    **kwargs,
):
    if component_name in kwargs:
        return kwargs.pop(component_name, None)

    if component_name in pipeline_components:
        return {}

    if component_name == "unet":
        unet_components = create_unet_model(
            pipeline_class_name, original_config, checkpoint, checkpoint_path_or_dict, **kwargs
        )
        return unet_components

    if component_name == "vae":
        vae_components = create_vae_model(
            pipeline_class_name, original_config, checkpoint, checkpoint_path_or_dict, **kwargs
        )
        return vae_components

    if component_name == "controlnet":
        controlnet_components = create_controlnet_model(
            pipeline_class_name, original_config, checkpoint, checkpoint_path_or_dict, **kwargs
        )
        return controlnet_components

    if component_name == "scheduler":
        scheduler_components = create_scheduler(
            pipeline_class_name, original_config, checkpoint, checkpoint_path_or_dict, **kwargs
        )
        return scheduler_components

    if component_name in ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"]:
        text_encoder_components = create_text_encoders_and_tokenizers(
            pipeline_class_name, original_config, checkpoint, checkpoint_path_or_dict, **kwargs
        )
        return text_encoder_components

    return


def build_additional_components(
    pipeline_components,
    pipeline_class_name,
    component_name,
    original_config,
    checkpoint,
    checkpoint_path_or_dict,
    **kwargs,
):
    if component_name in kwargs:
        return kwargs.pop(component_name, None)

    if component_name in pipeline_components:
        return None

    if pipeline_class_name == ["StableUnCLIPPipeline", "StableUnCLIPImg2ImgPipeline"]:
        stable_unclip_components = create_stable_unclip_components(
            pipeline_class_name, original_config, checkpoint, checkpoint_path_or_dict, **kwargs
        )
        return stable_unclip_components

    if pipeline_class_name == "PaintByExamplePipeline":
        paint_by_example_components = create_paint_by_example_components(
            pipeline_class_name, original_config, checkpoint, checkpoint_path_or_dict, **kwargs
        )
        return paint_by_example_components

    if pipeline_class_name in ["StableDiffusionXLImg2ImgPipeline", "StableDiffusionXLInpaintPipeline"]:
        model_type = infer_model_type(pipeline_class_name, original_config)
        is_refiner = model_type == "SDXL-Refiner"
        return {
            "requires_aesthetics_score": is_refiner,
            "force_zeros_for_empty_prompt": False if is_refiner else True,
        }


class FromSingleFileMixin:
    """
    Load model weights saved in the `.ckpt` format into a [`DiffusionPipeline`].
    """

    @classmethod
    @validate_hf_hub_args
    def from_single_file(cls, pretrained_model_link_or_path, **kwargs):
        r"""
        Instantiate a [`DiffusionPipeline`] from pretrained pipeline weights saved in the `.ckpt` or `.safetensors`
        format. The pipeline is set in evaluation mode (`model.eval()`) by default.

        Parameters:
            pretrained_model_link_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:
                    - A link to the `.ckpt` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.ckpt"`) on the Hub.
                    - A path to a *file* containing all pipeline weights.
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If `"auto"` is passed, the
                dtype is automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
        Examples:

        ```py
        >>> from diffusers import StableDiffusionPipeline

        >>> # Download pipeline from huggingface.co and cache.
        >>> pipeline = StableDiffusionPipeline.from_single_file(
        ...     "https://huggingface.co/WarriorMama777/OrangeMixs/blob/main/Models/AbyssOrangeMix/AbyssOrangeMix.safetensors"
        ... )

        >>> # Download pipeline from local file
        >>> # file is downloaded under ./v1-5-pruned-emaonly.ckpt
        >>> pipeline = StableDiffusionPipeline.from_single_file("./v1-5-pruned-emaonly")

        >>> # Enable float16 and move to GPU
        >>> pipeline = StableDiffusionPipeline.from_single_file(
        ...     "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt",
        ...     torch_dtype=torch.float16,
        ... )
        >>> pipeline.to("cuda")
        ```
        """
        original_config_file = kwargs.pop("original_config_file", None)
        config_files = kwargs.pop("config_files", None)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        cache_dir = kwargs.pop("cache_dir", None)
        local_files_only = kwargs.pop("local_files_only", None)
        revision = kwargs.pop("revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        pipeline_name = cls.__name__
        file_extension = pretrained_model_link_or_path.rsplit(".", 1)[-1]
        from_safetensors = file_extension == "safetensors"

        if from_safetensors and use_safetensors is False:
            raise ValueError("Make sure to install `safetensors` with `pip install safetensors`.")

        has_valid_url_prefix = check_valid_url(pretrained_model_link_or_path)

        # Code based on diffusers.pipelines.pipeline_utils.DiffusionPipeline.from_pretrained
        ckpt_path = Path(pretrained_model_link_or_path)
        if (not ckpt_path.is_file()) and (not has_valid_url_prefix):
            raise ValueError(
                f"The provided path is either not a file or a valid huggingface URL was not provided. Valid URLs begin with {', '.join(VALID_URL_PREFIXES)}"
            )
        if not ckpt_path.is_file():
            pretrained_model_link_or_path = download_model_checkpoint(
                ckpt_path,
                cache_dir=cache_dir,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
            )
            checkpoint = load_checkpoint(pretrained_model_link_or_path, from_safetensors=from_safetensors)
        else:
            checkpoint = load_checkpoint(pretrained_model_link_or_path, from_safetensors=from_safetensors)

        # NOTE: this while loop isn't great but this controlnet checkpoint has one additional
        # "state_dict" key https://huggingface.co/thibaud/controlnet-canny-sd21
        while "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        original_config = fetch_original_config(checkpoint, original_config_file, config_files)
        component_names = extract_pipeline_component_names(cls)

        pipeline_components = {}
        for component in component_names:
            components = build_component(
                pipeline_components,
                pipeline_name,
                component,
                original_config,
                checkpoint,
                pretrained_model_link_or_path,
                **kwargs,
            )
            if not components:
                continue
            pipeline_components.update(components)

        additional_components = set(pipeline_components.keys() - component_names)
        if additional_components:
            components = build_additional_components(pipeline_name, component, checkpoint, original_config, **kwargs)
            pipeline_components.update(components)

        pipe = cls(**pipeline_components)

        if torch_dtype is not None:
            pipe.to(dtype=torch_dtype)

        return pipe
