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
import os
import re

from huggingface_hub.utils import validate_hf_hub_args
from transformers import AutoFeatureExtractor

from ..models.modeling_utils import load_state_dict
from ..pipelines.pipeline_utils import _get_pipeline_class
from ..pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from ..utils import (
    is_accelerate_available,
    is_transformers_available,
    logging,
)
from ..utils.hub_utils import _get_model_file
from .single_file_utils import (
    create_diffusers_controlnet_model_from_ldm,
    create_diffusers_unet_model_from_ldm,
    create_diffusers_vae_model_from_ldm,
    create_scheduler_from_ldm,
    create_text_encoders_and_tokenizers_from_ldm,
    fetch_original_config,
    infer_model_type,
)


if is_transformers_available():
    pass

if is_accelerate_available():
    pass

logger = logging.get_logger(__name__)


VALID_URL_PREFIXES = ["https://huggingface.co/", "huggingface.co/", "hf.co/", "https://hf.co/"]
# Pipelines that support the SDXL Refiner checkpoint
REFINER_PIPELINES = [
    "StableDiffusionXLImg2ImgPipeline",
    "StableDiffusionXLInpaintPipeline",
    "StableDiffusionXLControlNetImg2ImgPipeline",
]


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


def build_sub_model_components(
    pipeline_components,
    pipeline_class_name,
    component_name,
    original_config,
    checkpoint,
    checkpoint_path_or_dict,
    local_files_only=False,
    load_safety_checker=False,
    **kwargs,
):
    if component_name in pipeline_components:
        return {}

    model_type = kwargs.get("model_type", None)
    image_size = kwargs.pop("image_size", None)

    if component_name == "unet":
        num_in_channels = kwargs.pop("num_in_channels", None)
        unet_components = create_diffusers_unet_model_from_ldm(
            pipeline_class_name, original_config, checkpoint, num_in_channels=num_in_channels, image_size=image_size
        )
        return unet_components

    if component_name == "vae":
        vae_components = create_diffusers_vae_model_from_ldm(
            pipeline_class_name, original_config, checkpoint, image_size
        )
        return vae_components

    if component_name == "scheduler":
        scheduler_type = kwargs.get("scheduler_type", "ddim")
        prediction_type = kwargs.get("prediction_type", None)

        scheduler_components = create_scheduler_from_ldm(
            pipeline_class_name,
            original_config,
            checkpoint,
            scheduler_type=scheduler_type,
            prediction_type=prediction_type,
            model_type=model_type,
        )

        return scheduler_components

    if component_name in ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"]:
        text_encoder_components = create_text_encoders_and_tokenizers_from_ldm(
            original_config,
            checkpoint,
            model_type=model_type,
            local_files_only=local_files_only,
        )
        return text_encoder_components

    if component_name == "safety_checker":
        if load_safety_checker:
            safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker", local_files_only=local_files_only
            )
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                "CompVis/stable-diffusion-safety-checker", local_files_only=local_files_only
            )
        else:
            safety_checker = None
            feature_extractor = None

        return {"safety_checker": safety_checker, "feature_extractor": feature_extractor}

    return


def set_additional_components(
    pipeline_class_name,
    original_config,
    **kwargs,
):
    components = {}
    model_type = kwargs.get("model_type", None)
    if pipeline_class_name in REFINER_PIPELINES:
        model_type = infer_model_type(original_config, model_type=model_type)
        is_refiner = model_type == "SDXL-Refiner"
        components.update(
            {
                "requires_aesthetics_score": is_refiner,
                "force_zeros_for_empty_prompt": False if is_refiner else True,
            }
        )

    return components


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
                Override the default `torch.dtype` and load the model with another dtype.
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
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        cache_dir = kwargs.pop("cache_dir", None)
        local_files_only = kwargs.pop("local_files_only", None)
        revision = kwargs.pop("revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        use_safetensors = kwargs.pop("use_safetensors", True)

        class_name = cls.__name__
        file_extension = pretrained_model_link_or_path.rsplit(".", 1)[-1]
        from_safetensors = file_extension == "safetensors"

        if from_safetensors and use_safetensors is False:
            raise ValueError("Make sure to install `safetensors` with `pip install safetensors`.")

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

        original_config = fetch_original_config(class_name, checkpoint, original_config_file, config_files)

        if class_name == "AutoencoderKL":
            image_size = kwargs.pop("image_size", None)
            component = create_diffusers_vae_model_from_ldm(
                class_name, original_config, checkpoint, image_size=image_size
            )
            return component["vae"]

        if class_name == "ControlNetModel":
            upcast_attention = kwargs.pop("upcast_attention", False)
            image_size = kwargs.pop("image_size", None)

            component = create_diffusers_controlnet_model_from_ldm(
                class_name, original_config, checkpoint, upcast_attention=upcast_attention, image_size=image_size
            )
            return component["controlnet"]

        pipeline_class = _get_pipeline_class(
            cls,
            config=None,
            cache_dir=cache_dir,
        )

        expected_modules, optional_kwargs = cls._get_signature_keys(pipeline_class)
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}

        init_kwargs = {}
        for name in expected_modules:
            if name in passed_class_obj:
                init_kwargs[name] = passed_class_obj[name]
            else:
                components = build_sub_model_components(
                    init_kwargs,
                    class_name,
                    name,
                    original_config,
                    checkpoint,
                    pretrained_model_link_or_path,
                    **kwargs,
                )
                if not components:
                    continue
                init_kwargs.update(components)

        additional_components = set_additional_components(class_name, original_config, **kwargs)
        if additional_components:
            init_kwargs.update(additional_components)

        init_kwargs.update(passed_pipe_kwargs)
        pipe = pipeline_class(**init_kwargs)

        if torch_dtype is not None:
            pipe.to(dtype=torch_dtype)

        return pipe
