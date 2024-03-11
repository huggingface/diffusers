# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from huggingface_hub.utils import validate_hf_hub_args

from ..utils import is_transformers_available, logging
from .single_file_utils import (
    create_diffusers_unet_model_from_ldm,
    create_diffusers_vae_model_from_ldm,
    create_scheduler_from_ldm,
    create_text_encoders_and_tokenizers_from_ldm,
    fetch_ldm_config_and_checkpoint,
    infer_model_type,
)


logger = logging.get_logger(__name__)

# Pipelines that support the SDXL Refiner checkpoint
REFINER_PIPELINES = [
    "StableDiffusionXLImg2ImgPipeline",
    "StableDiffusionXLInpaintPipeline",
    "StableDiffusionXLControlNetImg2ImgPipeline",
]

if is_transformers_available():
    from transformers import AutoFeatureExtractor


def build_sub_model_components(
    pipeline_components,
    pipeline_class_name,
    component_name,
    original_config,
    checkpoint,
    local_files_only=False,
    load_safety_checker=False,
    model_type=None,
    image_size=None,
    torch_dtype=None,
    **kwargs,
):
    if component_name in pipeline_components:
        return {}

    if component_name == "unet":
        num_in_channels = kwargs.pop("num_in_channels", None)
        unet_components = create_diffusers_unet_model_from_ldm(
            pipeline_class_name,
            original_config,
            checkpoint,
            num_in_channels=num_in_channels,
            image_size=image_size,
            torch_dtype=torch_dtype,
            model_type=model_type,
        )
        return unet_components

    if component_name == "vae":
        scaling_factor = kwargs.get("scaling_factor", None)
        vae_components = create_diffusers_vae_model_from_ldm(
            pipeline_class_name,
            original_config,
            checkpoint,
            image_size,
            scaling_factor,
            torch_dtype,
            model_type=model_type,
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
            torch_dtype=torch_dtype,
        )
        return text_encoder_components

    if component_name == "safety_checker":
        if load_safety_checker:
            from ..pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

            safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker", local_files_only=local_files_only, torch_dtype=torch_dtype
            )
        else:
            safety_checker = None
        return {"safety_checker": safety_checker}

    if component_name == "feature_extractor":
        if load_safety_checker:
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                "CompVis/stable-diffusion-safety-checker", local_files_only=local_files_only
            )
        else:
            feature_extractor = None
        return {"feature_extractor": feature_extractor}

    return


def set_additional_components(
    pipeline_class_name,
    original_config,
    checkpoint=None,
    model_type=None,
):
    components = {}
    if pipeline_class_name in REFINER_PIPELINES:
        model_type = infer_model_type(original_config, checkpoint=checkpoint, model_type=model_type)
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
            original_config_file (`str`, *optional*):
                The path to the original config file that was used to train the model. If not provided, the config file
                will be inferred from the checkpoint file.
            model_type (`str`, *optional*):
                The type of model to load. If not provided, the model type will be inferred from the checkpoint file.
            image_size (`int`, *optional*):
                The size of the image output. It's used to configure the `sample_size` parameter of the UNet and VAE model.
            load_safety_checker (`bool`, *optional*, defaults to `False`):
                Whether to load the safety checker model or not. By default, the safety checker is not loaded unless a `safety_checker` component is passed to the `kwargs`.
            num_in_channels (`int`, *optional*):
                Specify the number of input channels for the UNet model. Read more about how to configure UNet model with this parameter
                [here](https://huggingface.co/docs/diffusers/training/adapt_a_model#configure-unet2dconditionmodel-parameters).
            scaling_factor (`float`, *optional*):
                The scaling factor to use for the VAE model. If not provided, it is inferred from the config file first.
                If the scaling factor is not found in the config file, the default value 0.18215 is used.
            scheduler_type (`str`, *optional*):
                The type of scheduler to load. If not provided, the scheduler type will be inferred from the checkpoint file.
            prediction_type (`str`, *optional*):
                The type of prediction to load. If not provided, the prediction type will be inferred from the checkpoint file.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.

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
        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        cache_dir = kwargs.pop("cache_dir", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)

        class_name = cls.__name__

        original_config, checkpoint = fetch_ldm_config_and_checkpoint(
            pretrained_model_link_or_path=pretrained_model_link_or_path,
            class_name=class_name,
            original_config_file=original_config_file,
            resume_download=resume_download,
            force_download=force_download,
            proxies=proxies,
            token=token,
            revision=revision,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )

        from ..pipelines.pipeline_utils import _get_pipeline_class

        pipeline_class = _get_pipeline_class(
            cls,
            config=None,
            cache_dir=cache_dir,
        )

        expected_modules, optional_kwargs = cls._get_signature_keys(pipeline_class)
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}

        model_type = kwargs.pop("model_type", None)
        image_size = kwargs.pop("image_size", None)
        load_safety_checker = (kwargs.pop("load_safety_checker", False)) or (
            passed_class_obj.get("safety_checker", None) is not None
        )

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
                    model_type=model_type,
                    image_size=image_size,
                    load_safety_checker=load_safety_checker,
                    local_files_only=local_files_only,
                    torch_dtype=torch_dtype,
                    **kwargs,
                )
                if not components:
                    continue
                init_kwargs.update(components)

        additional_components = set_additional_components(class_name, original_config, model_type=model_type)
        if additional_components:
            init_kwargs.update(additional_components)

        init_kwargs.update(passed_pipe_kwargs)
        pipe = pipeline_class(**init_kwargs)

        if torch_dtype is not None:
            pipe.to(dtype=torch_dtype)

        return pipe
