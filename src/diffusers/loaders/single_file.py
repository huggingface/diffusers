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

import importlib

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import validate_hf_hub_args

from ..utils import logging
from .single_file_utils import (
    fetch_diffusers_config,
    fetch_original_config,
    load_single_file_model_checkpoint,
)


logger = logging.get_logger(__name__)


def load_single_file_sub_model(
    library_name,
    class_name,
    pretrained_model_name_or_path,
    name,
    checkpoint,
    pipelines,
    is_pipeline_module,
    original_config=None,
    local_files_only=False,
    torch_dtype=None,
    **kwargs,
):
    if is_pipeline_module:
        pipeline_module = getattr(pipelines, library_name)
        class_obj = getattr(pipeline_module, class_name)
    else:
        # else we just import it from the library.
        library = importlib.import_module(library_name)
        class_obj = getattr(library, class_name)

    diffusers_module = importlib.import_module(__name__.split(".")[0])
    is_single_file_model = issubclass(class_obj, diffusers_module.FromOriginalModelMixin)

    if is_single_file_model:
        load_method = getattr(class_obj, "from_single_file")
        if original_config is None:
            config = class_obj.load_config(
                pretrained_model_name_or_path, subfolder=name, local_files_only=local_files_only
            )
        else:
            config = None

        loaded_sub_model = load_method(
            checkpoint=checkpoint,
            original_config=original_config,
            config=config,
            torch_dtype=torch_dtype,
            **kwargs,
        )

    else:
        load_method = getattr(class_obj, "from_pretrained")
        loaded_sub_model = load_method(
            pretrained_model_name_or_path,
            subfolder=name,
            local_files_only=local_files_only,
            torch_dtype=torch_dtype,
            **kwargs,
        )

    return loaded_sub_model


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
        load_connected_pipeline = kwargs.pop("load_connected_pipeline", False)
        local_dir = kwargs.pop("local_dir", None)
        local_dir_use_symlinks = kwargs.pop("local_dir_use_symlinks", "auto")

        checkpoint = load_single_file_model_checkpoint(
            pretrained_model_link_or_path,
            resume_download=resume_download,
            force_download=force_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
        )
        if original_config_file is not None:
            original_config = fetch_original_config(
                checkpoint, original_config_file, local_files_only=local_files_only
            )
        else:
            original_config = None

        from ..pipelines.pipeline_utils import _get_pipeline_class

        pipeline_class = _get_pipeline_class(
            cls,
            config=None,
            cache_dir=cache_dir,
            load_connected_pipeline=load_connected_pipeline,
        )
        default_pipeline_config = fetch_diffusers_config(checkpoint)
        config_file = hf_hub_download(
            default_pipeline_config["pretrained_model_name_or_path"],
            filename=cls.config_name,
            cache_dir=cache_dir,
            revision=revision,
            proxies=proxies,
            force_download=force_download,
            resume_download=resume_download,
            token=token,
            local_files_only=local_files_only,
            local_dir=local_dir,
            local_dir_use_symlinks=local_dir_use_symlinks,
        )

        config_dict = pipeline_class._dict_from_json_file(config_file)
        # pop out "_ignore_files" as it is only needed for download
        config_dict.pop("_ignore_files", None)

        expected_modules, optional_kwargs = pipeline_class._get_signature_keys(cls)
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}

        init_dict, unused_kwargs, _ = pipeline_class.extract_init_dict(config_dict, **kwargs)

        from diffusers import pipelines

        init_kwargs = {
            k: init_dict.pop(k) for k in optional_kwargs if k in init_dict and k not in cls._optional_components
        }
        init_kwargs = {**init_kwargs, **passed_pipe_kwargs}

        for name, (library_name, class_name) in logging.tqdm(init_dict.items(), desc="Loading pipeline components..."):
            loaded_sub_model = None
            is_pipeline_module = hasattr(pipelines, library_name)

            if name in passed_class_obj:
                loaded_sub_model = passed_class_obj[name]

            else:
                loaded_sub_model = load_single_file_sub_model(
                    checkpoint=checkpoint,
                    library_name=library_name,
                    class_name=class_name,
                    pretrained_model_name_or_path=default_pipeline_config["pretrained_model_name_or_path"],
                    is_pipeline_module=is_pipeline_module,
                    pipelines=pipelines,
                    name=name,
                    torch_dtype=torch_dtype,
                    original_config=original_config,
                    **kwargs,
                )

            init_kwargs[name] = loaded_sub_model

        missing_modules = set(expected_modules) - set(init_kwargs.keys())
        passed_modules = list(passed_class_obj.keys())
        optional_modules = pipeline_class._optional_components

        if len(missing_modules) > 0 and missing_modules <= set(passed_modules + optional_modules):
            for module in missing_modules:
                init_kwargs[module] = passed_class_obj.get(module, None)
        elif len(missing_modules) > 0:
            passed_modules = set(list(init_kwargs.keys()) + list(passed_class_obj.keys())) - optional_kwargs
            raise ValueError(
                f"Pipeline {pipeline_class} expected {expected_modules}, but only {passed_modules} were passed."
            )

        pipe = pipeline_class(**init_kwargs)
        if torch_dtype is not None:
            pipe.to(dtype=torch_dtype)

        return pipe
