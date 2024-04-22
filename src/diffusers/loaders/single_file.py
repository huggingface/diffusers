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
import os

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError, validate_hf_hub_args
from packaging import version

from ..utils import deprecate, is_transformers_available, logging
from .single_file_utils import (
    _is_model_weights_in_cached_folder,
    _legacy_load_clip_tokenizer,
    _legacy_load_safety_checker,
    _legacy_load_scheduler,
    create_diffusers_clip_model_from_ldm,
    fetch_diffusers_config,
    fetch_original_config,
    is_clip_model_in_single_file,
    load_single_file_checkpoint,
)


logger = logging.get_logger(__name__)

if is_transformers_available():
    import transformers
    from transformers import PreTrainedModel, PreTrainedTokenizer


def load_single_file_sub_model(
    library_name,
    class_name,
    pretrained_model_name_or_path,
    name,
    checkpoint,
    pipelines,
    is_pipeline_module,
    cached_model_path,
    original_config=None,
    local_files_only=False,
    torch_dtype=None,
    use_safetensors=None,
    is_legacy_loading=False,
    cache_dir=None,
    **kwargs,
):
    if is_pipeline_module:
        pipeline_module = getattr(pipelines, library_name)
        class_obj = getattr(pipeline_module, class_name)
    else:
        # else we just import it from the library.
        library = importlib.import_module(library_name)
        class_obj = getattr(library, class_name)

    if is_transformers_available():
        transformers_version = version.parse(version.parse(transformers.__version__).base_version)
    else:
        transformers_version = "N/A"

    is_transformers_model = (
        is_transformers_available()
        and issubclass(class_obj, PreTrainedModel)
        and transformers_version >= version.parse("4.20.0")
    )
    is_tokenizer = (
        is_transformers_available()
        and issubclass(class_obj, PreTrainedTokenizer)
        and transformers_version >= version.parse("4.20.0")
    )

    diffusers_module = importlib.import_module(__name__.split(".")[0])
    is_diffusers_single_file_model = issubclass(class_obj, diffusers_module.FromOriginalModelMixin)
    is_diffusers_model = issubclass(class_obj, diffusers_module.ModelMixin)
    is_diffusers_scheduler = issubclass(class_obj, diffusers_module.SchedulerMixin)

    if is_diffusers_single_file_model:
        load_method = getattr(class_obj, "from_single_file")

        # We cannot provide two different config options to the `from_single_file` method
        # Here we have to ignore loading the config from `pretrained_model_name_or_path` if `original_config` is provided
        if original_config:
            pretrained_model_name_or_path = None

        loaded_sub_model = load_method(
            checkpoint=checkpoint,
            original_config=original_config,
            config=pretrained_model_name_or_path,
            subfolder=name,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            **kwargs,
        )

    elif is_transformers_model and is_clip_model_in_single_file(class_obj, checkpoint):
        loaded_sub_model = create_diffusers_clip_model_from_ldm(
            class_obj,
            checkpoint=checkpoint,
            config=pretrained_model_name_or_path,
            subfolder=name,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            is_legacy_loading=is_legacy_loading,
        )

    elif is_tokenizer and is_legacy_loading:
        loaded_sub_model = _legacy_load_clip_tokenizer(
            class_obj,
            config=pretrained_model_name_or_path,
            checkpoint=checkpoint,
        )

    elif is_diffusers_scheduler and is_legacy_loading:
        scheduler_type = kwargs.get("scheduler_type", None)
        prediction_type = kwargs.get("prediction_type", None)

        loaded_sub_model = _legacy_load_scheduler(
            class_obj,
            checkpoint=checkpoint,
            name=name,
            original_config=original_config,
            scheduler_type=scheduler_type,
            prediction_type=prediction_type,
        )

    else:
        if not hasattr(class_obj, "from_pretrained"):
            raise ValueError(
                (
                    f"The component {class_obj.__name__} cannot be loaded as it does not seem to have"
                    " a supported loading method."
                )
            )

        loading_kwargs = {}
        loading_kwargs.update(
            {
                "pretrained_model_name_or_path": cached_model_path,
                "subfolder": name,
                "local_files_only": local_files_only,
                "use_safetensors": use_safetensors,
            }
        )

        # Schedulers and Tokenizers don't make use of torch_dtype
        # Skip passing it to those objects
        if issubclass(class_obj, torch.nn.Module):
            loading_kwargs.update({"torch_dtype": torch_dtype})

        if is_diffusers_model:
            logger.warning(
                (
                    f"Pipeline component {name}'s weights do not appear to be included in the checkpoint"
                    f" or {class_obj.__name__} does not currently support single file loading."
                    f"Attempting to download the component using `from_pretrained` and the inferred model repository: {pretrained_model_name_or_path}."
                )
            )
            if not _is_model_weights_in_cached_folder(cached_model_path, name):
                loading_kwargs.update(
                    {
                        "pretrained_model_name_or_path": pretrained_model_name_or_path,
                        "local_files_only": False,
                        "cache_dir": cache_dir,
                    }
                )

        load_method = getattr(class_obj, "from_pretrained")
        loaded_sub_model = load_method(**loading_kwargs)

    return loaded_sub_model


def _map_component_types_to_config_dict(component_types):
    diffusers_module = importlib.import_module(__name__.split(".")[0])
    config_dict = {}
    component_types.pop("self", None)

    if is_transformers_available():
        transformers_version = version.parse(version.parse(transformers.__version__).base_version)
    else:
        transformers_version = "N/A"

    for component_name, component_value in component_types.items():
        is_diffusers_model = issubclass(component_value[0], diffusers_module.ModelMixin)
        is_scheduler_enum = component_value[0].__name__ == "KarrasDiffusionSchedulers"
        is_scheduler = issubclass(component_value[0], diffusers_module.SchedulerMixin)

        is_transformers_model = (
            is_transformers_available()
            and issubclass(component_value[0], PreTrainedModel)
            and transformers_version >= version.parse("4.20.0")
        )
        is_transformers_tokenizer = (
            is_transformers_available()
            and issubclass(component_value[0], PreTrainedTokenizer)
            and transformers_version >= version.parse("4.20.0")
        )

        if is_diffusers_model:
            config_dict[component_name] = ["diffusers", component_value[0].__name__]

        elif is_scheduler_enum or is_scheduler:
            if is_scheduler_enum:
                # Since we cannot fetch a scheduler config from the hub, we default to DDIMScheduler
                # if the type hint is a KarrassDiffusionSchedulers enum
                config_dict[component_name] = ["diffusers", "DDIMScheduler"]

            elif is_scheduler:
                config_dict[component_name] = ["diffusers", component_value[0].__name__]

        elif is_transformers_model or is_transformers_tokenizer:
            config_dict[component_name] = ["transformers", component_value[0].__name__]

        elif is_transformers_model and component_value[0].__name__ == "StableDiffusionSafetyChecker":
            config_dict[component_name] = ["stable_diffusion", component_value[0].__name__]

        else:
            config_dict[component_name] = [None, None]

    return config_dict


def _download_diffusers_model_config_from_hub(
    default_pretrained_model_name_or_path,
    cache_dir,
    revision,
    proxies,
    force_download=None,
    resume_download=None,
    local_files_only=None,
    token=None,
):
    allow_patterns = ["**/*.json", "*.json", "*.txt", "**/*.txt"]
    cached_model_path = snapshot_download(
        default_pretrained_model_name_or_path,
        cache_dir=cache_dir,
        revision=revision,
        proxies=proxies,
        force_download=force_download,
        resume_download=resume_download,
        local_files_only=local_files_only,
        token=token,
        allow_patterns=allow_patterns,
    )

    return cached_model_path


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
            local_dir (`str`, *optional*):
                If provided, the downloaded file will be placed under this directory, either as a symlink (default) or
                a regular file.
            local_dir_use_symlinks (`str` or `bool`, *optional*):
                To be used with local_dir. If set to “auto”, the cache directory will be used and the file will either
                be duplicated or symlinked to the local directory depending on its size. It set to True, a symlink will
                be created, no matter the file size. If set to False, the file will either be duplicated from cache (if
                already exists) or downloaded from the Hub and not cached. Find more information in the
                `huggingface_hub`
                [documentation](https://huggingface.co/docs/huggingface_hub/package_reference/file_download#huggingface_hub.hf_hub_download.local_dir_use_symlinks).
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            original_config_file (`str`, *optional*):
                The path to the original config file that was used to train the model. If not provided, the config file
                will be inferred from the checkpoint file.
            config (`str`, *optional*):
                Can be either:
                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing the pipeline
                      component configs in Diffusers format.
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
        >>> pipeline = StableDiffusionPipeline.from_single_file("./v1-5-pruned-emaonly.ckpt")

        >>> # Enable float16 and move to GPU
        >>> pipeline = StableDiffusionPipeline.from_single_file(
        ...     "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt",
        ...     torch_dtype=torch.float16,
        ... )
        >>> pipeline.to("cuda")
        ```
        """
        original_config_file = kwargs.pop("original_config_file", None)
        config = kwargs.pop("config", None)
        original_config = kwargs.pop("original_config", None)

        if original_config_file is not None:
            deprecation_message = (
                "`original_config_file` argument is deprecated and will be removed in future versions."
                "please use the `original_config` argument instead."
            )
            deprecate("original_config_file", "1.0.0", deprecation_message)
            original_config = original_config_file

        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        cache_dir = kwargs.pop("cache_dir", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        local_dir = kwargs.pop("local_dir", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        is_legacy_loading = False

        # We shouldn't allow configuring individual models components through a Pipeline creation method
        # These model kwargs should be deprecated
        scaling_factor = kwargs.get("scaling_factor", None)
        if scaling_factor is not None:
            deprecation_message = (
                "Passing the `scaling_factor` argument to `from_single_file is deprecated "
                "and will be ignored in future versions."
            )
            deprecate("scaling_factor", "1.0.0", deprecation_message)

        prediction_type = kwargs.get("prediction_type", None)
        if prediction_type is not None:
            deprecation_message = (
                "Passing the `prediction_type` argument to `from_single_file is deprecated "
                "and will be ignored in future versions."
            )
            deprecate("prediction_type", "1.0.0", deprecation_message)

        if original_config is not None:
            original_config = fetch_original_config(original_config, local_files_only=local_files_only)

        from ..pipelines.pipeline_utils import _get_pipeline_class

        pipeline_class = _get_pipeline_class(cls, config=None)

        checkpoint = load_single_file_checkpoint(
            pretrained_model_link_or_path,
            resume_download=resume_download,
            force_download=force_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            local_dir=local_dir,
        )

        # Always infer the default diffusers pretrained_model config from the checkpoint
        # although it might not be used to configure the pipeline
        config = {"pretrained_model_name_or_path": config} if config else fetch_diffusers_config(checkpoint)
        default_pretrained_model_name_or_path = config["pretrained_model_name_or_path"]

        if not os.path.isdir(default_pretrained_model_name_or_path):
            # Provided config is a repo_id
            if default_pretrained_model_name_or_path.count("/") > 1:
                raise ValueError(
                    f'The provided pretrained_model_name_or_path "{default_pretrained_model_name_or_path}"'
                    " is neither a valid local path nor a valid repo id. Please check the parameter."
                )
            try:
                # Attempt to download the config files for the pipeline
                cached_model_path = _download_diffusers_model_config_from_hub(
                    default_pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    revision=revision,
                    proxies=proxies,
                    force_download=force_download,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                )
                config_dict = pipeline_class.load_config(cached_model_path)

            except LocalEntryNotFoundError:
                # In this path, `local_files_only=True` but a local diffusers format model config is not available in the cache
                # If `original_config` is not provided, we need to force download the config files from hub so that we have a way
                # to configure the pipeline components.

                # If `original_config` is provided, then we need to assume we are using legacy loading of pipeline components to preserve
                # backwards compatibility
                if original_config is None:
                    logger.warning(
                        "`local_files_only` is True but no local configs were found for this checkpoint.\n"
                        "Attempting to download the necessary config files for this pipeline.\n"
                    )
                    cached_model_path = _download_diffusers_model_config_from_hub(
                        default_pretrained_model_name_or_path,
                        cache_dir=cache_dir,
                        revision=revision,
                        proxies=proxies,
                        force_download=force_download,
                        resume_download=resume_download,
                        local_files_only=False,
                        token=token,
                    )
                    config_dict = pipeline_class.load_config(cached_model_path)

                else:
                    logger.warning(
                        "Detected legacy `from_single_file` loading behavior. Attempting to create the pipeline based on inferred components.\n"
                        "This may lead to errors if the model components are not correctly inferred. "
                        "To avoid this warning, please explicity pass the `config` argument to `from_single_file` with a path to a local diffusers model repo "
                        "or run `from_single_file` with `local_files_only=False` first to update the local cache directory with "
                        "the necessary config files.\n"
                    )
                    is_legacy_loading = True
                    cached_model_path = None

                    component_types = pipeline_class._get_signature_types()
                    expected_modules, optional_kwargs = pipeline_class._get_signature_keys(cls)
                    component_types = {
                        k: v for k, v in component_types.items() if k not in pipeline_class._optional_components
                    }

                    config_dict = _map_component_types_to_config_dict(component_types)
                    config_dict["_class_name"] = pipeline_class.__name__

        else:
            # Provide config is a path to a local directory attempt to load directly.
            cached_model_path = default_pretrained_model_name_or_path
            config_dict = pipeline_class.load_config(cached_model_path)

        #   pop out "_ignore_files" as it is only needed for download
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

        # remove `null` components
        def load_module(name, value):
            if value[0] is None:
                return False
            if name in passed_class_obj and passed_class_obj[name] is None:
                return False
            return True

        init_dict = {k: v for k, v in init_dict.items() if load_module(k, v)}

        for name, (library_name, class_name) in logging.tqdm(
            sorted(init_dict.items()), desc="Loading pipeline components..."
        ):
            loaded_sub_model = None
            is_pipeline_module = hasattr(pipelines, library_name)

            if name in passed_class_obj:
                loaded_sub_model = passed_class_obj[name]

            else:
                loaded_sub_model = load_single_file_sub_model(
                    checkpoint=checkpoint,
                    library_name=library_name,
                    class_name=class_name,
                    pretrained_model_name_or_path=default_pretrained_model_name_or_path,
                    is_pipeline_module=is_pipeline_module,
                    cached_model_path=cached_model_path,
                    pipelines=pipelines,
                    name=name,
                    cache_dir=cache_dir,
                    torch_dtype=torch_dtype,
                    original_config=original_config,
                    local_files_only=local_files_only,
                    use_safetensors=use_safetensors,
                    is_legacy_loading=is_legacy_loading,
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

        # deprecated kwargs
        load_safety_checker = kwargs.pop("load_safety_checker", None)
        if load_safety_checker is not None:
            deprecation_message = (
                "Please pass instances of `StableDiffusionSafetyChecker` and `AutoImageProcessor`"
                "using the `safety_checker` and `feature_extractor` arguments in `from_single_file`"
                "If no safety checker components are provided, the safety checker will be loaded based"
                f"on the default config for the {pipeline_class.__name__}: {default_pretrained_model_name_or_path}."
            )
            deprecate("load_safety_checker", "1.0.0", deprecation_message)

            safety_checker_components = _legacy_load_safety_checker(local_files_only, torch_dtype)
            init_kwargs.update(safety_checker_components)

        scheduler_type = kwargs.pop("scheduler_type", None)
        if scheduler_type is not None:
            deprecation_message = (
                "The `scheduler_type` argument is deprecated and will be ignored in future versions. "
                "Please pass an instance of a Scheduler object directly to the `scheduler` argument in `from_single_file`."
                "If no scheduler is provided, it will be loaded based"
                f"on the default config for the {pipeline_class.__name__}: {default_pretrained_model_name_or_path}."
            )
            deprecate("scheduler_type", "1.0.0", deprecation_message)
            scheduler = _legacy_load_scheduler(
                pipeline_class_name=class_name,
                checkpoint=checkpoint,
                original_config=original_config,
                scheduler_type=scheduler_type,
                prediction_type=prediction_type,
            )
            init_kwargs.update({"scheduler": scheduler})

        pipe = pipeline_class(**init_kwargs)

        if torch_dtype is not None:
            pipe.to(dtype=torch_dtype)

        return pipe
