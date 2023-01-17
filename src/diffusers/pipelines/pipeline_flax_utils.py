# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import inspect
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np

import flax
import PIL
from flax.core.frozen_dict import FrozenDict
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm.auto import tqdm

from ..configuration_utils import ConfigMixin
from ..models.modeling_flax_utils import FLAX_WEIGHTS_NAME, FlaxModelMixin
from ..schedulers.scheduling_utils_flax import SCHEDULER_CONFIG_NAME, FlaxSchedulerMixin
from ..utils import CONFIG_NAME, DIFFUSERS_CACHE, BaseOutput, http_user_agent, is_transformers_available, logging


if is_transformers_available():
    from transformers import FlaxPreTrainedModel

INDEX_FILE = "diffusion_flax_model.bin"


logger = logging.get_logger(__name__)


LOADABLE_CLASSES = {
    "diffusers": {
        "FlaxModelMixin": ["save_pretrained", "from_pretrained"],
        "FlaxSchedulerMixin": ["save_pretrained", "from_pretrained"],
        "FlaxDiffusionPipeline": ["save_pretrained", "from_pretrained"],
    },
    "transformers": {
        "PreTrainedTokenizer": ["save_pretrained", "from_pretrained"],
        "PreTrainedTokenizerFast": ["save_pretrained", "from_pretrained"],
        "FlaxPreTrainedModel": ["save_pretrained", "from_pretrained"],
        "FeatureExtractionMixin": ["save_pretrained", "from_pretrained"],
        "ProcessorMixin": ["save_pretrained", "from_pretrained"],
        "ImageProcessingMixin": ["save_pretrained", "from_pretrained"],
    },
}

ALL_IMPORTABLE_CLASSES = {}
for library in LOADABLE_CLASSES:
    ALL_IMPORTABLE_CLASSES.update(LOADABLE_CLASSES[library])


def import_flax_or_no_model(module, class_name):
    try:
        # 1. First make sure that if a Flax object is present, import this one
        class_obj = getattr(module, "Flax" + class_name)
    except AttributeError:
        # 2. If this doesn't work, it's not a model and we don't append "Flax"
        class_obj = getattr(module, class_name)
    except AttributeError:
        raise ValueError(f"Neither Flax{class_name} nor {class_name} exist in {module}")

    return class_obj


@flax.struct.dataclass
class FlaxImagePipelineOutput(BaseOutput):
    """
    Output class for image pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


class FlaxDiffusionPipeline(ConfigMixin):
    r"""
    Base class for all models.

    [`FlaxDiffusionPipeline`] takes care of storing all components (models, schedulers, processors) for diffusion
    pipelines and handles methods for loading, downloading and saving models as well as a few methods common to all
    pipelines to:

        - enabling/disabling the progress bar for the denoising iteration

    Class attributes:

        - **config_name** ([`str`]) -- name of the config file that will store the class and module names of all
          components of the diffusion pipeline.
    """
    config_name = "model_index.json"

    def register_modules(self, **kwargs):
        # import it here to avoid circular import
        from diffusers import pipelines

        for name, module in kwargs.items():
            if module is None:
                register_dict = {name: (None, None)}
            else:
                # retrieve library
                library = module.__module__.split(".")[0]

                # check if the module is a pipeline module
                pipeline_dir = module.__module__.split(".")[-2]
                path = module.__module__.split(".")
                is_pipeline_module = pipeline_dir in path and hasattr(pipelines, pipeline_dir)

                # if library is not in LOADABLE_CLASSES, then it is a custom module.
                # Or if it's a pipeline module, then the module is inside the pipeline
                # folder so we set the library to module name.
                if library not in LOADABLE_CLASSES or is_pipeline_module:
                    library = pipeline_dir

                # retrieve class_name
                class_name = module.__class__.__name__

                register_dict = {name: (library, class_name)}

            # save model index config
            self.register_to_config(**register_dict)

            # set models
            setattr(self, name, module)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], params: Union[Dict, FrozenDict]):
        # TODO: handle inference_state
        """
        Save all variables of the pipeline that can be saved and loaded as well as the pipelines configuration file to
        a directory. A pipeline variable can be saved and loaded if its class implements both a save and loading
        method. The pipeline can easily be re-loaded using the `[`~FlaxDiffusionPipeline.from_pretrained`]` class
        method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
        """
        self.save_config(save_directory)

        model_index_dict = dict(self.config)
        model_index_dict.pop("_class_name")
        model_index_dict.pop("_diffusers_version")
        model_index_dict.pop("_module", None)

        for pipeline_component_name in model_index_dict.keys():
            sub_model = getattr(self, pipeline_component_name)
            if sub_model is None:
                # edge case for saving a pipeline with safety_checker=None
                continue

            model_cls = sub_model.__class__

            save_method_name = None
            # search for the model's base class in LOADABLE_CLASSES
            for library_name, library_classes in LOADABLE_CLASSES.items():
                library = importlib.import_module(library_name)
                for base_class, save_load_methods in library_classes.items():
                    class_candidate = getattr(library, base_class, None)
                    if class_candidate is not None and issubclass(model_cls, class_candidate):
                        # if we found a suitable base class in LOADABLE_CLASSES then grab its save method
                        save_method_name = save_load_methods[0]
                        break
                if save_method_name is not None:
                    break

            save_method = getattr(sub_model, save_method_name)
            expects_params = "params" in set(inspect.signature(save_method).parameters.keys())

            if expects_params:
                save_method(
                    os.path.join(save_directory, pipeline_component_name), params=params[pipeline_component_name]
                )
            else:
                save_method(os.path.join(save_directory, pipeline_component_name))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        r"""
        Instantiate a Flax diffusion pipeline from pre-trained pipeline weights.

        The pipeline is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated).

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *repo id* of a pretrained pipeline hosted inside a model repo on
                      https://huggingface.co/ Valid repo ids have to be located under a user or organization name, like
                      `CompVis/ldm-text2im-large-256`.
                    - A path to a *directory* containing pipeline weights saved using
                      [`~FlaxDiffusionPipeline.save_pretrained`], e.g., `./my_pipeline_directory/`.
            dtype (`str` or `jnp.dtype`, *optional*):
                Override the default `jnp.dtype` and load the model under this dtype. If `"auto"` is passed the dtype
                will be automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information. specify the folder name here.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load - and saveable variables - *i.e.* the pipeline components - of the
                specific pipeline class. The overwritten components are then directly passed to the pipelines
                `__init__` method. See example below for more information.

        <Tip>

         It is required to be logged in (`huggingface-cli login`) when you want to use private or [gated
         models](https://huggingface.co/docs/hub/models-gated#gated-models), *e.g.* `"runwayml/stable-diffusion-v1-5"`

        </Tip>

        <Tip>

        Activate the special ["offline-mode"](https://huggingface.co/diffusers/installation.html#offline-mode) to use
        this method in a firewalled environment.

        </Tip>

        Examples:

        ```py
        >>> from diffusers import FlaxDiffusionPipeline

        >>> # Download pipeline from huggingface.co and cache.
        >>> # Requires to be logged in to Hugging Face hub,
        >>> # see more in [the documentation](https://huggingface.co/docs/hub/security-tokens)
        >>> pipeline, params = FlaxDiffusionPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5",
        ...     revision="bf16",
        ...     dtype=jnp.bfloat16,
        ... )

        >>> # Download pipeline, but use a different scheduler
        >>> from diffusers import FlaxDPMSolverMultistepScheduler

        >>> model_id = "runwayml/stable-diffusion-v1-5"
        >>> sched, sched_state = FlaxDPMSolverMultistepScheduler.from_pretrained(
        ...     model_id,
        ...     subfolder="scheduler",
        ... )

        >>> dpm_pipe, dpm_params = FlaxStableDiffusionPipeline.from_pretrained(
        ...     model_id, revision="bf16", dtype=jnp.bfloat16, scheduler=dpmpp
        ... )
        >>> dpm_params["scheduler"] = dpmpp_state
        ```
        """
        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        from_pt = kwargs.pop("from_pt", False)
        dtype = kwargs.pop("dtype", None)

        # 1. Download the checkpoints and configs
        # use snapshot download here to get it working from from_pretrained
        if not os.path.isdir(pretrained_model_name_or_path):
            config_dict = cls.load_config(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
            )
            # make sure we only download sub-folders and `diffusers` filenames
            folder_names = [k for k in config_dict.keys() if not k.startswith("_")]
            allow_patterns = [os.path.join(k, "*") for k in folder_names]
            allow_patterns += [FLAX_WEIGHTS_NAME, SCHEDULER_CONFIG_NAME, CONFIG_NAME, cls.config_name]

            # make sure we don't download PyTorch weights, unless when using from_pt
            ignore_patterns = "*.bin" if not from_pt else []

            if cls != FlaxDiffusionPipeline:
                requested_pipeline_class = cls.__name__
            else:
                requested_pipeline_class = config_dict.get("_class_name", cls.__name__)
                requested_pipeline_class = (
                    requested_pipeline_class
                    if requested_pipeline_class.startswith("Flax")
                    else "Flax" + requested_pipeline_class
                )

            user_agent = {"pipeline_class": requested_pipeline_class}
            user_agent = http_user_agent(user_agent)

            # download all allow_patterns
            cached_folder = snapshot_download(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                user_agent=user_agent,
            )
        else:
            cached_folder = pretrained_model_name_or_path

        config_dict = cls.load_config(cached_folder)

        # 2. Load the pipeline class, if using custom module then load it from the hub
        # if we load from explicit class, let's use it
        if cls != FlaxDiffusionPipeline:
            pipeline_class = cls
        else:
            diffusers_module = importlib.import_module(cls.__module__.split(".")[0])
            class_name = (
                config_dict["_class_name"]
                if config_dict["_class_name"].startswith("Flax")
                else "Flax" + config_dict["_class_name"]
            )
            pipeline_class = getattr(diffusers_module, class_name)

        # some modules can be passed directly to the init
        # in this case they are already instantiated in `kwargs`
        # extract them here
        expected_modules = set(inspect.signature(pipeline_class.__init__).parameters.keys())
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}

        init_dict, _, _ = pipeline_class.extract_init_dict(config_dict, **kwargs)

        init_kwargs = {}

        # inference_params
        params = {}

        # import it here to avoid circular import
        from diffusers import pipelines

        # 3. Load each module in the pipeline
        for name, (library_name, class_name) in init_dict.items():
            if class_name is None:
                # edge case for when the pipeline was saved with safety_checker=None
                init_kwargs[name] = None
                continue

            is_pipeline_module = hasattr(pipelines, library_name)
            loaded_sub_model = None
            sub_model_should_be_defined = True

            # if the model is in a pipeline module, then we load it from the pipeline
            if name in passed_class_obj:
                # 1. check that passed_class_obj has correct parent class
                if not is_pipeline_module:
                    library = importlib.import_module(library_name)
                    class_obj = getattr(library, class_name)
                    importable_classes = LOADABLE_CLASSES[library_name]
                    class_candidates = {c: getattr(library, c, None) for c in importable_classes.keys()}

                    expected_class_obj = None
                    for class_name, class_candidate in class_candidates.items():
                        if class_candidate is not None and issubclass(class_obj, class_candidate):
                            expected_class_obj = class_candidate

                    if not issubclass(passed_class_obj[name].__class__, expected_class_obj):
                        raise ValueError(
                            f"{passed_class_obj[name]} is of type: {type(passed_class_obj[name])}, but should be"
                            f" {expected_class_obj}"
                        )
                elif passed_class_obj[name] is None:
                    logger.warning(
                        f"You have passed `None` for {name} to disable its functionality in {pipeline_class}. Note"
                        f" that this might lead to problems when using {pipeline_class} and is not recommended."
                    )
                    sub_model_should_be_defined = False
                else:
                    logger.warning(
                        f"You have passed a non-standard module {passed_class_obj[name]}. We cannot verify whether it"
                        " has the correct type"
                    )

                # set passed class object
                loaded_sub_model = passed_class_obj[name]
            elif is_pipeline_module:
                pipeline_module = getattr(pipelines, library_name)
                class_obj = import_flax_or_no_model(pipeline_module, class_name)

                importable_classes = ALL_IMPORTABLE_CLASSES
                class_candidates = {c: class_obj for c in importable_classes.keys()}
            else:
                # else we just import it from the library.
                library = importlib.import_module(library_name)
                class_obj = import_flax_or_no_model(library, class_name)

                importable_classes = LOADABLE_CLASSES[library_name]
                class_candidates = {c: getattr(library, c, None) for c in importable_classes.keys()}

            if loaded_sub_model is None and sub_model_should_be_defined:
                load_method_name = None
                for class_name, class_candidate in class_candidates.items():
                    if class_candidate is not None and issubclass(class_obj, class_candidate):
                        load_method_name = importable_classes[class_name][1]

                load_method = getattr(class_obj, load_method_name)

                # check if the module is in a subdirectory
                if os.path.isdir(os.path.join(cached_folder, name)):
                    loadable_folder = os.path.join(cached_folder, name)
                else:
                    loaded_sub_model = cached_folder

                if issubclass(class_obj, FlaxModelMixin):
                    loaded_sub_model, loaded_params = load_method(loadable_folder, from_pt=from_pt, dtype=dtype)
                    params[name] = loaded_params
                elif is_transformers_available() and issubclass(class_obj, FlaxPreTrainedModel):
                    if from_pt:
                        # TODO(Suraj): Fix this in Transformers. We should be able to use `_do_init=False` here
                        loaded_sub_model = load_method(loadable_folder, from_pt=from_pt)
                        loaded_params = loaded_sub_model.params
                        del loaded_sub_model._params
                    else:
                        loaded_sub_model, loaded_params = load_method(loadable_folder, _do_init=False)
                    params[name] = loaded_params
                elif issubclass(class_obj, FlaxSchedulerMixin):
                    loaded_sub_model, scheduler_state = load_method(loadable_folder)
                    params[name] = scheduler_state
                else:
                    loaded_sub_model = load_method(loadable_folder)

            init_kwargs[name] = loaded_sub_model  # UNet(...), # DiffusionSchedule(...)

        model = pipeline_class(**init_kwargs, dtype=dtype)
        return model, params

    @staticmethod
    def _get_signature_keys(obj):
        parameters = inspect.signature(obj.__init__).parameters
        required_parameters = {k: v for k, v in parameters.items() if v.default == inspect._empty}
        optional_parameters = set({k for k, v in parameters.items() if v.default != inspect._empty})
        expected_modules = set(required_parameters.keys()) - set(["self"])
        return expected_modules, optional_parameters

    @property
    def components(self) -> Dict[str, Any]:
        r"""

        The `self.components` property can be useful to run different pipelines with the same weights and
        configurations to not have to re-allocate memory.

        Examples:

        ```py
        >>> from diffusers import (
        ...     FlaxStableDiffusionPipeline,
        ...     FlaxStableDiffusionImg2ImgPipeline,
        ... )

        >>> text2img = FlaxStableDiffusionPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5", revision="bf16", dtype=jnp.bfloat16
        ... )
        >>> img2img = FlaxStableDiffusionImg2ImgPipeline(**text2img.components)
        ```

        Returns:
            A dictionary containing all the modules needed to initialize the pipeline.
        """
        expected_modules, optional_parameters = self._get_signature_keys(self)
        components = {
            k: getattr(self, k) for k in self.config.keys() if not k.startswith("_") and k not in optional_parameters
        }

        if set(components.keys()) != expected_modules:
            raise ValueError(
                f"{self} has been incorrectly initialized or {self.__class__} is incorrectly implemented. Expected"
                f" {expected_modules} to be defined, but {components} are defined."
            )

        return components

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    # TODO: make it compatible with jax.lax
    def progress_bar(self, iterable):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        return tqdm(iterable, **self._progress_bar_config)

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs
