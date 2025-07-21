# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from huggingface_hub.utils import validate_hf_hub_args
from typing_extensions import Self

from ..configuration_utils import ConfigMixin
from ..utils import PushToHubMixin, get_logger


if TYPE_CHECKING:
    from ..modular_pipelines.modular_pipeline import BlockState


GUIDER_CONFIG_NAME = "guider_config.json"


logger = get_logger(__name__)  # pylint: disable=invalid-name


class BaseGuidance(ConfigMixin, PushToHubMixin):
    r"""Base class providing the skeleton for implementing guidance techniques."""

    config_name = GUIDER_CONFIG_NAME
    _input_predictions = None
    _identifier_key = "__guidance_identifier__"

    def __init__(self, start: float = 0.0, stop: float = 1.0):
        self._start = start
        self._stop = stop
        self._step: int = None
        self._num_inference_steps: int = None
        self._timestep: torch.LongTensor = None
        self._count_prepared = 0
        self._input_fields: Dict[str, Union[str, Tuple[str, str]]] = None
        self._enabled = True

        if not (0.0 <= start < 1.0):
            raise ValueError(f"Expected `start` to be between 0.0 and 1.0, but got {start}.")
        if not (start <= stop <= 1.0):
            raise ValueError(f"Expected `stop` to be between {start} and 1.0, but got {stop}.")

        if self._input_predictions is None or not isinstance(self._input_predictions, list):
            raise ValueError(
                "`_input_predictions` must be a list of required prediction names for the guidance technique."
            )

    def disable(self):
        self._enabled = False

    def enable(self):
        self._enabled = True

    def set_state(self, step: int, num_inference_steps: int, timestep: torch.LongTensor) -> None:
        self._step = step
        self._num_inference_steps = num_inference_steps
        self._timestep = timestep
        self._count_prepared = 0

    def set_input_fields(self, **kwargs: Dict[str, Union[str, Tuple[str, str]]]) -> None:
        """
        Set the input fields for the guidance technique. The input fields are used to specify the names of the returned
        attributes containing the prepared data after `prepare_inputs` is called. The prepared data is obtained from
        the values of the provided keyword arguments to this method.

        Args:
            **kwargs (`Dict[str, Union[str, Tuple[str, str]]]`):
                A dictionary where the keys are the names of the fields that will be used to store the data once it is
                prepared with `prepare_inputs`. The values can be either a string or a tuple of length 2, which is used
                to look up the required data provided for preparation.

                If a string is provided, it will be used as the conditional data (or unconditional if used with a
                guidance method that requires it). If a tuple of length 2 is provided, the first element must be the
                conditional data identifier and the second element must be the unconditional data identifier or None.

                Example:
                ```
                data = {"prompt_embeds": <some tensor>, "negative_prompt_embeds": <some tensor>, "latents": <some tensor>}

                BaseGuidance.set_input_fields(
                    latents="latents",
                    prompt_embeds=("prompt_embeds", "negative_prompt_embeds"),
                )
                ```
        """
        for key, value in kwargs.items():
            is_string = isinstance(value, str)
            is_tuple_of_str_with_len_2 = (
                isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, str) for v in value)
            )
            if not (is_string or is_tuple_of_str_with_len_2):
                raise ValueError(
                    f"Expected `set_input_fields` to be called with a string or a tuple of string with length 2, but got {type(value)} for key {key}."
                )
        self._input_fields = kwargs

    def prepare_models(self, denoiser: torch.nn.Module) -> None:
        """
        Prepares the models for the guidance technique on a given batch of data. This method should be overridden in
        subclasses to implement specific model preparation logic.
        """
        self._count_prepared += 1

    def cleanup_models(self, denoiser: torch.nn.Module) -> None:
        """
        Cleans up the models for the guidance technique after a given batch of data. This method should be overridden
        in subclasses to implement specific model cleanup logic. It is useful for removing any hooks or other stateful
        modifications made during `prepare_models`.
        """
        pass

    def prepare_inputs(self, data: "BlockState") -> List["BlockState"]:
        raise NotImplementedError("BaseGuidance::prepare_inputs must be implemented in subclasses.")

    def __call__(self, data: List["BlockState"]) -> Any:
        if not all(hasattr(d, "noise_pred") for d in data):
            raise ValueError("Expected all data to have `noise_pred` attribute.")
        if len(data) != self.num_conditions:
            raise ValueError(
                f"Expected {self.num_conditions} data items, but got {len(data)}. Please check the input data."
            )
        forward_inputs = {getattr(d, self._identifier_key): d.noise_pred for d in data}
        return self.forward(**forward_inputs)

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("BaseGuidance::forward must be implemented in subclasses.")

    @property
    def is_conditional(self) -> bool:
        raise NotImplementedError("BaseGuidance::is_conditional must be implemented in subclasses.")

    @property
    def is_unconditional(self) -> bool:
        return not self.is_conditional

    @property
    def num_conditions(self) -> int:
        raise NotImplementedError("BaseGuidance::num_conditions must be implemented in subclasses.")

    @classmethod
    def _prepare_batch(
        cls,
        input_fields: Dict[str, Union[str, Tuple[str, str]]],
        data: "BlockState",
        tuple_index: int,
        identifier: str,
    ) -> "BlockState":
        """
        Prepares a batch of data for the guidance technique. This method is used in the `prepare_inputs` method of the
        `BaseGuidance` class. It prepares the batch based on the provided tuple index.

        Args:
            input_fields (`Dict[str, Union[str, Tuple[str, str]]]`):
                A dictionary where the keys are the names of the fields that will be used to store the data once it is
                prepared with `prepare_inputs`. The values can be either a string or a tuple of length 2, which is used
                to look up the required data provided for preparation. If a string is provided, it will be used as the
                conditional data (or unconditional if used with a guidance method that requires it). If a tuple of
                length 2 is provided, the first element must be the conditional data identifier and the second element
                must be the unconditional data identifier or None.
            data (`BlockState`):
                The input data to be prepared.
            tuple_index (`int`):
                The index to use when accessing input fields that are tuples.

        Returns:
            `BlockState`: The prepared batch of data.
        """
        from ..modular_pipelines.modular_pipeline import BlockState

        if input_fields is None:
            raise ValueError(
                "Input fields cannot be None. Please pass `input_fields` to `prepare_inputs` or call `set_input_fields` before preparing inputs."
            )
        data_batch = {}
        for key, value in input_fields.items():
            try:
                if isinstance(value, str):
                    data_batch[key] = getattr(data, value)
                elif isinstance(value, tuple):
                    data_batch[key] = getattr(data, value[tuple_index])
                else:
                    # We've already checked that value is a string or a tuple of strings with length 2
                    pass
            except AttributeError:
                logger.debug(f"`data` does not have attribute(s) {value}, skipping.")
        data_batch[cls._identifier_key] = identifier
        return BlockState(**data_batch)

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        subfolder: Optional[str] = None,
        return_unused_kwargs=False,
        **kwargs,
    ) -> Self:
        r"""
        Instantiate a guider from a pre-defined JSON configuration file in a local directory or Hub repository.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the guider configuration
                      saved with [`~BaseGuidance.save_pretrained`].
            subfolder (`str`, *optional*):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                Whether kwargs that are not consumed by the Python class should be returned or not.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.

            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.

        <Tip>

        To use private or [gated models](https://huggingface.co/docs/hub/models-gated#gated-models), log-in with
        `huggingface-cli login`. You can also activate the special
        ["offline-mode"](https://huggingface.co/diffusers/installation.html#offline-mode) to use this method in a
        firewalled environment.

        </Tip>

        """
        config, kwargs, commit_hash = cls.load_config(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder=subfolder,
            return_unused_kwargs=True,
            return_commit_hash=True,
            **kwargs,
        )
        return cls.from_config(config, return_unused_kwargs=return_unused_kwargs, **kwargs)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save a guider configuration object to a directory so that it can be reloaded using the
        [`~BaseGuidance.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face Hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        self.save_config(save_directory=save_directory, push_to_hub=push_to_hub, **kwargs)


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    r"""
    Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure. Based on
    Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
    Flawed](https://arxiv.org/pdf/2305.08891.pdf).

    Args:
        noise_cfg (`torch.Tensor`):
            The predicted noise tensor for the guided diffusion process.
        noise_pred_text (`torch.Tensor`):
            The predicted noise tensor for the text-guided diffusion process.
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            A rescale factor applied to the noise predictions.
    Returns:
        noise_cfg (`torch.Tensor`): The rescaled noise prediction tensor.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg
