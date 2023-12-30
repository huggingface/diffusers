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
from io import BytesIO
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import validate_hf_hub_args


class FromOriginalControlnetMixin:
    """
    Load pretrained ControlNet weights saved in the `.ckpt` or `.safetensors` format into a [`ControlNetModel`].
    """

    @classmethod
    @validate_hf_hub_args
    def from_single_file(cls, pretrained_model_link_or_path, **kwargs):
        r"""
        Instantiate a [`ControlNetModel`] from pretrained ControlNet weights saved in the original `.ckpt` or
        `.safetensors` format. The pipeline is set in evaluation mode (`model.eval()`) by default.

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
                Whether to only load local model weights and configuration files or not. If set to True, the model
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
            image_size (`int`, *optional*, defaults to 512):
                The image size the model was trained on. Use 512 for all Stable Diffusion v1 models and the Stable
                Diffusion v2 base model. Use 768 for Stable Diffusion v2.
            upcast_attention (`bool`, *optional*, defaults to `None`):
                Whether the attention computation should always be upcasted.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (for example the pipeline components of the
                specific pipeline class). The overwritten components are directly passed to the pipelines `__init__`
                method. See example below for more information.

        Examples:

        ```py
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

        url = "https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15_canny.pth"  # can also be a local path
        model = ControlNetModel.from_single_file(url)

        url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned.safetensors"  # can also be a local path
        pipe = StableDiffusionControlNetPipeline.from_single_file(url, controlnet=controlnet)
        ```
        """
        # import here to avoid circular dependency
        from ..pipelines.stable_diffusion.convert_from_ckpt import download_controlnet_from_original_ckpt

        config_file = kwargs.pop("config_file", None)
        cache_dir = kwargs.pop("cache_dir", None)
        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        num_in_channels = kwargs.pop("num_in_channels", None)
        use_linear_projection = kwargs.pop("use_linear_projection", None)
        revision = kwargs.pop("revision", None)
        extract_ema = kwargs.pop("extract_ema", False)
        image_size = kwargs.pop("image_size", None)
        upcast_attention = kwargs.pop("upcast_attention", None)

        torch_dtype = kwargs.pop("torch_dtype", None)

        use_safetensors = kwargs.pop("use_safetensors", None)

        file_extension = pretrained_model_link_or_path.rsplit(".", 1)[-1]
        from_safetensors = file_extension == "safetensors"

        if from_safetensors and use_safetensors is False:
            raise ValueError("Make sure to install `safetensors` with `pip install safetensors`.")

        # remove huggingface url
        for prefix in ["https://huggingface.co/", "huggingface.co/", "hf.co/", "https://hf.co/"]:
            if pretrained_model_link_or_path.startswith(prefix):
                pretrained_model_link_or_path = pretrained_model_link_or_path[len(prefix) :]

        # Code based on diffusers.pipelines.pipeline_utils.DiffusionPipeline.from_pretrained
        ckpt_path = Path(pretrained_model_link_or_path)
        if not ckpt_path.is_file():
            # get repo_id and (potentially nested) file path of ckpt in repo
            repo_id = "/".join(ckpt_path.parts[:2])
            file_path = "/".join(ckpt_path.parts[2:])

            if file_path.startswith("blob/"):
                file_path = file_path[len("blob/") :]

            if file_path.startswith("main/"):
                file_path = file_path[len("main/") :]

            pretrained_model_link_or_path = hf_hub_download(
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

        if config_file is None:
            config_url = "https://raw.githubusercontent.com/lllyasviel/ControlNet/main/models/cldm_v15.yaml"
            config_file = BytesIO(requests.get(config_url).content)

        image_size = image_size or 512

        controlnet = download_controlnet_from_original_ckpt(
            pretrained_model_link_or_path,
            original_config_file=config_file,
            image_size=image_size,
            extract_ema=extract_ema,
            num_in_channels=num_in_channels,
            upcast_attention=upcast_attention,
            from_safetensors=from_safetensors,
            use_linear_projection=use_linear_projection,
        )

        if torch_dtype is not None:
            controlnet.to(dtype=torch_dtype)

        return controlnet
