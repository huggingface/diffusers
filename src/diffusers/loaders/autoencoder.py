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

from .single_file_utils import (
    create_diffusers_vae_model_from_ldm,
    fetch_ldm_config_and_checkpoint,
)


class FromOriginalVAEMixin:
    """
    Load pretrained AutoencoderKL weights saved in the `.ckpt` or `.safetensors` format into a [`AutoencoderKL`].
    """

    @classmethod
    @validate_hf_hub_args
    def from_single_file(cls, pretrained_model_link_or_path, **kwargs):
        r"""
        Instantiate a [`AutoencoderKL`] from pretrained ControlNet weights saved in the original `.ckpt` or
        `.safetensors` format. The pipeline is set in evaluation mode (`model.eval()`) by default.

        Parameters:
            pretrained_model_link_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:
                    - A link to the `.ckpt` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.ckpt"`) on the Hub.
                    - A path to a *file* containing all pipeline weights.
            config_file (`str`, *optional*):
                Filepath to the configuration YAML file associated with the model. If not provided it will default to:
                https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml
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
            image_size (`int`, *optional*, defaults to 512):
                The image size the model was trained on. Use 512 for all Stable Diffusion v1 models and the Stable
                Diffusion v2 base model. Use 768 for Stable Diffusion v2.
            scaling_factor (`float`, *optional*, defaults to 0.18215):
                The component-wise standard deviation of the trained latent space computed using the first batch of the
                training set. This is used to scale the latent space to have unit variance when training the diffusion
                model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
                diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z
                = 1 / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution
                Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (for example the pipeline components of the
                specific pipeline class). The overwritten components are directly passed to the pipelines `__init__`
                method. See example below for more information.

        <Tip warning={true}>

            Make sure to pass both `image_size` and `scaling_factor` to `from_single_file()` if you're loading
            a VAE from SDXL or a Stable Diffusion v2 model or higher.

        </Tip>

        Examples:

        ```py
        from diffusers import AutoencoderKL

        url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"  # can also be local file
        model = AutoencoderKL.from_single_file(url)
        ```
        """

        original_config_file = kwargs.pop("original_config_file", None)
        config_file = kwargs.pop("config_file", None)
        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        cache_dir = kwargs.pop("cache_dir", None)
        local_files_only = kwargs.pop("local_files_only", None)
        revision = kwargs.pop("revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)

        class_name = cls.__name__

        if (config_file is not None) and (original_config_file is not None):
            raise ValueError(
                "You cannot pass both `config_file` and `original_config_file` to `from_single_file`. Please use only one of these arguments."
            )

        original_config_file = original_config_file or config_file
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

        image_size = kwargs.pop("image_size", None)
        scaling_factor = kwargs.pop("scaling_factor", None)
        component = create_diffusers_vae_model_from_ldm(
            class_name,
            original_config,
            checkpoint,
            image_size=image_size,
            scaling_factor=scaling_factor,
            torch_dtype=torch_dtype,
        )
        vae = component["vae"]
        if torch_dtype is not None:
            vae = vae.to(torch_dtype)

        return vae
