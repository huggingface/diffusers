from typing import Optional

from huggingface_hub.utils import validate_hf_hub_args

from ..utils import logging
from .single_file_utils import (
    create_diffusers_unet_from_ldm,
    create_diffusers_unet_from_stable_cascade,
    create_diffusers_vae_from_ldm,
    load_single_file_model_checkpoint,
)


SINGLE_FILE_LOADABLE_CLASSES = {
    "StableCascadeUNet": create_diffusers_unet_from_stable_cascade,
    "UNet2DConditionModel": create_diffusers_unet_from_ldm,
    "AutoencoderKL": create_diffusers_vae_from_ldm,
}

class FromOriginalModelMixin:
    """
    Load pretrained UNet model weights saved in the `.ckpt` or `.safetensors` format into a [`StableCascadeUNet`].
    """

    @classmethod
    @validate_hf_hub_args
    def from_single_file(cls, pretrained_model_link_or_path: Optional[str] = None, **kwargs):
        r"""
        Instantiate a Model from pretrained weights saved in the original `.ckpt`, `.bin`, or
        `.safetensors` format. The model is set in evaluation mode (`model.eval()`) by default.
        Currently supported checkpoints: StableCascade, SDXL, SD, Playground v2.5, etc.

        Parameters:
            pretrained_model_link_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:
                    - A link to the `.ckpt` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.ckpt"`) on the Hub.
                    - A path to a *file* containing all pipeline weights.
            config: (`dict`, *optional*):
                Dictionary containing the configuration of the model:
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
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables of the model.

        """
        class_name = cls.__name__
        if class_name not in SINGLE_FILE_LOADABLE_CLASSES:
            raise ValueError(
                f"FromOriginalUNetMixin is currently only compatible with {', '.join(SINGLE_FILE_LOADABLE_CLASSES.keys())}"
            )

        checkpoint = kwargs.pop("checkpoint", None)
        if pretrained_model_link_or_path is None and checkpoint is None:
            raise ValueError(
                "Please provide either a `pretrained_model_link_or_path` or a `checkpoint` to load the model from."
            )

        config = kwargs.pop("config", None)

        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        cache_dir = kwargs.pop("cache_dir", None)
        local_files_only = kwargs.pop("local_files_only", None)
        revision = kwargs.pop("revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)

        if checkpoint is None:
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

        model_loading_fn = SINGLE_FILE_LOADABLE_CLASSES[class_name]
        model = model_loading_fn(
            cls,
            checkpoint=checkpoint,
            config=config,
            torch_dtype=torch_dtype,
            **kwargs,
        )

        return model