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


if is_transformers_available():
    pass

if is_accelerate_available():
    from accelerate import init_empty_weights

logger = logging.get_logger(__name__)


class FromOriginalVAEMixin:
    """
    Load pretrained ControlNet weights saved in the `.ckpt` or `.safetensors` format into an [`AutoencoderKL`].
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
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            upcast_attention (`bool`, *optional*, defaults to `None`):
                Whether the attention computation should always be upcasted.
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
        if not is_omegaconf_available():
            raise ValueError(BACKENDS_MAPPING["omegaconf"][1])

        from omegaconf import OmegaConf

        from ..models import AutoencoderKL

        # import here to avoid circular dependency
        from ..pipelines.stable_diffusion.convert_from_ckpt import (
            convert_ldm_vae_checkpoint,
            create_vae_diffusers_config,
        )

        config_file = kwargs.pop("config_file", None)
        cache_dir = kwargs.pop("cache_dir", None)
        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        image_size = kwargs.pop("image_size", None)
        scaling_factor = kwargs.pop("scaling_factor", None)
        kwargs.pop("upcast_attention", None)

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

        if from_safetensors:
            from safetensors import safe_open

            checkpoint = {}
            with safe_open(pretrained_model_link_or_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    checkpoint[key] = f.get_tensor(key)
        else:
            checkpoint = torch.load(pretrained_model_link_or_path, map_location="cpu")

        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        if config_file is None:
            config_url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml"
            config_file = BytesIO(requests.get(config_url).content)

        original_config = OmegaConf.load(config_file)

        # default to sd-v1-5
        image_size = image_size or 512

        vae_config = create_vae_diffusers_config(original_config, image_size=image_size)
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)

        if scaling_factor is None:
            if (
                "model" in original_config
                and "params" in original_config.model
                and "scale_factor" in original_config.model.params
            ):
                vae_scaling_factor = original_config.model.params.scale_factor
            else:
                vae_scaling_factor = 0.18215  # default SD scaling factor

        vae_config["scaling_factor"] = vae_scaling_factor

        ctx = init_empty_weights if is_accelerate_available() else nullcontext
        with ctx():
            vae = AutoencoderKL(**vae_config)

        if is_accelerate_available():
            from ..models.modeling_utils import load_model_dict_into_meta

            load_model_dict_into_meta(vae, converted_vae_checkpoint, device="cpu")
        else:
            vae.load_state_dict(converted_vae_checkpoint)

        if torch_dtype is not None:
            vae.to(dtype=torch_dtype)

        return vae
