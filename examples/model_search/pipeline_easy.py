# coding=utf-8
# Copyright 2025 suzukimain
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
import re
import types
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Union

import requests
import torch
from huggingface_hub import hf_api, hf_hub_download
from huggingface_hub.file_download import http_get
from huggingface_hub.utils import validate_hf_hub_args

from diffusers.loaders.single_file_utils import (
    VALID_URL_PREFIXES,
    _extract_repo_id_and_weights_name,
    infer_diffusers_model_type,
    load_single_file_checkpoint,
)
from diffusers.pipelines.animatediff import AnimateDiffPipeline, AnimateDiffSDXLPipeline
from diffusers.pipelines.auto_pipeline import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
)
from diffusers.pipelines.controlnet import (
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetPipeline,
)
from diffusers.pipelines.flux import FluxImg2ImgPipeline, FluxPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    StableDiffusionUpscalePipeline,
)
from diffusers.pipelines.stable_diffusion_3 import StableDiffusion3Img2ImgPipeline, StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion_xl import (
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.utils import logging


logger = logging.get_logger(__name__)


SINGLE_FILE_CHECKPOINT_TEXT2IMAGE_PIPELINE_MAPPING = OrderedDict(
    [
        ("animatediff_rgb", AnimateDiffPipeline),
        ("animatediff_scribble", AnimateDiffPipeline),
        ("animatediff_sdxl_beta", AnimateDiffSDXLPipeline),
        ("animatediff_v1", AnimateDiffPipeline),
        ("animatediff_v2", AnimateDiffPipeline),
        ("animatediff_v3", AnimateDiffPipeline),
        ("autoencoder-dc-f128c512", None),
        ("autoencoder-dc-f32c32", None),
        ("autoencoder-dc-f32c32-sana", None),
        ("autoencoder-dc-f64c128", None),
        ("controlnet", StableDiffusionControlNetPipeline),
        ("controlnet_xl", StableDiffusionXLControlNetPipeline),
        ("controlnet_xl_large", StableDiffusionXLControlNetPipeline),
        ("controlnet_xl_mid", StableDiffusionXLControlNetPipeline),
        ("controlnet_xl_small", StableDiffusionXLControlNetPipeline),
        ("flux-depth", FluxPipeline),
        ("flux-dev", FluxPipeline),
        ("flux-fill", FluxPipeline),
        ("flux-schnell", FluxPipeline),
        ("hunyuan-video", None),
        ("inpainting", None),
        ("inpainting_v2", None),
        ("ltx-video", None),
        ("ltx-video-0.9.1", None),
        ("mochi-1-preview", None),
        ("playground-v2-5", StableDiffusionXLPipeline),
        ("sd3", StableDiffusion3Pipeline),
        ("sd35_large", StableDiffusion3Pipeline),
        ("sd35_medium", StableDiffusion3Pipeline),
        ("stable_cascade_stage_b", None),
        ("stable_cascade_stage_b_lite", None),
        ("stable_cascade_stage_c", None),
        ("stable_cascade_stage_c_lite", None),
        ("upscale", StableDiffusionUpscalePipeline),
        ("v1", StableDiffusionPipeline),
        ("v2", StableDiffusionPipeline),
        ("xl_base", StableDiffusionXLPipeline),
        ("xl_inpaint", None),
        ("xl_refiner", StableDiffusionXLPipeline),
    ]
)

SINGLE_FILE_CHECKPOINT_IMAGE2IMAGE_PIPELINE_MAPPING = OrderedDict(
    [
        ("animatediff_rgb", AnimateDiffPipeline),
        ("animatediff_scribble", AnimateDiffPipeline),
        ("animatediff_sdxl_beta", AnimateDiffSDXLPipeline),
        ("animatediff_v1", AnimateDiffPipeline),
        ("animatediff_v2", AnimateDiffPipeline),
        ("animatediff_v3", AnimateDiffPipeline),
        ("autoencoder-dc-f128c512", None),
        ("autoencoder-dc-f32c32", None),
        ("autoencoder-dc-f32c32-sana", None),
        ("autoencoder-dc-f64c128", None),
        ("controlnet", StableDiffusionControlNetImg2ImgPipeline),
        ("controlnet_xl", StableDiffusionXLControlNetImg2ImgPipeline),
        ("controlnet_xl_large", StableDiffusionXLControlNetImg2ImgPipeline),
        ("controlnet_xl_mid", StableDiffusionXLControlNetImg2ImgPipeline),
        ("controlnet_xl_small", StableDiffusionXLControlNetImg2ImgPipeline),
        ("flux-depth", FluxImg2ImgPipeline),
        ("flux-dev", FluxImg2ImgPipeline),
        ("flux-fill", FluxImg2ImgPipeline),
        ("flux-schnell", FluxImg2ImgPipeline),
        ("hunyuan-video", None),
        ("inpainting", None),
        ("inpainting_v2", None),
        ("ltx-video", None),
        ("ltx-video-0.9.1", None),
        ("mochi-1-preview", None),
        ("playground-v2-5", StableDiffusionXLImg2ImgPipeline),
        ("sd3", StableDiffusion3Img2ImgPipeline),
        ("sd35_large", StableDiffusion3Img2ImgPipeline),
        ("sd35_medium", StableDiffusion3Img2ImgPipeline),
        ("stable_cascade_stage_b", None),
        ("stable_cascade_stage_b_lite", None),
        ("stable_cascade_stage_c", None),
        ("stable_cascade_stage_c_lite", None),
        ("upscale", StableDiffusionUpscalePipeline),
        ("v1", StableDiffusionImg2ImgPipeline),
        ("v2", StableDiffusionImg2ImgPipeline),
        ("xl_base", StableDiffusionXLImg2ImgPipeline),
        ("xl_inpaint", None),
        ("xl_refiner", StableDiffusionXLImg2ImgPipeline),
    ]
)

SINGLE_FILE_CHECKPOINT_INPAINT_PIPELINE_MAPPING = OrderedDict(
    [
        ("animatediff_rgb", None),
        ("animatediff_scribble", None),
        ("animatediff_sdxl_beta", None),
        ("animatediff_v1", None),
        ("animatediff_v2", None),
        ("animatediff_v3", None),
        ("autoencoder-dc-f128c512", None),
        ("autoencoder-dc-f32c32", None),
        ("autoencoder-dc-f32c32-sana", None),
        ("autoencoder-dc-f64c128", None),
        ("controlnet", StableDiffusionControlNetInpaintPipeline),
        ("controlnet_xl", None),
        ("controlnet_xl_large", None),
        ("controlnet_xl_mid", None),
        ("controlnet_xl_small", None),
        ("flux-depth", None),
        ("flux-dev", None),
        ("flux-fill", None),
        ("flux-schnell", None),
        ("hunyuan-video", None),
        ("inpainting", StableDiffusionInpaintPipeline),
        ("inpainting_v2", StableDiffusionInpaintPipeline),
        ("ltx-video", None),
        ("ltx-video-0.9.1", None),
        ("mochi-1-preview", None),
        ("playground-v2-5", None),
        ("sd3", None),
        ("sd35_large", None),
        ("sd35_medium", None),
        ("stable_cascade_stage_b", None),
        ("stable_cascade_stage_b_lite", None),
        ("stable_cascade_stage_c", None),
        ("stable_cascade_stage_c_lite", None),
        ("upscale", StableDiffusionUpscalePipeline),
        ("v1", None),
        ("v2", None),
        ("xl_base", None),
        ("xl_inpaint", StableDiffusionXLInpaintPipeline),
        ("xl_refiner", None),
    ]
)


CONFIG_FILE_LIST = [
    "pytorch_model.bin",
    "pytorch_model.fp16.bin",
    "diffusion_pytorch_model.bin",
    "diffusion_pytorch_model.fp16.bin",
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.fp16.safetensors",
    "diffusion_pytorch_model.ckpt",
    "diffusion_pytorch_model.fp16.ckpt",
    "diffusion_pytorch_model.non_ema.bin",
    "diffusion_pytorch_model.non_ema.safetensors",
]

DIFFUSERS_CONFIG_DIR = [
    "safety_checker",
    "unet",
    "vae",
    "text_encoder",
    "text_encoder_2",
]

TOKENIZER_SHAPE_MAP = {
    768: [
        "SD 1.4",
        "SD 1.5",
        "SD 1.5 LCM",
        "SDXL 0.9",
        "SDXL 1.0",
        "SDXL 1.0 LCM",
        "SDXL Distilled",
        "SDXL Turbo",
        "SDXL Lightning",
        "PixArt a",
        "Playground v2",
        "Pony",
    ],
    1024: ["SD 2.0", "SD 2.0 768", "SD 2.1", "SD 2.1 768", "SD 2.1 Unclip"],
}


EXTENSION = [".safetensors", ".ckpt", ".bin"]

CACHE_HOME = os.path.expanduser("~/.cache")


@dataclass
class RepoStatus:
    r"""
    Data class for storing repository status information.

    Attributes:
        repo_id (`str`):
            The name of the repository.
        repo_hash (`str`):
            The hash of the repository.
        version (`str`):
            The version ID of the repository.
    """

    repo_id: str = ""
    repo_hash: str = ""
    version: str = ""


@dataclass
class ModelStatus:
    r"""
    Data class for storing model status information.

    Attributes:
        search_word (`str`):
            The search word used to find the model.
        download_url (`str`):
            The URL to download the model.
        file_name (`str`):
            The name of the model file.
        local (`bool`):
            Whether the model exists locally
        site_url (`str`):
            The URL of the site where the model is hosted.
    """

    search_word: str = ""
    download_url: str = ""
    file_name: str = ""
    local: bool = False
    site_url: str = ""


@dataclass
class ExtraStatus:
    r"""
    Data class for storing extra status information.

    Attributes:
        trained_words (`str`):
            The words used to trigger the model
    """

    trained_words: Union[List[str], None] = None


@dataclass
class SearchResult:
    r"""
    Data class for storing model data.

    Attributes:
        model_path (`str`):
            The path to the model.
        loading_method (`str`):
            The type of loading method used for the model ( None or 'from_single_file' or 'from_pretrained')
        checkpoint_format (`str`):
            The format of the model checkpoint (`single_file` or `diffusers`).
        repo_status (`RepoStatus`):
            The status of the repository.
        model_status (`ModelStatus`):
            The status of the model.
    """

    model_path: str = ""
    loading_method: Union[str, None] = None
    checkpoint_format: Union[str, None] = None
    repo_status: RepoStatus = field(default_factory=RepoStatus)
    model_status: ModelStatus = field(default_factory=ModelStatus)
    extra_status: ExtraStatus = field(default_factory=ExtraStatus)


@validate_hf_hub_args
def load_pipeline_from_single_file(pretrained_model_or_path, pipeline_mapping, **kwargs):
    r"""
    Instantiate a [`DiffusionPipeline`] from pretrained pipeline weights saved in the `.ckpt` or `.safetensors`
    format. The pipeline is set in evaluation mode (`model.eval()`) by default.

    Parameters:
        pretrained_model_or_path (`str` or `os.PathLike`, *optional*):
            Can be either:
                - A link to the `.ckpt` file (for example
                  `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.ckpt"`) on the Hub.
                - A path to a *file* containing all pipeline weights.
        pipeline_mapping (`dict`):
            A mapping of model types to their corresponding pipeline classes. This is used to determine
            which pipeline class to instantiate based on the model type inferred from the checkpoint.
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Override the default `torch.dtype` and load the model with another dtype.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force the (re-)download of the model weights and configuration files, overriding the
            cached versions if they exist.
        cache_dir (`Union[str, os.PathLike]`, *optional*):
            Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
            is not used.
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
        config (`str`, *optional*):
            Can be either:
                - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                  hosted on the Hub.
                - A path to a *directory* (for example `./my_pipeline_directory/`) containing the pipeline
                  component configs in Diffusers format.
        checkpoint (`dict`, *optional*):
            The loaded state dictionary of the model.
        kwargs (remaining dictionary of keyword arguments, *optional*):
            Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
            class). The overwritten components are passed directly to the pipelines `__init__` method. See example
            below for more information.
    """

    # Load the checkpoint from the provided link or path
    checkpoint = load_single_file_checkpoint(pretrained_model_or_path)

    # Infer the model type from the loaded checkpoint
    model_type = infer_diffusers_model_type(checkpoint)

    # Get the corresponding pipeline class from the pipeline mapping
    pipeline_class = pipeline_mapping[model_type]

    # For tasks not supported by this pipeline
    if pipeline_class is None:
        raise ValueError(
            f"{model_type} is not supported in this pipeline."
            "For `Text2Image`, please use `AutoPipelineForText2Image.from_pretrained`, "
            "for `Image2Image` , please use `AutoPipelineForImage2Image.from_pretrained`, "
            "and `inpaint` is only supported in `AutoPipelineForInpainting.from_pretrained`"
        )

    else:
        # Instantiate and return the pipeline with the loaded checkpoint and any additional kwargs
        return pipeline_class.from_single_file(pretrained_model_or_path, **kwargs)


def get_keyword_types(keyword):
    r"""
    Determine the type and loading method for a given keyword.

    Parameters:
        keyword (`str`):
            The input keyword to classify.

    Returns:
        `dict`: A dictionary containing the model format, loading method,
                and various types and extra types flags.
    """

    # Initialize the status dictionary with default values
    status = {
        "checkpoint_format": None,
        "loading_method": None,
        "type": {
            "other": False,
            "hf_url": False,
            "hf_repo": False,
            "civitai_url": False,
            "local": False,
        },
        "extra_type": {
            "url": False,
            "missing_model_index": None,
        },
    }

    # Check if the keyword is an HTTP or HTTPS URL
    status["extra_type"]["url"] = bool(re.search(r"^(https?)://", keyword))

    # Check if the keyword is a file
    if os.path.isfile(keyword):
        status["type"]["local"] = True
        status["checkpoint_format"] = "single_file"
        status["loading_method"] = "from_single_file"

    # Check if the keyword is a directory
    elif os.path.isdir(keyword):
        status["type"]["local"] = True
        status["checkpoint_format"] = "diffusers"
        status["loading_method"] = "from_pretrained"
        if not os.path.exists(os.path.join(keyword, "model_index.json")):
            status["extra_type"]["missing_model_index"] = True

    # Check if the keyword is a Civitai URL
    elif keyword.startswith("https://civitai.com/"):
        status["type"]["civitai_url"] = True
        status["checkpoint_format"] = "single_file"
        status["loading_method"] = None

    # Check if the keyword starts with any valid URL prefixes
    elif any(keyword.startswith(prefix) for prefix in VALID_URL_PREFIXES):
        repo_id, weights_name = _extract_repo_id_and_weights_name(keyword)
        if weights_name:
            status["type"]["hf_url"] = True
            status["checkpoint_format"] = "single_file"
            status["loading_method"] = "from_single_file"
        else:
            status["type"]["hf_repo"] = True
            status["checkpoint_format"] = "diffusers"
            status["loading_method"] = "from_pretrained"

    # Check if the keyword matches a Hugging Face repository format
    elif re.match(r"^[^/]+/[^/]+$", keyword):
        status["type"]["hf_repo"] = True
        status["checkpoint_format"] = "diffusers"
        status["loading_method"] = "from_pretrained"

    # If none of the above apply
    else:
        status["type"]["other"] = True
        status["checkpoint_format"] = None
        status["loading_method"] = None

    return status


def file_downloader(
    url,
    save_path,
    **kwargs,
) -> None:
    """
    Downloads a file from a given URL and saves it to the specified path.

    parameters:
        url (`str`):
            The URL of the file to download.
        save_path (`str`):
            The local path where the file will be saved.
        resume (`bool`, *optional*, defaults to `False`):
            Whether to resume an incomplete download.
        headers (`dict`, *optional*, defaults to `None`):
            Dictionary of HTTP Headers to send with the request.
        proxies (`dict`, *optional*, defaults to `None`):
            Dictionary mapping protocol to the URL of the proxy passed to `requests.request`.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether to force the download even if the file already exists.
        displayed_filename (`str`, *optional*):
            The filename of the file that is being downloaded. Value is used only to display a nice progress bar. If
            not set, the filename is guessed from the URL or the `Content-Disposition` header.

    returns:
        None
    """

    # Get optional parameters from kwargs, with their default values
    resume = kwargs.pop("resume", False)
    headers = kwargs.pop("headers", None)
    proxies = kwargs.pop("proxies", None)
    force_download = kwargs.pop("force_download", False)
    displayed_filename = kwargs.pop("displayed_filename", None)

    # Default mode for file writing and initial file size
    mode = "wb"
    file_size = 0

    # Create directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Check if the file already exists at the save path
    if os.path.exists(save_path):
        if not force_download:
            # If the file exists and force_download is False, skip the download
            logger.info(f"File already exists: {save_path}, skipping download.")
            return None
        elif resume:
            # If resuming, set mode to append binary and get current file size
            mode = "ab"
            file_size = os.path.getsize(save_path)

    # Open the file in the appropriate mode (write or append)
    with open(save_path, mode) as model_file:
        # Call the http_get function to perform the file download
        return http_get(
            url=url,
            temp_file=model_file,
            resume_size=file_size,
            displayed_filename=displayed_filename,
            headers=headers,
            proxies=proxies,
            **kwargs,
        )


def search_huggingface(search_word: str, **kwargs) -> Union[str, SearchResult, None]:
    r"""
    Downloads a model from Hugging Face.

    Parameters:
        search_word (`str`):
            The search query string.
        revision (`str`, *optional*):
            The specific version of the model to download.
        checkpoint_format (`str`, *optional*, defaults to `"single_file"`):
            The format of the model checkpoint.
        download (`bool`, *optional*, defaults to `False`):
            Whether to download the model.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether to force the download if the model already exists.
        include_params (`bool`, *optional*, defaults to `False`):
            Whether to include parameters in the returned data.
        pipeline_tag (`str`, *optional*):
            Tag to filter models by pipeline.
        token (`str`, *optional*):
            API token for Hugging Face authentication.
        gated (`bool`, *optional*, defaults to `False` ):
            A boolean to filter models on the Hub that are gated or not.
        skip_error (`bool`, *optional*, defaults to `False`):
            Whether to skip errors and return None.

    Returns:
        `Union[str,  SearchResult, None]`: The model path or  SearchResult or None.
    """
    # Extract additional parameters from kwargs
    revision = kwargs.pop("revision", None)
    checkpoint_format = kwargs.pop("checkpoint_format", "single_file")
    download = kwargs.pop("download", False)
    force_download = kwargs.pop("force_download", False)
    include_params = kwargs.pop("include_params", False)
    pipeline_tag = kwargs.pop("pipeline_tag", None)
    token = kwargs.pop("token", None)
    gated = kwargs.pop("gated", False)
    skip_error = kwargs.pop("skip_error", False)

    file_list = []
    hf_repo_info = {}
    hf_security_info = {}
    model_path = ""
    repo_id, file_name = "", ""
    diffusers_model_exists = False

    # Get the type and loading method for the keyword
    search_word_status = get_keyword_types(search_word)

    if search_word_status["type"]["hf_repo"]:
        hf_repo_info = hf_api.model_info(repo_id=search_word, securityStatus=True)
        if download:
            model_path = DiffusionPipeline.download(
                search_word,
                revision=revision,
                token=token,
                force_download=force_download,
                **kwargs,
            )
        else:
            model_path = search_word
    elif search_word_status["type"]["hf_url"]:
        repo_id, weights_name = _extract_repo_id_and_weights_name(search_word)
        if download:
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=weights_name,
                force_download=force_download,
                token=token,
            )
        else:
            model_path = search_word
    elif search_word_status["type"]["local"]:
        model_path = search_word
    elif search_word_status["type"]["civitai_url"]:
        if skip_error:
            return None
        else:
            raise ValueError("The URL for Civitai is invalid with `for_hf`. Please use `for_civitai` instead.")
    else:
        # Get model data from HF API
        hf_models = hf_api.list_models(
            search=search_word,
            direction=-1,
            limit=100,
            fetch_config=True,
            pipeline_tag=pipeline_tag,
            full=True,
            gated=gated,
            token=token,
        )
        model_dicts = [asdict(value) for value in list(hf_models)]

        # Loop through models to find a suitable candidate
        for repo_info in model_dicts:
            repo_id = repo_info["id"]
            file_list = []
            hf_repo_info = hf_api.model_info(repo_id=repo_id, securityStatus=True)
            # Lists files with security issues.
            hf_security_info = hf_repo_info.security_repo_status
            exclusion = [issue["path"] for issue in hf_security_info["filesWithIssues"]]

            # Checks for multi-folder diffusers model or valid files (models with security issues are excluded).
            if hf_security_info["scansDone"]:
                for info in repo_info["siblings"]:
                    file_path = info["rfilename"]
                    if "model_index.json" == file_path and checkpoint_format in [
                        "diffusers",
                        "all",
                    ]:
                        diffusers_model_exists = True
                        break

                    elif (
                        any(file_path.endswith(ext) for ext in EXTENSION)
                        and not any(config in file_path for config in CONFIG_FILE_LIST)
                        and not any(exc in file_path for exc in exclusion)
                        and os.path.basename(os.path.dirname(file_path)) not in DIFFUSERS_CONFIG_DIR
                    ):
                        file_list.append(file_path)

            # Exit from the loop if a multi-folder diffusers model or valid file is found
            if diffusers_model_exists or file_list:
                break
        else:
            # Handle case where no models match the criteria
            if skip_error:
                return None
            else:
                raise ValueError("No models matching your criteria were found on huggingface.")

        if diffusers_model_exists:
            if download:
                model_path = DiffusionPipeline.download(
                    repo_id,
                    token=token,
                    **kwargs,
                )
            else:
                model_path = repo_id

        elif file_list:
            # Sort and find the safest model
            file_name = next(
                (model for model in sorted(file_list, reverse=True) if re.search(r"(?i)[-_](safe|sfw)", model)),
                file_list[0],
            )

            if download:
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_name,
                    revision=revision,
                    token=token,
                    force_download=force_download,
                )

    # `pathlib.PosixPath` may be returned
    if model_path:
        model_path = str(model_path)

    if file_name:
        download_url = f"https://huggingface.co/{repo_id}/blob/main/{file_name}"
    else:
        download_url = f"https://huggingface.co/{repo_id}"

    output_info = get_keyword_types(model_path)

    if include_params:
        return SearchResult(
            model_path=model_path or download_url,
            loading_method=output_info["loading_method"],
            checkpoint_format=output_info["checkpoint_format"],
            repo_status=RepoStatus(repo_id=repo_id, repo_hash=hf_repo_info.sha, version=revision),
            model_status=ModelStatus(
                search_word=search_word,
                site_url=download_url,
                download_url=download_url,
                file_name=file_name,
                local=download,
            ),
            extra_status=ExtraStatus(trained_words=None),
        )

    else:
        return model_path


def search_civitai(search_word: str, **kwargs) -> Union[str, SearchResult, None]:
    r"""
    Downloads a model from Civitai.

    Parameters:
        search_word (`str`):
            The search query string.
        model_type (`str`, *optional*, defaults to `Checkpoint`):
            The type of model to search for.
        sort (`str`, *optional*):
            The order in which you wish to sort the results(for example, `Highest Rated`, `Most Downloaded`, `Newest`).
        base_model (`str`, *optional*):
            The base model to filter by.
        download (`bool`, *optional*, defaults to `False`):
            Whether to download the model.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether to force the download if the model already exists.
        token (`str`, *optional*):
            API token for Civitai authentication.
        include_params (`bool`, *optional*, defaults to `False`):
            Whether to include parameters in the returned data.
        cache_dir (`str`, `Path`, *optional*):
            Path to the folder where cached files are stored.
        resume (`bool`, *optional*, defaults to `False`):
            Whether to resume an incomplete download.
        skip_error (`bool`, *optional*, defaults to `False`):
            Whether to skip errors and return None.

    Returns:
        `Union[str,  SearchResult, None]`: The model path or ` SearchResult` or None.
    """

    # Extract additional parameters from kwargs
    model_type = kwargs.pop("model_type", "Checkpoint")
    sort = kwargs.pop("sort", None)
    download = kwargs.pop("download", False)
    base_model = kwargs.pop("base_model", None)
    force_download = kwargs.pop("force_download", False)
    token = kwargs.pop("token", None)
    include_params = kwargs.pop("include_params", False)
    resume = kwargs.pop("resume", False)
    cache_dir = kwargs.pop("cache_dir", None)
    skip_error = kwargs.pop("skip_error", False)

    # Initialize additional variables with default values
    model_path = ""
    repo_name = ""
    repo_id = ""
    version_id = ""
    trainedWords = ""
    models_list = []
    selected_repo = {}
    selected_model = {}
    selected_version = {}
    civitai_cache_dir = cache_dir or os.path.join(CACHE_HOME, "Civitai")

    # Set up parameters and headers for the CivitAI API request
    params = {
        "query": search_word,
        "types": model_type,
        "limit": 20,
    }
    if base_model is not None:
        if not isinstance(base_model, list):
            base_model = [base_model]
        params["baseModel"] = base_model

    if sort is not None:
        params["sort"] = sort

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        # Make the request to the CivitAI API
        response = requests.get("https://civitai.com/api/v1/models", params=params, headers=headers)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise requests.HTTPError(f"Could not get elements from the URL: {err}")
    else:
        try:
            data = response.json()
        except AttributeError:
            if skip_error:
                return None
            else:
                raise ValueError("Invalid JSON response")

    # Sort repositories by download count in descending order
    sorted_repos = sorted(data["items"], key=lambda x: x["stats"]["downloadCount"], reverse=True)

    for selected_repo in sorted_repos:
        repo_name = selected_repo["name"]
        repo_id = selected_repo["id"]

        # Sort versions within the selected repo by download count
        sorted_versions = sorted(
            selected_repo["modelVersions"],
            key=lambda x: x["stats"]["downloadCount"],
            reverse=True,
        )
        for selected_version in sorted_versions:
            version_id = selected_version["id"]
            trainedWords = selected_version["trainedWords"]
            models_list = []
            # When searching for textual inversion, results other than the values entered for the base model may come up, so check again.
            if base_model is None or selected_version["baseModel"] in base_model:
                for model_data in selected_version["files"]:
                    # Check if the file passes security scans and has a valid extension
                    file_name = model_data["name"]
                    if (
                        model_data["pickleScanResult"] == "Success"
                        and model_data["virusScanResult"] == "Success"
                        and any(file_name.endswith(ext) for ext in EXTENSION)
                        and os.path.basename(os.path.dirname(file_name)) not in DIFFUSERS_CONFIG_DIR
                    ):
                        file_status = {
                            "filename": file_name,
                            "download_url": model_data["downloadUrl"],
                        }
                        models_list.append(file_status)

            if models_list:
                # Sort the models list by filename and find the safest model
                sorted_models = sorted(models_list, key=lambda x: x["filename"], reverse=True)
                selected_model = next(
                    (
                        model_data
                        for model_data in sorted_models
                        if bool(re.search(r"(?i)[-_](safe|sfw)", model_data["filename"]))
                    ),
                    sorted_models[0],
                )

                break
        else:
            continue
        break

    # Exception handling when search candidates are not found
    if not selected_model:
        if skip_error:
            return None
        else:
            raise ValueError("No model found. Please try changing the word you are searching for.")

    # Define model file status
    file_name = selected_model["filename"]
    download_url = selected_model["download_url"]

    # Handle file download and setting model information
    if download:
        # The path where the model is to be saved.
        model_path = os.path.join(str(civitai_cache_dir), str(repo_id), str(version_id), str(file_name))
        # Download Model File
        file_downloader(
            url=download_url,
            save_path=model_path,
            resume=resume,
            force_download=force_download,
            displayed_filename=file_name,
            headers=headers,
            **kwargs,
        )

    else:
        model_path = download_url

    output_info = get_keyword_types(model_path)

    if not include_params:
        return model_path
    else:
        return SearchResult(
            model_path=model_path,
            loading_method=output_info["loading_method"],
            checkpoint_format=output_info["checkpoint_format"],
            repo_status=RepoStatus(repo_id=repo_name, repo_hash=repo_id, version=version_id),
            model_status=ModelStatus(
                search_word=search_word,
                site_url=f"https://civitai.com/models/{repo_id}?modelVersionId={version_id}",
                download_url=download_url,
                file_name=file_name,
                local=output_info["type"]["local"],
            ),
            extra_status=ExtraStatus(trained_words=trainedWords or None),
        )


def add_methods(pipeline):
    r"""
    Add methods from `AutoConfig` to the pipeline.

    Parameters:
        pipeline (`Pipeline`):
            The pipeline to which the methods will be added.
    """
    for attr_name in dir(AutoConfig):
        attr_value = getattr(AutoConfig, attr_name)
        if callable(attr_value) and not attr_name.startswith("__"):
            setattr(pipeline, attr_name, types.MethodType(attr_value, pipeline))
    return pipeline


class AutoConfig:
    def auto_load_textual_inversion(
        self,
        pretrained_model_name_or_path: Union[str, List[str]],
        token: Optional[Union[str, List[str]]] = None,
        base_model: Optional[Union[str, List[str]]] = None,
        tokenizer=None,
        text_encoder=None,
        **kwargs,
    ):
        r"""
        Load Textual Inversion embeddings into the text encoder of [`StableDiffusionPipeline`] (both 🤗 Diffusers and
        Automatic1111 formats are supported).

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike` or `List[str or os.PathLike]` or `Dict` or `List[Dict]`):
                Can be either one of the following or a list of them:

                    - Search keywords for pretrained model (for example `EasyNegative`).
                    - A string, the *model id* (for example `sd-concepts-library/low-poly-hd-logos-icons`) of a
                      pretrained model hosted on the Hub.
                    - A path to a *directory* (for example `./my_text_inversion_directory/`) containing the textual
                      inversion weights.
                    - A path to a *file* (for example `./my_text_inversions.pt`) containing textual inversion weights.
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            token (`str` or `List[str]`, *optional*):
                Override the token to use for the textual inversion weights. If `pretrained_model_name_or_path` is a
                list, then `token` must also be a list of equal length.
            text_encoder ([`~transformers.CLIPTextModel`], *optional*):
                Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
                If not specified, function will take self.tokenizer.
            tokenizer ([`~transformers.CLIPTokenizer`], *optional*):
                A `CLIPTokenizer` to tokenize text. If not specified, function will take self.tokenizer.
            weight_name (`str`, *optional*):
                Name of a custom weight file. This should be used when:

                    - The saved textual inversion file is in 🤗 Diffusers format, but was saved under a specific weight
                      name such as `text_inv.bin`.
                    - The saved textual inversion file is in the Automatic1111 format.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.

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
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you're downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.

        Examples:

        ```py
        >>> from auto_diffusers import EasyPipelineForText2Image

        >>> pipeline = EasyPipelineForText2Image.from_huggingface("stable-diffusion-v1-5")

        >>> pipeline.auto_load_textual_inversion("EasyNegative", token="EasyNegative")

        >>> image = pipeline(prompt).images[0]
        ```

        """
        # 1. Set tokenizer and text encoder
        tokenizer = tokenizer or getattr(self, "tokenizer", None)
        text_encoder = text_encoder or getattr(self, "text_encoder", None)

        # Check if tokenizer and text encoder are provided
        if tokenizer is None or text_encoder is None:
            raise ValueError("Tokenizer and text encoder must be provided.")

        # 2. Normalize inputs
        pretrained_model_name_or_paths = (
            [pretrained_model_name_or_path]
            if not isinstance(pretrained_model_name_or_path, list)
            else pretrained_model_name_or_path
        )

        # 2.1 Normalize tokens
        tokens = [token] if not isinstance(token, list) else token
        if tokens[0] is None:
            tokens = tokens * len(pretrained_model_name_or_paths)

        for check_token in tokens:
            # Check if token is already in tokenizer vocabulary
            if check_token in tokenizer.get_vocab():
                raise ValueError(
                    f"Token {token} already in tokenizer vocabulary. Please choose a different token name or remove {token} and embedding from the tokenizer and text encoder."
                )

        expected_shape = text_encoder.get_input_embeddings().weight.shape[-1]  # Expected shape of tokenizer

        for search_word in pretrained_model_name_or_paths:
            if isinstance(search_word, str):
                # Update kwargs to ensure the model is downloaded and parameters are included
                _status = {
                    "download": True,
                    "include_params": True,
                    "skip_error": False,
                    "model_type": "TextualInversion",
                }
                # Get tags for the base model of textual inversion compatible with tokenizer.
                # If the tokenizer is 768-dimensional, set tags for SD 1.x and SDXL.
                # If the tokenizer is 1024-dimensional, set tags for SD 2.x.
                if expected_shape in TOKENIZER_SHAPE_MAP:
                    # Retrieve the appropriate tags from the TOKENIZER_SHAPE_MAP based on the expected shape
                    tags = TOKENIZER_SHAPE_MAP[expected_shape]
                    if base_model is not None:
                        if isinstance(base_model, list):
                            tags.extend(base_model)
                        else:
                            tags.append(base_model)
                    _status["base_model"] = tags

                kwargs.update(_status)
                # Search for the model on Civitai and get the model status
                textual_inversion_path = search_civitai(search_word, **kwargs)
                logger.warning(
                    f"textual_inversion_path: {search_word} -> {textual_inversion_path.model_status.site_url}"
                )

                pretrained_model_name_or_paths[pretrained_model_name_or_paths.index(search_word)] = (
                    textual_inversion_path.model_path
                )

        self.load_textual_inversion(
            pretrained_model_name_or_paths, token=tokens, tokenizer=tokenizer, text_encoder=text_encoder, **kwargs
        )

    def auto_load_lora_weights(
        self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]], adapter_name=None, **kwargs
    ):
        r"""
        Load LoRA weights specified in `pretrained_model_name_or_path_or_dict` into `self.unet` and
        `self.text_encoder`.

        All kwargs are forwarded to `self.lora_state_dict`.

        See [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`] for more details on how the state dict is
        loaded.

        See [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_into_unet`] for more details on how the state dict is
        loaded into `self.unet`.

        See [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_into_text_encoder`] for more details on how the state
        dict is loaded into `self.text_encoder`.

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                See [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`].
            adapter_name (`str`, *optional*):
                Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
                `default_{i}` where i is the total number of adapters being loaded.
            low_cpu_mem_usage (`bool`, *optional*):
                Speed up model loading by only loading the pretrained LoRA weights and not initializing the random
                weights.
            kwargs (`dict`, *optional*):
                See [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`].
        """
        if isinstance(pretrained_model_name_or_path_or_dict, str):
            # Update kwargs to ensure the model is downloaded and parameters are included
            _status = {
                "download": True,
                "include_params": True,
                "skip_error": False,
                "model_type": "LORA",
            }
            kwargs.update(_status)
            # Search for the model on Civitai and get the model status
            lora_path = search_civitai(pretrained_model_name_or_path_or_dict, **kwargs)
            logger.warning(f"lora_path: {lora_path.model_status.site_url}")
            logger.warning(f"trained_words: {lora_path.extra_status.trained_words}")
            pretrained_model_name_or_path_or_dict = lora_path.model_path

        self.load_lora_weights(pretrained_model_name_or_path_or_dict, adapter_name=adapter_name, **kwargs)


class EasyPipelineForText2Image(AutoPipelineForText2Image):
    r"""
    [`EasyPipelineForText2Image`] is a generic pipeline class that instantiates a text-to-image pipeline class. The
    specific underlying pipeline class is automatically selected from either the
    [`~EasyPipelineForText2Image.from_pretrained`], [`~EasyPipelineForText2Image.from_pipe`], [`~EasyPipelineForText2Image.from_huggingface`] or [`~EasyPipelineForText2Image.from_civitai`] methods.

    This class cannot be instantiated using `__init__()` (throws an error).

    Class attributes:

        - **config_name** (`str`) -- The configuration filename that stores the class and module names of all the
          diffusion pipeline's components.

    """

    config_name = "model_index.json"

    def __init__(self, *args, **kwargs):
        # EnvironmentError is returned
        super().__init__()

    @classmethod
    @validate_hf_hub_args
    def from_huggingface(cls, pretrained_model_link_or_path, **kwargs):
        r"""
        Parameters:
            pretrained_model_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A keyword to search for Hugging Face (for example `Stable Diffusion`)
                    - Link to `.ckpt` or `.safetensors` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.safetensors"`) on the Hub.
                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                      saved using
                    [`~DiffusionPipeline.save_pretrained`].
            checkpoint_format (`str`, *optional*, defaults to `"single_file"`):
                The format of the model checkpoint.
            pipeline_tag (`str`, *optional*):
                Tag to filter models by pipeline.
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If "auto" is passed, the
                dtype is automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.

            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            custom_revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
                `revision` when loading a custom pipeline from the Hub. It can be a 🤗 Diffusers version when loading a
                custom pipeline from GitHub, otherwise it defaults to `"main"` when loading from the Hub.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn’t need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if device_map contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            gated (`bool`, *optional*, defaults to `False` ):
                A boolean to filter models on the Hub that are gated or not.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.
            variant (`str`, *optional*):
                Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.

        > [!TIP]
        > To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        > `hf auth login`.

        Examples:

        ```py
        >>> from auto_diffusers import EasyPipelineForText2Image

        >>> pipeline = EasyPipelineForText2Image.from_huggingface("stable-diffusion-v1-5")
        >>> image = pipeline(prompt).images[0]
        ```
        """
        # Update kwargs to ensure the model is downloaded and parameters are included
        _status = {
            "download": True,
            "include_params": True,
            "skip_error": False,
            "pipeline_tag": "text-to-image",
        }
        kwargs.update(_status)

        # Search for the model on Hugging Face and get the model status
        hf_checkpoint_status = search_huggingface(pretrained_model_link_or_path, **kwargs)
        logger.warning(f"checkpoint_path: {hf_checkpoint_status.model_status.download_url}")
        checkpoint_path = hf_checkpoint_status.model_path

        # Check the format of the model checkpoint
        if hf_checkpoint_status.loading_method == "from_single_file":
            # Load the pipeline from a single file checkpoint
            pipeline = load_pipeline_from_single_file(
                pretrained_model_or_path=checkpoint_path,
                pipeline_mapping=SINGLE_FILE_CHECKPOINT_TEXT2IMAGE_PIPELINE_MAPPING,
                **kwargs,
            )
        else:
            pipeline = cls.from_pretrained(checkpoint_path, **kwargs)
        return add_methods(pipeline)

    @classmethod
    def from_civitai(cls, pretrained_model_link_or_path, **kwargs):
        r"""
        Parameters:
            pretrained_model_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A keyword to search for Hugging Face (for example `Stable Diffusion`)
                    - Link to `.ckpt` or `.safetensors` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.safetensors"`) on the Hub.
                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                      saved using
                    [`~DiffusionPipeline.save_pretrained`].
            model_type (`str`, *optional*, defaults to `Checkpoint`):
                The type of model to search for. (for example `Checkpoint`, `TextualInversion`, `LORA`, `Controlnet`)
            base_model (`str`, *optional*):
                The base model to filter by.
            cache_dir (`str`, `Path`, *optional*):
                Path to the folder where cached files are stored.
            resume (`bool`, *optional*, defaults to `False`):
                Whether to resume an incomplete download.
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If "auto" is passed, the
                dtype is automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str`, *optional*):
                The token to use as HTTP bearer authorization for remote files.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn’t need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if device_map contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.

        > [!TIP]
        > To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        > `hf auth login`.

        Examples:

        ```py
        >>> from auto_diffusers import EasyPipelineForText2Image

        >>> pipeline = EasyPipelineForText2Image.from_huggingface("stable-diffusion-v1-5")
        >>> image = pipeline(prompt).images[0]
        ```
        """
        # Update kwargs to ensure the model is downloaded and parameters are included
        _status = {
            "download": True,
            "include_params": True,
            "skip_error": False,
            "model_type": "Checkpoint",
        }
        kwargs.update(_status)

        # Search for the model on Civitai and get the model status
        checkpoint_status = search_civitai(pretrained_model_link_or_path, **kwargs)
        logger.warning(f"checkpoint_path: {checkpoint_status.model_status.site_url}")
        checkpoint_path = checkpoint_status.model_path

        # Load the pipeline from a single file checkpoint
        pipeline = load_pipeline_from_single_file(
            pretrained_model_or_path=checkpoint_path,
            pipeline_mapping=SINGLE_FILE_CHECKPOINT_TEXT2IMAGE_PIPELINE_MAPPING,
            **kwargs,
        )
        return add_methods(pipeline)


class EasyPipelineForImage2Image(AutoPipelineForImage2Image):
    r"""

    [`EasyPipelineForImage2Image`] is a generic pipeline class that instantiates an image-to-image pipeline class. The
    specific underlying pipeline class is automatically selected from either the
    [`~EasyPipelineForImage2Image.from_pretrained`], [`~EasyPipelineForImage2Image.from_pipe`], [`~EasyPipelineForImage2Image.from_huggingface`] or [`~EasyPipelineForImage2Image.from_civitai`] methods.

    This class cannot be instantiated using `__init__()` (throws an error).

    Class attributes:

        - **config_name** (`str`) -- The configuration filename that stores the class and module names of all the
          diffusion pipeline's components.

    """

    config_name = "model_index.json"

    def __init__(self, *args, **kwargs):
        # EnvironmentError is returned
        super().__init__()

    @classmethod
    @validate_hf_hub_args
    def from_huggingface(cls, pretrained_model_link_or_path, **kwargs):
        r"""
        Parameters:
            pretrained_model_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A keyword to search for Hugging Face (for example `Stable Diffusion`)
                    - Link to `.ckpt` or `.safetensors` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.safetensors"`) on the Hub.
                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                      saved using
                    [`~DiffusionPipeline.save_pretrained`].
            checkpoint_format (`str`, *optional*, defaults to `"single_file"`):
                The format of the model checkpoint.
            pipeline_tag (`str`, *optional*):
                Tag to filter models by pipeline.
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If "auto" is passed, the
                dtype is automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.

            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            custom_revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
                `revision` when loading a custom pipeline from the Hub. It can be a 🤗 Diffusers version when loading a
                custom pipeline from GitHub, otherwise it defaults to `"main"` when loading from the Hub.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn’t need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if device_map contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            gated (`bool`, *optional*, defaults to `False` ):
                A boolean to filter models on the Hub that are gated or not.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.
            variant (`str`, *optional*):
                Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.

        > [!TIP]
        > To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        > `hf auth login`.

        Examples:

        ```py
        >>> from auto_diffusers import EasyPipelineForImage2Image

        >>> pipeline = EasyPipelineForImage2Image.from_huggingface("stable-diffusion-v1-5")
        >>> image = pipeline(prompt, image).images[0]
        ```
        """
        # Update kwargs to ensure the model is downloaded and parameters are included
        _parmas = {
            "download": True,
            "include_params": True,
            "skip_error": False,
            "pipeline_tag": "image-to-image",
        }
        kwargs.update(_parmas)

        # Search for the model on Hugging Face and get the model status
        hf_checkpoint_status = search_huggingface(pretrained_model_link_or_path, **kwargs)
        logger.warning(f"checkpoint_path: {hf_checkpoint_status.model_status.download_url}")
        checkpoint_path = hf_checkpoint_status.model_path

        # Check the format of the model checkpoint
        if hf_checkpoint_status.loading_method == "from_single_file":
            # Load the pipeline from a single file checkpoint
            pipeline = load_pipeline_from_single_file(
                pretrained_model_or_path=checkpoint_path,
                pipeline_mapping=SINGLE_FILE_CHECKPOINT_IMAGE2IMAGE_PIPELINE_MAPPING,
                **kwargs,
            )
        else:
            pipeline = cls.from_pretrained(checkpoint_path, **kwargs)

        return add_methods(pipeline)

    @classmethod
    def from_civitai(cls, pretrained_model_link_or_path, **kwargs):
        r"""
        Parameters:
            pretrained_model_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A keyword to search for Hugging Face (for example `Stable Diffusion`)
                    - Link to `.ckpt` or `.safetensors` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.safetensors"`) on the Hub.
                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                      saved using
                    [`~DiffusionPipeline.save_pretrained`].
            model_type (`str`, *optional*, defaults to `Checkpoint`):
                The type of model to search for. (for example `Checkpoint`, `TextualInversion`, `LORA`, `Controlnet`)
            base_model (`str`, *optional*):
                The base model to filter by.
            cache_dir (`str`, `Path`, *optional*):
                Path to the folder where cached files are stored.
            resume (`bool`, *optional*, defaults to `False`):
                Whether to resume an incomplete download.
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If "auto" is passed, the
                dtype is automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str`, *optional*):
                The token to use as HTTP bearer authorization for remote files.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn’t need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if device_map contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.

        > [!TIP]
        > To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        > `hf auth login`.

        Examples:

        ```py
        >>> from auto_diffusers import EasyPipelineForImage2Image

        >>> pipeline = EasyPipelineForImage2Image.from_huggingface("stable-diffusion-v1-5")
        >>> image = pipeline(prompt, image).images[0]
        ```
        """
        # Update kwargs to ensure the model is downloaded and parameters are included
        _status = {
            "download": True,
            "include_params": True,
            "skip_error": False,
            "model_type": "Checkpoint",
        }
        kwargs.update(_status)

        # Search for the model on Civitai and get the model status
        checkpoint_status = search_civitai(pretrained_model_link_or_path, **kwargs)
        logger.warning(f"checkpoint_path: {checkpoint_status.model_status.site_url}")
        checkpoint_path = checkpoint_status.model_path

        # Load the pipeline from a single file checkpoint
        pipeline = load_pipeline_from_single_file(
            pretrained_model_or_path=checkpoint_path,
            pipeline_mapping=SINGLE_FILE_CHECKPOINT_IMAGE2IMAGE_PIPELINE_MAPPING,
            **kwargs,
        )
        return add_methods(pipeline)


class EasyPipelineForInpainting(AutoPipelineForInpainting):
    r"""

    [`EasyPipelineForInpainting`] is a generic pipeline class that instantiates an inpainting pipeline class. The
    specific underlying pipeline class is automatically selected from either the
    [`~EasyPipelineForInpainting.from_pretrained`], [`~EasyPipelineForInpainting.from_pipe`], [`~EasyPipelineForInpainting.from_huggingface`] or [`~EasyPipelineForInpainting.from_civitai`] methods.

    This class cannot be instantiated using `__init__()` (throws an error).

    Class attributes:

        - **config_name** (`str`) -- The configuration filename that stores the class and module names of all the
          diffusion pipeline's components.

    """

    config_name = "model_index.json"

    def __init__(self, *args, **kwargs):
        # EnvironmentError is returned
        super().__init__()

    @classmethod
    @validate_hf_hub_args
    def from_huggingface(cls, pretrained_model_link_or_path, **kwargs):
        r"""
        Parameters:
            pretrained_model_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A keyword to search for Hugging Face (for example `Stable Diffusion`)
                    - Link to `.ckpt` or `.safetensors` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.safetensors"`) on the Hub.
                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                      saved using
                    [`~DiffusionPipeline.save_pretrained`].
            checkpoint_format (`str`, *optional*, defaults to `"single_file"`):
                The format of the model checkpoint.
            pipeline_tag (`str`, *optional*):
                Tag to filter models by pipeline.
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If "auto" is passed, the
                dtype is automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.

            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            custom_revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
                `revision` when loading a custom pipeline from the Hub. It can be a 🤗 Diffusers version when loading a
                custom pipeline from GitHub, otherwise it defaults to `"main"` when loading from the Hub.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn’t need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if device_map contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            gated (`bool`, *optional*, defaults to `False` ):
                A boolean to filter models on the Hub that are gated or not.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.
            variant (`str`, *optional*):
                Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.

        > [!TIP]
        > To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        > `hf auth login

        Examples:

        ```py
        >>> from auto_diffusers import EasyPipelineForInpainting

        >>> pipeline = EasyPipelineForInpainting.from_huggingface("stable-diffusion-2-inpainting")
        >>> image = pipeline(prompt, image=init_image, mask_image=mask_image).images[0]
        ```
        """
        # Update kwargs to ensure the model is downloaded and parameters are included
        _status = {
            "download": True,
            "include_params": True,
            "skip_error": False,
            "pipeline_tag": "image-to-image",
        }
        kwargs.update(_status)

        # Search for the model on Hugging Face and get the model status
        hf_checkpoint_status = search_huggingface(pretrained_model_link_or_path, **kwargs)
        logger.warning(f"checkpoint_path: {hf_checkpoint_status.model_status.download_url}")
        checkpoint_path = hf_checkpoint_status.model_path

        # Check the format of the model checkpoint
        if hf_checkpoint_status.loading_method == "from_single_file":
            # Load the pipeline from a single file checkpoint
            pipeline = load_pipeline_from_single_file(
                pretrained_model_or_path=checkpoint_path,
                pipeline_mapping=SINGLE_FILE_CHECKPOINT_INPAINT_PIPELINE_MAPPING,
                **kwargs,
            )
        else:
            pipeline = cls.from_pretrained(checkpoint_path, **kwargs)
        return add_methods(pipeline)

    @classmethod
    def from_civitai(cls, pretrained_model_link_or_path, **kwargs):
        r"""
        Parameters:
            pretrained_model_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A keyword to search for Hugging Face (for example `Stable Diffusion`)
                    - Link to `.ckpt` or `.safetensors` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.safetensors"`) on the Hub.
                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                      saved using
                    [`~DiffusionPipeline.save_pretrained`].
            model_type (`str`, *optional*, defaults to `Checkpoint`):
                The type of model to search for. (for example `Checkpoint`, `TextualInversion`, `LORA`, `Controlnet`)
            base_model (`str`, *optional*):
                The base model to filter by.
            cache_dir (`str`, `Path`, *optional*):
                Path to the folder where cached files are stored.
            resume (`bool`, *optional*, defaults to `False`):
                Whether to resume an incomplete download.
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If "auto" is passed, the
                dtype is automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str`, *optional*):
                The token to use as HTTP bearer authorization for remote files.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn’t need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if device_map contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.

        > [!TIP]
        > To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        > `hf auth login

        Examples:

        ```py
        >>> from auto_diffusers import EasyPipelineForInpainting

        >>> pipeline = EasyPipelineForInpainting.from_huggingface("stable-diffusion-2-inpainting")
        >>> image = pipeline(prompt, image=init_image, mask_image=mask_image).images[0]
        ```
        """
        # Update kwargs to ensure the model is downloaded and parameters are included
        _status = {
            "download": True,
            "include_params": True,
            "skip_error": False,
            "model_type": "Checkpoint",
        }
        kwargs.update(_status)

        # Search for the model on Civitai and get the model status
        checkpoint_status = search_civitai(pretrained_model_link_or_path, **kwargs)
        logger.warning(f"checkpoint_path: {checkpoint_status.model_status.site_url}")
        checkpoint_path = checkpoint_status.model_path

        # Load the pipeline from a single file checkpoint
        pipeline = load_pipeline_from_single_file(
            pretrained_model_or_path=checkpoint_path,
            pipeline_mapping=SINGLE_FILE_CHECKPOINT_INPAINT_PIPELINE_MAPPING,
            **kwargs,
        )
        return add_methods(pipeline)
