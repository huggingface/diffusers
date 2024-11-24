# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
import requests
from tqdm.auto import tqdm
from collections import OrderedDict
from dataclasses import (
    dataclass,
    asdict
)


from huggingface_hub.utils import validate_hf_hub_args
from huggingface_hub import (
    hf_api,
    hf_hub_download,
)

from ..configuration_utils import ConfigMixin
from ..utils import (
    logging,
    is_sentencepiece_available
)
from ..loaders.single_file_utils import (
    infer_diffusers_model_type,
    load_single_file_checkpoint,
    _extract_repo_id_and_weights_name,
    VALID_URL_PREFIXES,
)

from .aura_flow import AuraFlowPipeline
from .cogview3 import CogView3PlusPipeline
from .controlnet import (
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLControlNetPipeline,
)
from .deepfloyd_if import IFImg2ImgPipeline, IFInpaintingPipeline, IFPipeline
from .flux import (
    FluxControlNetImg2ImgPipeline,
    FluxControlNetInpaintPipeline,
    FluxControlNetPipeline,
    FluxImg2ImgPipeline,
    FluxInpaintPipeline,
    FluxPipeline,
)
from .hunyuandit import HunyuanDiTPipeline
from .kandinsky import (
    KandinskyCombinedPipeline,
    KandinskyImg2ImgCombinedPipeline,
    KandinskyImg2ImgPipeline,
    KandinskyInpaintCombinedPipeline,
    KandinskyInpaintPipeline,
    KandinskyPipeline,
)
from .kandinsky2_2 import (
    KandinskyV22CombinedPipeline,
    KandinskyV22Img2ImgCombinedPipeline,
    KandinskyV22Img2ImgPipeline,
    KandinskyV22InpaintCombinedPipeline,
    KandinskyV22InpaintPipeline,
    KandinskyV22Pipeline,
)
from .kandinsky3 import Kandinsky3Img2ImgPipeline, Kandinsky3Pipeline
from .latent_consistency_models import LatentConsistencyModelImg2ImgPipeline, LatentConsistencyModelPipeline
from .lumina import LuminaText2ImgPipeline
from .pag import (
    HunyuanDiTPAGPipeline,
    PixArtSigmaPAGPipeline,
    StableDiffusion3PAGPipeline,
    StableDiffusionControlNetPAGInpaintPipeline,
    StableDiffusionControlNetPAGPipeline,
    StableDiffusionPAGImg2ImgPipeline,
    StableDiffusionPAGPipeline,
    StableDiffusionXLControlNetPAGImg2ImgPipeline,
    StableDiffusionXLControlNetPAGPipeline,
    StableDiffusionXLPAGImg2ImgPipeline,
    StableDiffusionXLPAGInpaintPipeline,
    StableDiffusionXLPAGPipeline,
)
from .pixart_alpha import PixArtAlphaPipeline, PixArtSigmaPipeline
from .stable_cascade import StableCascadeCombinedPipeline, StableCascadeDecoderPipeline
from .stable_diffusion import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
)
from .stable_diffusion_3 import (
    StableDiffusion3Img2ImgPipeline,
    StableDiffusion3InpaintPipeline,
    StableDiffusion3Pipeline,
)
from .stable_diffusion_xl import (
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)
from .wuerstchen import WuerstchenCombinedPipeline, WuerstchenDecoderPipeline
from .pipeline_utils import DiffusionPipeline

logger = logging.get_logger(__name__)


AUTO_TEXT2IMAGE_PIPELINES_MAPPING = OrderedDict(
    [
        ("stable-diffusion", StableDiffusionPipeline),
        ("stable-diffusion-xl", StableDiffusionXLPipeline),
        ("stable-diffusion-3", StableDiffusion3Pipeline),
        ("stable-diffusion-3-pag", StableDiffusion3PAGPipeline),
        ("if", IFPipeline),
        ("hunyuan", HunyuanDiTPipeline),
        ("hunyuan-pag", HunyuanDiTPAGPipeline),
        ("kandinsky", KandinskyCombinedPipeline),
        ("kandinsky22", KandinskyV22CombinedPipeline),
        ("kandinsky3", Kandinsky3Pipeline),
        ("stable-diffusion-controlnet", StableDiffusionControlNetPipeline),
        ("stable-diffusion-xl-controlnet", StableDiffusionXLControlNetPipeline),
        ("wuerstchen", WuerstchenCombinedPipeline),
        ("cascade", StableCascadeCombinedPipeline),
        ("lcm", LatentConsistencyModelPipeline),
        ("pixart-alpha", PixArtAlphaPipeline),
        ("pixart-sigma", PixArtSigmaPipeline),
        ("stable-diffusion-pag", StableDiffusionPAGPipeline),
        ("stable-diffusion-controlnet-pag", StableDiffusionControlNetPAGPipeline),
        ("stable-diffusion-xl-pag", StableDiffusionXLPAGPipeline),
        ("stable-diffusion-xl-controlnet-pag", StableDiffusionXLControlNetPAGPipeline),
        ("pixart-sigma-pag", PixArtSigmaPAGPipeline),
        ("auraflow", AuraFlowPipeline),
        ("flux", FluxPipeline),
        ("flux-controlnet", FluxControlNetPipeline),
        ("lumina", LuminaText2ImgPipeline),
        ("cogview3", CogView3PlusPipeline),
    ]
)

AUTO_IMAGE2IMAGE_PIPELINES_MAPPING = OrderedDict(
    [
        ("stable-diffusion", StableDiffusionImg2ImgPipeline),
        ("stable-diffusion-xl", StableDiffusionXLImg2ImgPipeline),
        ("stable-diffusion-3", StableDiffusion3Img2ImgPipeline),
        ("if", IFImg2ImgPipeline),
        ("kandinsky", KandinskyImg2ImgCombinedPipeline),
        ("kandinsky22", KandinskyV22Img2ImgCombinedPipeline),
        ("kandinsky3", Kandinsky3Img2ImgPipeline),
        ("stable-diffusion-controlnet", StableDiffusionControlNetImg2ImgPipeline),
        ("stable-diffusion-pag", StableDiffusionPAGImg2ImgPipeline),
        ("stable-diffusion-xl-controlnet", StableDiffusionXLControlNetImg2ImgPipeline),
        ("stable-diffusion-xl-pag", StableDiffusionXLPAGImg2ImgPipeline),
        ("stable-diffusion-xl-controlnet-pag", StableDiffusionXLControlNetPAGImg2ImgPipeline),
        ("lcm", LatentConsistencyModelImg2ImgPipeline),
        ("flux", FluxImg2ImgPipeline),
        ("flux-controlnet", FluxControlNetImg2ImgPipeline),
    ]
)

AUTO_INPAINT_PIPELINES_MAPPING = OrderedDict(
    [
        ("stable-diffusion", StableDiffusionInpaintPipeline),
        ("stable-diffusion-xl", StableDiffusionXLInpaintPipeline),
        ("stable-diffusion-3", StableDiffusion3InpaintPipeline),
        ("if", IFInpaintingPipeline),
        ("kandinsky", KandinskyInpaintCombinedPipeline),
        ("kandinsky22", KandinskyV22InpaintCombinedPipeline),
        ("stable-diffusion-controlnet", StableDiffusionControlNetInpaintPipeline),
        ("stable-diffusion-controlnet-pag", StableDiffusionControlNetPAGInpaintPipeline),
        ("stable-diffusion-xl-controlnet", StableDiffusionXLControlNetInpaintPipeline),
        ("stable-diffusion-xl-pag", StableDiffusionXLPAGInpaintPipeline),
        ("flux", FluxInpaintPipeline),
        ("flux-controlnet", FluxControlNetInpaintPipeline),
    ]
)

_AUTO_TEXT2IMAGE_DECODER_PIPELINES_MAPPING = OrderedDict(
    [
        ("kandinsky", KandinskyPipeline),
        ("kandinsky22", KandinskyV22Pipeline),
        ("wuerstchen", WuerstchenDecoderPipeline),
        ("cascade", StableCascadeDecoderPipeline),
    ]
)
_AUTO_IMAGE2IMAGE_DECODER_PIPELINES_MAPPING = OrderedDict(
    [
        ("kandinsky", KandinskyImg2ImgPipeline),
        ("kandinsky22", KandinskyV22Img2ImgPipeline),
    ]
)
_AUTO_INPAINT_DECODER_PIPELINES_MAPPING = OrderedDict(
    [
        ("kandinsky", KandinskyInpaintPipeline),
        ("kandinsky22", KandinskyV22InpaintPipeline),
    ]
)


SINGLE_FILE_CHECKPOINT_TEXT2IMAGE_PIPELINE_MAPPING = OrderedDict(
    [
        ("xl_base", StableDiffusionXLPipeline),
        ("xl_refiner", StableDiffusionXLPipeline),
        ("xl_inpaint", None),
        ("playground-v2-5", StableDiffusionXLPipeline),
        ("upscale", None),
        ("inpainting", None),
        ("inpainting_v2", None),
        ("controlnet", StableDiffusionControlNetPipeline),
        ("v2", StableDiffusionPipeline),
        ("v1", StableDiffusionPipeline),
    ]
)

SINGLE_FILE_CHECKPOINT_IMAGE2IMAGE_PIPELINE_MAPPING = OrderedDict(
    [
        ("xl_base", StableDiffusionXLImg2ImgPipeline),
        ("xl_refiner", StableDiffusionXLImg2ImgPipeline),
        ("xl_inpaint", None),
        ("playground-v2-5", StableDiffusionXLImg2ImgPipeline),
        ("upscale", None),
        ("inpainting", None),
        ("inpainting_v2", None),
        ("controlnet", StableDiffusionControlNetImg2ImgPipeline),
        ("v2", StableDiffusionImg2ImgPipeline),
        ("v1", StableDiffusionImg2ImgPipeline),
    ]
)

SINGLE_FILE_CHECKPOINT_INPAINT_PIPELINE_MAPPING = OrderedDict(
    [
        ("xl_base", None),
        ("xl_refiner", None),
        ("xl_inpaint", StableDiffusionXLInpaintPipeline),
        ("playground-v2-5", None),
        ("upscale", None),
        ("inpainting", StableDiffusionInpaintPipeline),
        ("inpainting_v2", StableDiffusionInpaintPipeline),
        ("controlnet", StableDiffusionControlNetInpaintPipeline),
        ("v2", None),
        ("v1", None),
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

DIFFUSERS_CONFIG_DIR = ["safety_checker", "unet", "vae", "text_encoder", "text_encoder_2"]

INPAINT_PIPELINE_KEYS = [
    "xl_inpaint",
    "inpainting",
    "inpainting_v2",
]

EXTENSION =  [".safetensors", ".ckpt", ".bin"]


if is_sentencepiece_available():
    from .kolors import KolorsImg2ImgPipeline, KolorsPipeline
    from .pag import KolorsPAGPipeline

    AUTO_TEXT2IMAGE_PIPELINES_MAPPING["kolors"] = KolorsPipeline
    AUTO_TEXT2IMAGE_PIPELINES_MAPPING["kolors-pag"] = KolorsPAGPipeline
    AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["kolors"] = KolorsImg2ImgPipeline

SUPPORTED_TASKS_MAPPINGS = [
    AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
    AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
    AUTO_INPAINT_PIPELINES_MAPPING,
    _AUTO_TEXT2IMAGE_DECODER_PIPELINES_MAPPING,
    _AUTO_IMAGE2IMAGE_DECODER_PIPELINES_MAPPING,
    _AUTO_INPAINT_DECODER_PIPELINES_MAPPING,
]


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
    """
    search_word: str = ""
    download_url: str = ""
    file_name: str = ""
    local: bool = False

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
    loading_method: str = None  
    checkpoint_format: str = None
    repo_status: RepoStatus = RepoStatus()
    model_status: ModelStatus = ModelStatus()


def _get_connected_pipeline(pipeline_cls):
    # for now connected pipelines can only be loaded from decoder pipelines, such as kandinsky-community/kandinsky-2-2-decoder
    if pipeline_cls in _AUTO_TEXT2IMAGE_DECODER_PIPELINES_MAPPING.values():
        return _get_task_class(
            AUTO_TEXT2IMAGE_PIPELINES_MAPPING, pipeline_cls.__name__, throw_error_if_not_exist=False
        )
    if pipeline_cls in _AUTO_IMAGE2IMAGE_DECODER_PIPELINES_MAPPING.values():
        return _get_task_class(
            AUTO_IMAGE2IMAGE_PIPELINES_MAPPING, pipeline_cls.__name__, throw_error_if_not_exist=False
        )
    if pipeline_cls in _AUTO_INPAINT_DECODER_PIPELINES_MAPPING.values():
        return _get_task_class(AUTO_INPAINT_PIPELINES_MAPPING, pipeline_cls.__name__, throw_error_if_not_exist=False)


def _get_task_class(mapping, pipeline_class_name, throw_error_if_not_exist: bool = True):
    def get_model(pipeline_class_name):
        for task_mapping in SUPPORTED_TASKS_MAPPINGS:
            for model_name, pipeline in task_mapping.items():
                if pipeline.__name__ == pipeline_class_name:
                    return model_name

    model_name = get_model(pipeline_class_name)

    if model_name is not None:
        task_class = mapping.get(model_name, None)
        if task_class is not None:
            return task_class

    if throw_error_if_not_exist:
        raise ValueError(f"AutoPipeline can't find a pipeline linked to {pipeline_class_name} for {model_name}")


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
        return pipeline_class.from_single_file(pretrained_model_or_path, checkpoint=checkpoint, **kwargs)


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


def search_huggingface(search_word: str, **kwargs):
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

    # Get the type and loading method for the keyword
    search_word_status = get_keyword_types(search_word)

    if search_word_status["type"]["hf_repo"]:
        if download:
            model_path = DiffusionPipeline.download(
                search_word,
                revision=revision,
                token=token,
                force_download=force_download,
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
            token=token
        )
        model_dicts = [asdict(value) for value in list(hf_models)]
        
        hf_repo_info = {}
        file_list = []
        repo_id, file_name = "", ""
        diffusers_model_exists = False

        # Loop through models to find a suitable candidate
        for repo_info in model_dicts:
            repo_id = repo_info["id"]
            file_list = []
            hf_repo_info = hf_api.model_info(
                repo_id=repo_id,
                securityStatus=True
            )
            # Lists files with security issues.
            hf_security_info = hf_repo_info.security_repo_status
            exclusion = [issue['path'] for issue in hf_security_info['filesWithIssues']]

            # Checks for multi-folder diffusers model or valid files (models with security issues are excluded).
            if hf_security_info["scansDone"]:
                for info in repo_info["siblings"]:
                    file_path = info["rfilename"]
                    if (
                        "model_index.json" == file_path
                        and checkpoint_format in ["diffusers", "all"]
                    ):
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
                    repo_id=repo_id,
                    token=token,
                )
            else:
                model_path = repo_id
                
        elif file_list:
            # Sort and find the safest model
            file_name = next(
                (
                    model
                    for model in sorted(file_list, reverse=True)
                    if re.search(r"(?i)[-_](safe|sfw)", model)
                ),
                file_list[0]
            )
            

            if download:
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_name,
                    revision=revision,
                    token=token,
                    force_download=force_download,
                )
            
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
            repo_status=RepoStatus(
                repo_id=repo_id,
                repo_hash=hf_repo_info.sha,
                version=revision
            ),
            model_status=ModelStatus(
                search_word=search_word,
                download_url=download_url,
                file_name=file_name,
                local=download,
            )
        )
    
    else:
        return model_path
    

def search_civitai(search_word: str, **kwargs):
    r"""
    Downloads a model from Civitai.

    Parameters:
        search_word (`str`):
            The search query string.
        model_type (`str`, *optional*, defaults to `Checkpoint`):
            The type of model to search for.
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
        skip_error (`bool`, *optional*, defaults to `False`):
            Whether to skip errors and return None.

    Returns:
        `Union[str,  SearchResult, None]`: The model path or ` SearchResult` or None.
    """

    # Extract additional parameters from kwargs
    model_type = kwargs.pop("model_type", "Checkpoint")
    download = kwargs.pop("download", False)
    base_model = kwargs.pop("base_model", None)
    force_download = kwargs.pop("force_download", False)
    token = kwargs.pop("token", None)
    include_params = kwargs.pop("include_params", False)
    skip_error = kwargs.pop("skip_error", False)

    # Initialize additional variables with default values
    model_path = ""
    repo_name = ""
    repo_id = ""
    version_id = ""
    models_list = []
    selected_repo = {}
    selected_model = {}
    selected_version = {}

    # Set up parameters and headers for the CivitAI API request
    params = {
        "query": search_word,
        "types": model_type,
        "sort": "Most Downloaded",
        "limit": 20,
    }
    if base_model is not None:
        params["baseModel"] = base_model

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        # Make the request to the CivitAI API
        response = requests.get(
            "https://civitai.com/api/v1/models", params=params, headers=headers
        )
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
        sorted_versions = sorted(selected_repo["modelVersions"], key=lambda x: x["stats"]["downloadCount"], reverse=True)
        for selected_version in sorted_versions:
            version_id = selected_version["id"]
            models_list = []
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
                    sorted_models[0]
                )

                break
        else:
            continue
        break

    if not selected_model:
        if skip_error:
            return None
        else:
            raise ValueError("No model found. Please try changing the word you are searching for.")

    file_name = selected_model["filename"]
    download_url = selected_model["download_url"]

    # Handle file download and setting model information
    if download:
        model_path = f"/root/.cache/Civitai/{repo_id}/{version_id}/{file_name}"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        if (not os.path.exists(model_path)) or force_download:
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"

            try:
                response = requests.get(download_url, stream=True, headers=headers)
                response.raise_for_status()
            except requests.HTTPError:
                raise requests.HTTPError(f"Invalid URL: {download_url}, {response.status_code}")

            with tqdm.wrapattr(
                open(model_path, "wb"),
                "write",
                miniters=1,
                desc=file_name,
                total=int(response.headers.get("content-length", 0)),
            ) as fetched_model_info:
                for chunk in response.iter_content(chunk_size=8192):
                    fetched_model_info.write(chunk)
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
            repo_status=RepoStatus(
                repo_id=repo_name,
                repo_hash=repo_id,
                version=version_id
            ),
            model_status=ModelStatus(
                search_word=search_word,
                download_url=download_url,
                file_name=file_name,
                local=output_info["type"]["local"]
            )
        )


class AutoPipelineForText2Image(ConfigMixin):
    r"""

    [`AutoPipelineForText2Image`] is a generic pipeline class that instantiates a text-to-image pipeline class. The
    specific underlying pipeline class is automatically selected from either the
    [`~AutoPipelineForText2Image.from_pretrained`] or [`~AutoPipelineForText2Image.from_pipe`] methods.

    This class cannot be instantiated using `__init__()` (throws an error).

    Class attributes:

        - **config_name** (`str`) -- The configuration filename that stores the class and module names of all the
          diffusion pipeline's components.

    """

    config_name = "model_index.json"

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_pipe(pipeline)` methods."
        )

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_or_path, **kwargs):
        r"""
        Instantiates a text-to-image Pytorch diffusion pipeline from pretrained pipeline weight.

        The from_pretrained() method takes care of returning the correct pipeline class instance by:
            1. Detect the pipeline class of the pretrained_model_or_path based on the _class_name property of its
               config object
            2. Find the text-to-image pipeline linked to the pipeline class using pattern matching on pipeline class
               name.

        If a `controlnet` argument is passed, it will instantiate a [`StableDiffusionControlNetPipeline`] object.

        The pipeline is set in evaluation mode (`model.eval()`) by default.

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```
        Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
        - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
        You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
        ```

        Parameters:
            pretrained_model_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                      saved using
                    [`~DiffusionPipeline.save_pretrained`].
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
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.
            variant (`str`, *optional*):
                Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.

        <Tip>

        To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        `huggingface-cli login`.

        </Tip>

        Examples:

        ```py
        >>> from diffusers import AutoPipelineForText2Image

        >>> pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> image = pipeline(prompt).images[0]
        ```
        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)

        load_config_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "token": token,
            "local_files_only": local_files_only,
            "revision": revision,
        }

        config = cls.load_config(pretrained_model_or_path, **load_config_kwargs)
        orig_class_name = config["_class_name"]

        if "controlnet" in kwargs:
            orig_class_name = config["_class_name"].replace("Pipeline", "ControlNetPipeline")
        if "enable_pag" in kwargs:
            enable_pag = kwargs.pop("enable_pag")
            if enable_pag:
                orig_class_name = orig_class_name.replace("Pipeline", "PAGPipeline")

        text_2_image_cls = _get_task_class(AUTO_TEXT2IMAGE_PIPELINES_MAPPING, orig_class_name)

        kwargs = {**load_config_kwargs, **kwargs}
        return text_2_image_cls.from_pretrained(pretrained_model_or_path, **kwargs)

    @classmethod
    def from_pipe(cls, pipeline, **kwargs):
        r"""
        Instantiates a text-to-image Pytorch diffusion pipeline from another instantiated diffusion pipeline class.

        The from_pipe() method takes care of returning the correct pipeline class instance by finding the text-to-image
        pipeline linked to the pipeline class using pattern matching on pipeline class name.

        All the modules the pipeline contains will be used to initialize the new pipeline without reallocating
        additional memory.

        The pipeline is set in evaluation mode (`model.eval()`) by default.

        Parameters:
            pipeline (`DiffusionPipeline`):
                an instantiated `DiffusionPipeline` object

        ```py
        >>> from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

        >>> pipe_i2i = AutoPipelineForImage2Image.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5", requires_safety_checker=False
        ... )

        >>> pipe_t2i = AutoPipelineForText2Image.from_pipe(pipe_i2i)
        >>> image = pipe_t2i(prompt).images[0]
        ```
        """

        original_config = dict(pipeline.config)
        original_cls_name = pipeline.__class__.__name__

        # derive the pipeline class to instantiate
        text_2_image_cls = _get_task_class(AUTO_TEXT2IMAGE_PIPELINES_MAPPING, original_cls_name)

        if "controlnet" in kwargs:
            if kwargs["controlnet"] is not None:
                to_replace = "PAGPipeline" if "PAG" in text_2_image_cls.__name__ else "Pipeline"
                text_2_image_cls = _get_task_class(
                    AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
                    text_2_image_cls.__name__.replace("ControlNet", "").replace(to_replace, "ControlNet" + to_replace),
                )
            else:
                text_2_image_cls = _get_task_class(
                    AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
                    text_2_image_cls.__name__.replace("ControlNet", ""),
                )

        if "enable_pag" in kwargs:
            enable_pag = kwargs.pop("enable_pag")
            if enable_pag:
                text_2_image_cls = _get_task_class(
                    AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
                    text_2_image_cls.__name__.replace("PAG", "").replace("Pipeline", "PAGPipeline"),
                )
            else:
                text_2_image_cls = _get_task_class(
                    AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
                    text_2_image_cls.__name__.replace("PAG", ""),
                )

        # define expected module and optional kwargs given the pipeline signature
        expected_modules, optional_kwargs = text_2_image_cls._get_signature_keys(text_2_image_cls)

        pretrained_model_name_or_path = original_config.pop("_name_or_path", None)

        # allow users pass modules in `kwargs` to override the original pipeline's components
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
        original_class_obj = {
            k: pipeline.components[k]
            for k, v in pipeline.components.items()
            if k in expected_modules and k not in passed_class_obj
        }

        # allow users pass optional kwargs to override the original pipelines config attribute
        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}
        original_pipe_kwargs = {
            k: original_config[k]
            for k, v in original_config.items()
            if k in optional_kwargs and k not in passed_pipe_kwargs
        }

        # config that were not expected by original pipeline is stored as private attribute
        # we will pass them as optional arguments if they can be accepted by the pipeline
        additional_pipe_kwargs = [
            k[1:]
            for k in original_config.keys()
            if k.startswith("_") and k[1:] in optional_kwargs and k[1:] not in passed_pipe_kwargs
        ]
        for k in additional_pipe_kwargs:
            original_pipe_kwargs[k] = original_config.pop(f"_{k}")

        text_2_image_kwargs = {**passed_class_obj, **original_class_obj, **passed_pipe_kwargs, **original_pipe_kwargs}

        # store unused config as private attribute
        unused_original_config = {
            f"{'' if k.startswith('_') else '_'}{k}": original_config[k]
            for k, v in original_config.items()
            if k not in text_2_image_kwargs
        }

        missing_modules = set(expected_modules) - set(pipeline._optional_components) - set(text_2_image_kwargs.keys())

        if len(missing_modules) > 0:
            raise ValueError(
                f"Pipeline {text_2_image_cls} expected {expected_modules}, but only {set(list(passed_class_obj.keys()) + list(original_class_obj.keys()))} were passed"
            )

        model = text_2_image_cls(**text_2_image_kwargs)
        model.register_to_config(_name_or_path=pretrained_model_name_or_path)
        model.register_to_config(**unused_original_config)

        return model
    
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

        <Tip>

        To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        `huggingface-cli login`.

        </Tip>

        Examples:

        ```py
        >>> from diffusers import AutoPipelineForText2Image

        >>> pipeline = AutoPipelineForText2Image.from_huggingface("stable-diffusion-v1-5")
        >>> image = pipeline(prompt).images[0]
        ```
        """
        # Update kwargs to ensure the model is downloaded and parameters are included
        kwargs.update({"download": True, "include_params": True, "skip_error": False})

        # Search for the model on Hugging Face and get the model status
        model_status = search_huggingface(pretrained_model_link_or_path, **kwargs)   
        logger.warning(f"checkpoint_path: {model_status.model_status.download_url}")
        checkpoint_path = model_status.model_path
        
        # Check the format of the model checkpoint
        if model_status.checkpoint_format == "single_file":
            # Load the pipeline from a single file checkpoint
            return load_pipeline_from_single_file(
                pretrained_model_or_path=checkpoint_path,
                pipeline_mapping=SINGLE_FILE_CHECKPOINT_TEXT2IMAGE_PIPELINE_MAPPING,
                **kwargs
            )
        else:
            return cls.from_pretrained(checkpoint_path, **kwargs)
    
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
            pipeline_tag (`str`, *optional*):
                Tag to filter models by pipeline.
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

        <Tip>

        To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        `huggingface-cli login`.

        </Tip>

        Examples:

        ```py
        >>> from diffusers import AutoPipelineForText2Image

        >>> pipeline = AutoPipelineForText2Image.from_huggingface("stable-diffusion-v1-5")
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
        model_status = search_civitai(pretrained_model_link_or_path, **kwargs)
        logger.warning(f"checkpoint_path: {model_status.model_status.download_url}")
        checkpoint_path = model_status.model_path

        # Load the pipeline from a single file checkpoint
        return load_pipeline_from_single_file(
            pretrained_model_or_path=checkpoint_path,
            pipeline_mapping=SINGLE_FILE_CHECKPOINT_TEXT2IMAGE_PIPELINE_MAPPING,
            **kwargs
        )
    


class AutoPipelineForImage2Image(ConfigMixin):
    r"""

    [`AutoPipelineForImage2Image`] is a generic pipeline class that instantiates an image-to-image pipeline class. The
    specific underlying pipeline class is automatically selected from either the
    [`~AutoPipelineForImage2Image.from_pretrained`] or [`~AutoPipelineForImage2Image.from_pipe`] methods.

    This class cannot be instantiated using `__init__()` (throws an error).

    Class attributes:

        - **config_name** (`str`) -- The configuration filename that stores the class and module names of all the
          diffusion pipeline's components.

    """

    config_name = "model_index.json"

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_pipe(pipeline)` methods."
        )

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_or_path, **kwargs):
        r"""
        Instantiates a image-to-image Pytorch diffusion pipeline from pretrained pipeline weight.

        The from_pretrained() method takes care of returning the correct pipeline class instance by:
            1. Detect the pipeline class of the pretrained_model_or_path based on the _class_name property of its
               config object
            2. Find the image-to-image pipeline linked to the pipeline class using pattern matching on pipeline class
               name.

        If a `controlnet` argument is passed, it will instantiate a [`StableDiffusionControlNetImg2ImgPipeline`]
        object.

        The pipeline is set in evaluation mode (`model.eval()`) by default.

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```
        Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
        - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
        You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
        ```

        Parameters:
            pretrained_model_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                      saved using
                    [`~DiffusionPipeline.save_pretrained`].
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
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.
            variant (`str`, *optional*):
                Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.

        <Tip>

        To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        `huggingface-cli login`.

        </Tip>

        Examples:

        ```py
        >>> from diffusers import AutoPipelineForImage2Image

        >>> pipeline = AutoPipelineForImage2Image.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> image = pipeline(prompt, image).images[0]
        ```
        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)

        load_config_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "token": token,
            "local_files_only": local_files_only,
            "revision": revision,
        }

        config = cls.load_config(pretrained_model_or_path, **load_config_kwargs)
        orig_class_name = config["_class_name"]

        # the `orig_class_name` can be:
        # `- *Pipeline` (for regular text-to-image checkpoint)
        # `- *Img2ImgPipeline` (for refiner checkpoint)
        to_replace = "Img2ImgPipeline" if "Img2Img" in config["_class_name"] else "Pipeline"

        if "controlnet" in kwargs:
            orig_class_name = orig_class_name.replace(to_replace, "ControlNet" + to_replace)
        if "enable_pag" in kwargs:
            enable_pag = kwargs.pop("enable_pag")
            if enable_pag:
                orig_class_name = orig_class_name.replace(to_replace, "PAG" + to_replace)

        image_2_image_cls = _get_task_class(AUTO_IMAGE2IMAGE_PIPELINES_MAPPING, orig_class_name)

        kwargs = {**load_config_kwargs, **kwargs}
        return image_2_image_cls.from_pretrained(pretrained_model_or_path, **kwargs)

    @classmethod
    def from_pipe(cls, pipeline, **kwargs):
        r"""
        Instantiates a image-to-image Pytorch diffusion pipeline from another instantiated diffusion pipeline class.

        The from_pipe() method takes care of returning the correct pipeline class instance by finding the
        image-to-image pipeline linked to the pipeline class using pattern matching on pipeline class name.

        All the modules the pipeline contains will be used to initialize the new pipeline without reallocating
        additional memory.

        The pipeline is set in evaluation mode (`model.eval()`) by default.

        Parameters:
            pipeline (`DiffusionPipeline`):
                an instantiated `DiffusionPipeline` object

        Examples:

        ```py
        >>> from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

        >>> pipe_t2i = AutoPipelineForText2Image.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5", requires_safety_checker=False
        ... )

        >>> pipe_i2i = AutoPipelineForImage2Image.from_pipe(pipe_t2i)
        >>> image = pipe_i2i(prompt, image).images[0]
        ```
        """

        original_config = dict(pipeline.config)
        original_cls_name = pipeline.__class__.__name__

        # derive the pipeline class to instantiate
        image_2_image_cls = _get_task_class(AUTO_IMAGE2IMAGE_PIPELINES_MAPPING, original_cls_name)

        if "controlnet" in kwargs:
            if kwargs["controlnet"] is not None:
                to_replace = "Img2ImgPipeline"
                if "PAG" in image_2_image_cls.__name__:
                    to_replace = "PAG" + to_replace
                image_2_image_cls = _get_task_class(
                    AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
                    image_2_image_cls.__name__.replace("ControlNet", "").replace(
                        to_replace, "ControlNet" + to_replace
                    ),
                )
            else:
                image_2_image_cls = _get_task_class(
                    AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
                    image_2_image_cls.__name__.replace("ControlNet", ""),
                )

        if "enable_pag" in kwargs:
            enable_pag = kwargs.pop("enable_pag")
            if enable_pag:
                image_2_image_cls = _get_task_class(
                    AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
                    image_2_image_cls.__name__.replace("PAG", "").replace("Img2ImgPipeline", "PAGImg2ImgPipeline"),
                )
            else:
                image_2_image_cls = _get_task_class(
                    AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
                    image_2_image_cls.__name__.replace("PAG", ""),
                )

        # define expected module and optional kwargs given the pipeline signature
        expected_modules, optional_kwargs = image_2_image_cls._get_signature_keys(image_2_image_cls)

        pretrained_model_name_or_path = original_config.pop("_name_or_path", None)

        # allow users pass modules in `kwargs` to override the original pipeline's components
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
        original_class_obj = {
            k: pipeline.components[k]
            for k, v in pipeline.components.items()
            if k in expected_modules and k not in passed_class_obj
        }

        # allow users pass optional kwargs to override the original pipelines config attribute
        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}
        original_pipe_kwargs = {
            k: original_config[k]
            for k, v in original_config.items()
            if k in optional_kwargs and k not in passed_pipe_kwargs
        }

        # config attribute that were not expected by original pipeline is stored as its private attribute
        # we will pass them as optional arguments if they can be accepted by the pipeline
        additional_pipe_kwargs = [
            k[1:]
            for k in original_config.keys()
            if k.startswith("_") and k[1:] in optional_kwargs and k[1:] not in passed_pipe_kwargs
        ]
        for k in additional_pipe_kwargs:
            original_pipe_kwargs[k] = original_config.pop(f"_{k}")

        image_2_image_kwargs = {**passed_class_obj, **original_class_obj, **passed_pipe_kwargs, **original_pipe_kwargs}

        # store unused config as private attribute
        unused_original_config = {
            f"{'' if k.startswith('_') else '_'}{k}": original_config[k]
            for k, v in original_config.items()
            if k not in image_2_image_kwargs
        }

        missing_modules = set(expected_modules) - set(pipeline._optional_components) - set(image_2_image_kwargs.keys())

        if len(missing_modules) > 0:
            raise ValueError(
                f"Pipeline {image_2_image_cls} expected {expected_modules}, but only {set(list(passed_class_obj.keys()) + list(original_class_obj.keys()))} were passed"
            )

        model = image_2_image_cls(**image_2_image_kwargs)
        model.register_to_config(_name_or_path=pretrained_model_name_or_path)
        model.register_to_config(**unused_original_config)

        return model
    
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

        <Tip>

        To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        `huggingface-cli login`.

        </Tip>

        Examples:

        ```py
        >>> from diffusers import AutoPipelineForText2Image

        >>> pipeline = AutoPipelineForText2Image.from_huggingface("stable-diffusion-v1-5")
        >>> image = pipeline(prompt).images[0]
        ```
        """
        # Update kwargs to ensure the model is downloaded and parameters are included
        kwargs.update({"download": True, "include_params": True, "skip_error": False})

        # Search for the model on Hugging Face and get the model status
        model_status = search_huggingface(pretrained_model_link_or_path, **kwargs)   
        logger.warning(f"checkpoint_path: {model_status.model_status.download_url}")
        checkpoint_path = model_status.model_path
        
        # Check the format of the model checkpoint
        if model_status.checkpoint_format == "single_file":
            # Load the pipeline from a single file checkpoint
            return load_pipeline_from_single_file(
                pretrained_model_or_path=checkpoint_path,
                pipeline_mapping=SINGLE_FILE_CHECKPOINT_IMAGE2IMAGE_PIPELINE_MAPPING,
                **kwargs
            )
        else:
            return cls.from_pretrained(checkpoint_path, **kwargs)
    
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
            pipeline_tag (`str`, *optional*):
                Tag to filter models by pipeline.
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

        <Tip>

        To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        `huggingface-cli login`.

        </Tip>

        Examples:

        ```py
        >>> from diffusers import AutoPipelineForText2Image

        >>> pipeline = AutoPipelineForText2Image.from_huggingface("stable-diffusion-v1-5")
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
        model_status = search_civitai(pretrained_model_link_or_path, **kwargs)
        logger.warning(f"checkpoint_path: {model_status.model_status.download_url}")
        checkpoint_path = model_status.model_path

        # Load the pipeline from a single file checkpoint
        return load_pipeline_from_single_file(
            pretrained_model_or_path=checkpoint_path,
            pipeline_mapping=SINGLE_FILE_CHECKPOINT_IMAGE2IMAGE_PIPELINE_MAPPING,
            **kwargs
        )

    

class AutoPipelineForInpainting(ConfigMixin):
    r"""

    [`AutoPipelineForInpainting`] is a generic pipeline class that instantiates an inpainting pipeline class. The
    specific underlying pipeline class is automatically selected from either the
    [`~AutoPipelineForInpainting.from_pretrained`] or [`~AutoPipelineForInpainting.from_pipe`] methods.

    This class cannot be instantiated using `__init__()` (throws an error).

    Class attributes:

        - **config_name** (`str`) -- The configuration filename that stores the class and module names of all the
          diffusion pipeline's components.

    """

    config_name = "model_index.json"

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_pipe(pipeline)` methods."
        )

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_or_path, **kwargs):
        r"""
        Instantiates a inpainting Pytorch diffusion pipeline from pretrained pipeline weight.

        The from_pretrained() method takes care of returning the correct pipeline class instance by:
            1. Detect the pipeline class of the pretrained_model_or_path based on the _class_name property of its
               config object
            2. Find the inpainting pipeline linked to the pipeline class using pattern matching on pipeline class name.

        If a `controlnet` argument is passed, it will instantiate a [`StableDiffusionControlNetInpaintPipeline`]
        object.

        The pipeline is set in evaluation mode (`model.eval()`) by default.

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```
        Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
        - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
        You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
        ```

        Parameters:
            pretrained_model_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                      saved using
                    [`~DiffusionPipeline.save_pretrained`].
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
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.
            variant (`str`, *optional*):
                Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.

        <Tip>

        To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        `huggingface-cli login`.

        </Tip>

        Examples:

        ```py
        >>> from diffusers import AutoPipelineForInpainting

        >>> pipeline = AutoPipelineForInpainting.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> image = pipeline(prompt, image=init_image, mask_image=mask_image).images[0]
        ```
        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)

        load_config_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "token": token,
            "local_files_only": local_files_only,
            "revision": revision,
        }

        config = cls.load_config(pretrained_model_or_path, **load_config_kwargs)
        orig_class_name = config["_class_name"]

        # The `orig_class_name`` can be:
        # `- *InpaintPipeline` (for inpaint-specific checkpoint)
        #  - or *Pipeline (for regular text-to-image checkpoint)
        to_replace = "InpaintPipeline" if "Inpaint" in config["_class_name"] else "Pipeline"

        if "controlnet" in kwargs:
            orig_class_name = orig_class_name.replace(to_replace, "ControlNet" + to_replace)
        if "enable_pag" in kwargs:
            enable_pag = kwargs.pop("enable_pag")
            if enable_pag:
                orig_class_name = orig_class_name.replace(to_replace, "PAG" + to_replace)
        inpainting_cls = _get_task_class(AUTO_INPAINT_PIPELINES_MAPPING, orig_class_name)

        kwargs = {**load_config_kwargs, **kwargs}
        return inpainting_cls.from_pretrained(pretrained_model_or_path, **kwargs)

    @classmethod
    def from_pipe(cls, pipeline, **kwargs):
        r"""
        Instantiates a inpainting Pytorch diffusion pipeline from another instantiated diffusion pipeline class.

        The from_pipe() method takes care of returning the correct pipeline class instance by finding the inpainting
        pipeline linked to the pipeline class using pattern matching on pipeline class name.

        All the modules the pipeline class contain will be used to initialize the new pipeline without reallocating
        additional memory.

        The pipeline is set in evaluation mode (`model.eval()`) by default.

        Parameters:
            pipeline (`DiffusionPipeline`):
                an instantiated `DiffusionPipeline` object

        Examples:

        ```py
        >>> from diffusers import AutoPipelineForText2Image, AutoPipelineForInpainting

        >>> pipe_t2i = AutoPipelineForText2Image.from_pretrained(
        ...     "DeepFloyd/IF-I-XL-v1.0", requires_safety_checker=False
        ... )

        >>> pipe_inpaint = AutoPipelineForInpainting.from_pipe(pipe_t2i)
        >>> image = pipe_inpaint(prompt, image=init_image, mask_image=mask_image).images[0]
        ```
        """
        original_config = dict(pipeline.config)
        original_cls_name = pipeline.__class__.__name__

        # derive the pipeline class to instantiate
        inpainting_cls = _get_task_class(AUTO_INPAINT_PIPELINES_MAPPING, original_cls_name)

        if "controlnet" in kwargs:
            if kwargs["controlnet"] is not None:
                inpainting_cls = _get_task_class(
                    AUTO_INPAINT_PIPELINES_MAPPING,
                    inpainting_cls.__name__.replace("ControlNet", "").replace(
                        "InpaintPipeline", "ControlNetInpaintPipeline"
                    ),
                )
            else:
                inpainting_cls = _get_task_class(
                    AUTO_INPAINT_PIPELINES_MAPPING,
                    inpainting_cls.__name__.replace("ControlNetInpaintPipeline", "InpaintPipeline"),
                )

        if "enable_pag" in kwargs:
            enable_pag = kwargs.pop("enable_pag")
            if enable_pag:
                inpainting_cls = _get_task_class(
                    AUTO_INPAINT_PIPELINES_MAPPING,
                    inpainting_cls.__name__.replace("PAG", "").replace("InpaintPipeline", "PAGInpaintPipeline"),
                )
            else:
                inpainting_cls = _get_task_class(
                    AUTO_INPAINT_PIPELINES_MAPPING,
                    inpainting_cls.__name__.replace("PAGInpaintPipeline", "InpaintPipeline"),
                )

        # define expected module and optional kwargs given the pipeline signature
        expected_modules, optional_kwargs = inpainting_cls._get_signature_keys(inpainting_cls)

        pretrained_model_name_or_path = original_config.pop("_name_or_path", None)

        # allow users pass modules in `kwargs` to override the original pipeline's components
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
        original_class_obj = {
            k: pipeline.components[k]
            for k, v in pipeline.components.items()
            if k in expected_modules and k not in passed_class_obj
        }

        # allow users pass optional kwargs to override the original pipelines config attribute
        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}
        original_pipe_kwargs = {
            k: original_config[k]
            for k, v in original_config.items()
            if k in optional_kwargs and k not in passed_pipe_kwargs
        }

        # config that were not expected by original pipeline is stored as private attribute
        # we will pass them as optional arguments if they can be accepted by the pipeline
        additional_pipe_kwargs = [
            k[1:]
            for k in original_config.keys()
            if k.startswith("_") and k[1:] in optional_kwargs and k[1:] not in passed_pipe_kwargs
        ]
        for k in additional_pipe_kwargs:
            original_pipe_kwargs[k] = original_config.pop(f"_{k}")

        inpainting_kwargs = {**passed_class_obj, **original_class_obj, **passed_pipe_kwargs, **original_pipe_kwargs}

        # store unused config as private attribute
        unused_original_config = {
            f"{'' if k.startswith('_') else '_'}{k}": original_config[k]
            for k, v in original_config.items()
            if k not in inpainting_kwargs
        }

        missing_modules = set(expected_modules) - set(pipeline._optional_components) - set(inpainting_kwargs.keys())

        if len(missing_modules) > 0:
            raise ValueError(
                f"Pipeline {inpainting_cls} expected {expected_modules}, but only {set(list(passed_class_obj.keys()) + list(original_class_obj.keys()))} were passed"
            )

        model = inpainting_cls(**inpainting_kwargs)
        model.register_to_config(_name_or_path=pretrained_model_name_or_path)
        model.register_to_config(**unused_original_config)

        return model
    
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

        <Tip>

        To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        `huggingface-cli login`.

        </Tip>

        Examples:

        ```py
        >>> from diffusers import AutoPipelineForText2Image

        >>> pipeline = AutoPipelineForText2Image.from_huggingface("stable-diffusion-v1-5")
        >>> image = pipeline(prompt).images[0]
        ```
        """
        # Update kwargs to ensure the model is downloaded and parameters are included
        kwargs.update({"download": True, "include_params": True, "skip_error": False})

        # Search for the model on Hugging Face and get the model status
        model_status = search_huggingface(pretrained_model_link_or_path, **kwargs)   
        logger.warning(f"checkpoint_path: {model_status.model_status.download_url}")
        checkpoint_path = model_status.model_path
        
        # Check the format of the model checkpoint
        if model_status.checkpoint_format == "single_file":
            # Load the pipeline from a single file checkpoint
            return load_pipeline_from_single_file(
                pretrained_model_or_path=checkpoint_path,
                pipeline_mapping=SINGLE_FILE_CHECKPOINT_INPAINT_PIPELINE_MAPPING,
                **kwargs
            )
        else:
            return cls.from_pretrained(checkpoint_path, **kwargs)
    
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
            pipeline_tag (`str`, *optional*):
                Tag to filter models by pipeline.
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

        <Tip>

        To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        `huggingface-cli login`.

        </Tip>

        Examples:

        ```py
        >>> from diffusers import AutoPipelineForText2Image

        >>> pipeline = AutoPipelineForText2Image.from_huggingface("stable-diffusion-v1-5")
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
        model_status = search_civitai(pretrained_model_link_or_path, **kwargs)
        logger.warning(f"checkpoint_path: {model_status.model_status.download_url}")
        checkpoint_path = model_status.model_path

        # Load the pipeline from a single file checkpoint
        return load_pipeline_from_single_file(
            pretrained_model_or_path=checkpoint_path,
            pipeline_mapping=SINGLE_FILE_CHECKPOINT_INPAINT_PIPELINE_MAPPING,
            **kwargs
        )