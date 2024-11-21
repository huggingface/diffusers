import os
import re
import requests
from typing import (
    Union,
    List
)
from tqdm.auto import tqdm
from dataclasses import (
    asdict,
    dataclass
)
from huggingface_hub import (
    hf_api,
    hf_hub_download,
)

from diffusers.utils import logging
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.loaders.single_file_utils import (
    VALID_URL_PREFIXES,
    _extract_repo_id_and_weights_name,
)


CONFIG_FILE_LIST = [
    "preprocessor_config.json",
    "config.json",
    "model.safetensors",
    "model.fp16.safetensors",
    "model.ckpt",
    "pytorch_model.bin",
    "pytorch_model.fp16.bin",
    "scheduler_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
    "diffusion_pytorch_model.bin",
    "diffusion_pytorch_model.fp16.bin",
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.fp16.safetensors",
    "diffusion_pytorch_model.ckpt",
    "diffusion_pytorch_model.fp16.ckpt",
    "diffusion_pytorch_model.non_ema.bin",
    "diffusion_pytorch_model.non_ema.safetensors",
    "safety_checker/pytorch_model.bin",
    "safety_checker/model.safetensors",
    "safety_checker/model.ckpt",
    "safety_checker/model.fp16.safetensors",
    "safety_checker/model.fp16.ckpt",
    "unet/diffusion_pytorch_model.bin",
    "unet/diffusion_pytorch_model.safetensors",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "unet/diffusion_pytorch_model.ckpt",
    "unet/diffusion_pytorch_model.fp16.ckpt",
    "vae/diffusion_pytorch_model.bin",
    "vae/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.fp16.safetensors",
    "vae/diffusion_pytorch_model.ckpt",
    "vae/diffusion_pytorch_model.fp16.ckpt",
    "text_encoder/pytorch_model.bin",
    "text_encoder/model.safetensors",
    "text_encoder/model.fp16.safetensors",
    "text_encoder/model.ckpt",
    "text_encoder/model.fp16.ckpt",
    "text_encoder_2/model.safetensors",
    "text_encoder_2/model.ckpt"
]

EXTENSION =  [".safetensors", ".ckpt",".bin"]


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name



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
class SearchPipelineOutput:
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
            "search_word": False,
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
    
    # If none of the above, treat it as a search word
    else:
        status["type"]["search_word"] = True
        status["checkpoint_format"] = None
        status["loading_method"] = None
    
    return status


class HFSearchPipeline:
    """
    Search for models from Huggingface.
    """

    def __init__(self):
        pass
    
    @staticmethod
    def create_huggingface_url(repo_id, file_name):
        r"""
        Create a Hugging Face URL for a given repository ID and file name.

        Parameters:
            repo_id (`str`):
                The repository ID.
            file_name (`str`):
                The file name within the repository.

        Returns:
            `str`: The complete URL to the file or repository on Hugging Face.
        """
        if file_name:
            return f"https://huggingface.co/{repo_id}/blob/main/{file_name}"
        else:
            return f"https://huggingface.co/{repo_id}"
    
    @staticmethod
    def hf_find_safest_model(models) -> str:
        r"""
        Sort and find the safest model.

        Parameters:
            models (`list`):
                A list of model names to sort and check.

        Returns:
            `str`: The name of the safest model or the first model in the list if no safe model is found.
        """
        for model in sorted(models, reverse=True):
            if bool(re.search(r"(?i)[-_](safe|sfw)", model)):
                return model
        return models[0]
    
    @classmethod
    def for_HF(cls, search_word: str, **kwargs) -> Union[str, SearchPipelineOutput, None]:
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
            hf_token (`str`, *optional*):
                API token for Hugging Face authentication.
            skip_error (`bool`, *optional*, defaults to `False`):
                Whether to skip errors and return None.

        Returns:
            `Union[str, SearchPipelineOutput, None]`: The model path or SearchPipelineOutput or None.
        """
        # Extract additional parameters from kwargs
        revision = kwargs.pop("revision", None)
        checkpoint_format = kwargs.pop("checkpoint_format", "single_file")
        download = kwargs.pop("download", False)
        force_download = kwargs.pop("force_download", False)
        include_params = kwargs.pop("include_params", False)
        pipeline_tag = kwargs.pop("pipeline_tag", None)
        hf_token = kwargs.pop("hf_token", None)
        skip_error = kwargs.pop("skip_error", False)

        # Get the type and loading method for the keyword
        search_word_status = get_keyword_types(search_word)

        # Handle different types of keywords
        if search_word_status["type"]["hf_repo"]:
            if download:
                model_path = DiffusionPipeline.download(
                    search_word,
                    revision=revision,
                    token=hf_token
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
                    token=hf_token
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
                sort="downloads",
                direction=-1,
                limit=100,
                fetch_config=True,
                pipeline_tag=pipeline_tag,
                full=True,
                token=hf_token
            )
            model_dicts = [asdict(value) for value in list(hf_models)]
            
            hf_repo_info = {}
            file_list = []
            repo_id, file_name = "", ""
            
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
                diffusers_model_exists = False
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
                            and (file_path not in CONFIG_FILE_LIST)
                            and (file_path not in exclusion)
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
            
            download_url = cls.create_huggingface_url(
                repo_id=repo_id, file_name=file_name
            )
            if diffusers_model_exists:
                if download:
                    model_path = DiffusionPipeline.download(
                        repo_id=repo_id,
                        token=hf_token,
                    )
                else:
                    model_path = repo_id
            elif file_list:
                file_name = cls.hf_find_safest_model(file_list)
                if download:
                    model_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=file_name,
                        revision=revision,
                        token=hf_token
                    )
                else:
                    model_path = cls.create_huggingface_url(
                        repo_id=repo_id, file_name=file_name
                    )
        
        output_info = get_keyword_types(model_path)

        if include_params:
            return SearchPipelineOutput(
                model_path=model_path,
                loading_method=output_info["loading_method"],
                checkpoint_format=output_info["checkpoint_format"],
                repo_status=RepoStatus(
                    repo_id=repo_id,
                    repo_hash=hf_repo_info["sha"],
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



class CivitaiSearchPipeline:
    """
    Find checkpoints and more from Civitai.
    """

    def __init__(self):
        pass

    @staticmethod
    def civitai_find_safest_model(models: List[dict]) -> dict:
        r"""
        Sort and find the safest model.
        
        Parameters:
            models (`list`):
                A list of model dictionaries to check. Each dictionary should contain a 'filename' key.
        
        Returns:
            `dict`: The dictionary of the safest model or the first model in the list if no safe model is found.
        """
        
        for model_data in models:
            if bool(re.search(r"(?i)[-_](safe|sfw)", model_data["filename"])):
                return model_data
        return models[0]

    @classmethod
    def for_civitai(
        cls,
        search_word: str,
        **kwargs
    ) -> Union[str, SearchPipelineOutput, None]:
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
            civitai_token (`str`, *optional*):
                API token for Civitai authentication.
            include_params (`bool`, *optional*, defaults to `False`):
                Whether to include parameters in the returned data.
            skip_error (`bool`, *optional*, defaults to `False`):
                Whether to skip errors and return None.

        Returns:
            `Union[str, SearchPipelineOutput, None]`: The model path or `SearchPipelineOutput` or None.
        """

        # Extract additional parameters from kwargs
        model_type = kwargs.pop("model_type", "Checkpoint")
        download = kwargs.pop("download", False)
        base_model = kwargs.pop("base_model", None)
        force_download = kwargs.pop("force_download", False)
        civitai_token = kwargs.pop("civitai_token", None)
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
            "sort": "Highest Rated",
            "limit": 20
        }
        if base_model is not None:
            params["baseModel"] = base_model

        headers = {}
        if civitai_token:
            headers["Authorization"] = f"Bearer {civitai_token}"

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
                    if (
                        model_data["pickleScanResult"] == "Success"
                        and model_data["virusScanResult"] == "Success"
                        and any(model_data["name"].endswith(ext) for ext in EXTENSION)
                    ):
                        file_status = {
                            "filename": model_data["name"],
                            "download_url": model_data["downloadUrl"],
                        }
                        models_list.append(file_status)

                if models_list:
                    # Sort the models list by filename and find the safest model
                    sorted_models = sorted(models_list, key=lambda x: x["filename"], reverse=True)
                    selected_model = cls.civitai_find_safest_model(sorted_models)
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
                if civitai_token:
                    headers["Authorization"] = f"Bearer {civitai_token}"

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
            return SearchPipelineOutput(
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



class ModelSearchPipeline(
    HFSearchPipeline,
    CivitaiSearchPipeline
    ):

    def __init__(self):
        pass
    
    @classmethod
    def for_hubs(
        cls,
        search_word: str,
        **kwargs
    ) -> Union[None, str, SearchPipelineOutput]:
        r"""
        Search and download models from multiple hubs.

        Parameters:
            search_word (`str`):
                The search query string.
            model_type (`str`, *optional*, defaults to `Checkpoint`, Civitai only):
                The type of model to search for.
            revision (`str`, *optional*, Hugging Face only):
                The specific version of the model to download.
            include_params (`bool`, *optional*, defaults to `False`, both):
                Whether to include parameters in the returned data.
            checkpoint_format (`str`, *optional*, defaults to `"single_file"`, Hugging Face only):
                The format of the model checkpoint.
            download (`bool`, *optional*, defaults to `False`, both):
                Whether to download the model.
            pipeline_tag (`str`, *optional*, Hugging Face only):
                Tag to filter models by pipeline.
            base_model (`str`, *optional*, Civitai only):
                The base model to filter by.
            force_download (`bool`, *optional*, defaults to `False`, both):
                Whether to force the download if the model already exists.  
            hf_token (`str`, *optional*, Hugging Face only):
                API token for Hugging Face authentication.
            civitai_token (`str`, *optional*, Civitai only):
                API token for Civitai authentication.
            skip_error (`bool`, *optional*, defaults to `False`, both):
                Whether to skip errors and return None.

        Returns:
            `Union[None, str, SearchPipelineOutput]`: The model path, SearchPipelineOutput, or None if not found.
        """

        return (
            cls.for_HF(search_word=search_word, skip_error=True, **kwargs) 
            or cls.for_HF(search_word=search_word, skip_error=True, **kwargs)
        )