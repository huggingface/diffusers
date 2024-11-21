import os
import re
import requests
from typing import Union
from tqdm.auto import tqdm
from dataclasses import asdict
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

from .pipeline_output import (
    SearchPipelineOutput,
    ModelStatus,
    RepoStatus,
)



CUSTOM_SEARCH_KEY = {
    "sd" : "stabilityai/stable-diffusion-2-1",
    }


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



def get_keyword_types(keyword):
    """
    Determine the type and loading method for a given keyword.
    
    Args:
        keyword (str): The input keyword to classify.
        
    Returns:
        dict: A dictionary containing the model format, loading method,
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
    model_info = {
        "model_path": "",
        "load_type": "",
        "repo_status": {
            "repo_name": "",
            "repo_id": "",
            "revision": ""
        },
        "model_status": {
            "search_word": "",
            "download_url": "",
            "filename": "",
            "local": False,
            "single_file": False
        },
    }

    def __init__(self):
        pass
    

    @staticmethod
    def create_huggingface_url(repo_id, file_name):
        """
        Create a Hugging Face URL for a given repository ID and file name.
        
        Args:
            repo_id (str): The repository ID.
            file_name (str): The file name within the repository.
        
        Returns:
            str: The complete URL to the file or repository on Hugging Face.
        """
        if file_name:
            return f"https://huggingface.co/{repo_id}/blob/main/{file_name}"
        else:
            return f"https://huggingface.co/{repo_id}"
    
    @staticmethod
    def hf_find_safest_model(models) -> str:
        """
        Sort and find the safest model.

        Args:
            models (list): A list of model names to sort and check.

        Returns:
            The name of the safest model or the first model in the list if no safe model is found.
        """
        for model in sorted(models, reverse=True):
            if bool(re.search(r"(?i)[-_](safe|sfw)", model)):
                return model
        return models[0]


    @classmethod
    def for_HF(cls, search_word, **kwargs):
        """
        Class method to search and download models from Hugging Face.
        
        Args:
            search_word (str): The search keyword for finding models.
            **kwargs: Additional keyword arguments.
        
        Returns:
            str: The path to the downloaded model or search word.
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
    The Civitai class is used to search and download models from Civitai.

    Attributes:
        base_civitai_dir (str): Base directory for Civitai.
        max_number_of_choices (int): Maximum number of choices.
        chunk_size (int): Chunk size.

    Methods:
        for_civitai(search_word, auto, model_type, download, civitai_token, skip_error, include_hugface):
            Downloads a model from Civitai.
        civitai_security_check(value): Performs a security check.
        requests_civitai(query, auto, model_type, civitai_token, include_hugface): Retrieves models from Civitai.
        repo_select_civitai(state, auto, recursive, include_hugface): Selects a repository from Civitai.
        download_model(url, save_path, civitai_token): Downloads a model.
        version_select_civitai(state, auto, recursive): Selects a model version from Civitai.
        file_select_civitai(state_list, auto, recursive): Selects a file to download.
        civitai_save_path(): Sets the save path.
    """

    base_civitai_dir = "/root/.cache/Civitai"
    max_number_of_choices: int = 15
    chunk_size: int = 8192

    def __init__(self):
        pass

    @staticmethod
    def civitai_find_safest_model(models) -> str:
        """
        Sort and find the safest model.

        Args:
            models (list): A list of model names to check.

        Returns:
            The name of the safest model or the first model in the list if no safe model is found.
        """

        for model_data in models:
            if bool(re.search(r"(?i)[-_](safe|sfw)", model_data["filename"])):
                return model_data
        return models[0]
    

    @classmethod
    def for_civitai(
        cls,
        search_word,
        **kwargs
    ) -> Union[str,SearchPipelineOutput,None]:
        """
        Downloads a model from Civitai.

        Parameters:
        - search_word (str): Search query string.
        - auto (bool): Auto-select flag.
        - model_type (str): Type of model to search for.
        - download (bool): Whether to download the model.
        - include_params (bool): Whether to include parameters in the returned data.

        Returns:
        - SearchPipelineOutput
        """
        model_type = kwargs.pop("model_type", "Checkpoint")
        download = kwargs.pop("download", False)
        force_download = kwargs.pop("force_download", False)
        civitai_token = kwargs.pop("civitai_token", None)
        include_params = kwargs.pop("include_params", False)
        skip_error = kwargs.pop("skip_error", False)

        model_info = {
            "model_path" : "",
            "load_type" : "",
            "repo_status":{
                "repo_name":"",
                "repo_id":"",
                "revision":""
                },
            "model_status":{
                "search_word" : "",
                "download_url": "",
                "filename":"",
                "local" : False,
                "single_file" : False
                },
            }
        
        params = {
            "query": search_word,
            "types": model_type,
            "sort": "Highest Rated",
            "limit":20
            }

        headers = {}
        if civitai_token:
            headers["Authorization"] = f"Bearer {civitai_token}"

        try:
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
        # Put the repo sorting process on this line.
        sorted_repos = sorted(data["items"], key=lambda x: x["stats"]["downloadCount"], reverse=True)

        model_path = ""
        repo_name = ""
        repo_id = ""
        version_id = ""
        models_list = []
        selected_repo = {}
        selected_model = {}
        selected_version = {}

        for selected_repo in sorted_repos:
            repo_name = selected_repo["name"]
            repo_id = selected_repo["id"]
            
            sorted_versions = sorted(selected_repo["modelVersions"], key=lambda x: x["stats"]["downloadCount"], reverse=True)
            for selected_version in sorted_versions:
                version_id = selected_version["id"]
                models_list = []
                for model_data in selected_version["files"]:
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
                raise ValueError("No models found")

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

        # Return appropriate result based on include_params
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
        """
        Search and retrieve model information from various sources (e.g., Hugging Face or CivitAI).

        This method allows flexible searching of models across different hubs. It accepts several parameters 
        to customize the search behavior, such as filtering by model type, format, or priority hub. Additionally, 
        it supports authentication tokens for private or restricted access.

        Args:
            search_word (str): The search term or keyword used to locate the desired model.
            download (bool, optional): Whether to download the model locally after finding it. Defaults to False.
            model_type (str, optional): Type of the model to search for (e.g., "Checkpoint", "LORA"). Defaults to "Checkpoint".
            checkpoint_format (str, optional): Specifies the format of the model (e.g., "single_file", "diffusers"). Defaults to "single_file".
            branch (str, optional): The branch of the repository to search in. Defaults to "main".
            include_params (bool, optional): Whether to include additional parameters about the model in the output. Defaults to False.
            hf_token (str, optional): Hugging Face API token for authentication. Required for private or restricted models.
            civitai_token (str, optional): CivitAI API token for authentication. Required for private or restricted models.

        Returns:
            Union[None, str, SearchPipelineOutput]:
                - `None`: If no model is found or accessible.
                - `str`: A string path to the retrieved model if `include_params=False`.
                - `SearchPipelineOutput`: Detailed model information if `include_params=True`.
        """
        return (
            cls.for_HF(search_word=search_word, skip_error=True, **kwargs) 
            or cls.for_HF(search_word=search_word, skip_error=True, **kwargs)
        )