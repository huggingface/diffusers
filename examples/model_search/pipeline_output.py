from dataclasses import dataclass

@dataclass
class RepoStatus:
    """
    Data class for storing repository status information.

    Attributes:
        repo_id (str): The name of the repository.
        repo_hash (str): The hash of the repository.
        version (str): The version ID of the repository.
    """
    repo_id: str = ""
    repo_hash: str = ""
    version: str = ""


@dataclass
class ModelStatus:
    """
    Data class for storing model status information.

    Attributes:
        search_word (str): The search word used to find the model.
        download_url (str): The URL to download the model.
        file_name (str): The name of the model file.
        file_id (str): The ID of the model file.
        fp (str): Floating-point precision formats.
        local (bool): Whether the model is stored locally.
    """
    search_word: str = ""
    download_url: str = ""
    file_name: str = ""
    local: bool = False


@dataclass
class SearchPipelineOutput:
    """
    Data class for storing model data.

    Attributes:
        model_path (str): The path to the model.
        load_type (str): The type of loading method used for the model.
        repo_status (RepoStatus): The status of the repository.
        model_status (ModelStatus): The status of the model.
    """
    model_path: str = ""
    loading_method: str = ""  # "" or "from_single_file" or "from_pretrained"
    checkpoint_format: str = None # "single_file" or "diffusers"
    repo_status: RepoStatus = RepoStatus()
    model_status: ModelStatus = ModelStatus()