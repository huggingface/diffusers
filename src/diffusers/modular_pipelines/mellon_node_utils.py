import json
import logging
import os

# Simple typed wrapper for parameter overrides
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Union

from huggingface_hub import create_repo, hf_hub_download, upload_folder
from huggingface_hub.utils import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from ..utils import HUGGINGFACE_CO_RESOLVE_ENDPOINT


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MellonParam:
    """
    Parameter definition for Mellon nodes.

    Use factory methods for common params (e.g., MellonParam.seed()) or create custom ones with MellonParam(name="...",
    label="...", type="...").
    """

    name: str
    label: str
    type: str
    display: Optional[str] = None
    default: Any = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Any = None
    value: Any = None
    fieldOptions: Optional[Dict[str, Any]] = None
    onChange: Any = None
    onSignal: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for Mellon schema, excluding None values and name."""
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None and k != "name"}

    @classmethod
    def image(cls) -> "MellonParam":
        return cls(name="image", label="Image", type="image", display="input")

    @classmethod
    def images(cls) -> "MellonParam":
        return cls(name="images", label="Images", type="image", display="output")

    @classmethod
    def control_image(cls, display: str = "input") -> "MellonParam":
        return cls(name="control_image", label="Control Image", type="image", display=display)

    @classmethod
    def latents(cls, display: str = "input") -> "MellonParam":
        return cls(name="latents", label="Latents", type="latents", display=display)

    @classmethod
    def image_latents(cls, display: str = "input") -> "MellonParam":
        return cls(name="image_latents", label="Image Latents", type="latents", display=display)

    @classmethod
    def image_latents_with_strength(cls) -> "MellonParam":
        return cls(
            name="image_latents",
            label="Image Latents",
            type="latents",
            display="input",
            onChange={"false": ["height", "width"], "true": ["strength"]},
        )

    @classmethod
    def latents_preview(cls) -> "MellonParam":
        """
        `Latents Preview` is a special output parameter that is used to preview the latents in the UI.
        """
        return cls(name="latents_preview", label="Latents Preview", type="latent", display="output")

    @classmethod
    def embeddings(cls, display: str = "output") -> "MellonParam":
        return cls(name="embeddings", label="Text Embeddings", type="embeddings", display=display)

    @classmethod
    def controlnet_conditioning_scale(cls, default: float = 0.5) -> "MellonParam":
        return cls(
            name="controlnet_conditioning_scale",
            label="Controlnet Conditioning Scale",
            type="float",
            default=default,
            min=0.0,
            max=1.0,
            step=0.01,
        )

    @classmethod
    def control_guidance_start(cls, default: float = 0.0) -> "MellonParam":
        return cls(
            name="control_guidance_start",
            label="Control Guidance Start",
            type="float",
            default=default,
            min=0.0,
            max=1.0,
            step=0.01,
        )

    @classmethod
    def control_guidance_end(cls, default: float = 1.0) -> "MellonParam":
        return cls(
            name="control_guidance_end",
            label="Control Guidance End",
            type="float",
            default=default,
            min=0.0,
            max=1.0,
            step=0.01,
        )

    @classmethod
    def prompt(cls, default: str = "") -> "MellonParam":
        return cls(name="prompt", label="Prompt", type="string", default=default, display="textarea")

    @classmethod
    def negative_prompt(cls, default: str = "") -> "MellonParam":
        return cls(name="negative_prompt", label="Negative Prompt", type="string", default=default, display="textarea")

    @classmethod
    def strength(cls, default: float = 0.5) -> "MellonParam":
        return cls(name="strength", label="Strength", type="float", default=default, min=0.0, max=1.0, step=0.01)

    @classmethod
    def guidance_scale(cls, default: float = 5.0) -> "MellonParam":
        return cls(
            name="guidance_scale",
            label="Guidance Scale",
            type="float",
            display="slider",
            default=default,
            min=1.0,
            max=30.0,
            step=0.1,
        )

    @classmethod
    def height(cls, default: int = 1024) -> "MellonParam":
        return cls(name="height", label="Height", type="int", default=default, min=64, step=8)

    @classmethod
    def width(cls, default: int = 1024) -> "MellonParam":
        return cls(name="width", label="Width", type="int", default=default, min=64, step=8)

    @classmethod
    def seed(cls, default: int = 0) -> "MellonParam":
        return cls(name="seed", label="Seed", type="int", default=default, min=0, max=4294967295, display="random")

    @classmethod
    def num_inference_steps(cls, default: int = 25) -> "MellonParam":
        return cls(
            name="num_inference_steps", label="Steps", type="int", default=default, min=1, max=100, display="slider"
        )

    @classmethod
    def num_frames(cls, default: int = 81) -> "MellonParam":
        return cls(name="num_frames", label="Frames", type="int", default=default, min=1, max=480, display="slider")

    @classmethod
    def videos(cls) -> "MellonParam":
        return cls(name="videos", label="Videos", type="video", display="output")

    @classmethod
    def vae(cls) -> "MellonParam":
        """
        VAE model info dict.

        Contains keys like 'model_id', 'repo_id', 'execution_device' etc. Use components.get_one(model_id) to retrieve
        the actual model.
        """
        return cls(name="vae", label="VAE", type="diffusers_auto_model", display="input")

    @classmethod
    def unet(cls) -> "MellonParam":
        """
        Denoising model (UNet/Transformer) info dict.

        Contains keys like 'model_id', 'repo_id', 'execution_device' etc. Use components.get_one(model_id) to retrieve
        the actual model.
        """
        return cls(name="unet", label="Denoise Model", type="diffusers_auto_model", display="input")

    @classmethod
    def scheduler(cls) -> "MellonParam":
        """
        Scheduler model info dict.

        Contains keys like 'model_id', 'repo_id' etc. Use components.get_one(model_id) to retrieve the actual
        scheduler.
        """
        return cls(name="scheduler", label="Scheduler", type="diffusers_auto_model", display="input")

    @classmethod
    def controlnet(cls) -> "MellonParam":
        """
        ControlNet model info dict.

        Contains keys like 'model_id', 'repo_id', 'execution_device' etc. Use components.get_one(model_id) to retrieve
        the actual model.
        """
        return cls(name="controlnet", label="ControlNet Model", type="diffusers_auto_model", display="input")

    @classmethod
    def text_encoders(cls) -> "MellonParam":
        """
        Dict of text encoder model info dicts.

        Structure: {
            'text_encoder': {'model_id': ..., 'execution_device': ..., ...}, 'tokenizer': {'model_id': ..., ...},
            'repo_id': '...'
        } Use components.get_one(model_id) to retrieve each model.
        """
        return cls(name="text_encoders", label="Text Encoders", type="diffusers_auto_models", display="input")

    @classmethod
    def controlnet_bundle(cls, display: str = "input") -> "MellonParam":
        """
        ControlNet bundle containing model info and processed control inputs.

        Structure: {
            'controlnet': {'model_id': ..., ...}, # controlnet model info dict 'control_image': ..., # processed
            control image/embeddings 'controlnet_conditioning_scale': ..., ... # other inputs expected by denoise
            blocks
        }

        Output from Controlnet node, input to Denoise node.
        """
        return cls(name="controlnet_bundle", label="ControlNet", type="custom_controlnet", display=display)

    @classmethod
    def ip_adapter(cls) -> "MellonParam":
        return cls(name="ip_adapter", label="IP Adapter", type="custom_ip_adapter", display="input")

    @classmethod
    def guider(cls) -> "MellonParam":
        return cls(
            name="guider",
            label="Guider",
            type="custom_guider",
            display="input",
            onChange={False: ["guidance_scale"], True: []},
        )

    @classmethod
    def doc(cls) -> "MellonParam":
        return cls(name="doc", label="Doc", type="string", display="output")


def mark_required(label: str, marker: str = " *") -> str:
    """Add required marker to label if not already present."""
    if label.endswith(marker):
        return label
    return f"{label}{marker}"


def node_spec_to_mellon_dict(node_spec: Dict[str, Any], node_type: str) -> Dict[str, Any]:
    """
    Convert a node spec dict into Mellon format.

    A node spec is how we define a Mellon diffusers node in code. This function converts it into the `params` map
    format that Mellon UI expects.

    The `params` map is a dict where keys are parameter names and values are UI configuration:
        ```python
        {"seed": {"label": "Seed", "type": "int", "default": 0}}
        ```

    For Modular Mellon nodes, we need to distinguish:
        - `inputs`: Pipeline inputs (e.g., seed, prompt, image)
        - `model_inputs`: Model components (e.g., unet, vae, scheduler)
        - `outputs`: Node outputs (e.g., latents, images)

    The node spec also includes:
        - `required_inputs` / `required_model_inputs`: Which params are required (marked with *)
        - `block_name`: The modular pipeline block this node corresponds to on backend

    We provide factory methods for common parameters (e.g., `MellonParam.seed()`, `MellonParam.unet()`) so you don't
    have to manually specify all the UI configuration.

    Args:
        node_spec: Dict with `inputs`, `model_inputs`, `outputs` (lists of MellonParam),
                   plus `required_inputs`, `required_model_inputs`, `block_name`.
        node_type: The node type string (e.g., "denoise", "controlnet")

    Returns:
        Dict with:
            - `params`: Flat dict of all params in Mellon UI format
            - `input_names`: List of input parameter names
            - `model_input_names`: List of model input parameter names
            - `output_names`: List of output parameter names
            - `block_name`: The backend block name
            - `node_type`: The node type

    Example:
        ```python
        node_spec = {
            "inputs": [MellonParam.seed(), MellonParam.prompt()],
            "model_inputs": [MellonParam.unet()],
            "outputs": [MellonParam.latents(display="output")],
            "required_inputs": ["prompt"],
            "required_model_inputs": ["unet"],
            "block_name": "denoise",
        }

        result = node_spec_to_mellon_dict(node_spec, "denoise")
        # Returns:
        # {
        #     "params": {
        #         "seed": {"label": "Seed", "type": "int", "default": 0},
        #         "prompt": {"label": "Prompt *", "type": "string", "default": ""},  # * marks required
        #         "unet": {"label": "Denoise Model *", "type": "diffusers_auto_model", "display": "input"},
        #         "latents": {"label": "Latents", "type": "latents", "display": "output"},
        #     },
        #     "input_names": ["seed", "prompt"],
        #     "model_input_names": ["unet"],
        #     "output_names": ["latents"],
        #     "block_name": "denoise",
        #     "node_type": "denoise",
        # }
        ```
    """
    params = {}
    input_names = []
    model_input_names = []
    output_names = []

    required_inputs = node_spec.get("required_inputs", [])
    required_model_inputs = node_spec.get("required_model_inputs", [])

    # Process inputs
    for p in node_spec.get("inputs", []):
        param_dict = p.to_dict()
        if p.name in required_inputs:
            param_dict["label"] = mark_required(param_dict["label"])
        params[p.name] = param_dict
        input_names.append(p.name)

    # Process model_inputs
    for p in node_spec.get("model_inputs", []):
        param_dict = p.to_dict()
        if p.name in required_model_inputs:
            param_dict["label"] = mark_required(param_dict["label"])
        params[p.name] = param_dict
        model_input_names.append(p.name)

    # Process outputs
    for p in node_spec.get("outputs", []):
        params[p.name] = p.to_dict()
        output_names.append(p.name)

    return {
        "params": params,
        "input_names": input_names,
        "model_input_names": model_input_names,
        "output_names": output_names,
        "block_name": node_spec.get("block_name"),
        "node_type": node_type,
    }


class MellonPipelineConfig:
    """
    Configuration for an entire Mellon pipeline containing multiple nodes.

    Accepts node specs as dicts with inputs/model_inputs/outputs lists of MellonParam, converts them to Mellon-ready
    format, and handles save/load to Hub.

    Example:
        ```python
        config = MellonPipelineConfig(
            node_specs={
                "denoise": {
                    "inputs": [MellonParam.seed(), MellonParam.prompt()],
                    "model_inputs": [MellonParam.unet()],
                    "outputs": [MellonParam.latents(display="output")],
                    "required_inputs": ["prompt"],
                    "required_model_inputs": ["unet"],
                    "block_name": "denoise",
                },
                "decoder": {
                    "inputs": [MellonParam.latents(display="input")],
                    "outputs": [MellonParam.images()],
                    "block_name": "decoder",
                },
            },
            label="My Pipeline",
            default_repo="user/my-pipeline",
            default_dtype="float16",
        )

        # Access Mellon format dict
        denoise = config.node_params["denoise"]
        input_names = denoise["input_names"]
        params = denoise["params"]

        # Save to Hub
        config.save("./my_config", push_to_hub=True, repo_id="user/my-pipeline")

        # Load from Hub
        loaded = MellonPipelineConfig.load("user/my-pipeline")
        ```
    """

    config_name = "mellon_pipeline_config.json"

    def __init__(
        self,
        node_specs: Dict[str, Optional[Dict[str, Any]]],
        label: str = "",
        default_repo: str = "",
        default_dtype: str = "",
    ):
        """
        Args:
            node_specs: Dict mapping node_type to node spec or None.
                        Node spec has: inputs, model_inputs, outputs, required_inputs, required_model_inputs,
                        block_name (all optional)
            label: Human-readable label for the pipeline
            default_repo: Default HuggingFace repo for this pipeline
            default_dtype: Default dtype (e.g., "float16", "bfloat16")
        """
        # Convert all node specs to Mellon format immediately
        self.node_params = {}
        for node_type, spec in node_specs.items():
            if spec is None:
                self.node_params[node_type] = None
            else:
                self.node_params[node_type] = node_spec_to_mellon_dict(spec, node_type)

        self.label = label
        self.default_repo = default_repo
        self.default_dtype = default_dtype

    def __repr__(self) -> str:
        node_types = list(self.node_params.keys())
        return f"MellonPipelineConfig(label={self.label!r}, default_repo={self.default_repo!r}, default_dtype={self.default_dtype!r}, node_params={node_types})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "label": self.label,
            "default_repo": self.default_repo,
            "default_dtype": self.default_dtype,
            "node_params": self.node_params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MellonPipelineConfig":
        """
        Create from a dictionary (loaded from JSON).

        Note: The mellon_params are already in Mellon format when loading from JSON.
        """
        instance = cls.__new__(cls)
        instance.node_params = data.get("node_params", {})
        instance.label = data.get("label", "")
        instance.default_repo = data.get("default_repo", "")
        instance.default_dtype = data.get("default_dtype", "")
        return instance

    def to_json_string(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """Save to a JSON file."""
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    @classmethod
    def from_json_file(cls, json_file_path: Union[str, os.PathLike]) -> "MellonPipelineConfig":
        """Load from a JSON file."""
        with open(json_file_path, "r", encoding="utf-8") as reader:
            data = json.load(reader)
        return cls.from_dict(data)

    def save(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """Save the pipeline config to a directory."""
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)
        output_path = os.path.join(save_directory, self.config_name)
        self.to_json_file(output_path)
        logger.info(f"Pipeline config saved to {output_path}")

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            private = kwargs.pop("private", None)
            create_pr = kwargs.pop("create_pr", False)
            token = kwargs.pop("token", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = create_repo(repo_id, exist_ok=True, private=private, token=token).repo_id
            subfolder = kwargs.pop("subfolder", None)

            upload_folder(
                repo_id=repo_id,
                folder_path=save_directory,
                token=token,
                commit_message=commit_message or "Upload MellonPipelineConfig",
                create_pr=create_pr,
                path_in_repo=subfolder,
            )
            logger.info(f"Pipeline config pushed to hub: {repo_id}")

    @classmethod
    def load(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **kwargs,
    ) -> "MellonPipelineConfig":
        """Load a pipeline config from a local path or Hugging Face Hub."""
        cache_dir = kwargs.pop("cache_dir", None)
        local_dir = kwargs.pop("local_dir", None)
        local_dir_use_symlinks = kwargs.pop("local_dir_use_symlinks", "auto")
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        if os.path.isfile(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        elif os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, cls.config_name)
            if not os.path.isfile(config_file):
                raise EnvironmentError(f"No file named {cls.config_name} found in {pretrained_model_name_or_path}")
        else:
            try:
                config_file = hf_hub_download(
                    pretrained_model_name_or_path,
                    filename=cls.config_name,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    local_dir=local_dir,
                    local_dir_use_symlinks=local_dir_use_symlinks,
                )
            except RepositoryNotFoundError:
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier"
                    " listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a"
                    " token having permission to this repo with `token` or log in with `hf auth login`."
                )
            except RevisionNotFoundError:
                raise EnvironmentError(
                    f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for"
                    " this model name. Check the model page at"
                    f" 'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
                )
            except EntryNotFoundError:
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} does not appear to have a file named {cls.config_name}."
                )
            except HfHubHTTPError as err:
                raise EnvironmentError(
                    "There was a specific connection error when trying to load"
                    f" {pretrained_model_name_or_path}:\n{err}"
                )
            except ValueError:
                raise EnvironmentError(
                    f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it"
                    f" in the cached files and it looks like {pretrained_model_name_or_path} is not the path to a"
                    f" directory containing a {cls.config_name} file.\nCheckout your internet connection or see how to"
                    " run the library in offline mode at"
                    " 'https://huggingface.co/docs/diffusers/installation#offline-mode'."
                )
            except EnvironmentError:
                raise EnvironmentError(
                    f"Can't load config for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                    "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                    f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                    f"containing a {cls.config_name} file"
                )

        try:
            return cls.from_json_file(config_file)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise EnvironmentError(f"The config file at '{config_file}' is not a valid JSON file.")
