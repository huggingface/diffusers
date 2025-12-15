import json
import logging
import os

# Simple typed wrapper for parameter overrides
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from huggingface_hub import create_repo, hf_hub_download, upload_folder
from huggingface_hub.utils import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    validate_hf_hub_args,
)

from ..utils import HUGGINGFACE_CO_RESOLVE_ENDPOINT, PushToHubMixin, extract_commit_hash
from .modular_pipeline import ModularPipelineBlocks


logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class MellonParam:
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
    _map_to_input: Any = None  # the block input name this parameter maps to

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return {k: v for k, v in data.items() if not k.startswith("_") and v is not None}

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
    def image_latents(cls, display: str = "input", use_strength: bool = False) -> "MellonParam":
        """
        `image_latents_input` is used to accept the image latents from vae encoder node.
        When the node receives the `image_latents_input`, the height and width input widget will be hidden, and height and width will be determined by the image latents.
        if `use_strength` is True, the strength slider will be shown.
        """
        if display == "input":
            onChange = {False: ["height", "width"], True: ["strength"]} if use_strength else {False: ["height", "width"], True: []}
        else:
            onChange = None
        return cls(name="image_latents", label="Image Latents", type="latents", display=display, onChange=onChange)

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
        return cls(name="controlnet_conditioning_scale", label="Controlnet Conditioning Scale", type="float", default=default, min=0.0, max=1.0, step=0.01)

    @classmethod
    def control_guidance_start(cls, default: float = 0.0) -> "MellonParam":
        return cls(name="control_guidance_start", label="Control Guidance Start", type="float", default=default, min=0.0, max=1.0, step=0.01)

    @classmethod
    def control_guidance_end(cls, default: float = 1.0) -> "MellonParam":
        return cls(name="control_guidance_end", label="Control Guidance End", type="float", default=default, min=0.0, max=1.0, step=0.01)
    
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
        return cls(name="guidance_scale", label="Guidance Scale", type="float", display="slider", default=default, min=1.0, max=30.0, step=0.1)

    @classmethod
    def height(cls, default: int = 1024) -> "MellonParam":
        return cls(name="height", label="Height", type="int", default=default, min=64,step=8)

    @classmethod
    def width(cls, default: int = 1024) -> "MellonParam":
        return cls(name="width", label="Width", type="int", default=default, min=64, step=8)
    
    @classmethod
    def seed(cls, default: int = 0) -> "MellonParam":
        return cls(name="seed", label="Seed", type="int", default=default, min=0, max=4294967295, display="random")

    @classmethod
    def num_inference_steps(cls, default: int = 25) -> "MellonParam":
        return cls(name="num_inference_steps", label="Steps", type="int", default=default, min=1, max=100, display="slider")

    @classmethod
    def vae(cls) -> "MellonParam":
        """
        VAE model info dict.
        
        Contains keys like 'model_id', 'repo_id', 'execution_device' etc.
        Use components.get_one(model_id) to retrieve the actual model.
        """
        return cls(name="vae", label="VAE", type="diffusers_auto_model", display="input")

    @classmethod
    def unet(cls) -> "MellonParam":
        """
        Denoising model (UNet/Transformer) info dict.
        
        Contains keys like 'model_id', 'repo_id', 'execution_device' etc.
        Use components.get_one(model_id) to retrieve the actual model.
        """
        return cls(name="unet", label="Denoise Model", type="diffusers_auto_model", display="input")

    @classmethod
    def scheduler(cls) -> "MellonParam":
        """
        Scheduler model info dict.
        
        Contains keys like 'model_id', 'repo_id' etc.
        Use components.get_one(model_id) to retrieve the actual scheduler.
        """
        return cls(name="scheduler", label="Scheduler", type="diffusers_auto_model", display="input")

    @classmethod
    def controlnet(cls) -> "MellonParam":
        """
        ControlNet model info dict.
        
        Contains keys like 'model_id', 'repo_id', 'execution_device' etc.
        Use components.get_one(model_id) to retrieve the actual model.
        """
        return cls(name="controlnet", label="ControlNet Model", type="diffusers_auto_model", display="input")

    @classmethod
    def text_encoders(cls) -> "MellonParam":
        """
        Dict of text encoder model info dicts.
        
        Structure: {
            'text_encoder': {'model_id': ..., 'execution_device': ..., ...},
            'tokenizer': {'model_id': ..., ...},
            'repo_id': '...'
        }
        Use components.get_one(model_id) to retrieve each model.
        """
        return cls(name="text_encoders", label="Text Encoders", type="diffusers_auto_models", display="input")

    @classmethod
    def controlnet_bundle(cls, display: str = "input") -> "MellonParam":
        """
        ControlNet bundle containing model info and processed control inputs.
        
        Structure: {
            'controlnet': {'model_id': ..., ...},  # controlnet model info dict
            'control_image': ...,                   # processed control image/embeddings
            'controlnet_conditioning_scale': ...,
            ...  # other inputs expected by denoise blocks
        }
        
        Output from Controlnet node, input to Denoise node.
        """
        return cls(name="controlnet_bundle", label="ControlNet", type="custom_controlnet", display=display)

    @classmethod
    def ip_adapter(cls) -> "MellonParam":
        return cls(name="ip_adapter", label="IP Adapter", type="custom_ip_adapter", display="input")

    @classmethod
    def guider(cls) -> "MellonParam":
        return cls(name="guider", label="Guider", type="custom_guider", display="input", onChange={False: ["guidance_scale"], True: []})

    @classmethod
    def doc(cls) -> "MellonParam":
        return cls(name="doc", label="Doc", type="string", display="output")

@dataclass
class MellonNodeConfig:
    """
    A MellonNodeConfig is a base class to build Mellon nodes UI with modular diffusers.
    It is used to configure a single Mellon node.

    <Tip warning={true}>

        This is an experimental feature and is likely to change in the future.

    </Tip>
    """

    inputs: List[MellonParam]
    model_inputs: List[MellonParam]
    outputs: List[MellonParam]
    block_name: str
    node_type: str
    required_inputs: List[str] = field(default_factory=list)
    required_model_inputs: List[str] = field(default_factory=list)
    config_name = "mellon_config.json"

    def __post_init__(self):

        self.inputs = self._params_list_to_dict(self.inputs, required=self.required_inputs)
        self.model_inputs = self._params_list_to_dict(self.model_inputs, required=self.required_model_inputs)
        self.outputs = self._params_list_to_dict(self.outputs)

    @staticmethod
    def _params_list_to_dict(params: List[MellonParam], required: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Convert a list of MellonParam to the `Param` definition for Mellon nodes."""
        required = required or []
        resolved = {}
        
        for param in params:
            param_dict = param.to_dict()
            param_name = param_dict.pop("name")
            
            if param_name in resolved:
                raise ValueError(f"Duplicate param '{param_name}'")
            
            # Mark required params with asterisk in label
            if param_name in required and not param_dict["label"].endswith(" *"):
                param_dict["label"] = f"{param_dict['label']} *"
            
            resolved[param_name] = param_dict
            logger.info(f"Resolved param: {param_name}")
        
        return resolved

    def to_mellon_dict(self) -> Dict[str, Any]:
        """Return a Json-serializable dict for Mellon Schema. Include:

        - `node_type`: The type of the node. Currently we support the following node types: "controlnet", "denoise", "vae_encoder", "text_encoder", "decode".
        - `block_name`: The name of the sub-block in the modular pipeline blocks this node corresponds to.
        - `params`:  The `Param` definitions for Mellon nodes. It is a single flat dict composed as: {**inputs, **model_inputs, **outputs}.
                     The keys are the parameter names, and the values are the parameter definitions.
        """
        # inputs/model_inputs/outputs are already normalized dicts
        merged_params = {}
        merged_params.update(self.inputs or {})
        merged_params.update(self.model_inputs or {})
        merged_params.update(self.outputs or {})

        return {
            "node_type": self.node_type,
            "block_name": self.block_name,
            "params": merged_params,
        }

    def from_mellon_dict(cls, mellon_dict: Dict[str, Any]) -> "MellonNodeConfig":
        """
        Create a MellonNodeConfig from a Mellon schema dict.
        
        Splits the flat params dict back into inputs/model_inputs/outputs
        based on the 'display' and 'type' fields.
        """
        flat_params = mellon_dict.get("params", {})

        inputs: List[MellonParam] = []
        model_inputs: List[MellonParam] = []
        outputs: List[MellonParam] = []

        for param_name, param_dict in flat_params.items():
            # Reconstruct MellonParam
            param = MellonParam(
                name=param_name,
                label=param_dict.get("label", param_name),
                type=param_dict.get("type", ""),
                display=param_dict.get("display", "input"),
                default=param_dict.get("default"),
                min=param_dict.get("min"),
                max=param_dict.get("max"),
                step=param_dict.get("step"),
                onChange=param_dict.get("onChange"),
            )
            
            # Categorize based on display/type
            if param_dict.get("display") == "output":
                outputs.append(param)
            elif param_dict.get("type") in ("diffusers_auto_model", "diffusers_auto_models"):
                model_inputs.append(param)
            else:
                inputs.append(param)

        return cls(
            inputs=inputs,
            model_inputs=model_inputs,
            outputs=outputs,
            block_name=mellon_dict.get("block_name", ""),
            node_type=mellon_dict.get("node_type", ""),
        )

    def to_json_string(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_mellon_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """Save to a JSON file."""
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())



class MellonPipelineConfig:
    """
    Configuration for an entire pipeline in Mellon that contains multiple nodes.
    
    This allows saving/loading all node configurations for a modularpipeline
    (e.g., Flux, SDXL, etc.) in a single JSON file.
    """
    
    config_name = "mellon_pipeline_config.json"

    def __init__(
        self,
        node_configs: Dict[str, Optional[MellonNodeConfig]],
        label: str = "",
        default_repo: str = "",
        default_dtype: str = "",
    ):
        """
        Args:
            node_configs: Dict mapping node_type -> MellonNodeConfig (or None if not supported).
            label: Display label for the pipeline.
            default_repo: Default HuggingFace repo for models.
            default_dtype: Default dtype (e.g., "torch.bfloat16").
        """
        self.node_configs = node_configs
        self.label = label
        self.default_repo = default_repo
        self.default_dtype = default_dtype

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        nodes = {}
        for node_type, node_config in self.node_configs.items():
            if node_config is None:
                nodes[node_type] = None
            else:
                nodes[node_type] = node_config.to_mellon_dict()
        
        return {
            "label": self.label,
            "default_repo": self.default_repo,
            "default_dtype": self.default_dtype,
            "nodes": nodes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MellonPipelineConfig":
        """Create from a dictionary (inverse of to_dict)."""
        node_configs = {}
        for node_type, node_data in data.get("nodes", {}).items():
            if node_data is None:
                node_configs[node_type] = None
            else:
                node_configs[node_type] = MellonNodeConfig.from_mellon_dict(node_data)
        
        return cls(
            node_configs=node_configs,
            label=data.get("label", ""),
            default_repo=data.get("default_repo", ""),
            default_dtype=data.get("default_dtype", ""),
        )

    def to_json_string(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

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
        """
        Save the pipeline config to a directory.
        
        Args:
            save_directory: Directory where the config JSON file is saved.
            push_to_hub: Whether to push to Hugging Face Hub after saving.
            **kwargs: Additional arguments for hub upload (repo_id, token, private, etc.).
        """
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
        """
        Load a pipeline config from a local path or Hugging Face Hub.
        
        Args:
            pretrained_model_name_or_path: Either:
                - A model id on the Hub (e.g., "username/my-pipeline-config")
                - A local directory path containing the config file
                - A direct path to the JSON config file
            cache_dir: Path to cache directory for downloaded files.
            force_download: Whether to force re-download even if cached.
            proxies: Dict of proxy servers to use.
            token: HF token for private repos.
            local_files_only: Whether to only look for local files.
            revision: Git revision (branch, tag, commit) to use.
            subfolder: Subfolder within the repo.
        
        Returns:
            MellonPipelineConfig instance.
        """
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
        
        # Case 1: Direct path to JSON file
        if os.path.isfile(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        
        # Case 2: Local directory
        elif os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, cls.config_name)
            if not os.path.isfile(config_file):
                raise EnvironmentError(
                    f"No file named {cls.config_name} found in {pretrained_model_name_or_path}"
                )
        
        # Case 3: Download from Hugging Face Hub
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
            config_dict = cls.from_json_file(config_file)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise EnvironmentError(f"It looks like the config file at '{config_file}' is not a valid JSON file.")
        return config_dict

    def get_node_config(self, node_type: str) -> Optional[MellonNodeConfig]:
        """Get the config for a specific node type."""
        return self.node_configs.get(node_type)

    def __getitem__(self, node_type: str) -> Optional[MellonNodeConfig]:
        """Allow dict-like access: pipeline_config['denoise']"""
        return self.get_node_config(node_type)

    def __iter__(self):
        """Iterate over node types."""
        return iter(self.node_configs)

    def items(self):
        """Iterate over (node_type, config) pairs."""
        return self.node_configs.items()
    
    def __repr__(self) -> str:
        node_types = list(self.node_configs.keys())
        return f"MellonPipelineConfig(label={self.label!r}, nodes={node_types})"
