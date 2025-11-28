import json
import logging
import os

# Simple typed wrapper for parameter overrides
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from huggingface_hub import create_repo, hf_hub_download
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


SUPPORTED_NODE_TYPES = {"controlnet", "vae_encoder", "denoise", "text_encoder", "decoder"}


# Mellon Input Parameters (runtime parameters, not models)
MELLON_INPUT_PARAMS = {
    # controlnet
    "control_image": {
        "label": "Control Image",
        "type": "image",
        "display": "input",
    },
    "controlnet_conditioning_scale": {
        "label": "Scale",
        "type": "float",
        "default": 0.5,
        "min": 0,
        "max": 1,
    },
    "control_guidance_end": {
        "label": "End",
        "type": "float",
        "default": 1.0,
        "min": 0,
        "max": 1,
    },
    "control_guidance_start": {
        "label": "Start",
        "type": "float",
        "default": 0.0,
        "min": 0,
        "max": 1,
    },
    "controlnet": {
        "label": "Controlnet",
        "type": "custom_controlnet",
        "display": "input",
    },
    "embeddings": {
        "label": "Text Embeddings",
        "display": "input",
        "type": "embeddings",
    },
    "image": {
        "label": "Image",
        "type": "image",
        "display": "input",
    },
    "negative_prompt": {
        "label": "Negative Prompt",
        "type": "string",
        "default": "",
        "display": "textarea",
    },
    "prompt": {
        "label": "Prompt",
        "type": "string",
        "default": "",
        "display": "textarea",
    },
    "guidance_scale": {
        "label": "Guidance Scale",
        "type": "float",
        "display": "slider",
        "default": 5,
        "min": 1.0,
        "max": 30.0,
        "step": 0.1,
    },
    "height": {
        "label": "Height",
        "type": "int",
        "default": 1024,
        "min": 64,
        "step": 8,
    },
    "image_latents": {
        "label": "Image Latents",
        "type": "latents",
        "display": "input",
        "onChange": {False: ["height", "width"], True: ["strength"]},
    },
    "latents": {
        "label": "Latents",
        "type": "latents",
        "display": "input",
    },
    "num_inference_steps": {
        "label": "Steps",
        "type": "int",
        "display": "slider",
        "default": 25,
        "min": 1,
        "max": 100,
    },
    "seed": {
        "label": "Seed",
        "type": "int",
        "display": "random",
        "default": 0,
        "min": 0,
        "max": 4294967295,
    },
    "strength": {
        "label": "Strength",
        "type": "float",
        "default": 0.5,
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
    },
    "width": {
        "label": "Width",
        "type": "int",
        "default": 1024,
        "min": 64,
        "step": 8,
    },
    "ip_adapter": {
        "label": "IP Adapter",
        "type": "custom_ip_adapter",
        "display": "input",
    },
}

# Mellon Model Parameters (diffusers_auto_model types)
MELLON_MODEL_PARAMS = {
    "scheduler": {
        "label": "Scheduler",
        "display": "input",
        "type": "diffusers_auto_model",
    },
    "text_encoders": {
        "label": "Text Encoders",
        "type": "diffusers_auto_models",
        "display": "input",
    },
    "unet": {
        "label": "Unet",
        "display": "input",
        "type": "diffusers_auto_model",
        "onSignal": {
            "action": "signal",
            "target": "guider",
        },
    },
    "guider": {
        "label": "Guider",
        "display": "input",
        "type": "custom_guider",
        "onChange": {False: ["guidance_scale"], True: []},
    },
    "vae": {
        "label": "VAE",
        "display": "input",
        "type": "diffusers_auto_model",
    },
    "controlnet": {
        "label": "Controlnet Model",
        "type": "diffusers_auto_model",
        "display": "input",
    },
}

# Mellon Output Parameters (display = "output")
MELLON_OUTPUT_PARAMS = {
    "embeddings": {
        "label": "Text Embeddings",
        "display": "output",
        "type": "embeddings",
    },
    "images": {
        "label": "Images",
        "type": "image",
        "display": "output",
    },
    "image_latents": {
        "label": "Image Latents",
        "type": "latents",
        "display": "output",
    },
    "latents": {
        "label": "Latents",
        "type": "latents",
        "display": "output",
    },
    "latents_preview": {
        "label": "Latents Preview",
        "display": "output",
        "type": "latent",
    },
    "controlnet_out": {
        "label": "Controlnet",
        "display": "output",
        "type": "controlnet",
    },
}


# Default param selections per supported node_type
# from MELLON_INPUT_PARAMS / MELLON_MODEL_PARAMS / MELLON_OUTPUT_PARAMS.
NODE_TYPE_PARAMS_MAP = {
    "controlnet": {
        "inputs": [
            "control_image",
            "controlnet_conditioning_scale",
            "control_guidance_start",
            "control_guidance_end",
            "height",
            "width",
        ],
        "model_inputs": [
            "controlnet",
            "vae",
        ],
        "outputs": [
            "controlnet",
        ],
        "block_names": ["controlnet_vae_encoder"],
    },
    "denoise": {
        "inputs": [
            "embeddings",
            "width",
            "height",
            "seed",
            "num_inference_steps",
            "guidance_scale",
            "image_latents",
            "strength",
            # custom adapters coming in as inputs
            "controlnet",
            # ip_adapter is optional and custom; include if available
            "ip_adapter",
        ],
        "model_inputs": [
            "unet",
            "guider",
            "scheduler",
        ],
        "outputs": [
            "latents",
            "latents_preview",
        ],
        "block_names": ["denoise"],
    },
    "vae_encoder": {
        "inputs": [
            "image",
            "width",
            "height",
        ],
        "model_inputs": [
            "vae",
        ],
        "outputs": [
            "image_latents",
        ],
        "block_names": ["vae_encoder"],
    },
    "text_encoder": {
        "inputs": [
            "prompt",
            "negative_prompt",
            # optional image prompt input supported in embeddings node
            "image",
        ],
        "model_inputs": [
            "text_encoders",
        ],
        "outputs": [
            "embeddings",
        ],
        "block_names": ["text_encoder"],
    },
    "decoder": {
        "inputs": [
            "latents",
        ],
        "model_inputs": [
            "vae",
        ],
        "outputs": [
            "images",
        ],
        "block_names": ["decode"],
    },
}


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


@dataclass
class MellonNodeConfig(PushToHubMixin):
    """
    A MellonNodeConfig is a base class to build Mellon nodes UI with modular diffusers.

    <Tip warning={true}>

        This is an experimental feature and is likely to change in the future.

    </Tip>
    """

    inputs: List[Union[str, MellonParam]]
    model_inputs: List[Union[str, MellonParam]]
    outputs: List[Union[str, MellonParam]]
    blocks_names: list[str]
    node_type: str
    config_name = "mellon_config.json"

    def __post_init__(self):
        if isinstance(self.inputs, list):
            self.inputs = self._resolve_params_list(self.inputs, MELLON_INPUT_PARAMS)
        if isinstance(self.model_inputs, list):
            self.model_inputs = self._resolve_params_list(self.model_inputs, MELLON_MODEL_PARAMS)
        if isinstance(self.outputs, list):
            self.outputs = self._resolve_params_list(self.outputs, MELLON_OUTPUT_PARAMS)

    @staticmethod
    def _resolve_params_list(
        params: List[Union[str, MellonParam]], default_map: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        def _resolve_param(
            param: Union[str, MellonParam], default_params_map: Dict[str, Dict[str, Any]]
        ) -> Tuple[str, Dict[str, Any]]:
            if isinstance(param, str):
                if param not in default_params_map:
                    raise ValueError(f"Unknown param '{param}', please define a `MellonParam` object instead")
                return param, default_params_map[param].copy()
            elif isinstance(param, MellonParam):
                param_dict = param.to_dict()
                param_name = param_dict.pop("name")
                return param_name, param_dict
            else:
                raise ValueError(
                    f"Unknown param type '{type(param)}', please use a string or a  `MellonParam` object instead"
                )

        resolved = {}
        for p in params:
            logger.info(f" Resolving param: {p}")
            name, cfg = _resolve_param(p, default_map)
            if name in resolved:
                raise ValueError(f"Duplicate param '{name}'")
            resolved[name] = cfg
        return resolved

    @classmethod
    @validate_hf_hub_args
    def load_mellon_config(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        return_unused_kwargs=False,
        return_commit_hash=False,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        r"""
        Load a model or scheduler configuration.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing model weights saved with
                      [`~ConfigMixin.save_config`].

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
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
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            return_unused_kwargs (`bool`, *optional*, defaults to `False):
                Whether unused keyword arguments of the config are returned.
            return_commit_hash (`bool`, *optional*, defaults to `False):
                Whether the `commit_hash` of the loaded configuration are returned.

        Returns:
            `dict`:
                A dictionary of all the parameters stored in a JSON configuration file.

        """
        cache_dir = kwargs.pop("cache_dir", None)
        local_dir = kwargs.pop("local_dir", None)
        local_dir_use_symlinks = kwargs.pop("local_dir_use_symlinks", "auto")
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        if cls.config_name is None:
            raise ValueError(
                "`self.config_name` is not defined. Note that one should not load a config from "
                "`ConfigMixin`. Please make sure to define `config_name` in a class inheriting from `ConfigMixin`"
            )
        if os.path.isfile(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        elif os.path.isdir(pretrained_model_name_or_path):
            if os.path.isfile(os.path.join(pretrained_model_name_or_path, cls.config_name)):
                # Load from a PyTorch checkpoint
                config_file = os.path.join(pretrained_model_name_or_path, cls.config_name)
            else:
                raise EnvironmentError(
                    f"Error no file named {cls.config_name} found in directory {pretrained_model_name_or_path}."
                )
        else:
            try:
                # Load from URL or cache if already cached
                config_file = hf_hub_download(
                    pretrained_model_name_or_path,
                    filename=cls.config_name,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
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
            with open(config_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            config_dict = json.loads(text)

            commit_hash = extract_commit_hash(config_file)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise EnvironmentError(f"It looks like the config file at '{config_file}' is not a valid JSON file.")

        if not (return_unused_kwargs or return_commit_hash):
            return config_dict

        outputs = (config_dict,)

        if return_unused_kwargs:
            outputs += (kwargs,)

        if return_commit_hash:
            outputs += (commit_hash,)

        return outputs

    def save_mellon_config(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save the Mellon node definition to a JSON file.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file is saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face Hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        # If we save using the predefined names, we can load using `from_config`
        output_config_file = os.path.join(save_directory, self.config_name)

        self.to_json_file(output_config_file)
        logger.info(f"Mellon node definition saved in {output_config_file}")

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            private = kwargs.pop("private", None)
            create_pr = kwargs.pop("create_pr", False)
            token = kwargs.pop("token", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = create_repo(repo_id, exist_ok=True, private=private, token=token).repo_id
            subfolder = kwargs.pop("subfolder", None)

            self._upload_folder(
                save_directory,
                repo_id,
                token=token,
                commit_message=commit_message,
                create_pr=create_pr,
                subfolder=subfolder,
            )

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save the Mellon schema dictionary to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file to save a configuration instance's parameters.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string of the Mellon schema dict.

        Args:
        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """

        mellon_dict = self.to_mellon_dict()
        return json.dumps(mellon_dict, indent=2, sort_keys=True) + "\n"

    def to_mellon_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict focusing on the Mellon schema fields only.

        params is a single flat dict composed as: {**inputs, **model_inputs, **outputs}.
        """
        # inputs/model_inputs/outputs are already normalized dicts
        merged_params = {}
        merged_params.update(self.inputs or {})
        merged_params.update(self.model_inputs or {})
        merged_params.update(self.outputs or {})

        return {
            "node_type": self.node_type,
            "blocks_names": self.blocks_names,
            "params": merged_params,
        }

    @classmethod
    def from_mellon_dict(cls, mellon_dict: Dict[str, Any]) -> "MellonNodeConfig":
        """Create a config from a Mellon schema dict produced by to_mellon_dict().

        Splits the flat params dict back into inputs/model_inputs/outputs using the known key spaces from
        MELLON_INPUT_PARAMS / MELLON_MODEL_PARAMS / MELLON_OUTPUT_PARAMS. Unknown keys are treated as inputs by
        default.
        """
        flat_params = mellon_dict.get("params", {})

        inputs: Dict[str, Any] = {}
        model_inputs: Dict[str, Any] = {}
        outputs: Dict[str, Any] = {}

        for param_name, param_dict in flat_params.items():
            if param_dict.get("display", "") == "output":
                outputs[param_name] = param_dict
            elif param_dict.get("type", "") in ("diffusers_auto_model", "diffusers_auto_models"):
                model_inputs[param_name] = param_dict
            else:
                inputs[param_name] = param_dict

        return cls(
            inputs=inputs,
            model_inputs=model_inputs,
            outputs=outputs,
            blocks_names=mellon_dict.get("blocks_names", []),
            node_type=mellon_dict.get("node_type"),
        )

    # YiYi Notes: not used yet
    @classmethod
    def from_blocks(cls, blocks: ModularPipelineBlocks, node_type: str) -> "MellonNodeConfig":
        """
        Create an instance from a ModularPipeline object. If a preset exists in NODE_TYPE_PARAMS_MAP for the node_type,
        use it; otherwise fall back to deriving lists from the pipeline's expected inputs/components/outputs.
        """
        if node_type not in NODE_TYPE_PARAMS_MAP:
            raise ValueError(f"Node type {node_type} not supported")

        blocks_names = list(blocks.sub_blocks.keys())

        default_node_config = NODE_TYPE_PARAMS_MAP[node_type]
        inputs_list: List[Union[str, MellonParam]] = default_node_config.get("inputs", [])
        model_inputs_list: List[Union[str, MellonParam]] = default_node_config.get("model_inputs", [])
        outputs_list: List[Union[str, MellonParam]] = default_node_config.get("outputs", [])

        for required_input_name in blocks.required_inputs:
            if required_input_name not in inputs_list:
                inputs_list.append(
                    MellonParam(
                        name=required_input_name, label=required_input_name, type=required_input_name, display="input"
                    )
                )

        for component_spec in blocks.expected_components:
            if component_spec.name not in model_inputs_list:
                model_inputs_list.append(
                    MellonParam(
                        name=component_spec.name,
                        label=component_spec.name,
                        type="diffusers_auto_model",
                        display="input",
                    )
                )

        return cls(
            inputs=inputs_list,
            model_inputs=model_inputs_list,
            outputs=outputs_list,
            blocks_names=blocks_names,
            node_type=node_type,
        )


# Minimal modular registry for Mellon node configs
class ModularMellonNodeRegistry:
    """Registry mapping (pipeline class, blocks_name) -> list of MellonNodeConfig."""

    def __init__(self):
        self._registry = {}
        self._initialized = False

    def register(self, pipeline_cls: type, node_params: Dict[str, MellonNodeConfig]):
        if not self._initialized:
            _initialize_registry(self)
        self._registry[pipeline_cls] = node_params

    def get(self, pipeline_cls: type) -> MellonNodeConfig:
        if not self._initialized:
            _initialize_registry(self)
        return self._registry.get(pipeline_cls, None)

    def get_all(self) -> Dict[type, Dict[str, MellonNodeConfig]]:
        if not self._initialized:
            _initialize_registry(self)
        return self._registry


def _register_preset_node_types(
    pipeline_cls, params_map: Dict[str, Dict[str, Any]], registry: ModularMellonNodeRegistry
):
    """Register all node-type presets for a given pipeline class from a params map."""
    node_configs = {}
    for node_type, spec in params_map.items():
        node_config = MellonNodeConfig(
            inputs=spec.get("inputs", []),
            model_inputs=spec.get("model_inputs", []),
            outputs=spec.get("outputs", []),
            blocks_names=spec.get("block_names", []),
            node_type=node_type,
        )
        node_configs[node_type] = node_config
    registry.register(pipeline_cls, node_configs)


def _initialize_registry(registry: ModularMellonNodeRegistry):
    """Initialize the registry and register all available pipeline configs."""
    print("Initializing registry")

    registry._initialized = True

    try:
        from .qwenimage.modular_pipeline import QwenImageModularPipeline
        from .qwenimage.node_utils import QwenImage_NODE_TYPES_PARAMS_MAP

        _register_preset_node_types(QwenImageModularPipeline, QwenImage_NODE_TYPES_PARAMS_MAP, registry)
    except Exception:
        raise Exception("Failed to register QwenImageModularPipeline")

    try:
        from .stable_diffusion_xl.modular_pipeline import StableDiffusionXLModularPipeline
        from .stable_diffusion_xl.node_utils import SDXL_NODE_TYPES_PARAMS_MAP

        _register_preset_node_types(StableDiffusionXLModularPipeline, SDXL_NODE_TYPES_PARAMS_MAP, registry)
    except Exception:
        raise Exception("Failed to register StableDiffusionXLModularPipeline")
