import importlib
import os
from typing import Optional, Union

from huggingface_hub.utils import validate_hf_hub_args

from ..configuration_utils import ConfigMixin
from ..utils import CONFIG_NAME
from .modeling_utils import ModelMixin


class AutoModel(ConfigMixin):
    config_name = CONFIG_NAME

    """TODO"""

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            "from_config(config) methods."
        )

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        """TODO"""
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)

        load_config_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "token": token,
            "local_files_only": local_files_only,
            "revision": revision,
            "subfolder": subfolder,
        }
        config = cls.load_config(pretrained_model_name_or_path, **load_config_kwargs)
        class_name = config["_class_name"]
        diffusers_module = importlib.import_module(__name__.split(".")[0])

        try:
            model_cls: ModelMixin = getattr(diffusers_module, class_name)
        except Exception:
            raise ValueError(f"Could not import the `{class_name}` class from diffusers.")

        kwargs = {**load_config_kwargs, **kwargs}
        return model_cls.from_pretrained(pretrained_model_name_or_path, **kwargs)
