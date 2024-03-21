# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import platform
from argparse import ArgumentParser
from importlib import util

import huggingface_hub

from .. import __version__ as version
from ..utils import (
    is_accelerate_available,
    is_flax_available,
    is_google_colab,
    is_notebook,
    is_peft_available,
    is_safetensors_available,
    is_torch_available,
    is_transformers_available,
    is_xformers_available,
)
from . import BaseDiffusersCLICommand


def info_command_factory(_):
    return EnvironmentCommand()


def download_command_factory(args):
    return EnvironmentCommand(args.accelerate_config_file)


class EnvironmentCommand(BaseDiffusersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        download_parser = parser.add_parser("env")
        download_parser.set_defaults(func=info_command_factory)
        download_parser.add_argument(
            "--accelerate-config_file",
            default=None,
            help="The accelerate config file to use for the default values in the launching script.",
        )
        download_parser.set_defaults(func=download_command_factory)

    def __init__(self, accelerate_config_file, *args) -> None:
        self._accelerate_config_file = accelerate_config_file

    def run(self):
        hub_version = huggingface_hub.__version__

        if is_safetensors_available():
            import safetensors

            safetensors_version = safetensors.__version__
        elif util.find_spec("safetensors") is not None:
            import safetensors

            safetensors_version = f"{safetensors.__version__} but is ignored because of PyTorch version too old."
        else:
            safetensors_version = "not installed"

        pt_version = "not installed"
        pt_cuda_available = "NA"
        if is_torch_available():
            import torch

            pt_version = torch.__version__
            pt_cuda_available = torch.cuda.is_available()

        flax_version = "not installed"
        jax_version = "not installed"
        jaxlib_version = "not installed"
        jax_backend = "NA"
        if is_flax_available():
            import flax
            import jax
            import jaxlib

            flax_version = flax.__version__
            jax_version = jax.__version__
            jaxlib_version = jaxlib.__version__
            jax_backend = jax.lib.xla_bridge.get_backend().platform

        transformers_version = "not installed"
        if is_transformers_available():
            import transformers

            transformers_version = transformers.__version__

        accelerate_version = "not installed"
        accelerate_config = accelerate_config_str = "not found"
        if is_accelerate_available():
            import accelerate
            from accelerate.commands.config import default_config_file, load_config_from_file

            accelerate_version = accelerate.__version__
            # Get the default from the config file.
            if self._accelerate_config_file is not None or os.path.isfile(default_config_file):
                accelerate_config = load_config_from_file(self._accelerate_config_file).to_dict()

            accelerate_config_str = (
                "\n".join([f"\t- {prop}: {val}" for prop, val in accelerate_config.items()])
                if isinstance(accelerate_config, dict)
                else f"\t{accelerate_config}"
            )

        peft_version = "not installed"
        if is_peft_available():
            import peft

            peft_version = peft.__version__

        xformers_version = "not installed"
        if is_xformers_available():
            import xformers

            xformers_version = xformers.__version__

        is_notebook_str = "Yes" if is_notebook() else "No"

        is_google_colab_str = "Yes" if is_google_colab() else "No"

        info = {
            "`diffusers` version": version,
            "Platform": platform.platform(),
            "Running on a notebook?": is_notebook_str,
            "Running on Google Colab?": is_google_colab_str,
            "Python version": platform.python_version(),
            "PyTorch version (GPU?)": f"{pt_version} ({pt_cuda_available})",
            "Flax version (CPU?/GPU?/TPU?)": f"{flax_version} ({jax_backend})",
            "Jax version": f"{jax_version}",
            "JaxLib version": f"{jaxlib_version}",
            "Huggingface_hub version": hub_version,
            "Transformers version": transformers_version,
            "Accelerate version": accelerate_version,
            "Accelerate config": f"{accelerate_config_str}",
            "PEFT version": peft_version,
            "SafeTensors version": f"{safetensors_version}",
            "xFormers version": xformers_version,
            "Using GPU in script?": "<fill in>",
            "Using distributed or parallel set-up in script?": "<fill in>",
        }

        print("\nCopy-and-paste the text below in your GitHub issue and FILL OUT the two last points.\n")
        print(self.format_dict(info))

        return info

    @staticmethod
    def format_dict(d):
        return "\n".join([f"- {prop}: {val}" for prop, val in d.items()]) + "\n"
