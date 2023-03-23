# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import platform
from argparse import ArgumentParser

import huggingface_hub

from .. import __version__ as version
from ..utils import is_accelerate_available, is_torch_available, is_transformers_available, is_xformers_available
from . import BaseDiffusersCLICommand


def info_command_factory(_):
    return EnvironmentCommand()


class EnvironmentCommand(BaseDiffusersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        download_parser = parser.add_parser("env")
        download_parser.set_defaults(func=info_command_factory)

    def run(self):
        hub_version = huggingface_hub.__version__

        pt_version = "not installed"
        pt_cuda_available = "NA"
        if is_torch_available():
            import torch

            pt_version = torch.__version__
            pt_cuda_available = torch.cuda.is_available()

        transformers_version = "not installed"
        if is_transformers_available():
            import transformers

            transformers_version = transformers.__version__

        accelerate_version = "not installed"
        if is_accelerate_available():
            import accelerate

            accelerate_version = accelerate.__version__

        xformers_version = "not installed"
        if is_xformers_available():
            import xformers

            xformers_version = xformers.__version__

        info = {
            "`diffusers` version": version,
            "Platform": platform.platform(),
            "Python version": platform.python_version(),
            "PyTorch version (GPU?)": f"{pt_version} ({pt_cuda_available})",
            "Huggingface_hub version": hub_version,
            "Transformers version": transformers_version,
            "Accelerate version": accelerate_version,
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
