# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import subprocess
from argparse import ArgumentParser

import huggingface_hub

from .. import __version__ as version
from ..utils import (
    is_accelerate_available,
    is_bitsandbytes_available,
    is_flax_available,
    is_google_colab,
    is_peft_available,
    is_safetensors_available,
    is_torch_available,
    is_transformers_available,
    is_xformers_available,
)
from . import BaseDiffusersCLICommand


def info_command_factory(_):
    return EnvironmentCommand()


class EnvironmentCommand(BaseDiffusersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser) -> None:
        download_parser = parser.add_parser("env")
        download_parser.set_defaults(func=info_command_factory)

    def run(self) -> dict:
        hub_version = huggingface_hub.__version__

        safetensors_version = "not installed"
        if is_safetensors_available():
            import safetensors

            safetensors_version = safetensors.__version__

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
        if is_accelerate_available():
            import accelerate

            accelerate_version = accelerate.__version__

        peft_version = "not installed"
        if is_peft_available():
            import peft

            peft_version = peft.__version__

        bitsandbytes_version = "not installed"
        if is_bitsandbytes_available():
            import bitsandbytes

            bitsandbytes_version = bitsandbytes.__version__

        xformers_version = "not installed"
        if is_xformers_available():
            import xformers

            xformers_version = xformers.__version__

        platform_info = platform.platform()

        is_google_colab_str = "Yes" if is_google_colab() else "No"

        accelerator = "NA"
        if platform.system() in {"Linux", "Windows"}:
            try:
                sp = subprocess.Popen(
                    ["nvidia-smi", "--query-gpu=gpu_name,memory.total", "--format=csv,noheader"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                out_str, _ = sp.communicate()
                out_str = out_str.decode("utf-8")

                if len(out_str) > 0:
                    accelerator = out_str.strip()
            except FileNotFoundError:
                pass
        elif platform.system() == "Darwin":  # Mac OS
            try:
                sp = subprocess.Popen(
                    ["system_profiler", "SPDisplaysDataType"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                out_str, _ = sp.communicate()
                out_str = out_str.decode("utf-8")

                start = out_str.find("Chipset Model:")
                if start != -1:
                    start += len("Chipset Model:")
                    end = out_str.find("\n", start)
                    accelerator = out_str[start:end].strip()

                    start = out_str.find("VRAM (Total):")
                    if start != -1:
                        start += len("VRAM (Total):")
                        end = out_str.find("\n", start)
                        accelerator += " VRAM: " + out_str[start:end].strip()
            except FileNotFoundError:
                pass
        else:
            print("It seems you are running an unusual OS. Could you fill in the accelerator manually?")

        info = {
            "🤗 Diffusers version": version,
            "Platform": platform_info,
            "Running on Google Colab?": is_google_colab_str,
            "Python version": platform.python_version(),
            "PyTorch version (GPU?)": f"{pt_version} ({pt_cuda_available})",
            "Flax version (CPU?/GPU?/TPU?)": f"{flax_version} ({jax_backend})",
            "Jax version": jax_version,
            "JaxLib version": jaxlib_version,
            "Huggingface_hub version": hub_version,
            "Transformers version": transformers_version,
            "Accelerate version": accelerate_version,
            "PEFT version": peft_version,
            "Bitsandbytes version": bitsandbytes_version,
            "Safetensors version": safetensors_version,
            "xFormers version": xformers_version,
            "Accelerator": accelerator,
            "Using GPU in script?": "<fill in>",
            "Using distributed or parallel set-up in script?": "<fill in>",
        }

        print("\nCopy-and-paste the text below in your GitHub issue and FILL OUT the two last points.\n")
        print(self.format_dict(info))

        return info

    @staticmethod
    def format_dict(d: dict) -> str:
        return "\n".join([f"- {prop}: {val}" for prop, val in d.items()]) + "\n"
