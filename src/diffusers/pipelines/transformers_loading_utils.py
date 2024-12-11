# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
import contextlib
import tempfile
from typing import TYPE_CHECKING, Dict

from huggingface_hub import DDUFEntry

from ..utils import is_safetensors_available, is_transformers_available


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

if is_transformers_available():
    from transformers import PreTrainedModel, PreTrainedTokenizer

if is_safetensors_available():
    import safetensors.torch


def load_tokenizer_from_dduf(
    cls: "PreTrainedTokenizer", name: str, dduf_entries: Dict[str, DDUFEntry]
) -> "PreTrainedTokenizer":
    """
    Load a tokenizer from a DDUF archive.

    In practice, `transformers` do not provide a way to load a tokenizer from a DDUF archive. This function is a workaround
    by extracting the tokenizer files from the DDUF archive and loading the tokenizer from the extracted files. There is an
    extra cost of extracting the files, but of limited impact as the tokenizer files are usually small-ish.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        for entry_name, entry in dduf_entries.items():
            if entry_name.startswith(name + "/"):
                tmp_entry_path = os.path.join(tmp_dir, *entry_name.split("/"))
                with open(tmp_entry_path, "wb") as f:
                    with entry.as_mmap() as mm:
                        f.write(mm)
        return cls.from_pretrained(tmp_dir, **kwargs)


def load_transformers_model_from_dduf(
    cls: "PreTrainedModel", name: str, dduf_entries: Dict[str, DDUFEntry], **kwargs
) -> "PreTrainedModel":
    """
    Load a transformers model from a DDUF archive.

    In practice, `transformers` do not provide a way to load a model from a DDUF archive. This function is a workaround
    by instantiating a model from the config file and loading the weights from the DDUF archive directly.
    """
    config_file = dduf_entries.get(f"{name}/config.json")
    if config_file is None:
        raise EnvironmentError(
            f"Could not find a config.json file for component {name} in DDUF file (contains {dduf_entries.keys()})."
        )

    weight_files = [
        entry
        for entry_name, entry in dduf_entries.items()
        if entry_name.startswith(f"{name}/") and entry_name.endswith(".safetensors")
    ]
    if not weight_files:
        raise EnvironmentError(
            f"Could not find any weight file for component {name} in DDUF file (contains {dduf_entries.keys()})."
        )
    if not is_safetensors_available():
        raise EnvironmentError(
            "Safetensors is not available, cannot load model from DDUF. Please `pip install safetensors`."
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_config_file = os.path.join(tmp_dir, "config.json")
        with open(tmp_config_file, "w") as f:
            f.write(config_file.read_text())

        with contextlib.ExitStack() as stack:
            state_dict = {
                key: tensor
                for entry in weight_files  # loop over safetensors files
                for key, tensor in safetensors.torch.load(  # load tensors from mmap-ed bytes
                    stack.enter_context(entry.as_mmap())  # use enter_context to close the mmap when done
                ).items()
            }
            return cls.from_pretrained(tmp_dir, state_dict=state_dict, **kwargs)
