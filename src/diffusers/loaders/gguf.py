# coding=utf-8
# Copyright 2024 The ggml.ai team and The HuggingFace Inc. team. and pygguf author (github.com/99991)
# https://github.com/99991/pygguf
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

from array import array

import gguf
import torch
from tqdm import tqdm

from ..utils import is_torch_available
from ..utils.import_utils import is_gguf_available
from ..utils.logging import get_logger


TORCH_COMPATIBLE_QTYPES = {None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}

if is_torch_available():
    pass

logger = get_logger(__name__)


GGUF_TO_DIFFUSERS_MAPPING = {
    "ignore": {
        "GGUF": {
            "version": "version",
            "tensor_count": "tensor_count",
            "kv_count": "kv_count",
        },
        "general": {"file_type": "file_type", "quantization_version": "quantization_version"},
    },
}


class GGMLTensor(torch.Tensor):
    def __init__(self, dtype, axis):
        self._dtype = dtype
        self._axis = axis

    @property
    def axis(self):
        return self._axis

    @property
    def dtype(self):
        return self._dtype

    def numpy(self):
        return self.dequantize().cpu().numpy()

    def clone(self, *args, **kwargs):
        return self

    def detach(self, *args, **kwargs):
        return self


def _gguf_parse_value(_value, data_type):
    if not isinstance(data_type, list):
        data_type = [data_type]
    if len(data_type) == 1:
        data_type = data_type[0]
        array_data_type = None
    else:
        if data_type[0] != 9:
            raise ValueError("Received multiple types, therefore expected the first type to indicate an array.")
        data_type, array_data_type = data_type

    if data_type in [0, 1, 2, 3, 4, 5, 10, 11]:
        _value = int(_value[0])
    elif data_type in [6, 12]:
        _value = float(_value[0])
    elif data_type in [7]:
        _value = bool(_value[0])
    elif data_type in [8]:
        _value = array("B", list(_value)).tobytes().decode()
    elif data_type in [9]:
        _value = _gguf_parse_value(_value, array_data_type)
    return _value


def read_field(reader, field):
    value = reader.fields[field]
    return [_gguf_parse_value(value.parts[_data_index], value.types) for _data_index in value.data]


def load_gguf_checkpoint(gguf_checkpoint_path, return_tensors=False):
    """
    Load a GGUF file and return a dictionary of parsed parameters containing tensors, the parsed
    tokenizer and config attributes.

    Args:
        gguf_checkpoint_path (`str`):
            The path the to GGUF file to load
        return_tensors (`bool`, defaults to `True`):
            Whether to read the tensors from the file and return them. Not doing so is faster
            and only loads the metadata in memory.
    """

    if is_gguf_available() and is_torch_available():
        from gguf import GGUFReader
    else:
        logger.error(
            "Loading a GGUF checkpoint in PyTorch, requires both PyTorch and GGUF>=0.10.0 to be installed. Please see "
            "https://pytorch.org/ and https://github.com/ggerganov/llama.cpp/tree/master/gguf-py for installation instructions."
        )
        raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

    reader = GGUFReader(gguf_checkpoint_path)
    fields = reader.fields
    reader_keys = list(fields.keys())

    parsed_parameters = {}
    for tensor in tqdm(reader.tensors, desc="Converting and de-quantizing GGUF tensors..."):
        name = tensor.name
        weights = torch.from_numpy(tensor.data)

        parsed_parameters[name] = weights

    if len(reader_keys) > 0:
        logger.info(f"Some keys of the GGUF file were not considered: {reader_keys}")

    return parsed_parameters
