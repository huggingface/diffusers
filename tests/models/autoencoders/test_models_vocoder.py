# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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

import torch

from diffusers import MMAudioVocoder
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
)


enable_full_determinism()


class MMAudioVocoderTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return MMAudioVocoder

    @property
    def pretrained_model_name_or_path(self):
        return "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers"

    @property
    def pretrained_model_kwargs(self):
        return {"subfolder": "vocoder"}

    @property
    def main_input_name(self) -> str:
        return "mel"

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "num_mels": 8,
            "upsample_initial_channel": 8,
            "upsample_rates": (2, 2),
            "upsample_kernel_sizes": (4, 4),
            "resblock_kernel_sizes": (3,),
            "resblock_dilation_sizes": ((1, 3),),
        }

    def get_dummy_inputs(self) -> dict:
        return {"mel": randn_tensor((2, 8, 16), generator=self.generator, device=torch_device)}

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (8, 16)

    @property
    def output_shape(self) -> tuple[int, ...]:
        # `upsample_rates` multiply to 4, so 16 mel frames decode to 64 samples.
        return (1, 64)


class TestMMAudioVocoderModel(MMAudioVocoderTesterConfig, ModelTesterMixin):
    pass


class TestMMAudioVocoderMemory(MMAudioVocoderTesterConfig, MemoryTesterMixin):
    pass


class TestMMAudioVocoderTorchCompile(MMAudioVocoderTesterConfig, TorchCompileTesterMixin):
    pass
