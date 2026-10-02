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

from diffusers import Kandinsky6SRLatentUpscalerBank
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
)


enable_full_determinism()


class Kandinsky6SRLatentUpscalerBankTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return Kandinsky6SRLatentUpscalerBank

    @property
    def pretrained_model_name_or_path(self):
        return "kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers"

    @property
    def pretrained_model_kwargs(self):
        return {"subfolder": "latent_upscaler"}

    @property
    def main_input_name(self) -> str:
        return "latents"

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "in_channels": 4,
            "stage_channels": (8, 8, 4),
            "num_pre_blocks": 1,
            "num_mid_blocks": 1,
            "num_post_blocks": 1,
            "num_x2_adapter_blocks": 1,
            "scales": (2, 4),
        }

    def get_dummy_inputs(self) -> dict:
        return {
            "latents": randn_tensor((2, 4, 3, 4, 4), generator=self.generator, device=torch_device),
            "scale": 2,
        }

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (4, 3, 4, 4)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (4, 3, 8, 8)


class TestKandinsky6SRLatentUpscalerBankModel(Kandinsky6SRLatentUpscalerBankTesterConfig, ModelTesterMixin):
    def test_x4_scale(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        inputs = self.get_dummy_inputs()
        with torch.no_grad():
            output = model(inputs["latents"], scale=4).sample
        assert output.shape == (2, 4, 3, 16, 16)


class TestKandinsky6SRLatentUpscalerBankMemory(Kandinsky6SRLatentUpscalerBankTesterConfig, MemoryTesterMixin):
    pass


class TestKandinsky6SRLatentUpscalerBankTorchCompile(
    Kandinsky6SRLatentUpscalerBankTesterConfig, TorchCompileTesterMixin
):
    pass
