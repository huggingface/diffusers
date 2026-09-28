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

from diffusers import Kandinsky6SRVAE
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import assert_tensors_close, enable_full_determinism, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
)


enable_full_determinism()


class Kandinsky6SRVAETesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return Kandinsky6SRVAE

    @property
    def pretrained_model_name_or_path(self):
        return "kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers"

    @property
    def pretrained_model_kwargs(self):
        return {"subfolder": "vae"}

    @property
    def main_input_name(self) -> str:
        return "sample"

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "encoder_block_out_channels": (4, 8, 8),
            "decoder_block_out_channels": (4, 8, 8),
            "layers_per_block": 1,
            "temporal_compression_ratio": 4,
            "temporal_compression_start_level": 0,
        }

    def get_dummy_inputs(self) -> dict:
        return {"sample": randn_tensor((2, 3, 9, 16, 16), generator=self.generator, device=torch_device)}

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (3, 9, 16, 16)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (3, 9, 16, 16)


class TestKandinsky6SRVAEModel(Kandinsky6SRVAETesterConfig, ModelTesterMixin):
    def test_compression_ratios(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        with torch.no_grad():
            latents = model.encode(self.get_dummy_inputs()["sample"]).latent_dist.mode()
        assert model.spatial_compression_ratio == 4
        assert model.temporal_compression_ratio == 4
        assert latents.shape == (2, 4, 3, 4, 4)

    def test_segmented_processing_matches_single_pass(self):
        # 41 frames span three causal segments; the carried padding must reproduce a single-segment result.
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        sample = randn_tensor((1, 3, 41, 16, 16), generator=self.generator, device=torch_device)
        with torch.no_grad():
            latents = model.encode(sample).latent_dist.mode()
            latents_prefix = model.encode(sample[:, :, :17]).latent_dist.mode()
            decoded = model.decode(latents).sample
            decoded_prefix = model.decode(latents[:, :, :5]).sample
        assert_tensors_close(latents_prefix, latents[:, :, :5], atol=1e-5, rtol=0)
        assert_tensors_close(decoded_prefix, decoded[:, :, :17], atol=1e-4, rtol=0)


class TestKandinsky6SRVAEMemory(Kandinsky6SRVAETesterConfig, MemoryTesterMixin):
    pass


class TestKandinsky6SRVAETorchCompile(Kandinsky6SRVAETesterConfig, TorchCompileTesterMixin):
    pass
