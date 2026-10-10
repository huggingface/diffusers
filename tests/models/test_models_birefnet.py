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

import pytest
import torch

from diffusers import BiRefNetModel
from diffusers.utils.torch_utils import randn_tensor

from ..testing_utils import enable_full_determinism, torch_device
from .testing_utils import (
    AttentionTesterMixin,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
)


enable_full_determinism()


class BiRefNetTesterConfig:
    main_input_name = "sample"

    @property
    def model_class(self):
        return BiRefNetModel

    @property
    def pretrained_model_name_or_path(self):
        return None

    @property
    def pretrained_model_kwargs(self):
        return {}

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | list[int]]:
        return {"embed_dim": 8, "depths": (1, 1, 1, 1), "num_heads": (1, 1, 2, 4), "window_size": 2, "sample_size": 64}

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        return {"sample": randn_tensor((1, 3, 64, 64), generator=self.generator, device=torch_device)}

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (3, 64, 64)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (1, 64, 64)


class TestBiRefNetModel(BiRefNetTesterConfig, ModelTesterMixin):
    def test_torchvision_dependency_error(self, monkeypatch):
        from diffusers.utils import import_utils

        monkeypatch.setitem(
            import_utils.BACKENDS_MAPPING, "torchvision", (lambda: False, import_utils.TORCHVISION_IMPORT_ERROR)
        )
        with pytest.raises(ImportError, match="requires the torchvision library"):
            BiRefNetModel(**self.get_init_dict())


class TestBiRefNetMemory(BiRefNetTesterConfig, MemoryTesterMixin):
    @pytest.mark.skip(reason="This BiRefNet adapter contains inference heads and is not intended for training.")
    def test_layerwise_casting_training(self):
        pass


class TestBiRefNetTorchCompile(BiRefNetTesterConfig, TorchCompileTesterMixin):
    @property
    def different_shapes_for_compilation(self):
        return [(64, 64), (64, 128), (128, 128)]

    def get_dummy_inputs(self, height: int = 64, width: int = 64) -> dict[str, torch.Tensor]:
        return {"sample": randn_tensor((1, 3, height, width), generator=self.generator, device=torch_device)}


class TestBiRefNetAttention(BiRefNetTesterConfig, AttentionTesterMixin):
    pass
