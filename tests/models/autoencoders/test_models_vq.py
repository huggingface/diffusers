# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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

import numpy as np
import pytest
import torch

from diffusers import VQModel
from diffusers.models.autoencoders.vae import VectorQuantizer
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import backend_manual_seed, enable_full_determinism, torch_device
from ..testing_utils import BaseModelTesterConfig, MemoryTesterMixin, ModelTesterMixin, TrainingTesterMixin
from .testing_utils import NewAutoencoderTesterMixin


enable_full_determinism()


class VQModelTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return VQModel

    @property
    def main_input_name(self) -> str:
        return "sample"

    @property
    def output_shape(self) -> tuple:
        return (3, 32, 32)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "block_out_channels": [8, 16],
            "norm_num_groups": 8,
            "in_channels": 3,
            "out_channels": 3,
            "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
            "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
            "latent_channels": 3,
        }

    def get_dummy_inputs(self) -> dict:
        batch_size = 4
        num_channels = 3
        sizes = (32, 32)
        image = randn_tensor((batch_size, num_channels, *sizes), generator=self.generator, device=torch_device)
        return {"sample": image}


class TestVQModel(VQModelTesterConfig, ModelTesterMixin):
    def test_from_pretrained_hub(self):
        model, loading_info = VQModel.from_pretrained("fusing/vqgan-dummy", output_loading_info=True)
        assert model is not None
        assert len(loading_info["missing_keys"]) == 0

        model.to(torch_device)
        image = model(**self.get_dummy_inputs())

        assert image is not None, "Make sure output is not None"

    def test_output_pretrained(self):
        model = VQModel.from_pretrained("fusing/vqgan-dummy")
        model.to(torch_device).eval()

        torch.manual_seed(0)
        backend_manual_seed(torch_device, 0)

        image = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
        image = image.to(torch_device)
        with torch.no_grad():
            output = model(image).sample

        output_slice = output[0, -1, -3:, -3:].flatten().cpu()
        # fmt: off
        expected_output_slice = torch.tensor([-0.0153, -0.4044, -0.1880, -0.5161, -0.2418, -0.4072, -0.1612, -0.0633, -0.0143])
        # fmt: on
        assert torch.allclose(output_slice, expected_output_slice, atol=1e-3)

    def test_loss_pretrained(self):
        model = VQModel.from_pretrained("fusing/vqgan-dummy")
        model.to(torch_device).eval()

        torch.manual_seed(0)
        backend_manual_seed(torch_device, 0)

        image = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
        image = image.to(torch_device)
        with torch.no_grad():
            output = model(image).commit_loss.cpu()
        # fmt: off
        expected_output = torch.tensor([0.1936])
        # fmt: on
        assert torch.allclose(output, expected_output, atol=1e-3)

    def test_vector_quantizer_logs_remap_configuration(self, tmp_path):
        remap_path = tmp_path / "used.npy"
        np.save(remap_path, np.array([0, 2, 4], dtype=np.int64))

        with self.assertLogs("diffusers.models.autoencoders.vae", level="INFO") as captured_logs:
            VectorQuantizer(n_e=8, vq_embed_dim=4, beta=0.25, remap=str(remap_path), unknown_index="extra")

        assert any(
            "Remapping 8 indices to 4 indices. Using 3 for unknown indices." in message
            for message in captured_logs.output
        )


class TestVQModelTraining(VQModelTesterConfig, TrainingTesterMixin):
    """Training tests for VQModel."""

    @pytest.mark.skip("Test not supported.")
    def test_training(self):
        super().test_training()

    @pytest.mark.skip("Test not supported.")
    def test_training_with_ema(self):
        super().test_training_with_ema()


class TestVQModelMemory(VQModelTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for VQModel."""


class TestVQModelSlicingTiling(VQModelTesterConfig, NewAutoencoderTesterMixin):
    """Slicing and tiling tests for VQModel."""
