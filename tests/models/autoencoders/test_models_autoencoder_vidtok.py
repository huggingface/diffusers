# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

from diffusers import AutoencoderVidTok
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import BaseModelTesterConfig, MemoryTesterMixin, ModelTesterMixin, TrainingTesterMixin
from .testing_utils import NewAutoencoderTesterMixin


enable_full_determinism()


def _run_nondeterministic(fn):
    # avg_pool3d_backward_cuda has no deterministic CUDA implementation;
    # temporarily relax the requirement for tests that do backward passes.
    torch.use_deterministic_algorithms(False)
    try:
        fn()
    finally:
        torch.use_deterministic_algorithms(True)


class AutoencoderVidTokTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return AutoencoderVidTok

    @property
    def main_input_name(self) -> str:
        return "sample"

    @property
    def output_shape(self) -> tuple:
        return (3, 16, 32, 32)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "is_causal": False,
            "in_channels": 3,
            "out_channels": 3,
            "ch": 128,
            "ch_mult": [1, 2, 4, 4, 4],
            "z_channels": 6,
            "double_z": False,
            "num_res_blocks": 2,
            "regularizer": "fsq",
            "codebook_size": 262144,
        }

    def get_dummy_inputs(self) -> dict:
        batch_size = 4
        num_frames = 16
        num_channels = 3
        sizes = (32, 32)
        image = randn_tensor(
            (batch_size, num_channels, num_frames, *sizes), generator=self.generator, device=torch_device
        )
        return {"sample": image}


class TestAutoencoderVidTok(AutoencoderVidTokTesterConfig, ModelTesterMixin):
    @pytest.mark.skip("VidTok output structure not compatible with recursive output check.")
    def test_outputs_equivalence(self):
        super().test_outputs_equivalence()


class TestAutoencoderVidTokTraining(AutoencoderVidTokTesterConfig, TrainingTesterMixin):
    """Training tests for AutoencoderVidTok."""

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"VidTokEncoder3D", "VidTokDecoder3D"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)

    def test_training(self):
        _run_nondeterministic(super().test_training)

    def test_training_with_ema(self):
        _run_nondeterministic(super().test_training_with_ema)

    def test_mixed_precision_training(self):
        _run_nondeterministic(super().test_mixed_precision_training)


class TestAutoencoderVidTokMemory(AutoencoderVidTokTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for AutoencoderVidTok."""

    def test_layerwise_casting_training(self):
        _run_nondeterministic(super().test_layerwise_casting_training)


class TestAutoencoderVidTokSlicingTiling(AutoencoderVidTokTesterConfig, NewAutoencoderTesterMixin):
    """Slicing and tiling tests for AutoencoderVidTok."""

    def test_enable_disable_tiling(self):
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()

        torch.manual_seed(0)
        model = self.model_class(**init_dict).to(torch_device)

        torch.manual_seed(0)
        output_without_tiling = model(**inputs_dict, generator=torch.manual_seed(0))[0]

        torch.manual_seed(0)
        model.enable_tiling()
        output_with_tiling = model(**inputs_dict, generator=torch.manual_seed(0))[0]

        assert (
            output_without_tiling.detach().cpu().numpy() - output_with_tiling.detach().cpu().numpy()
        ).max() < 0.5, "VAE tiling should not affect the inference results"

        torch.manual_seed(0)
        model.disable_tiling()
        output_without_tiling_2 = model(**inputs_dict, generator=torch.manual_seed(0))[0]

        assert (
            output_without_tiling.detach().cpu().numpy().all() == output_without_tiling_2.detach().cpu().numpy().all()
        ), "Without tiling outputs should match with the outputs when tiling is manually disabled."

    def test_enable_disable_slicing(self):
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()

        torch.manual_seed(0)
        model = self.model_class(**init_dict).to(torch_device)
        inputs_dict.update({"return_dict": False})

        torch.manual_seed(0)
        output_without_slicing = model(**inputs_dict, generator=torch.manual_seed(0))[0]

        torch.manual_seed(0)
        model.enable_slicing()
        output_with_slicing = model(**inputs_dict, generator=torch.manual_seed(0))[0]

        assert (
            output_without_slicing.detach().cpu().numpy() - output_with_slicing.detach().cpu().numpy()
        ).max() < 0.5, "VAE slicing should not affect the inference results"

        torch.manual_seed(0)
        model.disable_slicing()
        output_without_slicing_2 = model(**inputs_dict, generator=torch.manual_seed(0))[0]

        assert (
            output_without_slicing.detach().cpu().numpy().all()
            == output_without_slicing_2.detach().cpu().numpy().all()
        ), "Without slicing outputs should match when slicing is manually disabled."

    def test_forward_with_norm_groups(self):
        """VidTok uses layernorm instead of groupnorm."""
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(**inputs_dict)
            if isinstance(output, dict):
                output = output.to_tuple()[0]

        assert output is not None
        expected_shape = inputs_dict["sample"].shape
        assert output.shape == expected_shape, "Input and output shapes do not match"
