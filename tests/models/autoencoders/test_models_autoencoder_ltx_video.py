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

import pytest
import torch

from diffusers import AutoencoderKLLTXVideo
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import BaseModelTesterConfig, MemoryTesterMixin, ModelTesterMixin, TrainingTesterMixin
from .testing_utils import NewAutoencoderTesterMixin


enable_full_determinism()


_LTX_VIDEO_GRADIENT_CKPT_EXPECTED = {
    "LTXVideoEncoder3d",
    "LTXVideoDecoder3d",
    "LTXVideoDownBlock3D",
    "LTXVideoMidBlock3d",
    "LTXVideoUpBlock3d",
}


class AutoencoderKLLTXVideo090TesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return AutoencoderKLLTXVideo

    @property
    def main_input_name(self) -> str:
        return "sample"

    @property
    def output_shape(self) -> tuple:
        return (3, 9, 16, 16)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 8,
            "block_out_channels": (8, 8, 8, 8),
            "decoder_block_out_channels": (8, 8, 8, 8),
            "layers_per_block": (1, 1, 1, 1, 1),
            "decoder_layers_per_block": (1, 1, 1, 1, 1),
            "spatio_temporal_scaling": (True, True, False, False),
            "decoder_spatio_temporal_scaling": (True, True, False, False),
            "decoder_inject_noise": (False, False, False, False, False),
            "upsample_residual": (False, False, False, False),
            "upsample_factor": (1, 1, 1, 1),
            "timestep_conditioning": False,
            "patch_size": 1,
            "patch_size_t": 1,
            "encoder_causal": True,
            "decoder_causal": False,
        }

    def get_dummy_inputs(self) -> dict:
        batch_size = 2
        num_frames = 9
        num_channels = 3
        sizes = (16, 16)
        image = randn_tensor(
            (batch_size, num_channels, num_frames, *sizes), generator=self.generator, device=torch_device
        )
        return {"sample": image}


class TestAutoencoderKLLTXVideo090(AutoencoderKLLTXVideo090TesterConfig, ModelTesterMixin):
    base_precision = 1e-2

    @pytest.mark.skip("Unsupported test.")
    def test_outputs_equivalence(self):
        super().test_outputs_equivalence()


class TestAutoencoderKLLTXVideo090Training(AutoencoderKLLTXVideo090TesterConfig, TrainingTesterMixin):
    """Training tests for AutoencoderKLLTXVideo (0.9.0 config)."""

    def test_gradient_checkpointing_is_applied(self):
        super().test_gradient_checkpointing_is_applied(expected_set=_LTX_VIDEO_GRADIENT_CKPT_EXPECTED)


class TestAutoencoderKLLTXVideo090Memory(AutoencoderKLLTXVideo090TesterConfig, MemoryTesterMixin):
    """Memory optimization tests for AutoencoderKLLTXVideo (0.9.0 config)."""


class TestAutoencoderKLLTXVideo090SlicingTiling(AutoencoderKLLTXVideo090TesterConfig, NewAutoencoderTesterMixin):
    """Slicing and tiling tests for AutoencoderKLLTXVideo (0.9.0 config)."""

    @pytest.mark.skip("AutoencoderKLLTXVideo does not support `norm_num_groups` because it does not use GroupNorm.")
    def test_forward_with_norm_groups(self):
        super().test_forward_with_norm_groups()


class AutoencoderKLLTXVideo091TesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return AutoencoderKLLTXVideo

    @property
    def main_input_name(self) -> str:
        return "sample"

    @property
    def output_shape(self) -> tuple:
        return (3, 9, 16, 16)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 8,
            "block_out_channels": (8, 8, 8, 8),
            "decoder_block_out_channels": (16, 32, 64),
            "layers_per_block": (1, 1, 1, 1),
            "decoder_layers_per_block": (1, 1, 1, 1),
            "spatio_temporal_scaling": (True, True, True, False),
            "decoder_spatio_temporal_scaling": (True, True, True),
            "decoder_inject_noise": (True, True, True, False),
            "upsample_residual": (True, True, True),
            "upsample_factor": (2, 2, 2),
            "timestep_conditioning": True,
            "patch_size": 1,
            "patch_size_t": 1,
            "encoder_causal": True,
            "decoder_causal": False,
        }

    def get_dummy_inputs(self) -> dict:
        batch_size = 2
        num_frames = 9
        num_channels = 3
        sizes = (16, 16)
        image = randn_tensor(
            (batch_size, num_channels, num_frames, *sizes), generator=self.generator, device=torch_device
        )
        timestep = torch.tensor([0.05] * batch_size, device=torch_device)
        return {"sample": image, "temb": timestep}


class TestAutoencoderKLLTXVideo091(AutoencoderKLLTXVideo091TesterConfig, ModelTesterMixin):
    base_precision = 1e-2

    @pytest.mark.skip("Unsupported test.")
    def test_outputs_equivalence(self):
        super().test_outputs_equivalence()


class TestAutoencoderKLLTXVideo091Training(AutoencoderKLLTXVideo091TesterConfig, TrainingTesterMixin):
    """Training tests for AutoencoderKLLTXVideo (0.9.1 config)."""

    def test_gradient_checkpointing_is_applied(self):
        super().test_gradient_checkpointing_is_applied(expected_set=_LTX_VIDEO_GRADIENT_CKPT_EXPECTED)


class TestAutoencoderKLLTXVideo091Memory(AutoencoderKLLTXVideo091TesterConfig, MemoryTesterMixin):
    """Memory optimization tests for AutoencoderKLLTXVideo (0.9.1 config)."""
