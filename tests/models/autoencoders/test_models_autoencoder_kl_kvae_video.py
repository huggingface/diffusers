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

from diffusers import AutoencoderKLKVAEVideo
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import BaseModelTesterConfig, MemoryTesterMixin, ModelTesterMixin, TrainingTesterMixin
from .testing_utils import NewAutoencoderTesterMixin


enable_full_determinism()


def _run_nondeterministic(fn):
    # reflection_pad3d_backward_out_cuda has no deterministic CUDA implementation;
    # temporarily relax the requirement for tests that do backward passes.
    torch.use_deterministic_algorithms(False)
    try:
        fn()
    finally:
        torch.use_deterministic_algorithms(True)


class AutoencoderKLKVAEVideoTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return AutoencoderKLKVAEVideo

    @property
    def main_input_name(self) -> str:
        return "sample"

    @property
    def output_shape(self) -> tuple:
        return (3, 3, 16, 16)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "ch": 32,
            "ch_mult": (1, 2),
            "num_res_blocks": 1,
            "in_channels": 3,
            "out_ch": 3,
            "z_channels": 4,
            "temporal_compress_times": 2,
        }

    def get_dummy_inputs(self) -> dict:
        batch_size = 2
        num_frames = 3  # satisfies (T-1) % temporal_compress_times == 0 with temporal_compress_times=2
        num_channels = 3
        sizes = (16, 16)
        video = randn_tensor(
            (batch_size, num_channels, num_frames, *sizes), generator=self.generator, device=torch_device
        )
        return {"sample": video}


class TestAutoencoderKLKVAEVideo(AutoencoderKLKVAEVideoTesterConfig, ModelTesterMixin):
    @pytest.mark.skip(
        "Multi-GPU inference is not supported due to the stateful cache_dict passing through the forward pass."
    )
    def test_model_parallelism(self):
        super().test_model_parallelism()


class TestAutoencoderKLKVAEVideoTraining(AutoencoderKLKVAEVideoTesterConfig, TrainingTesterMixin):
    """Training tests for AutoencoderKLKVAEVideo."""

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"KVAECachedEncoder3D", "KVAECachedDecoder3D"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)

    def test_training(self):
        _run_nondeterministic(super().test_training)

    def test_training_with_ema(self):
        _run_nondeterministic(super().test_training_with_ema)

    @pytest.mark.skip(
        "Gradient checkpointing recomputes the forward pass, but the model uses a stateful cache_dict "
        "that is mutated during the first forward. On recomputation the cache is already populated, "
        "causing a different execution path and numerically different gradients."
    )
    def test_gradient_checkpointing_equivalence(self):
        super().test_gradient_checkpointing_equivalence()

    def test_layerwise_casting_training(self):
        _run_nondeterministic(super().test_layerwise_casting_training)


class TestAutoencoderKLKVAEVideoMemory(AutoencoderKLKVAEVideoTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for AutoencoderKLKVAEVideo."""


class TestAutoencoderKLKVAEVideoSlicingTiling(AutoencoderKLKVAEVideoTesterConfig, NewAutoencoderTesterMixin):
    """Slicing and tiling tests for AutoencoderKLKVAEVideo."""
