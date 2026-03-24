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

import unittest

from diffusers import AutoencoderKLKVAEVideo

from ...testing_utils import enable_full_determinism, floats_tensor, torch_device
from ..test_modeling_common import ModelTesterMixin
from .testing_utils import AutoencoderTesterMixin


enable_full_determinism()


class AutoencoderKLKVAEVideoTests(ModelTesterMixin, AutoencoderTesterMixin, unittest.TestCase):
    model_class = AutoencoderKLKVAEVideo
    main_input_name = "sample"
    base_precision = 1e-2

    def get_autoencoder_kl_kvae_video_config(self):
        return {
            "ch": 32,
            "ch_mult": (1, 2),
            "num_res_blocks": 1,
            "in_channels": 3,
            "out_ch": 3,
            "z_channels": 4,
            "temporal_compress_times": 2,
        }

    @property
    def dummy_input(self):
        batch_size = 2
        num_frames = 3  # satisfies (T-1) % temporal_compress_times == 0 with temporal_compress_times=2
        num_channels = 3
        sizes = (16, 16)

        video = floats_tensor((batch_size, num_channels, num_frames) + sizes).to(torch_device)

        return {"sample": video}

    @property
    def input_shape(self):
        return (3, 3, 16, 16)

    @property
    def output_shape(self):
        return (3, 3, 16, 16)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = self.get_autoencoder_kl_kvae_video_config()
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {
            "KVAECachedEncoder3D",
            "KVAECachedDecoder3D",
        }
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)

    @unittest.skip("Unsupported test.")
    def test_outputs_equivalence(self):
        pass

    @unittest.skip(
        "Multi-GPU inference is not supported due to the stateful cache_dict passing through the forward pass."
    )
    def test_model_parallelism(self):
        pass

    @unittest.skip(
        "Multi-GPU inference is not supported due to the stateful cache_dict passing through the forward pass."
    )
    def test_sharded_checkpoints_device_map(self):
        pass

    def _run_nondeterministic(self, fn):
        # reflection_pad3d_backward_out_cuda has no deterministic CUDA implementation;
        # temporarily relax the requirement for training tests that do backward passes.
        import torch

        torch.use_deterministic_algorithms(False)
        try:
            fn()
        finally:
            torch.use_deterministic_algorithms(True)

    def test_training(self):
        self._run_nondeterministic(super().test_training)

    def test_ema_training(self):
        self._run_nondeterministic(super().test_ema_training)

    @unittest.skip(
        "Gradient checkpointing recomputes the forward pass, but the model uses a stateful cache_dict "
        "that is mutated during the first forward. On recomputation the cache is already populated, "
        "causing a different execution path and numerically different gradients. "
        "GC still reduces peak memory usage; gradient correctness in the presence of GC is a known limitation."
    )
    def test_effective_gradient_checkpointing(self):
        pass

    def test_layerwise_casting_training(self):
        self._run_nondeterministic(super().test_layerwise_casting_training)
