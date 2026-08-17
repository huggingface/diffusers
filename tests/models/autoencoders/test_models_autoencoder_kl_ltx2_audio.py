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

from diffusers import AutoencoderKLLTX2Audio
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import is_flaky, torch_device
from ..testing_utils import BaseModelTesterConfig, MemoryTesterMixin, ModelTesterMixin, TrainingTesterMixin
from .testing_utils import NewAutoencoderTesterMixin


class AutoencoderKLLTX2AudioTesterConfig(BaseModelTesterConfig):
    @property
    def main_input_name(self):
        return "sample"

    @property
    def model_class(self):
        return AutoencoderKLLTX2Audio

    @property
    def output_shape(self):
        return (2, 5, 16)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self):
        return {
            "in_channels": 2,  # stereo,
            "output_channels": 2,
            "latent_channels": 4,
            "base_channels": 16,
            "ch_mult": (1, 2, 4),
            "resolution": 16,
            "attn_resolutions": None,
            "num_res_blocks": 2,
            "norm_type": "pixel",
            "causality_axis": "height",
            "mid_block_add_attention": False,
            "sample_rate": 16000,
            "mel_hop_length": 160,
            "mel_bins": 16,
            "is_causal": True,
            "double_z": True,
        }

    def get_dummy_inputs(self, num_frames: int = 8):
        batch_size = 2
        num_channels = 2
        num_mel_bins = 16
        spectrogram = randn_tensor(
            (batch_size, num_channels, num_frames, num_mel_bins),
            generator=self.generator,
            device=torch_device,
        )
        return {"sample": spectrogram}


class TestAutoencoderKLLTX2Audio(AutoencoderKLLTX2AudioTesterConfig, ModelTesterMixin):
    base_precision = 1e-2

    def test_outputs_equivalence(self):
        pytest.skip("Unsupported test.")


class TestAutoencoderKLLTX2AudioTraining(AutoencoderKLLTX2AudioTesterConfig, TrainingTesterMixin):
    """Training tests for AutoencoderKLLTX2Audio."""


class TestAutoencoderKLLTX2AudioMemory(AutoencoderKLLTX2AudioTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for AutoencoderKLLTX2Audio."""

    # The memory tests run on a longer spectrogram than the rest of the file. `test_layerwise_casting_memory` asserts
    # that bfloat16 compute peaks lower than float32 compute, which only holds once the activations outweigh the
    # scratch space the convolutions ask for: at 8 frames the forward pass allocates ~0.2 MB of activations, while the
    # first bfloat16 convolution takes a cuDNN algorithm needing a 0.6 MB workspace where float32 takes one needing
    # 16 KB. 512 frames put ~4 MB of activations between the two dtypes and keep the peak under 15 MB. The autoencoder
    # trims 3 frames per round trip, hence the output length below.
    def get_dummy_inputs(self, num_frames: int = 512):
        return super().get_dummy_inputs(num_frames=num_frames)

    @property
    def output_shape(self):
        return (2, 509, 16)

    @is_flaky()
    @pytest.mark.parametrize("record_stream", [False, True])
    @pytest.mark.parametrize("offload_type", ["block_level", "leaf_level"])
    def test_group_offloading_with_disk(self, tmp_path, record_stream, offload_type, atol=1e-5, rtol=0):
        super().test_group_offloading_with_disk(tmp_path, record_stream, offload_type, atol=atol, rtol=rtol)


class TestAutoencoderKLLTX2AudioSlicingTiling(AutoencoderKLLTX2AudioTesterConfig, NewAutoencoderTesterMixin):
    """Slicing and tiling tests for AutoencoderKLLTX2Audio."""
