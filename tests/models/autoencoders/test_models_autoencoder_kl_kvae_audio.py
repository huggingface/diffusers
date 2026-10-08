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

from diffusers import AutoencoderKLKVAEAudio
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import BaseModelTesterConfig, MemoryTesterMixin, ModelTesterMixin, TorchCompileTesterMixin
from .testing_utils import NewAutoencoderTesterMixin


enable_full_determinism()


class AutoencoderKLKVAEAudioTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return AutoencoderKLKVAEAudio

    @property
    def main_input_name(self) -> str:
        return "sample"

    @property
    def output_shape(self) -> tuple:
        return (1, 37)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "encoder_dim": 4,
            # Includes an odd stride (3) so the odd-stride `output_padding` fix in
            # `KVAEAudioDecoderBlock` (needed for the decoder to invert the encoder's downsampling
            # exactly) is actually exercised by the test suite.
            "encoder_rates": [2, 3],
            "codebook_dim": 4,
            "decoder_dim": 16,
            "decoder_rates": [3, 2],
            "sample_rate": 48000,
            "num_channels": 1,
        }

    def get_dummy_inputs(self) -> dict:
        batch_size = 2
        num_channels = 1
        seq_len = 37
        waveform = randn_tensor((batch_size, num_channels, seq_len), generator=self.generator, device=torch_device)
        return {"sample": waveform, "sample_posterior": False}


class TestAutoencoderKLKVAEAudio(AutoencoderKLKVAEAudioTesterConfig, ModelTesterMixin):
    def test_forward_output_length_matches_input_length(self):
        # Regression test: the decoder must invert the encoder's downsampling exactly, for any
        # input length, including ones not a multiple of `hop_length`. Without `output_padding` on
        # the odd-stride `ConvTranspose1d`s, decode() undershoots the (hop-length-padded) input
        # length by a fixed amount per odd stride, so forward()'s trailing `[..., :length]` slice
        # becomes a no-op and silently returns short audio.
        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict).to(torch_device).eval()

        for seq_len in (1, 5, 6, 7, 37, 100):
            waveform = randn_tensor((1, 1, seq_len), generator=self.generator, device=torch_device)
            with torch.no_grad():
                output = model(waveform, sample_posterior=False)
            assert output.sample.shape[-1] == seq_len, (
                f"expected output length {seq_len}, got {output.sample.shape[-1]}"
            )


class TestAutoencoderKLKVAEAudioMemory(AutoencoderKLKVAEAudioTesterConfig, MemoryTesterMixin):
    pass


class TestAutoencoderKLKVAEAudioTorchCompile(AutoencoderKLKVAEAudioTesterConfig, TorchCompileTesterMixin):
    pass


class TestAutoencoderKLKVAEAudioSlicing(AutoencoderKLKVAEAudioTesterConfig, NewAutoencoderTesterMixin):
    pass


class AutoencoderKLKVAEAudioAttnTesterConfig(AutoencoderKLKVAEAudioTesterConfig):
    def get_init_dict(self) -> dict:
        init_dict = super().get_init_dict()
        init_dict.update({"use_attn": True, "attn_num_heads": 2})
        return init_dict


class TestAutoencoderKLKVAEAudioAttn(AutoencoderKLKVAEAudioAttnTesterConfig, ModelTesterMixin):
    pass
