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

from diffusers import MMAudioVAE
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
)


enable_full_determinism()


class MMAudioVAETesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return MMAudioVAE

    @property
    def pretrained_model_name_or_path(self):
        return "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers"

    @property
    def pretrained_model_kwargs(self):
        return {"subfolder": "audio_vae"}

    @property
    def main_input_name(self) -> str:
        return "sample"

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "mel_bins": 8,
            "latent_channels": 4,
            "hidden_channels": 8,
            "channel_multipliers": (1, 2),
            "layers_per_block": 1,
            "sample_rate": 64,
            "n_fft": 16,
            "hop_length": 4,
        }

    def get_dummy_inputs(self) -> dict:
        waveform = randn_tensor((2, 256), generator=self.generator, device=torch_device).clamp(-1, 1)
        return {"sample": waveform}

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (256,)

    @property
    def output_shape(self) -> tuple[int, ...]:
        # `decode`/`forward` return a mel spectrogram (`mel_bins`, num_mel_frames); the waveform is produced by the
        # separate `MMAudioVocoder`.
        return (8, 64)


class TestMMAudioVAEModel(MMAudioVAETesterConfig, ModelTesterMixin):
    def test_latent_shape(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        with torch.no_grad():
            latents = model.encode(self.get_dummy_inputs()["sample"]).latent_dist.mode()
        # 256 samples -> 64 mel frames (hop 4) -> 32 latent frames (one 2x downsample)
        assert model.latent_hop_length == 8
        assert latents.shape == (2, 4, 32)


class TestMMAudioVAEMemory(MMAudioVAETesterConfig, MemoryTesterMixin):
    pass


class TestMMAudioVAETorchCompile(MMAudioVAETesterConfig, TorchCompileTesterMixin):
    pass
