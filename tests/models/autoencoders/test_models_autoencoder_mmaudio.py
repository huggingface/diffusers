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

from diffusers import MMAudioVAE
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, require_accelerator, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
)


enable_full_determinism()


MEL_DTYPE = pytest.mark.xfail(
    reason="MMAudio's STFT and mel-filter multiplication require float32.",
    raises=RuntimeError,
    strict=True,
)
LAYERWISE_CASTING_BACKWARD = pytest.mark.xfail(
    reason="MMAudio's gain multiplication saves weights that are cast to float8 before backward.",
    raises=RuntimeError,
    strict=True,
)
NORMALIZATION_BUFFER_OFFLOAD = pytest.mark.xfail(
    reason="MMAudio encode/decode bypass the offloading hooks for normalization buffers.",
    raises=RuntimeError,
    strict=True,
)
INCOMPLETE_DEVICE_MAP = pytest.mark.xfail(
    reason="Automatic device maps omit MMAudio's normalization buffers.",
    raises=ValueError,
    strict=True,
)


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
    @MEL_DTYPE
    @require_accelerator
    @pytest.mark.skipif(
        torch_device not in ["cuda", "xpu"],
        reason="float16 and bfloat16 can only be use for inference with an accelerator",
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_from_save_pretrained_dtype_inference(self, tmp_path, dtype):
        super().test_from_save_pretrained_dtype_inference(tmp_path, dtype)

    def test_latent_shape(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        with torch.no_grad():
            latents = model.encode(self.get_dummy_inputs()["sample"]).latent_dist.mode()
        # 256 samples -> 64 mel frames (hop 4) -> 32 latent frames (one 2x downsample)
        assert model.latent_hop_length == 8
        assert latents.shape == (2, 4, 32)


class TestMMAudioVAEMemory(MMAudioVAETesterConfig, MemoryTesterMixin):
    @MEL_DTYPE
    def test_layerwise_casting_memory(self):
        super().test_layerwise_casting_memory()

    @LAYERWISE_CASTING_BACKWARD
    def test_layerwise_casting_training(self):
        super().test_layerwise_casting_training()

    @NORMALIZATION_BUFFER_OFFLOAD
    @pytest.mark.parametrize("record_stream", [False, True])
    def test_group_offloading(self, base_model_output, record_stream):
        super().test_group_offloading(base_model_output, record_stream)

    @pytest.mark.parametrize("record_stream", [False, True])
    @pytest.mark.parametrize(
        "offload_type", ["block_level", pytest.param("leaf_level", marks=NORMALIZATION_BUFFER_OFFLOAD)]
    )
    def test_group_offloading_with_layerwise_casting(self, record_stream, offload_type):
        super().test_group_offloading_with_layerwise_casting(record_stream, offload_type)

    @pytest.mark.parametrize("record_stream", [False, True])
    @pytest.mark.parametrize(
        "offload_type", ["block_level", pytest.param("leaf_level", marks=NORMALIZATION_BUFFER_OFFLOAD)]
    )
    def test_group_offloading_with_disk(self, tmp_path, record_stream, offload_type):
        super().test_group_offloading_with_disk(tmp_path, record_stream, offload_type)

    @INCOMPLETE_DEVICE_MAP
    def test_cpu_offload(self, base_model_output, tmp_path):
        super().test_cpu_offload(base_model_output, tmp_path)

    @INCOMPLETE_DEVICE_MAP
    def test_disk_offload_without_safetensors(self, base_model_output, tmp_path):
        super().test_disk_offload_without_safetensors(base_model_output, tmp_path)

    @INCOMPLETE_DEVICE_MAP
    def test_disk_offload_with_safetensors(self, base_model_output, tmp_path):
        super().test_disk_offload_with_safetensors(base_model_output, tmp_path)


class TestMMAudioVAETorchCompile(MMAudioVAETesterConfig, TorchCompileTesterMixin):
    pass
