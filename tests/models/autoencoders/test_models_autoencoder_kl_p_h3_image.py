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

from diffusers import AutoencoderKLPH3Image
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TrainingTesterMixin,
    run_nondeterministic,
)
from .testing_utils import AutoencoderTesterMixin


enable_full_determinism()


# Images are single-frame videos; at 16x compression a 32 x 32 image is 2 x 2 latents.
HEIGHT = WIDTH = 32


class AutoencoderKLPH3ImageTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return AutoencoderKLPH3Image

    @property
    def main_input_name(self) -> str:
        return "sample"

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (3, 1, HEIGHT, WIDTH)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (3, 1, HEIGHT, WIDTH)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "block_out_channels": (8, 16),
            "layers_per_block": 1,
            "spatial_downsample_factors": (4, 4),
            "temporal_downsample_factors": (2, 2),
            "norm_num_groups": 8,
            "decoder_num_layers": 2,
            "decoder_num_attention_heads": 2,
            "decoder_attention_head_dim": 8,
            "decoder_num_register_tokens": 2,
            "decoder_ffn_mult": 2,
            "latents_mean": (0.0,) * 4,
            "latents_std": (1.0,) * 4,
        }

    def get_dummy_inputs(self) -> dict:
        return {
            "sample": randn_tensor((2, 3, 1, HEIGHT, WIDTH), generator=self.generator, device=torch_device),
        }


class TestAutoencoderKLPH3Image(AutoencoderKLPH3ImageTesterConfig, ModelTesterMixin):
    """Core model tests for the P-H3-Image image autoencoder."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_from_save_pretrained_dtype(self, tmp_path, dtype):
        model = self.model_class(**self.get_init_dict())
        model.save_pretrained(tmp_path)
        new_model = self.model_class.from_pretrained(tmp_path, dtype=dtype)
        assert new_model.decoder.transformer_blocks[0].attn.to_q.weight.dtype == dtype
        assert new_model.encoder.conv_in.weight.dtype == torch.float32

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_decode_in_low_precision(self, tmp_path, dtype):
        # The output projection is float32, so the decoded pixels are too.
        self.model_class(**self.get_init_dict()).save_pretrained(tmp_path)
        model = self.model_class.from_pretrained(tmp_path, dtype=dtype).eval()
        with torch.no_grad():
            decoded = model.decode(torch.randn(1, 4, 1, HEIGHT // 16, WIDTH // 16), return_dict=False)[0]
        assert decoded.dtype == torch.float32
        assert decoded.shape == (1, 3, 1, HEIGHT, WIDTH)

    @pytest.mark.skip(
        "`forward` runs through the `apply_forward_hook`-decorated `encode` and `decode`, and that decorator's "
        "`pre_forward` call clears the input device accelerate's `AlignDevicesHook` recorded for the caller, so the "
        "output comes back on the last device of the split rather than the input device and the comparison raises. "
        "`test_cpu_offload` covers split placement instead; there every submodule executes on the same device."
    )
    def test_model_parallelism(self, base_model_output, tmp_path, atol=1e-5, rtol=0):
        pass

    def test_encode_decode_single_frame(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        sample = self.get_dummy_inputs()["sample"]

        with torch.no_grad():
            latents = model.encode(sample, return_dict=False)[0].mode()
            decoded = model.decode(latents, return_dict=False)[0]

        assert latents.shape == (2, 4, 1, HEIGHT // 16, WIDTH // 16)
        assert decoded.shape == sample.shape

    def test_rejects_videos(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        with pytest.raises(ValueError, match="single frames"):
            model.encode(torch.randn(1, 3, 2, HEIGHT, WIDTH, device=torch_device))
        with pytest.raises(ValueError, match="single latent frames"):
            model.decode(torch.randn(1, 4, 2, HEIGHT // 16, WIDTH // 16, device=torch_device))


class TestAutoencoderKLPH3ImageMemory(AutoencoderKLPH3ImageTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for the P-H3-Image image autoencoder."""

    # The encoder's reflect spatial padding has no deterministic CUDA backward
    # (reflection_pad3d_backward_out_cuda), so every test with a backward pass runs with determinism relaxed.
    def test_layerwise_casting_training(self):
        run_nondeterministic(super().test_layerwise_casting_training)


class TestAutoencoderKLPH3ImageTraining(AutoencoderKLPH3ImageTesterConfig, TrainingTesterMixin):
    """Training tests for the P-H3-Image image autoencoder."""

    def test_gradient_checkpointing_is_applied(self):
        super().test_gradient_checkpointing_is_applied(expected_set={"PH3ImageDownBlock3d", "PH3ImageViTDecoder"})

    # The encoder's reflect spatial padding has no deterministic CUDA backward
    # (reflection_pad3d_backward_out_cuda), so every test with a backward pass runs with determinism relaxed.
    def test_training(self):
        run_nondeterministic(super().test_training)

    def test_training_with_ema(self):
        run_nondeterministic(super().test_training_with_ema)

    def test_mixed_precision_training(self):
        run_nondeterministic(super().test_mixed_precision_training)

    def test_gradient_checkpointing_equivalence(self):
        run_nondeterministic(super().test_gradient_checkpointing_equivalence)


class TestAutoencoderKLPH3ImageAttention(AutoencoderKLPH3ImageTesterConfig, AttentionTesterMixin):
    """Attention processor tests for the P-H3-Image image autoencoder."""


class TestAutoencoderKLPH3ImageSlicingTiling(AutoencoderKLPH3ImageTesterConfig, AutoencoderTesterMixin):
    """Slicing and tiling tests for the P-H3-Image image autoencoder."""
