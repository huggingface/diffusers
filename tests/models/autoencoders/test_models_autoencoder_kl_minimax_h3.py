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

from diffusers import AutoencoderKLMiniMaxH3
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)
from .testing_utils import NewAutoencoderTesterMixin


enable_full_determinism()


def _run_nondeterministic(fn):
    # The encoder's reflect spatial padding has no deterministic CUDA backward
    # (reflection_pad3d_backward_out_cuda); temporarily relax the requirement for tests that do backward passes.
    torch.use_deterministic_algorithms(False)
    try:
        fn()
    finally:
        torch.use_deterministic_algorithms(True)


# The temporal geometry is fixed by the released `clip_length` / `token_drop` pair: `17 * n + 5` pixel frames map to
# `5 * n + 2` latent frames, so 22 frames is the shortest video this autoencoder can round-trip.
NUM_FRAMES = 22
HEIGHT = WIDTH = 8


class AutoencoderKLMiniMaxH3TesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return AutoencoderKLMiniMaxH3

    @property
    def main_input_name(self) -> str:
        return "sample"

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (3, NUM_FRAMES, HEIGHT, WIDTH)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (3, NUM_FRAMES, HEIGHT, WIDTH)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        # `clip_length` and `token_drop` are left at their released values, so the temporal chunking the decoder
        # re-derives from them (`frame_pre_padding`, `token_overlap`, `frame_overlap`) is the real one; so is
        # `decoder_rope_dim_ratio`, which keeps the partial-rotary path of the ViT decoder in play.
        return {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "block_out_channels": (8, 16),
            "layers_per_block": 1,
            "spatial_downsample_factors": (2, 2),
            "temporal_downsample_factors": (2, 2),
            "norm_num_groups": 8,
            "decoder_num_layers": 2,
            "decoder_num_attention_heads": 2,
            "decoder_attention_head_dim": 8,
            "decoder_num_register_tokens": 2,
            "decoder_ffn_mult": 2,
            "clip_length": 17,
            "token_drop": 3,
            "latents_mean": (0.0,) * 4,
            "latents_std": (1.0,) * 4,
        }

    def get_dummy_inputs(self) -> dict:
        return {
            "sample": randn_tensor((2, 3, NUM_FRAMES, HEIGHT, WIDTH), generator=self.generator, device=torch_device),
        }


class TestAutoencoderKLMiniMaxH3(AutoencoderKLMiniMaxH3TesterConfig, ModelTesterMixin):
    """Core model tests for the MiniMax-H3 video autoencoder."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_from_save_pretrained_dtype(self, tmp_path, dtype):
        # `_keep_in_fp32_modules` pins every module: the released checkpoint is float32 and decoding through
        # downcast weights degrades (the bfloat16 audio VAE decodes roughly 20 dB too quiet), so a requested
        # `torch_dtype` cast at load time must be refused and the weights must stay float32.
        model = self.model_class(**self.get_init_dict())
        model.save_pretrained(tmp_path)
        new_model = self.model_class.from_pretrained(tmp_path, torch_dtype=dtype)
        assert new_model.dtype == torch.float32

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_from_pretrained_dtype_alias(self, tmp_path, dtype):
        # Same contract through the `dtype` alias.
        model = self.model_class(**self.get_init_dict())
        model.save_pretrained(tmp_path)
        new_model = self.model_class.from_pretrained(tmp_path, dtype=dtype)
        assert new_model.dtype == torch.float32

    def test_encode_decode_temporal_geometry(self):
        r"""
        `17 * n + 5` pixel frames encode to `5 * n + 2` latent frames and decode back to the original length.

        The encoder pads up to a whole number of `clip_length` chunks and drops `token_drop` latent frames per
        encode, and the decoder cross-fades the resulting overlap back together, so the round trip only recovers the
        requested frame count if both halves agree on that geometry.
        """
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        sample = self.get_dummy_inputs()["sample"]

        with torch.no_grad():
            latents = model.encode(sample, return_dict=False)[0].mode()
            decoded = model.decode(latents, return_dict=False)[0]

        assert latents.shape == (2, 4, 7, HEIGHT // 4, WIDTH // 4)
        assert decoded.shape == sample.shape


class TestAutoencoderKLMiniMaxH3Memory(AutoencoderKLMiniMaxH3TesterConfig, MemoryTesterMixin):
    """Memory optimization tests for the MiniMax-H3 video autoencoder."""

    @pytest.mark.skip(
        "`_keep_in_fp32_modules` pins every module of this autoencoder, so layerwise casting has nothing to cast and "
        "the memory footprint does not change."
    )
    def test_layerwise_casting_memory(self):
        pass

    def test_layerwise_casting_training(self):
        _run_nondeterministic(super().test_layerwise_casting_training)


class TestAutoencoderKLMiniMaxH3Training(AutoencoderKLMiniMaxH3TesterConfig, TrainingTesterMixin):
    """Training tests for the MiniMax-H3 video autoencoder."""

    def test_gradient_checkpointing_is_applied(self):
        super().test_gradient_checkpointing_is_applied(
            expected_set={"MiniMaxH3VideoDownBlock3d", "MiniMaxH3VideoViTDecoder3d"}
        )

    def test_training(self):
        _run_nondeterministic(super().test_training)

    def test_training_with_ema(self):
        _run_nondeterministic(super().test_training_with_ema)

    def test_mixed_precision_training(self):
        _run_nondeterministic(super().test_mixed_precision_training)

    def test_gradient_checkpointing_equivalence(self):
        _run_nondeterministic(super().test_gradient_checkpointing_equivalence)


class TestAutoencoderKLMiniMaxH3Attention(AutoencoderKLMiniMaxH3TesterConfig, AttentionTesterMixin):
    """Attention processor tests for the MiniMax-H3 video autoencoder."""


class TestAutoencoderKLMiniMaxH3TorchCompile(AutoencoderKLMiniMaxH3TesterConfig, TorchCompileTesterMixin):
    """Torch compile tests for the MiniMax-H3 video autoencoder."""


class TestAutoencoderKLMiniMaxH3SlicingTiling(AutoencoderKLMiniMaxH3TesterConfig, NewAutoencoderTesterMixin):
    """Slicing and tiling tests for the MiniMax-H3 video autoencoder."""
