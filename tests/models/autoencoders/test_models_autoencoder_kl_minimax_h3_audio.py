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

from diffusers import AutoencoderKLMiniMaxH3Audio
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
    run_nondeterministic,
)


enable_full_determinism()


# `prod(encoder_rates)` is the hop length, so a waveform of `HOP_LENGTH * NUM_LATENTS` samples encodes to
# `NUM_LATENTS` latents. MiniMax-H3 carries stereo as two batch items of this mono autoencoder.
HOP_LENGTH = 4
NUM_LATENTS = 8
BATCH_SIZE = 2


class AutoencoderKLMiniMaxH3AudioTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return AutoencoderKLMiniMaxH3Audio

    @property
    def main_input_name(self) -> str:
        return "sample"

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (1, HOP_LENGTH * NUM_LATENTS)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (1, HOP_LENGTH * NUM_LATENTS)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        # `latent_dim // num_attention_heads` (16) is kept different from `latent_channels` (8), as it is at full
        # scale, so the adaptive-average-pool branch of the causal-attention projection is exercised.
        return {
            "encoder_dim": 4,
            "encoder_rates": (2, 2),
            "latent_dim": 32,
            "latent_channels": 8,
            "num_attention_heads": 2,
            "decoder_dim": 16,
            "decoder_rates": (2, 2),
            "decoder_kernel_sizes": (4, 4),
            "resblock_kernel_sizes": (3, 7),
            "resblock_dilation_sizes": ((1, 3), (1, 3)),
            "sampling_rate": 32000,
            "latents_mean": [0.0] * 8,
            "latents_std": [1.0] * 8,
        }

    def get_dummy_inputs(self) -> dict:
        return {
            "sample": randn_tensor(
                (BATCH_SIZE, 1, HOP_LENGTH * NUM_LATENTS), generator=self.generator, device=torch_device
            ),
        }


class TestAutoencoderKLMiniMaxH3Audio(AutoencoderKLMiniMaxH3AudioTesterConfig, ModelTesterMixin):
    """Core model tests for the MiniMax-H3 audio autoencoder."""

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

    def test_encode_pads_to_the_hop_length(self):
        r"""
        A waveform that is not a whole number of hops is right-padded rather than rejected, and decoding always
        returns `num_frames * hop_length` samples.
        """
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        unaligned = randn_tensor(
            (BATCH_SIZE, 1, HOP_LENGTH * NUM_LATENTS + 1), generator=self.generator, device=torch_device
        )

        with torch.no_grad():
            latents = model.encode(unaligned, return_dict=False)[0].mode()
            decoded = model.decode(latents, return_dict=False)[0]

        assert latents.shape == (BATCH_SIZE, 8, NUM_LATENTS + 1)
        assert decoded.shape == (BATCH_SIZE, 1, (NUM_LATENTS + 1) * HOP_LENGTH)


class TestAutoencoderKLMiniMaxH3AudioMemory(AutoencoderKLMiniMaxH3AudioTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for the MiniMax-H3 audio autoencoder."""

    @pytest.mark.skip(
        "`_keep_in_fp32_modules` pins every module of this autoencoder, so layerwise casting has nothing to cast and "
        "the memory footprint does not change."
    )
    def test_layerwise_casting_memory(self):
        pass

    # The causal-attention projection's `F.adaptive_avg_pool1d` has no deterministic CUDA backward
    # (adaptive_avg_pool2d_backward_cuda), so every test with a backward pass runs with determinism relaxed.
    def test_layerwise_casting_training(self):
        run_nondeterministic(super().test_layerwise_casting_training)


class TestAutoencoderKLMiniMaxH3AudioTraining(AutoencoderKLMiniMaxH3AudioTesterConfig, TrainingTesterMixin):
    """Training tests for the MiniMax-H3 audio autoencoder."""

    # The causal-attention projection's `F.adaptive_avg_pool1d` has no deterministic CUDA backward
    # (adaptive_avg_pool2d_backward_cuda), so every test with a backward pass runs with determinism relaxed.
    def test_training(self):
        run_nondeterministic(super().test_training)

    def test_training_with_ema(self):
        run_nondeterministic(super().test_training_with_ema)

    def test_mixed_precision_training(self):
        run_nondeterministic(super().test_mixed_precision_training)


class TestAutoencoderKLMiniMaxH3AudioAttention(AutoencoderKLMiniMaxH3AudioTesterConfig, AttentionTesterMixin):
    """Attention processor tests for the MiniMax-H3 audio autoencoder."""

    @pytest.mark.skip(
        "The audio autoencoder holds exactly one attention module (the `pre_block` latent projection), so a dict of "
        "processors can never have a length that mismatches the number of attention layers."
    )
    def test_attention_processor_count_mismatch_raises_error(self):
        pass


class TestAutoencoderKLMiniMaxH3AudioTorchCompile(AutoencoderKLMiniMaxH3AudioTesterConfig, TorchCompileTesterMixin):
    """Torch compile tests for the MiniMax-H3 audio autoencoder."""

    def test_torch_compile_recompilation_and_graph_break(self):
        # Under `torch.use_deterministic_algorithms(True)`, `F.pad(mode="replicate")` (used by the anti-aliased
        # resamplers) routes through an `importlib.import_module` call, which dynamo cannot trace with
        # fullgraph=True on torch <= 2.10.
        run_nondeterministic(super().test_torch_compile_recompilation_and_graph_break)
