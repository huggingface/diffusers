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

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffusers import AutoencoderKLMiniMaxH3
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


# The temporal geometry is fixed by the released `clip_length` / `token_drop` pair: `17 * n + 5` pixel frames map to
# `5 * n + 2` latent frames, so 22 frames is the shortest video this autoencoder can round-trip.
NUM_FRAMES = 22
HEIGHT = WIDTH = 8


def _parallel_tiled_decode_worker(rank, world_size, init_file, model_config):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size, timeout=timedelta(seconds=60)
    )
    try:
        torch.manual_seed(0)
        model = AutoencoderKLMiniMaxH3(**model_config).eval()
        subgroup = dist.new_group([1, 2])
        tile_settings = {
            "tile_sample_min_height": 8,
            "tile_sample_min_width": 8,
            "tile_sample_min_overlap_height": 4,
            "tile_sample_min_overlap_width": 4,
        }
        calls = []
        handle = model.decoder.register_forward_hook(lambda *_: calls.append(1))
        for group in (dist.group.WORLD, subgroup):
            if dist.get_rank(group) < 0:
                with pytest.raises(ValueError, match="must belong"):
                    model.enable_tiling(**tile_settings, process_group=group)
            else:
                # Multiple tiles, then more ranks than tiles with the released temporal overlap and autocast.
                for latent_shape, autocast in (((2, 4, 7, 3, 4), False), ((1, 4, 12, 1, 2), True)):
                    z = torch.randn(latent_shape, generator=torch.Generator().manual_seed(1))
                    model.enable_tiling(**tile_settings)
                    calls.clear()
                    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                        expected = model.decode(z).sample
                        serial_calls = len(calls)
                        calls.clear()
                        model.enable_tiling(**tile_settings, process_group=group)
                        actual = model.decode(z).sample
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    assert actual.dtype == (torch.bfloat16 if autocast else torch.float32)
                    local_calls = torch.tensor(len(calls))
                    dist.all_reduce(local_calls, group=group)
                    assert local_calls.item() == serial_calls
                with pytest.raises(RuntimeError, match="requires torch.no_grad"):
                    model.decode(z)
            dist.barrier()
        handle.remove()
    finally:
        dist.destroy_process_group()


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
        # The requested dtype reaches the decoder; only `_keep_in_fp32_modules` stays float32.
        model = self.model_class(**self.get_init_dict())
        model.save_pretrained(tmp_path)
        new_model = self.model_class.from_pretrained(tmp_path, torch_dtype=dtype)
        assert new_model.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_from_pretrained_dtype_alias(self, tmp_path, dtype):
        # Same contract through the `dtype` alias.
        model = self.model_class(**self.get_init_dict())
        model.save_pretrained(tmp_path)
        new_model = self.model_class.from_pretrained(tmp_path, dtype=dtype)
        assert new_model.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_decode_in_low_precision(self, tmp_path, dtype):
        # Decode is mixed precision with no autocast, so every dtype seam has to hold on its own.
        self.model_class(**self.get_init_dict()).save_pretrained(tmp_path)
        model = self.model_class.from_pretrained(tmp_path, dtype=dtype).eval()
        with torch.no_grad():
            decoded = model.decode(torch.randn(1, 4, 7, HEIGHT // 4, WIDTH // 4), return_dict=False)[0]
        assert decoded.dtype == dtype

    @pytest.mark.skip(
        "`forward` runs through the `apply_forward_hook`-decorated `encode` and `decode`, and that decorator's "
        "`pre_forward` call clears the input device accelerate's `AlignDevicesHook` recorded for the caller, so the "
        "output comes back on the last device of the split rather than the input device and the comparison raises. "
        "`test_cpu_offload` covers split placement instead — there every submodule executes on the same device."
    )
    def test_model_parallelism(self, base_model_output, tmp_path, atol=1e-5, rtol=0):
        pass

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

    # The encoder's reflect spatial padding has no deterministic CUDA backward
    # (reflection_pad3d_backward_out_cuda), so every test with a backward pass runs with determinism relaxed.
    def test_layerwise_casting_training(self):
        run_nondeterministic(super().test_layerwise_casting_training)


class TestAutoencoderKLMiniMaxH3Training(AutoencoderKLMiniMaxH3TesterConfig, TrainingTesterMixin):
    """Training tests for the MiniMax-H3 video autoencoder."""

    def test_gradient_checkpointing_is_applied(self):
        super().test_gradient_checkpointing_is_applied(
            expected_set={"MiniMaxH3VideoDownBlock3d", "MiniMaxH3VideoViTDecoder3d"}
        )

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


class TestAutoencoderKLMiniMaxH3Attention(AutoencoderKLMiniMaxH3TesterConfig, AttentionTesterMixin):
    """Attention processor tests for the MiniMax-H3 video autoencoder."""


class TestAutoencoderKLMiniMaxH3SlicingTiling(AutoencoderKLMiniMaxH3TesterConfig, AutoencoderTesterMixin):
    """Slicing and tiling tests for the MiniMax-H3 video autoencoder."""

    def test_parallel_tiled_decoding(self, tmp_path):
        """Original tiles and temporal blending match across WORLD and a subgroup without global rank zero."""
        if not dist.is_available() or not dist.is_gloo_available():
            pytest.skip("Parallel tiled decoding requires torch.distributed with Gloo.")
        mp.spawn(
            _parallel_tiled_decode_worker,
            args=(3, str(tmp_path / "gloo_init"), self.get_init_dict()),
            nprocs=3,
            join=True,
        )
