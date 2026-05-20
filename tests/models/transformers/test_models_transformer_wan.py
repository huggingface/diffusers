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

from diffusers import WanKVCache, WanTransformer3DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    BitsAndBytesTesterMixin,
    GGUFCompileTesterMixin,
    GGUFTesterMixin,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchAoTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class WanTransformer3DTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return WanTransformer3DModel

    @property
    def pretrained_model_name_or_path(self):
        return "hf-internal-testing/tiny-wan22-transformer"

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (4, 2, 16, 16)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (4, 2, 16, 16)

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | list[int] | tuple | str | bool]:
        return {
            "patch_size": (1, 2, 2),
            "num_attention_heads": 2,
            "attention_head_dim": 12,
            "in_channels": 4,
            "out_channels": 4,
            "text_dim": 16,
            "freq_dim": 256,
            "ffn_dim": 32,
            "num_layers": 2,
            "cross_attn_norm": True,
            "qk_norm": "rms_norm_across_heads",
            "rope_max_seq_len": 32,
        }

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        batch_size = 1
        num_channels = 4
        num_frames = 2
        height = 16
        width = 16
        text_encoder_embedding_dim = 16
        sequence_length = 12

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_channels, num_frames, height, width),
                generator=self.generator,
                device=torch_device,
                dtype=self.torch_dtype,
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, text_encoder_embedding_dim),
                generator=self.generator,
                device=torch_device,
                dtype=self.torch_dtype,
            ),
            "timestep": torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(torch_device),
        }


class TestWanTransformer3D(WanTransformer3DTesterConfig, ModelTesterMixin):
    """Core model tests for Wan Transformer 3D."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_from_save_pretrained_dtype_inference(self, tmp_path, dtype):
        # Skip: fp16/bf16 require very high atol to pass, providing little signal.
        # Dtype preservation is already tested by test_from_save_pretrained_dtype and test_keep_in_fp32_modules.
        pytest.skip("Tolerance requirements too high for meaningful test")


class TestWanTransformer3DMemory(WanTransformer3DTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for Wan Transformer 3D."""


class TestWanTransformer3DTraining(WanTransformer3DTesterConfig, TrainingTesterMixin):
    """Training tests for Wan Transformer 3D."""

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"WanTransformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestWanTransformer3DAttention(WanTransformer3DTesterConfig, AttentionTesterMixin):
    """Attention processor tests for Wan Transformer 3D."""


class TestWanTransformer3DCompile(WanTransformer3DTesterConfig, TorchCompileTesterMixin):
    """Torch compile tests for Wan Transformer 3D."""


class TestWanTransformer3DBitsAndBytes(WanTransformer3DTesterConfig, BitsAndBytesTesterMixin):
    """BitsAndBytes quantization tests for Wan Transformer 3D."""

    @property
    def torch_dtype(self):
        return torch.float16

    def get_dummy_inputs(self):
        """Override to provide inputs matching the tiny Wan model dimensions."""
        return {
            "hidden_states": randn_tensor(
                (1, 36, 2, 64, 64), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "encoder_hidden_states": randn_tensor(
                (1, 512, 4096), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "timestep": torch.tensor([1.0]).to(torch_device, self.torch_dtype),
        }


class TestWanTransformer3DTorchAo(WanTransformer3DTesterConfig, TorchAoTesterMixin):
    """TorchAO quantization tests for Wan Transformer 3D."""

    @property
    def torch_dtype(self):
        return torch.bfloat16

    def get_dummy_inputs(self):
        """Override to provide inputs matching the tiny Wan model dimensions."""
        return {
            "hidden_states": randn_tensor(
                (1, 36, 2, 64, 64), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "encoder_hidden_states": randn_tensor(
                (1, 512, 4096), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "timestep": torch.tensor([1.0]).to(torch_device, self.torch_dtype),
        }


class TestWanTransformer3DGGUF(WanTransformer3DTesterConfig, GGUFTesterMixin):
    """GGUF quantization tests for Wan Transformer 3D."""

    @property
    def gguf_filename(self):
        return "https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/blob/main/LowNoise/Wan2.2-I2V-A14B-LowNoise-Q2_K.gguf"

    @property
    def torch_dtype(self):
        return torch.bfloat16

    def _create_quantized_model(self, config_kwargs=None, **extra_kwargs):
        return super()._create_quantized_model(
            config_kwargs, config="Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="transformer", **extra_kwargs
        )

    def get_dummy_inputs(self):
        """Override to provide inputs matching the real Wan I2V model dimensions.

        Wan 2.2 I2V: in_channels=36, text_dim=4096
        """
        return {
            "hidden_states": randn_tensor(
                (1, 36, 2, 64, 64), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "encoder_hidden_states": randn_tensor(
                (1, 512, 4096), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "timestep": torch.tensor([1.0]).to(torch_device, self.torch_dtype),
        }


class TestWanTransformer3DGGUFCompile(WanTransformer3DTesterConfig, GGUFCompileTesterMixin):
    """GGUF + compile tests for Wan Transformer 3D."""

    @property
    def gguf_filename(self):
        return "https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/blob/main/LowNoise/Wan2.2-I2V-A14B-LowNoise-Q2_K.gguf"

    @property
    def torch_dtype(self):
        return torch.bfloat16

    def _create_quantized_model(self, config_kwargs=None, **extra_kwargs):
        return super()._create_quantized_model(
            config_kwargs, config="Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="transformer", **extra_kwargs
        )

    def get_dummy_inputs(self):
        """Override to provide inputs matching the real Wan I2V model dimensions.

        Wan 2.2 I2V: in_channels=36, text_dim=4096
        """
        return {
            "hidden_states": randn_tensor(
                (1, 36, 2, 64, 64), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "encoder_hidden_states": randn_tensor(
                (1, 512, 4096), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "timestep": torch.tensor([1.0]).to(torch_device, self.torch_dtype),
        }


class TestWanKVCache:
    NUM_BLOCKS = 2
    TOKENS_PER_CHUNK = 4
    _CONFIG = {
        "patch_size": [1, 2, 2],
        "num_attention_heads": 2,
        "attention_head_dim": 16,
        "in_channels": 16,
        "out_channels": 16,
        "text_dim": 32,
        "freq_dim": 32,
        "ffn_dim": 64,
        "num_layers": NUM_BLOCKS,
        "cross_attn_norm": False,
        "qk_norm": "rms_norm_across_heads",
        "eps": 1e-6,
        "rope_max_seq_len": 32,
    }

    def setup_method(self):
        self.transformer = WanTransformer3DModel.from_config(self._CONFIG).eval()

    def _make_chunk(self, seed):
        # centered around zero so RoPE has a measurable effect (all-positive inputs → uniform attn)
        n_lat, n_enc = 16 * 4 * 4, 10 * 32
        lat = (torch.arange(n_lat, dtype=torch.float32) - n_lat // 2 + seed * 7).reshape(1, 16, 1, 4, 4) / 50
        enc = (torch.arange(n_enc, dtype=torch.float32) - n_enc // 2 + seed * 7).reshape(1, 10, 32) / 50
        return lat, torch.zeros(1, dtype=torch.long), enc

    def _denoise_chunk(self, latents, timestep, encoder_hidden_states, *, cache, frame_offset=0):
        with torch.no_grad():
            return self.transformer(
                latents,
                timestep,
                encoder_hidden_states,
                frame_offset=frame_offset,
                return_dict=False,
                attention_kwargs={"kv_cache": cache},
            )[0]

    def _cached_len(self, cache, block=0):
        k = cache.block_caches[block].cached_key
        return 0 if k is None else k.shape[1]

    def _cached_keys(self, cache, block=0):
        return cache.block_caches[block].cached_key

    def _assert_equal(self, a, b):
        assert torch.equal(a, b)

    def _assert_not_equal(self, a, b):
        assert not torch.equal(a, b)

    def test_append_unbounded(self):
        """mode=append, window_size=-1: cache grows by TOKENS_PER_CHUNK each call; existing prefix is never disturbed."""
        T = self.TOKENS_PER_CHUNK
        cache = WanKVCache(num_blocks=self.NUM_BLOCKS, window_size=-1)
        cache.enable_append_mode()

        self._denoise_chunk(*self._make_chunk(1), cache=cache)
        assert self._cached_len(cache) == T
        snap1 = self._cached_keys(cache).clone()

        self._denoise_chunk(*self._make_chunk(2), cache=cache)
        assert self._cached_len(cache) == T * 2
        snap2 = self._cached_keys(cache).clone()
        self._assert_equal(snap1[:, :T], snap2[:, :T])

        self._denoise_chunk(*self._make_chunk(3), cache=cache)
        assert self._cached_len(cache) == T * 3
        snap3 = self._cached_keys(cache).clone()
        self._assert_equal(snap2[:, : 2 * T], snap3[:, : 2 * T])

        self._denoise_chunk(*self._make_chunk(4), cache=cache)
        assert self._cached_len(cache) == T * 4
        snap4 = self._cached_keys(cache).clone()
        self._assert_equal(snap3[:, : 3 * T], snap4[:, : 3 * T])

    def test_append_windowed_single_chunk(self):
        """mode=append, window_size=T: each new chunk fully evicts the previous; cache stays at T tokens."""
        T = self.TOKENS_PER_CHUNK
        cache = WanKVCache(num_blocks=self.NUM_BLOCKS, window_size=T)
        cache.enable_append_mode()

        self._denoise_chunk(*self._make_chunk(1), cache=cache)
        assert self._cached_len(cache) == T
        snap1 = self._cached_keys(cache).clone()

        self._denoise_chunk(*self._make_chunk(2), cache=cache)
        assert self._cached_len(cache) == T  # eviction kept size at T
        snap2 = self._cached_keys(cache).clone()
        self._assert_not_equal(snap1[:, :T], snap2[:, :T])  # chunk 1 fully evicted

        self._denoise_chunk(*self._make_chunk(3), cache=cache)
        assert self._cached_len(cache) == T  # eviction kept size at T
        snap3 = self._cached_keys(cache).clone()
        self._assert_not_equal(snap2[:, :T], snap3[:, :T])  # chunk 2 fully evicted

    def test_append_windowed_three_chunks(self):
        """mode=append, window_size=3*T: cache fills to 3 chunks then rolls — surviving chunks shift left by T per step."""
        T = self.TOKENS_PER_CHUNK
        cache = WanKVCache(num_blocks=self.NUM_BLOCKS, window_size=3 * T)
        cache.enable_append_mode()

        # Fill the window with chunks 1, 2, 3 (cache grows; no eviction yet)
        self._denoise_chunk(*self._make_chunk(1), cache=cache)
        assert self._cached_len(cache) == T
        self._denoise_chunk(*self._make_chunk(2), cache=cache)
        assert self._cached_len(cache) == 2 * T
        self._denoise_chunk(*self._make_chunk(3), cache=cache)
        assert self._cached_len(cache) == 3 * T
        snap_full = self._cached_keys(cache).clone()  # [chunk1, chunk2, chunk3]

        # Chunk 4: window full → chunk 1 evicted; chunks 2-3 shift left by T (size stays 3T)
        self._denoise_chunk(*self._make_chunk(4), cache=cache)
        assert self._cached_len(cache) == 3 * T
        snap_after_4 = self._cached_keys(cache).clone()  # [chunk2, chunk3, chunk4]
        # chunks 2-3 (snap_full[T:3T]) now sit at positions [0:2T] in the new cache
        self._assert_equal(snap_after_4[:, : 2 * T], snap_full[:, T : 3 * T])
        self._assert_not_equal(
            snap_full[:, 2 * T : 3 * T], snap_after_4[:, 2 * T : 3 * T]
        )  # last slot is chunk 4 (≠ chunk 3)

        # Chunk 5: chunk 2 evicted; chunks 3-4 shift left
        self._denoise_chunk(*self._make_chunk(5), cache=cache)
        assert self._cached_len(cache) == 3 * T
        snap_after_5 = self._cached_keys(cache).clone()  # [chunk3, chunk4, chunk5]
        self._assert_equal(snap_after_5[:, : 2 * T], snap_after_4[:, T : 3 * T])
        self._assert_not_equal(snap_after_4[:, 2 * T : 3 * T], snap_after_5[:, 2 * T : 3 * T])  # last slot is chunk 5

    def test_overwrite_end_replaces_last_chunk(self):
        """mode=overwrite_end, window_size=-1: simulates Self-Forcing multi-step denoising — append a chunk, then re-write it in place."""
        T = self.TOKENS_PER_CHUNK
        cache = WanKVCache(num_blocks=self.NUM_BLOCKS, window_size=-1)

        # Chunk 0: append (cache empty)
        cache.enable_append_mode()
        self._denoise_chunk(*self._make_chunk(1), cache=cache, frame_offset=0)
        assert self._cached_len(cache) == T
        snap_chunk0_v1 = self._cached_keys(cache).clone()

        # Re-run chunk 0 with different content via overwrite_end (subsequent denoising step)
        cache.enable_overwrite_mode()
        self._denoise_chunk(*self._make_chunk(99), cache=cache, frame_offset=0)
        assert self._cached_len(cache) == T
        snap_chunk0_v2 = self._cached_keys(cache).clone()
        self._assert_not_equal(snap_chunk0_v1[:, :T], snap_chunk0_v2[:, :T])  # last (only) chunk replaced

        # Chunk 1: append (extends cache)
        cache.enable_append_mode()
        self._denoise_chunk(*self._make_chunk(2), cache=cache, frame_offset=1)
        assert self._cached_len(cache) == 2 * T
        snap_after_append = self._cached_keys(cache).clone()
        self._assert_equal(snap_chunk0_v2[:, :T], snap_after_append[:, :T])  # chunk 0 untouched

        # Re-run chunk 1 with different content via overwrite_end
        cache.enable_overwrite_mode()
        self._denoise_chunk(*self._make_chunk(98), cache=cache, frame_offset=1)
        assert self._cached_len(cache) == 2 * T
        self._assert_equal(snap_after_append[:, :T], self._cached_keys(cache)[:, :T])  # chunk 0 untouched
        self._assert_not_equal(
            snap_after_append[:, T : 2 * T], self._cached_keys(cache)[:, T : 2 * T]
        )  # chunk 1 replaced

    def test_uses_prior_context(self):
        """The cache is read during forward pass: same input gives different output with vs. without context."""
        cache = WanKVCache(num_blocks=self.NUM_BLOCKS)
        self._denoise_chunk(*self._make_chunk(1), cache=cache)
        out_with = self._denoise_chunk(*self._make_chunk(2), cache=cache)
        out_without = self._denoise_chunk(*self._make_chunk(2), cache=WanKVCache(num_blocks=self.NUM_BLOCKS))
        self._assert_not_equal(out_with, out_without)

    def test_reset(self):
        """reset() sets keys, values, and offsets back to initial state across every block."""
        cache = WanKVCache(num_blocks=self.NUM_BLOCKS)
        self._denoise_chunk(*self._make_chunk(1), cache=cache)
        cache.reset()
        assert all(bc.cached_key is None and bc.cached_value is None for bc in cache.block_caches)

    def test_frame_offset_affects_rope(self):
        """frame_offset shifts RoPE positions; same chunk at different offsets produces different output."""
        chunk = self._make_chunk(42)
        out_0 = self._denoise_chunk(*chunk, cache=WanKVCache(num_blocks=self.NUM_BLOCKS), frame_offset=0)
        out_1 = self._denoise_chunk(*chunk, cache=WanKVCache(num_blocks=self.NUM_BLOCKS), frame_offset=1)
        self._assert_not_equal(out_0, out_1)
