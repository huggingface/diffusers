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

from diffusers import SanaWMLTX2RefinerTransformer3DModel
from diffusers.models.transformers.transformer_sana_wm_refiner import (
    SanaWMLTX2AudioVideoRotaryPosEmbed,
    SanaWMRefinerKVCache,
)
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class SanaWMLTX2RefinerTransformer3DTesterConfig(BaseModelTesterConfig):
    # Tiny stand-in for the LTX-2 based SANA-WM stage-2 refiner. The refiner forward is video-only, but the audio
    # submodules are still built (so LTX-2 checkpoints round-trip), hence the small audio arguments below.
    batch_size = 2
    num_frames = 2
    height = 4
    width = 4
    in_channels = 4
    num_attention_heads = 2
    attention_head_dim = 8
    caption_channels = 16
    sequence_length = 8
    fps = 24.0

    @property
    def model_class(self):
        return SanaWMLTX2RefinerTransformer3DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def input_shape(self) -> tuple:
        return (self.num_frames * self.height * self.width, self.in_channels)

    @property
    def output_shape(self) -> tuple:
        return (self.num_frames * self.height * self.width, self.in_channels)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "out_channels": self.in_channels,
            "patch_size": 1,
            "patch_size_t": 1,
            "num_attention_heads": self.num_attention_heads,
            "attention_head_dim": self.attention_head_dim,
            "cross_attention_dim": 16,
            "audio_in_channels": 4,
            "audio_out_channels": 4,
            "audio_num_attention_heads": 2,
            "audio_attention_head_dim": 4,
            "audio_cross_attention_dim": 8,
            "num_layers": 2,
            "qk_norm": "rms_norm_across_heads",
            "caption_channels": self.caption_channels,
            "rope_double_precision": False,
        }

    def get_video_rotary_emb(self, batch_size: int, num_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the caller-supplied video RoPE for frames `[0, num_frames)` with the model's default RoPE config."""
        rope = SanaWMLTX2AudioVideoRotaryPosEmbed(
            dim=self.num_attention_heads * self.attention_head_dim,
            patch_size=1,
            patch_size_t=1,
            base_num_frames=20,
            base_height=2048,
            base_width=2048,
            scale_factors=(8, 32, 32),
            modality="video",
            double_precision=False,
            num_attention_heads=self.num_attention_heads,
        )
        coords = rope.prepare_video_coords(
            batch_size, num_frames, self.height, self.width, device=torch_device, fps=self.fps
        )
        return rope(coords, device=torch_device)

    def get_dummy_inputs(self, batch_size: int | None = None) -> dict:
        batch_size = batch_size or self.batch_size
        num_tokens = self.num_frames * self.height * self.width

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_tokens, self.in_channels), generator=self.generator, device=torch_device
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, self.sequence_length, self.caption_channels),
                generator=self.generator,
                device=torch_device,
            ),
            # The refiner takes a per-token timestep that is already scaled by `timestep_scale_multiplier`.
            "timestep": torch.rand((batch_size, num_tokens), generator=self.generator).to(torch_device) * 1000,
            "video_rotary_emb": self.get_video_rotary_emb(batch_size, self.num_frames),
            "encoder_attention_mask": torch.ones((batch_size, self.sequence_length), device=torch_device),
        }


class TestSanaWMLTX2RefinerTransformer3D(SanaWMLTX2RefinerTransformer3DTesterConfig, ModelTesterMixin):
    def test_kv_cache_sink_injection_changes_output(self):
        """Drive the KV cache the way `SanaWMLTX2Refiner` does: seed the attention sink from a clean block with
        `capture_pre_rope`, then denoise the next block with `inject` and check the sink is actually attended to."""
        torch.manual_seed(0)
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        num_layers = len(model.transformer_blocks)

        batch_size = self.batch_size
        sink_size = 1
        tokens_per_frame = self.height * self.width
        encoder_hidden_states = randn_tensor(
            (batch_size, self.sequence_length, self.caption_channels), generator=self.generator, device=torch_device
        )
        encoder_attention_mask = torch.ones((batch_size, self.sequence_length), device=torch_device)

        # Sink block: frames `[0, sink_size)` at sigma=0.
        sink_tokens = randn_tensor(
            (batch_size, sink_size * tokens_per_frame, self.in_channels), generator=self.generator, device=torch_device
        )
        sink_inputs = {
            "hidden_states": sink_tokens,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": torch.zeros((batch_size, sink_tokens.shape[1]), device=torch_device),
            "video_rotary_emb": model.build_rotary_emb_for_absolute_positions(
                batch_size=batch_size,
                frame_positions=list(range(sink_size)),
                height=self.height,
                width=self.width,
                device=torch_device,
                fps=self.fps,
            ),
            "encoder_attention_mask": encoder_attention_mask,
        }

        # Active block: frames `[sink_size, sink_size + num_frames)` at some intermediate sigma.
        block_start = sink_size
        frame_positions = list(range(block_start, block_start + self.num_frames))
        active_tokens = randn_tensor(
            (batch_size, self.num_frames * tokens_per_frame, self.in_channels),
            generator=self.generator,
            device=torch_device,
        )
        active_inputs = {
            "hidden_states": active_tokens,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": torch.full((batch_size, active_tokens.shape[1]), 500.0, device=torch_device),
            "video_rotary_emb": model.build_rotary_emb_for_absolute_positions(
                batch_size=batch_size,
                frame_positions=frame_positions,
                height=self.height,
                width=self.width,
                device=torch_device,
                fps=self.fps,
            ),
            "encoder_attention_mask": encoder_attention_mask,
        }

        kv_cache = SanaWMRefinerKVCache(num_layers)
        with torch.no_grad():
            # 1. Capture the pre-RoPE sink K/V; capturing injects no prefix, so it must match a plain forward.
            sink_plain = model(**sink_inputs, return_dict=False)[0]
            sink_captured = model(
                **sink_inputs, kv_cache=kv_cache, kv_cache_mode="capture_pre_rope", return_dict=False
            )[0]
            assert torch.allclose(sink_plain, sink_captured, atol=1e-6)

            inner_dim = self.num_attention_heads * self.attention_head_dim
            for layer_idx in range(num_layers):
                layer_cache = kv_cache.get(layer_idx)
                sink_k_pre, sink_v = layer_cache.get_captured_pre_rope()
                assert sink_k_pre.shape == (batch_size, sink_size * tokens_per_frame, inner_dim)
                assert sink_v.shape == (batch_size, sink_size * tokens_per_frame, inner_dim)
                layer_cache.store_sink(sink_k_pre, sink_v)

            # 2. Slide the sink RoPE so it sits immediately before the active block (no history yet).
            history_frames = 0
            sink_rope_offset = block_start - history_frames - sink_size
            kv_cache.sink_pe = model.build_rotary_emb_for_absolute_positions(
                batch_size=batch_size,
                frame_positions=list(range(sink_rope_offset, sink_rope_offset + sink_size)),
                height=self.height,
                width=self.width,
                device=torch_device,
                fps=self.fps,
            )

            # 3. Denoise the active block with and without the injected sink.
            no_cache = model(**active_inputs, return_dict=False)[0]
            injected = model(**active_inputs, kv_cache=kv_cache, kv_cache_mode="inject", return_dict=False)[0]

        assert injected.shape == no_cache.shape == (batch_size, active_tokens.shape[1], self.in_channels)
        assert torch.isfinite(injected).all()
        assert not torch.allclose(injected, no_cache, atol=1e-4), "Injecting the sink K/V did not change the output."


class TestSanaWMLTX2RefinerTransformer3DMemory(SanaWMLTX2RefinerTransformer3DTesterConfig, MemoryTesterMixin):
    pass


class TestSanaWMLTX2RefinerTransformer3DCompile(SanaWMLTX2RefinerTransformer3DTesterConfig, TorchCompileTesterMixin):
    pass


class TestSanaWMLTX2RefinerTransformer3DTraining(SanaWMLTX2RefinerTransformer3DTesterConfig, TrainingTesterMixin):
    pass
