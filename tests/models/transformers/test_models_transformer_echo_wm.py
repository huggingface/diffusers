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

from diffusers import EchoWMTransformer3DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, require_accelerate, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class EchoWMTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return EchoWMTransformer3DModel

    @property
    def output_shape(self) -> tuple[int, int]:
        return (512, 4)

    @property
    def input_shape(self) -> tuple[int, int]:
        return (512, 4)

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self):
        return {
            "in_channels": 4,
            "out_channels": 4,
            "patch_size": 1,
            "patch_size_t": 1,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "cross_attention_dim": 16,
            "audio_in_channels": 4,
            "audio_out_channels": 4,
            "audio_num_attention_heads": 2,
            "audio_attention_head_dim": 4,
            "audio_cross_attention_dim": 8,
            "num_layers": 2,
            "qk_norm": "rms_norm_across_heads",
            "caption_channels": 16,
            "rope_double_precision": False,
        }

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        batch_size = 2
        num_frames = 2
        num_channels = 4
        height = 16
        width = 16
        audio_num_frames = 9
        audio_num_channels = 2
        num_mel_bins = 2
        embedding_dim = 16
        sequence_length = 16

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_frames * height * width, num_channels),
                generator=self.generator,
                device=torch_device,
            ),
            "audio_hidden_states": randn_tensor(
                (batch_size, audio_num_frames, audio_num_channels * num_mel_bins),
                generator=self.generator,
                device=torch_device,
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, embedding_dim), generator=self.generator, device=torch_device
            ),
            "audio_encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, embedding_dim), generator=self.generator, device=torch_device
            ),
            "timestep": (randn_tensor((batch_size,), generator=self.generator, device=torch_device).abs() * 1000),
            "encoder_attention_mask": torch.ones((batch_size, sequence_length)).bool().to(torch_device),
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "audio_num_frames": audio_num_frames,
            "fps": 25.0,
        }


class TestEchoWMTransformer(EchoWMTransformerTesterConfig, ModelTesterMixin):
    """Core model tests for Echo-WM Transformer."""

    def test_keyframes_abs_pos_embedding_marks_only_masked_tokens(self):
        init_dict = self.get_init_dict()
        init_dict["use_keyframes_abs_pos_embedding"] = True
        torch.manual_seed(0)
        model = self.model_class(**init_dict).to(torch_device).eval()
        # The parameter is zero-initialized, so an untrained checkpoint is an exact no-op.
        torch.nn.init.normal_(model.keyframes_abs_pos_embedding, std=0.1)

        inputs = self.get_dummy_inputs()
        num_tokens = inputs["hidden_states"].shape[1]
        keyframes_mask = torch.zeros(
            (inputs["hidden_states"].shape[0], num_tokens, 1), device=torch_device, dtype=torch.float32
        )

        with torch.no_grad():
            unmarked = model(**inputs, return_dict=False)[0]
            all_zero_mask = model(**inputs, video_keyframes_mask=keyframes_mask, return_dict=False)[0]
            keyframes_mask[:, : num_tokens // 2] = 1.0
            half_marked = model(**inputs, video_keyframes_mask=keyframes_mask, return_dict=False)[0]

        # An all-zero mask marks nothing, so it must match omitting the mask.
        assert torch.allclose(unmarked, all_zero_mask, atol=1e-5)
        assert not torch.allclose(unmarked, half_marked, atol=1e-5)

    def test_keyframes_mask_is_ignored_without_the_embedding(self):
        torch.manual_seed(0)
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        assert not hasattr(model, "keyframes_abs_pos_embedding")

        inputs = self.get_dummy_inputs()
        keyframes_mask = torch.ones(
            (inputs["hidden_states"].shape[0], inputs["hidden_states"].shape[1], 1),
            device=torch_device,
            dtype=torch.float32,
        )

        with torch.no_grad():
            without_mask = model(**inputs, return_dict=False)[0]
            with_mask = model(**inputs, video_keyframes_mask=keyframes_mask, return_dict=False)[0]

        assert torch.allclose(without_mask, with_mask, atol=1e-5)

    def test_echo_wm_ucpe_camera_conditioning(self):
        init_dict = self.get_init_dict()
        init_dict.update(
            ucpe_block_indices=(0, 1),
            ucpe_attention_dim=16,
            ucpe_num_attention_heads=2,
            ucpe_patches_x=16,
            ucpe_patches_y=16,
            ucpe_image_width=512,
            ucpe_image_height=512,
        )
        torch.manual_seed(0)
        model = self.model_class(**init_dict).to(torch_device).eval()
        for block in model.transformer_blocks:
            torch.nn.init.normal_(block.ucpe_out_proj.weight, std=0.1)

        inputs = self.get_dummy_inputs()
        batch_size, num_frames = inputs["hidden_states"].shape[0], inputs["num_frames"]
        viewmats = torch.eye(4, device=torch_device)[None, None].repeat(batch_size, num_frames, 1, 1)
        intrinsics = torch.eye(3, device=torch_device)[None, None].repeat(batch_size, num_frames, 1, 1)
        moved_viewmats = viewmats.clone()
        moved_viewmats[:, 1, 0, 3] = 0.25

        with torch.no_grad():
            stationary = model(**inputs, ucpe_viewmats=viewmats, ucpe_intrinsics=intrinsics, return_dict=False)[0]
            moving = model(**inputs, ucpe_viewmats=moved_viewmats, ucpe_intrinsics=intrinsics, return_dict=False)[0]

        assert stationary.shape == moving.shape == inputs["hidden_states"].shape
        assert not torch.allclose(stationary, moving, atol=1e-5)
        assert model.config.ucpe_block_indices == (0, 1)

    def test_echo_wm_ucpe_is_opt_in(self):
        model = self.model_class(**self.get_init_dict())
        assert model.config.ucpe_block_indices is None
        assert all(not block.ucpe_enabled for block in model.transformer_blocks)
        assert not any("ucpe_" in key for key in model.state_dict())

    def test_echo_wm_uses_native_rms_norm(self):
        model = self.model_class(**self.get_init_dict())
        for block in model.transformer_blocks:
            for name in (
                "norm1",
                "audio_norm1",
                "norm2",
                "audio_norm2",
                "audio_to_video_norm",
                "video_to_audio_norm",
                "norm3",
                "audio_norm3",
            ):
                assert isinstance(getattr(block, name), torch.nn.RMSNorm)

    def test_echo_wm_ucpe_recreates_fp32_coefficients_after_dtype_cast(self):
        from diffusers.models.transformers.transformer_echo_wm import EchoWMCameraRotaryPosEmbed

        kwargs = {
            "head_dim": 16,
            "patches_x": 2,
            "patches_y": 2,
            "image_width": 32,
            "image_height": 32,
        }
        reference = EchoWMCameraRotaryPosEmbed(**kwargs)
        cast = EchoWMCameraRotaryPosEmbed(**kwargs).to(dtype=torch.bfloat16)
        hidden_states = torch.randn(1, 2, 4, 16, dtype=torch.bfloat16)
        viewmats = torch.eye(4)[None, None].repeat(1, 2, 1, 1)
        intrinsics = torch.eye(3)[None, None].repeat(1, 2, 1, 1)

        reference_query = reference.prepare_transforms(viewmats, intrinsics)[0](hidden_states)
        cast_query = cast.prepare_transforms(viewmats, intrinsics)[0](hidden_states)

        assert cast.x_cos.dtype == torch.bfloat16
        assert torch.equal(reference_query, cast_query)

    def test_echo_wm_ucpe_requires_both_camera_inputs(self):
        init_dict = self.get_init_dict()
        init_dict.update(
            ucpe_block_indices=(0,),
            ucpe_attention_dim=16,
            ucpe_num_attention_heads=2,
            ucpe_patches_x=16,
            ucpe_patches_y=16,
        )
        model = self.model_class(**init_dict).to(torch_device).eval()
        inputs = self.get_dummy_inputs()
        viewmats = torch.eye(4, device=torch_device)[None, None].repeat(2, 2, 1, 1)
        with pytest.raises(ValueError, match="must be provided together"):
            model(**inputs, ucpe_viewmats=viewmats)

    def test_echo_wm_bounded_cache_replaces_noisy_chunk(self):
        model = self.model_class(**self.get_init_dict())
        cache = model.init_echo_wm_causal_caches(
            video_local_tokens=5, video_sink_tokens=2, audio_local_tokens=5, audio_sink_tokens=2
        )[0]["video_self"]
        from diffusers.models.transformers.transformer_echo_wm import _update_causal_kv_cache

        key = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        with torch.no_grad():
            _update_causal_kv_cache(cache, 0, key, key)
            replacement = torch.full_like(key[:, 1:], 99)
            cached_key, _ = _update_causal_kv_cache(cache, 1, replacement, replacement)
        assert cache["positions"].tolist() == [0, 1, 2]
        assert torch.equal(cached_key[:, 1:], replacement)

        appended = torch.full_like(key, 42)
        with torch.no_grad():
            _update_causal_kv_cache(cache, 3, appended, appended)
        assert cache["positions"].tolist() == [0, 1, 3, 4, 5]

    @require_accelerate
    @pytest.mark.parametrize("device", list(dict.fromkeys(["cpu", torch_device])))
    def test_echo_wm_causal_cache_survives_auto_offload(self, device):
        from diffusers.modular_pipelines.components_manager import custom_offload_with_hook

        device = torch.device(device)
        config = self.get_init_dict()
        config.update(
            use_prompt_embeddings=False,
            cross_attn_mod=True,
            audio_cross_attn_mod=True,
            gated_attn=True,
            audio_gated_attn=True,
            perturbed_attn=True,
            rope_type="split",
            ucpe_block_indices=(0, 1),
            ucpe_attention_dim=16,
            ucpe_num_attention_heads=2,
            ucpe_patches_x=2,
            ucpe_patches_y=2,
            ucpe_image_width=64,
            ucpe_image_height=64,
        )
        torch.manual_seed(0)
        model = self.model_class(**config).to(device).eval()
        for block in model.transformer_blocks:
            torch.nn.init.normal_(block.ucpe_out_proj.weight, std=0.1)

        video = torch.randn(1, 40, 4, device=device)
        audio = torch.randn(1, 77, 4, device=device)
        context = torch.randn(1, 5, 16, device=device)
        audio_context = torch.randn(1, 5, 8, device=device)
        video_coords = model.rope.prepare_video_coords(1, 10, 2, 2, device, fps=24)
        audio_coords = model.audio_rope.prepare_audio_coords(1, 77, device)
        poses = torch.eye(4, device=device)[None, None].repeat(1, 10, 1, 1)
        poses[:, :, 0, 3] = torch.arange(10, device=device) * 0.03
        intrinsics = torch.eye(3, device=device)[None, None].repeat(1, 10, 1, 1)

        @torch.no_grad()
        def rollout():
            caches = model.init_echo_wm_causal_caches(
                video_local_tokens=28, video_sink_tokens=4, audio_local_tokens=52, audio_sink_tokens=2
            )
            outputs = []
            for start, end, audio_start, audio_end in [(0, 1, 0, 2), (1, 4, 2, 27), (4, 7, 27, 52), (7, 10, 52, 77)]:
                # Repeated noisy writes followed by a clean commit, including cache overflow.
                for sigma in (1000.0, 500.0, 0.0):
                    outputs.append(
                        model(
                            hidden_states=video[:, start * 4 : end * 4],
                            audio_hidden_states=audio[:, audio_start:audio_end],
                            encoder_hidden_states=context,
                            audio_encoder_hidden_states=audio_context,
                            timestep=torch.full((1, (end - start) * 4), sigma, device=device),
                            audio_timestep=torch.full((1, audio_end - audio_start), sigma, device=device),
                            sigma=torch.full((1,), 1000.0, device=device),
                            audio_sigma=torch.full((1,), 1000.0, device=device),
                            video_coords=video_coords[:, :, start * 4 : end * 4],
                            audio_coords=audio_coords[:, :, audio_start:audio_end],
                            ucpe_viewmats=poses,
                            ucpe_intrinsics=intrinsics,
                            kv_caches=caches,
                            current_video_token_start=start * 4,
                            current_audio_token_start=audio_start,
                            return_dict=False,
                        )
                    )
                    for layer in caches:
                        for name in ("video_self", "audio_self", "video_ucpe", "a2v", "v2a"):
                            assert layer[name]["key"] is not None, f"Lost {name} cache at chunk {start}"
            return outputs, caches

        expected, expected_caches = rollout()
        hook = custom_offload_with_hook("transformer", model, device)
        try:
            actual, actual_caches = rollout()
        finally:
            hook.remove()

        for expected_pair, actual_pair in zip(expected, actual):
            for expected_output, actual_output in zip(expected_pair, actual_pair):
                torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
        for expected_layer, actual_layer in zip(expected_caches, actual_caches):
            for name in ("video_self", "audio_self", "video_ucpe", "a2v", "v2a"):
                for field in ("key", "value", "positions"):
                    torch.testing.assert_close(actual_layer[name][field], expected_layer[name][field], rtol=0, atol=0)

    def test_echo_wm_bounded_ucpe_transform_preserves_dtype(self):
        from diffusers.models.transformers.transformer_echo_wm import _ucpe_transform

        hidden_states = torch.randn(1, 2, 3, 4, dtype=torch.bfloat16)
        seen_dtypes = []

        def transform(value):
            seen_dtypes.append(value.dtype)
            return value

        output = _ucpe_transform(transform, hidden_states)

        assert seen_dtypes == [torch.float32]
        assert output.dtype == torch.bfloat16

    def test_echo_wm_bounded_cache_uses_local_rope(self):
        from diffusers.models.transformers.transformer_echo_wm import EchoWMAttention

        torch.manual_seed(0)
        attention = EchoWMAttention(
            query_dim=4,
            heads=1,
            kv_heads=1,
            dim_head=4,
            bias=False,
            out_bias=False,
            norm_elementwise_affine=False,
        ).eval()
        phases = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1).repeat(1, 1, 4)
        rotary_emb = (phases.cos(), phases.sin())
        chunks = [torch.randn(1, 2, 4) for _ in range(3)]
        cache = {
            "key": None,
            "value": None,
            "positions": None,
            "local_size": 4,
            "sink_size": 1,
            "local_rotary_emb": rotary_emb,
        }

        with torch.no_grad():
            attention(chunks[0], kv_cache=cache, kv_cache_start=0)
            cached_before_overflow = attention(chunks[1], kv_cache=cache, kv_cache_start=2)
            expected_before_overflow = attention(torch.cat(chunks[:2], dim=1), query_rotary_emb=rotary_emb)[:, -2:]
            cached_after_overflow = attention(chunks[2], kv_cache=cache, kv_cache_start=4)
            active_window = torch.cat((chunks[0][:, :1], chunks[1][:, 1:], chunks[2]), dim=1)
            expected_after_overflow = attention(active_window, query_rotary_emb=rotary_emb)[:, -2:]

        assert cache["positions"].tolist() == [0, 3, 4, 5]
        assert torch.allclose(cached_before_overflow, expected_before_overflow, atol=1e-6)
        assert torch.allclose(cached_after_overflow, expected_after_overflow, atol=1e-6)

    def test_echo_wm_cross_modal_cache_uses_local_rope_slices(self):
        from diffusers.models.transformers.transformer_echo_wm import EchoWMAttention, _slice_echo_wm_rotary_emb

        torch.manual_seed(0)
        attention = EchoWMAttention(
            query_dim=4,
            heads=1,
            kv_heads=1,
            dim_head=4,
            bias=False,
            out_bias=False,
            norm_elementwise_affine=False,
        ).eval()
        query_phases = (torch.arange(4, dtype=torch.float32) + 1).reshape(1, 4, 1).repeat(1, 1, 4)
        key_phases = (torch.arange(4, dtype=torch.float32) + 5).reshape(1, 4, 1).repeat(1, 1, 4)
        query_rotary_emb = (query_phases.cos(), query_phases.sin())
        key_rotary_emb = (key_phases.cos(), key_phases.sin())
        queries = [torch.randn(1, 2, 4) for _ in range(3)]
        encoder_chunks = [torch.randn(1, 2, 4) for _ in range(3)]
        cache = {
            "key": None,
            "value": None,
            "positions": None,
            "local_size": 4,
            "sink_size": 1,
            "local_query_rotary_emb": query_rotary_emb,
            "local_key_rotary_emb": key_rotary_emb,
            "local_query_slices": {(0, 2): (0, 2), (2, 4): (2, 4), (4, 6): (2, 4)},
        }

        with torch.no_grad():
            attention(queries[0], encoder_chunks[0], kv_cache=cache, kv_cache_start=0)
            cached_before_overflow = attention(queries[1], encoder_chunks[1], kv_cache=cache, kv_cache_start=2)
            expected_before_overflow = attention(
                queries[1],
                torch.cat(encoder_chunks[:2], dim=1),
                query_rotary_emb=_slice_echo_wm_rotary_emb(query_rotary_emb, 2, 4),
                key_rotary_emb=key_rotary_emb,
            )
            cached_after_overflow = attention(queries[2], encoder_chunks[2], kv_cache=cache, kv_cache_start=4)
            active_window = torch.cat((encoder_chunks[0][:, :1], encoder_chunks[1][:, 1:], encoder_chunks[2]), dim=1)
            expected_after_overflow = attention(
                queries[2],
                active_window,
                query_rotary_emb=_slice_echo_wm_rotary_emb(query_rotary_emb, 2, 4),
                key_rotary_emb=key_rotary_emb,
            )

        assert cache["positions"].tolist() == [0, 3, 4, 5]
        assert torch.allclose(cached_before_overflow, expected_before_overflow, atol=1e-6)
        assert torch.allclose(cached_after_overflow, expected_after_overflow, atol=1e-6)

    @pytest.mark.parametrize("processor_name", ["EchoWMAudioVideoAttnProcessor", "EchoWMPerturbedAttnProcessor"])
    def test_echo_wm_text_cross_attention_cache(self, processor_name):
        from diffusers.models.transformers import transformer_echo_wm

        torch.manual_seed(0)
        attention = transformer_echo_wm.EchoWMAttention(
            query_dim=4,
            cross_attention_dim=6,
            heads=1,
            kv_heads=1,
            dim_head=4,
            bias=False,
            out_bias=False,
            norm_elementwise_affine=False,
            processor=getattr(transformer_echo_wm, processor_name)(),
        ).eval()
        hidden_states = torch.randn(1, 2, 4)
        encoder_hidden_states = torch.randn(1, 3, 6)
        cache = {"key": None, "value": None}

        with torch.no_grad():
            first = attention(hidden_states, encoder_hidden_states, crossattn_cache=cache)
            cached = attention(hidden_states, encoder_hidden_states + 10, crossattn_cache=cache)

        assert cache["key"] is not None and cache["value"] is not None
        assert torch.equal(first, cached)


class TestEchoWMTransformerMemory(EchoWMTransformerTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for Echo-WM Transformer."""


class TestEchoWMTransformerTraining(EchoWMTransformerTesterConfig, TrainingTesterMixin):
    """Training tests for Echo-WM Transformer."""

    def test_gradient_checkpointing_is_applied(self):
        super().test_gradient_checkpointing_is_applied(expected_set={"EchoWMTransformer3DModel"})


class TestEchoWMTransformerAttention(EchoWMTransformerTesterConfig, AttentionTesterMixin):
    """Attention processor tests for Echo-WM Transformer."""

    @pytest.mark.skip(
        "EchoWMAttention does not set is_cross_attention, so fuse_projections tries to fuse Q+K+V together even for cross-attention modules with different input dimensions."
    )
    def test_fuse_unfuse_qkv_projections(self, atol=1e-3, rtol=0):
        pass


class TestEchoWMTransformerCompile(EchoWMTransformerTesterConfig, TorchCompileTesterMixin):
    """Torch compile tests for Echo-WM Transformer."""
