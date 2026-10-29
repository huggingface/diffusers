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

import copy
import math

import pytest
import torch

from diffusers import HyNomasTransformer3DModel
from diffusers.models.attention_dispatch import attention_backend
from diffusers.models.transformers.transformer_hy_nomas import (
    HyNomasAttnProcessor,
    HyNomasMagiAttnProcessor,
    HyNomasRotaryPosEmbed,
    HyNomasTimestepEmbedder,
    HyNomasTokenRefiner,
)
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import (
    enable_full_determinism,
    is_torch_compile,
    require_torch_gpu,
    require_torch_multi_gpu,
    torch_device,
)
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class HyNomasTransformerTesterConfig(BaseModelTesterConfig):
    main_input_name = "hidden_states"

    @property
    def model_class(self):
        return HyNomasTransformer3DModel

    def get_init_dict(self):
        return {
            "in_channels": 4,
            "out_channels": 4,
            "num_layers": 2,
            "num_attention_heads": 2,
            "attention_head_dim": 12,
            "text_embed_dim": 16,
            "mlp_ratio": 2.0,
            "rope_axes_dim": (4, 4, 4),
            "camera_embed_dim": 8,
            "camera_attention_dim": 24,
            "camera_num_attention_heads": 2,
            "camera_rope_axes_dim": (4, 4, 4),
        }

    def get_dummy_inputs(self, height=2, width=2):
        generator = torch.Generator("cpu").manual_seed(0)
        return {
            "hidden_states": randn_tensor((1, 4, 2, height, width), generator=generator, device=torch_device),
            "encoder_hidden_states": randn_tensor((1, 5, 16), generator=generator, device=torch_device),
            "timestep": torch.tensor([500.0], device=torch_device),
            "camera_extrinsics": torch.eye(4, device=torch_device).repeat(1, 5, 1, 1),
        }

    @property
    def input_shape(self):
        return (4, 2, 2, 2)

    @property
    def output_shape(self):
        return (4, 2, 2, 2)


class TestHyNomasTransformerModel(HyNomasTransformerTesterConfig, ModelTesterMixin):
    @pytest.mark.parametrize("reset_method", ["disable_reference_attention", "reset_attention_backend", "native"])
    @torch.no_grad()
    def test_reference_attention_can_restore_standard_inference(self, monkeypatch, reset_method):
        monkeypatch.setattr("diffusers.models.attention_dispatch._CAN_USE_FLASH_ATTN", True)
        monkeypatch.setattr("diffusers.models.attention_dispatch._CAN_USE_FLASH_ATTN_3", True)
        monkeypatch.setattr("diffusers.models.attention_dispatch._CAN_USE_MAGI_ATTN", True)
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        original_processors = {name: type(processor) for name, processor in model.attn_processors.items()}
        inputs = self.get_dummy_inputs()
        with attention_backend("native"):
            expected = model(**inputs).sample
            model.enable_reference_attention(deterministic=True)
            assert all(
                type(processor) != original_processors[name] for name, processor in model.attn_processors.items()
            )
            if reset_method == "native":
                model.set_attention_backend("native")
            else:
                getattr(model, reset_method)()
            assert {name: type(processor) for name, processor in model.attn_processors.items()} == original_processors
            torch.testing.assert_close(model(**inputs).sample, expected, atol=0, rtol=0)

    @pytest.mark.parametrize(
        "unavailable_backend,error",
        [
            ("_CAN_USE_FLASH_ATTN", "Flash Attention backend"),
            ("_CAN_USE_FLASH_ATTN_3", "Flash Attention 3 backend"),
            ("_CAN_USE_MAGI_ATTN", "Magi Attention backend"),
        ],
    )
    def test_reference_attention_reports_missing_dependency(self, monkeypatch, unavailable_backend, error):
        monkeypatch.setattr("diffusers.models.attention_dispatch._CAN_USE_FLASH_ATTN", True)
        monkeypatch.setattr("diffusers.models.attention_dispatch._CAN_USE_FLASH_ATTN_3", True)
        monkeypatch.setattr("diffusers.models.attention_dispatch._CAN_USE_MAGI_ATTN", True)
        monkeypatch.setattr(f"diffusers.models.attention_dispatch.{unavailable_backend}", False)
        model = self.model_class(**self.get_init_dict())
        processors = model.attn_processors
        with pytest.raises(RuntimeError, match=error):
            model.enable_reference_attention()
        assert model.attn_processors == processors

    @torch.no_grad()
    def test_reference_action_ranges_match_dense_visibility(self, monkeypatch):
        def dense_range_attention(query, key, value, backend, attention_kwargs):
            assert backend == "magi"
            mask = torch.zeros(query.shape[1], key.shape[1], dtype=torch.bool)
            for (start, end), (key_start, key_end) in zip(
                attention_kwargs["query_ranges"], attention_kwargs["key_ranges"]
            ):
                mask[start:end, key_start:key_end] = True
            return torch.nn.functional.scaled_dot_product_attention(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=mask
            ).transpose(1, 2)

        monkeypatch.setattr(
            "diffusers.models.transformers.transformer_hy_nomas.dispatch_attention_fn", dense_range_attention
        )
        generator = torch.Generator().manual_seed(17)
        query = torch.randn(1, 10, 2, 8, generator=generator)
        key = torch.randn(1, 13, 2, 8, generator=generator)
        value = torch.randn(1, 13, 2, 8, generator=generator)
        text_mask = torch.tensor([[True, True, True, True, True, False]])
        key_mask = torch.tensor([[True, False, True, True, True, True, True, True, True, True, True, True, False]])
        # History 0/2, current video 3:7, head text 7:9, actions 9 and 10:12.
        expected_mask = torch.zeros(10, 13, dtype=torch.bool)
        expected_mask[:9, [0, 2, 3, 4, 5, 6, 7, 8]] = True
        expected_mask[[0, 1, 6], 9] = True
        expected_mask[[2, 3, 7, 8], 10:12] = True
        expected = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=expected_mask
        ).transpose(1, 2)
        actual = HyNomasMagiAttnProcessor()._compute_attention(
            query, key, value, key_mask, text_mask, 3, 4, 6, text_mask, [[2, 3, 5]], [(0, 2, 0), (2, 4, 1)]
        )
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    @require_torch_gpu
    @torch.no_grad()
    def test_bfloat16_preserves_autocast_normalization_precision(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device, dtype=torch.bfloat16).eval()
        inputs = {
            name: value.to(torch.bfloat16) if value.is_floating_point() else value
            for name, value in self.get_dummy_inputs().items()
        }
        # Original inference uses CUDA autocast. Native inference must preserve its float32
        # LayerNorm/modulation boundary without requiring callers to enable autocast.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = model(**inputs).sample
        actual = model(**inputs).sample
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    @require_torch_gpu
    @torch.no_grad()
    def test_masked_context_preserves_autocast_reduction_precision(self):
        refiner = HyNomasTokenRefiner(16, 24, 2).to(torch_device, dtype=torch.bfloat16).eval()
        generator = torch.Generator("cpu").manual_seed(42)
        hidden_states = randn_tensor((1, 1280, 16), generator=generator, device=torch_device).to(torch.bfloat16) * 100
        mask = torch.arange(1280, device=torch_device)[None] < 251
        captured = []
        hook = refiner.c_embedder.register_forward_pre_hook(lambda module, args: captured.append(args[0].clone()))
        refiner(hidden_states, torch.tensor([1000.0], device=torch_device), mask)
        hook.remove()
        # CUDA autocast promotes sum, including its division denominator, to float32. The linear
        # layer then casts the quotient once. Reducing directly in bfloat16 loses information first.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            weights = mask.to(hidden_states.dtype).unsqueeze(-1)
            expected = (hidden_states * weights).sum(1) / weights.sum(1)
        torch.testing.assert_close(captured[0], expected.to(torch.bfloat16), atol=0, rtol=0)

    @require_torch_gpu
    @torch.no_grad()
    def test_timestep_frequencies_survive_weight_cast(self):
        embedder = HyNomasTimestepEmbedder(24).to(torch_device, dtype=torch.bfloat16)
        captured = []
        hook = embedder.mlp[0].register_forward_pre_hook(lambda module, args: captured.append(args[0].clone()))
        timestep = torch.tensor([1000.0, 916.6667, 500.0], device=torch_device)
        embedder(timestep, torch.bfloat16)
        hook.remove()
        # The published sinusoid uses CPU float32 frequencies before its GPU trigonometry.
        frequencies = torch.exp(-math.log(10000.0) * torch.arange(128, dtype=torch.float32) / 128)
        angles = timestep[:, None] * frequencies.to(torch_device)[None]
        expected = torch.cat((angles.cos(), angles.sin()), -1).to(torch.bfloat16)
        torch.testing.assert_close(captured[0], expected, atol=0, rtol=0)

    @require_torch_gpu
    @pytest.mark.parametrize("frame_offset,num_frames", [(36, 72), (108, 30), (138, 72), (210, 36)])
    def test_continuation_rotary_uses_cpu_rounding(self, frame_offset, num_frames):
        rotary = HyNomasRotaryPosEmbed((16, 56, 56), 256)
        expected = rotary(num_frames, 22, 40, "cpu", frame_offset)
        actual = rotary(num_frames, 22, 40, torch_device, frame_offset)
        for expected_component, actual_component in zip(expected, actual):
            torch.testing.assert_close(actual_component.cpu(), expected_component, atol=0, rtol=0)

    @torch.no_grad()
    def test_cache_is_explicit_and_reads_do_not_mutate(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        inputs = self.get_dummy_inputs()
        cache = [{} for _ in range(model.config.num_layers)]
        baseline = model(**inputs).sample
        first = model(**inputs, kv_cache=cache, update_cache=True, cache_action_start=2).sample
        torch.testing.assert_close(baseline, first)
        assert cache[0]["key"].shape[1] == 13
        assert cache[0]["mask"][0, :10].all()
        assert not cache[0]["mask"][0, 10:].any()
        before = copy.deepcopy(cache)
        continued = model(**inputs, kv_cache=cache, frame_offset=2).sample
        assert not torch.equal(continued, baseline)
        for previous, current in zip(before, cache):
            for key in previous:
                torch.testing.assert_close(previous[key], current[key])
        model(**inputs, kv_cache=cache, update_cache=True, frame_offset=2)
        assert cache[0]["key"].shape[1] == 26
        torch.testing.assert_close(model(**inputs).sample, baseline)

    @torch.no_grad()
    def test_camera_off_matches_no_camera(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        inputs = self.get_dummy_inputs()
        disabled = model(**inputs, camera_attention_mask=torch.zeros(1, device=torch_device)).sample
        inputs.pop("camera_extrinsics")
        torch.testing.assert_close(disabled, model(**inputs).sample)

    @torch.no_grad()
    def test_text_padding_does_not_change_output(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        inputs = self.get_dummy_inputs()
        inputs["encoder_attention_mask"] = torch.tensor([[1, 1, 1, 0, 0]], device=torch_device)
        expected = model(**inputs).sample
        inputs["encoder_hidden_states"][:, 3:] += 100
        torch.testing.assert_close(expected, model(**inputs).sample)

    @torch.no_grad()
    def test_action_groups_match_unrestricted_attention_when_all_text_visible(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        inputs = self.get_dummy_inputs()
        expected = model(**inputs).sample
        actual = model(**inputs, action_text_offsets=[[0, 5]], action_frame_windows=[(0, 2, 0)]).sample
        torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @torch.no_grad()
    def test_action_attention_matches_explicit_visibility(self, dtype):
        generator = torch.Generator().manual_seed(17)
        query = torch.randn(1, 10, 2, 8, generator=generator, dtype=dtype)
        key = torch.randn(1, 13, 2, 8, generator=generator, dtype=dtype)
        value = torch.randn(1, 13, 2, 8, generator=generator, dtype=dtype)
        text_mask = torch.tensor([[True, False, True, True, True, False]])
        key_mask = torch.cat((torch.tensor([[True, False, True, True, True, True, True]]), text_mask), dim=1)
        # Visible history: 0/2; video: 3:7; head text: 7; actions: 9 and 10:12.
        expected_mask = torch.zeros(10, 13, dtype=torch.bool)
        expected_mask[:8, [0, 2, 3, 4, 5, 6, 7]] = True
        expected_mask[[0, 1, 5], 9] = True
        expected_mask[[2, 3, 6, 7], 10:12] = True
        expected = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=expected_mask
        ).transpose(1, 2)
        actual = HyNomasAttnProcessor()._compute_attention(
            query, key, value, key_mask, text_mask, 3, 4, 6, text_mask, [[1, 2, 6]], [(0, 2, 0), (2, 4, 1)]
        )
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    @torch.no_grad()
    def test_hidden_cached_text_has_no_effect(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        inputs = self.get_dummy_inputs()
        cache = [{} for _ in range(model.config.num_layers)]
        model(**inputs, kv_cache=cache, update_cache=True, cache_action_start=2)
        expected = model(**inputs, kv_cache=cache, frame_offset=2).sample
        for layer in cache:
            layer["key"][:, 10:] += 100
            layer["value"][:, 10:] += 100
        torch.testing.assert_close(expected, model(**inputs, kv_cache=cache, frame_offset=2).sample)

    def test_cache_validation(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        with pytest.raises(ValueError, match="caller-owned"):
            model(**self.get_dummy_inputs(), update_cache=True)
        with pytest.raises(ValueError, match="one dictionary"):
            model(**self.get_dummy_inputs(), kv_cache=[{}])


class TestHyNomasTransformerMemory(HyNomasTransformerTesterConfig, MemoryTesterMixin):
    @require_torch_multi_gpu
    @pytest.mark.parametrize("camera_device", [0, 1])
    @torch.no_grad()
    def test_device_map_camera_residual(self, camera_device, tmp_path):
        torch.manual_seed(0)
        model = self.model_class(**self.get_init_dict()).to("cuda:0").eval()
        inputs = self.get_dummy_inputs()
        inputs.update(
            encoder_attention_mask=torch.tensor([[1, 1, 1, 0, 0]], device="cuda:0", dtype=torch.bool),
            action_text_offsets=[[1, 2, 3]],
            action_frame_windows=[(0, 1, 0), (1, 2, 1)],
            cache_action_start=1,
        )
        expected_cache = [{} for _ in range(model.config.num_layers)]
        expected = model(**inputs, kv_cache=expected_cache, update_cache=True).sample
        expected_continuation = model(**inputs, kv_cache=expected_cache, frame_offset=2).sample
        model.cpu().save_pretrained(tmp_path)
        device_map = {
            name: "cuda:0" for name, _ in model.named_children() if name not in ("double_blocks", "action_modules")
        }
        device_map["extrinsics_embedding_camctrl"] = f"cuda:{camera_device}"
        device_map["c2ws_hidden_states_layer2"] = f"cuda:{1 - camera_device}"
        device_map["final_layer"] = "cuda:1"
        for index in range(model.config.num_layers):
            device_map[f"double_blocks.{index}"] = f"cuda:{index % 2}"
            device_map[f"action_modules.{index}"] = f"cuda:{1 - index % 2}"
        loaded = self.model_class.from_pretrained(tmp_path, device_map=device_map)
        assert loaded.extrinsics_embedding_camctrl.weight.device == torch.device(f"cuda:{camera_device}")
        assert loaded.c2ws_hidden_states_layer2.weight.device == torch.device(f"cuda:{1 - camera_device}")
        actual_cache = [{} for _ in range(model.config.num_layers)]
        actual = loaded(**inputs, kv_cache=actual_cache, update_cache=True).sample
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)
        actual_continuation = loaded(**inputs, kv_cache=actual_cache, frame_offset=2).sample
        torch.testing.assert_close(actual_continuation, expected_continuation, atol=1e-5, rtol=0)
        for expected_layer, actual_layer in zip(expected_cache, actual_cache):
            for name in expected_layer:
                torch.testing.assert_close(actual_layer[name].cpu(), expected_layer[name].cpu(), atol=1e-5, rtol=0)


class TestHyNomasTransformerTorchCompile(HyNomasTransformerTesterConfig, TorchCompileTesterMixin):
    @property
    def different_shapes_for_compilation(self):
        return [(2, 2), (2, 4), (4, 4)]


@is_torch_compile
class TestHyNomasTransformerActionCompile(HyNomasTransformerTesterConfig):
    @pytest.mark.parametrize("cache_mode", ["none", "read", "write"])
    @torch.no_grad()
    def test_action_masks_and_history_compile(self, cache_mode):
        torch.compiler.reset()
        model = self.model_class(**self.get_init_dict()).cpu().eval()
        inputs = {name: value.cpu() for name, value in self.get_dummy_inputs().items()}
        history = None
        if cache_mode != "none":
            history = [{} for _ in range(model.config.num_layers)]
            model(**inputs, kv_cache=history, update_cache=True, cache_action_start=1)
        inputs.update(
            action_text_offsets=[[1, 3, 5]],
            action_frame_windows=[(0, 1, 0), (1, 2, 1)],
            frame_offset=0 if history is None else 2,
            update_cache=cache_mode == "write",
            return_dict=False,
        )
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        try:
            with torch._dynamo.config.patch(error_on_recompile=True):
                for mask in ([1, 1, 1, 0, 0], [1, 0, 1, 1, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]):
                    inputs["encoder_attention_mask"] = torch.tensor([mask], dtype=torch.bool)
                    eager_cache, compiled_cache = copy.deepcopy(history), copy.deepcopy(history)
                    expected = model(**inputs, kv_cache=eager_cache)[0]
                    actual = compiled(**inputs, kv_cache=compiled_cache)[0]
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    if history is not None:
                        for eager_layer, compiled_layer in zip(eager_cache, compiled_cache):
                            for key in eager_layer:
                                torch.testing.assert_close(compiled_layer[key], eager_layer[key], atol=0, rtol=0)
        finally:
            torch.compiler.reset()


class TestHyNomasTransformerTraining(HyNomasTransformerTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        super().test_gradient_checkpointing_is_applied(expected_set={"HyNomasTransformer3DModel"})


class TestHyNomasTransformerAttention(HyNomasTransformerTesterConfig, AttentionTesterMixin):
    pass
