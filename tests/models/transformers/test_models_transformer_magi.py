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

import math

import pytest
import torch

from diffusers import AutoencoderKLMagi, MagiTransformer3DModel
from diffusers.models.attention_dispatch import attention_backend
from diffusers.models.transformers.transformer_magi import MagiTimestepEmbedding
from diffusers.utils import is_flash_attn_available
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


enable_full_determinism()


class MagiTransformerTesterConfig(BaseModelTesterConfig):
    main_input_name = "hidden_states"

    @property
    def model_class(self):
        return MagiTransformer3DModel

    @property
    def pretrained_model_name_or_path(self):
        return None

    @property
    def pretrained_model_kwargs(self):
        return {}

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self):
        return {
            "in_channels": 4,
            "out_channels": 4,
            "num_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "attention_head_dim": 32,
            "ffn_dim": 96,
            "condition_dim": 16,
            "caption_channels": 16,
            "caption_max_length": 8,
            "frequency_embedding_size": 16,
        }

    def get_dummy_inputs(self):
        return {
            "hidden_states": randn_tensor((2, 4, 4, 4, 4), generator=self.generator, device=torch_device),
            "encoder_hidden_states": randn_tensor((2, 8, 16), generator=self.generator, device=torch_device),
            "timestep": torch.tensor([0.2, 0.5], device=torch_device),
        }

    @property
    def input_shape(self):
        return (4, 4, 4, 4)

    @property
    def output_shape(self):
        return self.input_shape


class TestMagiTransformerModel(MagiTransformerTesterConfig, ModelTesterMixin):
    @torch.no_grad()
    def test_empty_text_mask(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        inputs = self.get_dummy_inputs()
        inputs["encoder_attention_mask"] = torch.zeros(2, 8, dtype=torch.bool, device=torch_device)
        with pytest.raises(RuntimeError):
            model(**inputs)

    @pytest.mark.parametrize("regional", [False, True])
    @torch.no_grad()
    def test_masked_fullgraph_capture(self, regional):
        from copy import deepcopy

        torch.compiler.reset()
        reference = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        model = deepcopy(reference)
        inputs = self.get_dummy_inputs()
        masks = [
            [[True, False] * 4, [True, True, False, False] * 2],
            [[True] * 8, [True, False, False, False] * 2],
            [[True] * 8] * 2,
        ]
        try:
            with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
                if regional:
                    model.compile_repeated_blocks(backend="eager", fullgraph=True)
                else:
                    model = torch.compile(model, backend="eager", fullgraph=True)
                for mask in masks:
                    inputs["encoder_attention_mask"] = torch.tensor(mask, device=torch_device)
                    expected = reference(**inputs).sample
                    torch.testing.assert_close(model(**inputs).sample, expected, atol=0, rtol=0)
                    changed = dict(inputs)
                    changed["encoder_hidden_states"] = inputs["encoder_hidden_states"].clone()
                    changed["encoder_hidden_states"][~inputs["encoder_attention_mask"]] += 100
                    torch.testing.assert_close(model(**changed).sample, expected, atol=0, rtol=0)
        finally:
            torch.compiler.reset()

    @pytest.mark.parametrize("frames", [4, 16])
    @pytest.mark.parametrize(
        "duplicate_channels,distilled", [(False, False), (False, True), (True, False), (True, True)]
    )
    @torch.no_grad()
    def test_vae_transformer_dataflow(self, frames, duplicate_channels, distilled, tmp_path):
        torch.manual_seed(0)
        vae = (
            AutoencoderKLMagi(
                latent_channels=4,
                embed_dim=32,
                num_layers=1,
                num_attention_heads=4,
                mlp_ratio=2,
                patch_size=2,
                patch_length=4,
                sample_size=8,
                sample_frames=8,
            )
            .to(torch_device)
            .eval()
        )
        model = (
            self.model_class(
                **self.get_init_dict(),
                duplicate_channels=duplicate_channels,
                gated_linear_unit=duplicate_channels,
                x_rescale_factor=0.1 if duplicate_channels else 1.0,
                distilled=distilled,
            )
            .to(torch_device)
            .eval()
        )
        video = randn_tensor((2, 3, frames, 8, 12), generator=self.generator, device=torch_device)
        latent = vae.encode(video).latent_dist.mode()
        assert latent.shape == (2, 4, frames // 4, 4, 6)
        scale_factor = 0.18215
        inputs = self.get_dummy_inputs()
        inputs["hidden_states"] = latent * scale_factor
        if distilled:
            inputs["timestep_delta"] = torch.tensor(8.0, device=torch_device)
        predicted = model(**inputs).sample
        assert predicted.shape == latent.shape
        decoded = vae.decode(predicted / scale_factor, num_frames=frames).sample
        assert decoded.shape == video.shape
        assert decoded.isfinite().all()
        vae.save_pretrained(tmp_path / "vae")
        model.save_pretrained(tmp_path / "transformer")
        restored_vae = AutoencoderKLMagi.from_pretrained(tmp_path / "vae").to(torch_device)
        restored_model = self.model_class.from_pretrained(tmp_path / "transformer").to(torch_device)
        inputs["hidden_states"] = restored_vae.encode(video).latent_dist.mode() * scale_factor
        restored = restored_vae.decode(restored_model(**inputs).sample / scale_factor, num_frames=frames).sample
        torch.testing.assert_close(restored, decoded, atol=0, rtol=0)

    @torch.no_grad()
    def test_three_chunk_cache_with_per_chunk_masks(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        latent = randn_tensor((2, 4, 6, 4, 6), generator=self.generator, device=torch_device)
        text = randn_tensor((2, 3, 8, 16), generator=self.generator, device=torch_device)
        mask = torch.tensor(
            [[[True, False] * 4, [False, True] * 4, [True] * 3 + [False] * 5]] * 2,
            device=torch_device,
        )
        timestep = torch.tensor([[0.9999, 0.5, 0.1], [0.9999, 0.7, 0.3]], device=torch_device)
        expected = model(latent, text, timestep, encoder_attention_mask=mask, use_cache=True)
        cache = None
        outputs = []
        for index in range(3):
            output = model(
                latent[:, :, index * 2 : (index + 1) * 2],
                text[:, index],
                timestep[:, index],
                encoder_attention_mask=mask[:, index],
                kv_cache=cache,
                use_cache=True,
            )
            cache = output.kv_cache
            assert len(cache) == model.config.num_layers
            assert all(key.shape == value.shape == (2, (index + 1) * 12, 1, 32) for key, value in cache)
            outputs.append(output.sample)
        torch.testing.assert_close(torch.cat(outputs, dim=2), expected.sample, atol=1e-5, rtol=1e-5)
        for actual, full in zip(cache, expected.kv_cache):
            torch.testing.assert_close(actual, full, atol=1e-5, rtol=1e-5)

    @torch.no_grad()
    def test_timestep_frequency_rounding(self):
        embedder = MagiTimestepEmbedding(16, 256).to(torch_device).eval()
        timesteps = torch.tensor([0.9999, 0.3, 0.7], device=torch_device)
        frequencies = torch.exp(-math.log(10000) * torch.arange(128).float() / 128).to(torch_device)
        angles = timesteps[:, None] * frequencies[None] * 1000
        expected = embedder.mlp(torch.cat([angles.cos(), angles.sin()], dim=-1).bfloat16())
        actual = embedder(timesteps, torch.bfloat16)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    @torch.no_grad()
    def test_attention_output_projection_uses_fp32(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device, dtype=torch.bfloat16).eval()
        projection = model.transformer_blocks[0].self_attention.linear_proj
        hidden_states = randn_tensor((1, 4, 128), generator=self.generator, device=torch_device).bfloat16()
        actual = projection(hidden_states)
        expected = torch.nn.functional.linear(hidden_states.float(), projection.weight.float())
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    @torch.no_grad()
    def test_expanded_caption_matches_contiguous_projection(self):
        torch.manual_seed(0)
        config = self.get_init_dict()
        config["caption_channels"] = 128
        model = self.model_class(**config).to(torch_device).eval()
        embedder = model.condition_embedder.y_embedder
        captions = torch.randn(1, 8, 128, device=torch_device)[:, None].expand(-1, 3, -1, -1)
        mask = torch.zeros(1, device=torch_device, dtype=torch.bool)
        expected = embedder(captions.contiguous(), mask)
        actual = embedder(captions, mask)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    @torch.no_grad()
    def test_caption_dropout_only_changes_adaptive_embedding(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        embedder = model.condition_embedder.y_embedder
        embedder.null_caption_embedding[-2].fill_(0.1)
        embedder.null_caption_embedding[-1].fill_(-0.1)
        captions = self.get_dummy_inputs()["encoder_hidden_states"][:, None]
        conditional, condition = embedder(captions, torch.zeros(2, device=torch_device, dtype=torch.bool))
        unconditional, uncondition = embedder(captions, torch.ones(2, device=torch_device, dtype=torch.bool))
        torch.testing.assert_close(conditional, unconditional, atol=0, rtol=0)
        assert not torch.allclose(condition, uncondition)

    def test_invalid_cache_and_shapes(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        inputs = self.get_dummy_inputs()
        with pytest.raises(ValueError, match="one key/value pair"):
            model(**inputs, kv_cache=())
        inputs["hidden_states"] = inputs["hidden_states"][:, :, :, :3]
        with pytest.raises(ValueError, match="divisible"):
            model(**inputs)

    @torch.no_grad()
    def test_chunk_causality(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        inputs = self.get_dummy_inputs()
        inputs["timestep"] = inputs["timestep"][:, None].expand(-1, 2)
        expected = model(**inputs).sample
        inputs["hidden_states"][:, :, 2:] += 10
        actual = model(**inputs).sample
        torch.testing.assert_close(actual[:, :, :2], expected[:, :, :2], atol=1e-5, rtol=0)
        assert not torch.allclose(actual[:, :, 2:], expected[:, :, 2:])

    @torch.no_grad()
    @pytest.mark.parametrize("cache_device", [None, "cpu"])
    def test_prefix_cache_matches_full_forward(self, cache_device):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        inputs = self.get_dummy_inputs()
        inputs["timestep"] = inputs["timestep"][:, None].expand(-1, 2)
        expected = model(**inputs, use_cache=True)
        prefix = model(
            hidden_states=inputs["hidden_states"][:, :, :2],
            timestep=inputs["timestep"][:, :1],
            encoder_hidden_states=inputs["encoder_hidden_states"],
            use_cache=True,
            cache_device=cache_device,
        )
        cached = tuple((key.clone(), value.clone()) for key, value in prefix.kv_cache)
        suffix = model(
            hidden_states=inputs["hidden_states"][:, :, 2:],
            timestep=inputs["timestep"][:, 1:],
            encoder_hidden_states=inputs["encoder_hidden_states"],
            kv_cache=prefix.kv_cache,
            use_cache=True,
        )
        torch.testing.assert_close(suffix.sample, expected.sample[:, :, 2:], atol=1e-5, rtol=1e-5)
        for (key, value), (expected_key, expected_value) in zip(suffix.kv_cache, expected.kv_cache):
            torch.testing.assert_close(key.to(expected_key), expected_key, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(value.to(expected_value), expected_value, atol=1e-5, rtol=1e-5)
        for actual, previous in zip(prefix.kv_cache, cached):
            torch.testing.assert_close(actual, previous, atol=0, rtol=0)

    @torch.no_grad()
    @pytest.mark.parametrize("cache_device", [None, "cpu"])
    @pytest.mark.parametrize("retained_tokens", [4, 8])
    def test_compact_cache_matches_full_forward(self, cache_device, retained_tokens):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        inputs = self.get_dummy_inputs()
        expected = model(**inputs, use_cache=True)
        actual = model(**inputs, use_cache=True, cache_token_count=retained_tokens, cache_device=cache_device)
        torch.testing.assert_close(actual.sample, expected.sample, atol=0, rtol=0)
        for full_pair, compact_pair in zip(expected.kv_cache, actual.kv_cache):
            for full, compact in zip(full_pair, compact_pair):
                torch.testing.assert_close(compact.to(full), full[:, :retained_tokens], atol=0, rtol=0)
                assert compact.untyped_storage().nbytes() == compact.numel() * compact.element_size()
                if cache_device == "cpu":
                    assert compact.device.type == "cpu"

    @pytest.mark.parametrize(
        "options",
        [
            {"cache_token_count": 4},
            {"cache_device": "cpu"},
            {"cache_token_count": 3, "use_cache": True},
            {"cache_token_count": 100, "use_cache": True},
            {"cache_token_count": True, "use_cache": True},
        ],
    )
    def test_invalid_cache_retention(self, options):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        with pytest.raises(ValueError):
            model(**self.get_dummy_inputs(), **options)

    @torch.no_grad()
    def test_text_padding_is_ignored(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        inputs = self.get_dummy_inputs()
        inputs["encoder_attention_mask"] = torch.tensor([[True] * 4 + [False] * 4] * 2, device=torch_device)
        expected = model(**inputs).sample
        inputs["encoder_hidden_states"][:, 4:] += 100
        torch.testing.assert_close(model(**inputs).sample, expected, atol=0, rtol=0)

    @torch.no_grad()
    def test_per_chunk_text_conditioning(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        inputs = self.get_dummy_inputs()
        inputs["timestep"] = inputs["timestep"][:, None].expand(-1, 2)
        inputs["encoder_hidden_states"] = inputs["encoder_hidden_states"][:, None].repeat(1, 2, 1, 1)
        expected = model(**inputs).sample
        inputs["encoder_hidden_states"][:, 1] += 1
        actual = model(**inputs).sample
        torch.testing.assert_close(actual[:, :, :2], expected[:, :, :2], atol=0, rtol=0)
        assert not torch.allclose(actual[:, :, 2:], expected[:, :, 2:])

    @torch.no_grad()
    def test_distillation_condition(self):
        config = self.get_init_dict()
        model = self.model_class(**config, distilled=True).to(torch_device).eval()
        inputs = self.get_dummy_inputs()
        with pytest.raises(ValueError, match="timestep_delta"):
            model(**inputs)
        first = model(**inputs, timestep_delta=torch.tensor(2.0, device=torch_device)).sample
        second = model(**inputs, timestep_delta=torch.tensor(4.0, device=torch_device)).sample
        assert first.shape == second.shape == inputs["hidden_states"].shape
        assert not torch.allclose(first, second)

    @torch.no_grad()
    def test_24b_variant(self):
        model = (
            self.model_class(
                **self.get_init_dict(), duplicate_channels=True, gated_linear_unit=True, x_rescale_factor=0.1
            )
            .to(torch_device)
            .eval()
        )
        inputs = self.get_dummy_inputs()
        output = model(**inputs).sample
        assert output.shape == inputs["hidden_states"].shape
        assert torch.isfinite(output).all()

    @torch.no_grad()
    def test_explicit_kv_ranges(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        inputs = self.get_dummy_inputs()
        inputs["timestep"] = inputs["timestep"][:, None].expand(-1, 2)
        expected = model(**inputs).sample
        actual = model(**inputs, kv_ranges=((0, 8), (0, 16))).sample
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        local = model(**inputs, kv_ranges=((0, 8), (8, 16))).sample
        assert not torch.allclose(local[:, :, 2:], expected[:, :, 2:])
        with pytest.raises(ValueError, match="range"):
            model(**inputs, kv_ranges=((0, 8), (8, 17)))


class TestMagiTransformerMemory(MagiTransformerTesterConfig, MemoryTesterMixin):
    pass


class TestMagiTransformerTorchCompile(MagiTransformerTesterConfig, TorchCompileTesterMixin):
    @pytest.fixture(autouse=True)
    def capture_packed_text_shapes(self):
        with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
            yield

    @property
    def different_shapes_for_compilation(self):
        return [(4, 4), (4, 8), (8, 8)]

    def get_dummy_inputs(self, height=4, width=4):
        inputs = super().get_dummy_inputs()
        inputs["hidden_states"] = randn_tensor((2, 4, 4, height, width), generator=self.generator, device=torch_device)
        inputs["encoder_attention_mask"] = torch.tensor([[True, False] * 4] * 2, device=torch_device)
        return inputs


class TestMagiTransformerTraining(MagiTransformerTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        super().test_gradient_checkpointing_is_applied(expected_set={"MagiTransformer3DModel"})


class TestMagiTransformerAttention(MagiTransformerTesterConfig, AttentionTesterMixin):
    @pytest.mark.skipif(
        not is_flash_attn_available() or torch_device != "cuda", reason="Requires CUDA FlashAttention."
    )
    @pytest.mark.parametrize("backend", ["flash", "flash_varlen"])
    @torch.no_grad()
    def test_flash_backend(self, tmp_path, backend):
        with attention_backend("native"):
            model = self.model_class(**self.get_init_dict()).eval()
            model.save_pretrained(tmp_path)
            model = self.model_class.from_pretrained(tmp_path, torch_dtype=torch.bfloat16).to(torch_device)
            model.set_attention_backend(backend)
            inputs = self.get_dummy_inputs()
            if backend == "flash_varlen":
                inputs["encoder_attention_mask"] = torch.tensor([[True, False] * 4] * 2, device=torch_device)
            expected = model(**inputs).sample
            assert expected.shape == inputs["hidden_states"].shape
            assert torch.isfinite(expected).all()
            if backend == "flash_varlen":
                inputs["encoder_hidden_states"][:, 1::2] += 100
                torch.testing.assert_close(model(**inputs).sample, expected, atol=0, rtol=0)
