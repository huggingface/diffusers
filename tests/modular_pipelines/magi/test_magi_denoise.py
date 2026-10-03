# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from diffusers import (
    MagiClassifierFreeGuidance,
    MagiDenoiseStep,
    MagiEulerScheduler,
    MagiTransformer3DModel,
)
from diffusers.modular_pipelines.magi.denoise import (
    MagiDenoiseLoop,
    MagiPrefixDenoiseLoop,
    MagiPrefixDenoiseStep,
    MagiPrepareDenoiseStep,
    MagiPreparePrefixStep,
)


class TestMagiDenoise:
    def make_pipeline(self, **model_kwargs):
        torch.manual_seed(0)
        config = {
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
        config.update(model_kwargs)
        pipe = MagiDenoiseStep().init_pipeline()
        pipe.update_components(
            transformer=MagiTransformer3DModel(**config).eval(),
            scheduler=MagiEulerScheduler(),
            guider=MagiClassifierFreeGuidance(),
        )
        pipe.load_components()
        return pipe

    def inputs(self, batch=1, chunks=3):
        generator = torch.Generator().manual_seed(12)
        return {
            "latents": torch.randn(batch, 4, chunks * 2, 4, 4, generator=generator),
            "prompt_embeds": torch.randn(batch, chunks, 3, 16, generator=generator),
            "prompt_attention_mask": torch.ones(batch, chunks, 3, dtype=torch.bool),
            "negative_prompt_embeds": torch.randn(batch, 3, 16, generator=generator),
            "negative_prompt_attention_mask": torch.tensor([[True, False, True]]).expand(batch, -1),
            "chunk_width": 2,
            "window_size": 2,
            "num_inference_steps": 4,
        }

    def make_prepare_pipeline(self, **model_kwargs):
        denoise = self.make_pipeline(**model_kwargs)
        pipe = MagiPrepareDenoiseStep().init_pipeline()
        pipe.update_components(transformer=denoise.transformer, scheduler=denoise.scheduler)
        pipe.load_components()
        return pipe

    def make_loop_pipeline(self, prefix=False, **model_kwargs):
        denoise = self.make_pipeline(**model_kwargs)
        pipe = (MagiPrefixDenoiseLoop() if prefix else MagiDenoiseLoop()).init_pipeline()
        pipe.update_components(
            transformer=denoise.transformer,
            scheduler=MagiEulerScheduler(),
            guider=denoise.guider,
        )
        pipe.load_components()
        return pipe

    def make_prefix_pipeline(self):
        denoise = self.make_pipeline(num_layers=1)
        pipe = MagiPrefixDenoiseStep().init_pipeline()
        pipe.update_components(
            transformer=denoise.transformer,
            scheduler=denoise.scheduler,
            guider=denoise.guider,
        )
        pipe.load_components()
        return pipe

    @pytest.mark.parametrize("prefix_chunks", [0, 1])
    def test_cpu_cache_preserves_denoising(self, prefix_chunks):
        pipe = self.make_pipeline()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in self.inputs(chunks=4).items()}
        if prefix_chunks:
            inputs["prefix_latents"] = inputs["latents"][:, :, :2].clone()
        expected = pipe(**inputs, output=["latents", "clean_kv_cache"])
        actual = pipe(**inputs, cache_device="cpu", output=["latents", "clean_kv_cache"])
        torch.testing.assert_close(actual["latents"], expected["latents"], atol=0, rtol=0)
        for full_pair, cpu_pair in zip(expected["clean_kv_cache"], actual["clean_kv_cache"]):
            for full, compact in zip(full_pair, cpu_pair):
                assert compact.device.type == "cpu"
                torch.testing.assert_close(compact.to(full), full, atol=0, rtol=0)
                assert compact.untyped_storage().nbytes() == compact.numel() * compact.element_size()

    def test_window_branches_and_clean_cache(self):
        pipe = self.make_pipeline()
        inputs = self.inputs()
        original = {k: v.clone() for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        calls = []

        def capture(module, args, kwargs):
            calls.append({k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()})

        projected_rows = []
        projection_handle = pipe.transformer.transformer_blocks[
            0
        ].self_attention.linear_kv_xattn.register_forward_pre_hook(
            lambda module, args: projected_rows.append(tuple(args[0].shape))
        )
        handle = pipe.transformer.register_forward_pre_hook(capture, with_kwargs=True)
        output = pipe(**inputs, output=["latents", "clean_kv_cache", "completed_chunks"])
        handle.remove()
        projection_handle.remove()
        assert [shape[0] for shape in projected_rows[:3]] == [3, 2, 2]
        assert all(len(shape) == 2 for shape in projected_rows)
        assert len(calls) == 24
        assert output["completed_chunks"] == [0, 1, 2]
        assert output["latents"].dtype == torch.float32
        assert not output["latents"].requires_grad
        assert [x["hidden_states"].shape[2] for x in calls[::3]] == [2, 2, 4, 4, 6, 4, 4, 2]
        assert [i for i, x in enumerate(calls) if x["use_cache"]] == [13, 19]
        for i in range(0, len(calls), 3):
            text, prefix, unconditional = calls[i : i + 3]
            assert not text["caption_dropout_mask"].any()
            assert prefix["caption_dropout_mask"].all()
            assert unconditional["caption_dropout_mask"].all()
            assert unconditional["caption_dropout_mask"].numel() == 1
            assert text["kv_cache"] is prefix["kv_cache"]
            assert unconditional["kv_cache"] is None
            assert unconditional["kv_ranges"] is None
            assert unconditional["hidden_states"].shape[2] == 2
            assert unconditional["timestep"].shape[1] == 1
        for name, value in original.items():
            torch.testing.assert_close(inputs[name], value, atol=0, rtol=0)
        with torch.no_grad():
            clean = pipe.transformer(
                output["latents"][:, :, :4],
                inputs["negative_prompt_embeds"],
                torch.full((1, 2), 0.9999),
                encoder_attention_mask=inputs["negative_prompt_attention_mask"],
                caption_dropout_mask=torch.ones(1, dtype=torch.bool),
                kv_ranges=((0, 8), (8, 16)),
                use_cache=True,
            ).kv_cache
        for actual_pair, expected_pair in zip(output["clean_kv_cache"], clean):
            for actual, expected in zip(actual_pair, expected_pair):
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
                assert actual.shape[1] == 16
                assert actual.untyped_storage().nbytes() == actual.numel() * actual.element_size()

    def test_pipeline_roundtrip(self, tmp_path):
        from diffusers import ModularPipeline

        pipe = self.make_pipeline()
        inputs = self.inputs()
        expected = pipe(**inputs, output="latents")
        pipe.save_pretrained(str(tmp_path), overwrite_modular_index=True)
        restored = ModularPipeline.from_pretrained(str(tmp_path))
        restored.load_components()
        actual = restored(**inputs, output="latents")
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_repeat_run_resets_state(self):
        pipe = self.make_pipeline()
        inputs = self.inputs()
        first = pipe(**inputs, output="latents")
        second = pipe(**inputs, output="latents")
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        assert pipe.scheduler.step_index is None

    def test_standalone_denoise_loop_matches_composed_step(self):
        inputs = self.inputs()
        expected = self.make_pipeline()(**inputs, output=["latents", "clean_kv_cache", "completed_chunks"])
        state = self.make_prepare_pipeline()(**inputs)
        loop = self.make_loop_pipeline()
        actual = loop(state=state, output=["latents", "clean_kv_cache", "completed_chunks"])
        torch.testing.assert_close(actual["latents"], expected["latents"], atol=0, rtol=0)
        for actual_pair, expected_pair in zip(actual["clean_kv_cache"], expected["clean_kv_cache"]):
            for actual_tensor, expected_tensor in zip(actual_pair, expected_pair):
                torch.testing.assert_close(actual_tensor, expected_tensor, atol=0, rtol=0)
        assert actual["completed_chunks"] == expected["completed_chunks"]

        with pytest.raises(
            ValueError, match="timestep_schedule must match the configured scheduler and num_inference_steps"
        ):
            loop(state=state, timestep_schedule=state.get("timestep_schedule") + 0.1, output="latents")

    def test_standalone_prefix_denoise_loop_matches_composed_step(self):
        inputs = self.inputs(chunks=4)
        inputs.update(num_inference_steps=2, noise2clean_kvrange=(2, 1))
        conditioning_latents = torch.randn(1, 4, 3, 4, 4, generator=torch.Generator().manual_seed(13))
        expected = self.make_prefix_pipeline()(
            **inputs,
            conditioning_latents=conditioning_latents,
            output=["latents", "clean_kv_cache", "completed_chunks"],
        )

        prefix_prepare = MagiPreparePrefixStep().init_pipeline()
        prefix_prepare.load_components()
        state = prefix_prepare(
            latents=inputs["latents"],
            conditioning_latents=conditioning_latents,
            chunk_width=inputs["chunk_width"],
        )
        state = self.make_prepare_pipeline(num_layers=1)(state=state, **inputs)
        actual = self.make_loop_pipeline(prefix=True, num_layers=1)(
            state=state, output=["latents", "clean_kv_cache", "completed_chunks"]
        )

        torch.testing.assert_close(actual["latents"], expected["latents"], atol=0, rtol=0)
        assert actual["completed_chunks"] == expected["completed_chunks"]
        for actual_pair, expected_pair in zip(actual["clean_kv_cache"], expected["clean_kv_cache"]):
            for actual_tensor, expected_tensor in zip(actual_pair, expected_pair):
                torch.testing.assert_close(actual_tensor, expected_tensor, atol=0, rtol=0)

    def test_full_chunk_prefix(self):
        pipe = self.make_pipeline()
        inputs = self.inputs(chunks=4)
        inputs["prefix_latents"] = torch.randn(1, 4, 4, 4, 4)
        prefix = inputs["prefix_latents"].clone()
        output = pipe(**inputs, output=["latents", "clean_kv_cache", "completed_chunks"])
        torch.testing.assert_close(output["latents"][:, :, :4], prefix, atol=0, rtol=0)
        torch.testing.assert_close(inputs["prefix_latents"], prefix, atol=0, rtol=0)
        assert output["completed_chunks"] == [0, 1, 2, 3]
        assert output["clean_kv_cache"][0][0].shape[1] == 24

    def test_clean_chunk_kvrange_fallback(self):
        pipe = self.make_pipeline()
        inputs = self.inputs(chunks=4)
        inputs["prefix_latents"] = inputs["latents"][:, :, :2].clone()
        expected = pipe(**inputs, clean_chunk_kvrange=2, output=["latents", "clean_kv_cache"])
        actual = pipe(**inputs, clean_chunk_kvrange=-1, output=["latents", "clean_kv_cache"])
        torch.testing.assert_close(actual["latents"], expected["latents"], atol=0, rtol=0)
        for actual_pair, expected_pair in zip(actual["clean_kv_cache"], expected["clean_kv_cache"]):
            for actual_tensor, expected_tensor in zip(actual_pair, expected_pair):
                torch.testing.assert_close(actual_tensor, expected_tensor, atol=0, rtol=0)

    @pytest.mark.parametrize("prefix_length", [1, 3])
    def test_partial_prefix_branch_inputs_and_clean_cache_refresh(self, prefix_length):
        pipe = self.make_prefix_pipeline()
        inputs = self.inputs(chunks=4)
        inputs.update(num_inference_steps=2, noise2clean_kvrange=(2, 1))
        prefix = torch.randn(1, 4, prefix_length, 4, 4, generator=torch.Generator().manual_seed(13)) + 10
        calls = []

        def capture(module, args, kwargs):
            calls.append(
                {
                    "hidden_states": kwargs["hidden_states"].detach().clone(),
                    "use_cache": kwargs.get("use_cache", False),
                    "cache_token_count": kwargs.get("cache_token_count"),
                }
            )

        handle = pipe.transformer.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            output = pipe(
                **inputs,
                conditioning_latents=prefix,
                output=["latents", "clean_kv_cache"],
            )
        finally:
            handle.remove()

        chunk_width = inputs["chunk_width"]
        full_length = prefix_length // chunk_width * chunk_width
        partial_prefix = prefix[:, :, full_length:]
        initial_calls = int(full_length > 0)
        if initial_calls:
            torch.testing.assert_close(calls[0]["hidden_states"], prefix[:, :, :full_length], atol=0, rtol=0)
        for call in calls[initial_calls : initial_calls + 3]:
            torch.testing.assert_close(
                call["hidden_states"][:1, :, : partial_prefix.shape[2]], partial_prefix, atol=0, rtol=0
            )

        refresh_prefix_index = next(
            index for index, call in enumerate(calls) if call["use_cache"] and call["cache_token_count"] is not None
        )
        refresh_text = calls[refresh_prefix_index - 1]
        refresh_prefix = calls[refresh_prefix_index]
        refresh_independent = calls[refresh_prefix_index + 1]
        for call in (refresh_text, refresh_prefix):
            torch.testing.assert_close(
                call["hidden_states"][:1, :, : partial_prefix.shape[2]], partial_prefix, atol=0, rtol=0
            )
        assert not refresh_text["use_cache"]
        assert refresh_prefix["use_cache"]
        assert not refresh_independent["use_cache"]

        pt, ph, pw = pipe.transformer.config.patch_size
        chunk_tokens = (chunk_width // pt) * (inputs["latents"].shape[3] // ph) * (inputs["latents"].shape[4] // pw)
        assert refresh_prefix["cache_token_count"] == (full_length // chunk_width + 1) * chunk_tokens
        active = refresh_text["hidden_states"][:, :, chunk_width:]
        count = active.shape[2] // chunk_width
        expected_independent = active.reshape(1, 4, count, chunk_width, 4, 4)
        expected_independent = expected_independent.permute(0, 2, 1, 3, 4, 5).reshape(count, 4, chunk_width, 4, 4)
        torch.testing.assert_close(refresh_independent["hidden_states"], expected_independent, atol=0, rtol=0)

        if full_length:
            torch.testing.assert_close(
                output["latents"][:, :, :full_length], prefix[:, :, :full_length], atol=0, rtol=0
            )
        assert not torch.equal(output["latents"][:, :, full_length:prefix_length], partial_prefix)

        cached_length = inputs["latents"].shape[2] - chunk_width
        cache_source = output["latents"][:, :, :cached_length].clone()
        cache_source[:, :, full_length:prefix_length] = partial_prefix
        cached_chunks = cached_length // chunk_width
        with torch.no_grad():
            expected_cache = pipe.transformer(
                hidden_states=cache_source,
                encoder_hidden_states=inputs["negative_prompt_embeds"],
                encoder_attention_mask=inputs["negative_prompt_attention_mask"],
                caption_dropout_mask=torch.ones(1, dtype=torch.bool),
                timestep=torch.full((1, cached_chunks), 0.9999),
                kv_ranges=tuple((index * chunk_tokens, (index + 1) * chunk_tokens) for index in range(cached_chunks)),
                use_cache=True,
            ).kv_cache
        for actual_pair, expected_pair in zip(output["clean_kv_cache"], expected_cache):
            for actual, expected in zip(actual_pair, expected_pair):
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_batch_matches_individual(self):
        pipe = self.make_pipeline()
        inputs = self.inputs(batch=2)
        batched = pipe(**inputs, output="latents")
        for i in range(2):
            single = {k: v[i : i + 1] if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            expected = pipe(**single, output="latents")
            torch.testing.assert_close(batched[i : i + 1], expected, atol=2e-5, rtol=2e-5)

    def test_single_chunk_and_window_larger_than_video(self):
        pipe = self.make_pipeline()
        inputs = self.inputs(chunks=1)
        inputs["window_size"] = 4
        output = pipe(**inputs, output=["latents", "clean_kv_cache", "completed_chunks"])
        assert output["clean_kv_cache"] is None
        assert output["completed_chunks"] == [0]

    def test_shared_text_matches_per_chunk(self):
        pipe = self.make_pipeline()
        inputs = self.inputs()
        inputs["prompt_embeds"] = inputs["prompt_embeds"][:, 0]
        inputs["prompt_attention_mask"] = inputs["prompt_attention_mask"][:, 0]
        expected = pipe(**inputs, output="latents")
        inputs["prompt_embeds"] = inputs["prompt_embeds"][:, None].expand(-1, 3, -1, -1)
        inputs["prompt_attention_mask"] = inputs["prompt_attention_mask"][:, None].expand(-1, 3, -1)
        actual = pipe(**inputs, output="latents")
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_invalid_inputs(self):
        pipe = self.make_pipeline()
        cases = [
            {"num_inference_steps": 6},
            {"window_size": 0},
            {"noise2clean_kvrange": ()},
            {"clean_t": 1.1},
            {"chunk_width": 4},
            {"prefix_latents": torch.zeros(1, 4, 1, 4, 4)},
            {"prefix_latents": torch.zeros(1, 4, 6, 4, 4)},
            {"negative_prompt_attention_mask": torch.zeros(1, 3, dtype=torch.bool)},
        ]
        for changed in cases:
            with pytest.raises(ValueError):
                pipe(**(self.inputs() | changed), output="latents")
        with pytest.raises(ValueError, match="base models only"):
            self.make_pipeline(distilled=True)(**self.inputs())

    def test_scalar_parameter_validation(self):
        pipe = self.make_pipeline()
        cases = [
            ({"noise2clean_kvrange": 5}, "noise2clean_kvrange must be a nonempty sequence of positive integers"),
            (
                {"noise2clean_kvrange": (True, 4, 3, 2)},
                "noise2clean_kvrange must be a nonempty sequence of positive integers",
            ),
            ({"clean_chunk_kvrange": True}, "clean_chunk_kvrange must be a positive integer or -1"),
            ({"clean_chunk_kvrange": 0}, "clean_chunk_kvrange must be a positive integer or -1"),
            ({"clean_t": True}, r"clean_t must be a finite number in \[0, 1\]"),
            ({"clean_t": float("nan")}, r"clean_t must be a finite number in \[0, 1\]"),
            ({"clean_t": float("inf")}, r"clean_t must be a finite number in \[0, 1\]"),
        ]
        for changed, message in cases:
            with pytest.raises(ValueError, match=message):
                pipe(**(self.inputs() | changed), output="latents")

    def test_standalone_prepare_denoise_input_validation(self):
        pipe = self.make_prepare_pipeline()
        inputs = self.inputs()
        latent_message = "latents must be a nonempty finite floating-point tensor"
        for latents in [
            [[[[[0.0]]]]],
            torch.zeros(1, 4, 6, 4, 4, dtype=torch.int64),
            torch.full((1, 4, 6, 4, 4), float("nan")),
        ]:
            with pytest.raises(ValueError, match=latent_message):
                pipe(**(inputs | {"latents": latents}), output="latents")

        prefix_message = "prefix_latents must be a finite floating-point tensor"
        for prefix in [
            [[[[[0.0]]]]],
            torch.zeros(1, 4, 2, 4, 4, dtype=torch.int64),
            torch.full((1, 4, 2, 4, 4), float("nan")),
        ]:
            with pytest.raises(ValueError, match=prefix_message):
                pipe(**inputs, prefix_latents=prefix, output="latents")

    def test_standalone_prepare_prefix(self):
        pipe = MagiPreparePrefixStep().init_pipeline()
        pipe.load_components()
        latents = torch.randn(1, 4, 6, 4, 4)
        prefix = torch.randn(1, 4, 3, 4, 4)
        actual = pipe(latents=latents, conditioning_latents=prefix, chunk_width=2, output="prefix_latents")
        torch.testing.assert_close(actual, prefix[:, :, :2], atol=0, rtol=0)
        assert (
            pipe(latents=latents, conditioning_latents=prefix[:, :, :1], chunk_width=2, output="prefix_latents")
            is None
        )

        message = "latents must be a nonempty finite floating-point tensor"
        for invalid_latents in [
            [[[[[0.0]]]]],
            torch.zeros(1, 4, 6, 4, 4, dtype=torch.int64),
            torch.full((1, 4, 6, 4, 4), float("nan")),
        ]:
            with pytest.raises(ValueError, match=message):
                pipe(
                    latents=invalid_latents,
                    conditioning_latents=prefix,
                    chunk_width=2,
                    output="prefix_latents",
                )


class TestMagiGuidance:
    def test_formula_and_thresholds(self):
        guider = MagiClassifierFreeGuidance(prefix_scales=(1, 2, 3, 4, 5), text_scales=(10, 20, 30, 40, 50))
        times = torch.tensor([[0.0, 0.0217 - 2e-7, 0.0217, 0.1, 0.3, 0.999]])
        guider.set_state(step=0, num_inference_steps=64, timestep=times)
        cond, prefix, uncond = [torch.full((1, 4, 12, 2, 2), x) for x in (3.0, 2.0, 1.0)]
        actual = guider.forward(cond, prefix, uncond).pred
        expected = (
            torch.tensor([12.0, 12.0, 23.0, 34.0, 45.0, 56.0])
            .repeat_interleave(2)[None, None, :, None, None]
            .expand_as(actual)
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_config_roundtrip(self, tmp_path):
        guider = MagiClassifierFreeGuidance(text_scales=(4, 3, 2, 1, 0), enabled=False)
        guider.save_pretrained(str(tmp_path))
        loaded = MagiClassifierFreeGuidance.from_pretrained(str(tmp_path))
        assert tuple(loaded.config.text_scales) == (4, 3, 2, 1, 0)
        assert not loaded.get_state()["enabled"]
        assert loaded.num_conditions == 3

    def test_invalid_prediction_shapes(self):
        guider = MagiClassifierFreeGuidance()
        guider.set_state(step=0, num_inference_steps=64, timestep=torch.tensor([[0.1]]))
        prediction = torch.zeros(1, 4, 2, 2, 2)
        with pytest.raises(ValueError, match="same five-dimensional shape"):
            guider.forward(prediction, prediction[:, :, :1], prediction)
        with pytest.raises(ValueError, match="same five-dimensional shape"):
            guider.forward(prediction[..., 0], prediction[..., 0], prediction[..., 0])

    @pytest.mark.parametrize("timestep", [-0.1, 1.1, float("nan"), float("inf")])
    def test_invalid_guidance_timesteps(self, timestep):
        guider = MagiClassifierFreeGuidance()
        guider.set_state(step=0, num_inference_steps=64, timestep=torch.tensor([[timestep]]))
        prediction = torch.zeros(1, 4, 2, 2, 2)
        with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
            guider.forward(prediction, prediction, prediction)

    def test_invalid_configuration(self):
        for kwargs, message in (
            ({"timestep_thresholds": 0}, "same nonzero length"),
            ({"prefix_scales": (1,)}, "same nonzero length"),
            (
                {"prefix_scales": (True, 1.5, 1.5, 1.0, 1.0)},
                "must contain finite real numbers",
            ),
            (
                {"text_scales": (7.5, 7.5, "0.0", 0.0, 0.0)},
                "must contain finite real numbers",
            ),
            ({"timestep_thresholds": (0, 0.1, 0.1, 0.3, 1)}, r"stay in \[0, 1\]"),
            ({"timestep_thresholds": (-1, 0.1, 0.2, 0.3, 1)}, r"stay in \[0, 1\]"),
            ({"timestep_thresholds": (0, 0.1, 0.2, 0.3, 1.1)}, r"stay in \[0, 1\]"),
        ):
            with pytest.raises(ValueError, match=message):
                MagiClassifierFreeGuidance(**kwargs)
