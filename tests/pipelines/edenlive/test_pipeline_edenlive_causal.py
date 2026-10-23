# Copyright 2026 The VAST-AI Team. All rights reserved.
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

import numpy as np
import pytest
import torch

from diffusers import (
    EdenLiveCausalPipeline,
    EdenLiveFlowMatchEulerDiscreteScheduler,
)

from ...testing_utils import torch_device
from ..testing_utils import MemoryTesterMixin, PipelineTesterMixin
from ..testing_utils.common import cast_pipeline_to_dtype
from .test_pipeline_edenlive import (
    EdenLive14BRequiredComponentsTesterMixin,
    EdenLiveOptionalDecoderTesterMixin,
    EdenLivePipelineTesterConfig,
)


class EdenLiveCausalPipelineTesterConfig(EdenLivePipelineTesterConfig):
    pipeline_class = EdenLiveCausalPipeline

    def get_dummy_components(self):
        components = super().get_dummy_components()
        components["scheduler"] = EdenLiveFlowMatchEulerDiscreteScheduler(
            stochastic_sampling=True, default_timesteps=[1000, 500]
        )
        return components


class TestEdenLiveCausalPipeline(
    EdenLiveOptionalDecoderTesterMixin, EdenLiveCausalPipelineTesterConfig, PipelineTesterMixin
):
    def test_load_causal_pipeline_from_root(self, tmp_path):
        pipe = self.get_pipeline().to(torch_device)
        expected = pipe(**self.get_dummy_inputs()).frames
        pipe.save_pretrained(tmp_path)
        restored = self.pipeline_class.from_pretrained(tmp_path).to(torch_device)
        assert restored.decoder is None
        assert isinstance(restored.scheduler, EdenLiveFlowMatchEulerDiscreteScheduler)
        torch.testing.assert_close(restored(**self.get_dummy_inputs()).frames, expected, atol=1e-5, rtol=0)

    def test_short_tail_and_no_state_leak(self):
        pipe = self.get_pipeline().to(torch_device)

        def inputs():
            result = self.get_dummy_inputs()
            result["video"] = np.concatenate([result["video"], result["video"][:4]])
            result["num_frames"] = 13
            result["chunk_prompts"] = ["snow", "fire"]
            return result

        expected = pipe(**inputs()).frames
        actual = pipe(**inputs()).frames
        assert actual.shape == (1, 13, 3, 16, 16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert not any("cache" in name for name in pipe.transformer.state_dict())

    @pytest.mark.parametrize("guidance", [1.0, 3.0])
    def test_chunk_prompt_changes_tail(self, guidance):
        pipe = self.get_pipeline().to(torch_device)

        def inputs(tail):
            result = self.get_dummy_inputs()
            result["video"] = np.concatenate([result["video"], result["video"][:4]])
            result.update(num_frames=13, chunk_prompts=["snow", tail], output_type="latent", guidance_scale=guidance)
            return result

        first = pipe(**inputs("fire")).frames
        second = pipe(**inputs("ocean")).frames
        torch.testing.assert_close(first[:, :, :3], second[:, :, :3], atol=0, rtol=0)
        assert (first[:, :, 3:] - second[:, :, 3:]).abs().max() > 1e-6

    def test_invalid_chunk_prompts(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["chunk_prompts"] = ["one", "too many"]
        before = inputs["generator"].get_state().clone()
        with pytest.raises(ValueError, match="chunk_prompts"):
            pipe(**inputs)
        assert torch.equal(before, inputs["generator"].get_state())

    def test_chunk_prompts_reject_batched_prompt(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs.update(prompt=["snow", "fire"], chunk_prompts=["ocean"])
        with pytest.raises(ValueError, match="batch size one"):
            pipe(**inputs)

    @pytest.mark.parametrize("guidance", [1.0, 3.0])
    def test_chunk_prompts_encode_unique_texts_individually(self, guidance):
        fixture = self
        pipe = fixture.get_pipeline().to(torch_device)
        encoded, used = [], []

        def record_text(module, args, kwargs):
            encoded.append((args[0].clone(), kwargs["attention_mask"].clone()))

        def record_dit(module, args, kwargs):
            if kwargs.get("cache_update"):
                used.append(args[2].clone())

        pipe.text_encoder.register_forward_pre_hook(record_text, with_kwargs=True)
        pipe.transformer.register_forward_pre_hook(record_dit, with_kwargs=True)
        inputs = fixture.get_dummy_inputs()
        inputs.update(
            video=np.tile(inputs["video"], (4, 1, 1, 1))[:29],
            num_frames=29,
            chunk_prompts=["snow", "fire", "snow"],
            guidance_scale=guidance,
            output_type="latent",
        )
        pipe(**inputs)
        expected_texts = ["snow", "fire"] + ([inputs["negative_prompt"]] if guidance > 1 else [])
        assert len(encoded) == len(expected_texts)
        for (ids, mask), text in zip(encoded, expected_texts):
            tokens = pipe.tokenizer(
                [text],
                padding="max_length",
                max_length=inputs["max_sequence_length"],
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            torch.testing.assert_close(ids.cpu(), tokens.input_ids)
            torch.testing.assert_close(mask.cpu(), tokens.attention_mask)
        conditional = used[::2] if guidance > 1 else used
        assert len(conditional) == 3
        torch.testing.assert_close(conditional[0], conditional[2], rtol=0, atol=0)
        assert not torch.equal(conditional[0], conditional[1])

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_clean_commit_timestep_shape_and_cfg(self, dtype):
        fixture = self
        pipe = cast_pipeline_to_dtype(fixture.get_pipeline().to(torch_device), dtype)
        calls = []

        def capture(module, args, kwargs):
            calls.append((args[1].clone(), kwargs.get("cache_update", False)))

        pipe.transformer.register_forward_pre_hook(capture, with_kwargs=True)
        inputs = fixture.get_dummy_inputs()
        inputs.update(
            video=np.concatenate([inputs["video"], inputs["video"][:4]]),
            num_frames=13,
            output_type="latent",
            guidance_scale=3.0,
        )
        pipe(**inputs)
        commits = [time for time, commit in calls if commit]
        assert len(commits) == 4
        assert all(time.dtype == torch.float32 and not time.any() for time in commits)
        assert all(time.shape == (1,) for time in commits)
        noisy = calls[0][0]
        assert noisy.shape == (1, 3) and noisy[0, 0] == 0 and (noisy[0, 1:] > 0).all()

    @pytest.mark.parametrize("guidance", [1.0, 3.0])
    def test_repeated_chunk_prompt_matches_static(self, guidance):
        pipe = self.get_pipeline().to(torch_device)

        def inputs():
            result = self.get_dummy_inputs()
            result.update(
                video=np.tile(result["video"], (2, 1, 1, 1))[:13],
                num_frames=13,
                prompt="snow",
                guidance_scale=guidance,
                output_type="latent",
            )
            return result

        expected = pipe(**inputs()).frames
        actual = pipe(**inputs(), chunk_prompts=["snow", "snow"]).frames
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_next_call_is_independent_after_failure(self):
        pipe = self.get_pipeline().to(torch_device)

        def inputs():
            result = self.get_dummy_inputs()
            result.update(
                video=np.tile(result["video"], (2, 1, 1, 1))[:13],
                num_frames=13,
                chunk_prompts=["snow", "fire"],
                guidance_scale=3.0,
                output_type="latent",
            )
            return result

        expected = pipe(**inputs()).frames
        states = []

        def fail_in_second_chunk(module, args, kwargs):
            state = kwargs["causal_state"]
            if not any(state is seen for seen in states):
                states.append(state)
            if state.next_frame >= 3:
                raise RuntimeError("injected model failure")

        handle = pipe.transformer.register_forward_pre_hook(fail_in_second_chunk, with_kwargs=True)
        try:
            with pytest.raises(RuntimeError, match="injected model failure"):
                pipe(**inputs())
        finally:
            handle.remove()
        assert len(states) == 2
        actual = pipe(**inputs()).frames
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


class TestEdenLive14BCausalPipeline(EdenLive14BRequiredComponentsTesterMixin, TestEdenLiveCausalPipeline):
    profile = "14b"


class TestEdenLiveCausalPipelineMemory(EdenLiveCausalPipelineTesterConfig, MemoryTesterMixin):
    pass


class TestEdenLive14BCausalPipelineMemory(EdenLiveCausalPipelineTesterConfig, MemoryTesterMixin):
    profile = "14b"
