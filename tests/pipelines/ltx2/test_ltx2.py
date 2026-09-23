# Copyright 2025 The HuggingFace Team.
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

from diffusers import LTX2Pipeline

from ...testing_utils import assert_tensors_close, enable_full_determinism, torch_device
from ..testing_utils import PipelineTesterMixin
from .testing_utils import (
    LTX2BaseTesterConfig,
    LTX2LoraMemoryTesterMixin,
    LTX2LoraTesterMixin,
    LTX2MemoryTesterMixin,
)


enable_full_determinism()


class LTX2PipelineTesterConfig(LTX2BaseTesterConfig):
    pipeline_class = LTX2Pipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "guidance_scale", "negative_prompt", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt"])

    def get_dummy_inputs(self):
        return {
            "prompt": "a robot dancing",
            "negative_prompt": "",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            # Pin legacy sampling knobs so deterministic slice tests stay stable when
            # production defaults track LTX-2.3/2.5 (STG on, modality/rescale, cross-timestep).
            "stg_scale": 0.0,
            "modality_scale": 1.0,
            "guidance_rescale": 0.0,
            "audio_guidance_scale": 1.0,
            "audio_stg_scale": 0.0,
            "audio_modality_scale": 1.0,
            "audio_guidance_rescale": 0.0,
            "spatio_temporal_guidance_blocks": None,
            "use_cross_timestep": False,
            "height": 32,
            "width": 32,
            "num_frames": 5,
            "frame_rate": 25.0,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestLTX2Pipeline(LTX2PipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slices below are CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        output = pipe(**inputs)
        video = output.frames
        audio = output.audio

        assert video.shape == (1, *self.output_shape)
        assert audio.shape[0] == 1
        assert audio.shape[1] == pipe.vocoder.config.out_channels

        # fmt: off
        expected_video_slice = torch.tensor(
            [
                0.4331, 0.6203, 0.3245, 0.7294, 0.4822, 0.5703, 0.2999, 0.7700, 0.4961, 0.4242, 0.4581, 0.4351, 0.1137, 0.4437, 0.6304, 0.3184
            ]
        )
        expected_audio_slice = torch.tensor(
            [
                0.0263, 0.0528, 0.1217, 0.1104, 0.1632, 0.1072, 0.1789, 0.0949, 0.0672, -0.0069, 0.0688, 0.0097, 0.0808, 0.1231, 0.0986, 0.0739
            ]
        )
        # fmt: on

        video = video.flatten()
        audio = audio.flatten()
        generated_video_slice = torch.cat([video[:8], video[-8:]])
        generated_audio_slice = torch.cat([audio[:8], audio[-8:]])

        assert_tensors_close(generated_video_slice, expected_video_slice, atol=1e-4, rtol=1e-4)
        assert_tensors_close(generated_audio_slice, expected_audio_slice, atol=1e-4, rtol=1e-4)

    def test_two_stages_inference(self):
        # Run on CPU: the expected slices below are CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["output_type"] = "latent"
        first_stage_output = pipe(**inputs)
        video_latent = first_stage_output.frames
        audio_latent = first_stage_output.audio

        assert video_latent.shape == (1, 4, 3, 16, 16)
        assert audio_latent.shape == (1, 2, 5, 2)
        assert audio_latent.shape[1] == pipe.vocoder.config.out_channels

        inputs["latents"] = video_latent
        inputs["audio_latents"] = audio_latent
        inputs["output_type"] = "pt"
        second_stage_output = pipe(**inputs)
        video = second_stage_output.frames
        audio = second_stage_output.audio

        assert video.shape == (1, *self.output_shape)
        assert audio.shape[0] == 1
        assert audio.shape[1] == pipe.vocoder.config.out_channels

        # fmt: off
        expected_video_slice = torch.tensor(
            [
                0.5514, 0.5943, 0.4260, 0.5971, 0.4306, 0.6369, 0.3124, 0.6964, 0.5419, 0.2412, 0.3882, 0.4504, 0.1941, 0.3404, 0.6037, 0.2464
            ]
        )
        expected_audio_slice = torch.tensor(
            [
                0.0252, 0.0526, 0.1211, 0.1119, 0.1638, 0.1042, 0.1776, 0.0948, 0.0672, -0.0069, 0.0688, 0.0097, 0.0808, 0.1231, 0.0986, 0.0739
            ]
        )
        # fmt: on

        video = video.flatten()
        audio = audio.flatten()
        generated_video_slice = torch.cat([video[:8], video[-8:]])
        generated_audio_slice = torch.cat([audio[:8], audio[-8:]])

        assert_tensors_close(generated_video_slice, expected_video_slice, atol=1e-4, rtol=1e-4)
        assert_tensors_close(generated_audio_slice, expected_audio_slice, atol=1e-4, rtol=1e-4)

    def test_inference_batch_single_identical(self, batch_size=2, expected_max_diff=2e-2):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    def test_auto_duration_produces_a_grid_valid_frame_count(self):
        pipe = self.get_pipeline_with_duration_head()

        inputs = self.get_dummy_inputs()
        inputs.pop("num_frames")
        inputs["min_seconds"] = 0.5
        inputs["max_seconds"] = 2.0
        frames = pipe(**inputs).frames[0]

        ratio = pipe.vae_temporal_compression_ratio
        assert (len(frames) - 1) % ratio == 0
        assert 0 < len(frames) <= round(2.0 * inputs["frame_rate"])

    def test_omitting_num_frames_auto_predicts_when_a_head_is_present(self):
        pipe = self.get_pipeline_with_duration_head()

        inputs = self.get_dummy_inputs()
        inputs.pop("num_frames")
        inputs["min_seconds"] = 0.5
        inputs["max_seconds"] = 2.0
        bounded_frames = pipe(**inputs).frames[0]

        # Omitting `num_frames` entirely with default bounds must also take the auto path.
        inputs = self.get_dummy_inputs()
        inputs.pop("num_frames")
        default_frames = pipe(**inputs).frames[0]

        ratio = pipe.vae_temporal_compression_ratio
        assert (len(bounded_frames) - 1) % ratio == 0
        assert (len(default_frames) - 1) % ratio == 0
        assert len(default_frames) != 121, "omitting num_frames with a head present must not use the legacy default"

    def test_omitting_num_frames_uses_the_legacy_default_without_a_head(self):
        # Guards backwards compatibility: a pre-2.5 pipeline has no duration_head and must keep 121.
        components = self.get_dummy_components()
        assert components.get("duration_head") is None
        pipe = self.get_pipeline(**components).to(torch_device)

        inputs = self.get_dummy_inputs()
        inputs.pop("num_frames")
        # Decoding 121 frames is needlessly slow here; the latent frame count already pins num_frames down.
        inputs["output_type"] = "latent"
        latents = pipe(**inputs).frames

        # Latents come back unpacked as [batch, channels, latent_frames, height, width].
        expected_latent_frames = (121 - 1) // pipe.vae_temporal_compression_ratio + 1
        assert latents.shape[2] == expected_latent_frames

    def test_explicit_num_frames_wins_over_a_present_head(self):
        pipe = self.get_pipeline_with_duration_head()

        inputs = self.get_dummy_inputs()
        inputs["num_frames"] = 9
        frames = pipe(**inputs).frames[0]

        assert len(frames) == 9

    def test_auto_duration_with_multiple_prompts_raises(self):
        # The head predicts one duration, so it cannot serve prompts with different natural lengths.
        # Without this guard the pipeline silently applied the first prompt's length to all of them.
        pipe = self.get_pipeline_with_duration_head()

        inputs = self.get_dummy_inputs()
        inputs["prompt"] = ["a robot dancing", "a much longer and quite different scene"]
        inputs["negative_prompt"] = ["", ""]
        inputs.pop("num_frames")

        with pytest.raises(ValueError, match="2 prompts were supplied"):
            pipe(**inputs)

    def test_multiple_prompts_still_work_with_an_explicit_num_frames(self):
        # The guard must be scoped to the auto path -- batched prompts with an integer are unaffected.
        pipe = self.get_pipeline_with_duration_head()

        inputs = self.get_dummy_inputs()
        inputs["prompt"] = ["a robot dancing", "a much longer and quite different scene"]
        inputs["negative_prompt"] = ["", ""]
        inputs["num_frames"] = 5
        inputs["output_type"] = "latent"

        latents = pipe(**inputs).frames

        assert latents.shape[0] == 2

    def test_invalid_duration_bounds_raise(self):
        pipe = self.get_pipeline_with_duration_head()

        inputs = self.get_dummy_inputs()
        inputs.pop("num_frames")
        inputs["min_seconds"] = 5.0
        inputs["max_seconds"] = 2.0

        with pytest.raises(ValueError, match="min_seconds"):
            pipe(**inputs)


class TestLTX2PipelineMemory(LTX2PipelineTesterConfig, LTX2MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the LTX2 pipeline."""


class TestLTX2PipelineLoRA(LTX2PipelineTesterConfig, LTX2LoraTesterMixin):
    """LoRA tests for the LTX2 pipeline."""


class TestLTX2PipelineLoRAMemory(LTX2PipelineTesterConfig, LTX2LoraMemoryTesterMixin):
    """LoRA x memory-optimization tests (group offload, CPU offload) for the LTX2 pipeline."""
