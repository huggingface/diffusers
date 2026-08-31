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

from diffusers import LTX2ImageToVideoPipeline
from diffusers.pipelines.ltx2 import LTX2LatentUpsamplePipeline
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

from ...testing_utils import assert_tensors_close, enable_full_determinism, torch_device
from ..testing_utils import PipelineTesterMixin
from .testing_utils import (
    LTX2BaseTesterConfig,
    LTX2LoraMemoryTesterMixin,
    LTX2LoraTesterMixin,
    LTX2MemoryTesterMixin,
)


enable_full_determinism()


class LTX2ImageToVideoPipelineTesterConfig(LTX2BaseTesterConfig):
    pipeline_class = LTX2ImageToVideoPipeline
    required_input_params_in_call_signature = frozenset(
        [
            "image",
            "prompt",
            "height",
            "width",
            "guidance_scale",
            "negative_prompt",
            "prompt_embeds",
            "negative_prompt_embeds",
        ]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt", "image"])

    def get_dummy_upsample_component(self, in_channels=4, mid_channels=32, num_blocks_per_stage=1):
        return LTX2LatentUpsamplerModel(
            in_channels=in_channels,
            mid_channels=mid_channels,
            num_blocks_per_stage=num_blocks_per_stage,
        )

    def get_dummy_inputs(self):
        generator = self.get_generator(0)
        image = torch.rand((1, 3, 32, 32), generator=generator)

        return {
            "image": image,
            "prompt": "a robot dancing",
            "negative_prompt": "",
            "generator": generator,
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
            # Synthetic float tensors skip H.264 CRF re-compression (training path uses PIL).
            "image_crf": 0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestLTX2ImageToVideoPipeline(LTX2ImageToVideoPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slices below are CPU-specific.
        pipe = self.get_pipeline()

        output = pipe(**self.get_dummy_inputs())
        video = output.frames
        audio = output.audio

        assert video.shape == (1, *self.output_shape)
        assert audio.shape[0] == 1
        assert audio.shape[1] == pipe.vocoder.config.out_channels

        # fmt: off
        expected_video_slice = torch.tensor(
            [
                0.3573, 0.8382, 0.3581, 0.6114, 0.3682, 0.7969, 0.2552, 0.6399, 0.3113, 0.1497, 0.3249, 0.5395, 0.3498, 0.4526, 0.4536, 0.4555
            ]
        )
        expected_audio_slice = torch.tensor(
            [
                0.0294, 0.0498, 0.1269, 0.1135, 0.1639, 0.1116, 0.1730, 0.0931, 0.0672, -0.0069, 0.0688, 0.0097, 0.0808, 0.1231, 0.0986, 0.0739
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
                0.2665, 0.6915, 0.2939, 0.6767, 0.2552, 0.6215, 0.1765, 0.6248, 0.2800, 0.2356, 0.3480, 0.5395, 0.3190, 0.4128, 0.4784, 0.4086
            ]
        )
        expected_audio_slice = torch.tensor(
            [
                0.0273, 0.0490, 0.1253, 0.1129, 0.1655, 0.1057, 0.1707, 0.0943, 0.0672, -0.0069, 0.0688, 0.0097, 0.0808, 0.1231, 0.0986, 0.0739
            ]
        )
        # fmt: on

        video = video.flatten()
        audio = audio.flatten()
        generated_video_slice = torch.cat([video[:8], video[-8:]])
        generated_audio_slice = torch.cat([audio[:8], audio[-8:]])

        assert_tensors_close(generated_video_slice, expected_video_slice, atol=1e-4, rtol=1e-4)
        assert_tensors_close(generated_audio_slice, expected_audio_slice, atol=1e-4, rtol=1e-4)

    def test_two_stages_inference_with_upsampler(self):
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

        upsampler = self.get_dummy_upsample_component(in_channels=video_latent.shape[1])
        upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=upsampler)
        upscaled_video_latent = upsample_pipe(latents=video_latent, output_type="latent", return_dict=False)[0]
        assert upscaled_video_latent.shape == (1, 4, 3, 32, 32)

        inputs["latents"] = upscaled_video_latent
        inputs["audio_latents"] = audio_latent
        inputs["output_type"] = "pt"
        second_stage_output = pipe(**inputs)
        video = second_stage_output.frames
        audio = second_stage_output.audio

        assert video.shape == (1, 5, 3, 64, 64)
        assert audio.shape[0] == 1
        assert audio.shape[1] == pipe.vocoder.config.out_channels

        # fmt: off
        expected_video_slice = torch.tensor(
            [
                0.4497, 0.6757, 0.4219, 0.7686, 0.4525, 0.6483, 0.3969, 0.7404, 0.3541, 0.3039, 0.4592, 0.3521, 0.3665, 0.2785, 0.3336, 0.3079
            ]
        )
        expected_audio_slice = torch.tensor(
            [
                0.0271, 0.0492, 0.1249, 0.1126, 0.1661, 0.1060, 0.1717, 0.0944, 0.0672, -0.0069, 0.0688, 0.0097, 0.0808, 0.1231, 0.0986, 0.0739
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
        inputs["output_type"] = "latent"
        latents = pipe(**inputs).frames

        legacy_latent_frames = (121 - 1) // pipe.vae_temporal_compression_ratio + 1
        assert latents.shape[2] != legacy_latent_frames, (
            "omitting num_frames with a head present must not use the legacy default"
        )

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


class TestLTX2ImageToVideoPipelineMemory(LTX2ImageToVideoPipelineTesterConfig, LTX2MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the LTX2 I2V pipeline."""


class TestLTX2ImageToVideoPipelineLoRA(LTX2ImageToVideoPipelineTesterConfig, LTX2LoraTesterMixin):
    """LoRA tests for the LTX2 I2V pipeline."""


class TestLTX2ImageToVideoPipelineLoRAMemory(LTX2ImageToVideoPipelineTesterConfig, LTX2LoraMemoryTesterMixin):
    """LoRA x memory-optimization tests (group offload, CPU offload) for the LTX2 I2V pipeline."""
