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

import pytest
import torch

from ..testing_utils import (
    ModularLoadingTesterMixin,
    ModularMemoryTesterMixin,
    ModularPipelineTesterMixin,
    ModularWorkflowTesterMixin,
)
from .testing_utils import EchoModularPipelineTesterConfig


class TestEchoModularPipelineFast(EchoModularPipelineTesterConfig, ModularPipelineTesterMixin):
    def test_multi_prompt_multi_video(self):
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        inputs["prompt"] = ["a robot dancing", "a cat walking"]
        inputs["num_videos_per_prompt"] = 2
        output = pipe(**inputs, output=["videos", "audio"])
        assert output["videos"].shape == (4, 5, 3, 32, 32)
        assert output["audio"].shape[0] == 4
        assert torch.isfinite(output["videos"]).all()
        assert torch.isfinite(output["audio"]).all()

    def test_denoise_returns_initial_vae_form_at_zero_sigma(self):
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        inputs.pop("image")
        inputs.pop("memory_images")
        inputs["sigmas"] = [0.0, 0.0]
        inputs["latents"] = torch.randn(1, 4, 3, 16, 16)
        inputs["audio_latents"] = torch.randn(1, 2, 5, 2)
        output = pipe(**inputs, output=["latents", "audio_latents"])
        torch.testing.assert_close(output["latents"], inputs["latents"], rtol=0, atol=0)
        torch.testing.assert_close(output["audio_latents"], inputs["audio_latents"], rtol=0, atol=0)

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=1e-3)

    def test_video_and_audio_outputs(self):
        pipe = self.get_pipeline()
        output = pipe(**self.get_dummy_inputs(), output=["videos", "audio"])

        assert output["videos"].shape == (1, 5, 3, 32, 32)
        assert output["audio"].shape[0] == 1
        assert output["audio"].shape[1] == pipe.vocoder.config.out_channels
        assert torch.isfinite(output["videos"]).all()
        assert torch.isfinite(output["audio"]).all()

    def test_seeded_dmd_renoising_is_deterministic(self):
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs(seed=1)
        inputs["output_type"] = "latent"
        first = pipe(**inputs, output=["videos", "audio"])

        inputs = self.get_dummy_inputs(seed=1)
        inputs["output_type"] = "latent"
        second = pipe(**inputs, output=["videos", "audio"])

        assert torch.equal(first["videos"], second["videos"])
        assert torch.equal(first["audio"], second["audio"])

    def test_audio_decoder_supports_mixed_precision(self):
        pipe = self.get_pipeline()
        pipe.vocoder.to(dtype=torch.float64)
        output = pipe(**self.get_dummy_inputs(), output=["videos", "audio"])

        assert output["audio"].dtype == torch.float64
        assert torch.isfinite(output["audio"]).all()

    def test_user_latents_are_not_modified(self):
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        inputs["output_type"] = "latent"
        latent_num_frames = 1 + (inputs["num_frames"] - 1) // pipe.vae_temporal_compression_ratio
        latent_height = inputs["height"] // pipe.vae_spatial_compression_ratio
        latent_width = inputs["width"] // pipe.vae_spatial_compression_ratio
        latents = torch.randn(1, pipe.vae.config.latent_channels, latent_num_frames, latent_height, latent_width)
        original = latents.clone()
        inputs["latents"] = latents

        pipe(**inputs, output=["videos", "audio"])

        assert torch.equal(latents, original)

    @pytest.mark.parametrize("waveform_lengths", [(3200, None), (3200, 4800), (1, 512)])
    def test_raw_audio_memory(self, waveform_lengths):
        pytest.importorskip("torchaudio")
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        inputs["memory_audio_waveforms"] = [
            None if length is None else torch.rand(2, length) for length in waveform_lengths
        ]
        inputs["memory_audio_sample_rates"] = 16000
        inputs["output_type"] = "latent"

        output = pipe(**inputs, output=["videos", "audio"])

        assert torch.isfinite(output["videos"]).all()
        assert torch.isfinite(output["audio"]).all()

    @pytest.mark.parametrize(
        ("sigmas", "message"),
        [
            ([1.0], "at least two values"),
            ([1.0, 0.5], "end at 0"),
            ([0.5, 1.0, 0.0], "monotonically non-increasing"),
        ],
    )
    def test_invalid_dmd_schedules(self, sigmas, message):
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        inputs["sigmas"] = sigmas

        with pytest.raises(ValueError, match=message):
            pipe(**inputs)


class TestEchoModularPipelineLoading(EchoModularPipelineTesterConfig, ModularLoadingTesterMixin):
    pass


class TestEchoModularPipelineWorkflow(EchoModularPipelineTesterConfig, ModularWorkflowTesterMixin):
    pass


class TestEchoModularPipelineMemory(EchoModularPipelineTesterConfig, ModularMemoryTesterMixin):
    pass
