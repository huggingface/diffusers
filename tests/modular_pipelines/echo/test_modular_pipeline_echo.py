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

from types import SimpleNamespace

import pytest
import torch

from diffusers.modular_pipelines import EchoBlocks, EchoModularPipeline
from diffusers.modular_pipelines.echo.echo_encoders import (
    EchoConditionEncoderStep,
    _max_response_window_start,
    _validate_memory_slot_count,
)

from ..testing_utils import (
    BaseModularPipelineTesterConfig,
    ModularLoadingTesterMixin,
    ModularMemoryTesterMixin,
    ModularPipelineTesterMixin,
    ModularWorkflowTesterMixin,
)


class EchoModularPipelineTesterConfig(BaseModularPipelineTesterConfig):
    pipeline_class = EchoModularPipeline
    pipeline_blocks_class = EchoBlocks
    pretrained_model_name_or_path = "Echo-Team/tiny-echo-modular-pipe"
    params = frozenset(
        [
            "prompt",
            "image",
            "memory_images",
            "memory_audio_waveforms",
            "height",
            "width",
            "num_frames",
            "frame_rate",
            "model_frame_rate",
            "sigmas",
        ]
    )
    batch_params = frozenset(["prompt"])
    optional_params = frozenset(["num_videos_per_prompt", "latents", "audio_latents", "output_type"])
    not_params = frozenset(["negative_prompt", "guidance_scale", "num_inference_steps"])
    expected_workflow_blocks = {}
    output_name = "videos"

    def get_dummy_inputs(self, seed=0):
        generator = self.get_generator(seed)
        return {
            "prompt": "a robot dancing",
            "image": torch.rand((1, 3, 32, 32), generator=generator),
            "memory_images": [
                torch.rand((1, 3, 32, 32), generator=generator),
                torch.rand((1, 3, 32, 32), generator=generator),
            ],
            "generator": self.get_generator(seed),
            "sigmas": [1.0, 0.0],
            "height": 32,
            "width": 32,
            "num_frames": 5,
            "frame_rate": 25.0,
            "max_sequence_length": 16,
            "output_type": "pt",
        }


class TestEchoModularPipelineFast(EchoModularPipelineTesterConfig, ModularPipelineTesterMixin):
    def test_max_response_window_start(self):
        mel = torch.zeros(1, 2, 2000, 64)
        mel[:, :, 1900:] = 10

        assert _max_response_window_start(mel, 963) == 1037

    def test_rejects_more_than_seven_memory_slots(self):
        with pytest.raises(ValueError, match="at most 7 memory slots"):
            _validate_memory_slot_count(8)

    def test_audio_memory_is_cropped_to_max_duration(self):
        pytest.importorskip("torchaudio")

        class FakeAudioVAE:
            dtype = torch.float32
            config = SimpleNamespace(sample_rate=16000, mel_hop_length=160, mel_bins=64)

            def encode(self, mel):
                self.encoded_mel = mel
                latent_dist = SimpleNamespace(mode=lambda: torch.zeros(1, 1, 1, 1))
                return SimpleNamespace(latent_dist=latent_dist)

        audio_vae = FakeAudioVAE()
        components = SimpleNamespace(
            audio_vae=audio_vae,
            audio_latents_mean=torch.zeros(1),
            audio_latents_std=torch.ones(1),
        )
        waveform = torch.zeros(2, 12 * 16000)
        waveform[:, -16000:] = torch.randn(2, 16000)

        EchoConditionEncoderStep._encode_audio(components, waveform, 16000, torch.device("cpu"))

        assert audio_vae.encoded_mel.shape[2] == 963

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
        token_count = latent_num_frames * latent_height * latent_width
        latents = torch.randn(1, token_count, pipe.transformer.config.in_channels)
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
