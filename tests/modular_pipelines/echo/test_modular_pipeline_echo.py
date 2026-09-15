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
from diffusers.modular_pipelines.echo.before_denoise import (
    EchoInputsStep,
    EchoPrepareConditioningStep,
    _pack_audio_latents,
)
from diffusers.modular_pipelines.echo.decoders import _denormalize_audio_latents
from diffusers.modular_pipelines.echo.encoders import (
    EchoVaeEncoderStep,
    _encode_audio,
    _normalize_audio_latents,
    _validate_memory_slot_count,
)
from diffusers.modular_pipelines.echo.modular_blocks_echo import EchoDecoderStep

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
    def test_rejects_more_than_seven_memory_slots(self):
        with pytest.raises(ValueError, match="at most 7 memory slots"):
            _validate_memory_slot_count(8)

    def test_vae_encoder_runs_without_transformer(self):
        pytest.importorskip("torchaudio")
        pipe = EchoVaeEncoderStep().init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components(dtype=torch.float32)
        output = pipe(
            image=torch.rand(1, 3, 32, 32),
            memory_images=[torch.rand(1, 3, 32, 32)],
            memory_audio_waveforms=[torch.rand(2, 3200)],
            memory_audio_sample_rates=16000,
            height=32,
            width=32,
            output=["first_frame_latents", "memory_video_latents", "memory_audio_latents"],
        )

        assert set(pipe.components) == {"vae", "audio_vae"}
        assert output["first_frame_latents"].shape == (1, 4, 1, 16, 16)
        assert len(output["memory_video_latents"]) == 1
        assert output["memory_audio_latents"][0].shape == (1, 2, 6, 2)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_audio_normalization_matches_packed_statistics(self, dtype):
        latents = torch.randn(2, 2, 5, 3).to(dtype)
        mean = torch.linspace(-0.5, 0.5, 6)
        std = torch.linspace(0.5, 1.5, 6)
        packed = _pack_audio_latents(latents)
        expected = (packed - mean.to(dtype)) / std.to(dtype)
        normalized = _normalize_audio_latents(latents, mean, std)

        torch.testing.assert_close(_pack_audio_latents(normalized), expected, rtol=0, atol=0)
        expected_decoded = (expected * std.to(dtype) + mean.to(dtype)).unflatten(2, (2, 3)).transpose(1, 2)
        torch.testing.assert_close(_denormalize_audio_latents(normalized, mean, std), expected_decoded, rtol=0, atol=0)

    def test_conditioning_uses_latent_geometry(self):
        pipe = EchoPrepareConditioningStep().init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components(dtype=torch.float32)
        video_latents = [torch.rand(1, 4, 1, 6, 10) for _ in range(3)]
        audio_latents = [torch.rand(1, 2, 5, 2), None, torch.rand(1, 2, 7, 2)]
        output = pipe(
            memory_video_latents=video_latents,
            memory_audio_latents=audio_latents,
            output=["memory_video_tokens", "memory_video_coords", "memory_audio_tokens", "memory_audio_coords"],
        )

        assert set(pipe.components) == {"transformer"}
        assert output["memory_video_tokens"].shape == (1, 180, 4)
        assert output["memory_video_coords"].shape == (1, 3, 180, 2)
        assert output["memory_audio_tokens"].shape == (1, 17, 4)
        assert output["memory_audio_coords"].shape == (1, 1, 17, 2)
        torch.testing.assert_close(output["memory_audio_tokens"][:, :5], _pack_audio_latents(audio_latents[0]))
        assert torch.count_nonzero(output["memory_audio_tokens"][:, 5:10]) == 0
        torch.testing.assert_close(output["memory_audio_tokens"][:, 10:], _pack_audio_latents(audio_latents[2]))

    @pytest.mark.parametrize("num_videos", [1, 2])
    @pytest.mark.parametrize("condition_batch", [1, 2])
    def test_inputs_expand_cached_conditioning(self, num_videos, condition_batch):
        pipe = EchoInputsStep().init_pipeline()
        inputs = {
            "connector_prompt_embeds": torch.randn(2, 4, 8),
            "connector_audio_prompt_embeds": torch.randn(2, 4, 4),
            "connector_attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]),
            "first_frame_tokens": torch.randn(condition_batch, 6, 4),
            "memory_video_tokens": torch.randn(condition_batch, 12, 4),
            "memory_audio_tokens": torch.randn(condition_batch, 5, 4),
            "memory_video_coords": torch.randn(1, 3, 12, 2),
            "memory_audio_coords": torch.randn(1, 1, 5, 2),
        }
        originals = {name: value.clone() for name, value in inputs.items()}
        # Run the same cached tensors repeatedly with different requested multiplicities.
        for count in (num_videos, 1, 2):
            output = pipe(**inputs, num_videos_per_prompt=count, output=list(inputs))
            for name, value in inputs.items():
                expected = value.repeat_interleave(2 if value.shape[0] == 1 else 1, dim=0)
                torch.testing.assert_close(output[name], expected.repeat_interleave(count, dim=0))
                torch.testing.assert_close(value, originals[name], rtol=0, atol=0)

    def test_decoder_runs_with_only_vae_form_latents(self):
        pipe = self.get_pipeline()
        output = pipe(**self.get_dummy_inputs(), output=["latents", "audio_latents", "videos", "audio"])
        assert output["latents"].shape == (1, 4, 3, 16, 16)
        assert output["audio_latents"].shape == (1, 2, 5, 2)

        decoder = EchoDecoderStep().init_pipeline(self.pretrained_model_name_or_path)
        decoder.load_components(dtype=torch.float32)
        decoded = decoder(
            latents=output["latents"],
            audio_latents=output["audio_latents"],
            output_type="pt",
            output=["videos", "audio"],
        )
        assert "transformer" not in decoder.components
        torch.testing.assert_close(decoded["videos"], output["videos"], rtol=0, atol=0)
        torch.testing.assert_close(decoded["audio"], output["audio"], rtol=0, atol=0)

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
        pipe.blocks.sub_blocks.pop("decode")
        output = pipe(**inputs, output=["latents", "audio_latents"])
        torch.testing.assert_close(output["latents"], inputs["latents"], rtol=0, atol=0)
        torch.testing.assert_close(output["audio_latents"], inputs["audio_latents"], rtol=0, atol=0)

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
        waveform = torch.zeros(2, 12 * 16000)
        waveform[:, -16000:] = torch.randn(2, 16000)

        _encode_audio(
            audio_vae,
            torch.zeros(1),
            torch.ones(1),
            waveform,
            16000,
            torch.device("cpu"),
        )

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
