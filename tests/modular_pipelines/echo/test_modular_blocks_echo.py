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

from copy import deepcopy

import pytest
import torch

from diffusers.modular_pipelines.echo.before_denoise import EchoInputsStep, EchoPrepareConditioningStep
from diffusers.modular_pipelines.echo.decoders import EchoAudioDecoderStep
from diffusers.modular_pipelines.echo.encoders import EchoVaeEncoderStep
from diffusers.modular_pipelines.echo.modular_blocks_echo import EchoDecoderStep

from ..testing_utils import BaseModularPipelineOutputMixin
from .testing_utils import EchoModularPipelineTesterConfig


class TestEchoModularBlocks(EchoModularPipelineTesterConfig, BaseModularPipelineOutputMixin):
    def test_vae_encoder_rejects_more_than_seven_memory_slots(self):
        pipe = EchoVaeEncoderStep().init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components(dtype=torch.float32)

        with pytest.raises(ValueError, match="at most 7 memory slots"):
            pipe(memory_images=[torch.zeros(1, 3, 32, 32) for _ in range(8)], height=32, width=32)

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
    def test_audio_encoder_decoder_use_channel_mel_statistics(self, dtype):
        pytest.importorskip("torchaudio")
        encoder = EchoVaeEncoderStep().init_pipeline(self.pretrained_model_name_or_path)
        encoder.load_components(dtype={"default": torch.float32, "audio_vae": dtype})
        identity_vae = deepcopy(encoder.audio_vae)
        identity_vae.latents_mean.zero_()
        identity_vae.latents_std.fill_(1)
        encoder.update_components(audio_vae=identity_vae)
        inputs = {
            "memory_images": [torch.rand(1, 3, 32, 32, generator=self.get_generator(0))],
            "memory_audio_waveforms": [torch.rand(2, 3200, generator=self.get_generator(1))],
            "memory_audio_sample_rates": 16000,
            "height": 32,
            "width": 32,
        }
        raw_latents = encoder(**inputs, output="memory_audio_latents")[0].to(dtype)

        normalized_vae = deepcopy(identity_vae)
        mean = torch.linspace(-0.5, 0.5, raw_latents.shape[1] * raw_latents.shape[3]).to(dtype)
        std = torch.linspace(0.5, 1.5, mean.numel()).to(dtype)
        normalized_vae.latents_mean.copy_(mean)
        normalized_vae.latents_std.copy_(std)
        encoder.update_components(audio_vae=normalized_vae)
        normalized = encoder(**inputs, output="memory_audio_latents")[0]

        # Statistics are stored in flattened (channel, mel-bin) order, independent of the time axis.
        expected = (raw_latents.transpose(1, 2).flatten(2) - mean) / std
        torch.testing.assert_close(normalized.transpose(1, 2).flatten(2), expected.float(), rtol=0, atol=0)

        decoder = EchoAudioDecoderStep().init_pipeline(self.pretrained_model_name_or_path)
        decoder.load_components(dtype=torch.float32)
        decoder.update_components(audio_vae=normalized_vae)
        decoded = decoder(audio_latents=normalized.to(dtype), output_type="latent", output="audio")
        expected_decoded = (
            (expected * std + mean).unflatten(2, (raw_latents.shape[1], raw_latents.shape[3])).transpose(1, 2)
        )
        torch.testing.assert_close(decoded, expected_decoded, rtol=0, atol=0)

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
        torch.testing.assert_close(output["memory_audio_tokens"][:, :5], audio_latents[0].transpose(1, 2).flatten(2))
        assert torch.count_nonzero(output["memory_audio_tokens"][:, 5:10]) == 0
        torch.testing.assert_close(output["memory_audio_tokens"][:, 10:], audio_latents[2].transpose(1, 2).flatten(2))

    @pytest.mark.parametrize("num_videos", [1, 2])
    @pytest.mark.parametrize("condition_batch", [1, 2])
    def test_inputs_expand_cached_conditioning(self, num_videos, condition_batch):
        pipe = EchoInputsStep().init_pipeline()
        pipe.load_components()
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

    def test_audio_memory_is_cropped_to_highest_response_window(self):
        pytest.importorskip("torchaudio")
        pipe = EchoVaeEncoderStep().init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components(dtype=torch.float32)
        sample_rate = pipe.audio_vae.config.sample_rate
        waveform = torch.zeros(2, 12 * sample_rate)
        waveform[:, -sample_rate:] = torch.randn(2, sample_rate, generator=self.get_generator(0))
        inputs = {
            "memory_images": [torch.rand(1, 3, 32, 32, generator=self.get_generator(1))],
            "memory_audio_sample_rates": sample_rate,
            "height": 32,
            "width": 32,
        }
        output = pipe(**inputs, memory_audio_waveforms=[waveform], output="memory_audio_latents")[0]

        # Only the last second contains sound, so the highest-response 9.62-second window ends at the tail.
        cropped_waveform = waveform[:, -round(9.62 * sample_rate) :]
        expected = pipe(**inputs, memory_audio_waveforms=[cropped_waveform], output="memory_audio_latents")[0]
        mel_steps = cropped_waveform.shape[-1] // pipe.audio_vae.config.mel_hop_length + 1
        assert output.shape[2] == (mel_steps - 1) // pipe.audio_vae.temporal_compression_ratio + 1
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
