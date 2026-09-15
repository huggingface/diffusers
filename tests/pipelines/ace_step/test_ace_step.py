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


import math

import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3Model

from diffusers import AutoencoderOobleck, FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.ace_step_transformer import AceStepTransformer1DModel
from diffusers.pipelines.ace_step import (
    AceStepAudioTokenDetokenizer,
    AceStepAudioTokenizer,
    AceStepConditionEncoder,
    AceStepPipeline,
)

from ...testing_utils import enable_full_determinism
from ..testing_utils import (
    BasePipelineTesterConfig,
    LoraMemoryTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


class TestAceStepConditionEncoder:
    """Fast tests for the AceStepConditionEncoder."""

    def get_tiny_config(self):
        return {
            "hidden_size": 32,
            "intermediate_size": 64,
            "text_hidden_dim": 16,
            "timbre_hidden_dim": 8,
            "num_lyric_encoder_hidden_layers": 2,
            "num_timbre_encoder_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "rope_theta": 10000.0,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "rms_norm_eps": 1e-6,
            "sliding_window": 16,
        }

    def test_forward_shape(self):
        """Test that the condition encoder produces packed hidden states."""
        config = self.get_tiny_config()
        encoder = AceStepConditionEncoder(**config)
        encoder.eval()

        batch_size = 2
        text_seq_len = 8
        lyric_seq_len = 12
        text_dim = config["text_hidden_dim"]
        timbre_dim = config["timbre_hidden_dim"]
        timbre_time = 10

        text_hidden_states = torch.randn(batch_size, text_seq_len, text_dim)
        text_attention_mask = torch.ones(batch_size, text_seq_len)
        lyric_hidden_states = torch.randn(batch_size, lyric_seq_len, text_dim)
        lyric_attention_mask = torch.ones(batch_size, lyric_seq_len)

        # Packed reference audio: 3 references across 2 batch items
        refer_audio = torch.randn(3, timbre_time, timbre_dim)
        refer_order_mask = torch.tensor([0, 0, 1], dtype=torch.long)

        with torch.no_grad():
            enc_hidden, enc_mask = encoder(
                text_hidden_states=text_hidden_states,
                text_attention_mask=text_attention_mask,
                lyric_hidden_states=lyric_hidden_states,
                lyric_attention_mask=lyric_attention_mask,
                refer_audio_acoustic_hidden_states_packed=refer_audio,
                refer_audio_order_mask=refer_order_mask,
            )

        # Output should be packed: batch_size x (lyric + timbre + text seq_len) x hidden_size
        assert enc_hidden.shape[0] == batch_size
        assert enc_hidden.shape[2] == config["hidden_size"]
        assert enc_mask.shape[0] == batch_size
        assert enc_mask.shape[1] == enc_hidden.shape[1]

    def test_save_load_config(self, tmp_path):
        """Test that the condition encoder config can be saved and loaded."""
        config = self.get_tiny_config()
        encoder = AceStepConditionEncoder(**config)

        encoder.save_config(tmp_path)
        loaded = AceStepConditionEncoder.from_config(tmp_path)

        assert encoder.config.hidden_size == loaded.config.hidden_size
        assert encoder.config.text_hidden_dim == loaded.config.text_hidden_dim
        assert encoder.config.timbre_hidden_dim == loaded.config.timbre_hidden_dim


class AceStepPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = AceStepPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "lyrics", "audio_duration", "vocal_language", "guidance_scale", "shift"]
    )
    batch_input_params = frozenset(["prompt", "lyrics"])
    # ACE-Step generates audio, so there is no `num_images_per_prompt`.
    optional_input_params = frozenset(["num_inference_steps", "generator", "latents", "output_type", "return_dict"])
    # `(channels, samples)` for the short `audio_duration` used by the dummy inputs.
    output_shape = (2, 7)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = AceStepTransformer1DModel(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            in_channels=24,
            audio_acoustic_hidden_dim=8,
            patch_size=2,
            rope_theta=10000.0,
            sliding_window=16,
        )

        # Create a tiny Qwen3Model for testing (matching the real Qwen3-Embedding-0.6B architecture)
        torch.manual_seed(0)
        qwen3_config = Qwen3Config(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=151936,  # Qwen3 vocab size
            max_position_embeddings=256,
        )
        text_encoder = Qwen3Model(qwen3_config)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
        text_hidden_dim = qwen3_config.hidden_size  # 32

        torch.manual_seed(0)
        condition_encoder = AceStepConditionEncoder(
            hidden_size=32,
            intermediate_size=64,
            text_hidden_dim=text_hidden_dim,
            timbre_hidden_dim=8,
            num_lyric_encoder_hidden_layers=2,
            num_timbre_encoder_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            rope_theta=10000.0,
            sliding_window=16,
        )

        audio_tokenizer_kwargs = {
            "hidden_size": 32,
            "intermediate_size": 64,
            "audio_acoustic_hidden_dim": 8,
            "pool_window_size": 2,
            "fsq_dim": 32,
            "fsq_input_levels": [4, 4, 4],
            "fsq_input_num_quantizers": 1,
            "num_attention_pooler_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "rope_theta": 10000.0,
            "sliding_window": 16,
        }
        torch.manual_seed(0)
        audio_tokenizer = AceStepAudioTokenizer(**audio_tokenizer_kwargs)
        torch.manual_seed(0)
        audio_token_detokenizer = AceStepAudioTokenDetokenizer(
            hidden_size=32,
            intermediate_size=64,
            audio_acoustic_hidden_dim=8,
            pool_window_size=2,
            num_attention_pooler_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            rope_theta=10000.0,
            sliding_window=16,
        )

        torch.manual_seed(0)
        vae = AutoencoderOobleck(
            encoder_hidden_size=6,
            downsampling_ratios=[1, 2],
            decoder_channels=3,
            decoder_input_channels=8,
            audio_channels=2,
            channel_multiples=[2, 4],
            sampling_rate=4,
        )

        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1, shift=1.0)

        return {
            "transformer": transformer,
            "condition_encoder": condition_encoder,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "scheduler": scheduler,
            "audio_tokenizer": audio_tokenizer,
            "audio_token_detokenizer": audio_token_detokenizer,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "A beautiful piano piece",
            "lyrics": "[verse]\nSoft notes in the morning",
            # Short for a fast test, but long enough that the decoded waveform carries enough samples for the
            # output comparisons the common tests make (the tiny VAE here runs at `latents_per_second == 2`).
            "audio_duration": 2.0,
            "num_inference_steps": 2,
            "generator": self.get_generator(0),
            "max_text_length": 32,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestAceStepPipeline(AceStepPipelineTesterConfig, PipelineTesterMixin):
    """Fast end-to-end tests for AceStepPipeline with tiny models."""

    def test_ace_step_batch(self):
        """Test batch generation."""
        pipe = self.get_pipeline()

        audio = self.run_pipe(
            pipe, prompt=["Piano piece", "Guitar solo"], lyrics=["[verse]\nHello", "[chorus]\nWorld"]
        )
        assert audio.shape[0] == 2  # batch size = 2

    def test_ace_step_latent_output(self):
        """Test that output_type='latent' returns latents."""
        pipe = self.get_pipeline()

        latents = self.run_pipe(pipe, lyrics="", output_type="latent")
        # Latent shape: [batch, latent_length, acoustic_dim]
        assert latents.ndim == 3
        assert latents.shape[0] == 1

    def test_ace_step_return_dict_false(self):
        """Test that return_dict=False returns a tuple."""
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        output = pipe(**inputs, return_dict=False)
        assert isinstance(output, tuple)
        assert len(output) == 1

    def test_audio_codes_cover_path(self):
        pipe = self.get_pipeline()

        output = pipe(
            prompt="A test prompt",
            lyrics="",
            audio_codes="<|audio_code_1|><|audio_code_2|>",
            num_inference_steps=1,
            output_type="latent",
            max_text_length=32,
        )

        assert output.audios.shape[1] == 4

    def test_save_load_local(self, tmp_path, base_pipe_output, expected_max_difference=7e-3):
        # increase tolerance to account for large composite model
        super().test_save_load_local(tmp_path, base_pipe_output, expected_max_difference=expected_max_difference)

    def test_save_load_optional_components(self, tmp_path, expected_max_difference=7e-3):
        # increase tolerance to account for large composite model
        super().test_save_load_optional_components(tmp_path, expected_max_difference=expected_max_difference)

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=7e-3):
        # increase tolerance for audio pipeline
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    def test_dict_tuple_outputs_equivalent(self, expected_slice=None, expected_max_difference=7e-3):
        # increase tolerance for audio pipeline
        super().test_dict_tuple_outputs_equivalent(
            expected_slice=expected_slice, expected_max_difference=expected_max_difference
        )

    @pytest.mark.skip(
        "ACE-Step __call__ does not accept prompt_embeds, so encode_prompt isolation test is not applicable"
    )
    def test_encode_prompt_works_in_isolation(self):
        pass

    def test_encode_prompt(self):
        """Test that encode_prompt returns correct shapes."""
        pipe = self.get_pipeline()

        text_hidden, text_mask, lyric_hidden, lyric_mask = pipe.encode_prompt(
            prompt="A test prompt",
            lyrics="[verse]\nHello world",
            device="cpu",
            max_text_length=32,
            max_lyric_length=64,
        )

        assert text_hidden.ndim == 3  # [batch, seq_len, hidden_dim]
        assert text_mask.ndim == 2  # [batch, seq_len]
        assert lyric_hidden.ndim == 3
        assert lyric_mask.ndim == 2
        assert text_hidden.shape[0] == 1
        assert lyric_hidden.shape[0] == 1

    def test_prepare_latents(self):
        """Test that prepare_latents returns correct shapes."""
        pipe = self.get_pipeline()

        latents = pipe.prepare_latents(
            batch_size=2,
            audio_duration=1.0,
            dtype=torch.float32,
            device="cpu",
        )

        expected_length = math.ceil(1.0 * pipe.latents_per_second)
        assert latents.shape == (2, expected_length, 8)

    def test_timestep_schedule(self):
        """Test that the timestep schedule is generated correctly."""
        pipe = self.get_pipeline()

        # Test standard schedule
        schedule = pipe._get_timestep_schedule(num_inference_steps=8, shift=3.0)
        assert len(schedule) == 8
        assert schedule[0].item() == pytest.approx(1.0, abs=1e-5)

        # Test truncated schedule
        schedule = pipe._get_timestep_schedule(num_inference_steps=4, shift=3.0)
        assert len(schedule) == 4

    def test_format_prompt(self):
        """Test that prompt formatting works correctly."""
        pipe = self.get_pipeline()

        text, lyrics = pipe._format_prompt(
            prompt="A piano piece",
            lyrics="[verse]\nHello",
            vocal_language="en",
            audio_duration=30.0,
        )

        assert "A piano piece" in text
        assert "30 seconds" in text
        assert "[verse]" in lyrics
        assert "Hello" in lyrics
        assert "en" in lyrics


class TestAceStepPipelineMemory(AceStepPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the ACE-Step pipeline."""

    @pytest.mark.skip("Sequential CPU offloading produces NaN with tiny random models")
    def test_sequential_cpu_offload_forward_pass(self):
        pass

    @pytest.mark.skip("Sequential CPU offloading produces NaN with tiny random models")
    def test_sequential_offload_forward_pass_twice(self):
        pass


class TestAceStepPipelineLoRA(AceStepPipelineTesterConfig, LoraTesterMixin):
    """LoRA tests for the ACE-Step pipeline."""


class TestAceStepPipelineLoRAMemory(AceStepPipelineTesterConfig, LoraMemoryTesterMixin):
    """LoRA x memory-optimization tests (group offload, CPU offload) for the ACE-Step pipeline."""
