# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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


import unittest

import torch

from diffusers import AutoencoderOobleck
from diffusers.models.transformers.ace_step_transformer import AceStepDiTModel
from diffusers.pipelines.ace_step import AceStepConditionEncoder, AceStepPipeline

from ...testing_utils import enable_full_determinism


enable_full_determinism()


class AceStepDiTModelTests(unittest.TestCase):
    """Fast tests for the AceStepDiTModel (DiT transformer)."""

    def get_tiny_config(self):
        return {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "in_channels": 24,  # audio_acoustic_hidden_dim * 3 (hidden + context_latents)
            "audio_acoustic_hidden_dim": 8,
            "patch_size": 2,
            "max_position_embeddings": 256,
            "rope_theta": 10000.0,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "rms_norm_eps": 1e-6,
            "use_sliding_window": True,
            "sliding_window": 16,
        }

    def test_forward_shape(self):
        """Test that the DiT model produces output with correct shape."""
        config = self.get_tiny_config()
        model = AceStepDiTModel(**config)
        model.eval()

        batch_size = 2
        seq_len = 8
        acoustic_dim = config["audio_acoustic_hidden_dim"]
        hidden_size = config["hidden_size"]
        encoder_seq_len = 10

        hidden_states = torch.randn(batch_size, seq_len, acoustic_dim)
        timestep = torch.rand(batch_size)
        timestep_r = torch.rand(batch_size)
        encoder_hidden_states = torch.randn(batch_size, encoder_seq_len, hidden_size)
        # context_latents = src_latents + chunk_masks, each of dim acoustic_dim
        context_latents = torch.randn(batch_size, seq_len, acoustic_dim * 2)

        with torch.no_grad():
            output = model(
                hidden_states=hidden_states,
                timestep=timestep,
                timestep_r=timestep_r,
                encoder_hidden_states=encoder_hidden_states,
                context_latents=context_latents,
                return_dict=False,
            )

        self.assertEqual(output[0].shape, (batch_size, seq_len, acoustic_dim))

    def test_forward_return_dict(self):
        """Test that return_dict=True returns a Transformer2DModelOutput."""
        config = self.get_tiny_config()
        model = AceStepDiTModel(**config)
        model.eval()

        batch_size = 1
        seq_len = 4
        acoustic_dim = config["audio_acoustic_hidden_dim"]
        hidden_size = config["hidden_size"]

        hidden_states = torch.randn(batch_size, seq_len, acoustic_dim)
        timestep = torch.rand(batch_size)
        timestep_r = torch.rand(batch_size)
        encoder_hidden_states = torch.randn(batch_size, 6, hidden_size)
        context_latents = torch.randn(batch_size, seq_len, acoustic_dim * 2)

        with torch.no_grad():
            output = model(
                hidden_states=hidden_states,
                timestep=timestep,
                timestep_r=timestep_r,
                encoder_hidden_states=encoder_hidden_states,
                context_latents=context_latents,
                return_dict=True,
            )

        self.assertTrue(hasattr(output, "sample"))
        self.assertEqual(output.sample.shape, (batch_size, seq_len, acoustic_dim))


class AceStepConditionEncoderTests(unittest.TestCase):
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
            "max_position_embeddings": 256,
            "rope_theta": 10000.0,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "rms_norm_eps": 1e-6,
            "use_sliding_window": False,
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
        self.assertEqual(enc_hidden.shape[0], batch_size)
        self.assertEqual(enc_hidden.shape[2], config["hidden_size"])
        self.assertEqual(enc_mask.shape[0], batch_size)
        self.assertEqual(enc_mask.shape[1], enc_hidden.shape[1])


class AceStepPipelineFastTests(unittest.TestCase):
    """Fast end-to-end tests for AceStepPipeline with tiny models."""

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = AceStepDiTModel(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            in_channels=24,
            audio_acoustic_hidden_dim=8,
            patch_size=2,
            max_position_embeddings=256,
            rope_theta=10000.0,
            use_sliding_window=False,
            sliding_window=16,
        )

        # Use T5 as a small text encoder for testing (d_model=32)
        # In production, ACE-Step uses Qwen3-Embedding-0.6B (hidden_size=1024)
        from transformers import T5EncoderModel, T5Tokenizer

        torch.manual_seed(0)
        t5_id = "hf-internal-testing/tiny-random-T5ForConditionalGeneration"
        text_encoder = T5EncoderModel.from_pretrained(t5_id)
        tokenizer = T5Tokenizer.from_pretrained(t5_id, truncation=True, model_max_length=256)
        text_hidden_dim = text_encoder.config.d_model  # 32

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
            max_position_embeddings=256,
            rope_theta=10000.0,
            use_sliding_window=False,
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

        components = {
            "transformer": transformer,
            "condition_encoder": condition_encoder,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }
        return components

    def test_ace_step_basic(self):
        """Test basic text-to-music generation."""
        device = "cpu"
        components = self.get_dummy_components()
        pipe = AceStepPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=device).manual_seed(0)
        output = pipe(
            prompt="A beautiful piano piece",
            lyrics="[verse]\nSoft notes in the morning",
            audio_duration=0.4,  # Very short for fast test (10 latent frames at 25Hz)
            num_inference_steps=2,
            generator=generator,
            max_text_length=32,
        )
        audio = output.audios
        self.assertIsNotNone(audio)
        self.assertEqual(audio.ndim, 3)  # [batch, channels, samples]

    def test_ace_step_batch(self):
        """Test batch generation."""
        device = "cpu"
        components = self.get_dummy_components()
        pipe = AceStepPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=device).manual_seed(42)
        output = pipe(
            prompt=["Piano piece", "Guitar solo"],
            lyrics=["[verse]\nHello", "[chorus]\nWorld"],
            audio_duration=0.4,
            num_inference_steps=2,
            generator=generator,
            max_text_length=32,
        )
        audio = output.audios
        self.assertIsNotNone(audio)
        self.assertEqual(audio.shape[0], 2)  # batch size = 2

    def test_ace_step_latent_output(self):
        """Test that output_type='latent' returns latents."""
        device = "cpu"
        components = self.get_dummy_components()
        pipe = AceStepPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=device).manual_seed(0)
        output = pipe(
            prompt="A test prompt",
            lyrics="",
            audio_duration=0.4,
            num_inference_steps=2,
            generator=generator,
            output_type="latent",
            max_text_length=32,
        )
        latents = output.audios
        self.assertIsNotNone(latents)
        # Latent shape: [batch, latent_length, acoustic_dim]
        self.assertEqual(latents.ndim, 3)
        self.assertEqual(latents.shape[0], 1)

    def test_ace_step_return_dict_false(self):
        """Test that return_dict=False returns a tuple."""
        device = "cpu"
        components = self.get_dummy_components()
        pipe = AceStepPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=device).manual_seed(0)
        output = pipe(
            prompt="A test prompt",
            lyrics="",
            audio_duration=0.4,
            num_inference_steps=2,
            generator=generator,
            return_dict=False,
            max_text_length=32,
        )
        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 1)


if __name__ == "__main__":
    unittest.main()
