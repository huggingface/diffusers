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
from transformers import AutoTokenizer, Qwen3Config, Qwen3Model

from diffusers import AutoencoderOobleck
from diffusers.models.transformers.ace_step_transformer import AceStepDiTModel
from diffusers.pipelines.ace_step import AceStepConditionEncoder, AceStepPipeline

from ...testing_utils import enable_full_determinism
from ..test_pipelines_common import PipelineTesterMixin


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

    def test_gradient_checkpointing(self):
        """Test that gradient checkpointing can be enabled."""
        config = self.get_tiny_config()
        model = AceStepDiTModel(**config)
        model.enable_gradient_checkpointing()
        self.assertTrue(model.gradient_checkpointing)


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

    def test_save_load_config(self):
        """Test that the condition encoder config can be saved and loaded."""
        import tempfile

        config = self.get_tiny_config()
        encoder = AceStepConditionEncoder(**config)

        with tempfile.TemporaryDirectory() as tmpdir:
            encoder.save_config(tmpdir)
            loaded = AceStepConditionEncoder.from_config(tmpdir)

        self.assertEqual(encoder.config.hidden_size, loaded.config.hidden_size)
        self.assertEqual(encoder.config.text_hidden_dim, loaded.config.text_hidden_dim)
        self.assertEqual(encoder.config.timbre_hidden_dim, loaded.config.timbre_hidden_dim)


class AceStepPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    """Fast end-to-end tests for AceStepPipeline with tiny models."""

    pipeline_class = AceStepPipeline
    params = frozenset(
        [
            "prompt",
            "lyrics",
            "audio_duration",
            "vocal_language",
            "guidance_scale",
            "shift",
        ]
    )
    batch_params = frozenset(["prompt", "lyrics"])
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "generator",
            "latents",
            "output_type",
            "return_dict",
        ]
    )

    # ACE-Step uses custom attention, not standard diffusers attention processors
    test_attention_slicing = False
    test_xformers_attention = False
    supports_dduf = False

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

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A beautiful piano piece",
            "lyrics": "[verse]\nSoft notes in the morning",
            "audio_duration": 0.4,  # Very short for fast test (10 latent frames at 25Hz)
            "num_inference_steps": 2,
            "generator": generator,
            "max_text_length": 32,
        }
        return inputs

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
            audio_duration=0.4,
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

    def test_save_load_local(self, expected_max_difference=7e-3):
        # increase tolerance to account for large composite model
        super().test_save_load_local(expected_max_difference=expected_max_difference)

    def test_save_load_optional_components(self, expected_max_difference=7e-3):
        # increase tolerance to account for large composite model
        super().test_save_load_optional_components(expected_max_difference=expected_max_difference)

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=7e-3):
        # increase tolerance for audio pipeline
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    def test_dict_tuple_outputs_equivalent(self, expected_slice=None, expected_max_difference=7e-3):
        # increase tolerance for audio pipeline
        super().test_dict_tuple_outputs_equivalent(
            expected_slice=expected_slice, expected_max_difference=expected_max_difference
        )

    # ACE-Step does not use num_images_per_prompt
    def test_num_images_per_prompt(self):
        pass

    # ACE-Step does not use standard schedulers
    @unittest.skip("ACE-Step uses built-in flow matching schedule, not diffusers schedulers")
    def test_karras_schedulers_shape(self):
        pass

    # ACE-Step does not support prompt_embeds directly
    @unittest.skip("ACE-Step does not support prompt_embeds / negative_prompt_embeds")
    def test_cfg(self):
        pass

    def test_float16_inference(self, expected_max_diff=5e-2):
        super().test_float16_inference(expected_max_diff=expected_max_diff)

    @unittest.skip("ACE-Step __call__ does not accept prompt_embeds, so encode_prompt isolation test is not applicable")
    def test_encode_prompt_works_in_isolation(self):
        pass

    @unittest.skip("Sequential CPU offloading produces NaN with tiny random models")
    def test_sequential_cpu_offload_forward_pass(self):
        pass

    @unittest.skip("Sequential CPU offloading produces NaN with tiny random models")
    def test_sequential_offload_forward_pass_twice(self):
        pass

    def test_encode_prompt(self):
        """Test that encode_prompt returns correct shapes."""
        device = "cpu"
        components = self.get_dummy_components()
        pipe = AceStepPipeline(**components)
        pipe = pipe.to(device)

        text_hidden, text_mask, lyric_hidden, lyric_mask = pipe.encode_prompt(
            prompt="A test prompt",
            lyrics="[verse]\nHello world",
            device=device,
            max_text_length=32,
            max_lyric_length=64,
        )

        self.assertEqual(text_hidden.ndim, 3)  # [batch, seq_len, hidden_dim]
        self.assertEqual(text_mask.ndim, 2)  # [batch, seq_len]
        self.assertEqual(lyric_hidden.ndim, 3)
        self.assertEqual(lyric_mask.ndim, 2)
        self.assertEqual(text_hidden.shape[0], 1)
        self.assertEqual(lyric_hidden.shape[0], 1)

    def test_prepare_latents(self):
        """Test that prepare_latents returns correct shapes."""
        device = "cpu"
        components = self.get_dummy_components()
        pipe = AceStepPipeline(**components)
        pipe = pipe.to(device)

        latents = pipe.prepare_latents(
            batch_size=2,
            audio_duration=1.0,
            dtype=torch.float32,
            device=device,
        )

        # 25 Hz latent rate, 1s duration -> 25 frames
        self.assertEqual(latents.shape, (2, 25, 8))

    def test_timestep_schedule(self):
        """Test that the timestep schedule is generated correctly."""
        components = self.get_dummy_components()
        pipe = AceStepPipeline(**components)

        # Test standard schedule
        schedule = pipe._get_timestep_schedule(num_inference_steps=8, shift=3.0)
        self.assertEqual(len(schedule), 8)
        self.assertAlmostEqual(schedule[0].item(), 1.0, places=5)

        # Test truncated schedule
        schedule = pipe._get_timestep_schedule(num_inference_steps=4, shift=3.0)
        self.assertEqual(len(schedule), 4)

    def test_format_prompt(self):
        """Test that prompt formatting works correctly."""
        components = self.get_dummy_components()
        pipe = AceStepPipeline(**components)

        text, lyrics = pipe._format_prompt(
            prompt="A piano piece",
            lyrics="[verse]\nHello",
            vocal_language="en",
            audio_duration=30.0,
        )

        self.assertIn("A piano piece", text)
        self.assertIn("30 seconds", text)
        self.assertIn("[verse]", lyrics)
        self.assertIn("Hello", lyrics)
        self.assertIn("en", lyrics)


if __name__ == "__main__":
    unittest.main()
