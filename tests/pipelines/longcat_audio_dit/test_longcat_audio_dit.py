# Copyright 2026 The HuggingFace Team.
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

import json
import os
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import UMT5Config, UMT5EncoderModel

from diffusers import LongCatAudioDiTPipeline, LongCatAudioDiTTransformer, LongCatAudioDiTVae
from tests.testing_utils import require_torch_accelerator, slow, torch_device


class DummyTokenizer:
    model_max_length = 16

    def __call__(self, texts, padding="longest", truncation=True, max_length=None, return_tensors="pt"):
        if isinstance(texts, str):
            texts = [texts]
        batch = len(texts)
        return type(
            "TokenBatch",
            (),
            {
                "input_ids": torch.ones(batch, 4, dtype=torch.long),
                "attention_mask": torch.ones(batch, 4, dtype=torch.long),
            },
        )


class LongCatAudioDiTPipelineFastTests(unittest.TestCase):
    pipeline_class = LongCatAudioDiTPipeline

    def get_dummy_components(self):
        torch.manual_seed(0)
        text_encoder = UMT5EncoderModel(UMT5Config(d_model=32, num_layers=1, num_heads=4, d_ff=64, vocab_size=128))
        transformer = LongCatAudioDiTTransformer(
            dit_dim=64,
            dit_depth=2,
            dit_heads=4,
            dit_text_dim=32,
            latent_dim=8,
            text_conv=False,
        )
        vae = LongCatAudioDiTVae(
            in_channels=1,
            channels=16,
            c_mults=[1, 2],
            strides=[2],
            latent_dim=8,
            encoder_latent_dim=16,
            downsampling_ratio=2,
            sample_rate=24000,
        )

        return {
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": DummyTokenizer(),
            "transformer": transformer,
        }

    def get_dummy_inputs(self, device, seed=0, prompt="soft ocean ambience"):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        return {
            "prompt": prompt,
            "audio_end_in_s": 0.1,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "generator": generator,
            "output_type": "pt",
        }

    def test_inference(self):
        device = "cpu"
        pipe = self.pipeline_class(**self.get_dummy_components())
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        output = pipe(**self.get_dummy_inputs(device)).audios

        self.assertEqual(output.ndim, 3)
        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[1], 1)
        self.assertGreater(output.shape[-1], 0)

    def test_inference_batch_single_identical(self):
        device = "cpu"
        pipe = self.pipeline_class(**self.get_dummy_components())
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        output1 = pipe(**self.get_dummy_inputs(device, seed=42)).audios
        output2 = pipe(**self.get_dummy_inputs(device, seed=42)).audios

        self.assertTrue(torch.allclose(output1, output2, atol=1e-4))

    def test_inference_batch_multiple_prompts(self):
        device = "cpu"
        pipe = self.pipeline_class(**self.get_dummy_components())
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=device).manual_seed(42)
        output = pipe(
            prompt=["soft ocean ambience", "gentle rain ambience"],
            audio_end_in_s=0.1,
            num_inference_steps=2,
            guidance_scale=1.0,
            generator=generator,
            output_type="pt",
        ).audios

        self.assertEqual(output.ndim, 3)
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 1)
        self.assertGreater(output.shape[-1], 0)

    def test_save_pretrained_roundtrip(self):
        import tempfile

        device = "cpu"
        pipe = self.pipeline_class(**self.get_dummy_components())
        pipe.to(device)

        with tempfile.TemporaryDirectory() as tmp_dir:
            pipe.save_pretrained(tmp_dir)
            reloaded = self.pipeline_class.from_pretrained(tmp_dir, tokenizer=DummyTokenizer(), local_files_only=True)
            output = reloaded(**self.get_dummy_inputs(device, seed=0)).audios

        self.assertIsInstance(reloaded, LongCatAudioDiTPipeline)
        self.assertEqual(output.ndim, 3)
        self.assertGreater(output.shape[-1], 0)

    def test_from_pretrained_local_dir(self):
        import tempfile
        from unittest.mock import patch

        device = "cpu"
        components = self.get_dummy_components()
        text_encoder = components["text_encoder"]
        transformer = components["transformer"]
        vae = components["vae"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "longcat-audio-dit"
            model_dir.mkdir()

            config = {
                "dit_dim": 64,
                "dit_depth": 2,
                "dit_heads": 4,
                "dit_text_dim": 32,
                "latent_dim": 8,
                "dit_dropout": 0.0,
                "dit_bias": True,
                "dit_cross_attn": True,
                "dit_adaln_type": "global",
                "dit_adaln_use_text_cond": True,
                "dit_long_skip": True,
                "dit_text_conv": False,
                "dit_qk_norm": True,
                "dit_cross_attn_norm": False,
                "dit_eps": 1e-6,
                "dit_use_latent_condition": True,
                "sampling_rate": 24000,
                "latent_hop": 2,
                "max_wav_duration": 30.0,
                "text_norm_feat": True,
                "text_add_embed": True,
                "text_encoder_model": "dummy-umt5",
                "text_encoder_config": text_encoder.config.to_dict(),
                "vae_config": {**dict(vae.config), "model_type": "longcat_audio_dit_vae"},
            }
            with (model_dir / "config.json").open("w") as handle:
                json.dump(config, handle)

            state_dict = {}
            state_dict.update({f"text_encoder.{k}": v for k, v in text_encoder.state_dict().items() if k != "shared.weight"})
            state_dict.update({f"transformer.{k}": v for k, v in transformer.state_dict().items()})
            state_dict.update({f"vae.{k}": v for k, v in vae.state_dict().items()})
            save_file(state_dict, model_dir / "model.safetensors")

            with patch(
                "diffusers.pipelines.longcat_audio_dit.pipeline_longcat_audio_dit.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ):
                pipe = LongCatAudioDiTPipeline.from_pretrained(model_dir, local_files_only=True)

            output = pipe(**self.get_dummy_inputs(device, seed=0)).audios

            self.assertIsInstance(pipe, LongCatAudioDiTPipeline)
            self.assertEqual(pipe.sample_rate, 24000)
            self.assertEqual(pipe.latent_hop, 2)
            self.assertEqual(output.ndim, 3)
            self.assertGreater(output.shape[-1], 0)


def test_longcat_audio_top_level_imports():
    assert LongCatAudioDiTPipeline is not None
    assert LongCatAudioDiTTransformer is not None
    assert LongCatAudioDiTVae is not None


@slow
@require_torch_accelerator
def test_longcat_audio_pipeline_from_pretrained_real_local_weights():
    model_path = Path(os.getenv("LONGCAT_AUDIO_DIT_MODEL_PATH", "/data/models/meituan-longcat/LongCat-AudioDiT-1B"))
    tokenizer_path_env = os.getenv("LONGCAT_AUDIO_DIT_TOKENIZER_PATH")
    if tokenizer_path_env is None:
        raise unittest.SkipTest("LONGCAT_AUDIO_DIT_TOKENIZER_PATH is not set")
    tokenizer_path = Path(tokenizer_path_env)

    if not model_path.exists():
        raise unittest.SkipTest(f"LongCat-AudioDiT model path not found: {model_path}")
    if not tokenizer_path.exists():
        raise unittest.SkipTest(f"LongCat-AudioDiT tokenizer path not found: {tokenizer_path}")

    pipe = LongCatAudioDiTPipeline.from_pretrained(
        model_path,
        tokenizer=tokenizer_path,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    pipe = pipe.to(torch_device)

    result = pipe(
        prompt="A calm ocean wave ambience with soft wind in the background.",
        audio_end_in_s=2.0,
        num_inference_steps=2,
        guidance_scale=4.0,
        output_type="pt",
    )

    assert result.audios.ndim == 3
    assert result.audios.shape[0] == 1
    assert result.audios.shape[1] == 1
    assert result.audios.shape[-1] > 0
