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

import os
import unittest
from pathlib import Path

import torch
from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    LongCatAudioDiTPipeline,
    LongCatAudioDiTTransformer,
    LongCatAudioDiTVae,
)

from ...testing_utils import enable_full_determinism, require_torch_accelerator, slow, torch_device
from ..pipeline_params import TEXT_TO_AUDIO_BATCH_PARAMS, TEXT_TO_AUDIO_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class LongCatAudioDiTPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = LongCatAudioDiTPipeline
    params = (
        TEXT_TO_AUDIO_PARAMS
        - {"audio_length_in_s", "prompt_embeds", "negative_prompt_embeds", "cross_attention_kwargs"}
    ) | {"audio_duration_s"}
    batch_params = TEXT_TO_AUDIO_BATCH_PARAMS
    required_optional_params = PipelineTesterMixin.required_optional_params - {"num_images_per_prompt"}
    test_attention_slicing = False
    test_xformers_attention = False
    supports_dduf = False

    def get_dummy_components(self):
        torch.manual_seed(0)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder = UMT5EncoderModel(
            UMT5Config(d_model=32, num_layers=1, num_heads=4, d_ff=64, vocab_size=tokenizer.vocab_size)
        )
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
            "tokenizer": tokenizer,
            "transformer": transformer,
        }

    def get_dummy_inputs(self, device, seed=0, prompt="soft ocean ambience"):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        return {
            "prompt": prompt,
            "audio_duration_s": 0.1,
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

    def test_save_load_local(self):
        import tempfile

        device = "cpu"
        pipe = self.pipeline_class(**self.get_dummy_components())
        pipe.to(device)

        with tempfile.TemporaryDirectory() as tmp_dir:
            pipe.save_pretrained(tmp_dir)
            reloaded = self.pipeline_class.from_pretrained(tmp_dir, local_files_only=True)
            output = reloaded(**self.get_dummy_inputs(device, seed=0)).audios

        self.assertIsInstance(reloaded, LongCatAudioDiTPipeline)
        self.assertEqual(output.ndim, 3)
        self.assertGreater(output.shape[-1], 0)

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=2e-3)

    def test_model_cpu_offload_forward_pass(self):
        self.skipTest(
            "LongCatAudioDiTPipeline offload coverage is not ready for the standard PipelineTesterMixin test."
        )

    def test_cpu_offload_forward_pass_twice(self):
        self.skipTest(
            "LongCatAudioDiTPipeline offload coverage is not ready for the standard PipelineTesterMixin test."
        )

    def test_sequential_cpu_offload_forward_pass(self):
        self.skipTest(
            "LongCatAudioDiTPipeline uses `torch.nn.utils.weight_norm`, which is not compatible with "
            "sequential offloading."
        )

    def test_sequential_offload_forward_pass_twice(self):
        self.skipTest(
            "LongCatAudioDiTPipeline uses `torch.nn.utils.weight_norm`, which is not compatible with "
            "sequential offloading."
        )

    def test_pipeline_level_group_offloading_inference(self):
        self.skipTest(
            "LongCatAudioDiTPipeline group offloading coverage is not ready for the standard PipelineTesterMixin test."
        )

    def test_num_images_per_prompt(self):
        self.skipTest("LongCatAudioDiTPipeline does not support num_images_per_prompt.")

    def test_encode_prompt_works_in_isolation(self):
        self.skipTest("LongCatAudioDiTPipeline.encode_prompt has a custom signature.")

    def test_uniform_flow_match_scheduler_grid_matches_manual_updates(self):
        num_inference_steps = 6
        scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0, invert_sigmas=True)
        sigmas = torch.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=torch.float32).tolist()
        scheduler.set_timesteps(sigmas=sigmas, device="cpu")

        expected_grid = torch.linspace(0, 1, num_inference_steps + 1, dtype=torch.float32)
        actual_timesteps = scheduler.timesteps / scheduler.config.num_train_timesteps
        self.assertTrue(torch.allclose(actual_timesteps, expected_grid[:-1], atol=1e-6, rtol=0))

        sample = torch.zeros(1, 2, 3)
        model_output = torch.ones_like(sample)
        expected = sample.clone()
        for t0, t1, scheduler_t in zip(expected_grid[:-1], expected_grid[1:], scheduler.timesteps):
            expected = expected + model_output * (t1 - t0)
            sample = scheduler.step(model_output, scheduler_t, sample, return_dict=False)[0]

        self.assertTrue(torch.allclose(sample, expected, atol=1e-6, rtol=0))


def test_longcat_audio_top_level_imports():
    assert LongCatAudioDiTPipeline is not None
    assert LongCatAudioDiTTransformer is not None
    assert LongCatAudioDiTVae is not None


@slow
@require_torch_accelerator
class LongCatAudioDiTPipelineSlowTests(unittest.TestCase):
    pipeline_class = LongCatAudioDiTPipeline

    def test_longcat_audio_pipeline_from_pretrained_real_local_weights(self):
        model_path = Path(
            os.getenv("LONGCAT_AUDIO_DIT_MODEL_PATH", "/data/models/meituan-longcat/LongCat-AudioDiT-1B")
        )
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
            audio_duration_s=2.0,
            num_inference_steps=2,
            guidance_scale=4.0,
            output_type="pt",
        )

        assert result.audios.ndim == 3
        assert result.audios.shape[0] == 1
        assert result.audios.shape[1] == 1
        assert result.audios.shape[-1] > 0
