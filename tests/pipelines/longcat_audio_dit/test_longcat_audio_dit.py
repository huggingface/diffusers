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
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    LongCatAudioDiTPipeline,
    LongCatAudioDiTTransformer,
    LongCatAudioDiTVae,
)

from ...testing_utils import (
    assert_tensors_close,
    enable_full_determinism,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..pipeline_params import TEXT_TO_AUDIO_BATCH_PARAMS, TEXT_TO_AUDIO_PARAMS
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


enable_full_determinism()


class LongCatAudioDiTPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = LongCatAudioDiTPipeline
    # This pipeline sizes its output with `audio_duration_s` and takes no precomputed prompt embeddings.
    required_input_params_in_call_signature = (
        TEXT_TO_AUDIO_PARAMS
        - {"audio_length_in_s", "prompt_embeds", "negative_prompt_embeds", "cross_attention_kwargs"}
    ) | {"audio_duration_s"}
    batch_input_params = TEXT_TO_AUDIO_BATCH_PARAMS
    # Waveform length for `audio_duration_s=0.1` at the tiny VAE's 24 kHz sample rate, as one mono channel.
    output_shape = (1, 4800)
    # An audio pipeline: `__call__` has no `num_images_per_prompt`, and the noise is always sampled internally
    # rather than passed in as `latents`.
    optional_input_params = frozenset(["num_inference_steps", "generator", "output_type", "return_dict"])

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

    def get_dummy_inputs(self):
        return {
            "prompt": "soft ocean ambience",
            "audio_duration_s": 0.1,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "generator": self.get_generator(0),
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestLongCatAudioDiTPipeline(LongCatAudioDiTPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        pipe = self.get_pipeline().to(torch_device)

        audios = pipe(**self.get_dummy_inputs()).audios

        assert audios.shape == (1, *self.output_shape)

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=2e-3):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    @pytest.mark.skip("`LongCatAudioDiTPipeline.encode_prompt` has a custom signature.")
    def test_encode_prompt_works_in_isolation(self):
        pass

    def test_uniform_flow_match_scheduler_grid_matches_manual_updates(self):
        num_inference_steps = 6
        scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0, invert_sigmas=True)
        sigmas = torch.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=torch.float32).tolist()
        scheduler.set_timesteps(sigmas=sigmas, device="cpu")

        expected_grid = torch.linspace(0, 1, num_inference_steps + 1, dtype=torch.float32)
        actual_timesteps = scheduler.timesteps / scheduler.config.num_train_timesteps
        assert_tensors_close(actual_timesteps, expected_grid[:-1], atol=1e-6, rtol=0)

        sample = torch.zeros(1, 2, 3)
        model_output = torch.ones_like(sample)
        expected = sample.clone()
        for t0, t1, scheduler_t in zip(expected_grid[:-1], expected_grid[1:], scheduler.timesteps):
            expected = expected + model_output * (t1 - t0)
            sample = scheduler.step(model_output, scheduler_t, sample, return_dict=False)[0]

        assert_tensors_close(sample, expected, atol=1e-6, rtol=0)


class TestLongCatAudioDiTPipelineMemory(LongCatAudioDiTPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the LongCat AudioDiT pipeline."""

    @pytest.mark.skip("Offload coverage is not ready for this pipeline.")
    def test_model_cpu_offload_forward_pass(self):
        pass

    @pytest.mark.skip("Offload coverage is not ready for this pipeline.")
    def test_cpu_offload_forward_pass_twice(self):
        pass

    @pytest.mark.skip("The pipeline uses `torch.nn.utils.weight_norm`, incompatible with sequential offloading.")
    def test_sequential_cpu_offload_forward_pass(self):
        pass

    @pytest.mark.skip("The pipeline uses `torch.nn.utils.weight_norm`, incompatible with sequential offloading.")
    def test_sequential_offload_forward_pass_twice(self):
        pass

    @pytest.mark.skip("Group offloading coverage is not ready for this pipeline.")
    def test_pipeline_level_group_offloading_inference(self):
        pass


def test_longcat_audio_top_level_imports():
    assert LongCatAudioDiTPipeline is not None
    assert LongCatAudioDiTTransformer is not None
    assert LongCatAudioDiTVae is not None


@slow
@require_torch_accelerator
class TestLongCatAudioDiTPipelineIntegration:
    def test_longcat_audio_pipeline_from_pretrained_real_local_weights(self):
        model_path = Path(
            os.getenv("LONGCAT_AUDIO_DIT_MODEL_PATH", "/data/models/meituan-longcat/LongCat-AudioDiT-1B")
        )
        tokenizer_path_env = os.getenv("LONGCAT_AUDIO_DIT_TOKENIZER_PATH")
        if tokenizer_path_env is None:
            pytest.skip("LONGCAT_AUDIO_DIT_TOKENIZER_PATH is not set")
        tokenizer_path = Path(tokenizer_path_env)

        if not model_path.exists():
            pytest.skip(f"LongCat-AudioDiT model path not found: {model_path}")
        if not tokenizer_path.exists():
            pytest.skip(f"LongCat-AudioDiT tokenizer path not found: {tokenizer_path}")

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
