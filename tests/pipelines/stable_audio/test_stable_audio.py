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


import gc
import unittest

import numpy as np
import torch
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
)

from diffusers import (
    AutoencoderOobleck,
    CosineDPMSolverMultistepScheduler,
    StableAudioDiTModel,
    StableAudioPipeline,
    StableAudioProjectionModel,
)
from diffusers.utils import is_xformers_available

from ...testing_utils import (
    Expectations,
    backend_empty_cache,
    enable_full_determinism,
    nightly,
    require_torch_accelerator,
    torch_device,
)
from ..pipeline_params import TEXT_TO_AUDIO_BATCH_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class StableAudioPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableAudioPipeline
    params = frozenset(
        [
            "prompt",
            "audio_end_in_s",
            "audio_start_in_s",
            "guidance_scale",
            "negative_prompt",
            "prompt_embeds",
            "negative_prompt_embeds",
            "initial_audio_waveforms",
        ]
    )
    batch_params = TEXT_TO_AUDIO_BATCH_PARAMS
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "num_waveforms_per_prompt",
            "generator",
            "latents",
            "output_type",
            "return_dict",
            "callback",
            "callback_steps",
        ]
    )
    # There is not xformers version of the StableAudioPipeline custom attention processor
    test_xformers_attention = False
    supports_dduf = False

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = StableAudioDiTModel(
            sample_size=4,
            in_channels=3,
            num_layers=2,
            attention_head_dim=4,
            num_key_value_attention_heads=2,
            out_channels=3,
            cross_attention_dim=4,
            time_proj_dim=8,
            global_states_input_dim=8,
            cross_attention_input_dim=4,
        )
        scheduler = CosineDPMSolverMultistepScheduler(
            solver_order=2,
            prediction_type="v_prediction",
            sigma_data=1.0,
            sigma_schedule="exponential",
        )
        torch.manual_seed(0)
        vae = AutoencoderOobleck(
            encoder_hidden_size=6,
            downsampling_ratios=[1, 2],
            decoder_channels=3,
            decoder_input_channels=3,
            audio_channels=2,
            channel_multiples=[2, 4],
            sampling_rate=4,
        )
        torch.manual_seed(0)
        t5_repo_id = "hf-internal-testing/tiny-random-T5ForConditionalGeneration"
        text_encoder = T5EncoderModel.from_pretrained(t5_repo_id)
        tokenizer = T5Tokenizer.from_pretrained(t5_repo_id, truncation=True, model_max_length=25)

        torch.manual_seed(0)
        projection_model = StableAudioProjectionModel(
            text_encoder_dim=text_encoder.config.d_model,
            conditioning_dim=4,
            min_value=0,
            max_value=32,
        )

        components = {
            "transformer": transformer,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "projection_model": projection_model,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A hammer hitting a wooden surface",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
        }
        return inputs

    def test_save_load_local(self):
        # increase tolerance from 1e-4 -> 7e-3 to account for large composite model
        super().test_save_load_local(expected_max_difference=7e-3)

    def test_save_load_optional_components(self):
        # increase tolerance from 1e-4 -> 7e-3 to account for large composite model
        super().test_save_load_optional_components(expected_max_difference=7e-3)

    def test_stable_audio_ddim(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components()
        stable_audio_pipe = StableAudioPipeline(**components)
        stable_audio_pipe = stable_audio_pipe.to(torch_device)
        stable_audio_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = stable_audio_pipe(**inputs)
        audio = output.audios[0]

        assert audio.ndim == 2
        assert audio.shape == (2, 7)

    def test_stable_audio_without_prompts(self):
        components = self.get_dummy_components()
        stable_audio_pipe = StableAudioPipeline(**components)
        stable_audio_pipe = stable_audio_pipe.to(torch_device)
        stable_audio_pipe = stable_audio_pipe.to(torch_device)
        stable_audio_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        inputs["prompt"] = 3 * [inputs["prompt"]]

        # forward
        output = stable_audio_pipe(**inputs)
        audio_1 = output.audios[0]

        inputs = self.get_dummy_inputs(torch_device)
        prompt = 3 * [inputs.pop("prompt")]

        text_inputs = stable_audio_pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=stable_audio_pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(torch_device)
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask

        prompt_embeds = stable_audio_pipe.text_encoder(
            text_input_ids,
            attention_mask=attention_mask,
        )[0]

        inputs["prompt_embeds"] = prompt_embeds
        inputs["attention_mask"] = attention_mask

        # forward
        output = stable_audio_pipe(**inputs)
        audio_2 = output.audios[0]

        assert (audio_1 - audio_2).abs().max() < 1e-2

    def test_stable_audio_negative_without_prompts(self):
        components = self.get_dummy_components()
        stable_audio_pipe = StableAudioPipeline(**components)
        stable_audio_pipe = stable_audio_pipe.to(torch_device)
        stable_audio_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        negative_prompt = 3 * ["this is a negative prompt"]
        inputs["negative_prompt"] = negative_prompt
        inputs["prompt"] = 3 * [inputs["prompt"]]

        # forward
        output = stable_audio_pipe(**inputs)
        audio_1 = output.audios[0]

        inputs = self.get_dummy_inputs(torch_device)
        prompt = 3 * [inputs.pop("prompt")]

        text_inputs = stable_audio_pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=stable_audio_pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(torch_device)
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask

        prompt_embeds = stable_audio_pipe.text_encoder(
            text_input_ids,
            attention_mask=attention_mask,
        )[0]

        inputs["prompt_embeds"] = prompt_embeds
        inputs["attention_mask"] = attention_mask

        negative_text_inputs = stable_audio_pipe.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=stable_audio_pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(torch_device)
        negative_text_input_ids = negative_text_inputs.input_ids
        negative_attention_mask = negative_text_inputs.attention_mask

        negative_prompt_embeds = stable_audio_pipe.text_encoder(
            negative_text_input_ids,
            attention_mask=negative_attention_mask,
        )[0]

        inputs["negative_prompt_embeds"] = negative_prompt_embeds
        inputs["negative_attention_mask"] = negative_attention_mask

        # forward
        output = stable_audio_pipe(**inputs)
        audio_2 = output.audios[0]

        assert (audio_1 - audio_2).abs().max() < 1e-2

    def test_stable_audio_negative_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        stable_audio_pipe = StableAudioPipeline(**components)
        stable_audio_pipe = stable_audio_pipe.to(device)
        stable_audio_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        negative_prompt = "egg cracking"
        output = stable_audio_pipe(**inputs, negative_prompt=negative_prompt)
        audio = output.audios[0]

        assert audio.ndim == 2
        assert audio.shape == (2, 7)

    def test_stable_audio_num_waveforms_per_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        stable_audio_pipe = StableAudioPipeline(**components)
        stable_audio_pipe = stable_audio_pipe.to(device)
        stable_audio_pipe.set_progress_bar_config(disable=None)

        prompt = "A hammer hitting a wooden surface"

        # test num_waveforms_per_prompt=1 (default)
        audios = stable_audio_pipe(prompt, num_inference_steps=2).audios

        assert audios.shape == (1, 2, 7)

        # test num_waveforms_per_prompt=1 (default) for batch of prompts
        batch_size = 2
        audios = stable_audio_pipe([prompt] * batch_size, num_inference_steps=2).audios

        assert audios.shape == (batch_size, 2, 7)

        # test num_waveforms_per_prompt for single prompt
        num_waveforms_per_prompt = 2
        audios = stable_audio_pipe(
            prompt, num_inference_steps=2, num_waveforms_per_prompt=num_waveforms_per_prompt
        ).audios

        assert audios.shape == (num_waveforms_per_prompt, 2, 7)

        # test num_waveforms_per_prompt for batch of prompts
        batch_size = 2
        audios = stable_audio_pipe(
            [prompt] * batch_size, num_inference_steps=2, num_waveforms_per_prompt=num_waveforms_per_prompt
        ).audios

        assert audios.shape == (batch_size * num_waveforms_per_prompt, 2, 7)

    def test_stable_audio_audio_end_in_s(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        stable_audio_pipe = StableAudioPipeline(**components)
        stable_audio_pipe = stable_audio_pipe.to(torch_device)
        stable_audio_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = stable_audio_pipe(audio_end_in_s=1.5, **inputs)
        audio = output.audios[0]

        assert audio.ndim == 2
        assert audio.shape[1] / stable_audio_pipe.vae.sampling_rate == 1.5

        output = stable_audio_pipe(audio_end_in_s=1.1875, **inputs)
        audio = output.audios[0]

        assert audio.ndim == 2
        assert audio.shape[1] / stable_audio_pipe.vae.sampling_rate == 1.0

    def test_attention_slicing_forward_pass(self):
        self._test_attention_slicing_forward_pass(test_mean_pixel_difference=False)

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=5e-4)

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_xformers_attention_forwardGenerator_pass(self):
        self._test_xformers_attention_forwardGenerator_pass(test_mean_pixel_difference=False)

    def test_stable_audio_input_waveform(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        stable_audio_pipe = StableAudioPipeline(**components)
        stable_audio_pipe = stable_audio_pipe.to(device)
        stable_audio_pipe.set_progress_bar_config(disable=None)

        prompt = "A hammer hitting a wooden surface"

        initial_audio_waveforms = torch.ones((1, 5))

        # test raises error when no sampling rate
        with self.assertRaises(ValueError):
            audios = stable_audio_pipe(
                prompt, num_inference_steps=2, initial_audio_waveforms=initial_audio_waveforms
            ).audios

        # test raises error when wrong sampling rate
        with self.assertRaises(ValueError):
            audios = stable_audio_pipe(
                prompt,
                num_inference_steps=2,
                initial_audio_waveforms=initial_audio_waveforms,
                initial_audio_sampling_rate=stable_audio_pipe.vae.sampling_rate - 1,
            ).audios

        audios = stable_audio_pipe(
            prompt,
            num_inference_steps=2,
            initial_audio_waveforms=initial_audio_waveforms,
            initial_audio_sampling_rate=stable_audio_pipe.vae.sampling_rate,
        ).audios
        assert audios.shape == (1, 2, 7)

        # test works with num_waveforms_per_prompt
        num_waveforms_per_prompt = 2
        audios = stable_audio_pipe(
            prompt,
            num_inference_steps=2,
            num_waveforms_per_prompt=num_waveforms_per_prompt,
            initial_audio_waveforms=initial_audio_waveforms,
            initial_audio_sampling_rate=stable_audio_pipe.vae.sampling_rate,
        ).audios

        assert audios.shape == (num_waveforms_per_prompt, 2, 7)

        # test num_waveforms_per_prompt for batch of prompts and input audio (two channels)
        batch_size = 2
        initial_audio_waveforms = torch.ones((batch_size, 2, 5))
        audios = stable_audio_pipe(
            [prompt] * batch_size,
            num_inference_steps=2,
            num_waveforms_per_prompt=num_waveforms_per_prompt,
            initial_audio_waveforms=initial_audio_waveforms,
            initial_audio_sampling_rate=stable_audio_pipe.vae.sampling_rate,
        ).audios

        assert audios.shape == (batch_size * num_waveforms_per_prompt, 2, 7)

    @unittest.skip("Not supported yet")
    def test_sequential_cpu_offload_forward_pass(self):
        pass

    @unittest.skip("Not supported yet")
    def test_sequential_offload_forward_pass_twice(self):
        pass

    @unittest.skip("Test not supported because `rotary_embed_dim` doesn't have any sensible default.")
    def test_encode_prompt_works_in_isolation(self):
        pass


@nightly
@require_torch_accelerator
class StableAudioPipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        latents = np.random.RandomState(seed).standard_normal((1, 64, 1024))
        latents = torch.from_numpy(latents).to(device=device, dtype=dtype)
        inputs = {
            "prompt": "A hammer hitting a wooden surface",
            "latents": latents,
            "generator": generator,
            "num_inference_steps": 3,
            "audio_end_in_s": 30,
            "guidance_scale": 2.5,
        }
        return inputs

    def test_stable_audio(self):
        stable_audio_pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0")
        stable_audio_pipe = stable_audio_pipe.to(torch_device)
        stable_audio_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        inputs["num_inference_steps"] = 25
        audio = stable_audio_pipe(**inputs).audios[0]

        assert audio.ndim == 2
        assert audio.shape == (2, int(inputs["audio_end_in_s"] * stable_audio_pipe.vae.sampling_rate))
        # check the portion of the generated audio with the largest dynamic range (reduces flakiness)
        audio_slice = audio[0, 447590:447600]
        # fmt: off
        expected_slices = Expectations(
            {
                ("xpu", 3): np.array([-0.0285, 0.1083, 0.1863, 0.3165, 0.5312, 0.6971, 0.6958, 0.6177, 0.5598, 0.5048]),
                ("cuda", 7): np.array([-0.0278, 0.1096, 0.1877, 0.3178, 0.5329, 0.6990, 0.6972, 0.6186, 0.5608, 0.5060]),
                ("cuda", 8): np.array([-0.0285, 0.1082, 0.1862, 0.3163, 0.5306, 0.6964, 0.6953, 0.6172, 0.5593, 0.5044]),
            }
        )
        # fmt: on

        expected_slice = expected_slices.get_expectation()
        max_diff = np.abs(expected_slice - audio_slice.detach().cpu().numpy()).max()
        assert max_diff < 1.5e-3
