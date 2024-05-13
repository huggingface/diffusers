# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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
    ClapAudioConfig,
    ClapConfig,
    ClapFeatureExtractor,
    ClapModel,
    ClapTextConfig,
    GPT2Config,
    GPT2Model,
    RobertaTokenizer,
    SpeechT5HifiGan,
    SpeechT5HifiGanConfig,
    T5Config,
    T5EncoderModel,
    T5Tokenizer,
)

from diffusers import (
    AudioLDM2Pipeline,
    AudioLDM2ProjectionModel,
    AudioLDM2UNet2DConditionModel,
    AutoencoderKL,
    DDIMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils.testing_utils import enable_full_determinism, nightly, torch_device

from ..pipeline_params import TEXT_TO_AUDIO_BATCH_PARAMS, TEXT_TO_AUDIO_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class AudioLDM2PipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = AudioLDM2Pipeline
    params = TEXT_TO_AUDIO_PARAMS
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

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = AudioLDM2UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=([None, 16, 32], [None, 16, 32]),
        )
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=1,
            out_channels=1,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
        torch.manual_seed(0)
        text_branch_config = ClapTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=16,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=2,
            num_hidden_layers=2,
            pad_token_id=1,
            vocab_size=1000,
            projection_dim=16,
        )
        audio_branch_config = ClapAudioConfig(
            spec_size=64,
            window_size=4,
            num_mel_bins=64,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            depths=[2, 2],
            num_attention_heads=[2, 2],
            num_hidden_layers=2,
            hidden_size=192,
            projection_dim=16,
            patch_size=2,
            patch_stride=2,
            patch_embed_input_channels=4,
        )
        text_encoder_config = ClapConfig.from_text_audio_configs(
            text_config=text_branch_config, audio_config=audio_branch_config, projection_dim=16
        )
        text_encoder = ClapModel(text_encoder_config)
        tokenizer = RobertaTokenizer.from_pretrained("hf-internal-testing/tiny-random-roberta", model_max_length=77)
        feature_extractor = ClapFeatureExtractor.from_pretrained(
            "hf-internal-testing/tiny-random-ClapModel", hop_length=7900
        )

        torch.manual_seed(0)
        text_encoder_2_config = T5Config(
            vocab_size=32100,
            d_model=32,
            d_ff=37,
            d_kv=8,
            num_heads=2,
            num_layers=2,
        )
        text_encoder_2 = T5EncoderModel(text_encoder_2_config)
        tokenizer_2 = T5Tokenizer.from_pretrained("hf-internal-testing/tiny-random-T5Model", model_max_length=77)

        torch.manual_seed(0)
        language_model_config = GPT2Config(
            n_embd=16,
            n_head=2,
            n_layer=2,
            vocab_size=1000,
            n_ctx=99,
            n_positions=99,
        )
        language_model = GPT2Model(language_model_config)
        language_model.config.max_new_tokens = 8

        torch.manual_seed(0)
        projection_model = AudioLDM2ProjectionModel(text_encoder_dim=16, text_encoder_1_dim=32, langauge_model_dim=16)

        vocoder_config = SpeechT5HifiGanConfig(
            model_in_dim=8,
            sampling_rate=16000,
            upsample_initial_channel=16,
            upsample_rates=[2, 2],
            upsample_kernel_sizes=[4, 4],
            resblock_kernel_sizes=[3, 7],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]],
            normalize_before=False,
        )

        vocoder = SpeechT5HifiGan(vocoder_config)

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "feature_extractor": feature_extractor,
            "language_model": language_model,
            "projection_model": projection_model,
            "vocoder": vocoder,
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

    def test_audioldm2_ddim(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components()
        audioldm_pipe = AudioLDM2Pipeline(**components)
        audioldm_pipe = audioldm_pipe.to(torch_device)
        audioldm_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = audioldm_pipe(**inputs)
        audio = output.audios[0]

        assert audio.ndim == 1
        assert len(audio) == 256

        audio_slice = audio[:10]
        expected_slice = np.array(
            [0.0025, 0.0018, 0.0018, -0.0023, -0.0026, -0.0020, -0.0026, -0.0021, -0.0027, -0.0020]
        )

        assert np.abs(audio_slice - expected_slice).max() < 1e-4

    def test_audioldm2_prompt_embeds(self):
        components = self.get_dummy_components()
        audioldm_pipe = AudioLDM2Pipeline(**components)
        audioldm_pipe = audioldm_pipe.to(torch_device)
        audioldm_pipe = audioldm_pipe.to(torch_device)
        audioldm_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        inputs["prompt"] = 3 * [inputs["prompt"]]

        # forward
        output = audioldm_pipe(**inputs)
        audio_1 = output.audios[0]

        inputs = self.get_dummy_inputs(torch_device)
        prompt = 3 * [inputs.pop("prompt")]

        text_inputs = audioldm_pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=audioldm_pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = text_inputs["input_ids"].to(torch_device)

        clap_prompt_embeds = audioldm_pipe.text_encoder.get_text_features(text_inputs)
        clap_prompt_embeds = clap_prompt_embeds[:, None, :]

        text_inputs = audioldm_pipe.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=True,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = text_inputs["input_ids"].to(torch_device)

        t5_prompt_embeds = audioldm_pipe.text_encoder_2(
            text_inputs,
        )
        t5_prompt_embeds = t5_prompt_embeds[0]

        projection_embeds = audioldm_pipe.projection_model(clap_prompt_embeds, t5_prompt_embeds)[0]
        generated_prompt_embeds = audioldm_pipe.generate_language_model(projection_embeds, max_new_tokens=8)

        inputs["prompt_embeds"] = t5_prompt_embeds
        inputs["generated_prompt_embeds"] = generated_prompt_embeds

        # forward
        output = audioldm_pipe(**inputs)
        audio_2 = output.audios[0]

        assert np.abs(audio_1 - audio_2).max() < 1e-2

    def test_audioldm2_negative_prompt_embeds(self):
        components = self.get_dummy_components()
        audioldm_pipe = AudioLDM2Pipeline(**components)
        audioldm_pipe = audioldm_pipe.to(torch_device)
        audioldm_pipe = audioldm_pipe.to(torch_device)
        audioldm_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        negative_prompt = 3 * ["this is a negative prompt"]
        inputs["negative_prompt"] = negative_prompt
        inputs["prompt"] = 3 * [inputs["prompt"]]

        # forward
        output = audioldm_pipe(**inputs)
        audio_1 = output.audios[0]

        inputs = self.get_dummy_inputs(torch_device)
        prompt = 3 * [inputs.pop("prompt")]

        embeds = []
        generated_embeds = []
        for p in [prompt, negative_prompt]:
            text_inputs = audioldm_pipe.tokenizer(
                p,
                padding="max_length",
                max_length=audioldm_pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_inputs = text_inputs["input_ids"].to(torch_device)

            clap_prompt_embeds = audioldm_pipe.text_encoder.get_text_features(text_inputs)
            clap_prompt_embeds = clap_prompt_embeds[:, None, :]

            text_inputs = audioldm_pipe.tokenizer_2(
                prompt,
                padding="max_length",
                max_length=True if len(embeds) == 0 else embeds[0].shape[1],
                truncation=True,
                return_tensors="pt",
            )
            text_inputs = text_inputs["input_ids"].to(torch_device)

            t5_prompt_embeds = audioldm_pipe.text_encoder_2(
                text_inputs,
            )
            t5_prompt_embeds = t5_prompt_embeds[0]

            projection_embeds = audioldm_pipe.projection_model(clap_prompt_embeds, t5_prompt_embeds)[0]
            generated_prompt_embeds = audioldm_pipe.generate_language_model(projection_embeds, max_new_tokens=8)

            embeds.append(t5_prompt_embeds)
            generated_embeds.append(generated_prompt_embeds)

        inputs["prompt_embeds"], inputs["negative_prompt_embeds"] = embeds
        inputs["generated_prompt_embeds"], inputs["negative_generated_prompt_embeds"] = generated_embeds

        # forward
        output = audioldm_pipe(**inputs)
        audio_2 = output.audios[0]

        assert np.abs(audio_1 - audio_2).max() < 1e-2

    def test_audioldm2_negative_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        components["scheduler"] = PNDMScheduler(skip_prk_steps=True)
        audioldm_pipe = AudioLDM2Pipeline(**components)
        audioldm_pipe = audioldm_pipe.to(device)
        audioldm_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        negative_prompt = "egg cracking"
        output = audioldm_pipe(**inputs, negative_prompt=negative_prompt)
        audio = output.audios[0]

        assert audio.ndim == 1
        assert len(audio) == 256

        audio_slice = audio[:10]
        expected_slice = np.array(
            [0.0025, 0.0018, 0.0018, -0.0023, -0.0026, -0.0020, -0.0026, -0.0021, -0.0027, -0.0020]
        )

        assert np.abs(audio_slice - expected_slice).max() < 1e-4

    def test_audioldm2_num_waveforms_per_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        components["scheduler"] = PNDMScheduler(skip_prk_steps=True)
        audioldm_pipe = AudioLDM2Pipeline(**components)
        audioldm_pipe = audioldm_pipe.to(device)
        audioldm_pipe.set_progress_bar_config(disable=None)

        prompt = "A hammer hitting a wooden surface"

        # test num_waveforms_per_prompt=1 (default)
        audios = audioldm_pipe(prompt, num_inference_steps=2).audios

        assert audios.shape == (1, 256)

        # test num_waveforms_per_prompt=1 (default) for batch of prompts
        batch_size = 2
        audios = audioldm_pipe([prompt] * batch_size, num_inference_steps=2).audios

        assert audios.shape == (batch_size, 256)

        # test num_waveforms_per_prompt for single prompt
        num_waveforms_per_prompt = 2
        audios = audioldm_pipe(prompt, num_inference_steps=2, num_waveforms_per_prompt=num_waveforms_per_prompt).audios

        assert audios.shape == (num_waveforms_per_prompt, 256)

        # test num_waveforms_per_prompt for batch of prompts
        batch_size = 2
        audios = audioldm_pipe(
            [prompt] * batch_size, num_inference_steps=2, num_waveforms_per_prompt=num_waveforms_per_prompt
        ).audios

        assert audios.shape == (batch_size * num_waveforms_per_prompt, 256)

    def test_audioldm2_audio_length_in_s(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        audioldm_pipe = AudioLDM2Pipeline(**components)
        audioldm_pipe = audioldm_pipe.to(torch_device)
        audioldm_pipe.set_progress_bar_config(disable=None)
        vocoder_sampling_rate = audioldm_pipe.vocoder.config.sampling_rate

        inputs = self.get_dummy_inputs(device)
        output = audioldm_pipe(audio_length_in_s=0.016, **inputs)
        audio = output.audios[0]

        assert audio.ndim == 1
        assert len(audio) / vocoder_sampling_rate == 0.016

        output = audioldm_pipe(audio_length_in_s=0.032, **inputs)
        audio = output.audios[0]

        assert audio.ndim == 1
        assert len(audio) / vocoder_sampling_rate == 0.032

    def test_audioldm2_vocoder_model_in_dim(self):
        components = self.get_dummy_components()
        audioldm_pipe = AudioLDM2Pipeline(**components)
        audioldm_pipe = audioldm_pipe.to(torch_device)
        audioldm_pipe.set_progress_bar_config(disable=None)

        prompt = ["hey"]

        output = audioldm_pipe(prompt, num_inference_steps=1)
        audio_shape = output.audios.shape
        assert audio_shape == (1, 256)

        config = audioldm_pipe.vocoder.config
        config.model_in_dim *= 2
        audioldm_pipe.vocoder = SpeechT5HifiGan(config).to(torch_device)
        output = audioldm_pipe(prompt, num_inference_steps=1)
        audio_shape = output.audios.shape
        # waveform shape is unchanged, we just have 2x the number of mel channels in the spectrogram
        assert audio_shape == (1, 256)

    def test_attention_slicing_forward_pass(self):
        self._test_attention_slicing_forward_pass(test_mean_pixel_difference=False)

    @unittest.skip("Raises a not implemented error in AudioLDM2")
    def test_xformers_attention_forwardGenerator_pass(self):
        pass

    def test_dict_tuple_outputs_equivalent(self):
        # increase tolerance from 1e-4 -> 2e-4 to account for large composite model
        super().test_dict_tuple_outputs_equivalent(expected_max_difference=2e-4)

    def test_inference_batch_single_identical(self):
        # increase tolerance from 1e-4 -> 2e-4 to account for large composite model
        self._test_inference_batch_single_identical(expected_max_diff=2e-4)

    def test_save_load_local(self):
        # increase tolerance from 1e-4 -> 2e-4 to account for large composite model
        super().test_save_load_local(expected_max_difference=2e-4)

    def test_save_load_optional_components(self):
        # increase tolerance from 1e-4 -> 2e-4 to account for large composite model
        super().test_save_load_optional_components(expected_max_difference=2e-4)

    def test_to_dtype(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        # The method component.dtype returns the dtype of the first parameter registered in the model, not the
        # dtype of the entire model. In the case of CLAP, the first parameter is a float64 constant (logit scale)
        model_dtypes = {key: component.dtype for key, component in components.items() if hasattr(component, "dtype")}

        # Without the logit scale parameters, everything is float32
        model_dtypes.pop("text_encoder")
        self.assertTrue(all(dtype == torch.float32 for dtype in model_dtypes.values()))

        # the CLAP sub-models are float32
        model_dtypes["clap_text_branch"] = components["text_encoder"].text_model.dtype
        self.assertTrue(all(dtype == torch.float32 for dtype in model_dtypes.values()))

        # Once we send to fp16, all params are in half-precision, including the logit scale
        pipe.to(dtype=torch.float16)
        model_dtypes = {key: component.dtype for key, component in components.items() if hasattr(component, "dtype")}
        self.assertTrue(all(dtype == torch.float16 for dtype in model_dtypes.values()))

    def test_sequential_cpu_offload_forward_pass(self):
        pass


@nightly
class AudioLDM2PipelineSlowTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        latents = np.random.RandomState(seed).standard_normal((1, 8, 128, 16))
        latents = torch.from_numpy(latents).to(device=device, dtype=dtype)
        inputs = {
            "prompt": "A hammer hitting a wooden surface",
            "latents": latents,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 2.5,
        }
        return inputs

    def get_inputs_tts(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        latents = np.random.RandomState(seed).standard_normal((1, 8, 128, 16))
        latents = torch.from_numpy(latents).to(device=device, dtype=dtype)
        inputs = {
            "prompt": "A men saying",
            "transcription": "hello my name is John",
            "latents": latents,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 2.5,
        }
        return inputs

    def test_audioldm2(self):
        audioldm_pipe = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2")
        audioldm_pipe = audioldm_pipe.to(torch_device)
        audioldm_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        inputs["num_inference_steps"] = 25
        audio = audioldm_pipe(**inputs).audios[0]

        assert audio.ndim == 1
        assert len(audio) == 81952

        # check the portion of the generated audio with the largest dynamic range (reduces flakiness)
        audio_slice = audio[17275:17285]
        expected_slice = np.array([0.0791, 0.0666, 0.1158, 0.1227, 0.1171, -0.2880, -0.1940, -0.0283, -0.0126, 0.1127])
        max_diff = np.abs(expected_slice - audio_slice).max()
        assert max_diff < 1e-3

    def test_audioldm2_lms(self):
        audioldm_pipe = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2")
        audioldm_pipe.scheduler = LMSDiscreteScheduler.from_config(audioldm_pipe.scheduler.config)
        audioldm_pipe = audioldm_pipe.to(torch_device)
        audioldm_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        audio = audioldm_pipe(**inputs).audios[0]

        assert audio.ndim == 1
        assert len(audio) == 81952

        # check the portion of the generated audio with the largest dynamic range (reduces flakiness)
        audio_slice = audio[31390:31400]
        expected_slice = np.array(
            [-0.1318, -0.0577, 0.0446, -0.0573, 0.0659, 0.1074, -0.2600, 0.0080, -0.2190, -0.4301]
        )
        max_diff = np.abs(expected_slice - audio_slice).max()
        assert max_diff < 1e-3

    def test_audioldm2_large(self):
        audioldm_pipe = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2-large")
        audioldm_pipe = audioldm_pipe.to(torch_device)
        audioldm_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        audio = audioldm_pipe(**inputs).audios[0]

        assert audio.ndim == 1
        assert len(audio) == 81952

        # check the portion of the generated audio with the largest dynamic range (reduces flakiness)
        audio_slice = audio[8825:8835]
        expected_slice = np.array(
            [-0.1829, -0.1461, 0.0759, -0.1493, -0.1396, 0.5783, 0.3001, -0.3038, -0.0639, -0.2244]
        )
        max_diff = np.abs(expected_slice - audio_slice).max()
        assert max_diff < 1e-3

    def test_audioldm2_tts(self):
        audioldm_tts_pipe = AudioLDM2Pipeline.from_pretrained("anhnct/audioldm2_gigaspeech")
        audioldm_tts_pipe = audioldm_tts_pipe.to(torch_device)
        audioldm_tts_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs_tts(torch_device)
        audio = audioldm_tts_pipe(**inputs).audios[0]

        assert audio.ndim == 1
        assert len(audio) == 81952

        # check the portion of the generated audio with the largest dynamic range (reduces flakiness)
        audio_slice = audio[8825:8835]
        expected_slice = np.array(
            [-0.1829, -0.1461, 0.0759, -0.1493, -0.1396, 0.5783, 0.3001, -0.3038, -0.0639, -0.2244]
        )
        max_diff = np.abs(expected_slice - audio_slice).max()
        assert max_diff < 1e-3
