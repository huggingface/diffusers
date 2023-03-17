# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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
import time
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    ClapTextConfig,
    ClapTextModelWithProjection,
    RobertaTokenizer,
    SpeechT5HifiGan,
    SpeechT5HifiGanConfig,
)

from diffusers import (
    AudioLDMPipeline,
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
    logging,
)
from diffusers.utils import load_numpy, nightly, slow, torch_device
from diffusers.utils.testing_utils import CaptureLogger, require_torch_gpu

from ...test_pipelines_common import PipelineTesterMixin


class AudioLDMPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = AudioLDMPipeline

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=(32, 64),
            class_embed_type="simple_projection",
            projection_class_embeddings_input_dim=32,
            class_embeddings_concat=True,
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
        text_encoder_config = ClapTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            projection_dim=32,
        )
        text_encoder = ClapTextModelWithProjection(text_encoder_config)
        tokenizer = RobertaTokenizer.from_pretrained("hf-internal-testing/tiny-random-roberta", model_max_length=77)

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
            "tokenizer": tokenizer,
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

    def test_audioldm_ddim(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components()
        audioldm_pipe = AudioLDMPipeline(**components)
        audioldm_pipe = audioldm_pipe.to(torch_device)
        audioldm_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = audioldm_pipe(**inputs)
        audio = output.audios[0]

        assert audio.ndim == 1
        assert len(audio) == 256

        audio_slice = audio[:10]
        expected_slice = np.array(
            [-0.0050, 0.0050, -0.0060, 0.0033, -0.0026, 0.0033, -0.0027, 0.0033, -0.0028, 0.0033]
        )

        assert np.abs(audio_slice - expected_slice).max() < 1e-3

    def test_audioldm_prompt_embeds(self):
        components = self.get_dummy_components()
        audioldm_pipe = AudioLDMPipeline(**components)
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

        prompt_embeds = audioldm_pipe.text_encoder(
            text_inputs,
        )
        prompt_embeds = prompt_embeds.text_embeds
        # additional L_2 normalization over each hidden-state
        prompt_embeds = F.normalize(prompt_embeds, dim=-1)

        inputs["prompt_embeds"] = prompt_embeds

        # forward
        output = audioldm_pipe(**inputs)
        audio_2 = output.audios[0]

        assert np.abs(audio_1 - audio_2).max() < 1e-4

    def test_audioldm_negative_prompt_embeds(self):
        components = self.get_dummy_components()
        audioldm_pipe = AudioLDMPipeline(**components)
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
        for p in [prompt, negative_prompt]:
            text_inputs = audioldm_pipe.tokenizer(
                p,
                padding="max_length",
                max_length=audioldm_pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_inputs = text_inputs["input_ids"].to(torch_device)

            text_embeds = audioldm_pipe.text_encoder(
                text_inputs,
            )
            text_embeds = text_embeds.text_embeds
            # additional L_2 normalization over each hidden-state
            text_embeds = F.normalize(text_embeds, dim=-1)

            embeds.append(text_embeds)

        inputs["prompt_embeds"], inputs["negative_prompt_embeds"] = embeds

        # forward
        output = audioldm_pipe(**inputs)
        audio_2 = output.audios[0]

        assert np.abs(audio_1 - audio_2).max() < 1e-4

    def test_audioldm_ddim_factor_8(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components()
        audioldm_pipe = AudioLDMPipeline(**components)
        audioldm_pipe = audioldm_pipe.to(device)
        audioldm_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = audioldm_pipe(**inputs, height=136)  # width has to stay fixed for the vocoder
        audio = output.audios[0]

        assert audio.ndim == 1
        assert len(audio) == 544

        audio_slice = audio[-10:]
        expected_slice = np.array([-0.0029, 0.0036, -0.0027, 0.0032, -0.0029, 0.0034, -0.0028, 0.0073, 0.0039, 0.0058])

        assert np.abs(audio_slice - expected_slice).max() < 1e-3

    def test_audioldm_pndm(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        audioldm_pipe = AudioLDMPipeline(**components)
        audioldm_pipe.scheduler = PNDMScheduler(skip_prk_steps=True)
        audioldm_pipe = audioldm_pipe.to(device)
        audioldm_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = audioldm_pipe(**inputs)
        audio = output.audios[0]

        assert audio.ndim == 1
        assert len(audio) == 256

        audio_slice = audio[:10]
        expected_slice = np.array(
            [-0.0051, 0.0050, -0.0060, 0.0034, -0.0026, 0.0033, -0.0027, 0.0033, -0.0028, 0.0032]
        )

        assert np.abs(audio_slice - expected_slice).max() < 1e-3

    def test_audioldm_k_lms(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components()
        audioldm_pipe = AudioLDMPipeline(**components)
        audioldm_pipe.scheduler = LMSDiscreteScheduler.from_config(audioldm_pipe.scheduler.config)
        audioldm_pipe = audioldm_pipe.to(device)
        audioldm_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = audioldm_pipe(**inputs)
        audio = output.audios[0]

        assert audio.ndim == 1
        assert len(audio) == 256

        audio_slice = audio[:10]
        expected_slice = np.array(
            [-0.0051, 0.0050, -0.0060, 0.0034, -0.0026, 0.0033, -0.0027, 0.0033, -0.0028, 0.0032]
        )

        assert np.abs(audio_slice - expected_slice).max() < 1e-3

    def test_audioldm_k_euler_ancestral(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components()
        audioldm_pipe = AudioLDMPipeline(**components)
        audioldm_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(audioldm_pipe.scheduler.config)
        audioldm_pipe = audioldm_pipe.to(device)
        audioldm_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = audioldm_pipe(**inputs)
        audio = output.audios[0]

        assert audio.ndim == 1
        assert len(audio) == 256

        audio_slice = audio[:10]
        expected_slice = np.array(
            [-0.0051, 0.0050, -0.0060, 0.0034, -0.0026, 0.0033, -0.0027, 0.0033, -0.0028, 0.0032]
        )

        assert np.abs(audio_slice - expected_slice).max() < 1e-3

    def test_audioldm_k_euler(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components()
        audioldm_pipe = AudioLDMPipeline(**components)
        audioldm_pipe.scheduler = EulerDiscreteScheduler.from_config(audioldm_pipe.scheduler.config)
        audioldm_pipe = audioldm_pipe.to(device)
        audioldm_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = audioldm_pipe(**inputs)
        audio = output.audios[0]

        assert audio.ndim == 1
        assert len(audio) == 256

        audio_slice = audio[:10]
        expected_slice = np.array(
            [-0.0051, 0.0050, -0.0060, 0.0034, -0.0026, 0.0033, -0.0027, 0.0033, -0.0028, 0.0032]
        )

        assert np.abs(audio_slice - expected_slice).max() < 1e-3

    def test_audioldm_vae_slicing(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        components["scheduler"] = LMSDiscreteScheduler.from_config(components["scheduler"].config)
        audioldm_pipe = AudioLDMPipeline(**components)
        audioldm_pipe = audioldm_pipe.to(device)
        audioldm_pipe.set_progress_bar_config(disable=None)

        image_count = 4

        inputs = self.get_dummy_inputs(device)
        inputs["prompt"] = [inputs["prompt"]] * image_count
        output_1 = audioldm_pipe(**inputs)

        # make sure sliced vae decode yields the same result
        audioldm_pipe.enable_vae_slicing()
        inputs = self.get_dummy_inputs(device)
        inputs["prompt"] = [inputs["prompt"]] * image_count
        output_2 = audioldm_pipe(**inputs)

        # there is a small discrepancy at spectrogram borders vs. full batch decode
        assert np.abs(output_2.audios - output_1.audios).max() < 1e-4

    def test_audioldm_negative_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        components["scheduler"] = PNDMScheduler(skip_prk_steps=True)
        audioldm_pipe = AudioLDMPipeline(**components)
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
            [-0.0051, 0.0050, -0.0060, 0.0034, -0.0026, 0.0033, -0.0027, 0.0033, -0.0028, 0.0032]
        )

        assert np.abs(audio_slice - expected_slice).max() < 1e-3

    def test_audioldm_num_waveforms_per_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        components["scheduler"] = PNDMScheduler(skip_prk_steps=True)
        audioldm_pipe = AudioLDMPipeline(**components)
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

    def test_audioldm_long_prompt(self):
        components = self.get_dummy_components()
        components["scheduler"] = LMSDiscreteScheduler.from_config(components["scheduler"].config)
        audioldm_pipe = AudioLDMPipeline(**components)
        audioldm_pipe = audioldm_pipe.to(torch_device)
        audioldm_pipe.set_progress_bar_config(disable=None)

        do_classifier_free_guidance = True
        negative_prompt = None
        num_images_per_prompt = 1
        logger = logging.get_logger("diffusers.pipelines.audioldm.pipeline_audioldm")

        prompt = 25 * "@"
        with CaptureLogger(logger) as cap_logger_3:
            text_embeddings_3 = audioldm_pipe._encode_prompt(
                prompt, torch_device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )

        prompt = 100 * "@"
        with CaptureLogger(logger) as cap_logger:
            text_embeddings = audioldm_pipe._encode_prompt(
                prompt, torch_device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )

        negative_prompt = "Hello"
        with CaptureLogger(logger) as cap_logger_2:
            text_embeddings_2 = audioldm_pipe._encode_prompt(
                prompt, torch_device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )

        assert text_embeddings_3.shape == text_embeddings_2.shape == text_embeddings.shape

        assert text_embeddings.shape[1] == 32

        assert cap_logger.out == cap_logger_2.out
        # 100 - 77 + 1 (BOS token) + 1 (EOS token) = 25
        assert cap_logger.out.count("@") == 25
        assert cap_logger_3.out == ""

    def test_audioldm_height_opt(self):
        components = self.get_dummy_components()
        components["scheduler"] = LMSDiscreteScheduler.from_config(components["scheduler"].config)
        audioldm_pipe = AudioLDMPipeline(**components)
        audioldm_pipe = audioldm_pipe.to(torch_device)
        audioldm_pipe.set_progress_bar_config(disable=None)

        prompt = ["hey"]

        output = audioldm_pipe(prompt, num_inference_steps=1)
        audio_shape = output.audios.shape
        assert audio_shape == (1, 256)

        output = audioldm_pipe(prompt, num_inference_steps=1, height=96, width=8)
        audio_shape = output.audios.shape
        assert audio_shape == (1, 384)

        config = dict(audioldm_pipe.unet.config)
        config["sample_size"] = 96
        audioldm_pipe.unet = UNet2DConditionModel.from_config(config).to(torch_device)
        output = audioldm_pipe(prompt, num_inference_steps=1, width=8)  # need to keep width fixed for vocoder
        audio_shape = output.audios.shape
        assert audio_shape == (1, 768)

    def test_audioldm_width_opt(self):
        components = self.get_dummy_components()
        components["scheduler"] = LMSDiscreteScheduler.from_config(components["scheduler"].config)
        audioldm_pipe = AudioLDMPipeline(**components)
        audioldm_pipe = audioldm_pipe.to(torch_device)
        audioldm_pipe.set_progress_bar_config(disable=None)

        prompt = ["hey"]

        width = audioldm_pipe.vocoder.config.model_in_dim

        output = audioldm_pipe(prompt, num_inference_steps=1, width=width)
        audio_shape = output.audios.shape
        assert audio_shape == (1, 256)

        config = audioldm_pipe.vocoder.config
        config.model_in_dim = width * 2
        audioldm_pipe.vocoder = SpeechT5HifiGan(config).to(torch_device)
        output = audioldm_pipe(prompt, num_inference_steps=1, width=width * 2)
        audio_shape = output.audios.shape
        # waveform shape is unchanged, we just have 2x the number of mel channels in the spectrogram
        assert audio_shape == (1, 256)

    def test_attention_slicing_forward_pass(self):
        self._test_attention_slicing_forward_pass(test_mean_pixel_difference=False)

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(test_mean_pixel_difference=False)


@slow
@require_torch_gpu
class AudioLDMPipelineSlowTests(unittest.TestCase):
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

    def test_audioldm_ddim(self):
        audioldm_pipe = AudioLDMPipeline.from_pretrained("cvssp/audioldm")
        audioldm_pipe.scheduler = DDIMScheduler.from_config(audioldm_pipe.scheduler.config)
        audioldm_pipe = audioldm_pipe.to(torch_device)
        audioldm_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        audio = audioldm_pipe(**inputs).audios[0]

        assert audio.ndim == 1
        assert len(audio) == 81952

        audio_slice = audio[3880:3890]
        expected_slice = np.array([-0.0574, 0.2462, 0.3955, 0.4213, 0.3901, 0.3770, 0.2762, 0.0206, -0.2208, -0.3282])
        max_diff = np.abs(expected_slice - audio_slice).max()
        assert max_diff < 1e-3

    def test_audioldm_lms(self):
        audioldm_pipe = AudioLDMPipeline.from_pretrained("cvssp/audioldm")
        audioldm_pipe.scheduler = LMSDiscreteScheduler.from_config(audioldm_pipe.scheduler.config)
        audioldm_pipe = audioldm_pipe.to(torch_device)
        audioldm_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        audio = audioldm_pipe(**inputs).audios[0]

        assert audio.ndim == 1
        assert len(audio) == 81952

        audio_slice = audio[27780:27790]
        expected_slice = np.array([-0.2131, -0.0873, -0.0124, -0.0189, 0.0569, 0.1373, 0.1883, 0.2886, 0.3297, 0.2212])
        max_diff = np.abs(expected_slice - audio_slice).max()
        assert max_diff < 1e-3

    def test_audioldm_dpm(self):
        audioldm_pipe = AudioLDMPipeline.from_pretrained("cvssp/audioldm")
        audioldm_pipe.scheduler = DPMSolverMultistepScheduler.from_config(audioldm_pipe.scheduler.config)
        audioldm_pipe = audioldm_pipe.to(torch_device)
        audioldm_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        audio = audioldm_pipe(**inputs).audios[0]

        assert audio.ndim == 1
        assert len(audio) == 81952

        audio_slice = audio[69310:69320]
        expected_slice = np.array([0.1842, 0.2411, 0.3127, 0.3069, 0.2287, 0.0948, -0.0071, -0.041, -0.1293, -0.2075])
        max_diff = np.abs(expected_slice - audio_slice).max()
        assert max_diff < 1e-3

    @unittest.skip("TODO(SG): fix or remove. This test yields the same memory for with / without attn slicing")
    def test_audioldm_attention_slicing(self):
        torch.cuda.reset_peak_memory_stats()
        pipe = AudioLDMPipeline.from_pretrained("cvssp/audioldm", torch_dtype=torch.float16)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # enable attention slicing
        pipe.enable_attention_slicing(slice_size="max")
        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        audios_sliced = pipe(**inputs).audios

        mem_bytes = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        # make sure that less than 3.75 GB is allocated
        assert mem_bytes < 3.75 * 10**9

        # disable slicing
        pipe.disable_attention_slicing()
        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        audios = pipe(**inputs).audios

        # make sure that more than 3.75 GB is allocated
        mem_bytes = torch.cuda.max_memory_allocated()
        assert mem_bytes > 3.75 * 10**9
        assert np.abs(audios_sliced - audios).max() < 1e-3

    def test_audioldm_vae_slicing(self):
        torch.cuda.reset_peak_memory_stats()
        pipe = AudioLDMPipeline.from_pretrained("cvssp/audioldm", torch_dtype=torch.float16)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        # enable vae slicing
        pipe.enable_vae_slicing()
        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        inputs["prompt"] = [inputs["prompt"]] * 4
        inputs["latents"] = torch.cat([inputs["latents"]] * 4)
        audio_sliced = pipe(**inputs).audios

        mem_bytes = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        # make sure that less than 1.1 GB is allocated
        assert mem_bytes < 1.1 * 10**9

        # disable vae slicing
        pipe.disable_vae_slicing()
        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        inputs["prompt"] = [inputs["prompt"]] * 4
        inputs["latents"] = torch.cat([inputs["latents"]] * 4)
        audio = pipe(**inputs).audios

        # make sure that more than 1.1 GB is allocated
        mem_bytes = torch.cuda.max_memory_allocated()
        assert mem_bytes > 1.1 * 10**9
        # There is a small discrepancy at the spectrogram borders vs. a fully batched version.
        assert np.abs(audio_sliced - audio).max() < 1e-2

    def test_audioldm_fp16_vs_autocast(self):
        # this test makes sure that the original model with autocast
        # and the new model with fp16 yield the same result
        pipe = AudioLDMPipeline.from_pretrained("cvssp/audioldm", torch_dtype=torch.float16)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        audio_fp16 = pipe(**inputs).audios

        with torch.autocast(torch_device):
            inputs = self.get_inputs(torch_device)
            audio_autocast = pipe(**inputs).audios

        # Make sure results are close enough
        diff = np.abs(audio_fp16.flatten() - audio_autocast.flatten())
        # They ARE different since ops are not run always at the same precision
        # however, they should be extremely close.
        assert diff.mean() < 1e-3

    def test_audioldm_intermediate_state(self):
        number_of_steps = 0

        def callback_fn(step: int, timestep: int, latents: torch.FloatTensor) -> None:
            callback_fn.has_been_called = True
            nonlocal number_of_steps
            number_of_steps += 1
            if step == 1:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 8, 128, 16)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([-0.6730, -0.9062, 1.0400, 0.4220, -0.9785, 1.817, 0.1906, -1.3430, 1.3330])

                assert np.abs(latents_slice.flatten() - expected_slice).max() < 1e-3
            elif step == 2:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 8, 128, 16)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([-0.6763, -0.9062, 1.0520, 0.4200, -0.9750, 1.8220, 0.1879, -1.3490, 1.3190])

                assert np.abs(latents_slice.flatten() - expected_slice).max() < 1e-3

        callback_fn.has_been_called = False

        pipe = AudioLDMPipeline.from_pretrained("cvssp/audioldm", torch_dtype=torch.float16)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        pipe(**inputs, callback=callback_fn, callback_steps=1)
        assert callback_fn.has_been_called
        assert number_of_steps == inputs["num_inference_steps"]

    def test_audioldm_low_cpu_mem_usage(self):
        pipeline_id = "cvssp/audioldm"

        start_time = time.time()
        pipeline_low_cpu_mem_usage = AudioLDMPipeline.from_pretrained(pipeline_id, torch_dtype=torch.float16)
        pipeline_low_cpu_mem_usage.to(torch_device)
        low_cpu_mem_usage_time = time.time() - start_time

        start_time = time.time()
        _ = AudioLDMPipeline.from_pretrained(pipeline_id, torch_dtype=torch.float16, low_cpu_mem_usage=False)
        normal_load_time = time.time() - start_time

        assert 2 * low_cpu_mem_usage_time < normal_load_time

    def test_audioldm_pipeline_with_sequential_cpu_offloading(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        pipe = AudioLDMPipeline.from_pretrained("cvssp/audioldm", torch_dtype=torch.float16)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing(1)
        pipe.enable_sequential_cpu_offload()

        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        _ = pipe(**inputs)

        mem_bytes = torch.cuda.max_memory_allocated()
        # make sure that less than 2.8 GB is allocated
        assert mem_bytes < 2.8 * 10**9
