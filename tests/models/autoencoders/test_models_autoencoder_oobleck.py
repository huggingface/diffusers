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

import torch
from datasets import load_dataset
from parameterized import parameterized

from diffusers import AutoencoderOobleck

from ...testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    floats_tensor,
    slow,
    torch_all_close,
    torch_device,
)
from ..test_modeling_common import ModelTesterMixin
from .testing_utils import AutoencoderTesterMixin


enable_full_determinism()


class AutoencoderOobleckTests(ModelTesterMixin, AutoencoderTesterMixin, unittest.TestCase):
    model_class = AutoencoderOobleck
    main_input_name = "sample"
    base_precision = 1e-2

    def get_autoencoder_oobleck_config(self, block_out_channels=None):
        init_dict = {
            "encoder_hidden_size": 12,
            "decoder_channels": 12,
            "decoder_input_channels": 6,
            "audio_channels": 2,
            "downsampling_ratios": [2, 4],
            "channel_multiples": [1, 2],
        }
        return init_dict

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 2
        seq_len = 24

        waveform = floats_tensor((batch_size, num_channels, seq_len)).to(torch_device)

        return {"sample": waveform, "sample_posterior": False}

    @property
    def input_shape(self):
        return (2, 24)

    @property
    def output_shape(self):
        return (2, 24)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = self.get_autoencoder_oobleck_config()
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_enable_disable_slicing(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        torch.manual_seed(0)
        model = self.model_class(**init_dict).to(torch_device)

        inputs_dict.update({"return_dict": False})

        torch.manual_seed(0)
        output_without_slicing = model(**inputs_dict, generator=torch.manual_seed(0))[0]

        torch.manual_seed(0)
        model.enable_slicing()
        output_with_slicing = model(**inputs_dict, generator=torch.manual_seed(0))[0]

        self.assertLess(
            (output_without_slicing.detach().cpu().numpy() - output_with_slicing.detach().cpu().numpy()).max(),
            0.5,
            "VAE slicing should not affect the inference results",
        )

        torch.manual_seed(0)
        model.disable_slicing()
        output_without_slicing_2 = model(**inputs_dict, generator=torch.manual_seed(0))[0]

        self.assertEqual(
            output_without_slicing.detach().cpu().numpy().all(),
            output_without_slicing_2.detach().cpu().numpy().all(),
            "Without slicing outputs should match with the outputs when slicing is manually disabled.",
        )

    @unittest.skip("No attention module used in this model")
    def test_set_attn_processor_for_determinism(self):
        return

    @unittest.skip(
        "Test not supported because of 'weight_norm_fwd_first_dim_kernel' not implemented for 'Float8_e4m3fn'"
    )
    def test_layerwise_casting_training(self):
        return super().test_layerwise_casting_training()

    @unittest.skip(
        "The convolution layers of AutoencoderOobleck are wrapped with torch.nn.utils.weight_norm. This causes the hook's pre_forward to not "
        "cast the module weights to compute_dtype (as required by forward pass). As a result, forward pass errors out. To fix:\n"
        "1. Make sure `nn::Module::to` works with `torch.nn.utils.weight_norm` wrapped convolution layer.\n"
        "2. Unskip this test."
    )
    def test_layerwise_casting_inference(self):
        pass

    @unittest.skip(
        "The convolution layers of AutoencoderOobleck are wrapped with torch.nn.utils.weight_norm. This causes the hook's pre_forward to not "
        "cast the module weights to compute_dtype (as required by forward pass). As a result, forward pass errors out. To fix:\n"
        "1. Make sure `nn::Module::to` works with `torch.nn.utils.weight_norm` wrapped convolution layer.\n"
        "2. Unskip this test."
    )
    def test_layerwise_casting_memory(self):
        pass


@slow
class AutoencoderOobleckIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def _load_datasamples(self, num_samples):
        ds = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True
        )
        # automatic decoding with librispeech
        speech_samples = ds.sort("id").select(range(num_samples))[:num_samples]["audio"]

        return torch.nn.utils.rnn.pad_sequence(
            [torch.from_numpy(x["array"]) for x in speech_samples], batch_first=True
        )

    def get_audio(self, audio_sample_size=2097152, fp16=False):
        dtype = torch.float16 if fp16 else torch.float32
        audio = self._load_datasamples(2).to(torch_device).to(dtype)

        # pad / crop to audio_sample_size
        audio = torch.nn.functional.pad(audio[:, :audio_sample_size], pad=(0, audio_sample_size - audio.shape[-1]))

        # todo channel
        audio = audio.unsqueeze(1).repeat(1, 2, 1).to(torch_device)

        return audio

    def get_oobleck_vae_model(self, model_id="stabilityai/stable-audio-open-1.0", fp16=False):
        torch_dtype = torch.float16 if fp16 else torch.float32

        model = AutoencoderOobleck.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch_dtype,
        )
        model.to(torch_device)

        return model

    def get_generator(self, seed=0):
        generator_device = "cpu" if not torch_device.startswith(torch_device) else torch_device
        if torch_device != "mps":
            return torch.Generator(device=generator_device).manual_seed(seed)
        return torch.manual_seed(seed)

    @parameterized.expand(
        [
            # fmt: off
            [33, [1.193e-4, 6.56e-05, 1.314e-4, 3.80e-05, -4.01e-06], 0.001192],
            [44, [2.77e-05, -2.65e-05, 1.18e-05, -6.94e-05, -9.57e-05], 0.001196],
            # fmt: on
        ]
    )
    def test_stable_diffusion(self, seed, expected_slice, expected_mean_absolute_diff):
        model = self.get_oobleck_vae_model()
        audio = self.get_audio()
        generator = self.get_generator(seed)

        with torch.no_grad():
            sample = model(audio, generator=generator, sample_posterior=True).sample

        assert sample.shape == audio.shape
        assert ((sample - audio).abs().mean() - expected_mean_absolute_diff).abs() <= 1e-6

        output_slice = sample[-1, 1, 5:10].cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=1e-5)

    def test_stable_diffusion_mode(self):
        model = self.get_oobleck_vae_model()
        audio = self.get_audio()

        with torch.no_grad():
            sample = model(audio, sample_posterior=False).sample

        assert sample.shape == audio.shape

    @parameterized.expand(
        [
            # fmt: off
            [33, [1.193e-4, 6.56e-05, 1.314e-4, 3.80e-05, -4.01e-06], 0.001192],
            [44, [2.77e-05, -2.65e-05, 1.18e-05, -6.94e-05, -9.57e-05], 0.001196],
            # fmt: on
        ]
    )
    def test_stable_diffusion_encode_decode(self, seed, expected_slice, expected_mean_absolute_diff):
        model = self.get_oobleck_vae_model()
        audio = self.get_audio()
        generator = self.get_generator(seed)

        with torch.no_grad():
            x = audio
            posterior = model.encode(x).latent_dist
            z = posterior.sample(generator=generator)
            sample = model.decode(z).sample

        # (batch_size, latent_dim, sequence_length)
        assert posterior.mean.shape == (audio.shape[0], model.config.decoder_input_channels, 1024)

        assert sample.shape == audio.shape
        assert ((sample - audio).abs().mean() - expected_mean_absolute_diff).abs() <= 1e-6

        output_slice = sample[-1, 1, 5:10].cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=1e-5)
