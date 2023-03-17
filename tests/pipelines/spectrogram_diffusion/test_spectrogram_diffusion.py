# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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
import scipy
import torch

from diffusers import DDPMScheduler, MidiProcessor, SpectrogramDiffusionPipeline
from diffusers.pipelines.spectrogram_diffusion import SpectrogramContEncoder, SpectrogramNotesEncoder, T5FilmDecoder
from diffusers.utils import require_torch_gpu, skip_mps, slow, torch_device
from diffusers.utils.testing_utils import is_onnx_available, require_note_seq, require_onnxruntime

from ...pipeline_params import TOKENS_TO_AUDIO_GENERATION_PARAMS, TOKENS_TO_AUDIO_GENERATION_BATCH_PARAMS
from ...test_pipelines_common import PipelineTesterMixin


if is_onnx_available():
    from diffusers import OnnxRuntimeModel

torch.backends.cuda.matmul.allow_tf32 = False


MIDI_FILE = "./tests/fixtures/elise_format0.mid"


class SpectrogramDiffusionPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = SpectrogramDiffusionPipeline
    required_optional_params = PipelineTesterMixin.required_optional_params - {
        "callback",
        "latents",
        "callback_steps",
        "output_type",
        "num_images_per_prompt",
    }
    test_attention_slicing = False
    test_cpu_offload = False
    batch_params = TOKENS_TO_AUDIO_GENERATION_PARAMS
    params = TOKENS_TO_AUDIO_GENERATION_BATCH_PARAMS

    def get_dummy_components(self):
        torch.manual_seed(0)
        notes_encoder = SpectrogramNotesEncoder(
            max_length=2048,
            vocab_size=1536,
            d_model=768,
            dropout_rate=0.1,
            num_layers=1,
            num_heads=1,
            d_kv=4,
            d_ff=2048,
            feed_forward_proj="gated-gelu",
        )

        continuous_encoder = SpectrogramContEncoder(
            input_dims=128,
            targets_context_length=256,
            d_model=768,
            dropout_rate=0.1,
            num_layers=1,
            num_heads=1,
            d_kv=4,
            d_ff=2048,
            feed_forward_proj="gated-gelu",
        )

        decoder = T5FilmDecoder(
            input_dims=128,
            targets_length=256,
            max_decoder_noise_time=20000.0,
            d_model=768,
            num_layers=1,
            num_heads=1,
            d_kv=4,
            d_ff=2048,
            dropout_rate=0.1,
        )

        scheduler = DDPMScheduler()

        components = {
            "notes_encoder": notes_encoder,
            "continuous_encoder": continuous_encoder,
            "decoder": decoder,
            "scheduler": scheduler,
            "melgan": None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "input_tokens": [
                [1134, 90, 1135, 1133, 1080, 112, 1132, 1080, 1133, 1079, 133, 1132, 1079, 1133, 1] + [0] * 2033
            ],
            "generator": generator,
            "num_inference_steps": 4,
            "output_type": "mel",
        }
        return inputs

    def test_spectrogram_diffusion(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        pipe = SpectrogramDiffusionPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = pipe(**inputs)
        mel = output.audios

        mel_slice = mel[0, -3:, -3:]

        assert mel_slice.shape == (3, 3)
        expected_slice = np.array(
            [-4.783236, 4.0, -2.2628813, -4.4896817, -10.321411, -11.162924, -11.512925, 4.0, 4.0]
        )
        assert np.abs(mel_slice.flatten() - expected_slice).max() < 1e-2

    @skip_mps
    def test_save_load_local(self):
        return super().test_save_load_local()

    @skip_mps
    def test_dict_tuple_outputs_equivalent(self):
        return super().test_dict_tuple_outputs_equivalent()

    @skip_mps
    def test_save_load_optional_components(self):
        return super().test_save_load_optional_components()

    @skip_mps
    def test_attention_slicing_forward_pass(self):
        return super().test_attention_slicing_forward_pass()


@slow
@require_torch_gpu
@require_onnxruntime
@require_note_seq
class PipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_callback(self):
        # TODO - test that pipeline can decode tokens in a callback
        # so that music can be played live
        device = torch_device

        pipe = SpectrogramDiffusionPipeline.from_pretrained("kashif/music-spectrogram-diffusion", melgan=None)
        melgan = OnnxRuntimeModel.from_pretrained("kashif/soundstream_mel_decoder")

        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        def callback(step, mel_output):
            # decode mel to audio
            audio = melgan(input_features=mel_output.astype(np.float32))[0]
            assert len(audio[0]) == 81920 * (step + 1)
            # simulate that audio is played
            return audio

        processor = MidiProcessor()
        input_tokens = processor(MIDI_FILE)

        input_tokens = input_tokens[:3]
        generator = torch.manual_seed(0)
        pipe(input_tokens, num_inference_steps=5, generator=generator, callback=callback, output_type="mel")

    def test_spectrogram_fast(self):
        device = torch_device

        pipe = SpectrogramDiffusionPipeline.from_pretrained("kashif/music-spectrogram-diffusion")
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)
        processor = MidiProcessor()

        input_tokens = processor(MIDI_FILE)
        # just run two denoising loops
        input_tokens = input_tokens[:2]

        generator = torch.manual_seed(0)
        output = pipe(input_tokens, num_inference_steps=2, generator=generator)

        audio = output.audios[0]

        assert abs(np.abs(audio).sum() - 3612.841) < 1e-1

    def test_spectrogram(self):
        device = torch_device

        pipe = SpectrogramDiffusionPipeline.from_pretrained("kashif/music-spectrogram-diffusion")
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        processor = MidiProcessor()

        input_tokens = processor(MIDI_FILE)

        # just run 4 denoising loops
        input_tokens = input_tokens[:4]

        generator = torch.manual_seed(0)
        output = pipe(input_tokens, num_inference_steps=100, generator=generator)

        audio = output.audios[0]
        assert abs(np.abs(audio).sum() - 9389.1111) < 5e-2

        audio = output.audios[0]
        rate = 16_000
        scipy.io.wavfile.write("/home/patrick_huggingface_co/audios/beet.wav", rate, audio[0])
