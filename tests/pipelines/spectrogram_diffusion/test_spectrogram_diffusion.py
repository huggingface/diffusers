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

from diffusers import MidiProcessor, SpectrogramDiffusionPipeline
from diffusers.utils import require_torch_gpu, slow, torch_device
from diffusers.utils.testing_utils import require_note_seq, require_onnxruntime


torch.backends.cuda.matmul.allow_tf32 = False


MIDI_FILE = "./tests/fixtures/elise_format0.mid"


# Add more fast tests without MidiProcessor and onnx melgan, so that just PyTorch is needed


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
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        def callback(step, mel_output):
            # TODO - decode tokens to audio
            # ... melgan(...)
            # simulate that audio is played
            pass

        processor = MidiProcessor()

        input_tokens = processor(MIDI_FILE)
        input_tokens = input_tokens[:3]
        generator = torch.manual_seed(0)

        pipe(input_tokens, num_inference_steps=5, generator=generator, callback=callback)

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
