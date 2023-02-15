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
import torch
import os

from diffusers import SpectrogramDiffusionPipeline
from diffusers.utils import slow, require_torch_gpu, torch_device
import scipy


torch.backends.cuda.matmul.allow_tf32 = False


@slow
@require_torch_gpu
class PipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_spectrogram(self):
        device = torch_device

        url = "http://www.piano-midi.de/midis/beethoven/elise_format0.mid"

        os.system(f"wget {url}")

        pipe = SpectrogramDiffusionPipeline.from_pretrained("kashif/music-spectrogram-diffusion")
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        output = pipe(url.split("/")[-1], num_inference_steps=100)
        audio = output.audios[0]
        rate = 16_000
        scipy.io.wavfile.write("/home/patrick_huggingface_co/audios/beet.wav", rate, audio[0])

        print("Finished")
