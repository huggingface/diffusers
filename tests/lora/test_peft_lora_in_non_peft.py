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

import unittest

import numpy as np
import torch

from diffusers import DiffusionPipeline
from diffusers.utils.testing_utils import torch_device


class PEFTLoRALoading(unittest.TestCase):
    def get_dummy_inputs(self):
        pipeline_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
            "generator": torch.manual_seed(0),
        }
        return pipeline_inputs

    def test_stable_diffusion_peft_lora_loading_in_non_peft(self):
        sd_pipe = DiffusionPipeline.from_pretrained("hf-internal-testing/tiny-sd-pipe").to(torch_device)
        # This LoRA was obtained using similarly as how it's done in the training scripts.
        # For details on how the LoRA was obtained, refer to:
        # https://colab.research.google.com/gist/sayakpaul/4a00d0223c03225f82735ff93930f43d/scratchpad.ipynb
        sd_pipe.load_lora_weights("hf-internal-testing/tiny-sd-lora-peft")

        inputs = self.get_dummy_inputs()
        outputs = sd_pipe(**inputs).images

        predicted_slice = outputs[0, -3:, -3:, -1].flatten()
        expected_slice = np.array([0.5708, 0.6001, 0.4895, 0.5191, 0.5518, 0.4583, 0.5059, 0.4866, 0.4759])

        self.assertTrue(outputs.shape == (1, 64, 64, 3))
        assert np.allclose(expected_slice, predicted_slice, atol=1e-3, rtol=1e-3)

    def test_stable_diffusion_xl_peft_lora_loading_in_non_peft(self):
        sd_pipe = DiffusionPipeline.from_pretrained("hf-internal-testing/tiny-sdxl-pipe").to(torch_device)
        # This LoRA was obtained using similarly as how it's done in the training scripts.
        sd_pipe.load_lora_weights("hf-internal-testing/tiny-sdxl-lora-peft")

        inputs = self.get_dummy_inputs()
        outputs = sd_pipe(**inputs).images

        predicted_slice = outputs[0, -3:, -3:, -1].flatten()
        expected_slice = np.array([0.6079, 0.551, 0.521, 0.4106, 0.3948, 0.4648, 0.5277, 0.501, 0.49])

        self.assertTrue(outputs.shape == (1, 64, 64, 3))
        assert np.allclose(expected_slice, predicted_slice, atol=1e-3, rtol=1e-3)
