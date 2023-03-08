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
import torch
import numpy as np

from diffusers import VaeImageProcessor

class ImageProcessorTest(unittest.TestCase):
    
    @property
    def dummy_sample(self):
        batch_size = 4
        num_channels = 3
        height = 8
        width = 8

        sample = torch.rand((batch_size, num_channels, height, width))

        return sample
    
    def test_encode_input_pt(self):
        image_processor = VaeImageProcessor(do_resize=False, do_normalize=False)
        
        input_pt = self.dummy_sample
        out_pt = image_processor.decode(
            image_processor.encode(input_pt),
            output_type='pt')
        assert np.abs(input_pt.cpu().numpy() - out_pt.cpu().numpy()).max() < 1e-6

    def test_encode_input_np(self):
        image_processor = VaeImageProcessor(do_resize=False, do_normalize=False)

        input_np = self.dummy_sample.cpu().numpy().transpose(0, 2, 3, 1)
        out_np = image_processor.decode(
            image_processor.encode(input_np), 
            output_type='np')
        assert np.abs(input_np - out_np).max() < 1e-6

    def test_encode_input_pil(self):
        image_processor = VaeImageProcessor(do_resize=False, do_normalize=False)

        input_np = self.dummy_sample.cpu().numpy().transpose(0, 2, 3, 1)
        input_pil = image_processor.numpy_to_pil(input_np)
        
        out_pil = image_processor.decode(
            image_processor.encode(input_pil), 
            output_type='pil')
        for i, o in zip(input_pil, out_pil):
            assert np.abs(np.array(i) - np.array(o)).max() == 0