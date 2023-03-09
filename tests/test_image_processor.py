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

import PIL

from diffusers import VaeImageProcessor


class ImageProcessorTest(unittest.TestCase):
    
    @property
    def dummy_sample(self):
        batch_size = 1
        num_channels = 3
        height = 8
        width = 8

        sample = torch.rand((batch_size, num_channels, height, width))

        return sample

    def to_np(self, image):
        if isinstance(image[0], PIL.Image.Image):
            return np.stack([np.array(i) for i in image],axis=0)
        elif isinstance(image, torch.Tensor):
            return image.cpu().numpy().transpose(0, 2, 3, 1)
        return image

    def test_encode_input_pt(self):
        image_processor = VaeImageProcessor(do_resize=False, do_normalize=False)
    
        input_pt = self.dummy_sample
        input_np = self.to_np(input_pt)
        
        for output_type in ['pt','np','pil']:
            out = image_processor.decode(
                image_processor.encode(input_pt),
                output_type=output_type,
                )
            out_np = self.to_np(out)
            in_np = (input_np * 255).round() if output_type == 'pil' else input_np
            assert np.abs(in_np - out_np).max() < 1e-6, f"decoded output does not match input for output_type {output_type}"

    def test_encode_input_np(self):
        image_processor = VaeImageProcessor(do_resize=False, do_normalize=False)
        input_np = self.dummy_sample.cpu().numpy().transpose(0, 2, 3, 1)
        
        for output_type in ['pt','np','pil']:
            out = image_processor.decode(
                image_processor.encode(input_np), 
                output_type=output_type)
            
            out_np = self.to_np(out)
            in_np = (input_np * 255).round() if output_type == 'pil' else input_np
            assert np.abs(in_np - out_np).max() < 1e-6, f"decoded output does not match input for output_type {output_type}"

    def test_encode_input_pil(self):
        image_processor = VaeImageProcessor(do_resize=False, do_normalize=False)

        input_np = self.dummy_sample.cpu().numpy().transpose(0, 2, 3, 1)
        input_pil = image_processor.numpy_to_pil(input_np)
        
        for output_type in ['pt','np','pil']:
            out = image_processor.decode(
                image_processor.encode(input_pil), 
                output_type=output_type)
            for i, o in zip(input_pil, out):
                in_np = np.array(i)
                out_np = self.to_np(out) if output_type == 'pil' else (self.to_np(out) * 255).round()
                assert np.abs(in_np - out_np).max() < 1e-6, f"decoded output does not match input for output_type {output_type}"