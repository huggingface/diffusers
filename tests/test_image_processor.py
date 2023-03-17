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
import PIL
import torch

from diffusers.image_processor import VaeImageProcessor


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
            return np.stack([np.array(i) for i in image], axis=0)
        elif isinstance(image, torch.Tensor):
            return image.cpu().numpy().transpose(0, 2, 3, 1)
        return image

    def test_vae_image_processor_pt(self):
        image_processor = VaeImageProcessor(do_resize=False, do_normalize=False)

        input_pt = self.dummy_sample
        input_np = self.to_np(input_pt)

        for output_type in ["pt", "np", "pil"]:
            out = image_processor.postprocess(
                image_processor.preprocess(input_pt),
                output_type=output_type,
            )
            out_np = self.to_np(out)
            in_np = (input_np * 255).round() if output_type == "pil" else input_np
            assert (
                np.abs(in_np - out_np).max() < 1e-6
            ), f"decoded output does not match input for output_type {output_type}"

    def test_vae_image_processor_np(self):
        image_processor = VaeImageProcessor(do_resize=False, do_normalize=False)
        input_np = self.dummy_sample.cpu().numpy().transpose(0, 2, 3, 1)

        for output_type in ["pt", "np", "pil"]:
            out = image_processor.postprocess(image_processor.preprocess(input_np), output_type=output_type)

            out_np = self.to_np(out)
            in_np = (input_np * 255).round() if output_type == "pil" else input_np
            assert (
                np.abs(in_np - out_np).max() < 1e-6
            ), f"decoded output does not match input for output_type {output_type}"

    def test_vae_image_processor_pil(self):
        image_processor = VaeImageProcessor(do_resize=False, do_normalize=False)

        input_np = self.dummy_sample.cpu().numpy().transpose(0, 2, 3, 1)
        input_pil = image_processor.numpy_to_pil(input_np)

        for output_type in ["pt", "np", "pil"]:
            out = image_processor.postprocess(image_processor.preprocess(input_pil), output_type=output_type)
            for i, o in zip(input_pil, out):
                in_np = np.array(i)
                out_np = self.to_np(out) if output_type == "pil" else (self.to_np(out) * 255).round()
                assert (
                    np.abs(in_np - out_np).max() < 1e-6
                ), f"decoded output does not match input for output_type {output_type}"

    def test_preprocess_input_3d(self):
        image_processor = VaeImageProcessor(do_resize=False, do_normalize=False)

        input_pt_4d = self.dummy_sample
        input_pt_3d = input_pt_4d.squeeze(0)

        out_pt_4d = image_processor.postprocess(
            image_processor.preprocess(input_pt_4d),
            output_type="np",
        )
        out_pt_3d = image_processor.postprocess(
            image_processor.preprocess(input_pt_3d),
            output_type="np",
        )

        input_np_4d = self.to_np(self.dummy_sample)
        input_np_3d = input_np_4d.squeeze(0)

        out_np_4d = image_processor.postprocess(
            image_processor.preprocess(input_np_4d),
            output_type="np",
        )
        out_np_3d = image_processor.postprocess(
            image_processor.preprocess(input_np_3d),
            output_type="np",
        )

        assert np.abs(out_pt_4d - out_pt_3d).max() < 1e-6
        assert np.abs(out_np_4d - out_np_3d).max() < 1e-6

    def test_preprocess_input_list(self):
        image_processor = VaeImageProcessor(do_resize=False, do_normalize=False)

        input_pt_4d = self.dummy_sample
        input_pt_list = list(input_pt_4d)

        out_pt_4d = image_processor.postprocess(
            image_processor.preprocess(input_pt_4d),
            output_type="np",
        )

        out_pt_list = image_processor.postprocess(
            image_processor.preprocess(input_pt_list),
            output_type="np",
        )

        input_np_4d = self.to_np(self.dummy_sample)
        list(input_np_4d)

        out_np_4d = image_processor.postprocess(
            image_processor.preprocess(input_pt_4d),
            output_type="np",
        )

        out_np_list = image_processor.postprocess(
            image_processor.preprocess(input_pt_list),
            output_type="np",
        )

        assert np.abs(out_pt_4d - out_pt_list).max() < 1e-6
        assert np.abs(out_np_4d - out_np_list).max() < 1e-6
