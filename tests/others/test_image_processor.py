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
import PIL.Image
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

    @property
    def dummy_mask(self):
        batch_size = 1
        num_channels = 1
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
        image_processor = VaeImageProcessor(do_resize=False, do_normalize=True)

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
        image_processor = VaeImageProcessor(do_resize=False, do_normalize=True)
        input_np = self.dummy_sample.cpu().numpy().transpose(0, 2, 3, 1)

        for output_type in ["pt", "np", "pil"]:
            out = image_processor.postprocess(image_processor.preprocess(input_np), output_type=output_type)

            out_np = self.to_np(out)
            in_np = (input_np * 255).round() if output_type == "pil" else input_np
            assert (
                np.abs(in_np - out_np).max() < 1e-6
            ), f"decoded output does not match input for output_type {output_type}"

    def test_vae_image_processor_pil(self):
        image_processor = VaeImageProcessor(do_resize=False, do_normalize=True)

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
        input_np_list = list(input_np_4d)

        out_np_4d = image_processor.postprocess(
            image_processor.preprocess(input_np_4d),
            output_type="np",
        )

        out_np_list = image_processor.postprocess(
            image_processor.preprocess(input_np_list),
            output_type="np",
        )

        assert np.abs(out_pt_4d - out_pt_list).max() < 1e-6
        assert np.abs(out_np_4d - out_np_list).max() < 1e-6

    def test_preprocess_input_mask_3d(self):
        image_processor = VaeImageProcessor(
            do_resize=False, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )

        input_pt_4d = self.dummy_mask
        input_pt_3d = input_pt_4d.squeeze(0)
        input_pt_2d = input_pt_3d.squeeze(0)

        out_pt_4d = image_processor.postprocess(
            image_processor.preprocess(input_pt_4d),
            output_type="np",
        )
        out_pt_3d = image_processor.postprocess(
            image_processor.preprocess(input_pt_3d),
            output_type="np",
        )

        out_pt_2d = image_processor.postprocess(
            image_processor.preprocess(input_pt_2d),
            output_type="np",
        )

        input_np_4d = self.to_np(self.dummy_mask)
        input_np_3d = input_np_4d.squeeze(0)
        input_np_3d_1 = input_np_4d.squeeze(-1)
        input_np_2d = input_np_3d.squeeze(-1)

        out_np_4d = image_processor.postprocess(
            image_processor.preprocess(input_np_4d),
            output_type="np",
        )
        out_np_3d = image_processor.postprocess(
            image_processor.preprocess(input_np_3d),
            output_type="np",
        )

        out_np_3d_1 = image_processor.postprocess(
            image_processor.preprocess(input_np_3d_1),
            output_type="np",
        )

        out_np_2d = image_processor.postprocess(
            image_processor.preprocess(input_np_2d),
            output_type="np",
        )

        assert np.abs(out_pt_4d - out_pt_3d).max() == 0
        assert np.abs(out_pt_4d - out_pt_2d).max() == 0
        assert np.abs(out_np_4d - out_np_3d).max() == 0
        assert np.abs(out_np_4d - out_np_3d_1).max() == 0
        assert np.abs(out_np_4d - out_np_2d).max() == 0

    def test_preprocess_input_mask_list(self):
        image_processor = VaeImageProcessor(do_resize=False, do_normalize=False, do_convert_grayscale=True)

        input_pt_4d = self.dummy_mask
        input_pt_3d = input_pt_4d.squeeze(0)
        input_pt_2d = input_pt_3d.squeeze(0)

        inputs_pt = [input_pt_4d, input_pt_3d, input_pt_2d]
        inputs_pt_list = [[input_pt] for input_pt in inputs_pt]

        for input_pt, input_pt_list in zip(inputs_pt, inputs_pt_list):
            out_pt = image_processor.postprocess(
                image_processor.preprocess(input_pt),
                output_type="np",
            )
            out_pt_list = image_processor.postprocess(
                image_processor.preprocess(input_pt_list),
                output_type="np",
            )
            assert np.abs(out_pt - out_pt_list).max() < 1e-6

        input_np_4d = self.to_np(self.dummy_mask)
        input_np_3d = input_np_4d.squeeze(0)
        input_np_2d = input_np_3d.squeeze(-1)

        inputs_np = [input_np_4d, input_np_3d, input_np_2d]
        inputs_np_list = [[input_np] for input_np in inputs_np]

        for input_np, input_np_list in zip(inputs_np, inputs_np_list):
            out_np = image_processor.postprocess(
                image_processor.preprocess(input_np),
                output_type="np",
            )
            out_np_list = image_processor.postprocess(
                image_processor.preprocess(input_np_list),
                output_type="np",
            )
            assert np.abs(out_np - out_np_list).max() < 1e-6

    def test_preprocess_input_mask_3d_batch(self):
        image_processor = VaeImageProcessor(do_resize=False, do_normalize=False, do_convert_grayscale=True)

        # create a dummy mask input with batch_size 2
        dummy_mask_batch = torch.cat([self.dummy_mask] * 2, axis=0)

        # squeeze out the channel dimension
        input_pt_3d = dummy_mask_batch.squeeze(1)
        input_np_3d = self.to_np(dummy_mask_batch).squeeze(-1)

        input_pt_3d_list = list(input_pt_3d)
        input_np_3d_list = list(input_np_3d)

        out_pt_3d = image_processor.postprocess(
            image_processor.preprocess(input_pt_3d),
            output_type="np",
        )
        out_pt_3d_list = image_processor.postprocess(
            image_processor.preprocess(input_pt_3d_list),
            output_type="np",
        )

        assert np.abs(out_pt_3d - out_pt_3d_list).max() < 1e-6

        out_np_3d = image_processor.postprocess(
            image_processor.preprocess(input_np_3d),
            output_type="np",
        )
        out_np_3d_list = image_processor.postprocess(
            image_processor.preprocess(input_np_3d_list),
            output_type="np",
        )

        assert np.abs(out_np_3d - out_np_3d_list).max() < 1e-6

    def test_vae_image_processor_resize_pt(self):
        image_processor = VaeImageProcessor(do_resize=True, vae_scale_factor=1)
        input_pt = self.dummy_sample
        b, c, h, w = input_pt.shape
        scale = 2
        out_pt = image_processor.resize(image=input_pt, height=h // scale, width=w // scale)
        exp_pt_shape = (b, c, h // scale, w // scale)
        assert (
            out_pt.shape == exp_pt_shape
        ), f"resized image output shape '{out_pt.shape}' didn't match expected shape '{exp_pt_shape}'."

    def test_vae_image_processor_resize_np(self):
        image_processor = VaeImageProcessor(do_resize=True, vae_scale_factor=1)
        input_pt = self.dummy_sample
        b, c, h, w = input_pt.shape
        scale = 2
        input_np = self.to_np(input_pt)
        out_np = image_processor.resize(image=input_np, height=h // scale, width=w // scale)
        exp_np_shape = (b, h // scale, w // scale, c)
        assert (
            out_np.shape == exp_np_shape
        ), f"resized image output shape '{out_np.shape}' didn't match expected shape '{exp_np_shape}'."
