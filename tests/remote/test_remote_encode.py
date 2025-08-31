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

import unittest

import PIL.Image
import torch

from diffusers.utils import load_image
from diffusers.utils.constants import (
    DECODE_ENDPOINT_FLUX,
    DECODE_ENDPOINT_SD_V1,
    DECODE_ENDPOINT_SD_XL,
    ENCODE_ENDPOINT_FLUX,
    ENCODE_ENDPOINT_SD_V1,
    ENCODE_ENDPOINT_SD_XL,
)
from diffusers.utils.remote_utils import (
    remote_decode,
    remote_encode,
)

from ..testing_utils import (
    enable_full_determinism,
    slow,
)


enable_full_determinism()

IMAGE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg?download=true"


class RemoteAutoencoderKLEncodeMixin:
    channels: int = None
    endpoint: str = None
    decode_endpoint: str = None
    dtype: torch.dtype = None
    scaling_factor: float = None
    shift_factor: float = None
    image: PIL.Image.Image = None

    def get_dummy_inputs(self):
        if self.image is None:
            self.image = load_image(IMAGE)
        inputs = {
            "endpoint": self.endpoint,
            "image": self.image,
            "scaling_factor": self.scaling_factor,
            "shift_factor": self.shift_factor,
        }
        return inputs

    def test_image_input(self):
        inputs = self.get_dummy_inputs()
        height, width = inputs["image"].height, inputs["image"].width
        output = remote_encode(**inputs)
        self.assertEqual(list(output.shape), [1, self.channels, height // 8, width // 8])
        decoded = remote_decode(
            tensor=output,
            endpoint=self.decode_endpoint,
            scaling_factor=self.scaling_factor,
            shift_factor=self.shift_factor,
            image_format="png",
        )
        self.assertEqual(decoded.height, height)
        self.assertEqual(decoded.width, width)
        # image_slice = torch.from_numpy(np.array(inputs["image"])[0, -3:, -3:].flatten())
        # decoded_slice = torch.from_numpy(np.array(decoded)[0, -3:, -3:].flatten())
        # TODO: how to test this? encode->decode is lossy. expected slice of encoded latent?


class RemoteAutoencoderKLSDv1Tests(
    RemoteAutoencoderKLEncodeMixin,
    unittest.TestCase,
):
    channels = 4
    endpoint = ENCODE_ENDPOINT_SD_V1
    decode_endpoint = DECODE_ENDPOINT_SD_V1
    dtype = torch.float16
    scaling_factor = 0.18215
    shift_factor = None


class RemoteAutoencoderKLSDXLTests(
    RemoteAutoencoderKLEncodeMixin,
    unittest.TestCase,
):
    channels = 4
    endpoint = ENCODE_ENDPOINT_SD_XL
    decode_endpoint = DECODE_ENDPOINT_SD_XL
    dtype = torch.float16
    scaling_factor = 0.13025
    shift_factor = None


class RemoteAutoencoderKLFluxTests(
    RemoteAutoencoderKLEncodeMixin,
    unittest.TestCase,
):
    channels = 16
    endpoint = ENCODE_ENDPOINT_FLUX
    decode_endpoint = DECODE_ENDPOINT_FLUX
    dtype = torch.bfloat16
    scaling_factor = 0.3611
    shift_factor = 0.1159


class RemoteAutoencoderKLEncodeSlowTestMixin:
    channels: int = 4
    endpoint: str = None
    decode_endpoint: str = None
    dtype: torch.dtype = None
    scaling_factor: float = None
    shift_factor: float = None
    image: PIL.Image.Image = None

    def get_dummy_inputs(self):
        if self.image is None:
            self.image = load_image(IMAGE)
        inputs = {
            "endpoint": self.endpoint,
            "image": self.image,
            "scaling_factor": self.scaling_factor,
            "shift_factor": self.shift_factor,
        }
        return inputs

    def test_multi_res(self):
        inputs = self.get_dummy_inputs()
        for height in {
            320,
            512,
            640,
            704,
            896,
            1024,
            1208,
            1384,
            1536,
            1608,
            1864,
            2048,
        }:
            for width in {
                320,
                512,
                640,
                704,
                896,
                1024,
                1208,
                1384,
                1536,
                1608,
                1864,
                2048,
            }:
                inputs["image"] = inputs["image"].resize(
                    (
                        width,
                        height,
                    )
                )
                output = remote_encode(**inputs)
                self.assertEqual(list(output.shape), [1, self.channels, height // 8, width // 8])
                decoded = remote_decode(
                    tensor=output,
                    endpoint=self.decode_endpoint,
                    scaling_factor=self.scaling_factor,
                    shift_factor=self.shift_factor,
                    image_format="png",
                )
                self.assertEqual(decoded.height, height)
                self.assertEqual(decoded.width, width)
                decoded.save(f"test_multi_res_{height}_{width}.png")


@slow
class RemoteAutoencoderKLSDv1SlowTests(
    RemoteAutoencoderKLEncodeSlowTestMixin,
    unittest.TestCase,
):
    endpoint = ENCODE_ENDPOINT_SD_V1
    decode_endpoint = DECODE_ENDPOINT_SD_V1
    dtype = torch.float16
    scaling_factor = 0.18215
    shift_factor = None


@slow
class RemoteAutoencoderKLSDXLSlowTests(
    RemoteAutoencoderKLEncodeSlowTestMixin,
    unittest.TestCase,
):
    endpoint = ENCODE_ENDPOINT_SD_XL
    decode_endpoint = DECODE_ENDPOINT_SD_XL
    dtype = torch.float16
    scaling_factor = 0.13025
    shift_factor = None


@slow
class RemoteAutoencoderKLFluxSlowTests(
    RemoteAutoencoderKLEncodeSlowTestMixin,
    unittest.TestCase,
):
    channels = 16
    endpoint = ENCODE_ENDPOINT_FLUX
    decode_endpoint = DECODE_ENDPOINT_FLUX
    dtype = torch.bfloat16
    scaling_factor = 0.3611
    shift_factor = 0.1159
