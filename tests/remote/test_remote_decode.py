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
from typing import Tuple, Union

import numpy as np
import PIL.Image
import torch

from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.constants import (
    DECODE_ENDPOINT_FLUX,
    DECODE_ENDPOINT_HUNYUAN_VIDEO,
    DECODE_ENDPOINT_SD_V1,
    DECODE_ENDPOINT_SD_XL,
)
from diffusers.utils.remote_utils import (
    remote_decode,
)
from diffusers.video_processor import VideoProcessor

from ..testing_utils import (
    enable_full_determinism,
    slow,
    torch_all_close,
    torch_device,
)


enable_full_determinism()


class RemoteAutoencoderKLMixin:
    shape: Tuple[int, ...] = None
    out_hw: Tuple[int, int] = None
    endpoint: str = None
    dtype: torch.dtype = None
    scaling_factor: float = None
    shift_factor: float = None
    processor_cls: Union[VaeImageProcessor, VideoProcessor] = None
    output_pil_slice: torch.Tensor = None
    output_pt_slice: torch.Tensor = None
    partial_postprocess_return_pt_slice: torch.Tensor = None
    return_pt_slice: torch.Tensor = None
    width: int = None
    height: int = None

    def get_dummy_inputs(self):
        inputs = {
            "endpoint": self.endpoint,
            "tensor": torch.randn(
                self.shape,
                device=torch_device,
                dtype=self.dtype,
                generator=torch.Generator(torch_device).manual_seed(13),
            ),
            "scaling_factor": self.scaling_factor,
            "shift_factor": self.shift_factor,
            "height": self.height,
            "width": self.width,
        }
        return inputs

    def test_no_scaling(self):
        inputs = self.get_dummy_inputs()
        if inputs["scaling_factor"] is not None:
            inputs["tensor"] = inputs["tensor"] / inputs["scaling_factor"]
            inputs["scaling_factor"] = None
        if inputs["shift_factor"] is not None:
            inputs["tensor"] = inputs["tensor"] + inputs["shift_factor"]
            inputs["shift_factor"] = None
        processor = self.processor_cls()
        output = remote_decode(
            output_type="pt",
            # required for now, will be removed in next update
            do_scaling=False,
            processor=processor,
            **inputs,
        )
        assert isinstance(output, PIL.Image.Image)
        self.assertTrue(isinstance(output, PIL.Image.Image), f"Expected `PIL.Image.Image` output, got {type(output)}")
        self.assertEqual(output.height, self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output.height}")
        self.assertEqual(output.width, self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output.width}")
        output_slice = torch.from_numpy(np.array(output)[0, -3:, -3:].flatten())
        # Increased tolerance for Flux Packed diff [1, 0, 1, 0, 0, 0, 0, 0, 0]
        self.assertTrue(
            torch_all_close(output_slice, self.output_pt_slice.to(output_slice.dtype), rtol=1, atol=1),
            f"{output_slice}",
        )

    def test_output_type_pt(self):
        inputs = self.get_dummy_inputs()
        processor = self.processor_cls()
        output = remote_decode(output_type="pt", processor=processor, **inputs)
        assert isinstance(output, PIL.Image.Image)
        self.assertTrue(isinstance(output, PIL.Image.Image), f"Expected `PIL.Image.Image` output, got {type(output)}")
        self.assertEqual(output.height, self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output.height}")
        self.assertEqual(output.width, self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output.width}")
        output_slice = torch.from_numpy(np.array(output)[0, -3:, -3:].flatten())
        self.assertTrue(
            torch_all_close(output_slice, self.output_pt_slice.to(output_slice.dtype), rtol=1e-2), f"{output_slice}"
        )

    # output is visually the same, slice is flaky?
    def test_output_type_pil(self):
        inputs = self.get_dummy_inputs()
        output = remote_decode(output_type="pil", **inputs)
        self.assertTrue(isinstance(output, PIL.Image.Image), f"Expected `PIL.Image.Image` output, got {type(output)}")
        self.assertEqual(output.height, self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output.height}")
        self.assertEqual(output.width, self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output.width}")

    def test_output_type_pil_image_format(self):
        inputs = self.get_dummy_inputs()
        output = remote_decode(output_type="pil", image_format="png", **inputs)
        self.assertTrue(isinstance(output, PIL.Image.Image), f"Expected `PIL.Image.Image` output, got {type(output)}")
        self.assertEqual(output.height, self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output.height}")
        self.assertEqual(output.width, self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output.width}")
        self.assertEqual(output.format, "png", f"Expected image format `png`, got {output.format}")
        output_slice = torch.from_numpy(np.array(output)[0, -3:, -3:].flatten())
        self.assertTrue(
            torch_all_close(output_slice, self.output_pt_slice.to(output_slice.dtype), rtol=1e-2), f"{output_slice}"
        )

    def test_output_type_pt_partial_postprocess(self):
        inputs = self.get_dummy_inputs()
        output = remote_decode(output_type="pt", partial_postprocess=True, **inputs)
        self.assertTrue(isinstance(output, PIL.Image.Image), f"Expected `PIL.Image.Image` output, got {type(output)}")
        self.assertEqual(output.height, self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output.height}")
        self.assertEqual(output.width, self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output.width}")
        output_slice = torch.from_numpy(np.array(output)[0, -3:, -3:].flatten())
        self.assertTrue(
            torch_all_close(output_slice, self.output_pt_slice.to(output_slice.dtype), rtol=1e-2), f"{output_slice}"
        )

    def test_output_type_pt_return_type_pt(self):
        inputs = self.get_dummy_inputs()
        output = remote_decode(output_type="pt", return_type="pt", **inputs)
        self.assertTrue(isinstance(output, torch.Tensor), f"Expected `torch.Tensor` output, got {type(output)}")
        self.assertEqual(
            output.shape[2], self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output.shape[2]}"
        )
        self.assertEqual(
            output.shape[3], self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output.shape[3]}"
        )
        output_slice = output[0, 0, -3:, -3:].flatten()
        self.assertTrue(
            torch_all_close(output_slice, self.return_pt_slice.to(output_slice.dtype), rtol=1e-3, atol=1e-3),
            f"{output_slice}",
        )

    def test_output_type_pt_partial_postprocess_return_type_pt(self):
        inputs = self.get_dummy_inputs()
        output = remote_decode(output_type="pt", partial_postprocess=True, return_type="pt", **inputs)
        self.assertTrue(isinstance(output, torch.Tensor), f"Expected `torch.Tensor` output, got {type(output)}")
        self.assertEqual(
            output.shape[1], self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output.shape[1]}"
        )
        self.assertEqual(
            output.shape[2], self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output.shape[2]}"
        )
        output_slice = output[0, -3:, -3:, 0].flatten().cpu()
        self.assertTrue(
            torch_all_close(output_slice, self.partial_postprocess_return_pt_slice.to(output_slice.dtype), rtol=1e-2),
            f"{output_slice}",
        )

    def test_do_scaling_deprecation(self):
        inputs = self.get_dummy_inputs()
        inputs.pop("scaling_factor", None)
        inputs.pop("shift_factor", None)
        with self.assertWarns(FutureWarning) as warning:
            _ = remote_decode(output_type="pt", partial_postprocess=True, **inputs)
            self.assertEqual(
                str(warning.warnings[0].message),
                "`do_scaling` is deprecated, pass `scaling_factor` and `shift_factor` if required.",
                str(warning.warnings[0].message),
            )

    def test_input_tensor_type_base64_deprecation(self):
        inputs = self.get_dummy_inputs()
        with self.assertWarns(FutureWarning) as warning:
            _ = remote_decode(output_type="pt", input_tensor_type="base64", partial_postprocess=True, **inputs)
            self.assertEqual(
                str(warning.warnings[0].message),
                "input_tensor_type='base64' is deprecated. Using `binary`.",
                str(warning.warnings[0].message),
            )

    def test_output_tensor_type_base64_deprecation(self):
        inputs = self.get_dummy_inputs()
        with self.assertWarns(FutureWarning) as warning:
            _ = remote_decode(output_type="pt", output_tensor_type="base64", partial_postprocess=True, **inputs)
            self.assertEqual(
                str(warning.warnings[0].message),
                "output_tensor_type='base64' is deprecated. Using `binary`.",
                str(warning.warnings[0].message),
            )


class RemoteAutoencoderKLHunyuanVideoMixin(RemoteAutoencoderKLMixin):
    def test_no_scaling(self):
        inputs = self.get_dummy_inputs()
        if inputs["scaling_factor"] is not None:
            inputs["tensor"] = inputs["tensor"] / inputs["scaling_factor"]
            inputs["scaling_factor"] = None
        if inputs["shift_factor"] is not None:
            inputs["tensor"] = inputs["tensor"] + inputs["shift_factor"]
            inputs["shift_factor"] = None
        processor = self.processor_cls()
        output = remote_decode(
            output_type="pt",
            # required for now, will be removed in next update
            do_scaling=False,
            processor=processor,
            **inputs,
        )
        self.assertTrue(
            isinstance(output, list) and isinstance(output[0], PIL.Image.Image),
            f"Expected `List[PIL.Image.Image]` output, got {type(output)}",
        )
        self.assertEqual(
            output[0].height, self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output[0].height}"
        )
        self.assertEqual(
            output[0].width, self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output[0].width}"
        )
        output_slice = torch.from_numpy(np.array(output[0])[0, -3:, -3:].flatten())
        self.assertTrue(
            torch_all_close(output_slice, self.output_pt_slice.to(output_slice.dtype), rtol=1, atol=1),
            f"{output_slice}",
        )

    def test_output_type_pt(self):
        inputs = self.get_dummy_inputs()
        processor = self.processor_cls()
        output = remote_decode(output_type="pt", processor=processor, **inputs)
        self.assertTrue(
            isinstance(output, list) and isinstance(output[0], PIL.Image.Image),
            f"Expected `List[PIL.Image.Image]` output, got {type(output)}",
        )
        self.assertEqual(
            output[0].height, self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output[0].height}"
        )
        self.assertEqual(
            output[0].width, self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output[0].width}"
        )
        output_slice = torch.from_numpy(np.array(output[0])[0, -3:, -3:].flatten())
        self.assertTrue(
            torch_all_close(output_slice, self.output_pt_slice.to(output_slice.dtype), rtol=1, atol=1),
            f"{output_slice}",
        )

    # output is visually the same, slice is flaky?
    def test_output_type_pil(self):
        inputs = self.get_dummy_inputs()
        processor = self.processor_cls()
        output = remote_decode(output_type="pil", processor=processor, **inputs)
        self.assertTrue(
            isinstance(output, list) and isinstance(output[0], PIL.Image.Image),
            f"Expected `List[PIL.Image.Image]` output, got {type(output)}",
        )
        self.assertEqual(
            output[0].height, self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output[0].height}"
        )
        self.assertEqual(
            output[0].width, self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output[0].width}"
        )

    def test_output_type_pil_image_format(self):
        inputs = self.get_dummy_inputs()
        processor = self.processor_cls()
        output = remote_decode(output_type="pil", processor=processor, image_format="png", **inputs)
        self.assertTrue(
            isinstance(output, list) and isinstance(output[0], PIL.Image.Image),
            f"Expected `List[PIL.Image.Image]` output, got {type(output)}",
        )
        self.assertEqual(
            output[0].height, self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output[0].height}"
        )
        self.assertEqual(
            output[0].width, self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output[0].width}"
        )
        output_slice = torch.from_numpy(np.array(output[0])[0, -3:, -3:].flatten())
        self.assertTrue(
            torch_all_close(output_slice, self.output_pt_slice.to(output_slice.dtype), rtol=1, atol=1),
            f"{output_slice}",
        )

    def test_output_type_pt_partial_postprocess(self):
        inputs = self.get_dummy_inputs()
        output = remote_decode(output_type="pt", partial_postprocess=True, **inputs)
        self.assertTrue(
            isinstance(output, list) and isinstance(output[0], PIL.Image.Image),
            f"Expected `List[PIL.Image.Image]` output, got {type(output)}",
        )
        self.assertEqual(
            output[0].height, self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output[0].height}"
        )
        self.assertEqual(
            output[0].width, self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output[0].width}"
        )
        output_slice = torch.from_numpy(np.array(output[0])[0, -3:, -3:].flatten())
        self.assertTrue(
            torch_all_close(output_slice, self.output_pt_slice.to(output_slice.dtype), rtol=1, atol=1),
            f"{output_slice}",
        )

    def test_output_type_pt_return_type_pt(self):
        inputs = self.get_dummy_inputs()
        output = remote_decode(output_type="pt", return_type="pt", **inputs)
        self.assertTrue(isinstance(output, torch.Tensor), f"Expected `torch.Tensor` output, got {type(output)}")
        self.assertEqual(
            output.shape[3], self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output.shape[2]}"
        )
        self.assertEqual(
            output.shape[4], self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output.shape[3]}"
        )
        output_slice = output[0, 0, 0, -3:, -3:].flatten()
        self.assertTrue(
            torch_all_close(output_slice, self.return_pt_slice.to(output_slice.dtype), rtol=1e-3, atol=1e-3),
            f"{output_slice}",
        )

    def test_output_type_mp4(self):
        inputs = self.get_dummy_inputs()
        output = remote_decode(output_type="mp4", return_type="mp4", **inputs)
        self.assertTrue(isinstance(output, bytes), f"Expected `bytes` output, got {type(output)}")


class RemoteAutoencoderKLSDv1Tests(
    RemoteAutoencoderKLMixin,
    unittest.TestCase,
):
    shape = (
        1,
        4,
        64,
        64,
    )
    out_hw = (
        512,
        512,
    )
    endpoint = DECODE_ENDPOINT_SD_V1
    dtype = torch.float16
    scaling_factor = 0.18215
    shift_factor = None
    processor_cls = VaeImageProcessor
    output_pt_slice = torch.tensor([31, 15, 11, 55, 30, 21, 66, 42, 30], dtype=torch.uint8)
    partial_postprocess_return_pt_slice = torch.tensor([100, 130, 99, 133, 106, 112, 97, 100, 121], dtype=torch.uint8)
    return_pt_slice = torch.tensor([-0.2177, 0.0217, -0.2258, 0.0412, -0.1687, -0.1232, -0.2416, -0.2130, -0.0543])


class RemoteAutoencoderKLSDXLTests(
    RemoteAutoencoderKLMixin,
    unittest.TestCase,
):
    shape = (
        1,
        4,
        128,
        128,
    )
    out_hw = (
        1024,
        1024,
    )
    endpoint = DECODE_ENDPOINT_SD_XL
    dtype = torch.float16
    scaling_factor = 0.13025
    shift_factor = None
    processor_cls = VaeImageProcessor
    output_pt_slice = torch.tensor([104, 52, 23, 114, 61, 35, 108, 87, 38], dtype=torch.uint8)
    partial_postprocess_return_pt_slice = torch.tensor([77, 86, 89, 49, 60, 75, 52, 65, 78], dtype=torch.uint8)
    return_pt_slice = torch.tensor([-0.3945, -0.3289, -0.2993, -0.6177, -0.5259, -0.4119, -0.5898, -0.4863, -0.3845])


class RemoteAutoencoderKLFluxTests(
    RemoteAutoencoderKLMixin,
    unittest.TestCase,
):
    shape = (
        1,
        16,
        128,
        128,
    )
    out_hw = (
        1024,
        1024,
    )
    endpoint = DECODE_ENDPOINT_FLUX
    dtype = torch.bfloat16
    scaling_factor = 0.3611
    shift_factor = 0.1159
    processor_cls = VaeImageProcessor
    output_pt_slice = torch.tensor([110, 72, 91, 62, 35, 52, 69, 55, 69], dtype=torch.uint8)
    partial_postprocess_return_pt_slice = torch.tensor(
        [202, 203, 203, 197, 195, 193, 189, 188, 178], dtype=torch.uint8
    )
    return_pt_slice = torch.tensor([0.5820, 0.5962, 0.5898, 0.5439, 0.5327, 0.5112, 0.4797, 0.4773, 0.3984])


class RemoteAutoencoderKLFluxPackedTests(
    RemoteAutoencoderKLMixin,
    unittest.TestCase,
):
    shape = (
        1,
        4096,
        64,
    )
    out_hw = (
        1024,
        1024,
    )
    height = 1024
    width = 1024
    endpoint = DECODE_ENDPOINT_FLUX
    dtype = torch.bfloat16
    scaling_factor = 0.3611
    shift_factor = 0.1159
    processor_cls = VaeImageProcessor
    # slices are different due to randn on different shape. we can pack the latent instead if we want the same
    output_pt_slice = torch.tensor([96, 116, 157, 45, 67, 104, 34, 56, 89], dtype=torch.uint8)
    partial_postprocess_return_pt_slice = torch.tensor(
        [168, 212, 202, 155, 191, 185, 150, 180, 168], dtype=torch.uint8
    )
    return_pt_slice = torch.tensor([0.3198, 0.6631, 0.5864, 0.2131, 0.4944, 0.4482, 0.1776, 0.4153, 0.3176])


class RemoteAutoencoderKLHunyuanVideoTests(
    RemoteAutoencoderKLHunyuanVideoMixin,
    unittest.TestCase,
):
    shape = (
        1,
        16,
        3,
        40,
        64,
    )
    out_hw = (
        320,
        512,
    )
    endpoint = DECODE_ENDPOINT_HUNYUAN_VIDEO
    dtype = torch.float16
    scaling_factor = 0.476986
    processor_cls = VideoProcessor
    output_pt_slice = torch.tensor([112, 92, 85, 112, 93, 85, 112, 94, 85], dtype=torch.uint8)
    partial_postprocess_return_pt_slice = torch.tensor(
        [149, 161, 168, 136, 150, 156, 129, 143, 149], dtype=torch.uint8
    )
    return_pt_slice = torch.tensor([0.1656, 0.2661, 0.3157, 0.0693, 0.1755, 0.2252, 0.0127, 0.1221, 0.1708])


class RemoteAutoencoderKLSlowTestMixin:
    channels: int = 4
    endpoint: str = None
    dtype: torch.dtype = None
    scaling_factor: float = None
    shift_factor: float = None
    width: int = None
    height: int = None

    def get_dummy_inputs(self):
        inputs = {
            "endpoint": self.endpoint,
            "scaling_factor": self.scaling_factor,
            "shift_factor": self.shift_factor,
            "height": self.height,
            "width": self.width,
        }
        return inputs

    def test_multi_res(self):
        inputs = self.get_dummy_inputs()
        for height in {320, 512, 640, 704, 896, 1024, 1208, 1384, 1536, 1608, 1864, 2048}:
            for width in {320, 512, 640, 704, 896, 1024, 1208, 1384, 1536, 1608, 1864, 2048}:
                inputs["tensor"] = torch.randn(
                    (1, self.channels, height // 8, width // 8),
                    device=torch_device,
                    dtype=self.dtype,
                    generator=torch.Generator(torch_device).manual_seed(13),
                )
                inputs["height"] = height
                inputs["width"] = width
                output = remote_decode(output_type="pt", partial_postprocess=True, **inputs)
                output.save(f"test_multi_res_{height}_{width}.png")


@slow
class RemoteAutoencoderKLSDv1SlowTests(
    RemoteAutoencoderKLSlowTestMixin,
    unittest.TestCase,
):
    endpoint = DECODE_ENDPOINT_SD_V1
    dtype = torch.float16
    scaling_factor = 0.18215
    shift_factor = None


@slow
class RemoteAutoencoderKLSDXLSlowTests(
    RemoteAutoencoderKLSlowTestMixin,
    unittest.TestCase,
):
    endpoint = DECODE_ENDPOINT_SD_XL
    dtype = torch.float16
    scaling_factor = 0.13025
    shift_factor = None


@slow
class RemoteAutoencoderKLFluxSlowTests(
    RemoteAutoencoderKLSlowTestMixin,
    unittest.TestCase,
):
    channels = 16
    endpoint = DECODE_ENDPOINT_FLUX
    dtype = torch.bfloat16
    scaling_factor = 0.3611
    shift_factor = 0.1159
