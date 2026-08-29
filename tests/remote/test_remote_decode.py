# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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

import re

import numpy as np
import PIL.Image
import pytest
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
    shape: tuple[int, ...] = None
    out_hw: tuple[int, int] = None
    endpoint: str = None
    dtype: torch.dtype = None
    scaling_factor: float = None
    shift_factor: float = None
    processor_cls: VaeImageProcessor | VideoProcessor = None
    output_pil_slice: torch.Tensor = None
    output_pt_slice: torch.Tensor = None
    partial_postprocess_return_pt_slice: torch.Tensor = None
    return_pt_slice: torch.Tensor = None
    width: int = None
    height: int = None

    def get_dummy_inputs(self):
        # The tensor is serialized and decoded remotely, so the local device has no bearing on the result.
        # Seed it on CPU and move it afterwards, otherwise `torch.randn` yields different latents on a CUDA
        # runner than on a CPU one and the reference slices below only match on whichever recorded them.
        inputs = {
            "endpoint": self.endpoint,
            "tensor": torch.randn(
                self.shape,
                device="cpu",
                dtype=self.dtype,
                generator=torch.Generator("cpu").manual_seed(13),
            ).to(torch_device),
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
        assert isinstance(output, PIL.Image.Image), f"Expected `PIL.Image.Image` output, got {type(output)}"
        assert output.height == self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output.height}"
        assert output.width == self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output.width}"
        output_slice = torch.from_numpy(np.array(output)[0, -3:, -3:].flatten())
        # Increased tolerance for Flux Packed diff [1, 0, 1, 0, 0, 0, 0, 0, 0]
        assert torch_all_close(output_slice, self.output_pt_slice.to(output_slice.dtype), rtol=1, atol=1)

    def test_output_type_pt(self):
        inputs = self.get_dummy_inputs()
        processor = self.processor_cls()
        output = remote_decode(output_type="pt", processor=processor, **inputs)
        assert isinstance(output, PIL.Image.Image), f"Expected `PIL.Image.Image` output, got {type(output)}"
        assert output.height == self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output.height}"
        assert output.width == self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output.width}"
        output_slice = torch.from_numpy(np.array(output)[0, -3:, -3:].flatten())
        assert torch_all_close(output_slice, self.output_pt_slice.to(output_slice.dtype), rtol=1e-2)

    # output is visually the same, slice is flaky?
    def test_output_type_pil(self):
        inputs = self.get_dummy_inputs()
        output = remote_decode(output_type="pil", **inputs)
        assert isinstance(output, PIL.Image.Image), f"Expected `PIL.Image.Image` output, got {type(output)}"
        assert output.height == self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output.height}"
        assert output.width == self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output.width}"

    def test_output_type_pil_image_format(self):
        inputs = self.get_dummy_inputs()
        output = remote_decode(output_type="pil", image_format="png", **inputs)
        assert isinstance(output, PIL.Image.Image), f"Expected `PIL.Image.Image` output, got {type(output)}"
        assert output.height == self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output.height}"
        assert output.width == self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output.width}"
        assert output.format == "png", f"Expected image format `png`, got {output.format}"
        output_slice = torch.from_numpy(np.array(output)[0, -3:, -3:].flatten())
        assert torch_all_close(output_slice, self.output_pt_slice.to(output_slice.dtype), rtol=1e-2)

    def test_output_type_pt_partial_postprocess(self):
        inputs = self.get_dummy_inputs()
        output = remote_decode(output_type="pt", partial_postprocess=True, **inputs)
        assert isinstance(output, PIL.Image.Image), f"Expected `PIL.Image.Image` output, got {type(output)}"
        assert output.height == self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output.height}"
        assert output.width == self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output.width}"
        output_slice = torch.from_numpy(np.array(output)[0, -3:, -3:].flatten())
        assert torch_all_close(output_slice, self.output_pt_slice.to(output_slice.dtype), rtol=1e-2)

    def test_output_type_pt_return_type_pt(self):
        inputs = self.get_dummy_inputs()
        output = remote_decode(output_type="pt", return_type="pt", **inputs)
        assert isinstance(output, torch.Tensor), f"Expected `torch.Tensor` output, got {type(output)}"
        assert output.shape[2] == self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output.shape[2]}"
        assert output.shape[3] == self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output.shape[3]}"
        output_slice = output[0, 0, -3:, -3:].flatten()
        assert torch_all_close(output_slice, self.return_pt_slice.to(output_slice.dtype), rtol=1e-3, atol=1e-3)

    def test_output_type_pt_partial_postprocess_return_type_pt(self):
        inputs = self.get_dummy_inputs()
        output = remote_decode(output_type="pt", partial_postprocess=True, return_type="pt", **inputs)
        assert isinstance(output, torch.Tensor), f"Expected `torch.Tensor` output, got {type(output)}"
        assert output.shape[1] == self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output.shape[1]}"
        assert output.shape[2] == self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output.shape[2]}"
        output_slice = output[0, -3:, -3:, 0].flatten().cpu()
        assert torch_all_close(
            output_slice, self.partial_postprocess_return_pt_slice.to(output_slice.dtype), rtol=1e-2
        )

    def test_do_scaling_deprecation(self):
        inputs = self.get_dummy_inputs()
        inputs.pop("scaling_factor", None)
        inputs.pop("shift_factor", None)
        expected = "`do_scaling` is deprecated, pass `scaling_factor` and `shift_factor` if required."
        with pytest.warns(FutureWarning, match=re.escape(expected)):
            _ = remote_decode(output_type="pt", partial_postprocess=True, **inputs)

    def test_input_tensor_type_base64_deprecation(self):
        inputs = self.get_dummy_inputs()
        expected = "input_tensor_type='base64' is deprecated. Using `binary`."
        with pytest.warns(FutureWarning, match=re.escape(expected)):
            _ = remote_decode(output_type="pt", input_tensor_type="base64", partial_postprocess=True, **inputs)

    def test_output_tensor_type_base64_deprecation(self):
        inputs = self.get_dummy_inputs()
        expected = "output_tensor_type='base64' is deprecated. Using `binary`."
        with pytest.warns(FutureWarning, match=re.escape(expected)):
            _ = remote_decode(output_type="pt", output_tensor_type="base64", partial_postprocess=True, **inputs)


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
        assert isinstance(output, list) and isinstance(output[0], PIL.Image.Image), (
            f"Expected `List[PIL.Image.Image]` output, got {type(output)}"
        )
        assert output[0].height == self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output[0].height}"
        assert output[0].width == self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output[0].width}"
        output_slice = torch.from_numpy(np.array(output[0])[0, -3:, -3:].flatten())
        assert torch_all_close(output_slice, self.output_pt_slice.to(output_slice.dtype), rtol=1, atol=1)

    def test_output_type_pt(self):
        inputs = self.get_dummy_inputs()
        processor = self.processor_cls()
        output = remote_decode(output_type="pt", processor=processor, **inputs)
        assert isinstance(output, list) and isinstance(output[0], PIL.Image.Image), (
            f"Expected `List[PIL.Image.Image]` output, got {type(output)}"
        )
        assert output[0].height == self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output[0].height}"
        assert output[0].width == self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output[0].width}"
        output_slice = torch.from_numpy(np.array(output[0])[0, -3:, -3:].flatten())
        assert torch_all_close(output_slice, self.output_pt_slice.to(output_slice.dtype), rtol=1, atol=1)

    # output is visually the same, slice is flaky?
    def test_output_type_pil(self):
        inputs = self.get_dummy_inputs()
        processor = self.processor_cls()
        output = remote_decode(output_type="pil", processor=processor, **inputs)
        assert isinstance(output, list) and isinstance(output[0], PIL.Image.Image), (
            f"Expected `List[PIL.Image.Image]` output, got {type(output)}"
        )
        assert output[0].height == self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output[0].height}"
        assert output[0].width == self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output[0].width}"

    def test_output_type_pil_image_format(self):
        inputs = self.get_dummy_inputs()
        processor = self.processor_cls()
        output = remote_decode(output_type="pil", processor=processor, image_format="png", **inputs)
        assert isinstance(output, list) and isinstance(output[0], PIL.Image.Image), (
            f"Expected `List[PIL.Image.Image]` output, got {type(output)}"
        )
        assert output[0].height == self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output[0].height}"
        assert output[0].width == self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output[0].width}"
        output_slice = torch.from_numpy(np.array(output[0])[0, -3:, -3:].flatten())
        assert torch_all_close(output_slice, self.output_pt_slice.to(output_slice.dtype), rtol=1, atol=1)

    def test_output_type_pt_partial_postprocess(self):
        inputs = self.get_dummy_inputs()
        output = remote_decode(output_type="pt", partial_postprocess=True, **inputs)
        assert isinstance(output, list) and isinstance(output[0], PIL.Image.Image), (
            f"Expected `List[PIL.Image.Image]` output, got {type(output)}"
        )
        assert output[0].height == self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output[0].height}"
        assert output[0].width == self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output[0].width}"
        output_slice = torch.from_numpy(np.array(output[0])[0, -3:, -3:].flatten())
        assert torch_all_close(output_slice, self.output_pt_slice.to(output_slice.dtype), rtol=1, atol=1)

    def test_output_type_pt_return_type_pt(self):
        inputs = self.get_dummy_inputs()
        output = remote_decode(output_type="pt", return_type="pt", **inputs)
        assert isinstance(output, torch.Tensor), f"Expected `torch.Tensor` output, got {type(output)}"
        assert output.shape[3] == self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output.shape[3]}"
        assert output.shape[4] == self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output.shape[4]}"
        output_slice = output[0, 0, 0, -3:, -3:].flatten()
        assert torch_all_close(output_slice, self.return_pt_slice.to(output_slice.dtype), rtol=1e-3, atol=1e-3)

    def test_output_type_mp4(self):
        inputs = self.get_dummy_inputs()
        output = remote_decode(output_type="mp4", return_type="mp4", **inputs)
        assert isinstance(output, bytes), f"Expected `bytes` output, got {type(output)}"


# The tests below hit live HF Inference Endpoints, which are not part of the fast CI contract, so they are
# gated behind `RUN_SLOW` instead of running on every push. The reference slices are recorded against the
# deployed `DECODE_ENDPOINT_*` models and drift if those are redeployed.
@slow
class TestRemoteAutoencoderKLSDv1(RemoteAutoencoderKLMixin):
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
    output_pt_slice = torch.tensor([162, 131, 123, 158, 131, 124, 148, 128, 115], dtype=torch.uint8)
    partial_postprocess_return_pt_slice = torch.tensor(
        [133, 146, 130, 161, 141, 132, 148, 134, 135], dtype=torch.uint8
    )
    return_pt_slice = torch.tensor([0.0400, 0.1448, 0.0217, 0.2649, 0.1049, 0.0376, 0.1622, 0.0552, 0.0574])


@slow
class TestRemoteAutoencoderKLSDXL(RemoteAutoencoderKLMixin):
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
    output_pt_slice = torch.tensor([133, 182, 166, 131, 179, 163, 122, 173, 151], dtype=torch.uint8)
    partial_postprocess_return_pt_slice = torch.tensor(
        [172, 160, 147, 149, 136, 126, 118, 112, 107], dtype=torch.uint8
    )
    return_pt_slice = torch.tensor([0.3511, 0.2539, 0.1550, 0.1720, 0.0688, -0.0117, -0.0730, -0.1215, -0.1581])


@slow
class TestRemoteAutoencoderKLFlux(RemoteAutoencoderKLMixin):
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
    output_pt_slice = torch.tensor([236, 222, 171, 226, 208, 160, 210, 193, 154], dtype=torch.uint8)
    partial_postprocess_return_pt_slice = torch.tensor(
        [173, 174, 173, 188, 182, 178, 183, 175, 167], dtype=torch.uint8
    )
    return_pt_slice = torch.tensor([0.3538, 0.3613, 0.3582, 0.4749, 0.4294, 0.3977, 0.4336, 0.3738, 0.3098])


@slow
class TestRemoteAutoencoderKLFluxPacked(RemoteAutoencoderKLMixin):
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
    output_pt_slice = torch.tensor([203, 188, 127, 163, 140, 89, 113, 86, 50], dtype=torch.uint8)
    partial_postprocess_return_pt_slice = torch.tensor(
        [194, 173, 131, 192, 168, 126, 187, 158, 129], dtype=torch.uint8
    )
    return_pt_slice = torch.tensor([0.5259, 0.3564, 0.0272, 0.5093, 0.3171, -0.0111, 0.4639, 0.2371, 0.0083])


@slow
class TestRemoteAutoencoderKLHunyuanVideo(RemoteAutoencoderKLHunyuanVideoMixin):
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
    output_pt_slice = torch.tensor([126, 129, 195, 128, 130, 196, 129, 130, 196], dtype=torch.uint8)
    partial_postprocess_return_pt_slice = torch.tensor([19, 16, 13, 30, 25, 21, 39, 35, 31], dtype=torch.uint8)
    return_pt_slice = torch.tensor([-0.8501, -0.8750, -0.8979, -0.7681, -0.8027, -0.8389, -0.6958, -0.7222, -0.7593])


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

    def test_multi_res(self, tmp_path):
        inputs = self.get_dummy_inputs()
        for height in {320, 512, 640, 704, 896, 1024, 1208, 1384, 1536, 1608, 1864, 2048}:
            for width in {320, 512, 640, 704, 896, 1024, 1208, 1384, 1536, 1608, 1864, 2048}:
                inputs["tensor"] = torch.randn(
                    (1, self.channels, height // 8, width // 8),
                    device="cpu",
                    dtype=self.dtype,
                    generator=torch.Generator("cpu").manual_seed(13),
                ).to(torch_device)
                inputs["height"] = height
                inputs["width"] = width
                output = remote_decode(output_type="pt", partial_postprocess=True, **inputs)
                output.save(tmp_path / f"test_multi_res_{height}_{width}.png")


@slow
class TestRemoteAutoencoderKLSDv1Slow(RemoteAutoencoderKLSlowTestMixin):
    endpoint = DECODE_ENDPOINT_SD_V1
    dtype = torch.float16
    scaling_factor = 0.18215
    shift_factor = None


@slow
class TestRemoteAutoencoderKLSDXLSlow(RemoteAutoencoderKLSlowTestMixin):
    endpoint = DECODE_ENDPOINT_SD_XL
    dtype = torch.float16
    scaling_factor = 0.13025
    shift_factor = None


@slow
class TestRemoteAutoencoderKLFluxSlow(RemoteAutoencoderKLSlowTestMixin):
    channels = 16
    endpoint = DECODE_ENDPOINT_FLUX
    dtype = torch.bfloat16
    scaling_factor = 0.3611
    shift_factor = 0.1159
