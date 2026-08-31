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

import pytest
import torch

from diffusers import KandinskyCombinedPipeline, KandinskyImg2ImgCombinedPipeline, KandinskyInpaintCombinedPipeline
from diffusers.utils import is_transformers_version

from ...testing_utils import assert_tensors_close, enable_full_determinism
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
)
from .test_kandinsky import KandinskyPipelineTesterConfig
from .test_kandinsky_img2img import KandinskyImg2ImgPipelineTesterConfig
from .test_kandinsky_inpaint import KandinskyInpaintPipelineTesterConfig
from .test_kandinsky_prior import KandinskyPriorPipelineTesterConfig


enable_full_determinism()


# The combined pipelines chain the prior onto a decoder pipeline, so their components are the decoder's plus the
# prior's under a `prior_` prefix, and their inputs are the prior's (the image embeddings the decoder would take are
# produced internally).
DEVICE_MAP_SKIP_REASON = "Combined pipelines are not supported by `device_map`."


class KandinskyCombinedPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = KandinskyCombinedPipeline
    required_input_params_in_call_signature = frozenset(["prompt"])
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    output_shape = (3, 64, 64)

    def get_dummy_components(self):
        components = KandinskyPipelineTesterConfig().get_dummy_components()
        components.update(
            {f"prior_{k}": v for k, v in KandinskyPriorPipelineTesterConfig().get_dummy_components().items()}
        )
        return components

    def get_dummy_inputs(self):
        inputs = KandinskyPriorPipelineTesterConfig().get_dummy_inputs()
        inputs.update({"height": 64, "width": 64})
        return inputs


class TestKandinskyCombinedPipeline(KandinskyCombinedPipelineTesterConfig, PipelineTesterMixin):
    @pytest.mark.xfail(
        condition=is_transformers_version(">=", "4.56.2"),
        reason="Latest transformers changes the slices",
        strict=False,
    )
    def test_kandinsky(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs()).images
        image_from_tuple = pipe(**self.get_dummy_inputs(), return_dict=False)[0]

        assert image.shape == (1, *self.output_shape)

        # The decoder pipeline only denormalizes for `output_type` "np"/"pil", so `"pt"` hands back the raw decoder
        # output. Map it into the [0, 1] range the expected slice below was recorded in.
        image = (image * 0.5 + 0.5).clamp(0, 1)
        image_from_tuple = (image_from_tuple * 0.5 + 0.5).clamp(0, 1)

        # fmt: off
        expected_slice = torch.tensor([0.2893, 0.1464, 0.4603, 0.3529, 0.4612, 0.7701, 0.4027, 0.3051, 0.5155])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)
        assert_tensors_close(image_from_tuple[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-1):
        # Batched inference is only approximately equal to single inference here: the batch pads to the longest
        # prompt and the tiny 2-step denoising loop amplifies the resulting attention differences. Tolerance set
        # from the measured drift.
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    def test_dict_tuple_outputs_equivalent(self, expected_slice=None, expected_max_difference=5e-4):
        super().test_dict_tuple_outputs_equivalent(
            expected_slice=expected_slice, expected_max_difference=expected_max_difference
        )


class TestKandinskyCombinedPipelineMemory(KandinskyCombinedPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the combined Kandinsky
    pipeline."""

    @pytest.mark.skip(DEVICE_MAP_SKIP_REASON)
    def test_pipeline_with_accelerator_device_map(self):
        pass


class KandinskyImg2ImgCombinedPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = KandinskyImg2ImgCombinedPipeline
    required_input_params_in_call_signature = frozenset(["prompt", "image"])
    batch_input_params = frozenset(["prompt", "negative_prompt", "image"])
    output_shape = (3, 64, 64)

    def get_dummy_components(self):
        components = KandinskyImg2ImgPipelineTesterConfig().get_dummy_components()
        components.update(
            {f"prior_{k}": v for k, v in KandinskyPriorPipelineTesterConfig().get_dummy_components().items()}
        )
        return components

    def get_dummy_inputs(self):
        inputs = KandinskyPriorPipelineTesterConfig().get_dummy_inputs()
        inputs.update(KandinskyImg2ImgPipelineTesterConfig().get_dummy_inputs())
        # The decoder's image embeddings come from the prior, not from the caller.
        inputs.pop("image_embeds")
        inputs.pop("negative_image_embeds")
        return inputs


class TestKandinskyImg2ImgCombinedPipeline(KandinskyImg2ImgCombinedPipelineTesterConfig, PipelineTesterMixin):
    @pytest.mark.xfail(
        condition=is_transformers_version(">=", "4.56.2"),
        reason="Latest transformers changes the slices",
        strict=False,
    )
    def test_kandinsky(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs()).images
        image_from_tuple = pipe(**self.get_dummy_inputs(), return_dict=False)[0]

        assert image.shape == (1, *self.output_shape)

        # fmt: off
        expected_slice = torch.tensor([0.4852, 0.4136, 0.4539, 0.4781, 0.4680, 0.5217, 0.4973, 0.4089, 0.4977])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)
        assert_tensors_close(image_from_tuple[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-2):
        # Batched inference is only approximately equal to single inference here: the batch pads to the longest
        # prompt and the tiny 2-step denoising loop amplifies the resulting attention differences. Tolerance set
        # from the measured drift.
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    def test_dict_tuple_outputs_equivalent(self, expected_slice=None, expected_max_difference=5e-4):
        super().test_dict_tuple_outputs_equivalent(
            expected_slice=expected_slice, expected_max_difference=expected_max_difference
        )

    def test_save_load_optional_components(self, tmp_path, expected_max_difference=5e-4):
        super().test_save_load_optional_components(tmp_path, expected_max_difference=expected_max_difference)


class TestKandinskyImg2ImgCombinedPipelineMemory(KandinskyImg2ImgCombinedPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the combined Kandinsky img2img
    pipeline."""

    @pytest.mark.skip(DEVICE_MAP_SKIP_REASON)
    def test_pipeline_with_accelerator_device_map(self):
        pass


class KandinskyInpaintCombinedPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = KandinskyInpaintCombinedPipeline
    required_input_params_in_call_signature = frozenset(["prompt", "image", "mask_image"])
    batch_input_params = frozenset(["prompt", "negative_prompt", "image", "mask_image"])
    output_shape = (3, 64, 64)

    def get_dummy_components(self):
        components = KandinskyInpaintPipelineTesterConfig().get_dummy_components()
        components.update(
            {f"prior_{k}": v for k, v in KandinskyPriorPipelineTesterConfig().get_dummy_components().items()}
        )
        return components

    def get_dummy_inputs(self):
        inputs = KandinskyPriorPipelineTesterConfig().get_dummy_inputs()
        inputs.update(KandinskyInpaintPipelineTesterConfig().get_dummy_inputs())
        # The decoder's image embeddings come from the prior, not from the caller.
        inputs.pop("image_embeds")
        inputs.pop("negative_image_embeds")
        return inputs


class TestKandinskyInpaintCombinedPipeline(KandinskyInpaintCombinedPipelineTesterConfig, PipelineTesterMixin):
    @pytest.mark.xfail(
        condition=is_transformers_version(">=", "4.56.2"),
        reason="Latest transformers changes the slices",
        strict=False,
    )
    def test_kandinsky(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs()).images
        image_from_tuple = pipe(**self.get_dummy_inputs(), return_dict=False)[0]

        assert image.shape == (1, *self.output_shape)

        # The decoder pipeline only denormalizes for `output_type` "np"/"pil", so `"pt"` hands back the raw decoder
        # output. Map it into the [0, 1] range the expected slice below was recorded in.
        image = (image * 0.5 + 0.5).clamp(0, 1)
        image_from_tuple = (image_from_tuple * 0.5 + 0.5).clamp(0, 1)

        # fmt: off
        expected_slice = torch.tensor([0.0320, 0.0860, 0.4013, 0.0518, 0.2484, 0.5847, 0.4411, 0.2321, 0.4593])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)
        assert_tensors_close(image_from_tuple[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-1):
        # Batched inference is only approximately equal to single inference here: the batch pads to the longest
        # prompt and the tiny 2-step denoising loop amplifies the resulting attention differences. Tolerance set
        # from the measured drift.
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    def test_dict_tuple_outputs_equivalent(self, expected_slice=None, expected_max_difference=5e-4):
        super().test_dict_tuple_outputs_equivalent(
            expected_slice=expected_slice, expected_max_difference=expected_max_difference
        )

    def test_save_load_optional_components(self, tmp_path, expected_max_difference=5e-4):
        super().test_save_load_optional_components(tmp_path, expected_max_difference=expected_max_difference)

    def test_save_load_local(self, tmp_path, base_pipe_output, expected_max_difference=5e-3):
        super().test_save_load_local(tmp_path, base_pipe_output, expected_max_difference=expected_max_difference)


class TestKandinskyInpaintCombinedPipelineMemory(KandinskyInpaintCombinedPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the combined Kandinsky inpaint
    pipeline."""

    @pytest.mark.skip(DEVICE_MAP_SKIP_REASON)
    def test_pipeline_with_accelerator_device_map(self):
        pass
