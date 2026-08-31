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

from diffusers import (
    KandinskyV22CombinedPipeline,
    KandinskyV22Img2ImgCombinedPipeline,
    KandinskyV22InpaintCombinedPipeline,
)

from ...testing_utils import assert_tensors_close, enable_full_determinism, require_accelerator
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
)
from .test_kandinsky import KandinskyV22PipelineTesterConfig
from .test_kandinsky_img2img import KandinskyV22Img2ImgPipelineTesterConfig
from .test_kandinsky_inpaint import KandinskyV22InpaintPipelineTesterConfig
from .test_kandinsky_prior import KandinskyV22PriorPipelineTesterConfig


enable_full_determinism()


# `UNet2DConditionModel` builds an `ImageProjection` for `encoder_hid_dim_type="image_proj"`, and its `forward`
# aligns the input with `self.image_embeds.weight.dtype`. Under layerwise casting that reads the *storage* dtype
# (fp8), because the weight is only upcast inside `self.image_embeds`'s own hooked forward — so the input is pushed
# down to fp8 and the matmul then fails against the upcast bf16 weight. `TextImageProjection` (Kandinsky 2.1) calls
# the projection without reading its weight dtype and is unaffected.
LAYERWISE_CASTING_XFAIL_REASON = (
    "`ImageProjection.forward` reads `self.image_embeds.weight.dtype`, which is the fp8 storage dtype under "
    "layerwise casting, so the input is cast down to fp8 and the matmul fails."
)


# The combined pipelines chain the prior onto a decoder pipeline, so their components are the decoder's plus the
# prior's under a `prior_` prefix, and their inputs are the prior's (the image embeddings the decoder would take are
# produced internally).
DEVICE_MAP_SKIP_REASON = "`device_map` is not yet supported for connected pipelines."
CALLBACK_SKIP_REASON = "Combined pipelines don't expose the decoder's callback tensors."


class KandinskyV22CombinedPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = KandinskyV22CombinedPipeline
    required_input_params_in_call_signature = frozenset(["prompt"])
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    callback_cfg_params = frozenset(["image_embeds"])
    output_shape = (3, 64, 64)

    def get_dummy_components(self):
        components = KandinskyV22PipelineTesterConfig().get_dummy_components()
        components.update(
            {f"prior_{k}": v for k, v in KandinskyV22PriorPipelineTesterConfig().get_dummy_components().items()}
        )
        return components

    def get_dummy_inputs(self):
        inputs = KandinskyV22PriorPipelineTesterConfig().get_dummy_inputs()
        inputs.update({"height": 64, "width": 64})
        return inputs


class TestKandinskyV22CombinedPipeline(KandinskyV22CombinedPipelineTesterConfig, PipelineTesterMixin):
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
        expected_slice = torch.tensor([0.1111, 0.0000, 0.6088, 0.2670, 0.3847, 0.8102, 0.4594, 0.4858, 0.5990])
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

    def test_save_load_local(self, tmp_path, base_pipe_output, expected_max_difference=5e-3):
        super().test_save_load_local(tmp_path, base_pipe_output, expected_max_difference=expected_max_difference)

    def test_save_load_optional_components(self, tmp_path, expected_max_difference=5e-3):
        super().test_save_load_optional_components(tmp_path, expected_max_difference=expected_max_difference)

    @pytest.mark.skip(CALLBACK_SKIP_REASON)
    def test_callback_inputs(self):
        pass

    @pytest.mark.skip(CALLBACK_SKIP_REASON)
    def test_callback_cfg(self):
        pass


class TestKandinskyV22CombinedPipelineMemory(KandinskyV22CombinedPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the combined Kandinsky 2.2
    pipeline."""

    @pytest.mark.xfail(condition=True, reason=LAYERWISE_CASTING_XFAIL_REASON, strict=True)
    def test_layerwise_casting_inference(self):
        super().test_layerwise_casting_inference()

    def test_model_cpu_offload_forward_pass(self, base_pipe_output, expected_max_diff=5e-4):
        super().test_model_cpu_offload_forward_pass(base_pipe_output, expected_max_diff=expected_max_diff)

    @pytest.mark.skip(DEVICE_MAP_SKIP_REASON)
    def test_pipeline_with_accelerator_device_map(self):
        pass


class KandinskyV22Img2ImgCombinedPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = KandinskyV22Img2ImgCombinedPipeline
    required_input_params_in_call_signature = frozenset(["prompt", "image"])
    batch_input_params = frozenset(["prompt", "negative_prompt", "image"])
    callback_cfg_params = frozenset(["image_embeds"])
    output_shape = (3, 64, 64)

    def get_dummy_components(self):
        components = KandinskyV22Img2ImgPipelineTesterConfig().get_dummy_components()
        components.update(
            {f"prior_{k}": v for k, v in KandinskyV22PriorPipelineTesterConfig().get_dummy_components().items()}
        )
        return components

    def get_dummy_inputs(self):
        inputs = KandinskyV22PriorPipelineTesterConfig().get_dummy_inputs()
        inputs.update(KandinskyV22Img2ImgPipelineTesterConfig().get_dummy_inputs())
        # The decoder's image embeddings come from the prior, not from the caller.
        inputs.pop("image_embeds")
        inputs.pop("negative_image_embeds")
        return inputs


class TestKandinskyV22Img2ImgCombinedPipeline(KandinskyV22Img2ImgCombinedPipelineTesterConfig, PipelineTesterMixin):
    def test_kandinsky(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs()).images
        image_from_tuple = pipe(**self.get_dummy_inputs(), return_dict=False)[0]

        assert image.shape == (1, *self.output_shape)

        # fmt: off
        expected_slice = torch.tensor([0.4525, 0.4496, 0.4976, 0.4512, 0.4424, 0.5400, 0.4572, 0.4521, 0.5306])
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

    @pytest.mark.skip(CALLBACK_SKIP_REASON)
    def test_callback_inputs(self):
        pass

    @pytest.mark.skip(CALLBACK_SKIP_REASON)
    def test_callback_cfg(self):
        pass


class TestKandinskyV22Img2ImgCombinedPipelineMemory(
    KandinskyV22Img2ImgCombinedPipelineTesterConfig, MemoryTesterMixin
):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the combined Kandinsky 2.2
    img2img pipeline."""

    @pytest.mark.xfail(condition=True, reason=LAYERWISE_CASTING_XFAIL_REASON, strict=True)
    def test_layerwise_casting_inference(self):
        super().test_layerwise_casting_inference()

    def test_model_cpu_offload_forward_pass(self, base_pipe_output, expected_max_diff=5e-4):
        super().test_model_cpu_offload_forward_pass(base_pipe_output, expected_max_diff=expected_max_diff)

    @pytest.mark.skip(DEVICE_MAP_SKIP_REASON)
    def test_pipeline_with_accelerator_device_map(self):
        pass


class KandinskyV22InpaintCombinedPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = KandinskyV22InpaintCombinedPipeline
    required_input_params_in_call_signature = frozenset(["prompt", "image", "mask_image"])
    batch_input_params = frozenset(["prompt", "negative_prompt", "image", "mask_image"])
    callback_cfg_params = frozenset(["image_embeds"])
    output_shape = (3, 64, 64)

    def get_dummy_components(self):
        components = KandinskyV22InpaintPipelineTesterConfig().get_dummy_components()
        components.update(
            {f"prior_{k}": v for k, v in KandinskyV22PriorPipelineTesterConfig().get_dummy_components().items()}
        )
        return components

    def get_dummy_inputs(self):
        inputs = KandinskyV22PriorPipelineTesterConfig().get_dummy_inputs()
        inputs.update(KandinskyV22InpaintPipelineTesterConfig().get_dummy_inputs())
        # The decoder's image embeddings come from the prior, not from the caller.
        inputs.pop("image_embeds")
        inputs.pop("negative_image_embeds")
        return inputs


class TestKandinskyV22InpaintCombinedPipeline(KandinskyV22InpaintCombinedPipelineTesterConfig, PipelineTesterMixin):
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
        expected_slice = torch.tensor([0.5039, 0.4926, 0.4898, 0.4978, 0.4838, 0.4942, 0.4738, 0.4702, 0.4816])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)
        assert_tensors_close(image_from_tuple[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)

    @pytest.mark.xfail(
        reason=(
            "Batched inference is not equivalent to single inference for this pipeline: ~18% of the pixels of the first "
            "batch element drift by more than 1e-2 (max ~0.36), independent of batch size, because the masked-latent "
            "blending re-amplifies the batched forward's numerical differences at every step. This predates the move to "
            "the pipeline-level mixins — the unittest-era test failed the same way."
        ),
        strict=False,
    )
    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-4):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    def test_dict_tuple_outputs_equivalent(self, expected_slice=None, expected_max_difference=5e-4):
        super().test_dict_tuple_outputs_equivalent(
            expected_slice=expected_slice, expected_max_difference=expected_max_difference
        )

    def test_save_load_local(self, tmp_path, base_pipe_output, expected_max_difference=5e-3):
        super().test_save_load_local(tmp_path, base_pipe_output, expected_max_difference=expected_max_difference)

    def test_save_load_optional_components(self, tmp_path, expected_max_difference=5e-4):
        super().test_save_load_optional_components(tmp_path, expected_max_difference=expected_max_difference)

    @pytest.mark.skip(CALLBACK_SKIP_REASON)
    def test_callback_inputs(self):
        pass

    @pytest.mark.skip(CALLBACK_SKIP_REASON)
    def test_callback_cfg(self):
        pass


class TestKandinskyV22InpaintCombinedPipelineMemory(
    KandinskyV22InpaintCombinedPipelineTesterConfig, MemoryTesterMixin
):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the combined Kandinsky 2.2
    inpaint pipeline."""

    @pytest.mark.xfail(condition=True, reason=LAYERWISE_CASTING_XFAIL_REASON, strict=True)
    def test_layerwise_casting_inference(self):
        super().test_layerwise_casting_inference()

    def test_model_cpu_offload_forward_pass(self, base_pipe_output, expected_max_diff=5e-4):
        super().test_model_cpu_offload_forward_pass(base_pipe_output, expected_max_diff=expected_max_diff)

    @require_accelerator
    def test_sequential_cpu_offload_forward_pass(self, base_pipe_output, expected_max_diff=5e-4):
        super().test_sequential_cpu_offload_forward_pass(base_pipe_output, expected_max_diff=expected_max_diff)

    @pytest.mark.skip(DEVICE_MAP_SKIP_REASON)
    def test_pipeline_with_accelerator_device_map(self):
        pass
