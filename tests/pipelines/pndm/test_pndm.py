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

import numpy as np
import pytest
import torch

from diffusers import PNDMPipeline, PNDMScheduler, UNet2DModel

from ...testing_utils import (
    enable_full_determinism,
    nightly,
    require_accelerator,
    require_torch,
    torch_device,
)
from ..pipeline_params import UNCONDITIONAL_IMAGE_GENERATION_BATCH_PARAMS, UNCONDITIONAL_IMAGE_GENERATION_PARAMS
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


enable_full_determinism()


# `PNDMPipeline.__call__` postprocesses unconditionally — it always runs `image.cpu().permute(0, 2, 3, 1).numpy()`
# and only branches afterwards to decide whether to wrap the result in PIL. There is no `output_type="pt"` path, so
# `get_dummy_inputs` below has to ask for `"np"`, and every shared test that compares outputs with
# `assert_tensors_close` (torch-only) or calls `torch.isnan` on them fails on the numpy array it gets back.
#
# The sibling unconditional pipelines already have a `"pt"` path (`DDIMPipeline`, `DDPMPipeline`); adding one here
# is a `src/` change and out of scope for this test migration, so the affected tests are marked `xfail` rather than
# skipped: whoever adds the `"pt"` branch will see them XPASS and can drop these markers.
NO_PT_OUTPUT = pytest.mark.xfail(
    reason="`PNDMPipeline` has no `output_type='pt'` path and always returns a numpy array.",
    strict=True,
)

# `PNDMPipeline` samples its initial noise with `randn_tensor(..., device=self.device)`: no `dtype=`, so the noise
# stays float32 and disagrees with a half-precision or layerwise-cast UNet. Same story as above — a `src/` gap, so
# the accelerator-only tests it breaks are marked `xfail`.
UNSUPPORTED_DTYPE = pytest.mark.xfail(
    reason="`PNDMPipeline` samples its initial noise without `dtype=self.unet.dtype`, so it stays float32.",
    strict=True,
)

# The memory mixin trips over three separate `src/` gaps at once, so its tests carry one marker between them:
#   - the numpy output above (`test_group_offloading_inference`, `test_pipeline_level_group_offloading_inference`,
#     `test_pipeline_with_accelerator_device_map`),
#   - the float32 noise above (`test_layerwise_casting_inference`),
#   - `randn_tensor(..., device=self.device)` reading the *pipeline's* device rather than `self._execution_device`,
#     which under sequential offload is `meta` (`test_sequential_cpu_offload_forward_pass`,
#     `test_sequential_offload_forward_pass_twice`),
#   - and no `model_cpu_offload_seq`, which `enable_model_cpu_offload` requires
#     (`test_model_cpu_offload_forward_pass`, `test_cpu_offload_forward_pass_twice`).
# `strict=False` because `test_pipeline_level_group_offloading_sanity_checks` never runs the pipeline and so passes
# — it reports XPASS. The class-level marker is what keeps `MemoryTesterMixin`'s own `@is_memory` /
# `@require_accelerator` marks intact; overriding the eight failing tests individually would drop the
# `@require_accelerate_version_greater` gates they are declared with.
UNSUPPORTED_MEMORY_OPTIMIZATIONS = pytest.mark.xfail(
    reason=(
        "`PNDMPipeline` returns numpy, samples noise without `dtype=`/`_execution_device`, and declares no "
        "`model_cpu_offload_seq`."
    ),
    strict=False,
)


class PNDMPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = PNDMPipeline
    required_input_params_in_call_signature = UNCONDITIONAL_IMAGE_GENERATION_PARAMS
    batch_input_params = UNCONDITIONAL_IMAGE_GENERATION_BATCH_PARAMS
    # PNDM is unconditional and samples its own noise: there is no prompt to repeat
    # (`num_images_per_prompt`) and no user-suppliable `latents`.
    optional_input_params = frozenset(["num_inference_steps", "generator", "output_type", "return_dict"])
    # `(height, width, channels)` — the numpy layout, not the `(channels, height, width)` the other configs get
    # from `output_type="pt"` (see `NO_PT_OUTPUT` above).
    output_shape = (8, 8, 3)

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DModel(
            block_out_channels=(4, 8),
            layers_per_block=1,
            norm_num_groups=4,
            sample_size=8,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        )
        scheduler = PNDMScheduler()
        return {"unet": unet, "scheduler": scheduler}

    def get_dummy_inputs(self):
        return {
            "batch_size": 1,
            "generator": self.get_generator(0),
            # `PNDMScheduler` runs Runge-Kutta warm-up steps, so it needs at least 4 inference steps.
            "num_inference_steps": 4,
            # `"np"` rather than the usual `"pt"` — see `NO_PT_OUTPUT` above.
            "output_type": "np",
        }


class TestPNDMPipeline(PNDMPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs()).images
        generated_image = image[0]
        assert generated_image.shape == self.output_shape

        expected_slice = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        assert np.abs(generated_image[-3:, -3:, -1].flatten() - expected_slice).max() < 1e-2

    @NO_PT_OUTPUT
    def test_save_load_local(self, tmp_path, base_pipe_output, expected_max_difference=5e-4):
        super().test_save_load_local(tmp_path, base_pipe_output, expected_max_difference)

    @NO_PT_OUTPUT
    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical()

    @NO_PT_OUTPUT
    def test_dict_tuple_outputs_equivalent(self):
        super().test_dict_tuple_outputs_equivalent()

    # The three overrides below re-declare the base methods' skip decorators: overriding a test drops the marks
    # the base declared it with, and without them these would run (and xfail for the wrong reason) on CPU.
    @NO_PT_OUTPUT
    @require_accelerator
    def test_to_device(self):
        super().test_to_device()

    @UNSUPPORTED_DTYPE
    @pytest.mark.skipif(torch_device not in ["cuda", "xpu"], reason="half-precision inference requires CUDA or XPU")
    @require_accelerator
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=str)
    def test_half_precision_inference_no_nan(self, dtype):
        super().test_half_precision_inference_no_nan(dtype)

    @UNSUPPORTED_DTYPE
    @pytest.mark.skipif(torch_device not in ["cuda", "xpu"], reason="float16 requires CUDA or XPU")
    @require_accelerator
    def test_save_load_float16(self, tmp_path, expected_max_diff=1e-2):
        super().test_save_load_float16(tmp_path, expected_max_diff)


@UNSUPPORTED_MEMORY_OPTIMIZATIONS
class TestPNDMPipelineMemory(PNDMPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the PNDM pipeline."""


@nightly
@require_torch
class TestPNDMPipelineIntegration:
    def test_inference_cifar10(self):
        model_id = "google/ddpm-cifar10-32"

        unet = UNet2DModel.from_pretrained(model_id)
        scheduler = PNDMScheduler()

        pndm = PNDMPipeline(unet=unet, scheduler=scheduler)
        pndm.to(torch_device)
        pndm.set_progress_bar_config(disable=None)
        generator = torch.manual_seed(0)
        image = pndm(generator=generator, output_type="np").images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.1564, 0.14645, 0.1406, 0.14715, 0.12425, 0.14045, 0.13115, 0.12175, 0.125])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
