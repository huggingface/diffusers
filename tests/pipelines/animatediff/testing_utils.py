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

from diffusers import DPMSolverMultistepScheduler, LCMScheduler, UNetMotionModel
from diffusers.models.attention import FreeNoiseTransformerBlock

from ...testing_utils import assert_tensors_close, require_accelerator, torch_device
from ..testing_utils import BasePipelineTesterConfig, PipelineTesterMixin
from ..testing_utils.common import BasePipelineOutputMixin


# `PipelineFromPipeTesterMixin` (tests/pipelines/test_pipelines_common.py) still covers `from_pipe` forward-pass
# parity and the model-CPU-offload round trip, but it is unittest-era: its tests call `self.get_dummy_inputs(device,
# seed=0)` and `self.assertLess`, neither of which exists on a `BasePipelineTesterConfig` outside a
# `unittest.TestCase`. Un-skipping the parked classes below without porting the mixin first would error, not fail.
# The mixin is still live for the ten `tests/pipelines/pag/` files and `stable_diffusion_adapter`; rewrite it
# pytest-style when those are migrated, then drop these skips.
FROM_PIPE_SKIP_REASON = (
    "`PipelineFromPipeTesterMixin` is still unittest-style and cannot run against `BasePipelineTesterConfig` — "
    "these error rather than fail if un-skipped. Port the mixin to pytest (due when `tests/pipelines/pag/` is "
    "migrated), then remove this skip."
)


class MotionPipelineTesterConfig(BasePipelineTesterConfig):
    """`BasePipelineTesterConfig` for the AnimateDiff pipelines in this directory."""

    # AnimateDiff pipelines generate video, so they expose `num_videos_per_prompt`, not `num_images_per_prompt`.
    optional_input_params = frozenset(
        ["num_inference_steps", "num_videos_per_prompt", "generator", "latents", "output_type", "return_dict"]
    )


class MotionPipelineTesterMixin(PipelineTesterMixin):
    """`PipelineTesterMixin` for the AnimateDiff pipelines in this directory.

    They wrap the `unet` they are constructed with in a `UNetMotionModel`, so the device/dtype checks have to read
    the components off the built pipeline rather than off the dict `get_dummy_components()` returned.
    """

    def test_motion_unet_loading(self):
        pipe = self.get_pipeline()

        assert isinstance(pipe.unet, UNetMotionModel)

    @require_accelerator
    def test_to_device(self):
        pipe = self.get_pipeline()

        pipe.to("cpu")
        model_devices = [
            component.device.type for component in pipe.components.values() if getattr(component, "device", None)
        ]
        assert all(device == "cpu" for device in model_devices)

        output_cpu = pipe(**self.get_dummy_inputs())[0]
        assert torch.isnan(output_cpu).sum() == 0

        pipe.to(torch_device)
        model_devices = [
            component.device.type for component in pipe.components.values() if getattr(component, "device", None)
        ]
        assert all(device == torch_device for device in model_devices)

        output_device = pipe(**self.get_dummy_inputs())[0]
        assert torch.isnan(output_device).sum() == 0

    def test_to_dtype(self):
        pipe = self.get_pipeline()

        model_dtypes = [component.dtype for component in pipe.components.values() if getattr(component, "dtype", None)]
        assert all(dtype == torch.float32 for dtype in model_dtypes)

        pipe.to(dtype=torch.float16)
        model_dtypes = [component.dtype for component in pipe.components.values() if getattr(component, "dtype", None)]
        assert all(dtype == torch.float16 for dtype in model_dtypes)

    def test_inference_batch_single_identical(self, batch_size=2, expected_max_diff=1e-4):
        # A batch of 3 makes the longest prompt in the batch 100 * "very long", which the tiny text encoder here
        # truncates differently per element; 2 keeps the comparison meaningful while still exercising batching.
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    def test_prompt_embeds(self):
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        inputs.pop("prompt")
        inputs["prompt_embeds"] = torch.randn((1, 4, pipe.text_encoder.config.hidden_size), device=torch_device)
        pipe(**inputs)

    def test_vae_slicing(self, video_count=2):
        # Run on CPU to keep the device-dependent `torch.Generator` deterministic.
        pipe = self.get_pipeline()

        def batched_inputs():
            inputs = self.get_dummy_inputs()
            for name in self.batch_input_params:
                if name in inputs:
                    inputs[name] = [inputs[name]] * video_count
            return inputs

        output_1 = pipe(**batched_inputs())[0]

        # make sure sliced vae decode yields the same result
        pipe.vae.enable_slicing()
        output_2 = pipe(**batched_inputs())[0]

        assert_tensors_close(output_2, output_1, atol=1e-2, msg="VAE slicing should not affect the inference results.")

    def test_encode_prompt_works_in_isolation(self):
        extra_required_param_value_dict = {
            "device": torch.device(torch_device).type,
            "num_images_per_prompt": 1,
            "do_classifier_free_guidance": self.get_dummy_inputs().get("guidance_scale", 1.0) > 1.0,
        }
        return super().test_encode_prompt_works_in_isolation(extra_required_param_value_dict)


class FreeInitTesterMixin(BasePipelineOutputMixin):
    """FreeInit tests shared by the AnimateDiff pipelines in this directory."""

    def test_free_init(self):
        pipe = self.get_pipeline().to(torch_device)

        frames_normal = self.run_pipe(pipe)[0]

        pipe.enable_free_init(
            num_iters=2,
            use_fast_sampling=True,
            method="butterworth",
            order=4,
            spatial_stop_frequency=0.25,
            temporal_stop_frequency=0.25,
        )
        frames_enable_free_init = self.run_pipe(pipe)[0]

        pipe.disable_free_init()
        frames_disable_free_init = self.run_pipe(pipe)[0]

        sum_enabled = (frames_normal - frames_enable_free_init).abs().sum()
        assert sum_enabled > 1e1, (
            "Enabling of FreeInit should lead to results different from the default pipeline results"
        )
        assert_tensors_close(
            frames_disable_free_init,
            frames_normal,
            atol=1e-4,
            msg="Disabling of FreeInit should lead to results similar to the default pipeline results",
        )

    def test_free_init_with_schedulers(self):
        components = self.get_dummy_components()
        pipe = self.get_pipeline(**components).to(torch_device)

        frames_normal = self.run_pipe(pipe)[0]

        schedulers_to_test = [
            DPMSolverMultistepScheduler.from_config(
                components["scheduler"].config,
                timestep_spacing="linspace",
                beta_schedule="linear",
                algorithm_type="dpmsolver++",
                steps_offset=1,
                clip_sample=False,
            ),
            LCMScheduler.from_config(
                components["scheduler"].config,
                timestep_spacing="linspace",
                beta_schedule="linear",
                steps_offset=1,
                clip_sample=False,
            ),
        ]
        components.pop("scheduler")

        for scheduler in schedulers_to_test:
            components["scheduler"] = scheduler
            pipe = self.get_pipeline(**components).to(torch_device)
            pipe.enable_free_init(num_iters=2, use_fast_sampling=False)

            frames_enable_free_init = self.run_pipe(pipe)[0]
            sum_enabled = (frames_normal - frames_enable_free_init).abs().sum()

            assert sum_enabled > 1e1, (
                "Enabling of FreeInit should lead to results different from the default pipeline results"
            )


class FreeNoiseTesterMixin(BasePipelineOutputMixin):
    """FreeNoise tests shared by the AnimateDiff pipelines in this directory."""

    def get_free_noise_inputs(self):
        """Dummy inputs for the longer (16-frame) runs the FreeNoise context windows need.

        Override on pipelines whose frame count is derived from an input (a conditioning video, for example)
        rather than from the `num_frames` argument.
        """
        return {**self.get_dummy_inputs(), "num_frames": 16}

    def test_free_noise_blocks(self):
        pipe = self.get_pipeline().to(torch_device)

        pipe.enable_free_noise()
        for block in pipe.unet.down_blocks:
            for motion_module in block.motion_modules:
                for transformer_block in motion_module.transformer_blocks:
                    assert isinstance(transformer_block, FreeNoiseTransformerBlock), (
                        "Motion module transformer blocks must be an instance of `FreeNoiseTransformerBlock` after enabling FreeNoise."
                    )

        pipe.disable_free_noise()
        for block in pipe.unet.down_blocks:
            for motion_module in block.motion_modules:
                for transformer_block in motion_module.transformer_blocks:
                    assert not isinstance(transformer_block, FreeNoiseTransformerBlock), (
                        "Motion module transformer blocks must not be an instance of `FreeNoiseTransformerBlock` after disabling FreeNoise."
                    )

    def test_free_noise(self):
        pipe = self.get_pipeline().to(torch_device)

        torch.manual_seed(0)
        frames_normal = pipe(**self.get_free_noise_inputs()).frames[0]

        for context_length in [8, 9]:
            for context_stride in [4, 6]:
                pipe.enable_free_noise(context_length, context_stride)

                torch.manual_seed(0)
                frames_enable_free_noise = pipe(**self.get_free_noise_inputs()).frames[0]

                pipe.disable_free_noise()

                torch.manual_seed(0)
                frames_disable_free_noise = pipe(**self.get_free_noise_inputs()).frames[0]

                sum_enabled = (frames_normal - frames_enable_free_noise).abs().sum()
                assert sum_enabled > 1e1, (
                    "Enabling of FreeNoise should lead to results different from the default pipeline results"
                )
                assert_tensors_close(
                    frames_disable_free_noise,
                    frames_normal,
                    atol=1e-4,
                    msg="Disabling of FreeNoise should lead to results similar to the default pipeline results",
                )

    def test_free_noise_multi_prompt(self):
        pipe = self.get_pipeline().to(torch_device)

        context_length = 8
        context_stride = 4
        pipe.enable_free_noise(context_length, context_stride)

        # Make sure that pipeline works when prompt indices are within num_frames bounds
        inputs = self.get_free_noise_inputs()
        inputs["prompt"] = {0: "Caterpillar on a leaf", 10: "Butterfly on a leaf"}
        pipe(**inputs)

        with pytest.raises(ValueError):
            # Ensure that prompt indices are within bounds
            inputs = self.get_free_noise_inputs()
            inputs["prompt"] = {0: "Caterpillar on a leaf", 10: "Butterfly on a leaf", 42: "Error on a leaf"}
            pipe(**inputs)


class FreeNoiseSplitInferenceTesterMixin(FreeNoiseTesterMixin):
    """Adds the FreeNoise split-inference memory optimization test to `FreeNoiseTesterMixin`."""

    def test_free_noise_split_inference(self):
        pipe = self.get_pipeline().to(torch_device)

        pipe.enable_free_noise(8, 4)

        torch.manual_seed(0)
        frames_normal = pipe(**self.get_free_noise_inputs()).frames[0]

        # Test FreeNoise with split inference memory-optimization
        pipe.enable_free_noise_split_inference(spatial_split_size=16, temporal_split_size=4)

        torch.manual_seed(0)
        frames_enable_split_inference = pipe(**self.get_free_noise_inputs()).frames[0]

        # Split inference only reorders the same math, so compare per-element: summing the absolute differences
        # instead would scale the tolerance with the number of pixels and fail on accumulated float noise alone.
        assert_tensors_close(
            frames_enable_split_inference,
            frames_normal,
            atol=1e-4,
            msg=(
                "Enabling FreeNoise Split Inference memory-optimizations should lead to results similar to the "
                "default pipeline results"
            ),
        )
