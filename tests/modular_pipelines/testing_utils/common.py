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

import gc
from typing import Callable

import pytest
import torch

import diffusers
from diffusers import ModularPipeline, ModularPipelineBlocks
from diffusers.utils import logging

from ...testing_utils import (
    backend_empty_cache,
    numpy_cosine_similarity_distance,
    require_accelerator,
    torch_device,
)


class BaseModularPipelineTesterConfig:
    """
    Base class defining the configuration interface for modular pipeline testing.

    A concrete config must set `pipeline_blocks_class` and `pretrained_model_name_or_path` and implement
    `get_dummy_inputs()`; `params` and `batch_params` declare which inputs the blocks are expected to accept and which
    of them are batched. This class only declares the testing contract; the pipeline builder and the cached reference
    output live on `BaseModularPipelineOutputMixin` (mirroring the non-modular `BasePipelineTesterConfig`).
    """

    # Canonical parameters that are passed to `__call__` regardless of the type of pipeline. They are always
    # optional and have common sense default values.
    optional_params = frozenset(["num_inference_steps", "num_images_per_prompt", "latents", "output_type"])
    # Parameters the pipeline deliberately does NOT accept — e.g. `negative_prompt` on a
    # guidance-distilled pipeline. `test_pipeline_call_signature` asserts they are absent,
    # so accidentally (re)introducing one fails the test.
    not_params = frozenset()
    # this is modular specific: generator needs to be a intermediate input because it's mutable
    intermediate_params = frozenset(["generator"])
    # Output type for the pipeline (e.g., "images" for image pipelines, "videos" for video pipelines)
    # Subclasses can override this to change the expected output type
    output_name = "images"

    # ==================== Required interface ====================

    @property
    def pipeline_class(self) -> Callable | ModularPipeline:
        raise NotImplementedError(
            "You need to set the attribute `pipeline_class = ClassNameOfPipeline` in the child test class. "
            "See existing pipeline tests for reference."
        )

    @property
    def pretrained_model_name_or_path(self) -> str:
        raise NotImplementedError(
            "You need to set the attribute `pretrained_model_name_or_path` in the child test class. See existing pipeline tests for reference."
        )

    @property
    def pipeline_blocks_class(self) -> Callable | ModularPipelineBlocks:
        raise NotImplementedError(
            "You need to set the attribute `pipeline_blocks_class = ClassNameOfPipelineBlocks` in the child test class. "
            "See existing pipeline tests for reference."
        )

    def get_dummy_inputs(self, seed=0):
        raise NotImplementedError(
            "You need to implement `get_dummy_inputs(self, device, seed)` in the child test class. "
            "See existing pipeline tests for reference."
        )

    @property
    def params(self) -> frozenset:
        raise NotImplementedError(
            "You need to set the attribute `params` in the child test class. "
            "`params` are checked for if all values are present in `__call__`'s signature."
            " You can set `params` using one of the common set of parameters defined in `pipeline_params.py`"
            " e.g., `TEXT_TO_IMAGE_PARAMS` defines the common parameters used in text to  "
            "image pipelines, including prompts and prompt embedding overrides."
            "If your pipeline's set of arguments has minor changes from one of the common sets of arguments, "
            "do not make modifications to the existing common sets of arguments. I.e. a text to image pipeline "
            "with non-configurable height and width arguments should set the attribute as "
            "`params = TEXT_TO_IMAGE_PARAMS - {'height', 'width'}`. "
            "See existing pipeline tests for reference."
        )

    @property
    def batch_params(self) -> frozenset:
        raise NotImplementedError(
            "You need to set the attribute `batch_params` in the child test class. "
            "`batch_params` are the parameters required to be batched when passed to the pipeline's "
            "`__call__` method. `pipeline_params.py` provides some common sets of parameters such as "
            "`TEXT_TO_IMAGE_BATCH_PARAMS`, `IMAGE_VARIATION_BATCH_PARAMS`, etc... If your pipeline's "
            "set of batch arguments has minor changes from one of the common sets of batch arguments, "
            "do not make modifications to the existing common sets of batch arguments. I.e. a text to "
            "image pipeline `negative_prompt` is not batched should set the attribute as "
            "`batch_params = TEXT_TO_IMAGE_BATCH_PARAMS - {'negative_prompt'}`. "
            "See existing pipeline tests for reference."
        )

    @property
    def expected_workflow_blocks(self) -> dict:
        raise NotImplementedError(
            "You need to set the attribute `expected_workflow_blocks` in the child test class. "
            "`expected_workflow_blocks` is a dictionary that maps workflow names to list of block names. "
            "See existing pipeline tests for reference."
        )

    # ==================== Shared helpers ====================

    def get_generator(self, seed=0):
        # Always build the generator on CPU: a CPU generator works with a pipeline placed on any device (the tensor
        # is created on CPU and moved), whereas an accelerator generator cannot seed a CPU tensor, which the tests
        # that run the pipeline on CPU rely on.
        return torch.Generator("cpu").manual_seed(seed)

    # ==================== Fixtures ====================

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Free VRAM before/after each test (replaces unittest setUp/tearDown)."""
        torch.compiler.reset()
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        torch.compiler.reset()
        gc.collect()
        backend_empty_cache(torch_device)


class BaseModularPipelineOutputMixin:
    """Provides the `get_pipeline` builder and the class-scoped `base_pipe_output` fixture shared across tester
    mixins.

    Kept separate from `BaseModularPipelineTesterConfig` — which only declares the testing contract and performs no
    computation — so any mixin that needs to build a pipeline or read the cached reference output
    (`ModularPipelineTesterMixin`, the loading and memory mixins, ...) can inherit it without duplicating the
    build-and-forward.
    """

    def get_pipeline(self, components_manager=None, dtype=torch.float32):
        """Build the pipeline under test from `pipeline_blocks_class` and load its components.

        The pipeline is left wherever `load_components` put it — callers that need it elsewhere should chain
        `.to(torch_device)`.
        """
        pipeline = self.pipeline_blocks_class().init_pipeline(
            self.pretrained_model_name_or_path, components_manager=components_manager
        )
        pipeline.load_components(dtype=dtype)
        pipeline.set_progress_bar_config(disable=None)
        return pipeline

    @pytest.fixture(scope="class")
    def base_pipe_output(self):
        """Output of a freshly built pipeline on the standard dummy inputs, computed once per test class."""
        pipe = self.get_pipeline().to(torch_device)
        return pipe(**self.get_dummy_inputs(), output=self.output_name)


class ModularPipelineTesterMixin(BaseModularPipelineOutputMixin):
    """
    Common inference tests for each modular pipeline: call signature, batching, dtype/device handling and NaN-free
    outputs.

    Designed to be composed with `BaseModularPipelineTesterConfig` (which provides `pipeline_blocks_class`,
    `pretrained_model_name_or_path`, `get_dummy_inputs()` and the shared fixtures).
    """

    def test_pipeline_call_signature(self):
        pipe = self.get_pipeline()
        input_parameters = pipe.blocks.input_names
        optional_parameters = pipe.default_call_parameters

        def _check_for_parameters(parameters, expected_parameters, param_type):
            remaining_parameters = {param for param in parameters if param not in expected_parameters}
            assert len(remaining_parameters) == 0, (
                f"Required {param_type} parameters not present: {remaining_parameters}"
            )

        _check_for_parameters(self.params, input_parameters, "input")
        _check_for_parameters(self.optional_params, optional_parameters, "optional")

        unsupported_parameters = {param for param in self.not_params if param in input_parameters}
        assert len(unsupported_parameters) == 0, (
            f"Parameters declared in `not_params` unexpectedly present in the pipeline inputs: {unsupported_parameters}"
        )

    def test_inference_batch_consistent(self, batch_sizes=[2], batch_generator=True):
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        inputs["generator"] = self.get_generator(0)

        logger = logging.get_logger(pipe.__module__)
        logger.setLevel(level=diffusers.logging.FATAL)

        # prepare batched inputs
        batched_inputs = []
        for batch_size in batch_sizes:
            batched_input = {}
            batched_input.update(inputs)

            for name in self.batch_params:
                if name not in inputs:
                    continue

                value = inputs[name]
                batched_input[name] = batch_size * [value]

            if batch_generator and "generator" in inputs:
                batched_input["generator"] = [self.get_generator(i) for i in range(batch_size)]

            if "batch_size" in inputs:
                batched_input["batch_size"] = batch_size

            batched_inputs.append(batched_input)

        logger.setLevel(level=diffusers.logging.WARNING)
        for batch_size, batched_input in zip(batch_sizes, batched_inputs):
            output = pipe(**batched_input, output=self.output_name)
            assert len(output) == batch_size, "Output is different from expected batch size"

    def test_inference_batch_single_identical(
        self,
        batch_size=2,
        expected_max_diff=1e-4,
    ):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()

        # Reset generator in case it is has been used in self.get_dummy_inputs
        inputs["generator"] = self.get_generator(0)

        logger = logging.get_logger(pipe.__module__)
        logger.setLevel(level=diffusers.logging.FATAL)

        # batchify inputs
        batched_inputs = {}
        batched_inputs.update(inputs)

        for name in self.batch_params:
            if name not in inputs:
                continue

            value = inputs[name]
            batched_inputs[name] = batch_size * [value]

        if "generator" in inputs:
            batched_inputs["generator"] = [self.get_generator(i) for i in range(batch_size)]

        if "batch_size" in inputs:
            batched_inputs["batch_size"] = batch_size

        output = pipe(**inputs, output=self.output_name)
        output_batch = pipe(**batched_inputs, output=self.output_name)

        assert output_batch.shape[0] == batch_size

        # For batch comparison, we only need to compare the first item
        if output_batch.shape[0] == batch_size and output.shape[0] == 1:
            output_batch = output_batch[0:1]

        max_diff = torch.abs(output_batch - output).max()
        assert max_diff < expected_max_diff, "Batch inference results different from single inference results"

    @require_accelerator
    def test_float16_inference(self, expected_max_diff=5e-2):
        pipe = self.get_pipeline()
        pipe.to(torch_device, torch.float32)

        pipe_fp16 = self.get_pipeline()
        pipe_fp16.to(torch_device, torch.float16)

        inputs = self.get_dummy_inputs()
        # Reset generator in case it is used inside dummy inputs
        if "generator" in inputs:
            inputs["generator"] = self.get_generator(0)

        output = pipe(**inputs, output=self.output_name)

        fp16_inputs = self.get_dummy_inputs()
        # Reset generator in case it is used inside dummy inputs
        if "generator" in fp16_inputs:
            fp16_inputs["generator"] = self.get_generator(0)

        output_fp16 = pipe_fp16(**fp16_inputs, output=self.output_name)

        output_tensor = output.float().cpu()
        output_fp16_tensor = output_fp16.float().cpu()

        # Check for NaNs in outputs (can happen with tiny models in FP16)
        if torch.isnan(output_tensor).any() or torch.isnan(output_fp16_tensor).any():
            pytest.skip("FP16 inference produces NaN values - this is a known issue with tiny models")

        max_diff = numpy_cosine_similarity_distance(
            output_tensor.flatten().numpy(), output_fp16_tensor.flatten().numpy()
        )

        # Check if cosine similarity is NaN (which can happen if vectors are zero or very small)
        if torch.isnan(torch.tensor(max_diff)):
            pytest.skip("Cosine similarity is NaN - outputs may be too small for reliable comparison")

        assert max_diff < expected_max_diff, f"FP16 inference is different from FP32 inference (max_diff: {max_diff})"

    @require_accelerator
    def test_to_device(self):
        pipe = self.get_pipeline().to("cpu")

        model_devices = [
            component.device.type for component in pipe.components.values() if hasattr(component, "device")
        ]
        assert all(device == "cpu" for device in model_devices), "All pipeline components are not on CPU"

        pipe.to(torch_device)
        model_devices = [
            component.device.type for component in pipe.components.values() if hasattr(component, "device")
        ]
        assert all(device == torch_device for device in model_devices), (
            "All pipeline components are not on accelerator device"
        )

    def test_inference_is_not_nan_cpu(self):
        pipe = self.get_pipeline().to("cpu")

        inputs = self.get_dummy_inputs()
        output = pipe(**inputs, output=self.output_name)
        assert torch.isnan(output).sum() == 0, "CPU Inference returns NaN"

    @require_accelerator
    def test_inference_is_not_nan(self, base_pipe_output):
        assert torch.isnan(base_pipe_output).sum() == 0, "Accelerator Inference returns NaN"

    def test_num_images_per_prompt(self, batch_sizes=[1, 2], num_images_per_prompts=[1, 2]):
        pipe = self.get_pipeline().to(torch_device)

        if "num_images_per_prompt" not in pipe.blocks.input_names:
            pytest.skip("Skipping test as `num_images_per_prompt` is not present in input names.")

        for batch_size in batch_sizes:
            for num_images_per_prompt in num_images_per_prompts:
                inputs = self.get_dummy_inputs()

                for key in inputs.keys():
                    if key in self.batch_params:
                        inputs[key] = batch_size * [inputs[key]]

                images = pipe(**inputs, num_images_per_prompt=num_images_per_prompt, output=self.output_name)

                assert images.shape[0] == batch_size * num_images_per_prompt
