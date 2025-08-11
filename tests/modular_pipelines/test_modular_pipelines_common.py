import gc
import tempfile
import unittest
from typing import Callable, Union

import numpy as np
import torch

import diffusers
from diffusers import ComponentsManager, ModularPipeline, ModularPipelineBlocks
from diffusers.utils import logging
from diffusers.utils.testing_utils import (
    backend_empty_cache,
    numpy_cosine_similarity_distance,
    require_accelerator,
    require_torch,
    torch_device,
)


def to_np(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    return tensor


@require_torch
class ModularPipelineTesterMixin:
    """
    This mixin is designed to be used with unittest.TestCase classes.
    It provides a set of common tests for each modular pipeline,
    including:
    - test_pipeline_call_signature: check if the pipeline's __call__ method has all required parameters
    - test_inference_batch_consistent: check if the pipeline's __call__ method can handle batch inputs
    - test_inference_batch_single_identical: check if the pipeline's __call__ method can handle single input
    - test_float16_inference: check if the pipeline's __call__ method can handle float16 inputs
    - test_to_device: check if the pipeline's __call__ method can handle different devices
    """

    # Canonical parameters that are passed to `__call__` regardless
    # of the type of pipeline. They are always optional and have common
    # sense default values.
    optional_params = frozenset(
        [
            "num_inference_steps",
            "num_images_per_prompt",
            "latents",
            "output_type",
        ]
    )
    # this is modular specific: generator needs to be a intermediate input because it's mutable
    intermediate_params = frozenset(
        [
            "generator",
        ]
    )

    def get_generator(self, seed):
        device = torch_device if torch_device != "mps" else "cpu"
        generator = torch.Generator(device).manual_seed(seed)
        return generator

    @property
    def pipeline_class(self) -> Union[Callable, ModularPipeline]:
        raise NotImplementedError(
            "You need to set the attribute `pipeline_class = ClassNameOfPipeline` in the child test class. "
            "See existing pipeline tests for reference."
        )

    @property
    def repo(self) -> str:
        raise NotImplementedError(
            "You need to set the attribute `repo` in the child test class. See existing pipeline tests for reference."
        )

    @property
    def pipeline_blocks_class(self) -> Union[Callable, ModularPipelineBlocks]:
        raise NotImplementedError(
            "You need to set the attribute `pipeline_blocks_class = ClassNameOfPipelineBlocks` in the child test class. "
            "See existing pipeline tests for reference."
        )

    def get_pipeline(self):
        raise NotImplementedError(
            "You need to implement `get_pipeline(self)` in the child test class. "
            "See existing pipeline tests for reference."
        )

    def get_dummy_inputs(self, device, seed=0):
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

    def setUp(self):
        # clean up the VRAM before each test
        super().setUp()
        torch.compiler.reset()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        # clean up the VRAM after each test in case of CUDA runtime errors
        super().tearDown()
        torch.compiler.reset()
        gc.collect()
        backend_empty_cache(torch_device)

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

    def test_inference_batch_consistent(self, batch_sizes=[2], batch_generator=True):
        pipe = self.get_pipeline()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
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
            output = pipe(**batched_input, output="images")
            assert len(output) == batch_size, "Output is different from expected batch size"

    def test_inference_batch_single_identical(
        self,
        batch_size=2,
        expected_max_diff=1e-4,
    ):
        pipe = self.get_pipeline()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs(torch_device)

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

        output = pipe(**inputs, output="images")
        output_batch = pipe(**batched_inputs, output="images")

        assert output_batch.shape[0] == batch_size

        max_diff = np.abs(to_np(output_batch[0]) - to_np(output[0])).max()
        assert max_diff < expected_max_diff, "Batch inference results different from single inference results"

    @unittest.skipIf(torch_device not in ["cuda", "xpu"], reason="float16 requires CUDA or XPU")
    @require_accelerator
    def test_float16_inference(self, expected_max_diff=5e-2):
        pipe = self.get_pipeline()
        pipe.to(torch_device, torch.float32)
        pipe.set_progress_bar_config(disable=None)

        pipe_fp16 = self.get_pipeline()
        pipe_fp16.to(torch_device, torch.float16)
        pipe_fp16.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        # Reset generator in case it is used inside dummy inputs
        if "generator" in inputs:
            inputs["generator"] = self.get_generator(0)
        output = pipe(**inputs, output="images")

        fp16_inputs = self.get_dummy_inputs(torch_device)
        # Reset generator in case it is used inside dummy inputs
        if "generator" in fp16_inputs:
            fp16_inputs["generator"] = self.get_generator(0)
        output_fp16 = pipe_fp16(**fp16_inputs, output="images")

        if isinstance(output, torch.Tensor):
            output = output.cpu()
            output_fp16 = output_fp16.cpu()

        max_diff = numpy_cosine_similarity_distance(output.flatten(), output_fp16.flatten())
        assert max_diff < expected_max_diff, "FP16 inference is different from FP32 inference"

    @require_accelerator
    def test_to_device(self):
        pipe = self.get_pipeline()
        pipe.set_progress_bar_config(disable=None)

        pipe.to("cpu")
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
        pipe = self.get_pipeline()
        pipe.set_progress_bar_config(disable=None)
        pipe.to("cpu")

        output = pipe(**self.get_dummy_inputs("cpu"), output="images")
        assert np.isnan(to_np(output)).sum() == 0, "CPU Inference returns NaN"

    @require_accelerator
    def test_inference_is_not_nan(self):
        pipe = self.get_pipeline()
        pipe.set_progress_bar_config(disable=None)
        pipe.to(torch_device)

        output = pipe(**self.get_dummy_inputs(torch_device), output="images")
        assert np.isnan(to_np(output)).sum() == 0, "Accelerator Inference returns NaN"

    def test_num_images_per_prompt(self):
        pipe = self.get_pipeline()

        if "num_images_per_prompt" not in pipe.blocks.input_names:
            return

        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        batch_sizes = [1, 2]
        num_images_per_prompts = [1, 2]

        for batch_size in batch_sizes:
            for num_images_per_prompt in num_images_per_prompts:
                inputs = self.get_dummy_inputs(torch_device)

                for key in inputs.keys():
                    if key in self.batch_params:
                        inputs[key] = batch_size * [inputs[key]]

                images = pipe(**inputs, num_images_per_prompt=num_images_per_prompt, output="images")

                assert images.shape[0] == batch_size * num_images_per_prompt

    @require_accelerator
    def test_components_auto_cpu_offload_inference_consistent(self):
        base_pipe = self.get_pipeline().to(torch_device)

        cm = ComponentsManager()
        cm.enable_auto_cpu_offload(device=torch_device)
        offload_pipe = self.get_pipeline(components_manager=cm)

        image_slices = []
        for pipe in [base_pipe, offload_pipe]:
            inputs = self.get_dummy_inputs(torch_device)
            image = pipe(**inputs, output="images")

            image_slices.append(image[0, -3:, -3:, -1].flatten())

        assert np.abs(image_slices[0] - image_slices[1]).max() < 1e-3

    def test_save_from_pretrained(self):
        pipes = []
        base_pipe = self.get_pipeline().to(torch_device)
        pipes.append(base_pipe)

        with tempfile.TemporaryDirectory() as tmpdirname:
            base_pipe.save_pretrained(tmpdirname)
            pipe = ModularPipeline.from_pretrained(tmpdirname).to(torch_device)
            pipe.load_default_components(torch_dtype=torch.float32)
            pipe.to(torch_device)

        pipes.append(pipe)

        image_slices = []
        for pipe in pipes:
            inputs = self.get_dummy_inputs(torch_device)
            image = pipe(**inputs, output="images")

            image_slices.append(image[0, -3:, -3:, -1].flatten())

        assert np.abs(image_slices[0] - image_slices[1]).max() < 1e-3
