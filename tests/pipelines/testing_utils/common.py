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

import gc
import inspect
import json
import os
from typing import Callable

import pytest
import torch
import torch.nn as nn

import diffusers
from diffusers import DiffusionPipeline
from diffusers.utils import logging
from diffusers.utils.source_code_parsing_utils import ReturnNameVisitor

from ...testing_utils import (
    CaptureLogger,
    assert_tensors_close,
    backend_empty_cache,
    require_accelerator,
    torch_device,
)


class BasePipelineTesterConfig:
    """
    Base class defining the configuration interface for pipeline testing.

    A concrete pipeline test config must set `pipeline_class` and implement `get_dummy_components()` and
    `get_dummy_inputs()`. `required_input_params_in_call_signature` and `batch_input_params` should be set from the canonical sets in
    `tests/pipelines/pipeline_params.py`. This class only declares the testing contract; the cached reference
    output lives on `BasePipelineOutputMixin` (mirroring the model-level `BaseModelOutputMixin`).

    NOTE: `get_dummy_inputs` should request torch outputs (`output_type="pt"`) so tests operate on torch tensors
    exclusively and compare with `assert_tensors_close` directly (no numpy round-trip). Keep in mind that `"pt"`
    image outputs are laid out as `(batch, channels, height, width)`, unlike the `(batch, height, width, channels)`
    layout produced by `output_type="np"`.
    """

    # Canonical parameters that are passed to `__call__` regardless of the type of pipeline. They are always
    # optional and have common sense default values.
    optional_input_params = frozenset(
        [
            "num_inference_steps",
            "num_images_per_prompt",
            "generator",
            "latents",
            "output_type",
            "return_dict",
        ]
    )

    # ==================== Required interface ====================

    @property
    def pipeline_class(self) -> Callable | DiffusionPipeline:
        raise NotImplementedError(
            "You need to set the attribute `pipeline_class = ClassNameOfPipeline` in the child test class. "
            "See existing pipeline tests for reference."
        )

    def get_dummy_components(self):
        raise NotImplementedError(
            "You need to implement `get_dummy_components(self)` in the child test class. "
            "See existing pipeline tests for reference."
        )

    def get_dummy_inputs(self):
        raise NotImplementedError(
            "You need to implement `get_dummy_inputs(self)` in the child test class. Build the inputs on "
            "`torch_device` with a fixed seed (use the `get_generator` helper for the generator). "
            "Set `output_type='pt'` so the pipeline returns torch tensors (see the class docstring). "
            "See existing pipeline tests for reference."
        )

    @property
    def required_input_params_in_call_signature(self) -> frozenset:
        raise NotImplementedError(
            "You need to set the attribute `required_input_params_in_call_signature` in the child test class. "
            "`required_input_params_in_call_signature` are checked for if all values are present in `__call__`'s signature. "
            "You can set `required_input_params_in_call_signature` using one of the common set of parameters defined in `pipeline_params.py`."
        )

    @property
    def batch_input_params(self) -> frozenset:
        raise NotImplementedError(
            "You need to set the attribute `batch_input_params` in the child test class. "
            "`batch_input_params` are the parameters required to be batched when passed to the pipeline's `__call__` "
            "method. `pipeline_params.py` provides some common sets such as `TEXT_TO_IMAGE_BATCH_PARAMS`."
        )

    @property
    def callback_cfg_params(self) -> frozenset:
        raise NotImplementedError(
            "You need to set the attribute `callback_cfg_params` in the child test class that requires to run "
            "test_callback_cfg. `callback_cfg_params` are the parameters that need to be passed to the pipeline's "
            "callback function when dynamically adjusting `guidance_scale`."
        )

    # ==================== Shared helpers ====================

    def get_generator(self, seed=0):
        # Always build the generator on CPU: a CPU generator works with a pipeline placed on any device (the tensor
        # is created on CPU and moved), whereas an accelerator generator cannot seed a CPU tensor (see
        # `randn_tensor`), which `test_to_device` relies on when it runs the pipeline on CPU.
        return torch.Generator("cpu").manual_seed(seed)

    # ==================== Fixtures ====================

    @pytest.fixture(autouse=True)
    def skip_if_deprecated(self):
        """Skip deprecated pipelines."""
        from diffusers.pipelines.pipeline_utils import DeprecatedPipelineMixin

        if issubclass(self.pipeline_class, DeprecatedPipelineMixin):
            pytest.skip(reason=f"Deprecated Pipeline: {self.pipeline_class.__name__}")

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


class BasePipelineOutputMixin:
    """Provides the `get_pipeline` builder and the class-scoped `base_pipe_output` fixture shared across tester
    mixins.

    Kept separate from `BasePipelineTesterConfig` — which only declares the testing contract and performs no
    computation — so any mixin that needs to build a pipeline or read the cached reference output
    (`PipelineTesterMixin`, the memory offload mixins, ...) can inherit it without duplicating the
    build-and-forward. Mirrors the model-level `BaseModelOutputMixin`.
    """

    def get_pipeline(self, **components):
        """Build the pipeline under test, with every component in eval mode and the progress bar disabled.

        Pass components as keyword arguments to override the full `get_dummy_components()` set. The pipeline is
        left on CPU — callers that need it elsewhere should chain `.to(torch_device)`.
        """
        components = components or self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "eval"):
                component.eval()
        pipe.set_progress_bar_config(disable=None)

        return pipe

    @pytest.fixture(scope="class")
    def base_pipe_output(self):
        """Output of a freshly constructed pipeline on the standard dummy inputs, computed once per test class."""
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        torch.manual_seed(0)
        return pipe(**inputs)[0]


class PipelineTesterMixin(BasePipelineOutputMixin):
    """
    Common tests for each PyTorch pipeline: saving and loading, equivalence of dict and tuple outputs, batching,
    dtype/device handling, callbacks, and variants.

    Designed to be composed with `BasePipelineTesterConfig` (which provides `pipeline_class`,
    `get_dummy_components()`, `get_dummy_inputs()` and the shared fixtures).
    """

    def test_save_load_local(self, tmp_path, base_pipe_output, expected_max_difference=5e-4):
        pipe = self.get_pipeline().to(torch_device)

        logger = logging.get_logger("diffusers.pipelines.pipeline_utils")
        logger.setLevel(diffusers.logging.INFO)

        pipe.save_pretrained(tmp_path, safe_serialization=False)

        with CaptureLogger(logger) as cap_logger:
            pipe_loaded = self.pipeline_class.from_pretrained(tmp_path)

        for name in pipe_loaded.components.keys():
            if name not in pipe_loaded._optional_components:
                assert name in str(cap_logger)

        pipe_loaded.to(torch_device)
        pipe_loaded.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        torch.manual_seed(0)
        output_loaded = pipe_loaded(**inputs)[0]

        assert_tensors_close(
            output_loaded, base_pipe_output, atol=expected_max_difference, msg="Loaded pipeline output changed."
        )

    def test_pipeline_call_signature(self):
        assert hasattr(self.pipeline_class, "__call__"), f"{self.pipeline_class} should have a `__call__` method"

        parameters = inspect.signature(self.pipeline_class.__call__).parameters

        optional_parameters = set()
        for k, v in parameters.items():
            if v.default != inspect._empty:
                optional_parameters.add(k)

        parameters = set(parameters.keys())
        parameters.remove("self")
        parameters.discard("kwargs")  # kwargs can be added if arguments of pipeline call function are deprecated

        remaining_required_parameters = {
            param for param in self.required_input_params_in_call_signature if param not in parameters
        }
        assert len(remaining_required_parameters) == 0, (
            f"Required parameters not present: {remaining_required_parameters}"
        )

        remaining_required_optional_parameters = {
            param for param in self.optional_input_params if param not in optional_parameters
        }
        assert len(remaining_required_optional_parameters) == 0, (
            f"Required optional parameters not present: {remaining_required_optional_parameters}"
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

            for name in self.batch_input_params:
                if name not in inputs:
                    continue

                value = inputs[name]
                if name == "prompt":
                    len_prompt = len(value)
                    # make unequal batch sizes
                    batched_input[name] = [value[: len_prompt // i] for i in range(1, batch_size + 1)]
                    # make last batch super long
                    batched_input[name][-1] = 100 * "very long"
                else:
                    batched_input[name] = batch_size * [value]

            if batch_generator and "generator" in inputs:
                batched_input["generator"] = [self.get_generator(i) for i in range(batch_size)]

            if "batch_size" in inputs:
                batched_input["batch_size"] = batch_size

            batched_inputs.append(batched_input)

        logger.setLevel(level=diffusers.logging.WARNING)
        for batch_size, batched_input in zip(batch_sizes, batched_inputs):
            output = pipe(**batched_input)
            assert len(output[0]) == batch_size

    def test_inference_batch_single_identical(
        self, batch_size=3, expected_max_diff=1e-4, additional_params_copy_to_batched_inputs=["num_inference_steps"]
    ):
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        # Reset generator in case it has been used in self.get_dummy_inputs
        inputs["generator"] = self.get_generator(0)

        logger = logging.get_logger(pipe.__module__)
        logger.setLevel(level=diffusers.logging.FATAL)

        # batchify inputs
        batched_inputs = {}
        batched_inputs.update(inputs)

        for name in self.batch_input_params:
            if name not in inputs:
                continue

            value = inputs[name]
            if name == "prompt":
                len_prompt = len(value)
                batched_inputs[name] = [value[: len_prompt // i] for i in range(1, batch_size + 1)]
                batched_inputs[name][-1] = 100 * "very long"
            else:
                batched_inputs[name] = batch_size * [value]

        if "generator" in inputs:
            batched_inputs["generator"] = [self.get_generator(i) for i in range(batch_size)]

        if "batch_size" in inputs:
            batched_inputs["batch_size"] = batch_size

        for arg in additional_params_copy_to_batched_inputs:
            batched_inputs[arg] = inputs[arg]

        output = pipe(**inputs)
        output_batch = pipe(**batched_inputs)

        assert output_batch[0].shape[0] == batch_size

        assert_tensors_close(
            output_batch[0][0], output[0][0], atol=expected_max_diff, msg="Batched output differs from single."
        )

    def test_dict_tuple_outputs_equivalent(self, expected_slice=None, expected_max_difference=1e-4):
        pipe = self.get_pipeline().to(torch_device)

        if expected_slice is None:
            output = pipe(**self.get_dummy_inputs())[0]
        else:
            output = expected_slice

        output_tuple = pipe(**self.get_dummy_inputs(), return_dict=False)[0]

        if expected_slice is None:
            assert_tensors_close(
                output_tuple, output, atol=expected_max_difference, msg="Dict and tuple outputs are not equal."
            )
        else:
            if output_tuple.ndim != 5:
                output_tuple_slice = output_tuple[0, -3:, -3:, -1].flatten()
            else:
                output_tuple_slice = output_tuple[0, -3:, -3:, -1, -1].flatten()
            assert_tensors_close(
                output_tuple_slice, output, atol=expected_max_difference, msg="Dict and tuple outputs are not equal."
            )

    def test_components_function(self):
        init_components = self.get_dummy_components()
        init_components = {k: v for k, v in init_components.items() if not isinstance(v, (str, int, float))}

        pipe = self.get_pipeline(**init_components)

        assert hasattr(pipe, "components")
        assert set(pipe.components.keys()) == set(init_components.keys())

    @pytest.mark.skipif(torch_device not in ["cuda", "xpu"], reason="half-precision inference requires CUDA or XPU")
    @require_accelerator
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=str)
    def test_half_precision_inference_no_nan(self, dtype):
        # Models are usually run in half precision (fp16/bf16), so rather than comparing against an fp32 reference
        # (which carries little signal) we just run half-precision inference and check the output has no NaNs.
        pipe = self.get_pipeline().to(torch_device, dtype)

        inputs = self.get_dummy_inputs()
        if "generator" in inputs:
            inputs["generator"] = self.get_generator(0)
        output = pipe(**inputs)[0]

        assert not torch.isnan(output).any(), (
            f"`{self.pipeline_class.__name__}` produced NaNs during {dtype} inference."
        )

    @pytest.mark.skipif(torch_device not in ["cuda", "xpu"], reason="float16 requires CUDA or XPU")
    @require_accelerator
    def test_save_load_float16(self, tmp_path, expected_max_diff=1e-2):
        components = self.get_dummy_components()
        for name, module in components.items():
            # Account for components with _keep_in_fp32_modules
            if hasattr(module, "_keep_in_fp32_modules") and module._keep_in_fp32_modules is not None:
                for name, param in module.named_parameters():
                    if any(
                        module_to_keep_in_fp32 in name.split(".")
                        for module_to_keep_in_fp32 in module._keep_in_fp32_modules
                    ):
                        param.data = param.data.to(torch_device).to(torch.float32)
                    else:
                        param.data = param.data.to(torch_device).to(torch.float16)
                for name, buf in module.named_buffers():
                    if not buf.is_floating_point():
                        buf.data = buf.data.to(torch_device)
                    elif any(
                        module_to_keep_in_fp32 in name.split(".")
                        for module_to_keep_in_fp32 in module._keep_in_fp32_modules
                    ):
                        buf.data = buf.data.to(torch_device).to(torch.float32)
                    else:
                        buf.data = buf.data.to(torch_device).to(torch.float16)

            elif hasattr(module, "half"):
                components[name] = module.to(torch_device).half()

        pipe = self.get_pipeline(**components).to(torch_device)

        inputs = self.get_dummy_inputs()
        output = pipe(**inputs)[0]

        pipe.save_pretrained(tmp_path)
        pipe_loaded = self.pipeline_class.from_pretrained(tmp_path, torch_dtype=torch.float16)
        pipe_loaded.to(torch_device)
        pipe_loaded.set_progress_bar_config(disable=None)

        for name, component in pipe_loaded.components.items():
            if hasattr(component, "dtype"):
                assert component.dtype == torch.float16, (
                    f"`{name}.dtype` switched from `float16` to {component.dtype} after loading."
                )

        inputs = self.get_dummy_inputs()
        output_loaded = pipe_loaded(**inputs)[0]
        assert_tensors_close(
            output_loaded,
            output,
            atol=expected_max_diff,
            msg="The output of the fp16 pipeline changed after save/load.",
        )

    def test_save_load_optional_components(self, tmp_path, expected_max_difference=1e-4):
        if not getattr(self.pipeline_class, "_optional_components", None):
            pytest.skip(f"Skipping test because {self.pipeline_class} has no `_optional_components`.")

        pipe = self.get_pipeline().to(torch_device)

        # set all optional components to None
        for optional_component in pipe._optional_components:
            setattr(pipe, optional_component, None)

        inputs = self.get_dummy_inputs()
        torch.manual_seed(0)
        output = pipe(**inputs)[0]

        pipe.save_pretrained(tmp_path, safe_serialization=False)
        pipe_loaded = self.pipeline_class.from_pretrained(tmp_path)
        pipe_loaded.to(torch_device)
        pipe_loaded.set_progress_bar_config(disable=None)

        for optional_component in pipe._optional_components:
            assert getattr(pipe_loaded, optional_component) is None, (
                f"`{optional_component}` did not stay set to None after loading."
            )

        inputs = self.get_dummy_inputs()
        torch.manual_seed(0)
        output_loaded = pipe_loaded(**inputs)[0]

        assert_tensors_close(
            output_loaded,
            output,
            atol=expected_max_difference,
            msg="Output changed after dropping optional components.",
        )

    @require_accelerator
    def test_to_device(self):
        components = self.get_dummy_components()
        pipe = self.get_pipeline(**components)

        pipe.to("cpu")
        model_devices = [
            component.device.type for component in components.values() if getattr(component, "device", None)
        ]
        assert all(device == "cpu" for device in model_devices)

        output_cpu = pipe(**self.get_dummy_inputs())[0]
        assert torch.isnan(output_cpu).sum() == 0

        pipe.to(torch_device)
        model_devices = [
            component.device.type for component in components.values() if getattr(component, "device", None)
        ]
        assert all(device == torch_device for device in model_devices)

        output_device = pipe(**self.get_dummy_inputs())[0]
        assert torch.isnan(output_device).sum() == 0

    def test_to_dtype(self):
        components = self.get_dummy_components()
        pipe = self.get_pipeline(**components)

        model_dtypes = [component.dtype for component in components.values() if getattr(component, "dtype", None)]
        assert all(dtype == torch.float32 for dtype in model_dtypes)

        pipe.to(dtype=torch.float16)
        model_dtypes = [component.dtype for component in components.values() if getattr(component, "dtype", None)]
        assert all(dtype == torch.float16 for dtype in model_dtypes)

    def test_num_images_per_prompt(self):
        sig = inspect.signature(self.pipeline_class.__call__)

        if "num_images_per_prompt" not in sig.parameters:
            pytest.skip(
                f"Skipping test because `num_images_per_prompt` wasn't found in the args accepted in {self.pipeline_class}'s call."
            )

        pipe = self.get_pipeline().to(torch_device)

        batch_sizes = [1, 2]
        num_images_per_prompts = [1, 2]

        for batch_size in batch_sizes:
            for num_images_per_prompt in num_images_per_prompts:
                inputs = self.get_dummy_inputs()

                for key in inputs.keys():
                    if key in self.batch_input_params:
                        inputs[key] = batch_size * [inputs[key]]

                images = pipe(**inputs, num_images_per_prompt=num_images_per_prompt)[0]

                assert images.shape[0] == batch_size * num_images_per_prompt

    def test_cfg(self):
        sig = inspect.signature(self.pipeline_class.__call__)

        if "guidance_scale" not in sig.parameters:
            pytest.skip(
                f"Skipping test because `guidance_scale` wasn't found in the args accepted in {self.pipeline_class}'s call."
            )

        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()

        inputs["guidance_scale"] = 1.0
        out_no_cfg = pipe(**inputs)[0]

        inputs["guidance_scale"] = 7.5
        out_cfg = pipe(**inputs)[0]

        assert out_cfg.shape == out_no_cfg.shape

    def test_callback_inputs(self):
        sig = inspect.signature(self.pipeline_class.__call__)
        has_callback_tensor_inputs = "callback_on_step_end_tensor_inputs" in sig.parameters
        has_callback_step_end = "callback_on_step_end" in sig.parameters

        if not (has_callback_tensor_inputs and has_callback_step_end):
            pytest.skip(
                f"Skipping test because `callback_on_step_end` and `callback_on_step_end_tensor_inputs` weren't both found in the args accepted in {self.pipeline_class}'s call."
            )

        pipe = self.get_pipeline().to(torch_device)
        assert hasattr(pipe, "_callback_tensor_inputs"), (
            f"{self.pipeline_class} should have `_callback_tensor_inputs` that defines a list of tensor variables "
            "its callback function can use as inputs"
        )

        def callback_inputs_subset(pipe, i, t, callback_kwargs):
            # iterate over callback args
            for tensor_name, tensor_value in callback_kwargs.items():
                # check that we're only passing in allowed tensor inputs
                assert tensor_name in pipe._callback_tensor_inputs
            return callback_kwargs

        def callback_inputs_all(pipe, i, t, callback_kwargs):
            for tensor_name in pipe._callback_tensor_inputs:
                assert tensor_name in callback_kwargs

            # iterate over callback args
            for tensor_name, tensor_value in callback_kwargs.items():
                # check that we're only passing in allowed tensor inputs
                assert tensor_name in pipe._callback_tensor_inputs
            return callback_kwargs

        inputs = self.get_dummy_inputs()

        # Test passing in a subset
        inputs["callback_on_step_end"] = callback_inputs_subset
        inputs["callback_on_step_end_tensor_inputs"] = ["latents"]
        inputs["output_type"] = "latent"
        output = pipe(**inputs)[0]

        # Test passing in everything
        inputs["callback_on_step_end"] = callback_inputs_all
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        inputs["output_type"] = "latent"
        output = pipe(**inputs)[0]

        def callback_inputs_change_tensor(pipe, i, t, callback_kwargs):
            is_last = i == (pipe.num_timesteps - 1)
            if is_last:
                callback_kwargs["latents"] = torch.zeros_like(callback_kwargs["latents"])
            return callback_kwargs

        inputs["callback_on_step_end"] = callback_inputs_change_tensor
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        inputs["output_type"] = "latent"
        output = pipe(**inputs)[0]
        assert output.abs().sum() == 0

    def test_callback_cfg(self):
        sig = inspect.signature(self.pipeline_class.__call__)
        has_callback_tensor_inputs = "callback_on_step_end_tensor_inputs" in sig.parameters
        has_callback_step_end = "callback_on_step_end" in sig.parameters

        if not (has_callback_tensor_inputs and has_callback_step_end):
            pytest.skip(
                f"Skipping test because `callback_on_step_end` and `callback_on_step_end_tensor_inputs` weren't both found in the args accepted in {self.pipeline_class}'s call."
            )

        if "guidance_scale" not in sig.parameters:
            pytest.skip(
                f"Skipping test because `guidance_scale` wasn't found in the args accepted in {self.pipeline_class}'s call."
            )

        pipe = self.get_pipeline().to(torch_device)
        assert hasattr(pipe, "_callback_tensor_inputs"), (
            f"{self.pipeline_class} should have `_callback_tensor_inputs` that defines a list of tensor variables "
            "its callback function can use as inputs"
        )

        def callback_increase_guidance(pipe, i, t, callback_kwargs):
            pipe._guidance_scale += 1.0
            return callback_kwargs

        inputs = self.get_dummy_inputs()

        # use cfg guidance because some pipelines modify the shape of the latents outside of the denoising loop
        inputs["guidance_scale"] = 2.0
        inputs["callback_on_step_end"] = callback_increase_guidance
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        _ = pipe(**inputs)[0]

        # we increase the guidance scale by 1.0 at every step
        # check that the guidance scale is increased by the number of scheduler timesteps
        # accounts for models that modify the number of inference steps based on strength
        assert pipe.guidance_scale == (inputs["guidance_scale"] + pipe.num_timesteps)

    def test_serialization_with_variants(self, tmp_path):
        pipe = self.get_pipeline()
        model_components = [
            component_name for component_name, component in pipe.components.items() if isinstance(component, nn.Module)
        ]
        variant = "fp16"

        pipe.save_pretrained(tmp_path, variant=variant, safe_serialization=False)

        with open(f"{tmp_path}/model_index.json", "r") as f:
            config = json.load(f)

        for subfolder in os.listdir(tmp_path):
            if not os.path.isfile(subfolder) and subfolder in model_components:
                folder_path = os.path.join(tmp_path, subfolder)
                is_folder = os.path.isdir(folder_path) and subfolder in config
                assert is_folder and any(p.split(".")[1].startswith(variant) for p in os.listdir(folder_path))

    def test_loading_with_variants(self, tmp_path):
        pipe = self.get_pipeline()
        variant = "fp16"

        def is_nan(tensor):
            if tensor.ndimension() == 0:
                has_nan = torch.isnan(tensor).item()
            else:
                has_nan = torch.isnan(tensor).any()
            return has_nan

        pipe.save_pretrained(tmp_path, variant=variant, safe_serialization=False)
        pipe_loaded = self.pipeline_class.from_pretrained(tmp_path, variant=variant)

        model_components_pipe = {
            component_name: component
            for component_name, component in pipe.components.items()
            if isinstance(component, nn.Module)
        }
        model_components_pipe_loaded = {
            component_name: component
            for component_name, component in pipe_loaded.components.items()
            if isinstance(component, nn.Module)
        }
        for component_name in model_components_pipe:
            pipe_component = model_components_pipe[component_name]
            pipe_loaded_component = model_components_pipe_loaded[component_name]

            model_loaded_params = dict(pipe_loaded_component.named_parameters())
            model_original_params = dict(pipe_component.named_parameters())

            for name, p1 in model_original_params.items():
                # Skip tied weights that aren't saved with variants (transformers v5 behavior)
                if name not in model_loaded_params:
                    continue

                p2 = model_loaded_params[name]
                # nan check for luminanext (mps).
                if not (is_nan(p1) and is_nan(p2)):
                    assert torch.equal(p1, p2)

    def test_loading_with_incorrect_variants_raises_error(self, tmp_path):
        pipe = self.get_pipeline()
        variant = "fp16"

        # Don't save with variants.
        pipe.save_pretrained(tmp_path, safe_serialization=False)

        with pytest.raises(ValueError) as error:
            _ = self.pipeline_class.from_pretrained(tmp_path, variant=variant)

        assert f"You are trying to load the model files of the `variant={variant}`" in str(error.value)

    def test_encode_prompt_works_in_isolation(self, extra_required_param_value_dict=None, atol=1e-4, rtol=1e-4):
        if not hasattr(self.pipeline_class, "encode_prompt"):
            pytest.skip(f"Skipping test because {self.pipeline_class} doesn't have an `encode_prompt` method.")

        components = self.get_dummy_components()

        # We initialize the pipeline with only text encoders and tokenizers, mimicking a real-world scenario.
        components_with_text_encoders = {}
        for k in components:
            if "text" in k or "tokenizer" in k:
                components_with_text_encoders[k] = components[k]
            else:
                components_with_text_encoders[k] = None
        pipe_with_just_text_encoder = self.get_pipeline(**components_with_text_encoders).to(torch_device)

        # Get inputs and also the args of `encode_prompts`.
        inputs = self.get_dummy_inputs()
        encode_prompt_signature = inspect.signature(pipe_with_just_text_encoder.encode_prompt)
        encode_prompt_parameters = list(encode_prompt_signature.parameters.values())

        # Required args in encode_prompt with those with no default.
        required_params = []
        for param in encode_prompt_parameters:
            if param.name == "self" or param.name == "kwargs":
                continue
            if param.default is inspect.Parameter.empty:
                required_params.append(param.name)

        # Craft inputs for the `encode_prompt()` method to run in isolation.
        encode_prompt_param_names = [p.name for p in encode_prompt_parameters if p.name != "self"]
        input_keys = list(inputs.keys())
        encode_prompt_inputs = {k: inputs.pop(k) for k in input_keys if k in encode_prompt_param_names}

        pipe_call_signature = inspect.signature(pipe_with_just_text_encoder.__call__)
        pipe_call_parameters = pipe_call_signature.parameters

        # For each required arg in encode_prompt, check if it's missing in encode_prompt_inputs. If so, see if
        # __call__ has a default for that arg and use it if available.
        for required_param_name in required_params:
            if required_param_name not in encode_prompt_inputs:
                pipe_call_param = pipe_call_parameters.get(required_param_name, None)
                if pipe_call_param is not None and pipe_call_param.default is not inspect.Parameter.empty:
                    # Use the default from pipe.__call__
                    encode_prompt_inputs[required_param_name] = pipe_call_param.default
                elif extra_required_param_value_dict is not None and isinstance(extra_required_param_value_dict, dict):
                    encode_prompt_inputs[required_param_name] = extra_required_param_value_dict[required_param_name]
                else:
                    raise ValueError(
                        f"Required parameter '{required_param_name}' in "
                        f"encode_prompt has no default in either encode_prompt or __call__."
                    )

        # Compute `encode_prompt()`.
        with torch.no_grad():
            encoded_prompt_outputs = pipe_with_just_text_encoder.encode_prompt(**encode_prompt_inputs)

        # Programmatically determine the return names of `encode_prompt.`
        ast_visitor = ReturnNameVisitor()
        encode_prompt_tree = ast_visitor.get_ast_tree(cls=self.pipeline_class)
        ast_visitor.visit(encode_prompt_tree)
        prompt_embed_kwargs = ast_visitor.return_names
        prompt_embeds_kwargs = dict(zip(prompt_embed_kwargs, encoded_prompt_outputs))

        # Pack the outputs of `encode_prompt`.
        adapted_prompt_embeds_kwargs = {
            k: prompt_embeds_kwargs.pop(k) for k in list(prompt_embeds_kwargs.keys()) if k in pipe_call_parameters
        }

        # now initialize a pipeline without text encoders and compute outputs with the `encode_prompt()` outputs
        # and other relevant inputs.
        components_with_text_encoders = {}
        for k in components:
            if "text" in k or "tokenizer" in k:
                components_with_text_encoders[k] = None
            else:
                components_with_text_encoders[k] = components[k]
        pipe_without_text_encoders = self.get_pipeline(**components_with_text_encoders).to(torch_device)

        # Set `negative_prompt` to None as we have already calculated its embeds if it was present in `inputs`.
        # This is because otherwise we will interfere wrongly for non-None `negative_prompt` values as defaults
        # (PixArt for example).
        pipe_without_tes_inputs = {**inputs, **adapted_prompt_embeds_kwargs}
        if (
            pipe_call_parameters.get("negative_prompt", None) is not None
            and pipe_call_parameters.get("negative_prompt").default is not None
        ):
            pipe_without_tes_inputs.update({"negative_prompt": None})

        # Pipelines like attend and excite have `prompt` as a required argument.
        if (
            pipe_call_parameters.get("prompt", None) is not None
            and pipe_call_parameters.get("prompt").default is inspect.Parameter.empty
            and pipe_call_parameters.get("prompt_embeds", None) is not None
            and pipe_call_parameters.get("prompt_embeds").default is None
        ):
            pipe_without_tes_inputs.update({"prompt": None})

        pipe_out = pipe_without_text_encoders(**pipe_without_tes_inputs)[0]

        # Compare against regular pipeline outputs.
        full_pipe = self.get_pipeline(**components).to(torch_device)
        inputs = self.get_dummy_inputs()
        pipe_out_2 = full_pipe(**inputs)[0]

        assert_tensors_close(
            pipe_out, pipe_out_2, atol=atol, rtol=rtol, msg="`encode_prompt` in isolation changed the output."
        )

    def test_torch_dtype_dict(self, tmp_path):
        components = self.get_dummy_components()
        if not components:
            pytest.skip("No dummy components defined.")

        pipe = self.get_pipeline(**components)
        specified_key = next(iter(components.keys()))

        pipe.save_pretrained(str(tmp_path), safe_serialization=False)
        torch_dtype_dict = {specified_key: torch.bfloat16, "default": torch.float16}
        loaded_pipe = self.pipeline_class.from_pretrained(str(tmp_path), torch_dtype=torch_dtype_dict)

        for name, component in loaded_pipe.components.items():
            if isinstance(component, torch.nn.Module) and hasattr(component, "dtype"):
                expected_dtype = torch_dtype_dict.get(name, torch_dtype_dict.get("default", torch.float32))
                assert component.dtype == expected_dtype, (
                    f"Component '{name}' has dtype {component.dtype} but expected {expected_dtype}"
                )
