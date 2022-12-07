import contextlib
import gc
import inspect
import io
import re
import tempfile
import time
import unittest
from typing import Callable, Union

import numpy as np
import torch

from diffusers import CycleDiffusionPipeline, DanceDiffusionPipeline, DiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers.utils.import_utils import is_accelerate_available, is_xformers_available
from diffusers.utils.testing_utils import require_torch, torch_device


torch.backends.cuda.matmul.allow_tf32 = False


@require_torch
class PipelineTesterMixin:
    """
    This mixin is designed to be used with unittest.TestCase classes.
    It provides a set of common tests for each PyTorch pipeline, e.g. saving and loading the pipeline,
    equivalence of dict and tuple outputs, etc.
    """

    # set these parameters to False in the child class if the pipeline does not support the corresponding functionality
    test_attention_slicing = True
    test_cpu_offload = True
    test_xformers_attention = True

    @property
    def pipeline_class(self) -> Union[Callable, DiffusionPipeline]:
        raise NotImplementedError(
            "You need to set the attribute `pipeline_class = ClassNameOfPipeline` in the child test class. "
            "See existing pipeline tests for reference."
        )

    def get_dummy_components(self):
        raise NotImplementedError(
            "You need to implement `get_dummy_components(self)` in the child test class. "
            "See existing pipeline tests for reference."
        )

    def get_dummy_inputs(self, device, seed=0):
        raise NotImplementedError(
            "You need to implement `get_dummy_inputs(self, device, seed)` in the child test class. "
            "See existing pipeline tests for reference."
        )

    def tearDown(self):
        # clean up the VRAM after each test in case of CUDA runtime errors
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_save_load_local(self):
        if torch_device == "mps" and self.pipeline_class in (
            DanceDiffusionPipeline,
            CycleDiffusionPipeline,
            StableDiffusionImg2ImgPipeline,
        ):
            # FIXME: inconsistent outputs on MPS
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # Warmup pass when using mps (see #372)
        if torch_device == "mps":
            _ = pipe(**self.get_dummy_inputs(torch_device))

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output_loaded = pipe_loaded(**inputs)[0]

        max_diff = np.abs(output - output_loaded).max()
        self.assertLess(max_diff, 1e-5)

    def test_dict_tuple_outputs_equivalent(self):
        if torch_device == "mps" and self.pipeline_class in (
            DanceDiffusionPipeline,
            CycleDiffusionPipeline,
            StableDiffusionImg2ImgPipeline,
        ):
            # FIXME: inconsistent outputs on MPS
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # Warmup pass when using mps (see #372)
        if torch_device == "mps":
            _ = pipe(**self.get_dummy_inputs(torch_device))

        output = pipe(**self.get_dummy_inputs(torch_device))[0]
        output_tuple = pipe(**self.get_dummy_inputs(torch_device), return_dict=False)[0]

        max_diff = np.abs(output - output_tuple).max()
        self.assertLess(max_diff, 1e-5)

    def test_pipeline_call_implements_required_args(self):
        required_args = ["num_inference_steps", "generator", "return_dict"]

        for arg in required_args:
            self.assertTrue(arg in inspect.signature(self.pipeline_class.__call__).parameters)

    def test_num_inference_steps_consistent(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # Warmup pass when using mps (see #372)
        if torch_device == "mps":
            _ = pipe(**self.get_dummy_inputs(torch_device))

        outputs = []
        times = []
        for num_steps in [3, 6, 9]:
            inputs = self.get_dummy_inputs(torch_device)
            inputs["num_inference_steps"] = num_steps

            start_time = time.time()
            output = pipe(**inputs)[0]
            inference_time = time.time() - start_time

            outputs.append(output)
            times.append(inference_time)

        # check that all outputs have the same shape
        self.assertTrue(all(outputs[0].shape == output.shape for output in outputs))
        # check that the inference time increases with the number of inference steps
        self.assertTrue(all(times[i] > times[i - 1] for i in range(1, len(times))))

    def test_components_function(self):
        init_components = self.get_dummy_components()
        pipe = self.pipeline_class(**init_components)

        self.assertTrue(hasattr(pipe, "components"))
        self.assertTrue(set(pipe.components.keys()) == set(init_components.keys()))

    @unittest.skipIf(torch_device != "cuda", reason="float16 requires CUDA")
    def test_float16_inference(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        for name, module in components.items():
            if hasattr(module, "half"):
                components[name] = module.half()
        pipe_fp16 = self.pipeline_class(**components)
        pipe_fp16.to(torch_device)
        pipe_fp16.set_progress_bar_config(disable=None)

        output = pipe(**self.get_dummy_inputs(torch_device))[0]
        output_fp16 = pipe_fp16(**self.get_dummy_inputs(torch_device))[0]

        max_diff = np.abs(output - output_fp16).max()
        self.assertLess(max_diff, 1e-2, "The outputs of the fp16 and fp32 pipelines are too different.")

    @unittest.skipIf(torch_device != "cuda", reason="float16 requires CUDA")
    def test_save_load_float16(self):
        components = self.get_dummy_components()
        for name, module in components.items():
            if hasattr(module, "half"):
                components[name] = module.to(torch_device).half()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        for name, component in pipe_loaded.components.items():
            if hasattr(component, "dtype"):
                self.assertTrue(
                    component.dtype == torch.float16,
                    f"`{name}.dtype` switched from `float16` to {component.dtype} after loading.",
                )

        inputs = self.get_dummy_inputs(torch_device)
        output_loaded = pipe_loaded(**inputs)[0]

        max_diff = np.abs(output - output_loaded).max()
        self.assertLess(max_diff, 3e-3, "The output of the fp16 pipeline changed after saving and loading.")

    def test_save_load_optional_components(self):
        if not hasattr(self.pipeline_class, "_optional_components"):
            return

        if torch_device == "mps" and self.pipeline_class in (
            DanceDiffusionPipeline,
            CycleDiffusionPipeline,
            StableDiffusionImg2ImgPipeline,
        ):
            # FIXME: inconsistent outputs on MPS
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # Warmup pass when using mps (see #372)
        if torch_device == "mps":
            _ = pipe(**self.get_dummy_inputs(torch_device))

        # set all optional components to None
        for optional_component in pipe._optional_components:
            setattr(pipe, optional_component, None)

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        for optional_component in pipe._optional_components:
            self.assertTrue(
                getattr(pipe_loaded, optional_component) is None,
                f"`{optional_component}` did not stay set to None after loading.",
            )

        inputs = self.get_dummy_inputs(torch_device)
        output_loaded = pipe_loaded(**inputs)[0]

        max_diff = np.abs(output - output_loaded).max()
        self.assertLess(max_diff, 1e-5)

    @unittest.skipIf(torch_device != "cuda", reason="CUDA and CPU are required to switch devices")
    def test_to_device(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        pipe.to("cpu")
        model_devices = [component.device.type for component in components.values() if hasattr(component, "device")]
        self.assertTrue(all(device == "cpu" for device in model_devices))

        output_cpu = pipe(**self.get_dummy_inputs("cpu"))[0]
        self.assertTrue(np.isnan(output_cpu).sum() == 0)

        pipe.to("cuda")
        model_devices = [component.device.type for component in components.values() if hasattr(component, "device")]
        self.assertTrue(all(device == "cuda" for device in model_devices))

        output_cuda = pipe(**self.get_dummy_inputs("cuda"))[0]
        self.assertTrue(np.isnan(output_cuda).sum() == 0)

    def test_attention_slicing_forward_pass(self):
        if not self.test_attention_slicing:
            return

        if torch_device == "mps" and self.pipeline_class in (
            DanceDiffusionPipeline,
            CycleDiffusionPipeline,
            StableDiffusionImg2ImgPipeline,
        ):
            # FIXME: inconsistent outputs on MPS
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # Warmup pass when using mps (see #372)
        if torch_device == "mps":
            _ = pipe(**self.get_dummy_inputs(torch_device))

        inputs = self.get_dummy_inputs(torch_device)
        output_without_slicing = pipe(**inputs)[0]

        pipe.enable_attention_slicing(slice_size=1)
        inputs = self.get_dummy_inputs(torch_device)
        output_with_slicing = pipe(**inputs)[0]

        max_diff = np.abs(output_with_slicing - output_without_slicing).max()
        self.assertLess(max_diff, 1e-5, "Attention slicing should not affect the inference results")

    @unittest.skipIf(
        torch_device != "cuda" or not is_accelerate_available(),
        reason="CPU offload is only available with CUDA and `accelerate` installed",
    )
    def test_cpu_offload_forward_pass(self):
        if not self.test_cpu_offload:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output_without_offload = pipe(**inputs)[0]

        pipe.enable_sequential_cpu_offload()
        inputs = self.get_dummy_inputs(torch_device)
        output_with_offload = pipe(**inputs)[0]

        max_diff = np.abs(output_with_offload - output_without_offload).max()
        self.assertLess(max_diff, 1e-5, "CPU offloading should not affect the inference results")

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_xformers_attention_forward_pass(self):
        if not self.test_xformers_attention:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output_without_offload = pipe(**inputs)[0]

        pipe.enable_xformers_memory_efficient_attention()
        inputs = self.get_dummy_inputs(torch_device)
        output_with_offload = pipe(**inputs)[0]

        max_diff = np.abs(output_with_offload - output_without_offload).max()
        self.assertLess(max_diff, 1e-5, "XFormers attention should not affect the inference results")

    def test_progress_bar(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)

        inputs = self.get_dummy_inputs(torch_device)
        with io.StringIO() as stderr, contextlib.redirect_stderr(stderr):
            _ = pipe(**inputs)
            stderr = stderr.getvalue()
            # we can't calculate the number of progress steps beforehand e.g. for strength-dependent img2img,
            # so we just match "5" in "#####| 1/5 [00:01<00:00]"
            max_steps = re.search("/(.*?) ", stderr).group(1)
            self.assertTrue(max_steps is not None and len(max_steps) > 0)
            self.assertTrue(
                f"{max_steps}/{max_steps}" in stderr, "Progress bar should be enabled and stopped at the max step"
            )

        pipe.set_progress_bar_config(disable=True)
        with io.StringIO() as stderr, contextlib.redirect_stderr(stderr):
            _ = pipe(**inputs)
            self.assertTrue(stderr.getvalue() == "", "Progress bar should be disabled")
