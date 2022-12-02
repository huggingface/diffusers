import tempfile
import unittest

import numpy as np
import torch

from diffusers.utils.testing_utils import require_torch, torch_device


torch.backends.cuda.matmul.allow_tf32 = False


@require_torch
class PipelineTesterMixin:
    """
    This mixin is designed to be used with unittest.TestCase classes.
    It provides a set of common tests for each PyTorch pipeline, e.g. saving and loading the pipeline,
    equivalence of dict and tuple outputs, etc.
    """

    def test_save_load(self):
        device = "cpu"
        pipe = self.pipeline_class(**self.get_common_pipeline_components())
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        output = pipe(**self.get_common_inputs(device))[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
            pipe_loaded.to(device)
            pipe_loaded.set_progress_bar_config(disable=None)

        output_loaded = pipe_loaded(**self.get_common_inputs(device))[0]

        max_diff = np.abs(output - output_loaded).max()
        self.assertLessEqual(max_diff, 1e-5)

    def test_tuple_output(self):
        device = "cpu"
        pipe = self.pipeline_class(**self.get_common_pipeline_components())
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        output = pipe(**self.get_common_inputs(device))[0]
        output_tuple = pipe(**self.get_common_inputs(device), return_dict=False)[0]

        max_diff = np.abs(output - output_tuple).max()
        self.assertLessEqual(max_diff, 1e-5)

    @unittest.skipIf(torch_device != "cuda", reason="float16 requires CUDA")
    def test_float16(self):
        device = "cuda"
        components = self.get_common_pipeline_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        for name, module in components.items():
            if hasattr(module, "half"):
                components[name] = module.half()
        pipe_fp16 = self.pipeline_class(**components)
        pipe_fp16.to(device)
        pipe_fp16.set_progress_bar_config(disable=None)

        output = pipe(**self.get_common_inputs(device))[0]
        output_fp16 = pipe_fp16(**self.get_common_inputs(device))[0]

        max_diff = np.abs(output - output_fp16).max()
        # the outputs can be different, but not too much
        self.assertLessEqual(max_diff, 1e-2)

    @unittest.skipIf(torch_device != "cuda", reason="CUDA and CPU are required to switch devices")
    def test_to_device(self):
        components = self.get_common_pipeline_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        pipe.to("cpu")
        model_devices = [component.device.type for component in components.values() if hasattr(component, "device")]
        self.assertTrue(all(device == "cpu" for device in model_devices))

        output_cpu = pipe(**self.get_common_inputs("cpu"))[0]
        self.assertTrue(np.isnan(output_cpu).sum() == 0)

        pipe.to("cuda")
        model_devices = [component.device.type for component in components.values() if hasattr(component, "device")]
        self.assertTrue(all(device == "cuda" for device in model_devices))

        output_cuda = pipe(**self.get_common_inputs("cuda"))[0]
        self.assertTrue(np.isnan(output_cuda).sum() == 0)
