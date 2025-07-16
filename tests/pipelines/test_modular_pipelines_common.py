import gc
import unittest
from typing import Any, Callable, Dict, Union

import numpy as np
import torch

import diffusers
from diffusers import (
    ClassifierFreeGuidance,
    DiffusionPipeline,
)
from diffusers.loaders import ModularIPAdapterMixin
from diffusers.utils import logging
from diffusers.utils.testing_utils import (
    backend_empty_cache,
    numpy_cosine_similarity_distance,
    require_accelerator,
    require_torch,
    torch_device,
)

from ..models.unets.test_models_unet_2d_condition import (
    create_ip_adapter_faceid_state_dict,
    create_ip_adapter_state_dict,
)


def to_np(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    return tensor


class ModularIPAdapterTesterMixin:
    """
    This mixin is designed to be used with PipelineTesterMixin and unittest.TestCase classes.
    It provides a set of common tests for pipelines that support IP Adapters.
    """

    def test_pipeline_inputs_and_blocks(self):
        blocks = self.pipeline_blocks_class()
        parameters = blocks.input_names

        assert issubclass(self.pipeline_class, ModularIPAdapterMixin)
        self.assertIn(
            "ip_adapter_image",
            parameters,
            "`ip_adapter_image` argument must be supported by the `__call__` method",
        )
        self.assertIn(
            "ip_adapter",
            blocks.sub_blocks,
            "pipeline must contain an IPAdapter block",
        )

        _ = blocks.sub_blocks.pop("ip_adapter")
        parameters = blocks.input_names
        intermediate_parameters = blocks.intermediate_input_names
        self.assertNotIn(
            "ip_adapter_image",
            parameters,
            "`ip_adapter_image` argument must be removed from the `__call__` method",
        )
        self.assertNotIn(
            "ip_adapter_image_embeds",
            intermediate_parameters,
            "`ip_adapter_image_embeds` argument must be supported by the `__call__` method",
        )

    def _get_dummy_image_embeds(self, cross_attention_dim: int = 32):
        return torch.randn((1, 1, cross_attention_dim), device=torch_device)

    def _get_dummy_faceid_image_embeds(self, cross_attention_dim: int = 32):
        return torch.randn((1, 1, 1, cross_attention_dim), device=torch_device)

    def _get_dummy_masks(self, input_size: int = 64):
        _masks = torch.zeros((1, 1, input_size, input_size), device=torch_device)
        _masks[0, :, :, : int(input_size / 2)] = 1
        return _masks

    def _modify_inputs_for_ip_adapter_test(self, inputs: Dict[str, Any]):
        blocks = self.pipeline_blocks_class()
        _ = blocks.sub_blocks.pop("ip_adapter")
        parameters = blocks.input_names
        if "image" in parameters and "strength" in parameters:
            inputs["num_inference_steps"] = 4

        inputs["output_type"] = "np"
        return inputs

    def test_ip_adapter(self, expected_max_diff: float = 1e-4, expected_pipe_slice=None):
        r"""Tests for IP-Adapter.

        The following scenarios are tested:
          - Single IP-Adapter with scale=0 should produce same output as no IP-Adapter.
          - Multi IP-Adapter with scale=0 should produce same output as no IP-Adapter.
          - Single IP-Adapter with scale!=0 should produce different output compared to no IP-Adapter.
          - Multi IP-Adapter with scale!=0 should produce different output compared to no IP-Adapter.
        """
        # Raising the tolerance for this test when it's run on a CPU because we
        # compare against static slices and that can be shaky (with a VVVV low probability).
        expected_max_diff = 9e-4 if torch_device == "cpu" else expected_max_diff

        blocks = self.pipeline_blocks_class()
        _ = blocks.sub_blocks.pop("ip_adapter")
        pipe = blocks.init_pipeline(self.repo)
        pipe.load_default_components(torch_dtype=torch.float32)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        cross_attention_dim = pipe.unet.config.get("cross_attention_dim")

        # forward pass without ip adapter
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        if expected_pipe_slice is None:
            output_without_adapter = pipe(**inputs, output="images")
        else:
            output_without_adapter = expected_pipe_slice

        # 1. Single IP-Adapter test cases
        adapter_state_dict = create_ip_adapter_state_dict(pipe.unet)
        pipe.unet._load_ip_adapter_weights(adapter_state_dict)

        # forward pass with single ip adapter, but scale=0 which should have no effect
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        inputs["ip_adapter_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)]
        inputs["negative_ip_adapter_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)]
        pipe.set_ip_adapter_scale(0.0)
        output_without_adapter_scale = pipe(**inputs, output="images")
        if expected_pipe_slice is not None:
            output_without_adapter_scale = output_without_adapter_scale[0, -3:, -3:, -1].flatten()

        # forward pass with single ip adapter, but with scale of adapter weights
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        inputs["ip_adapter_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)]
        inputs["negative_ip_adapter_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)]
        pipe.set_ip_adapter_scale(42.0)
        output_with_adapter_scale = pipe(**inputs, output="images")
        if expected_pipe_slice is not None:
            output_with_adapter_scale = output_with_adapter_scale[0, -3:, -3:, -1].flatten()

        max_diff_without_adapter_scale = np.abs(output_without_adapter_scale - output_without_adapter).max()
        max_diff_with_adapter_scale = np.abs(output_with_adapter_scale - output_without_adapter).max()

        self.assertLess(
            max_diff_without_adapter_scale,
            expected_max_diff,
            "Output without ip-adapter must be same as normal inference",
        )
        self.assertGreater(
            max_diff_with_adapter_scale, 1e-2, "Output with ip-adapter must be different from normal inference"
        )

        # 2. Multi IP-Adapter test cases
        adapter_state_dict_1 = create_ip_adapter_state_dict(pipe.unet)
        adapter_state_dict_2 = create_ip_adapter_state_dict(pipe.unet)
        pipe.unet._load_ip_adapter_weights([adapter_state_dict_1, adapter_state_dict_2])

        # forward pass with multi ip adapter, but scale=0 which should have no effect
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        inputs["ip_adapter_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)] * 2
        inputs["negative_ip_adapter_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)] * 2
        pipe.set_ip_adapter_scale([0.0, 0.0])
        output_without_multi_adapter_scale = pipe(**inputs, output="images")
        if expected_pipe_slice is not None:
            output_without_multi_adapter_scale = output_without_multi_adapter_scale[0, -3:, -3:, -1].flatten()

        # forward pass with multi ip adapter, but with scale of adapter weights
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        inputs["ip_adapter_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)] * 2
        inputs["negative_ip_adapter_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)] * 2
        pipe.set_ip_adapter_scale([42.0, 42.0])
        output_with_multi_adapter_scale = pipe(**inputs, output="images")
        if expected_pipe_slice is not None:
            output_with_multi_adapter_scale = output_with_multi_adapter_scale[0, -3:, -3:, -1].flatten()

        max_diff_without_multi_adapter_scale = np.abs(
            output_without_multi_adapter_scale - output_without_adapter
        ).max()
        max_diff_with_multi_adapter_scale = np.abs(output_with_multi_adapter_scale - output_without_adapter).max()
        self.assertLess(
            max_diff_without_multi_adapter_scale,
            expected_max_diff,
            "Output without multi-ip-adapter must be same as normal inference",
        )
        self.assertGreater(
            max_diff_with_multi_adapter_scale,
            1e-2,
            "Output with multi-ip-adapter scale must be different from normal inference",
        )

    def test_ip_adapter_cfg(self, expected_max_diff: float = 1e-4):
        blocks = self.pipeline_blocks_class()
        _ = blocks.sub_blocks.pop("ip_adapter")
        pipe = blocks.init_pipeline(self.repo)
        pipe.load_default_components(torch_dtype=torch.float32)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        cross_attention_dim = pipe.unet.config.get("cross_attention_dim", 32)

        adapter_state_dict = create_ip_adapter_state_dict(pipe.unet)
        pipe.unet._load_ip_adapter_weights(adapter_state_dict)
        pipe.set_ip_adapter_scale(1.0)

        # forward pass with CFG not applied
        guider = ClassifierFreeGuidance(guidance_scale=1.0)
        pipe.update_components(guider=guider)

        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        inputs["ip_adapter_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)[0].unsqueeze(0)]
        out_no_cfg = pipe(**inputs, output="images")

        # forward pass with CFG applied
        guider = ClassifierFreeGuidance(guidance_scale=7.5)
        pipe.update_components(guider=guider)
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        inputs["ip_adapter_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)]
        inputs["negative_ip_adapter_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)]
        out_cfg = pipe(**inputs, output="images")

        assert out_cfg.shape == out_no_cfg.shape

    def test_ip_adapter_masks(self, expected_max_diff: float = 1e-4):
        blocks = self.pipeline_blocks_class()
        _ = blocks.sub_blocks.pop("ip_adapter")
        pipe = blocks.init_pipeline(self.repo)
        pipe.load_default_components(torch_dtype=torch.float32)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        cross_attention_dim = pipe.unet.config.get("cross_attention_dim", 32)
        sample_size = pipe.unet.config.get("sample_size", 32)
        block_out_channels = pipe.vae.config.get("block_out_channels", [128, 256, 512, 512])
        input_size = sample_size * (2 ** (len(block_out_channels) - 1))

        # forward pass without ip adapter
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        output_without_adapter = pipe(**inputs, output="images")
        output_without_adapter = output_without_adapter[0, -3:, -3:, -1].flatten()

        adapter_state_dict = create_ip_adapter_state_dict(pipe.unet)
        pipe.unet._load_ip_adapter_weights(adapter_state_dict)

        # forward pass with single ip adapter and masks, but scale=0 which should have no effect
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        inputs["ip_adapter_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)]
        inputs["negative_ip_adapter_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)]
        inputs["cross_attention_kwargs"] = {"ip_adapter_masks": [self._get_dummy_masks(input_size)]}
        pipe.set_ip_adapter_scale(0.0)
        output_without_adapter_scale = pipe(**inputs, output="images")
        output_without_adapter_scale = output_without_adapter_scale[0, -3:, -3:, -1].flatten()

        # forward pass with single ip adapter and masks, but with scale of adapter weights
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        inputs["ip_adapter_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)]
        inputs["negative_ip_adapter_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)]
        inputs["cross_attention_kwargs"] = {"ip_adapter_masks": [self._get_dummy_masks(input_size)]}
        pipe.set_ip_adapter_scale(42.0)
        output_with_adapter_scale = pipe(**inputs, output="images")
        output_with_adapter_scale = output_with_adapter_scale[0, -3:, -3:, -1].flatten()

        max_diff_without_adapter_scale = np.abs(output_without_adapter_scale - output_without_adapter).max()
        max_diff_with_adapter_scale = np.abs(output_with_adapter_scale - output_without_adapter).max()

        self.assertLess(
            max_diff_without_adapter_scale,
            expected_max_diff,
            "Output without ip-adapter must be same as normal inference",
        )
        self.assertGreater(
            max_diff_with_adapter_scale, 1e-3, "Output with ip-adapter must be different from normal inference"
        )

    def test_ip_adapter_faceid(self, expected_max_diff: float = 1e-4):
        blocks = self.pipeline_blocks_class()
        _ = blocks.sub_blocks.pop("ip_adapter")
        pipe = blocks.init_pipeline(self.repo)
        pipe.load_default_components(torch_dtype=torch.float32)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        cross_attention_dim = pipe.unet.config.get("cross_attention_dim", 32)

        # forward pass without ip adapter
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        output_without_adapter = pipe(**inputs, output="images")
        output_without_adapter = output_without_adapter[0, -3:, -3:, -1].flatten()

        adapter_state_dict = create_ip_adapter_faceid_state_dict(pipe.unet)
        pipe.unet._load_ip_adapter_weights(adapter_state_dict)

        # forward pass with single ip adapter, but scale=0 which should have no effect
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        inputs["ip_adapter_embeds"] = [self._get_dummy_faceid_image_embeds(cross_attention_dim)]
        inputs["negative_ip_adapter_embeds"] = [self._get_dummy_faceid_image_embeds(cross_attention_dim)]
        pipe.set_ip_adapter_scale(0.0)
        output_without_adapter_scale = pipe(**inputs, output="images")
        output_without_adapter_scale = output_without_adapter_scale[0, -3:, -3:, -1].flatten()

        # forward pass with single ip adapter, but with scale of adapter weights
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        inputs["ip_adapter_embeds"] = [self._get_dummy_faceid_image_embeds(cross_attention_dim)]
        inputs["negative_ip_adapter_embeds"] = [self._get_dummy_faceid_image_embeds(cross_attention_dim)]
        pipe.set_ip_adapter_scale(42.0)
        output_with_adapter_scale = pipe(**inputs, output="images")
        output_with_adapter_scale = output_with_adapter_scale[0, -3:, -3:, -1].flatten()

        max_diff_without_adapter_scale = np.abs(output_without_adapter_scale - output_without_adapter).max()
        max_diff_with_adapter_scale = np.abs(output_with_adapter_scale - output_without_adapter).max()

        self.assertLess(
            max_diff_without_adapter_scale,
            expected_max_diff,
            "Output without ip-adapter must be same as normal inference",
        )
        self.assertGreater(
            max_diff_with_adapter_scale, 1e-3, "Output with ip-adapter must be different from normal inference"
        )


@require_torch
class ModularPipelineTesterMixin:
    """
    This mixin is designed to be used with unittest.TestCase classes.
    It provides a set of common tests for each PyTorch pipeline, e.g. saving and loading the pipeline,
    equivalence of dict and tuple outputs, etc.
    """

    # Canonical parameters that are passed to `__call__` regardless
    # of the type of pipeline. They are always optional and have common
    # sense default values.
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "num_images_per_prompt",
            "latents",
            "output_type",
        ]
    )
    # generator needs to be a intermediate input because it's mutable
    required_intermediate_params = frozenset(
        [
            "generator",
        ]
    )

    def get_generator(self, seed):
        device = torch_device if torch_device != "mps" else "cpu"
        generator = torch.Generator(device).manual_seed(seed)
        return generator

    @property
    def pipeline_class(self) -> Union[Callable, DiffusionPipeline]:
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
    def pipeline_blocks_class(self) -> Union[Callable, DiffusionPipeline]:
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
        parameters = pipe.blocks.input_names
        optional_parameters = pipe.default_call_parameters
        intermediate_parameters = pipe.blocks.intermediate_input_names

        remaining_required_parameters = set()

        for param in self.params:
            if param not in parameters:
                remaining_required_parameters.add(param)

        self.assertTrue(
            len(remaining_required_parameters) == 0,
            f"Required parameters not present: {remaining_required_parameters}",
        )

        remaining_required_intermediate_parameters = set()

        for param in self.required_intermediate_params:
            if param not in intermediate_parameters:
                remaining_required_intermediate_parameters.add(param)

        self.assertTrue(
            len(remaining_required_intermediate_parameters) == 0,
            f"Required intermediate parameters not present: {remaining_required_intermediate_parameters}",
        )

        remaining_required_optional_parameters = set()

        for param in self.required_optional_params:
            if param not in optional_parameters:
                remaining_required_optional_parameters.add(param)

        self.assertTrue(
            len(remaining_required_optional_parameters) == 0,
            f"Required optional parameters not present: {remaining_required_optional_parameters}",
        )

    def test_inference_batch_consistent(self, batch_sizes=[2]):
        self._test_inference_batch_consistent(batch_sizes=batch_sizes)

    def _test_inference_batch_consistent(
        self, batch_sizes=[2], additional_params_copy_to_batched_inputs=["num_inference_steps"], batch_generator=True
    ):
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
            output = pipe(**batched_input, output="images")
            assert len(output) == batch_size

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-4):
        self._test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    def _test_inference_batch_single_identical(
        self,
        batch_size=2,
        expected_max_diff=1e-4,
        additional_params_copy_to_batched_inputs=["num_inference_steps"],
    ):
        pipe = self.get_pipeline()
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()

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

        output = pipe(**inputs, output="images")
        output_batch = pipe(**batched_inputs, output="images")

        assert output_batch.shape[0] == batch_size

        max_diff = np.abs(to_np(output_batch[0]) - to_np(output[0])).max()
        assert max_diff < expected_max_diff

    @unittest.skipIf(torch_device not in ["cuda", "xpu"], reason="float16 requires CUDA or XPU")
    @require_accelerator
    def test_float16_inference(self, expected_max_diff=5e-2):
        pipe = self.get_pipeline(torch_dtype=torch.float32)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()

        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        pipe_fp16 = self.get_pipeline(torch_dtype=torch.float16)
        for component in pipe_fp16.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
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
        assert max_diff < expected_max_diff

    @require_accelerator
    def test_to_device(self):
        pipe = self.get_pipeline()
        pipe.set_progress_bar_config(disable=None)

        pipe.to("cpu")
        model_devices = [
            component.device.type for component in pipe.components.values() if hasattr(component, "device")
        ]
        self.assertTrue(all(device == "cpu" for device in model_devices))

        output_cpu = pipe(**self.get_dummy_inputs("cpu"), output="images")
        self.assertTrue(np.isnan(output_cpu).sum() == 0)

        pipe.to(torch_device)
        model_devices = [
            component.device.type for component in pipe.components.values() if hasattr(component, "device")
        ]
        self.assertTrue(all(device == torch_device for device in model_devices))

        output_device = pipe(**self.get_dummy_inputs(torch_device), output="images")
        self.assertTrue(np.isnan(to_np(output_device)).sum() == 0)

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

    def test_cfg(self):
        pipe = self.get_pipeline()
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        guider = ClassifierFreeGuidance(guidance_scale=1.0)
        pipe.update_components(guider=guider)

        inputs = self.get_dummy_inputs(torch_device)
        out_no_cfg = pipe(**inputs, output="images")

        guider = ClassifierFreeGuidance(guidance_scale=7.5)
        pipe.update_components(guider=guider)

        out_cfg = pipe(**inputs, output="images")

        assert out_cfg.shape == out_no_cfg.shape
