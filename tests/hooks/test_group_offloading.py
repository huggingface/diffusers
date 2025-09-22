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

import contextlib
import gc
import unittest

import torch
from parameterized import parameterized

from diffusers.hooks import HookRegistry, ModelHook
from diffusers.models import ModelMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import get_logger
from diffusers.utils.import_utils import compare_versions

from ..testing_utils import (
    backend_empty_cache,
    backend_max_memory_allocated,
    backend_reset_peak_memory_stats,
    require_torch_accelerator,
    torch_device,
)


class DummyBlock(torch.nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()

        self.proj_in = torch.nn.Linear(in_features, hidden_features)
        self.activation = torch.nn.ReLU()
        self.proj_out = torch.nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        x = self.activation(x)
        x = self.proj_out(x)
        return x


class DummyModel(ModelMixin):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, num_layers: int) -> None:
        super().__init__()

        self.linear_1 = torch.nn.Linear(in_features, hidden_features)
        self.activation = torch.nn.ReLU()
        self.blocks = torch.nn.ModuleList(
            [DummyBlock(hidden_features, hidden_features, hidden_features) for _ in range(num_layers)]
        )
        self.linear_2 = torch.nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.activation(x)
        for block in self.blocks:
            x = block(x)
        x = self.linear_2(x)
        return x


# This model implementation contains one type of block (single_blocks) instantiated before another type of block (double_blocks).
# The invocation order of these blocks, however, is first the double_blocks and then the single_blocks.
# With group offloading implementation before https://github.com/huggingface/diffusers/pull/11375, such a modeling implementation
# would result in a device mismatch error because of the assumptions made by the code. The failure case occurs when using:
#   offload_type="block_level", num_blocks_per_group=2, use_stream=True
# Post the linked PR, the implementation will work as expected.
class DummyModelWithMultipleBlocks(ModelMixin):
    def __init__(
        self, in_features: int, hidden_features: int, out_features: int, num_layers: int, num_single_layers: int
    ) -> None:
        super().__init__()

        self.linear_1 = torch.nn.Linear(in_features, hidden_features)
        self.activation = torch.nn.ReLU()
        self.single_blocks = torch.nn.ModuleList(
            [DummyBlock(hidden_features, hidden_features, hidden_features) for _ in range(num_single_layers)]
        )
        self.double_blocks = torch.nn.ModuleList(
            [DummyBlock(hidden_features, hidden_features, hidden_features) for _ in range(num_layers)]
        )
        self.linear_2 = torch.nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.activation(x)
        for block in self.double_blocks:
            x = block(x)
        for block in self.single_blocks:
            x = block(x)
        x = self.linear_2(x)
        return x


# Test for https://github.com/huggingface/diffusers/pull/12077
class DummyModelWithLayerNorm(ModelMixin):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, num_layers: int) -> None:
        super().__init__()

        self.linear_1 = torch.nn.Linear(in_features, hidden_features)
        self.activation = torch.nn.ReLU()
        self.blocks = torch.nn.ModuleList(
            [DummyBlock(hidden_features, hidden_features, hidden_features) for _ in range(num_layers)]
        )
        self.layer_norm = torch.nn.LayerNorm(hidden_features, elementwise_affine=True)
        self.linear_2 = torch.nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.activation(x)
        for block in self.blocks:
            x = block(x)
        x = self.layer_norm(x)
        x = self.linear_2(x)
        return x


class DummyPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "model"

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()

        self.register_modules(model=model)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(2):
            x = x + 0.1 * self.model(x)
        return x


class LayerOutputTrackerHook(ModelHook):
    def __init__(self):
        super().__init__()
        self.outputs = []

    def post_forward(self, module, output):
        self.outputs.append(output)
        return output


@require_torch_accelerator
class GroupOffloadTests(unittest.TestCase):
    in_features = 64
    hidden_features = 256
    out_features = 64
    num_layers = 4

    def setUp(self):
        with torch.no_grad():
            self.model = self.get_model()
            self.input = torch.randn((4, self.in_features)).to(torch_device)

    def tearDown(self):
        super().tearDown()

        del self.model
        del self.input
        gc.collect()
        backend_empty_cache(torch_device)
        backend_reset_peak_memory_stats(torch_device)

    def get_model(self):
        torch.manual_seed(0)
        return DummyModel(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            num_layers=self.num_layers,
        )

    def test_offloading_forward_pass(self):
        @torch.no_grad()
        def run_forward(model):
            gc.collect()
            backend_empty_cache(torch_device)
            backend_reset_peak_memory_stats(torch_device)
            self.assertTrue(
                all(
                    module._diffusers_hook.get_hook("group_offloading") is not None
                    for module in model.modules()
                    if hasattr(module, "_diffusers_hook")
                )
            )
            model.eval()
            output = model(self.input)[0].cpu()
            max_memory_allocated = backend_max_memory_allocated(torch_device)
            return output, max_memory_allocated

        self.model.to(torch_device)
        output_without_group_offloading, mem_baseline = run_forward(self.model)
        self.model.to("cpu")

        model = self.get_model()
        model.enable_group_offload(torch_device, offload_type="block_level", num_blocks_per_group=3)
        output_with_group_offloading1, mem1 = run_forward(model)

        model = self.get_model()
        model.enable_group_offload(torch_device, offload_type="block_level", num_blocks_per_group=1)
        output_with_group_offloading2, mem2 = run_forward(model)

        model = self.get_model()
        model.enable_group_offload(torch_device, offload_type="block_level", num_blocks_per_group=1, use_stream=True)
        output_with_group_offloading3, mem3 = run_forward(model)

        model = self.get_model()
        model.enable_group_offload(torch_device, offload_type="leaf_level")
        output_with_group_offloading4, mem4 = run_forward(model)

        model = self.get_model()
        model.enable_group_offload(torch_device, offload_type="leaf_level", use_stream=True)
        output_with_group_offloading5, mem5 = run_forward(model)

        # Precision assertions - offloading should not impact the output
        self.assertTrue(torch.allclose(output_without_group_offloading, output_with_group_offloading1, atol=1e-5))
        self.assertTrue(torch.allclose(output_without_group_offloading, output_with_group_offloading2, atol=1e-5))
        self.assertTrue(torch.allclose(output_without_group_offloading, output_with_group_offloading3, atol=1e-5))
        self.assertTrue(torch.allclose(output_without_group_offloading, output_with_group_offloading4, atol=1e-5))
        self.assertTrue(torch.allclose(output_without_group_offloading, output_with_group_offloading5, atol=1e-5))

        # Memory assertions - offloading should reduce memory usage
        self.assertTrue(mem4 <= mem5 < mem2 <= mem3 < mem1 < mem_baseline)

    def test_warning_logged_if_group_offloaded_module_moved_to_accelerator(self):
        if torch.device(torch_device).type not in ["cuda", "xpu"]:
            return
        self.model.enable_group_offload(torch_device, offload_type="block_level", num_blocks_per_group=3)
        logger = get_logger("diffusers.models.modeling_utils")
        logger.setLevel("INFO")
        with self.assertLogs(logger, level="WARNING") as cm:
            self.model.to(torch_device)
        self.assertIn(f"The module '{self.model.__class__.__name__}' is group offloaded", cm.output[0])

    def test_warning_logged_if_group_offloaded_pipe_moved_to_accelerator(self):
        if torch.device(torch_device).type not in ["cuda", "xpu"]:
            return
        pipe = DummyPipeline(self.model)
        self.model.enable_group_offload(torch_device, offload_type="block_level", num_blocks_per_group=3)
        logger = get_logger("diffusers.pipelines.pipeline_utils")
        logger.setLevel("INFO")
        with self.assertLogs(logger, level="WARNING") as cm:
            pipe.to(torch_device)
        self.assertIn(f"The module '{self.model.__class__.__name__}' is group offloaded", cm.output[0])

    def test_error_raised_if_streams_used_and_no_accelerator_device(self):
        torch_accelerator_module = getattr(torch, torch_device, torch.cuda)
        original_is_available = torch_accelerator_module.is_available
        torch_accelerator_module.is_available = lambda: False
        with self.assertRaises(ValueError):
            self.model.enable_group_offload(
                onload_device=torch.device(torch_device), offload_type="leaf_level", use_stream=True
            )
        torch_accelerator_module.is_available = original_is_available

    def test_error_raised_if_supports_group_offloading_false(self):
        self.model._supports_group_offloading = False
        with self.assertRaisesRegex(ValueError, "does not support group offloading"):
            self.model.enable_group_offload(onload_device=torch.device(torch_device))

    def test_error_raised_if_model_offloading_applied_on_group_offloaded_module(self):
        pipe = DummyPipeline(self.model)
        pipe.model.enable_group_offload(torch_device, offload_type="block_level", num_blocks_per_group=3)
        with self.assertRaisesRegex(ValueError, "You are trying to apply model/sequential CPU offloading"):
            pipe.enable_model_cpu_offload()

    def test_error_raised_if_sequential_offloading_applied_on_group_offloaded_module(self):
        pipe = DummyPipeline(self.model)
        pipe.model.enable_group_offload(torch_device, offload_type="block_level", num_blocks_per_group=3)
        with self.assertRaisesRegex(ValueError, "You are trying to apply model/sequential CPU offloading"):
            pipe.enable_sequential_cpu_offload()

    def test_error_raised_if_group_offloading_applied_on_model_offloaded_module(self):
        pipe = DummyPipeline(self.model)
        pipe.enable_model_cpu_offload()
        with self.assertRaisesRegex(ValueError, "Cannot apply group offloading"):
            pipe.model.enable_group_offload(torch_device, offload_type="block_level", num_blocks_per_group=3)

    def test_error_raised_if_group_offloading_applied_on_sequential_offloaded_module(self):
        pipe = DummyPipeline(self.model)
        pipe.enable_sequential_cpu_offload()
        with self.assertRaisesRegex(ValueError, "Cannot apply group offloading"):
            pipe.model.enable_group_offload(torch_device, offload_type="block_level", num_blocks_per_group=3)

    def test_block_level_stream_with_invocation_order_different_from_initialization_order(self):
        if torch.device(torch_device).type not in ["cuda", "xpu"]:
            return

        model = DummyModelWithMultipleBlocks(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            num_layers=self.num_layers,
            num_single_layers=self.num_layers + 1,
        )
        model.enable_group_offload(torch_device, offload_type="block_level", num_blocks_per_group=1, use_stream=True)

        context = contextlib.nullcontext()
        if compare_versions("diffusers", "<=", "0.33.0"):
            # Will raise a device mismatch RuntimeError mentioning weights are on CPU but input is on device
            context = self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device")

        with context:
            model(self.input)

    @parameterized.expand([("block_level",), ("leaf_level",)])
    def test_block_level_offloading_with_parameter_only_module_group(self, offload_type: str):
        if torch.device(torch_device).type not in ["cuda", "xpu"]:
            return

        def apply_layer_output_tracker_hook(model: DummyModelWithLayerNorm):
            for name, module in model.named_modules():
                registry = HookRegistry.check_if_exists_or_initialize(module)
                hook = LayerOutputTrackerHook()
                registry.register_hook(hook, "layer_output_tracker")

        model_ref = DummyModelWithLayerNorm(128, 256, 128, 2)
        model = DummyModelWithLayerNorm(128, 256, 128, 2)

        model.load_state_dict(model_ref.state_dict(), strict=True)

        model_ref.to(torch_device)
        model.enable_group_offload(torch_device, offload_type=offload_type, num_blocks_per_group=1, use_stream=True)

        apply_layer_output_tracker_hook(model_ref)
        apply_layer_output_tracker_hook(model)

        x = torch.randn(2, 128).to(torch_device)

        out_ref = model_ref(x)
        out = model(x)
        self.assertTrue(torch.allclose(out_ref, out, atol=1e-5), "Outputs do not match.")

        num_repeats = 4
        for i in range(num_repeats):
            out_ref = model_ref(x)
            out = model(x)

        self.assertTrue(torch.allclose(out_ref, out, atol=1e-5), "Outputs do not match after multiple invocations.")

        for (ref_name, ref_module), (name, module) in zip(model_ref.named_modules(), model.named_modules()):
            assert ref_name == name
            ref_outputs = (
                HookRegistry.check_if_exists_or_initialize(ref_module).get_hook("layer_output_tracker").outputs
            )
            outputs = HookRegistry.check_if_exists_or_initialize(module).get_hook("layer_output_tracker").outputs
            cumulated_absmax = 0.0
            for i in range(len(outputs)):
                diff = ref_outputs[0] - outputs[i]
                absdiff = diff.abs()
                absmax = absdiff.max().item()
                cumulated_absmax += absmax
            self.assertLess(
                cumulated_absmax, 1e-5, f"Output differences for {name} exceeded threshold: {cumulated_absmax:.5f}"
            )
