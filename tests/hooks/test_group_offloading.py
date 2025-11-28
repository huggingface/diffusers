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

from typing import Any, Iterable, List, Optional, Sequence, Union

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
    

# Test for https://github.com/huggingface/diffusers/pull/12747
class DummyCallableBySubmodule:
    """
    Callable group offloading pinner that pins first and last DummyBlock
    called in the program by callable(submodule)
    """
    def __init__(self, pin_targets: Iterable[torch.nn.Module]) -> None:
        self.pin_targets = set(pin_targets)
        self.calls_track = [] # testing only

    def __call__(self, submodule: torch.nn.Module) -> bool:
        self.calls_track.append(submodule)
        return self._normalize_module_type(submodule) in self.pin_targets
    
    def _normalize_module_type(self, obj: Any) -> Optional[torch.nn.Module]:
        # group might be a single module, or a container of modules
        # The group-offloading code may pass either:
        #   - a single `torch.nn.Module`, or
        #   - a container (list/tuple) of modules.

        # Only return a module when the mapping is unambiguous:
        #   - if `obj` is a module -> return it
        #   - if `obj` is a list/tuple containing exactly one module -> return that module
        #   - otherwise -> return None (won't be considered as a target candidate)
        if isinstance(obj, torch.nn.Module):
            return obj
        if isinstance(obj, (list, tuple)):
            mods = [m for m in obj if isinstance(m, torch.nn.Module)]
            return mods[0] if len(mods) == 1 else None
        return None
    
class DummyCallableByNameSubmodule(DummyCallableBySubmodule):
    """
    Callable group offloading pinner that pins first and last DummyBlock
    Same behaviour with DummyCallableBySubmodule, only with different call signature
    called in the program by callable(name, submodule)
    """
    def __call__(self, name: str, submodule: torch.nn.Module) -> bool:
        self.calls_track.append((name, submodule))
        return self._normalize_module_type(submodule) in self.pin_targets
    
class DummyCallableByNameSubmoduleIdx(DummyCallableBySubmodule):
    """
    Callable group offloading pinner that pins first and last DummyBlock.
    Same behaviour with DummyCallableBySubmodule, only with different call signature
    Called in the program by callable(name, submodule, idx)
    """
    def __call__(self, name: str, submodule: torch.nn.Module, idx: int) -> bool:
        self.calls_track.append((name, submodule, idx))
        return self._normalize_module_type(submodule) in self.pin_targets
    
class DummyInvalidCallable(DummyCallableBySubmodule):
    """
    Callable group offloading pinner that uses invalid call signature
    """
    def __call__(self, name: str, submodule: torch.nn.Module, idx: int, extra: Any) -> bool:
        self.calls_track.append((name, submodule, idx, extra))
        return self._normalize_module_type(submodule) in self.pin_targets


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
    
    def test_block_level_offloading_with_pin_groups_stay_on_device(self):
        if torch.device(torch_device).type not in ["cuda", "xpu"]:
            return

        def assert_all_modules_on_expected_device(modules: Sequence[torch.nn.Module],
                                                  expected_device: Union[torch.device, str],
                                                  header_error_msg: str = "") -> None:
            def first_param_device(modules: torch.nn.Module) -> torch.device:
                p = next(modules.parameters(), None)
                self.assertIsNotNone(p, f"No parameters found for module {modules}")
                return p.device

            if isinstance(expected_device, torch.device):
                expected_device = expected_device.type
            
            bad = []
            for i, m in enumerate(modules):
                dev_type = first_param_device(m).type
                if dev_type != expected_device:
                    bad.append((i, m.__class__.__name__, dev_type))
            self.assertTrue(
                len(bad) == 0,
                (header_error_msg + "\n" if header_error_msg else "")
                + f"Expected all modules on {expected_device}, but found mismatches: {bad}",
            )

        def get_param_modules_from_execution_order(model: DummyModel) -> List[torch.nn.Module]:
            model.eval()
            root_registry = HookRegistry.check_if_exists_or_initialize(model)

            lazy_hook = root_registry.get_hook("lazy_prefetch_group_offloading")
            self.assertIsNotNone(lazy_hook, "lazy_prefetch_group_offloading hook was not registered")

            #record execution order with first forward
            with torch.no_grad():
                model(self.input)

            mods = [m for _, m in lazy_hook.execution_order]
            param_modules = [m for m in mods if next(m.parameters(), None) is not None]
            return param_modules
        
        def assert_callables_offloading_tests(
            param_modules: Sequence[torch.nn.Module],
            callable: Any,
            header_error_msg: str = "",
        ) -> None:
            pinned_modules = [m for m in param_modules if m in callable.pin_targets]
            unpinned_modules = [m for m in param_modules if m not in callable.pin_targets]
            self.assertTrue(len(callable.calls_track) > 0, f"{header_error_msg}: callable should have been called at least once")
            assert_all_modules_on_expected_device(pinned_modules, torch_device, f"{header_error_msg}: pinned blocks should stay on device")
            assert_all_modules_on_expected_device(unpinned_modules, "cpu", f"{header_error_msg}: unpinned blocks should be offloaded")


        default_parameters = {
            "onload_device": torch_device,
            "offload_type": "block_level",
            "num_blocks_per_group": 1,
            "use_stream": True,
        }
        model_default_no_pin = self.get_model()
        model_default_no_pin.enable_group_offload(
            **default_parameters
        )
        param_modules = get_param_modules_from_execution_order(model_default_no_pin)
        assert_all_modules_on_expected_device(param_modules, 
                                              expected_device="cpu", 
                                              header_error_msg="default pin_groups: expected ALL modules offloaded to CPU")

        model_pin_all = self.get_model()
        model_pin_all.enable_group_offload(
            **default_parameters,
            pin_groups="all",
        )
        param_modules = get_param_modules_from_execution_order(model_pin_all)
        assert_all_modules_on_expected_device(param_modules, 
                                              expected_device=torch_device, 
                                              header_error_msg="pin_groups = all: expected ALL layers on accelerator device")


        model_pin_first_last = self.get_model()
        model_pin_first_last.enable_group_offload(
            **default_parameters,
            pin_groups="first_last",
        )
        param_modules = get_param_modules_from_execution_order(model_pin_first_last)
        assert_all_modules_on_expected_device([param_modules[0], param_modules[-1]], 
                                              expected_device=torch_device, 
                                              header_error_msg="pin_groups = first_last: expected first and last layers on accelerator device")
        assert_all_modules_on_expected_device(param_modules[1:-1], 
                                              expected_device="cpu", 
                                              header_error_msg="pin_groups = first_last: expected ALL middle layers offloaded to CPU")

        
        model = self.get_model()
        callable_by_submodule = DummyCallableBySubmodule(pin_targets=[model.blocks[0], model.blocks[-1]])
        model.enable_group_offload(**default_parameters, 
                                   pin_groups=callable_by_submodule)
        param_modules = get_param_modules_from_execution_order(model)
        assert_callables_offloading_tests(param_modules, 
                                          callable_by_submodule,
                                          header_error_msg="pin_groups with callable(submodule)")

        model = self.get_model()
        callable_by_name_submodule = DummyCallableByNameSubmodule(pin_targets=[model.blocks[0], model.blocks[-1]])
        model.enable_group_offload(**default_parameters, 
                                   pin_groups=callable_by_name_submodule)
        param_modules = get_param_modules_from_execution_order(model)
        assert_callables_offloading_tests(param_modules, 
                                          callable_by_name_submodule,
                                          header_error_msg="pin_groups with callable(name, submodule)")

        model = self.get_model()
        callable_by_name_submodule_idx = DummyCallableByNameSubmoduleIdx(pin_targets=[model.blocks[0], model.blocks[-1]])
        model.enable_group_offload(**default_parameters, 
                                   pin_groups=callable_by_name_submodule_idx)
        param_modules = get_param_modules_from_execution_order(model)
        assert_callables_offloading_tests(param_modules, 
                                          callable_by_name_submodule_idx,
                                          header_error_msg="pin_groups with callable(name, submodule, idx)")
        
    def test_error_raised_if_pin_groups_received_invalid_value(self):
        default_parameters = {
            "onload_device": torch_device,
            "offload_type": "block_level",
            "num_blocks_per_group": 1,
            "use_stream": True,
        }
        model = self.get_model()
        with self.assertRaisesRegex(ValueError, 
                                    "`pin_groups` must be one of `None`, 'first_last', 'all', or a callable."):
            model.enable_group_offload(
                **default_parameters,
                pin_groups="invalid value",
            )

    def test_error_raised_if_pin_groups_received_invalid_callables(self):
        default_parameters = {
            "onload_device": torch_device,
            "offload_type": "block_level",
            "num_blocks_per_group": 1,
            "use_stream": True,
        }
        model = self.get_model()
        invalid_callable = DummyInvalidCallable(pin_targets=[model.blocks[0], model.blocks[-1]])
        model.enable_group_offload(
            **default_parameters,
            pin_groups=invalid_callable,
        )
        with self.assertRaisesRegex(TypeError, 
                                    r"missing\s+\d+\s+required\s+positional\s+argument(s)?:"):
            with torch.no_grad():
                model(self.input)
            
        

        