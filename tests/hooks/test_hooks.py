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
import unittest

import torch

from diffusers.hooks import HookRegistry, ModelHook
from diffusers.training_utils import free_memory
from diffusers.utils.logging import get_logger

from ..testing_utils import CaptureLogger, torch_device


logger = get_logger(__name__)  # pylint: disable=invalid-name


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


class DummyModel(torch.nn.Module):
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


class AddHook(ModelHook):
    def __init__(self, value: int):
        super().__init__()
        self.value = value

    def pre_forward(self, module: torch.nn.Module, *args, **kwargs):
        logger.debug("AddHook pre_forward")
        args = ((x + self.value) if torch.is_tensor(x) else x for x in args)
        return args, kwargs

    def post_forward(self, module, output):
        logger.debug("AddHook post_forward")
        return output


class MultiplyHook(ModelHook):
    def __init__(self, value: int):
        super().__init__()
        self.value = value

    def pre_forward(self, module, *args, **kwargs):
        logger.debug("MultiplyHook pre_forward")
        args = ((x * self.value) if torch.is_tensor(x) else x for x in args)
        return args, kwargs

    def post_forward(self, module, output):
        logger.debug("MultiplyHook post_forward")
        return output

    def __repr__(self):
        return f"MultiplyHook(value={self.value})"


class StatefulAddHook(ModelHook):
    _is_stateful = True

    def __init__(self, value: int):
        super().__init__()
        self.value = value
        self.increment = 0

    def pre_forward(self, module, *args, **kwargs):
        logger.debug("StatefulAddHook pre_forward")
        add_value = self.value + self.increment
        self.increment += 1
        args = ((x + add_value) if torch.is_tensor(x) else x for x in args)
        return args, kwargs

    def reset_state(self, module):
        self.increment = 0


class SkipLayerHook(ModelHook):
    def __init__(self, skip_layer: bool):
        super().__init__()
        self.skip_layer = skip_layer

    def pre_forward(self, module, *args, **kwargs):
        logger.debug("SkipLayerHook pre_forward")
        return args, kwargs

    def new_forward(self, module, *args, **kwargs):
        logger.debug("SkipLayerHook new_forward")
        if self.skip_layer:
            return args[0]
        return self.fn_ref.original_forward(*args, **kwargs)

    def post_forward(self, module, output):
        logger.debug("SkipLayerHook post_forward")
        return output


class HookTests(unittest.TestCase):
    in_features = 4
    hidden_features = 8
    out_features = 4
    num_layers = 2

    def setUp(self):
        params = self.get_module_parameters()
        self.model = DummyModel(**params)
        self.model.to(torch_device)

    def tearDown(self):
        super().tearDown()

        del self.model
        gc.collect()
        free_memory()

    def get_module_parameters(self):
        return {
            "in_features": self.in_features,
            "hidden_features": self.hidden_features,
            "out_features": self.out_features,
            "num_layers": self.num_layers,
        }

    def get_generator(self):
        return torch.manual_seed(0)

    def test_hook_registry(self):
        registry = HookRegistry.check_if_exists_or_initialize(self.model)
        registry.register_hook(AddHook(1), "add_hook")
        registry.register_hook(MultiplyHook(2), "multiply_hook")

        registry_repr = repr(registry)
        expected_repr = "HookRegistry(\n  (0) add_hook - AddHook\n  (1) multiply_hook - MultiplyHook(value=2)\n)"

        self.assertEqual(len(registry.hooks), 2)
        self.assertEqual(registry._hook_order, ["add_hook", "multiply_hook"])
        self.assertEqual(registry_repr, expected_repr)

        registry.remove_hook("add_hook")

        self.assertEqual(len(registry.hooks), 1)
        self.assertEqual(registry._hook_order, ["multiply_hook"])

    def test_stateful_hook(self):
        registry = HookRegistry.check_if_exists_or_initialize(self.model)
        registry.register_hook(StatefulAddHook(1), "stateful_add_hook")

        self.assertEqual(registry.hooks["stateful_add_hook"].increment, 0)

        input = torch.randn(1, 4, device=torch_device, generator=self.get_generator())
        num_repeats = 3

        for i in range(num_repeats):
            result = self.model(input)
            if i == 0:
                output1 = result

        self.assertEqual(registry.get_hook("stateful_add_hook").increment, num_repeats)

        registry.reset_stateful_hooks()
        output2 = self.model(input)

        self.assertEqual(registry.get_hook("stateful_add_hook").increment, 1)
        self.assertTrue(torch.allclose(output1, output2))

    def test_inference(self):
        registry = HookRegistry.check_if_exists_or_initialize(self.model)
        registry.register_hook(AddHook(1), "add_hook")
        registry.register_hook(MultiplyHook(2), "multiply_hook")

        input = torch.randn(1, 4, device=torch_device, generator=self.get_generator())
        output1 = self.model(input).mean().detach().cpu().item()

        registry.remove_hook("multiply_hook")
        new_input = input * 2
        output2 = self.model(new_input).mean().detach().cpu().item()

        registry.remove_hook("add_hook")
        new_input = input * 2 + 1
        output3 = self.model(new_input).mean().detach().cpu().item()

        self.assertAlmostEqual(output1, output2, places=5)
        self.assertAlmostEqual(output1, output3, places=5)
        self.assertAlmostEqual(output2, output3, places=5)

    def test_skip_layer_hook(self):
        registry = HookRegistry.check_if_exists_or_initialize(self.model)
        registry.register_hook(SkipLayerHook(skip_layer=True), "skip_layer_hook")

        input = torch.zeros(1, 4, device=torch_device)
        output = self.model(input).mean().detach().cpu().item()
        self.assertEqual(output, 0.0)

        registry.remove_hook("skip_layer_hook")
        registry.register_hook(SkipLayerHook(skip_layer=False), "skip_layer_hook")
        output = self.model(input).mean().detach().cpu().item()
        self.assertNotEqual(output, 0.0)

    def test_skip_layer_internal_block(self):
        registry = HookRegistry.check_if_exists_or_initialize(self.model.linear_1)
        input = torch.zeros(1, 4, device=torch_device)

        registry.register_hook(SkipLayerHook(skip_layer=True), "skip_layer_hook")
        with self.assertRaises(RuntimeError) as cm:
            self.model(input).mean().detach().cpu().item()
        self.assertIn("mat1 and mat2 shapes cannot be multiplied", str(cm.exception))

        registry.remove_hook("skip_layer_hook")
        output = self.model(input).mean().detach().cpu().item()
        self.assertNotEqual(output, 0.0)

        registry = HookRegistry.check_if_exists_or_initialize(self.model.blocks[1])
        registry.register_hook(SkipLayerHook(skip_layer=True), "skip_layer_hook")
        output = self.model(input).mean().detach().cpu().item()
        self.assertNotEqual(output, 0.0)

    def test_invocation_order_stateful_first(self):
        registry = HookRegistry.check_if_exists_or_initialize(self.model)
        registry.register_hook(StatefulAddHook(1), "add_hook")
        registry.register_hook(AddHook(2), "add_hook_2")
        registry.register_hook(MultiplyHook(3), "multiply_hook")

        input = torch.randn(1, 4, device=torch_device, generator=self.get_generator())

        logger = get_logger(__name__)
        logger.setLevel("DEBUG")

        with CaptureLogger(logger) as cap_logger:
            self.model(input)
        output = cap_logger.out.replace(" ", "").replace("\n", "")
        expected_invocation_order_log = (
            (
                "MultiplyHook pre_forward\n"
                "AddHook pre_forward\n"
                "StatefulAddHook pre_forward\n"
                "AddHook post_forward\n"
                "MultiplyHook post_forward\n"
            )
            .replace(" ", "")
            .replace("\n", "")
        )
        self.assertEqual(output, expected_invocation_order_log)

        registry.remove_hook("add_hook")
        with CaptureLogger(logger) as cap_logger:
            self.model(input)
        output = cap_logger.out.replace(" ", "").replace("\n", "")
        expected_invocation_order_log = (
            ("MultiplyHook pre_forward\nAddHook pre_forward\nAddHook post_forward\nMultiplyHook post_forward\n")
            .replace(" ", "")
            .replace("\n", "")
        )
        self.assertEqual(output, expected_invocation_order_log)

    def test_invocation_order_stateful_middle(self):
        registry = HookRegistry.check_if_exists_or_initialize(self.model)
        registry.register_hook(AddHook(2), "add_hook")
        registry.register_hook(StatefulAddHook(1), "add_hook_2")
        registry.register_hook(MultiplyHook(3), "multiply_hook")

        input = torch.randn(1, 4, device=torch_device, generator=self.get_generator())

        logger = get_logger(__name__)
        logger.setLevel("DEBUG")

        with CaptureLogger(logger) as cap_logger:
            self.model(input)
        output = cap_logger.out.replace(" ", "").replace("\n", "")
        expected_invocation_order_log = (
            (
                "MultiplyHook pre_forward\n"
                "StatefulAddHook pre_forward\n"
                "AddHook pre_forward\n"
                "AddHook post_forward\n"
                "MultiplyHook post_forward\n"
            )
            .replace(" ", "")
            .replace("\n", "")
        )
        self.assertEqual(output, expected_invocation_order_log)

        registry.remove_hook("add_hook")
        with CaptureLogger(logger) as cap_logger:
            self.model(input)
        output = cap_logger.out.replace(" ", "").replace("\n", "")
        expected_invocation_order_log = (
            ("MultiplyHook pre_forward\nStatefulAddHook pre_forward\nMultiplyHook post_forward\n")
            .replace(" ", "")
            .replace("\n", "")
        )
        self.assertEqual(output, expected_invocation_order_log)

        registry.remove_hook("add_hook_2")
        with CaptureLogger(logger) as cap_logger:
            self.model(input)
        output = cap_logger.out.replace(" ", "").replace("\n", "")
        expected_invocation_order_log = (
            ("MultiplyHook pre_forward\nMultiplyHook post_forward\n").replace(" ", "").replace("\n", "")
        )
        self.assertEqual(output, expected_invocation_order_log)

    def test_invocation_order_stateful_last(self):
        registry = HookRegistry.check_if_exists_or_initialize(self.model)
        registry.register_hook(AddHook(1), "add_hook")
        registry.register_hook(MultiplyHook(2), "multiply_hook")
        registry.register_hook(StatefulAddHook(3), "add_hook_2")

        input = torch.randn(1, 4, device=torch_device, generator=self.get_generator())

        logger = get_logger(__name__)
        logger.setLevel("DEBUG")

        with CaptureLogger(logger) as cap_logger:
            self.model(input)
        output = cap_logger.out.replace(" ", "").replace("\n", "")
        expected_invocation_order_log = (
            (
                "StatefulAddHook pre_forward\n"
                "MultiplyHook pre_forward\n"
                "AddHook pre_forward\n"
                "AddHook post_forward\n"
                "MultiplyHook post_forward\n"
            )
            .replace(" ", "")
            .replace("\n", "")
        )
        self.assertEqual(output, expected_invocation_order_log)

        registry.remove_hook("add_hook")
        with CaptureLogger(logger) as cap_logger:
            self.model(input)
        output = cap_logger.out.replace(" ", "").replace("\n", "")
        expected_invocation_order_log = (
            ("StatefulAddHook pre_forward\nMultiplyHook pre_forward\nMultiplyHook post_forward\n")
            .replace(" ", "")
            .replace("\n", "")
        )
        self.assertEqual(output, expected_invocation_order_log)
